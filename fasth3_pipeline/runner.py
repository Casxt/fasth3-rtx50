"""Shared stage-at-a-time inference for FastH3 Omni PDD and V2 DMD2."""
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import wraps
import gc
from pathlib import Path
from time import perf_counter
import torch
from .attention import TileLayout
from .dmd2 import DMD2Plan, RUNGS
from .pdd import PDDPlan
from .transformer import OMNI_PROFILE, T2VA_PROFILE, load_transformer, install_vsa


_RUN_RESOURCES = ContextVar('fasth3_run_resources', default=None)


def _managed_run(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with ExitStack() as resources:
            token = _RUN_RESOURCES.set(resources)
            try:
                return function(*args, **kwargs)
            finally:
                _RUN_RESOURCES.reset(token)
    return wrapped


def _own_stream(stream, diagnostics=None):
    resources = _RUN_RESOURCES.get()
    if resources is None:
        raise RuntimeError('Streaming resource must belong to a managed run')
    holder = [stream]
    def cleanup():
        if holder:
            holder.pop().close()
    resources.callback(cleanup)
    stream.add_close_callback(holder.clear)
    if diagnostics is not None:
        stream.add_close_callback(lambda: diagnostics.streaming.__setitem__('dit', stream.stats_snapshot()))
    return stream


def load_references(items, *, video_canvas=None, video_frames=None):
    from diffusers.modular_pipelines.minimax_h3.references import (
        MiniMaxH3ImageReference, MiniMaxH3VideoReference, MiniMaxH3AudioReference)
    classes = {"image": MiniMaxH3ImageReference, "video": MiniMaxH3VideoReference, "audio": MiniMaxH3AudioReference}
    references = []
    for item in items:
        kind, separator, path = item.partition(":")
        if not separator or kind not in classes:
            raise ValueError("Each --reference must be image:PATH, video:PATH or audio:PATH")
        path = Path(path).resolve(strict=True)
        # Keep input order. Official PyAV decoder retains video fps AND embedded audio.
        if kind == 'video' and video_canvas is not None:
            from .long_video import aligned_video_frames, read_video_reference
            references.append(read_video_reference(path, video_canvas, min(362, aligned_video_frames(video_frames))))
        else:
            references.append(classes[kind].from_file(path))
    if not references:
        raise ValueError("Ref2VA requires at least one reference")
    return references


def layout_from_state(state, patch_size, device, *, task='r2va'):
    _, ph, pw = patch_size
    segments = [int(state.get("prompt_embeds").shape[1])]
    if task == 'r2va':
        visual = iter(state.get('condition_latents'))
        audio = iter(state.get('audio_condition_latents'))
        for reference in state.get('normalized_references'):
            if reference.has_audio:
                segments.append(int(next(audio).shape[0]))
            if reference.kind in ('image', 'video'):
                shape = next(visual).shape
                segments.append((int(shape[2]), int(shape[3])//ph, int(shape[4])//pw))
    elif task in ('f2va', 'fl2va'):
        # Endpoint conditions are separate dense rows at the target rotary times.
        segments.append(int(state.get('num_condition_video_rows')))
    segments += [int(state.get("num_audio_latents"))*2,
                 (int(state.get("num_latent_frames")), int(state.get("latent_height"))//ph,
                  int(state.get("latent_width"))//pw)]
    return TileLayout.build(segments, device, tile_size=64 if task == 't2va' else 128)


def validate_task_inputs(task, references=(), first_frame=None, last_frame=None):
    if task not in ('t2va', 'f2va', 'fl2va', 'r2va'):
        raise ValueError(f'Unknown FastH3 task: {task}')
    if task == 'r2va' and not any(ref.kind in ('image', 'video') for ref in references):
        raise ValueError('r2va requires at least one image or video reference')
    if task != 'r2va' and references:
        raise ValueError('Only r2va accepts reference media')
    expected = (task in ('f2va', 'fl2va'), task == 'fl2va')
    if (first_frame is not None, last_frame is not None) != expected:
        raise ValueError(f'Invalid endpoint frames for {task}: f2va takes first only, fl2va takes first and last')
    if first_frame is not None:
        from PIL import Image
        if not isinstance(first_frame, Image.Image) or last_frame is not None and not isinstance(last_frame, Image.Image):
            raise TypeError('Endpoint frames must be PIL images')
    return task


def _prepare_denoising(core, pipe, state, task):
    names = {
        'r2va': ('prepare_layout', 'prepare_condition_latents', 'prepare_latents', 'prepare_latents_ref2va'),
        't2va': ('no_keyframe_anchors', 'prepare_layout', 'prepare_latents'),
        'f2va': ('prepare_layout', 'prepare_condition_latents', 'prepare_latents', 'prepare_latents_fl2va'),
        'fl2va': ('prepare_layout', 'prepare_condition_latents', 'prepare_latents', 'prepare_latents_fl2va'),
    }
    for name in names[task]:
        state = _run_block(core.sub_blocks[name], pipe, state)
        if task == 'f2va' and name == 'prepare_condition_latents':
            from diffusers.utils.torch_utils import randn_tensor

            # Reserve the closing anchor's RNG draw so equal-seed frame-control
            # requests use the same target noise on the same canvas and duration.
            first, = state.get('condition_latents')
            randn_tensor(first.shape, generator=state.get('generator'),
                         device=pipe._execution_device, dtype=torch.float32)
    return state


def _clear_device(device):
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


def configure_lightweight_components(pipe):
    """Match the official blocks' processor contracts, not generic defaults."""
    from diffusers import MiniMaxH3Scheduler
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.video_processor import VideoProcessor
    # This scheduler only noises anchors. The task sampler owns the updates.
    # H3 decode already reverts ImageNet normalization into [0, 1].
    pipe.update_components(scheduler=MiniMaxH3Scheduler(),
                           image_processor=VaeImageProcessor(vae_scale_factor=16),
                           video_processor=VideoProcessor(vae_scale_factor=16, do_normalize=False))


def load_local_components(pipe, names, root):
    """Override every remote spec location; fail if selective loading did not succeed."""
    root = str(Path(root).resolve(strict=True))
    unknown = set(names)-set(pipe._component_specs)
    if unknown:
        raise ValueError(f"Unknown components: {sorted(unknown)}")
    pipe.load_components(names=names, pretrained_model_name_or_path=root,
                         subfolder={name: name for name in names},
                         local_files_only=True, torch_dtype=torch.bfloat16)
    missing = [name for name in names if getattr(pipe, name, None) is None]
    if missing:
        raise RuntimeError(f"Local component loading failed: {missing}, root={root}")


def offload_vae(vae, device, guard, mode='cached'):
    from types import SimpleNamespace
    from .streaming import CPUWeightStore, ManagedStream
    from .offload import BlockStreamedWeights, GIB, vae_groups

    if mode not in ('fixed', 'cached'):
        raise ValueError('Unknown VAE offload mode')
    if hasattr(vae, 'enable_tiling'):
        vae.enable_tiling()
    store = CPUWeightStore(vae)
    try:
        tensor_guard = getattr(guard, 'check_device', None) if torch.device(device).type == 'cuda' else None
        stream = BlockStreamedWeights(vae, store, device, guard, groups=vae_groups(vae),
                                      tensor_guard=tensor_guard, guard_each_tensor=False,
                                      resident_cache_bytes=19 * GIB // 4 if mode == 'cached' else 0)
    except BaseException:
        store.close()
        raise
    vae._hf_hook = SimpleNamespace(execution_device=torch.device(device))
    return ManagedStream(stream, store, mode)


def _run_block(block, pipe, state):
    # Direct block invocation intentionally avoids ModularPipeline.__call__'s
    # deepcopy(state), which duplicates reference videos and all live latents.
    return block(pipe, state)[1]


@contextmanager
def _vae_execution(pipe, device, guard, mode, diagnostics, phase, *, names=('vae', 'audio_vae')):
    def close_and_record(name, handle):
        try:
            handle.close()
        finally:
            diagnostics.streaming.setdefault(phase, {})[name] = handle.stats_snapshot()
    with ExitStack() as stage:
        for name in names:
            handle = offload_vae(getattr(pipe, name), device, guard, mode)
            stage.callback(close_and_record, name, handle)
            stage.enter_context(handle.execution())
        yield


def decode_output(block, pipe, state, device, guard, mode, diagnostics, *,
                  vae_mxfp8=None, debug_checks=False):
    if vae_mxfp8 is not None:
        from .vae_mxfp8 import install_vae_mxfp8
        install_vae_mxfp8(pipe.vae, vae_mxfp8, device, guard, debug_checks=debug_checks)
    available = guard.budget.available_bytes() if guard.budget is not None else 0
    # The released 768p pairwise decoder peaked at about 5.4 GiB on SM120.
    pair_tiles = mode == 'cached' and available >= (6 << 30)
    diagnostics.streaming['decode_vae_policy'] = {'pair_tiles': pair_tiles, 'available_bytes': available}
    with _vae_execution(pipe, device, guard, mode, diagnostics, 'decode_vae'):
        from .vae_temporal import decode_video_to_host
        state = decode_video_to_host(pipe, state, pair_tiles=pair_tiles)
        return _run_block(block.sub_blocks['audio'], pipe, state)


def _install_dit_warmup(vae, stream, diagnostics):
    # First encoder compute block, not model.forward (encode bypasses it).
    paths = [name for name, module in vae.named_modules()
             if name.startswith('encoder.') and type(module).__name__ in
             ('MiniMaxH3VideoResnetBlock3d', 'MiniMaxH3VideoDownBlock3d')]
    if not paths:
        raise ValueError('Cannot locate a known VAE compute block for DiT warmup')
    info = diagnostics.streaming['dit_warmup'] = {'trigger': paths[0], 'attempted': False, 'ready': False}
    def after_compute(module, args, output):
        if not info['attempted']:
            info['attempted'] = True
            info['ready'] = bool(stream.prime_first())
    return vae.get_submodule(paths[0]).register_forward_hook(after_compute)


@_managed_run
@torch.no_grad()
def run_h3(*, task, base_model, transformer_path, text_encoder_path, prompt, guard,
           references=(), first_frame=None, last_frame=None, device='cuda:0', height=768, width=1344,
           num_frames=124, seed=1234, shared_components=None, reference_image_short_edge=256,
           diagnostics=None, canvas_config=None, debug_checks=False, adaln_table=None,
           reference_cache=None, vae_mxfp8=None):
    from .diagnostics import PipelineDiagnostics
    from .long_video import aligned_video_frames, allow_released_maximum
    references = tuple(references or ())
    task = validate_task_inputs(task, references, first_frame, last_frame)
    aligned_frames = aligned_video_frames(num_frames)
    if task == 'fl2va' and aligned_frames != num_frames:
        raise ValueError(f'fl2va requires a VAE-aligned frame count to retain the closing-frame time; '
                         f'{num_frames} rounds up to {aligned_frames}')
    is_pdd = task != 't2va'
    profile = OMNI_PROFILE if is_pdd else T2VA_PROFILE
    if diagnostics is None:
        diagnostics = PipelineDiagnostics(guard=guard, debug_checks=debug_checks)
    debug_checks = debug_checks or diagnostics.debug_checks
    diagnostics.debug_checks = debug_checks
    from diffusers.modular_pipelines.modular_pipeline import PipelineState
    from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import MiniMaxH3Blocks
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3SetTimestepsStep
    from .encoder import load_text_encoder
    blocks = MiniMaxH3Blocks()
    base_model = Path(base_model).resolve(strict=True)
    plan = PDDPlan.create() if is_pdd else DMD2Plan.from_checkpoint(base_model, 'cpu')
    diagnostics.streaming['model'] = {'task': task, 'profile': profile,
                                      'sampler': 'pdd' if is_pdd else 'dmd2',
                                      'vsa_tile_size': 128 if is_pdd else 64}
    shared_components = Path(shared_components).resolve(strict=True) if shared_components else base_model.parent / "shared"
    pipe = blocks.init_pipeline(str(base_model))
    allow_released_maximum(pipe)
    if vae_mxfp8 is None:
        overlay = shared_components / 'vae-ffn-mxfp8.safetensors'
        vae_mxfp8 = overlay if overlay.is_file() else None
    if adaln_table is None and is_pdd:
        table = Path(transformer_path).resolve().parent / 'adaln-pdd8.safetensors'
        adaln_table = table if table.is_file() else None
    if not is_pdd and adaln_table is not None:
        raise ValueError('The PDD AdaLN table is incompatible with the V2 DMD2 schedule')
    diagnostics.streaming['assets'] = {
        'adaln_table': str(adaln_table) if adaln_table is not None else None,
        'vae_mxfp8': str(vae_mxfp8) if vae_mxfp8 is not None else None,
    }
    if canvas_config is not None:
        if pipe.canvas_multiple != canvas_config['canvas_multiple']:
            raise ValueError('Released canvas multiple differs from local model geometry')
        canvas_options = {key: canvas_config[key] for key in ('canvas_short_edge', 'canvas_max_pixels')}
    else:
        canvas_options = dict(canvas_short_edge=min(height, width), canvas_max_pixels=height*width)
    pipe.update_components(**canvas_options, reference_image_short_edge=reference_image_short_edge)
    state = PipelineState()
    for param in blocks.inputs:
        if param.name:
            state.set(param.name, param.default, param.kwargs_type)
    for key, value in dict(prompt=prompt, references=list(references) if task == 'r2va' else None,
                           image=first_frame.convert('RGB') if first_frame is not None else None,
                           last_image=last_frame.convert('RGB') if last_frame is not None else None,
                           height=height, width=width, num_frames=num_frames if task == 'r2va' else aligned_frames,
                           num_inference_steps=8 if is_pdd else 9, guidance_scale=1.,
                           output_type="np", generator=torch.Generator(device=device).manual_seed(seed)).items():
        state.set(key, value)
    guard("load lightweight processors")
    configure_lightweight_components(pipe)
    load_local_components(pipe, ["tokenizer", "processor"], shared_components)
    state = _run_block(blocks.sub_blocks["before_encode"], pipe, state)
    if diagnostics.checkpoint("preprocess", height=state.get("height"), width=state.get("width"),
                              num_frames=state.get("num_frames"), reference_count=len(references)):
        return diagnostics.stopped("preprocess")

    guard("load text encoder")
    encoder, stream = load_text_encoder(text_encoder_path, device, guard, bounded_attention=True, prefetch=True)
    try:
        if reference_cache is not None:
            from .reference_cache import memoize_qwen_vision
            memoize_qwen_vision(encoder, reference_cache, text_encoder_path, stream)
        pipe.update_components(text_encoder=encoder)
        with stream.execution():
            state = _run_block(blocks.sub_blocks["text_encoder"], pipe, state)
    finally:
        stream.close()
        diagnostics.streaming['text_encoder'] = stream.stats_snapshot()
        if reference_cache is not None:
            diagnostics.streaming['reference_cache'] = reference_cache.snapshot()
        pipe.update_components(text_encoder=None)
        del encoder, stream
        _clear_device(device)
    guard("text encoder released")
    if diagnostics.checkpoint("encoder", {"prompt_embeds": state.get("prompt_embeds")}):
        return diagnostics.stopped("encoder")

    # Construct only meta DiT ahead of VAE work; its first real group is primed
    # from a VAE compute posthook, subject to the same global budget.
    model, stream = load_transformer(transformer_path, device, guard,
                                     profile=profile, debug_checks=debug_checks, adaln_table=adaln_table)
    _own_stream(stream, diagnostics)
    cached_reference = None
    if reference_cache is not None and task == 'r2va':
        from .reference_cache import vae_reference_key
        reference_key = vae_reference_key(pipe, state, shared_components)
        cached_reference = reference_cache.get(reference_key)
    if task == 't2va':
        state.set('condition_latents', [])
        state.set('audio_condition_latents', [])
    elif cached_reference is None:
        names = ['vae']
        if task == 'r2va' and any(ref.has_audio for ref in state.get('normalized_references')):
            names.append('audio_vae')
        started = perf_counter()
        load_local_components(pipe, names, shared_components)
        diagnostics.streaming['reference_vae_setup'] = {
            'components': names, 'seconds': perf_counter() - started,
            'video_encoder_precision': 'bf16_autocast' if 'vae' in names else None,
        }
        from .vae_temporal import encode_keyframes, encode_references
        with _vae_execution(pipe, device, guard, 'cached', diagnostics, 'reference_vae', names=names):
            with ExitStack() as warmup:
                hook = _install_dit_warmup(pipe.vae, stream, diagnostics)
                warmup.callback(hook.remove)
                state = (encode_references if task == 'r2va' else encode_keyframes)(pipe, state)
        if reference_cache is not None and task == 'r2va':
            reference_cache.put(reference_key, (state.get("condition_latents"), state.get("audio_condition_latents")))
        # Release VAE CPU sources before DiT; decode reloads VAEs.
        pipe.update_components(**dict.fromkeys(names))
        _clear_device(device)
    else:
        state.set("condition_latents", cached_reference[0])
        state.set("audio_condition_latents", cached_reference[1])
    if reference_cache is not None:
        diagnostics.streaming['reference_cache'] = reference_cache.snapshot()
    guard("reference VAEs released")
    condition_tensors = {f"video_{i}": value for i, value in enumerate(state.get("condition_latents"))}
    condition_tensors.update({f"audio_{i}": value for i, value in enumerate(state.get("audio_condition_latents"))})
    if diagnostics.checkpoint("reference-vae", condition_tensors):
        return diagnostics.stopped("reference-vae")
    del condition_tensors

    component = 'transformer_ref' if is_pdd else 'transformer'
    core_name = {'r2va': 'ref2va', 'f2va': 'fl2va', 'fl2va': 'fl2va', 't2va': 't2va'}[task]
    try:
        pipe.update_components(**{component: model})
        core = blocks.sub_blocks['denoise'].sub_blocks[core_name]
        state = _prepare_denoising(core, pipe, state, task)
        layout = layout_from_state(state, pipe.patch_size, device, task=task)
        policy = install_vsa(model, layout, guard.budget, sparsity=.9 if is_pdd else .8, debug_checks=debug_checks)
        diagnostics.streaming['dit_policy'] = policy
        if adaln_table is not None or not is_pdd:
            # Text refinement and rotary positions are fixed for this run.
            model._long_static = {}
        fields = {name: state.get(name) for name in ("token_tags", "position_ids", "video_indices", "audio_indices", "text_indices")}
        video, audio = state.get("latents")[None], state.get("audio_latents")[None]
        cv, ca = state.get("num_condition_video_rows"), state.get("num_condition_audio_rows")
        prompt_embeds = state.get("prompt_embeds")
        diagnostics.checkpoint("noise-unscaled", {"video": video, "audio": audio, **fields},
                               condition_video_rows=cv, condition_audio_rows=ca)
        video = plan.scale_initial_noise(video, "video", cv)
        audio = plan.scale_initial_noise(audio, "audio", ca)
        diagnostics.checkpoint("noise-scaled", {"video": video, "audio": audio},
                               video_sigma=float(plan.sigmas["video"][0]), audio_sigma=float(plan.sigmas["audio"][0]))
        for step in range(8):
            guard(f"{'PDD' if is_pdd else 'DMD2'} forward {step+1}/8")
            times, indices = MiniMaxH3SetTimestepsStep.build_row_timesteps(
                fields["video_indices"].cpu(), fields["audio_indices"].cpu(), cv, ca,
                prompt_embeds.shape[1], plan.clean_time("video", step),
                plan.clean_time("audio", step), max(plan.clean_time("video", step), pipe.keyframe_noise_aug), 1.)
            with ExitStack() as stack:
                if adaln_table is not None:
                    stack.enter_context(model._adaln_table.use(times))
                if is_pdd:
                    start, end = plan.indices[step:step+2]
                    stack.enter_context(model.proj_out.fuse(start, end, plan.weights['video']))
                    stack.enter_context(model.audio_proj_out.fuse(start, end, plan.weights['audio']))
                vp, ap = model(hidden_states=video, audio_hidden_states=audio,
                               encoder_hidden_states=prompt_embeds, timestep=times.to(device),
                               timestep_indices=indices.to(device), return_dict=False, **fields)
            diagnostics.forward_count += 1
            video = plan.update(video, vp, "video", step, cv)
            audio = plan.update(audio, ap, "audio", step, ca)
            step_contract = dict(head_start=start, head_end=end) if is_pdd else dict(dmd2_rung=RUNGS[step])
            diagnostics.checkpoint(f"step-{step+1:02d}", {"video": video, "audio": audio,
                                   "video_prediction": vp, "audio_prediction": ap, "times": times,
                                   "timestep_indices": indices}, **step_contract)
            if step == 0 and diagnostics.checkpoint("first-denoise"):
                return diagnostics.stopped("first-denoise")
        state.set("latents", video[0])
        state.set("audio_latents", audio[0])
        state = _run_block(core.sub_blocks["after_denoise"], pipe, state)
    finally:
        stream.close()
        diagnostics.streaming['dit'] = stream.stats_snapshot()
        pipe.update_components(**{component: None})
        del model, stream
        _clear_device(device)
        guard("DiT released")
    if diagnostics.checkpoint("denoise", {"video": state.get("latents"), "audio": state.get("audio_latents")}):
        return diagnostics.stopped("denoise")
    guard("DiT released before decode")
    load_local_components(pipe, ["vae", "audio_vae"], shared_components)
    state = decode_output(blocks.sub_blocks['decode'], pipe, state, device, guard, 'cached', diagnostics,
                          vae_mxfp8=vae_mxfp8, debug_checks=debug_checks)
    guard("decode complete")
    if diagnostics.checkpoint("decode", {"audio": state.get("audio")}, sampling_rate=state.get("sampling_rate")):
        return diagnostics.stopped("decode")
    return {name: state.get(name) for name in ("videos", "audio", "sampling_rate")}


def save_result(result, output, *, num_frames=None):
    import subprocess
    import tempfile
    import wave
    import imageio_ffmpeg
    import numpy as np
    from diffusers.utils import export_to_video
    # Video postprocessing returns NumPy, unlike the tensor-valued diagnostics.
    # Reject bad decoded values before uint8/PCM conversion can hide NaN/Inf.
    frames = result["videos"][0]
    already_rgb8 = isinstance(frames, np.ndarray) and frames.dtype == np.uint8
    if not already_rgb8 and not np.isfinite(np.asarray(frames)).all():
        raise FloatingPointError("Non-finite decoded video before export")
    if not bool(torch.isfinite(result["audio"]).all().item()):
        raise FloatingPointError("Non-finite decoded audio before export")
    if num_frames is not None:
        if not 0 < num_frames <= len(frames):
            raise ValueError('Requested export frames exceed decoded video')
        frames = frames[:num_frames]
    sampling_rate = int(result["sampling_rate"])
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="fasth3-mux-") as tmp:
        tmp = Path(tmp)
        video_path, audio_path = tmp/"video.mp4", tmp/"audio.wav"
        if already_rgb8:
            import imageio
            with imageio.get_writer(str(video_path), fps=24, quality=5, macro_block_size=16) as writer:
                for frame in frames:
                    writer.append_data(frame)
        else:
            export_to_video(frames, str(video_path), fps=24)
        audio = result["audio"].detach().float().cpu().numpy()[0].T
        if num_frames is not None:
            samples = (num_frames * sampling_rate + 12) // 24
            missing = samples - len(audio)
            if missing > (sampling_rate + 23) // 24:
                raise ValueError('Decoded audio is more than one frame shorter than the requested export')
            audio = np.pad(audio, ((0, missing), (0, 0))) if missing > 0 else audio[:samples]
        pcm = (np.clip(audio, -1, 1)*32767).astype("<i2")
        with wave.open(str(audio_path), "wb") as writer:
            writer.setnchannels(2)
            writer.setsampwidth(2)
            writer.setframerate(sampling_rate)
            writer.writeframes(pcm.tobytes())
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(video_path), "-i", str(audio_path),
                        "-c:v", "copy", "-c:a", "aac", "-movflags", "+faststart",
                        *(["-shortest"] if num_frames is None else []), str(output)], check=True)
    return output
