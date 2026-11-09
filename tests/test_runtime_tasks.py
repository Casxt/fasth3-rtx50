import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from fasth3_pipeline.dmd2 import DMD2Plan, RUNGS
from fasth3_pipeline.long_video import _project_output_rows
from fasth3_pipeline.pdd import PDDPlan
from fasth3_pipeline.runner import (
    _prepare_denoising, _run_block, configure_lightweight_components, layout_from_state, run_h3, validate_task_inputs,
)


def test_generation_tasks_require_their_own_inputs():
    image = Image.new('RGB', (16, 16))
    image_reference = SimpleNamespace(kind='image')
    audio_reference = SimpleNamespace(kind='audio')
    assert validate_task_inputs('t2va') == 't2va'
    assert validate_task_inputs('f2va', first_frame=image) == 'f2va'
    assert validate_task_inputs('fl2va', first_frame=image, last_frame=image) == 'fl2va'
    assert validate_task_inputs('r2va', references=[image_reference, audio_reference]) == 'r2va'
    with pytest.raises(ValueError, match='endpoint'):
        validate_task_inputs('f2va', first_frame=image, last_frame=image)
    with pytest.raises(ValueError, match='endpoint'):
        validate_task_inputs('fl2va', first_frame=image)
    with pytest.raises(ValueError, match='reference media'):
        validate_task_inputs('t2va', references=[image_reference])
    with pytest.raises(ValueError, match='image or video'):
        validate_task_inputs('r2va', references=[audio_reference])


def test_fl2va_rejects_unaligned_duration_before_loading_models():
    endpoint = Image.new('RGB', (16, 16))
    with pytest.raises(ValueError, match='191 rounds up to 192'):
        run_h3(task='fl2va', base_model=None, transformer_path=None, text_encoder_path=None,
               prompt='Move slowly.', guard=None, first_frame=endpoint, last_frame=endpoint, num_frames=191)


def test_v2_ordinary_output_projection_preserves_packed_row_order():
    projection = torch.nn.Linear(4, 3)
    packed = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4)
    rows = torch.tensor([3, 1, 4])
    time_indices = torch.tensor([0, 1, 0, 2, 1])
    time_values = torch.tensor([.1, .2, .3])

    def norm(hidden, values, indices):
        return hidden * values[indices, None]

    actual = _project_output_rows(packed, rows, time_indices, norm, time_values, projection)
    expected = projection(packed[:, rows] * time_values[time_indices[rows], None])
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('modality,shift', [('video', 10.), ('audio', 3.)])
def test_v2_dmd2_uses_published_fp32_grid_and_unscaled_noise(modality, shift):
    plan = DMD2Plan()
    base = torch.tensor([.999, .874, .749, .624, .5, .375, .25, .125, 0.], dtype=torch.float32)
    sigma = shift * base / (1 + (shift - 1) * base)
    clean = 1 - sigma[:-1]
    assert torch.equal(plan.sigmas[modality], sigma)
    assert len(RUNGS) == len(clean) == 8
    sample = torch.randn(1, 7, 3, generator=torch.Generator().manual_seed(23))
    assert plan.scale_initial_noise(sample, modality) is sample
    for step, timestep in enumerate(clean):
        prediction = torch.full_like(sample, .0625 * (step + 1)).bfloat16()
        estimate = sample + (1 - timestep) * prediction.float()
        ratio = sigma[step + 1] / sigma[step]
        expected = ratio * sample + (1 - ratio) * estimate
        assert plan.clean_time(modality, step) == timestep.item()
        sample = plan.update(sample, prediction, modality, step)
        assert torch.equal(sample, expected)


def test_v2_metadata_rejects_wrong_schedule_before_model_load(tmp_path):
    contract = {
        'schema_version': 'fasth3-inference-contract-v1', 'task': 't2av',
        'dmd_denoising_steps': list(RUNGS), 'transformer_forwards': 8,
        'num_inference_steps': 9, 'video_scheduler_shift': 10., 'audio_scheduler_shift': 3.,
        'attention_backend': 'VIDEO_SPARSE_ATTN_H3', 'vsa_tile_size': 64, 'vsa_sparsity': .8,
        'guidance_scale': 1.,
    }
    for subfolder, shift in (('scheduler', 10.), ('audio_scheduler', 3.)):
        directory = tmp_path / subfolder
        directory.mkdir()
        (directory / 'scheduler_config.json').write_text(json.dumps(
            {'_class_name': 'MiniMaxH3Scheduler', 'shift': shift}), encoding='utf-8')
    path = tmp_path / 'fastvideo_inference.json'
    path.write_text(json.dumps(contract), encoding='utf-8')
    assert isinstance(DMD2Plan.from_checkpoint(tmp_path, 'cpu'), DMD2Plan)
    contract['vsa_tile_size'] = 128
    path.write_text(json.dumps(contract), encoding='utf-8')
    with pytest.raises(ValueError, match='vsa_tile_size'):
        DMD2Plan.from_checkpoint(tmp_path, 'cpu')


def _prepared_task(task, anchors):
    from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import MiniMaxH3Blocks
    from diffusers.modular_pipelines.modular_pipeline import PipelineState

    blocks = MiniMaxH3Blocks()
    pipe = blocks.init_pipeline()
    configure_lightweight_components(pipe)
    state = PipelineState()
    for param in blocks.inputs:
        if param.name:
            state.set(param.name, param.default, param.kwargs_type)
    first = Image.new('RGB', (48, 32), (200, 5, 10)) if anchors else None
    last = Image.new('RGB', (32, 48), (5, 10, 200)) if len(anchors) == 2 else None
    for key, value in dict(image=first, last_image=last, height=64, width=64, num_frames=124,
                           generator=torch.Generator().manual_seed(29),
                           text_token_tags=torch.ones(7, dtype=torch.long), prompt_embeds=torch.zeros(1, 7, 3),
                           condition_latents=[torch.zeros(1, 24, 1, 4, 4) for _ in anchors]).items():
        state.set(key, value)
    state = _run_block(blocks.sub_blocks['before_encode'], pipe, state)
    if anchors:
        assert tuple(state.get('keyframe_anchors')) == anchors
        assert [frame.size for frame in state.get('keyframes')] == [(64, 64)] * len(anchors)
    core = blocks.sub_blocks['denoise'].sub_blocks['fl2va' if anchors else 't2va']
    return pipe, _prepare_denoising(core, pipe, state, task)


@pytest.mark.parametrize('task,anchors', [('t2va', ()), ('f2va', ('first',)), ('fl2va', ('first', 'last'))])
def test_official_core_packs_and_freezes_endpoint_latents(task, anchors):
    pipe, state = _prepared_task(task, anchors)
    layout = layout_from_state(state, pipe.patch_size, 'cpu', task=task)
    assert layout.inverse.numel() == len(state.get('token_tags'))
    assert layout.tile_size == (128 if anchors else 64)
    assert len(layout.video_spans) == 1
    condition_rows = state.get('num_condition_video_rows')
    assert condition_rows == 4 * len(anchors)
    positions = state.get('position_ids')[state.get('video_indices')]
    if anchors:
        # The first anchor shares the first frame time. The released last anchor
        # is at the end of the last latent span minus one pixel-frame step.
        assert torch.equal(positions[:4], positions[condition_rows:condition_rows + 4])
        if len(anchors) == 2:
            expected_last = positions[-4:].clone()
            expected_last[:, 0] += 5.0
            torch.testing.assert_close(positions[4:8], expected_last, rtol=0, atol=1e-12)
        plan = PDDPlan.create()
        latent = state.get('latents')[None]
        frozen = latent[:, :condition_rows].clone()
        latent = plan.scale_initial_noise(latent, 'video', condition_rows)
        for step in range(8):
            latent = plan.update(latent, torch.ones_like(latent), 'video', step, condition_rows)
            assert torch.equal(latent[:, :condition_rows], frozen)



def test_frame_control_shares_target_noise_without_adding_a_condition():
    from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents

    _, opening = _prepared_task('f2va', ('first',))
    pipe, both = _prepared_task('fl2va', ('first', 'last'))
    opening_rows = opening.get('num_condition_video_rows')
    both_rows = both.get('num_condition_video_rows')
    assert (opening_rows, both_rows) == (4, 8)
    assert torch.equal(opening.get('latents')[:opening_rows], both.get('latents')[:opening_rows])

    generator = torch.Generator().manual_seed(29)
    for _ in range(2):
        torch.randn((1, 24, 1, 4, 4), generator=generator)
    video_shape = (1, 24, both.get('num_latent_frames'), both.get('latent_height'), both.get('latent_width'))
    expected_video = patchify_video_latents(torch.randn(video_shape, generator=generator), pipe.patch_size)
    expected_audio = torch.randn(both.get('audio_latents').shape, generator=generator)
    assert torch.equal(both.get('latents')[both_rows:], expected_video)
    assert torch.equal(opening.get('latents')[opening_rows:], expected_video)
    assert torch.equal(both.get('audio_latents'), expected_audio)
    assert torch.equal(opening.get('audio_latents'), expected_audio)
