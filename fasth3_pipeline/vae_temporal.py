"""Bounded Ref2VA VAE input and output using the pinned H3 temporal schedule."""
from contextlib import contextmanager, ExitStack
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F


def _channel_values(values, device=None):
    return torch.tensor(values, device=device).view(1, -1, 1, 1, 1)


def _encoder_fusion_blocks(vae):
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import (
        MiniMaxH3VideoCausalConv3d, MiniMaxH3VideoDownBlock3d, MiniMaxH3VideoDownsample3d,
        MiniMaxH3VideoGroupNorm, MiniMaxH3VideoResnetBlock3d,
    )

    def compatible_conv(conv, cin, cout, padding, stride):
        return (type(conv) is MiniMaxH3VideoCausalConv3d and conv.kernel_size == (3, 3, 3)
                and (conv.in_channels, conv.out_channels) == (cin, cout)
                and conv.stride == stride and conv.padding == (0, 0, 0) and conv.groups == 1
                and conv.dilation == (1, 1, 1) and conv.spatial_padding == padding
                and conv.temporal_padding == 2 and conv.spatial_padding_mode == 'reflect')

    blocks = tuple(getattr(vae.encoder, 'down_blocks', ()))[:2]
    if len(blocks) != 2:
        return ()
    for block, cout, temporal_stride in zip(blocks, (128, 256), (1, 2)):
        if (type(block) is not MiniMaxH3VideoDownBlock3d or len(block.resnets) != 2
                or block.downsamplers is None or len(block.downsamplers) != 1):
            return ()
        for layer, cin in zip(block.resnets, (128, cout)):
            if (type(layer) is not MiniMaxH3VideoResnetBlock3d
                    or not compatible_conv(layer.conv1, cin, cout, 1, (1, 1, 1))
                    or not compatible_conv(layer.conv2, cout, cout, 1, (1, 1, 1))
                    or any(type(norm) is not MiniMaxH3VideoGroupNorm or norm.num_groups != 32
                           or norm.eps != 1e-6 or not norm.affine for norm in (layer.norm1, layer.norm2))):
                return ()
        layer = block.downsamplers[0]
        if (type(layer) is not MiniMaxH3VideoDownsample3d or layer.spatial_stride != 2
                or layer.spatial_padding_mode != 'reflect'
                or not compatible_conv(layer.conv, cout, cout, 0, (temporal_stride, 2, 2))):
            return ()
    return blocks


def _compatible_vae_tile(x, conv, size):
    return (x.ndim == 5 and x.shape[:2] == (1, conv.in_channels)
            and x.shape[2] in (1, 17) and x.shape[-2:] == (size, size)
            and x.is_cuda and x.dtype == torch.bfloat16 and x.data_ptr() % 16 == 0
            and (x.is_contiguous() or x.is_contiguous(memory_format=torch.channels_last_3d))
            and conv.weight.device == x.device)


def _frame_group_norm(norm, x):
    b, c, t, h, w = x.shape
    frames = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, 1, h, w)
    # Preserve the pinned GroupNorm reduction and leave the frame-major storage in place.
    frames = torch.nn.GroupNorm.forward(norm, frames)
    return frames.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)


def _fused_ingress_conv(conv, x, *, silu, padding):
    from fasth3_kernels._cute_vae_ingress import prepare_conv_input

    x = prepare_conv_input(x, padding, silu=silu)
    weight = conv.weight.to(dtype=torch.bfloat16, memory_format=torch.channels_last_3d)
    bias = None if conv.bias is None else conv.bias.to(torch.bfloat16)
    return F.conv3d(x, weight, bias, stride=conv.stride)


def _fused_vae_resnet(layer, original, size, x):
    if not _compatible_vae_tile(x, layer.conv1, size):
        return original(x)
    residual = x
    x = _fused_ingress_conv(layer.conv1, _frame_group_norm(layer.norm1, x), silu=True, padding=1)
    x = _fused_ingress_conv(layer.conv2, _frame_group_norm(layer.norm2, x), silu=True, padding=1)
    if layer.conv_shortcut is not None:
        residual = layer.conv_shortcut(residual)
    return residual + x


def _fused_vae_downsample(layer, original, size, x):
    if not _compatible_vae_tile(x, layer.conv, size):
        return original(x)
    return _fused_ingress_conv(layer.conv, x, silu=False, padding=0)


@contextmanager
def _vae_encoder_fusion(vae, device):
    device = torch.device(device)
    blocks = (_encoder_fusion_blocks(vae) if device.type == 'cuda' and not torch.is_grad_enabled()
              and torch.cuda.get_device_capability(device) == (12, 0) else ())
    with ExitStack() as stack:
        for block, size in zip(blocks, (256, 128)):
            for layers, fused in ((block.resnets, _fused_vae_resnet),
                                  (block.downsamplers, _fused_vae_downsample)):
                for layer in layers:
                    if 'forward' in layer.__dict__:
                        continue
                    original = layer.forward
                    stack.callback(delattr, layer, 'forward')
                    layer.forward = partial(fused, layer, original, size)
        yield


def _encode_video(pipe, frames):
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    from diffusers.models.modeling_utils import get_parameter_dtype

    vae, device = pipe.vae, pipe._execution_device
    clip_length = vae.config.clip_length
    # Same snap-down as the pinned Ref2VA reference encoder. A very short
    # reference still uses only its available frames before repeating the tail.
    frame_limit = (max(1, (len(frames) - pipe.vae_latents_per_chunk) // pipe.vae_frames_per_chunk)
                   * pipe.vae_frames_per_chunk + pipe.vae_latents_per_chunk)
    frames = frames[:frame_limit]
    pixel_mean = _channel_values(pipe.pixel_mean, device)
    pixel_std = _channel_values(pipe.pixel_std, device)
    compute_dtype = get_parameter_dtype(vae.encoder)
    moments = []
    with _vae_encoder_fusion(vae, device):
        for start in range(0, len(frames), clip_length):
            part = frames[start:start + clip_length]
            if len(part) < clip_length and len(frames) > 1:
                padded = np.empty((clip_length, *part.shape[1:]), dtype=part.dtype)
                padded[:len(part)] = part
                padded[len(part):] = part[-1]
                part = padded
            elif not (part.flags.c_contiguous and part.flags.writeable):
                part = np.array(part, copy=True, order='C')
            pixels = torch.from_numpy(part).to(device).permute(3, 0, 1, 2)[None]
            pixels = (pixels.float().div(255.0) - pixel_mean) / pixel_std
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=pixels.is_cuda):
                encoded = vae._encode_clip(pixels.to(compute_dtype))
            # Keep posterior sampling in FP32 before the released FP16 rounding.
            moments.append(encoded.float() if pixels.is_cuda else encoded)
            del encoded
            del pixels
    packed = moments[0] if len(moments) == 1 else torch.cat(moments, dim=2)
    del moments
    if vae.config.token_drop and len(frames) > 1:
        packed = packed[:, :, :-vae.config.token_drop]
    posterior = DiagonalGaussianDistribution(packed)
    generator = torch.Generator().manual_seed(pipe.keyframe_encode_seed)
    # Sample once over the original shape, preserving seed-to-element mapping.
    latent = posterior.sample(generator=generator).to(torch.float16).float().cpu()
    return (latent - _channel_values(vae.config.latents_mean)) / _channel_values(vae.config.latents_std)


def encode_references(pipe, state):
    device = pipe._execution_device
    references = state.get('normalized_references')
    if any(reference.has_audio for reference in references):
        audio_mean = torch.tensor(pipe.audio_vae.config.latents_mean).view(1, 1, -1)
        audio_std = torch.tensor(pipe.audio_vae.config.latents_std).view(1, 1, -1)
    visual, audio = [], []
    for reference in references:
        if reference.kind == 'image':
            visual.append(_encode_video(pipe, np.array(reference.image)[None]))
        elif reference.kind == 'video':
            visual.append(_encode_video(pipe, reference.frames))
        if reference.has_audio:
            posterior = pipe.audio_vae.encode(reference.audio.to(device)[:, None], return_dict=False)[0]
            latent = posterior.mode().float().cpu().transpose(1, 2)
            audio.append(((latent - audio_mean) / audio_std).reshape(-1, pipe.audio_latent_channels))
    state.set('condition_latents', visual)
    state.set('audio_condition_latents', audio)
    return state


def encode_keyframes(pipe, state):
    state.set('condition_latents', [_encode_video(pipe, np.array(frame)[None])
                                    for frame in state.get('keyframes')])
    state.set('audio_condition_latents', [])
    return state


def _decode_clip_pairwise(vae, z):
    """Batch only equal spatial tiles and retain Diffusers' seam order."""
    if not vae.use_tiling:
        return vae._decode_clip(z)

    ratio = vae.spatial_compression_ratio
    ys, heights, y_overlap = vae._split_tiles(z.shape[-2] * ratio, vae.tile_sample_min_height,
                                             vae.tile_sample_min_overlap_height)
    xs, widths, x_overlap = vae._split_tiles(z.shape[-1] * ratio, vae.tile_sample_min_width,
                                            vae.tile_sample_min_overlap_width)
    if len(ys) * len(xs) < 2 or len(set(heights)) != 1 or len(set(widths)) != 1:
        return vae._decode_clip(z)

    tiles = [z[..., y // ratio:y // ratio + height // ratio, x // ratio:x // ratio + width // ratio]
             for y, height in zip(ys, heights) for x, width in zip(xs, widths)]
    decoded = []
    for start in range(0, len(tiles), 2):
        group = tiles[start:start + 2]
        batch = torch.cat(group, dim=0) if len(group) == 2 else group[0]
        decoded.extend(vae.decoder(vae.post_quant_conv(batch)).split(z.shape[0], dim=0))
    rows = [decoded[start:start + len(xs)] for start in range(0, len(decoded), len(xs))]
    return vae._stitch_tiles(rows, y_overlap, x_overlap)


def _decode_chunks(vae, z, *, pair_tiles=False):
    """Yield completed chunks after the released VAE's GPU temporal blend."""
    size = vae.tokens_chunk_size
    drop = vae.config.token_drop
    ratio = vae.temporal_compression_ratio
    count = size * ratio
    padded = (-(z.shape[2] + drop)) % size
    num_chunks = (z.shape[2] + drop + padded) // size - int(drop > 0)
    tail = vae.config.clip_length % ratio
    discard = sum(tail if tail and (z.shape[2] + k) % size == 0 else ratio for k in range(padded))
    if padded:
        z = torch.cat((z, z[:, :, -1:].expand(-1, -1, padded, -1, -1)), dim=2)

    overlap = None
    for index in range(num_chunks):
        start = index * size
        latent_clip = z[:, :, start:start + size + vae.token_overlap]
        clip = _decode_clip_pairwise(vae, latent_clip) if pair_tiles else vae._decode_clip(latent_clip)
        chunk = clip[:, :, vae.frame_pre_padding:count]
        if overlap is not None:
            chunk = vae._blend(overlap, chunk, vae.frame_overlap, dim=-3)
        if drop:
            overlap = clip[:, :, count + vae.frame_pre_padding:2 * count].contiguous()
        if discard and index == num_chunks - 1:
            overlap_frames = 0 if overlap is None else overlap.shape[2]
            if discard > overlap_frames:
                chunk = chunk[:, :, :-(discard-overlap_frames)]
            if overlap is not None:
                overlap = overlap[:, :, :max(0, overlap_frames-discard)]
        yield chunk
        del chunk, clip
    if overlap is not None and overlap.shape[2]:
        yield overlap


def decode_video_to_host(pipe, state, *, pair_tiles=False):
    """Retain only the current decoded temporal clip and blend state on GPU."""
    from diffusers.models.modeling_utils import get_parameter_dtype

    vae, device = pipe.vae, pipe._execution_device
    z = state.get('latents')
    z = z * _channel_values(vae.config.latents_std, device) + _channel_values(vae.config.latents_mean, device)
    z = z.to(get_parameter_dtype(vae.post_quant_conv))
    mean, std = _channel_values(pipe.pixel_mean, device), _channel_values(pipe.pixel_std, device)
    expected_frames = state.get('num_frames')
    result = None
    chunks, offset = [], 0
    for decoded in _decode_chunks(vae, z, pair_tiles=pair_tiles):
        host = _decode_rgb8(decoded, mean, std)
        if expected_frames is None:
            chunks.append(host)
        else:
            if result is None:
                result = np.empty((host.shape[0], expected_frames, *host.shape[2:]), dtype=np.uint8)
            count = host.shape[1]
            if offset + count > expected_frames:
                raise RuntimeError('VAE emitted more frames than the requested aligned duration')
            result[:, offset:offset + count] = host
            offset += count
        del host, decoded
    if result is not None and offset != expected_frames:
        raise RuntimeError('VAE emitted fewer frames than the requested aligned duration')
    state.set('videos', np.concatenate(chunks, axis=1) if result is None else result)
    return state


def _decode_rgb8(decoded, mean, std):
    if (decoded.is_cuda and decoded.shape[1] == 3 and decoded.dtype in (torch.bfloat16, torch.float32)
            and mean.dtype == std.dtype == torch.float32
            and torch.cuda.get_device_capability(decoded.device) == (12, 0)):
        from fasth3_kernels._cute_vae_rgb8 import rgb8_to_host
        return rgb8_to_host(decoded, mean, std)
    rgb = (decoded.float() * std).add_(mean)
    if not bool(torch.isfinite(rgb).all()):
        raise FloatingPointError('Non-finite decoded video before export')
    return rgb.clamp_(0, 1).mul_(255).to(torch.uint8).permute(0, 2, 3, 4, 1).contiguous().cpu().numpy()
