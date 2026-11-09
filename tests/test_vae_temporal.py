import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from fasth3_pipeline.vae_temporal import (
    _decode_chunks, _decode_clip_pairwise, _encode_video, _encoder_fusion_blocks,
    _vae_encoder_fusion, decode_video_to_host, encode_references,
)


class TinyTemporalVAE:
    """Use the pinned temporal routines with a tiny deterministic spatial core."""
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3

    _encode = AutoencoderKLMiniMaxH3._encode
    _decode = AutoencoderKLMiniMaxH3._decode
    _blend = AutoencoderKLMiniMaxH3._blend
    tokens_chunk_size = 5
    token_overlap = 2
    frame_pre_padding = 3
    frame_overlap = 5
    temporal_compression_ratio = 4

    def __init__(self):
        self.config = SimpleNamespace(clip_length=17, token_drop=3, latents_mean=(0.25,) * 3,
                                      latents_std=(0.5,) * 3)
        self.encoder = nn.Linear(1, 1)
        self.post_quant_conv = nn.Linear(1, 1)

    def _encode_clip(self, pixels):
        values = pixels[:, :, ::4]
        return torch.cat((values, values / 2), dim=1)

    def encode(self, pixels, return_dict=False):
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
        return (DiagonalGaussianDistribution(self._encode(pixels)),)

    def _decode_clip(self, latent):
        return latent.repeat_interleave(self.temporal_compression_ratio, dim=2)


def _pipe():
    return SimpleNamespace(vae=TinyTemporalVAE(), _execution_device='cpu', pixel_mean=(.485, .456, .406),
                           pixel_std=(.229, .224, .225), keyframe_encode_seed=42,
                           vae_frames_per_chunk=17, vae_latents_per_chunk=5)


@pytest.mark.parametrize('frames', [1, 12, 36, 362])
@pytest.mark.parametrize('layout', ['contiguous', 'read-only', 'strided'])
def test_bounded_reference_encode_matches_released_sample_and_snap(frames, layout):
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition

    pipe = _pipe()
    video = np.random.default_rng(7).integers(0, 256, (frames, 2, 3, 3), dtype=np.uint8)
    if layout == 'read-only':
        video.flags.writeable = False
    elif layout == 'strided':
        video = video[:, :, ::-1]
    snap = max(1, (frames - 5) // 17) * 17 + 5
    pixels = torch.from_numpy(video[:snap].copy()).permute(3, 0, 1, 2)[None]
    expected = encode_vae_condition(pipe.vae, pixels, pipe.pixel_mean, pipe.pixel_std,
                                    pipe.keyframe_encode_seed)
    assert torch.equal(_encode_video(pipe, video), expected)


@pytest.mark.parametrize('kind', ['image', 'video'])
def test_visual_only_reference_encodes_without_loading_audio_vae(kind):
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
    from diffusers.modular_pipelines.modular_pipeline import PipelineState

    pipe = _pipe()
    frames = np.zeros((1 if kind == 'image' else 12, 2, 3, 3), dtype=np.uint8)
    state = PipelineState()
    state.set('normalized_references', [SimpleNamespace(kind=kind, image=frames[0], frames=frames, has_audio=False)])

    encode_references(pipe, state)

    pixels = torch.from_numpy(frames).permute(3, 0, 1, 2)[None]
    expected = encode_vae_condition(pipe.vae, pixels, pipe.pixel_mean, pipe.pixel_std, pipe.keyframe_encode_seed)
    assert torch.equal(state.get('condition_latents')[0], expected)
    assert state.get('audio_condition_latents') == []


def test_encoder_fusion_checks_architecture_and_restores_scoped_forwards(monkeypatch):
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoEncoder3d

    with torch.device('meta'):
        vae = SimpleNamespace(encoder=MiniMaxH3VideoEncoder3d())
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (12, 0))
    blocks = _encoder_fusion_blocks(vae)
    assert len(blocks) == 2
    layer = blocks[0].resnets[0]
    with torch.no_grad(), pytest.raises(RuntimeError, match='test failure'):
        with _vae_encoder_fusion(vae, 'cuda'):
            assert 'forward' in layer.__dict__
            raise RuntimeError('test failure')
    assert 'forward' not in layer.__dict__
    layer.norm1.eps = 1e-5
    assert _encoder_fusion_blocks(vae) == ()


@pytest.mark.skipif(os.environ.get('FASTH3_TEST_CUDA') != '1', reason='requires a coordinated GPU window')
def test_cute_encoder_ingress_preserves_bf16_rounding_and_padding():
    from fasth3_kernels._cute_vae_ingress import prepare_conv_input
    from fasth3_platform import BudgetGuard
    import torch.nn.functional as F

    with BudgetGuard(), torch.no_grad():
        # Bracket representable SiLU inputs on either side of many BF16 rounding ties.
        bits = torch.arange(0x3b80, 0x4280, device='cuda', dtype=torch.int16)
        ties = (bits.view(torch.bfloat16).float() + (bits + 1).view(torch.bfloat16).float()) * .5
        low, high = ties.clone(), 2 * ties
        for _ in range(30):
            mid = (low + high) * .5
            below = F.silu(mid) < ties
            low = torch.where(below, mid, low)
            high = torch.where(below, high, mid)
        extremes = torch.tensor([-104., -90., -16., -3., -1.25, -0., 0., 80.], device='cuda')
        values = torch.cat((low, high, -low, -high, extremes))
        for frames in (1, 17):
            frame_major = values.repeat(1 + frames * 32 * 16**2 // values.numel())[:frames * 32 * 16**2]
            x = frame_major.view(1, frames, 32, 16, 16).transpose(1, 2)
            for layout in (x, x.contiguous(memory_format=torch.channels_last_3d)):
                for silu, padding in ((True, 1), (False, 0)):
                    expected = F.silu(layout) if silu else layout
                    expected = F.pad(expected, (padding, 1, padding, 1, 0, 0), mode='reflect')
                    expected = F.pad(expected, (0, 0, 0, 0, 2, 0)).to(torch.bfloat16)
                    actual = prepare_conv_input(layout, padding, silu=silu)
                    assert torch.equal(actual.contiguous().view(torch.uint16), expected.contiguous().view(torch.uint16))


@pytest.mark.parametrize('latent_frames', [7, 8, 9, 10, 37, 107])
def test_temporal_stream_preserves_overlap_and_tail(latent_frames):
    pipe = _pipe()
    z = torch.rand((1, 3, latent_frames, 2, 3), generator=torch.Generator().manual_seed(4))
    expected = pipe.vae._decode(z)
    assert torch.equal(torch.cat(list(_decode_chunks(pipe.vae, z)), dim=2), expected)


def test_temporal_video_rgb_conversion_matches_released_truncation():
    from diffusers.modular_pipelines.modular_pipeline import PipelineState

    pipe = _pipe()
    latent = torch.rand((1, 3, 107, 2, 3), generator=torch.Generator().manual_seed(9))
    expected = pipe.vae._decode(latent * .5 + .25)
    std = torch.tensor(pipe.pixel_std).view(1, -1, 1, 1, 1)
    mean = torch.tensor(pipe.pixel_mean).view(1, -1, 1, 1, 1)
    expected = ((expected * std + mean).clamp(0, 1).permute(0, 2, 3, 4, 1).numpy() * 255).astype(np.uint8)
    state = PipelineState()
    state.set('latents', latent)
    state.set('num_frames', expected.shape[1])
    np.testing.assert_array_equal(decode_video_to_host(pipe, state).get('videos'), expected)


def test_spatial_tile_pair_keeps_batch_and_single_tail_in_stitch_order():
    vae = TinyTemporalVAE()
    original = vae.AutoencoderKLMiniMaxH3
    vae._split_tiles = original._split_tiles.__get__(vae)
    vae._stitch_tiles = original._stitch_tiles.__get__(vae)
    vae.use_tiling = True
    vae.spatial_compression_ratio = 1
    vae.tile_sample_min_height = vae.tile_sample_min_width = 4
    vae.tile_sample_min_overlap_height = vae.tile_sample_min_overlap_width = 1
    vae.post_quant_conv = nn.Identity()
    batches = []

    def decoder(x):
        batches.append(x.shape[0])
        return x * 2

    vae.decoder = decoder
    latent = torch.arange(2 * 3 * 9, dtype=torch.float32).reshape(2, 1, 1, 3, 9)
    expected = original._decode_clip(vae, latent)
    batches.clear()
    assert torch.equal(_decode_clip_pairwise(vae, latent), expected)
    assert batches == [4, 2]
