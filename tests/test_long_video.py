import os

import pytest
import torch
from torch import nn

from fasth3_pipeline.long_video import (
    _add_residual, _ffn_residual, _modulate, _project_into, _project_output_rows,
    aligned_video_frames, allow_released_maximum,
)


def test_aligned_released_limit_is_instance_local():
    class Pipeline:
        fps = 24
        max_duration = 15.0

    a, b = Pipeline(), Pipeline()
    allow_released_maximum(a)
    assert aligned_video_frames(360) == 362
    assert a.max_duration == 362 / 24
    assert b.max_duration == 15.0


def test_host_attention_scratch_reuses_and_releases_staging(monkeypatch):
    from types import SimpleNamespace
    from fasth3_pipeline import long_video

    drained = []

    def make_slots(host, columns, device):
        return SimpleNamespace(copy_stream=SimpleNamespace(synchronize=lambda: drained.append(True)))

    monkeypatch.setattr(long_video, '_PinnedHeadOutput', make_slots)
    scratch = long_video.LongVideoScratch()
    flat = torch.empty(3, 4, dtype=torch.bfloat16)
    host, slots = scratch.acquire(flat, 4, 2)
    next_host, next_slots = scratch.acquire(flat, 4, 2)
    assert next_host is host
    assert next_slots is slots
    shape = (1, 3, 4)
    assert scratch.prepare_cuda(shape, 'cpu', SimpleNamespace(available_bytes=lambda: 47)) == 0
    assert scratch.prepare_cuda(shape, 'cpu', SimpleNamespace(available_bytes=lambda: 48)) == 48
    assert scratch.packed.shape == scratch.modulated.shape == shape
    assert scratch.packed.data_ptr() != scratch.modulated.data_ptr()
    scratch.close()
    scratch.close()
    assert scratch.host is scratch.pinned is None
    assert scratch.packed is scratch.modulated is None
    assert drained == [True]


def test_host_projection_can_overwrite_disposable_modulation(monkeypatch):
    from types import SimpleNamespace
    from fasth3_pipeline import long_video
    from fasth3_pipeline.attention import H3VSAProcessor

    class Projection:
        out_features = 4

        def __call__(self, x, *, output):
            output.copy_(x[:, :self.out_features])

    def attend(processor, attn, flat, rotary, first, last, cache):
        return (flat[:, :1] + first + 1).expand(-1, (last - first) * 128)

    monkeypatch.setattr(long_video, '_PinnedHeadOutput', lambda *args: None)
    monkeypatch.setattr(long_video, '_attend_heads', attend)
    attn = SimpleNamespace(heads=2, to_out=(Projection(), nn.Identity()))
    x = torch.arange(16, dtype=torch.bfloat16).reshape(1, 4, 4)
    reference = long_video.grouped_attention(H3VSAProcessor(None), attn, x, None, heads_per_group=1)
    modulated = x.clone()
    processor = H3VSAProcessor(None, reuse_input=True)
    output = long_video.grouped_attention(processor, attn, modulated, None, heads_per_group=1)
    assert output.data_ptr() == modulated.data_ptr()
    torch.testing.assert_close(output, reference, rtol=0, atol=0)


def test_long_modulation_preserves_rounding_at_chunk_boundary():
    torch.manual_seed(34)
    x = torch.randn(1, 2051, 32, dtype=torch.bfloat16)
    shift, scale, gate = (torch.randn(4, 32, dtype=x.dtype) for _ in range(3))
    indices = torch.arange(x.shape[1]) % len(shift)
    norm = nn.RMSNorm(32, eps=1e-5, dtype=x.dtype)
    expected = norm(x) * (1 + scale[indices]) + shift[indices]
    actual = _modulate(x, norm, shift, scale, indices)
    assert torch.equal(actual, expected)
    torch.testing.assert_close(_add_residual(actual, x.clone(), gate, indices),
                               x + gate[indices] * expected, rtol=0, atol=0)


def test_ffn_residual_uses_bounded_chunks_and_original_rounding():
    from types import SimpleNamespace

    torch.manual_seed(35)
    x = torch.randn(1, 4101, 32, dtype=torch.bfloat16)
    shift, scale, gate = (torch.randn(4, 32) for _ in range(3))
    indices = torch.arange(x.shape[1]) % len(shift)
    norm = nn.RMSNorm(32, eps=1e-5, dtype=x.dtype)
    calls = []

    def ff(chunk):
        calls.append(chunk.shape[1])
        assert chunk.dtype == x.dtype
        return torch.nn.functional.silu(chunk)

    block = SimpleNamespace(norm2=norm, ff=ff)
    with torch.no_grad():
        expected = _add_residual(ff(_modulate(x, norm, shift, scale, indices)), x.clone(), gate, indices)
        states = x.clone()
        actual = _ffn_residual(block, states, shift, scale, gate, indices)
    assert actual is states
    assert calls == [4101, 2048, 2048, 5]
    assert torch.equal(actual, expected)


def test_project_into_packed_rows_without_replacing_storage():
    torch.manual_seed(88)
    source = torch.randn(1, 4099, 5)
    layer = nn.Linear(5, 7)
    packed = torch.zeros(1, 4102, 7, dtype=torch.bfloat16)
    rows = torch.randperm(packed.shape[1])[:source.shape[1]]
    expected = packed.index_copy(1, rows, layer(source).bfloat16())
    pointer = packed.data_ptr()
    with torch.no_grad():
        _project_into(packed, rows, layer, source)
    assert packed.data_ptr() == pointer
    assert torch.equal(packed, expected)


def test_chunked_pdd_head_keeps_row_order_and_timestep_assignment():
    from fasth3_pipeline.pdd import PDDProjection

    torch.manual_seed(81)
    states = torch.randn(1, 8300, 8)
    rows = torch.randperm(states.shape[1])[:8200]
    indices = torch.arange(states.shape[1]) % 3
    shift = torch.randn(3, 8)
    projection = PDDProjection(8, 5, grid_size=4)

    def norm(part, temb, times):
        return part + temb[times]

    with torch.no_grad(), projection.fuse(0, 4, torch.ones(4)):
        expected = projection(norm(states, shift, indices)).index_select(1, rows)
        actual = _project_output_rows(states, rows, indices, norm, shift, projection)
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(os.getenv('FASTH3_TEST_CUDA') != '1', reason='Coordinated SM120 GPU only')
def test_head_groups_match_resident_attention_with_two_regions():
    from fasth3_kernels.mxfp8 import quantize_mxfp8_cute
    from fasth3_pipeline.attention import H3VSAProcessor, TileLayout
    from fasth3_pipeline.long_video import grouped_attention
    from fasth3_pipeline.mxfp8_linear import MXFP8Linear
    from fasth3_platform import BudgetGuard

    def linear(k, n):
        module = MXFP8Linear(k, n, bias=False, device='cuda', backend='cute-optimized', chunk_tokens=512)
        weight = quantize_mxfp8_cute(torch.randn(n, k, device='cuda', dtype=torch.bfloat16) * .04)
        module.weight.copy_(weight.values)
        module.weight_scale.copy_(weight.scales)
        return module

    with BudgetGuard(), torch.no_grad():
        torch.manual_seed(4277)
        layout = TileLayout.build([90, (2, 5, 9), (4, 11, 11)], 'cuda')
        x = torch.randn(1, layout.inverse.numel(), 128, device='cuda', dtype=torch.bfloat16)
        attn = nn.Module()
        attn.heads = 2
        attn.to_q, attn.to_k, attn.to_v = (linear(128, 256) for _ in range(3))
        attn.norm_q, attn.norm_k = (nn.RMSNorm(128, eps=1e-5, device='cuda', dtype=x.dtype) for _ in range(2))
        attn.to_gate_compress = nn.Linear(128, 256, bias=False, device='cuda', dtype=x.dtype)
        attn.to_out = nn.Sequential(linear(256, 128), nn.Identity())
        angles = torch.randn(x.shape[1], 96, device='cuda')
        rotary = angles.cos(), angles.sin()
        processor = H3VSAProcessor(layout)
        expected = grouped_attention(processor, attn, x, rotary, heads_per_group=2)
        actual = grouped_attention(processor, attn, x, rotary, heads_per_group=1)
        assert (actual.float() - expected.float()).norm() / expected.float().norm() < .002
