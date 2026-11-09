"""Targeted SM120 regression checks for the optional DiT fusions."""
import os

import pytest
import torch
from torch import nn

from fasth3_platform import BudgetGuard


pytestmark = pytest.mark.skipif(os.environ.get('FASTH3_TEST_CUDA') != '1',
                                reason='Coordinated SM120 GPU only')


def test_swiglu_pack_preserves_bf16_rounding_and_mxfp8_codes():
    from fasth3_kernels._cute_mxfp8 import launch_quantizer

    with BudgetGuard(), torch.no_grad():
        torch.manual_seed(4277)
        projected = torch.randn(512, 28672, device='cuda', dtype=torch.bfloat16)
        value, gate = projected.chunk(2, dim=-1)
        expected, scales = launch_quantizer(value * nn.functional.silu(gate))
        actual, fused_scales = launch_quantizer(projected, swiglu=True)
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        assert torch.equal(fused_scales, scales)


@pytest.mark.parametrize('rms', [False, True])
def test_adaln_pack_preserves_affine_rounding_and_strided_tables(rms):
    from fasth3_kernels._cute_adaln import pack_adaln
    from fasth3_kernels.mxfp8 import quantize_mxfp8_cute

    with BudgetGuard(), torch.no_grad():
        torch.manual_seed(426)
        opts = {'device': 'cuda', 'dtype': torch.bfloat16}
        width = 5376
        shift, scale = torch.randn(9, 6 * width, **opts).chunk(6, dim=-1)[3:5]
        for count in (136, 512):
            x = torch.randn(count, width, **opts)
            rows = torch.arange(count, device='cuda') % len(scale)
            weight = torch.randn(width, **opts) if rms else None
            normalized = torch.nn.functional.rms_norm(x, (width,), weight, 1e-5) if rms else x
            expected = quantize_mxfp8_cute(normalized * (1.0 + scale[rows]) + shift[rows], check_finite=False)
            actual = pack_adaln(x, shift, scale, rows, rms_weight=weight, eps=1e-5 if rms else None)
            assert torch.equal(actual.values.view(torch.uint8), expected.values.view(torch.uint8))
            assert torch.equal(actual.scales, expected.scales)


def test_mxfp8_ffn_pipeline_reuses_slots_and_preserves_ragged_output():
    from fasth3_kernels._cute_mxfp8 import launch, launch_quantizer
    from fasth3_kernels._cute_mxfp8_ffn import Schedule, launch_down, launch_up
    from fasth3_kernels.mxfp8 import MXFP8Tensor, quantize_mxfp8_cute

    with BudgetGuard(), torch.no_grad():
        torch.manual_seed(4277)
        sms = torch.cuda.get_device_properties('cuda').multi_processor_count
        m, n, k = 129, 128 * (sms // 3 + 2), 384
        opts = {'device': 'cuda', 'dtype': torch.bfloat16}
        a = quantize_mxfp8_cute(torch.randn(m, k, **opts), check_finite=False)
        b = quantize_mxfp8_cute(torch.randn(n, k, **opts), check_finite=False)
        scales = b.scales.view(n // 32, 32, k // 128, 4).transpose(1, 2).contiguous()
        packed = MXFP8Tensor(b.values, scales, k)
        bias = torch.randn(n, **opts)
        projected = torch.empty(m, n, **opts)
        launch(a, packed, bias=bias, out=projected, packed_b=True, optimized=(64, 128, 2, 2, 2),
               producer_warps=1, tma=True)
        codes, scales = launch_quantizer(projected, swiglu=True)
        up = launch_up(a, packed, bias, Schedule(rows=64, ctas_per_sm=1))
        assert torch.equal(up.values.view(torch.uint8), codes.view(torch.uint8))
        assert torch.equal(up.scales, scales)
        for slots in (1, 2):
            schedule = Schedule(rows=64, output_slots=slots, ctas_per_sm=1)
            down = launch_down(a, packed, bias, torch.empty_like(projected), schedule)
            assert torch.equal(down, projected)


@pytest.mark.parametrize('tile_size', (64, 128))
def test_attention_layout_and_gate_on_partial_tiles(tile_size):
    from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb
    from fasth3_kernels._cute_h3_epilogues import attention_layout, compress_gate
    from fasth3_pipeline.attention import TileLayout

    with BudgetGuard(), torch.no_grad():
        torch.manual_seed(4277)
        layout = TileLayout.build([130, (2, 5, 9), (3, 8, 12)], 'cuda', tile_size=tile_size)
        tokens = layout.inverse.numel()
        x = torch.randn(1, tokens, 4, 128, device='cuda', dtype=torch.bfloat16)
        norm = nn.RMSNorm(128, eps=1e-5, device='cuda', dtype=x.dtype)
        norm.weight.copy_(torch.randn(128, device='cuda') * .25 + 1)
        angles = torch.randn(tokens, 96, device='cuda')
        rotary = (angles.cos(), angles.sin())
        expected = layout.tile(_apply_rotary_emb(norm(x), *rotary)).transpose(1, 2)
        actual = attention_layout(x, layout.source, norm, rotary)
        assert (actual.float() - expected.float()).norm() / expected.float().norm() < 2e-4
        assert torch.equal(attention_layout(x, layout.source), layout.tile(x).transpose(1, 2))
        gate = torch.randn_like(x)
        coarse = torch.randn(1, 4, len(layout.lengths), 128, device='cuda')
        expanded = coarse.transpose(1, 2).to(x.dtype).repeat_interleave(tile_size, dim=1)
        expected = torch.addcmul(actual.transpose(1, 2), expanded, layout.tile(gate))[:, layout.inverse]
        assert torch.equal(compress_gate(actual, coarse, gate, layout.inverse, tile_size=tile_size), expected)


def _linear(in_features, out_features):
    from fasth3_kernels.mxfp8 import quantize_mxfp8_cute
    from fasth3_pipeline.mxfp8_linear import MXFP8Linear

    layer = MXFP8Linear(in_features, out_features, bias=False, device='cuda',
                        backend='cute-optimized', chunk_tokens=512)
    weight = quantize_mxfp8_cute(torch.randn(out_features, in_features, device='cuda', dtype=torch.bfloat16))
    layer.weight.copy_(weight.values)
    layer.weight_scale.copy_(weight.scales)
    return layer


def test_qkv_share_one_pack_per_chunk_and_preserve_outputs(monkeypatch):
    from fasth3_kernels import mxfp8
    from fasth3_pipeline.long_video import _head_projections

    with BudgetGuard(), torch.no_grad():
        torch.manual_seed(4277)
        attn = nn.Module()
        attn.heads = 2
        for name in ('to_q', 'to_k', 'to_v'):
            setattr(attn, name, _linear(128, 256))
        x = torch.randn(1, 648, 128, device='cuda', dtype=torch.bfloat16)
        expected = tuple(layer(x).unflatten(-1, (attn.heads, -1)) for layer in (attn.to_q, attn.to_k, attn.to_v))
        pack = mxfp8.quantize_mxfp8_cute
        chunks = []

        def count_chunks(activation, **kwargs):
            chunks.append(len(activation))
            return pack(activation, **kwargs)

        monkeypatch.setattr(mxfp8, 'quantize_mxfp8_cute', count_chunks)
        actual = _head_projections(attn, x[0], (0, attn.heads))
        assert chunks == [512, 136]
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))


def test_streamed_ffn_preserves_tail_and_projection_output():
    from types import SimpleNamespace
    from fasth3_pipeline.transformer import _streamed_fused_ffn

    with BudgetGuard(), torch.no_grad():
        torch.manual_seed(4277)
        up, down = _linear(128, 256), _linear(128, 128)
        x = torch.randn(1, 648, 128, device='cuda', dtype=torch.bfloat16)
        value, gate = up(x).chunk(2, dim=-1)
        expected = down(value * nn.functional.silu(gate))
        ff = SimpleNamespace(net=(SimpleNamespace(proj=up), None, down))
        actual = _streamed_fused_ffn(ff, x)
        assert torch.equal(actual, expected)
