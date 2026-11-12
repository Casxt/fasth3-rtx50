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


def test_resident_scratch_reserves_output_but_allocates_it_lazily():
    from types import SimpleNamespace
    from fasth3_pipeline.long_video import LongVideoScratch

    scratch = LongVideoScratch()
    shape, width = (1, 3, 4), 128
    required = 48 + 3 * 132 + 17
    budget = lambda amount: SimpleNamespace(available_bytes=lambda: amount)
    assert scratch.prepare_cuda(shape, 'cpu', budget(required - 1), output_width=width, workspace_bytes=17) == 0
    assert scratch.packed is None
    assert scratch.prepare_cuda(shape, 'cpu', budget(required), output_width=width, workspace_bytes=17) == 48
    assert scratch.values is scratch.scales is None
    values, scales = scratch.acquire_output(scratch.packed[0], width)
    assert values.shape == (3, 128) and values.dtype == torch.float8_e4m3fn
    assert scales.shape == (3, 4) and scales.dtype == torch.uint8
    assert scratch.acquire_output(scratch.packed[0], width)[0] is values
    assert scratch.prepare_cuda(shape, 'cpu', budget(0)) == 48
    with pytest.raises(ValueError, match='fixed request geometry'):
        scratch.acquire_output(torch.empty(2, 4), width)
    scratch.close()
    assert scratch.values is scratch.scales is scratch.packed is scratch.modulated is None


def test_resident_output_partial_allocation_is_not_retained(monkeypatch):
    from fasth3_pipeline.long_video import LongVideoScratch

    scratch = LongVideoScratch()
    flat = torch.empty(3, 4, dtype=torch.bfloat16)
    allocate = torch.empty

    def fail_scales(*args, **kwargs):
        if kwargs.get('dtype') == torch.uint8:
            raise torch.OutOfMemoryError('scale allocation refused')
        return allocate(*args, **kwargs)

    monkeypatch.setattr(torch, 'empty', fail_scales)
    with pytest.raises(torch.OutOfMemoryError):
        scratch.acquire_output(flat, 128)
    assert scratch.values is scratch.scales is None


def test_packed_projection_inplace_and_scratch_reuse(monkeypatch):
    from types import SimpleNamespace
    from fasth3_kernels import mxfp8
    from fasth3_pipeline import long_video
    from fasth3_pipeline.attention import H3VSAProcessor

    monkeypatch.setattr(mxfp8, 'quantize_mxfp8_cute', lambda x, **kwargs: SimpleNamespace(
        values=x.to(torch.float8_e4m3fn), scales=torch.full((len(x), x.shape[1] // 32), 127, dtype=torch.uint8)))
    monkeypatch.setattr(long_video, '_attend_heads', lambda p, a, x, r, f, l, c:
                        (x[:, :1] + f + 1).expand(-1, (l - f) * 128))

    class Projection:
        in_features = 256
        out_features = 4
        chunk_tokens = 2
        debug_checks = False

        def __call__(self, x, *, packed, output):
            output.copy_(packed.values[:, :4].float())

    attn = SimpleNamespace(heads=2, to_out=(Projection(), nn.Identity()))
    x = torch.arange(16, dtype=torch.bfloat16).reshape(1, 4, 4)
    scratch = long_video.LongVideoScratch()
    processor = H3VSAProcessor(None, packed_output=True, host_scratch=scratch, reuse_input=True)
    residual = x.clone()
    output = long_video.grouped_attention(processor, attn, x.clone(), None, heads_per_group=1)
    values = scratch.values
    modulated = x.clone()
    actual = long_video.grouped_attention(processor, attn, modulated, None, heads_per_group=1)
    assert actual.data_ptr() == modulated.data_ptr()
    assert scratch.values is values and scratch.host is None
    torch.testing.assert_close(actual, output, rtol=0, atol=0)
    assert torch.equal(residual, x)


def test_resident_allocation_refusal_uses_host_for_rest_of_request(monkeypatch):
    from types import SimpleNamespace
    from fasth3_pipeline import long_video
    from fasth3_pipeline.attention import H3VSAProcessor

    scratch = long_video.LongVideoScratch()
    attempts = []

    def refuse(*args):
        attempts.append(True)
        raise torch.OutOfMemoryError('test refusal')

    monkeypatch.setattr(scratch, 'acquire_output', refuse)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    monkeypatch.setattr(long_video, '_PinnedHeadOutput', lambda *args: None)
    monkeypatch.setattr(long_video, '_attend_heads', lambda p, a, x, r, f, l, c:
                        (x[:, :1] + 1).expand(-1, (l-f) * 128))

    class Projection:
        out_features = 4

        def __call__(self, x, *, output):
            output.copy_(x[:, :4])

    scratch.policy = {}
    processor = H3VSAProcessor(None, packed_output=True, host_scratch=scratch, reuse_input=True)
    attn = SimpleNamespace(heads=1, to_out=(Projection(), nn.Identity()))
    for _ in range(2):
        x = torch.ones((1, 3, 4), dtype=torch.bfloat16)
        actual = long_video.grouped_attention(processor, attn, x, None)
        assert actual.data_ptr() == x.data_ptr()
        assert torch.equal(actual, torch.full_like(x, 2))
    assert attempts == [True]
    assert scratch.policy['attention_output_runtime'] == 'host'


def test_block_rejects_modulation_storage_aliasing_residual():
    from types import SimpleNamespace
    from fasth3_pipeline import long_video

    states = torch.ones(1, 2, 4, dtype=torch.bfloat16)
    block = SimpleNamespace(adaln_proj=lambda _: (None,) * 6,
                            _long_scratch=SimpleNamespace(modulated=states.view_as(states)))
    with pytest.raises(ValueError, match='separate from the residual'):
        long_video._block_forward(block, states, None, None, None)


@pytest.mark.parametrize('chunk_tokens', [512, 2048])
def test_fused_head_projection_obeys_projection_chunk_size(monkeypatch, chunk_tokens):
    from types import SimpleNamespace
    from fasth3_kernels import mxfp8
    from fasth3_pipeline import long_video, mxfp8_linear

    chunks = []
    monkeypatch.setattr(mxfp8, 'quantize_mxfp8_cute', lambda x, **kwargs: SimpleNamespace(values=x))

    def gemm(activation, weight, *, out, **kwargs):
        chunks.append(len(activation.values))
        out.fill_(1)

    monkeypatch.setattr(mxfp8_linear, '_native_validated_gemm', gemm)
    layers = [SimpleNamespace(chunk_tokens=chunk_tokens, packed_scales=True, debug_checks=False, bias=None,
                              tma_store=False, weight=torch.empty(512, 5376, dtype=torch.float8_e4m3fn),
                              weight_scale=torch.empty(16, 42, 32, 4, dtype=torch.uint8)) for _ in range(3)]
    attn = SimpleNamespace(to_q=layers[0], to_k=layers[1], to_v=layers[2])
    flat = torch.empty(2051, 5376, dtype=torch.bfloat16)
    actual = long_video._fused_head_projections(attn, flat, 0, 4, None)
    assert chunks == ([512] * 4 + [3] if chunk_tokens == 512 else [2048, 3])
    assert actual.shape == (1, 2051, 12, 128)
    assert torch.equal(actual, torch.ones_like(actual))


@pytest.mark.parametrize('available_gib,resident', [(8.56, True), (7.0, False)])
def test_long_policy_fits_accepted_geometry_and_falls_back(monkeypatch, available_gib, resident):
    from types import SimpleNamespace
    from fasth3_pipeline import long_video, transformer

    # The accepted 15s+15s packed sequence, without allocating its full matrices.
    layout = SimpleNamespace(inverse=torch.empty(226195, device='meta'),
                             source=torch.empty(260992, device='meta'),
                             lengths=torch.empty(2039, device='meta'))
    attn = SimpleNamespace(heads=56, to_q=SimpleNamespace(chunk_tokens=512),
                           to_k=SimpleNamespace(chunk_tokens=512), to_v=SimpleNamespace(chunk_tokens=512))
    attn.set_processor = lambda processor: setattr(attn, 'processor', processor)
    up = SimpleNamespace(debug_checks=False, in_features=5376, out_features=28672)
    down = SimpleNamespace(debug_checks=False, in_features=14336, out_features=5376)
    block = SimpleNamespace(attn=attn, ff=SimpleNamespace(net=[SimpleNamespace(proj=up), None, down]))
    model = SimpleNamespace(transformer_blocks=[block], context_embedder=SimpleNamespace(out_features=5376),
                            norm_out=SimpleNamespace(), modules=lambda: [], _dit_weight_arena_bytes=474486272)
    scratch = long_video.LongVideoScratch()
    budget = SimpleNamespace(available_bytes=lambda: int(available_gib * (1 << 30)))
    policy = transformer.install_vsa(model, layout, budget, long_scratch=scratch)
    assert policy['attention_output'] == ('mxfp8' if resident else 'host')
    assert policy['qkv_cache_bytes'] == 0
    assert policy['qkv_chunk_tokens'] == 2048
    assert policy['activation_scratch_reuse'] and policy['attention_output_inplace']
    assert (scratch.packed is not None) == resident
    assert scratch.values is scratch.scales is None  # Kept out of the text refiner phase.
    assert model._long_scratch is block._long_scratch is scratch
    assert attn.processor.reuse_input
    assert attn.to_q.chunk_tokens == attn.to_k.chunk_tokens == attn.to_v.chunk_tokens == 2048


def test_short_packed_policy_preserves_cache_and_projection_schedule():
    from types import SimpleNamespace
    from fasth3_pipeline import long_video, transformer

    layout = SimpleNamespace(inverse=torch.empty(118693, device='meta'))
    attn = SimpleNamespace(heads=56, to_q=SimpleNamespace(chunk_tokens=512),
                           to_k=SimpleNamespace(chunk_tokens=512), to_v=SimpleNamespace(chunk_tokens=512))
    attn.set_processor = lambda processor: setattr(attn, 'processor', processor)
    up = SimpleNamespace(debug_checks=False, in_features=5376, out_features=28672)
    down = SimpleNamespace(debug_checks=False, in_features=14336, out_features=5376)
    block = SimpleNamespace(attn=attn, ff=SimpleNamespace(net=[SimpleNamespace(proj=up), None, down]))
    model = SimpleNamespace(transformer_blocks=[block], norm_out=SimpleNamespace(), modules=lambda: [])
    scratch = long_video.LongVideoScratch()
    policy = transformer.install_vsa(model, layout, SimpleNamespace(available_bytes=lambda: 9 << 30),
                                     long_scratch=scratch)
    assert policy['attention_output'] == 'mxfp8'
    assert policy['qkv_cache_bytes'] == 384 << 20
    assert policy['qkv_chunk_tokens'] == 512
    assert not policy['activation_scratch_reuse']
    assert not attn.processor.reuse_input
    assert attn.processor.host_scratch is None
    assert model._long_scratch is None


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
        from fasth3_pipeline.long_video import LongVideoScratch
        scratch = LongVideoScratch()
        processor = H3VSAProcessor(layout, packed_output=True, host_scratch=scratch, reuse_input=True)
        modulated = x.clone()
        resident = grouped_attention(processor, attn, modulated, rotary, heads_per_group=1)
        assert resident.data_ptr() == modulated.data_ptr()
        assert (resident.float() - expected.float()).norm() / expected.float().norm() < .002
        scratch.close()


@pytest.mark.skipif(os.getenv('FASTH3_TEST_CUDA') != '1', reason='Coordinated SM120 GPU only')
def test_fused_qkv_2048_matches_512_token_chunks():
    from types import SimpleNamespace
    from fasth3_kernels.mxfp8 import quantize_mxfp8_cute
    from fasth3_pipeline.long_video import _fused_head_projections
    from fasth3_pipeline.mxfp8_linear import MXFP8Linear
    from fasth3_platform import BudgetGuard

    with BudgetGuard(), torch.no_grad():
        torch.manual_seed(4277)
        layers = []
        for _ in range(3):
            layer = MXFP8Linear(5376, 512, bias=False, device='cuda', backend='cute-optimized',
                                chunk_tokens=512, packed_scales=True)
            weight = quantize_mxfp8_cute(torch.randn(512, 5376, device='cuda', dtype=torch.bfloat16) * .04)
            layer.weight.copy_(weight.values)
            layer.weight_scale.copy_(weight.scales.view(16, 32, 42, 4).permute(0, 2, 1, 3).contiguous())
            layers.append(layer)
        attn = SimpleNamespace(to_q=layers[0], to_k=layers[1], to_v=layers[2])
        x = torch.randn(2051, 5376, device='cuda', dtype=torch.bfloat16)
        expected = _fused_head_projections(attn, x, 0, 4, None)
        for layer in layers:
            layer.chunk_tokens = 2048
        actual = _fused_head_projections(attn, x, 0, 4, None)
        assert torch.equal(actual, expected)
