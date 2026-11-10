"""Small real correctness tests. GPU is opt-in: FASTH3_TEST_CUDA=1.

Never initializes CUDA by default; coordinate the opt-in window with P0. All
GPU allocations follow the shared 10 GiB/256 MiB guard; no model is loaded.
"""
import os
import math
import pytest
import torch
from fasth3_kernels.attention import sparse_attention
from fasth3_kernels.nvfp4 import dequantize_nvfp4, linear_nvfp4
from fasth3_kernels.oracle import nvfp4_oracle, sparse_attention_oracle


@pytest.fixture(scope="module")
def gpu():
    if os.environ.get("FASTH3_TEST_CUDA") != "1":
        pytest.skip("set FASTH3_TEST_CUDA=1 during a coordinated GPU window")
    from fasth3_platform import configure_gpu_budget, check_gpu_headroom
    budget = configure_gpu_budget(max_gib=10, reserve_mib=256)
    yield "cuda"
    torch.cuda.synchronize()
    check_gpu_headroom(budget)


def _case(device="cpu", dtype=torch.float32, d=32):
    torch.manual_seed(2026)
    q = torch.randn(1, 2, 259, d, device=device, dtype=dtype)
    k, v = [torch.randn(1, 2, 277, d, device=device, dtype=dtype) for _ in range(2)]
    mask = torch.tensor([[[[1, 0, 1], [0, 0, 0], [1, 1, 1]],
                          [[0, 1, 0], [1, 0, 1], [0, 0, 1]]]], device=device, dtype=torch.bool)
    lengths = torch.tensor([83, 128, 7], device=device, dtype=torch.int32)
    return q, k, v, mask, lengths


@pytest.mark.parametrize("causal", [False, True])
def test_attention_oracle_matches_masked_sdpa(causal):
    q, k, v, mask, lengths = _case()
    actual = sparse_attention_oracle(q, k, v, mask, lengths, causal=causal)
    qi, ki = torch.arange(q.shape[2]), torch.arange(k.shape[2])
    allowed = mask.repeat_interleave(128, -2).repeat_interleave(128, -1)[..., :259, :277]
    allowed &= (ki % 128) < lengths[ki // 128]
    if causal:
        allowed &= qi[:, None] >= ki[None, :]
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert not actual[0, 0, 128:256].count_nonzero()


def test_attention_global_softmax_not_per_block_average():
    q = torch.ones(1, 1, 1, 32)
    k = torch.zeros(1, 1, 129, 32)
    k[..., 128, :] = 1
    v = torch.zeros_like(k)
    v[..., 128, :] = 1
    mask = torch.ones(1, 1, 1, 2, dtype=torch.bool)
    actual = sparse_attention_oracle(q, k, v, mask, torch.tensor([128, 1]))
    expected = math.exp(math.sqrt(32)) / (128 + math.exp(math.sqrt(32)))
    torch.testing.assert_close(actual, torch.full_like(actual, expected))


def test_nvfp4_all_nibbles_and_quantization_global_divisor():
    packed = torch.arange(16, dtype=torch.uint8)
    packed = (packed | (packed.roll(1) << 4)).reshape(1, 16)
    scales = torch.tensor([[0.5, 2.]])
    actual = dequantize_nvfp4(packed, scales, 4., backend="torch", dtype=torch.float32)
    expected = nvfp4_oracle(packed, scales, 4., dtype=torch.float32)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    multiplied = dequantize_nvfp4(packed, scales, 0.25, backend="torch", dtype=torch.float32,
                                  global_scale_mode="dequantization")
    torch.testing.assert_close(actual, multiplied, atol=0, rtol=0)


def _swizzle(scales):
    rows, cols = scales.shape
    pr, pc = math.ceil(rows / 128) * 128, math.ceil(cols / 4) * 4
    padded = torch.zeros(pr, pc, device=scales.device)
    padded[:rows, :cols] = scales.float()
    return padded.reshape(pr // 128, 4, 32, pc // 4, 4).permute(0, 3, 2, 1, 4).contiguous().reshape(pr, pc)


@pytest.mark.parametrize("n,k", [(128, 128), (131, 80)])
def test_nvfp4_128x4_uint8_scales_and_padding(n, k):
    torch.manual_seed(5)
    packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8)
    scales = torch.randint(1, 8, (n, k // 16)).float().to(torch.float8_e4m3fn)
    swizzled = _swizzle(scales).to(torch.float8_e4m3fn).view(torch.uint8)
    actual = dequantize_nvfp4(packed, swizzled, 2., scale_layout="128x4", backend="torch")
    expected = nvfp4_oracle(packed, scales, 2.)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_nvfp4_chunked_linear_awq_and_bias():
    torch.manual_seed(2)
    p = torch.randint(0, 256, (131, 32), dtype=torch.uint8)
    s = torch.rand(131, 4).to(torch.float8_e4m3fn)
    x = torch.randn(2, 3, 64, dtype=torch.bfloat16)
    pre = torch.linspace(.5, 1.5, 64)
    bias = torch.randn(131, dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(x * pre.to(x.dtype), nvfp4_oracle(p, s, 8.)) + bias
    actual = linear_nvfp4(x, p, s, 8., bias, pre_quant_scale=pre, row_chunk=17, backend="torch")
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_no_silent_backend_fallback():
    q, k, v, m, l = _case()
    with pytest.raises(ValueError, match="requires matching SM120"):
        sparse_attention(q, k, v, m, tile_lengths=l, query_tile_lengths=l, dense_prefix=0)
    with pytest.raises(ValueError, match="requires CUDA"):
        dequantize_nvfp4(torch.zeros(1, 8, dtype=torch.uint8), torch.ones(1, 1))


@pytest.mark.parametrize('tile_size', [0, 32, 96, 64.0])
def test_attention_rejects_invalid_tile_size(tile_size):
    q, k, v, m, l = _case()
    with pytest.raises(ValueError, match='tile_size 64 or 128'):
        sparse_attention(q, k, v, m, tile_lengths=l, query_tile_lengths=l,
                         dense_prefix=0, tile_size=tile_size)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cute_nvfp4_real_gpu_and_nondefault_stream(gpu, dtype):
    torch.manual_seed(3)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        p = torch.randint(0, 256, (131, 40), device=gpu, dtype=torch.uint8)
        s = torch.rand(131, 5, device=gpu).to(torch.float8_e4m3fn)
        ss = _swizzle(s).to(torch.float8_e4m3fn).view(torch.uint8)
        actual = dequantize_nvfp4(p, ss, 4., scale_layout="128x4", dtype=dtype)
        expected = nvfp4_oracle(p, s, 4., dtype=dtype)
    stream.synchronize()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_split_queries_matches_oracle_on_nondefault_stream(gpu):
    from fasth3_pipeline.attention import TileLayout, select_blocks, select_block_indices

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        layout = TileLayout.build([142, (1, 4, 10), (2, 4, 12)], gpu)
        tiles = layout.lengths.numel()
        q, k, v = [torch.randn(1, 2, tiles * 128, 128, dtype=torch.bfloat16, device=gpu) for _ in range(3)]
        q = q.transpose(-1, -2).contiguous().transpose(-1, -2)
        scores = torch.rand(1, 2, tiles, tiles, device=gpu)
        mask = select_blocks(scores, layout)
        indices = select_block_indices(scores, layout)
        reference = sparse_attention_oracle(q, k, v, mask, layout.lengths)
        reference.masked_fill_((layout.source < 0)[None, None, :, None], 0)
        actual = sparse_attention(q, k, v, indices, tile_lengths=layout.lengths,
                                  query_tile_lengths=layout.lengths, dense_prefix=layout.prefix_tiles)
    stream.synchronize()
    torch.testing.assert_close(actual, reference, atol=.008, rtol=.02)
    assert not actual[..., layout.source < 0, :].count_nonzero()


@pytest.mark.parametrize('prefix', [0, 1])
def test_tile64_attention_matches_dense_mask_with_partial_and_empty_tiles(gpu, prefix):
    torch.manual_seed(19)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        q = torch.randn(1, 2, 193, 128, dtype=torch.bfloat16, device=gpu)
        k, v = [torch.randn(1, 2, 210, 128, dtype=torch.bfloat16, device=gpu) for _ in range(2)]
        indices = torch.tensor([[[[2, 3], [-1, 7], [2, 3], [-1, -1]],
                                 [[1, 3], [1, 2], [1, 2], [2, 3]]]], device=gpu, dtype=torch.int32)
        kl = torch.tensor([37, 0, 64, 55], device=gpu)
        ql = torch.tensor([57, 0, 43, 8], device=gpu)
        qi, ki = torch.arange(193, device=gpu), torch.arange(210, device=gpu)
        qt, kt = qi // 64, ki // 64
        selected = (indices[..., qt, :, None] == kt).any(-2)
        allowed = (selected | (qt[:, None] < prefix) | (kt < prefix))
        allowed &= ((qi % 64) < ql[qt])[:, None] & ((ki % 64) < kl[kt])
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), attn_mask=allowed).to(torch.bfloat16)
        actual = sparse_attention(q, k, v, indices, tile_lengths=kl, query_tile_lengths=ql,
                                  dense_prefix=prefix, tile_size=64)
    stream.synchronize()
    torch.testing.assert_close(actual, expected, atol=.008, rtol=.02)
    assert not actual.masked_select(~allowed.any(-1, keepdim=True)).count_nonzero()


def test_tile64_and_default_tile128_are_exact_for_matching_routes(gpu):
    torch.manual_seed(37)
    q, k, v = [torch.randn(1, 1, 273, 128, dtype=torch.bfloat16, device=gpu) for _ in range(3)]
    lengths128 = torch.tensor([128, 128, 17], device=gpu)
    lengths64 = torch.tensor([64, 64, 64, 64, 17], device=gpu)
    routes128 = torch.tensor([[[[2], [2], [2]]]], dtype=torch.int32, device=gpu)
    routes64 = torch.tensor([[[[4], [4], [4], [4], [4]]]], dtype=torch.int32, device=gpu)
    default = sparse_attention(q, k, v, routes128, tile_lengths=lengths128,
                               query_tile_lengths=lengths128, dense_prefix=1)
    actual = sparse_attention(q, k, v, routes64, tile_lengths=lengths64,
                              query_tile_lengths=lengths64, dense_prefix=2, tile_size=64)
    torch.testing.assert_close(actual, default, atol=0, rtol=0)


def test_cute_nvfp4_chunked_linear_real_gpu(gpu):
    torch.manual_seed(2)
    p = torch.randint(0, 256, (131, 32), device=gpu, dtype=torch.uint8)
    s = torch.rand(131, 4, device=gpu).to(torch.float8_e4m3fn)
    x = torch.randn(2, 3, 64, device=gpu, dtype=torch.bfloat16)
    pre = torch.linspace(.5, 1.5, 64, device=gpu)
    bias = torch.randn(131, device=gpu, dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(x * pre.to(x.dtype), nvfp4_oracle(p, s, 8.)) + bias
    actual = linear_nvfp4(x, p, s, 8., bias, pre_quant_scale=pre, row_chunk=17)
    torch.testing.assert_close(actual, expected, atol=.03125, rtol=.008)


def test_cute_nvfp4_packed_slabs_preserve_chunked_linear_bits(gpu):
    torch.manual_seed(7)
    # The 385x80 matrix has a partial scale tile in both dimensions and a
    # second slab whose row offset must be rebased before unpacking its scales.
    packed = torch.randint(0, 256, (385, 40), dtype=torch.uint8, device=gpu)
    scale = torch.rand(385, 5, device=gpu).to(torch.float8_e4m3fn)
    scale = _swizzle(scale).to(torch.float8_e4m3fn).view(torch.uint8)
    x = torch.randn(2, 3, 80, device=gpu, dtype=torch.bfloat16)
    pre = torch.linspace(.5, 1.5, 80, device=gpu)
    bias = torch.randn(385, device=gpu, dtype=torch.bfloat16)
    options = dict(pre_quant_scale=pre, scale_layout="128x4")
    expected = linear_nvfp4(x, packed, scale, 8., bias, **options)
    actual = linear_nvfp4(x, packed, scale, 8., bias, decode_slab_bytes=48 << 10, **options)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
