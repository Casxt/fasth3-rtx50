"""Opt-in real checkpoint probes, never a model load.

FASTH3_NVFP4_CHECKPOINT enables a <1 MiB CPU prefix decode. Additionally setting
FASTH3_TEST_CUDA=1 enables a tiny M=2 full-k_proj linear (~3 MiB packed input),
with bounded 128-row BF16 weight chunks and the shared 10 GiB GPU guard.
"""
import os
from pathlib import Path
import json
import pytest
import torch
from fasth3_kernels.nvfp4 import dequantize_nvfp4


def test_real_nvfp4_checkpoint_prefix_decode():
    folder = os.environ.get("FASTH3_NVFP4_CHECKPOINT")
    if not folder:
        pytest.skip("set FASTH3_NVFP4_CHECKPOINT for opt-in checkpoint-prefix test")
    from safetensors import safe_open
    folder = Path(folder)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    quant = config["quantization_config"]
    assert quant["fmt"] == "e2m1" and quant["group_size"] == 16 and quant["scale_layout"] == "128x4"
    index = json.loads((folder / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    prefix = "model.language_model.layers.0.self_attn.k_proj"
    names = [prefix + "." + suffix for suffix in ("weight_packed", "weight_scale", "weight_global_scale")]
    tensors = []
    for name in names:
        with safe_open(folder / index[name], framework="pt", device="cpu") as handle:
            if name.endswith("weight_packed"):
                tensors.append(handle.get_slice(name)[:128, :32])
            else:
                tensors.append(handle.get_tensor(name))
    packed, all_scales, gs = tensors
    # First col_tile and first row_tile are exactly the first 512 swizzled bytes.
    scales = all_scales.flatten()[:512].reshape(128, 4)
    actual = dequantize_nvfp4(packed, scales, gs, scale_layout="128x4", backend="torch", dtype=torch.float32)
    # Scalar byte/nibble decode and independently addressed scale geometry.
    raw = scales.view(torch.float8_e4m3fn).flatten().float()
    levels = [0., .5, 1., 1.5, 2., 3., 4., 6.]
    expected = torch.empty(128, 64)
    multiplier = 1.0 / float(gs.item())
    for row in range(128):
        for col in range(64):
            byte = int(packed[row, col // 2])
            code = (byte >> (4 * (col % 2))) & 15
            sf_index = ((row % 32) * 4 + row // 32) * 4 + col // 16
            expected[row, col] = levels[code & 7] * (-1 if code & 8 else 1) * float(raw[sf_index]) * multiplier
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=2e-7)
    assert actual.isfinite().all() and actual.count_nonzero()


def test_real_nvfp4_checkpoint_cute_linear_small_gpu():
    folder = os.environ.get("FASTH3_NVFP4_CHECKPOINT")
    if not folder or os.environ.get("FASTH3_TEST_CUDA") != "1":
        pytest.skip("requires checkpoint path AND coordinated FASTH3_TEST_CUDA=1")
    from fasth3_platform import configure_gpu_budget, check_gpu_headroom
    from fasth3_kernels.nvfp4 import linear_nvfp4
    from fasth3_kernels.oracle import nvfp4_oracle
    from safetensors import safe_open
    budget = configure_gpu_budget(max_gib=10.0, reserve_mib=256)
    check_gpu_headroom(budget)
    folder = Path(folder)
    index = json.loads((folder / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    prefix = "model.language_model.layers.0.self_attn.k_proj"

    def read(suffix):
        name = prefix + "." + suffix
        with safe_open(folder / index[name], framework="pt", device="cpu") as handle:
            return handle.get_tensor(name).to("cuda")

    packed, raw_scales, gs = [read(s) for s in ("weight_packed", "weight_scale", "weight_global_scale")]
    assert packed.shape == (1024, 2560) and raw_scales.dtype == torch.uint8
    pre = read("pre_quant_scale") if prefix + ".pre_quant_scale" in index else None
    bias = read("bias") if prefix + ".bias" in index else None
    torch.manual_seed(7)
    x = torch.randn(2, 5120, device="cuda", dtype=torch.bfloat16)
    actual = linear_nvfp4(x, packed, raw_scales, gs, bias, pre_quant_scale=pre,
                         scale_layout="128x4", backend="cute", row_chunk=128)
    # Independent direct scale addressing, not the production reshape/permute.
    # Each 128-row/4-group tile contains 512 swizzled scale bytes.
    rows = torch.arange(1024, device="cuda")[:, None]
    groups = torch.arange(320, device="cuda")[None, :]
    addresses = ((rows // 128 * 80 + groups // 4) * 512
                 + (rows % 32) * 16 + ((rows // 32) % 4) * 4 + groups % 4)
    scales = raw_scales.flatten().view(torch.float8_e4m3fn).float()[addresses]
    smooth = x if pre is None else x * pre.to(x.dtype)
    expected = torch.empty_like(actual)
    for start in range(0, 1024, 128):
        weight = nvfp4_oracle(packed[start:start + 128], scales[start:start + 128], gs)
        expected[:, start:start + 128] = torch.nn.functional.linear(smooth, weight)
        del weight
    if bias is not None:
        expected += bias.to(expected.dtype)
    torch.cuda.synchronize()
    assert actual.isfinite().all() and actual.count_nonzero()
    torch.testing.assert_close(actual, expected, atol=.002, rtol=.008)
