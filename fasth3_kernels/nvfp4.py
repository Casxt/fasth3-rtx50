"""NVFP4 E2M1 -> BF16 fallback and bounded row-chunked linear.

Contract: packed[N,K/2] uint8, low nibble is the earlier K coordinate;
weight = E2M1 * block_scale[E4M3, K/16] / global_scale[FP32]
by default, matching the checkpoint's quantizer multiplier. Explicit
``global_scale_mode='dequantization'`` uses multiplication instead.
Checkpoint ``128x4`` scale layout must be passed explicitly. This is NOT an
INT4/AWQ nibble layout and NOT a native W4A4 GEMM. Checkpoint AWQ smoothing
uses linear's explicit pre_quant_scale multiplier. Generic input_scale is an
alternative divisor; pass only one convention.
"""
import math
import torch


def _validate(packed, scales, global_scale, scale_layout):
    if packed.ndim != 2 or packed.dtype != torch.uint8 or packed.shape[1] % 8:
        raise ValueError("packed must be uint8 [N,K/2] with K divisible by 16")
    n, pk = packed.shape
    if n == 0 or pk == 0:
        raise ValueError("weight must be nonempty")
    if scales.device != packed.device or not (scales.is_floating_point() or scales.dtype == torch.uint8):
        raise ValueError("block scales must be floating point or raw E4M3 uint8 on packed's device")
    cols = pk // 8
    if scale_layout == "row-major":
        if scales.shape != (n, cols):
            raise ValueError(f"row-major scales must have shape {(n, cols)}")
    elif scale_layout == "128x4":
        count = math.ceil(n / 128) * 128 * math.ceil(cols / 4) * 4
        if scales.numel() != count:
            raise ValueError(f"128x4 scales must have {count} elements including padding")
    else:
        raise ValueError("scale_layout must be 'row-major' or '128x4'")
    global_scale = torch.as_tensor(global_scale, device=packed.device, dtype=torch.float32)
    if global_scale.numel() != 1:
        raise ValueError("global_scale must be scalar; see global_scale_mode for its convention")
    return global_scale.reshape(1).contiguous()


def _row_scales(scales, rows, cols, layout):
    if scales.dtype == torch.uint8:
        scales = scales.contiguous().view(torch.float8_e4m3fn)
    if layout == "row-major":
        return scales.float().contiguous()
    pr, pc = math.ceil(rows / 128) * 128, math.ceil(cols / 4) * 4
    return (scales.float().reshape(pr // 128, pc // 4, 32, 4, 4)
            .permute(0, 3, 2, 1, 4).reshape(pr, pc)[:rows, :cols].contiguous())


def _scale_multiplier(global_scale, mode):
    if mode == "quantization":
        return global_scale.reciprocal()
    if mode == "dequantization":
        return global_scale
    raise ValueError("global_scale_mode must be 'quantization' or 'dequantization'")


def dequantize_nvfp4(packed, block_scales, global_scale=1.0, *,
                     scale_layout="row-major", backend="cute", dtype=torch.bfloat16,
                     global_scale_mode="quantization"):
    """Explicit CuTe or PyTorch fallback. Full decoded weight allocated by caller.

    ``backend='torch'`` is a usable vectorized BF16 fallback, independently
    tested against a table-lookup oracle. No auto substitution on CuTe errors.
    """
    gs = _scale_multiplier(_validate(packed, block_scales, global_scale, scale_layout), global_scale_mode)
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("output dtype must be BF16, FP16 or FP32")
    scales = _row_scales(block_scales, packed.shape[0], packed.shape[1] // 8, scale_layout)
    return _decode_row_major(packed, scales, gs, backend, dtype)


def _decode_row_major(packed, scales, gs, backend, dtype):
    if backend == "cute":
        if packed.device.type != "cuda":
            raise ValueError("CuTe backend requires CUDA; use backend='torch' on CPU")
        from ._cute_nvfp4 import launch
        return launch(packed.contiguous(), scales, gs, dtype)
    if backend != "torch":
        raise ValueError("backend must be 'cute' or 'torch'")
    codes = torch.stack((packed & 15, packed >> 4), -1).reshape(packed.shape[0], -1)
    mag = codes & 7
    # Arithmetic E2M1 decode independent from oracle's lookup table.
    exponent = mag >> 1
    mantissa = mag & 1
    val = torch.where(exponent == 0, mantissa.float() * .5,
                      (1. + mantissa.float() * .5) * torch.pow(2., exponent.float() - 1.))
    val = torch.where((codes & 8) != 0, -val, val)
    return (val * scales.repeat_interleave(16, -1) * gs).to(dtype)


def linear_nvfp4(x, packed, block_scales, global_scale=1.0, bias=None, *,
                 input_scale=None, pre_quant_scale=None, scale_layout="row-major", backend="cute", row_chunk=128,
                 global_scale_mode="quantization", decode_slab_bytes=0):
    """BF16 dequantization and GEMM with bounded temporary weights.

    The packed-scale path batches decoding within a slab without widening the GEMM.
    Input activation precision is preserved through smoothing, then BF16 GEMM.
    Not a quantized-activation/native FP4 performance claim. Bias added once.
    """
    gs = _scale_multiplier(_validate(packed, block_scales, global_scale, scale_layout), global_scale_mode)
    if x.ndim < 1 or x.device != packed.device or x.shape[-1] != packed.shape[-1] * 2:
        raise ValueError("input device/K mismatch")
    if not x.is_floating_point() or not isinstance(row_chunk, int) or row_chunk < 1:
        raise ValueError("input must float and row_chunk must be positive")
    if bias is not None and (bias.shape != (packed.shape[0],) or bias.device != x.device):
        raise ValueError("bias must have shape [N] on input's device")
    if pre_quant_scale is not None:
        if input_scale is not None:
            raise ValueError("pass only pre_quant_scale multiplier OR input_scale divisor")
        if pre_quant_scale.shape != (x.shape[-1],) or pre_quant_scale.device != x.device:
            raise ValueError("pre_quant_scale must be [K] on input's device")
        # Match serialized AWQ loader: smoothing performed in activation dtype.
        x = x * pre_quant_scale.to(x.dtype)
    if input_scale is not None:
        if input_scale.shape != (x.shape[-1],) or input_scale.device != x.device:
            raise ValueError("input_scale must be [K] on input's device")
        x = x.float() / input_scale.float()
    x = x.to(torch.bfloat16)
    out = torch.empty((*x.shape[:-1], packed.shape[0]), device=x.device, dtype=x.dtype)
    if decode_slab_bytes and backend == "cute" and scale_layout == "128x4" and row_chunk == 128:
        if not packed.is_cuda:
            raise ValueError("CuTe backend requires CUDA; use backend='torch' on CPU")
        from ._cute_nvfp4 import launch_packed

        scales = block_scales.contiguous()
        if scales.dtype == torch.uint8:
            scales = scales.view(torch.float8_e4m3fn)
        scales = scales.view(math.ceil(packed.shape[0] / 128), -1)
        slab_rows = max(128, decode_slab_bytes // (x.shape[-1] * 2 * 128) * 128)
        for slab_start in range(0, packed.shape[0], slab_rows):
            slab_end = min(slab_start + slab_rows, packed.shape[0])
            slab_scales = scales[slab_start // 128:math.ceil(slab_end / 128)]
            weight = launch_packed(packed[slab_start:slab_end].contiguous(), slab_scales, gs)
            # Keep the original GEMM dimensions and BF16 output rounding.
            for offset in range(0, len(weight), row_chunk):
                end = min(offset + row_chunk, len(weight))
                out[..., slab_start + offset:slab_start + end] = torch.nn.functional.linear(x, weight[offset:end])
            del weight
    else:
        scales = _row_scales(block_scales, packed.shape[0], packed.shape[1] // 8, scale_layout)
        for start in range(0, packed.shape[0], row_chunk):
            end = min(start + row_chunk, packed.shape[0])
            w = _decode_row_major(packed[start:end], scales[start:end], gs, backend, torch.bfloat16)
            out[..., start:end] = torch.nn.functional.linear(x, w)
            del w  # Permit same-stream allocator reuse before decoding the next chunk.
    if bias is not None:
        out += bias.to(out.dtype)
    return out
