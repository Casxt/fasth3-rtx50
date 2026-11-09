"""Independent, deliberately straightforward PyTorch test oracles.

These do not import CuTe or call production implementations/third-party kernels.
Dense attention expansion is test-only, not a large-model execution backend.
"""
import math
import torch


def sparse_attention_oracle(q, k, v, mask, lengths=None, *, causal=False, scale=None):
    sq, sk, d = q.shape[-2], k.shape[-2], q.shape[-1]
    if q.shape[0] * q.shape[1] * sq * sk > 16_777_216:
        raise ValueError("dense oracle is restricted to small tests")
    qi = torch.arange(sq, device=q.device)
    ki = torch.arange(sk, device=q.device)
    allowed = mask[..., qi[:, None] // 128, ki[None, :] // 128]
    if lengths is not None:
        allowed = allowed & ((ki % 128) < lengths[ki // 128])
    if causal:
        allowed = allowed & (qi[:, None] >= ki[None, :])
    scores = q.float() @ k.float().transpose(-2, -1)
    scores *= d ** -0.5 if scale is None else scale
    scores.masked_fill_(~allowed, float("-inf"))
    # Define empty rows directly, without depending on SDPA's NaN handling.
    nonempty = allowed.any(-1, keepdim=True)
    scores = torch.where(nonempty, scores, 0.0)
    probs = scores.softmax(-1).masked_fill(~allowed, 0.0)
    return (probs @ v.float()).to(q.dtype)


def nvfp4_oracle(packed, scales, global_scale=1.0, *, scale_layout="row-major", dtype=torch.bfloat16,
                  global_scale_mode="quantization"):
    """Decode byte-by-byte using explicit E2M1 table; low nibble first."""
    rows, packed_k = packed.shape
    k = packed_k * 2
    codes = torch.stack((packed & 15, packed >> 4), -1).reshape(rows, k).long()
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                          -0., -.5, -1., -1.5, -2., -3., -4., -6.], device=packed.device)
    if scales.dtype == torch.uint8:
        scales = scales.contiguous().view(torch.float8_e4m3fn)
    if scale_layout == "row-major":
        decoded_scales = scales.float().reshape(rows, k // 16)
    elif scale_layout == "128x4":
        padded_rows = math.ceil(rows / 128) * 128
        padded_cols = math.ceil((k // 16) / 4) * 4
        decoded_scales = scales.float().reshape(padded_rows // 128, padded_cols // 4, 32, 4, 4)
        decoded_scales = decoded_scales.permute(0, 3, 2, 1, 4).reshape(padded_rows, padded_cols)
        decoded_scales = decoded_scales[:rows, :k // 16]
    else:
        raise ValueError("unknown scale layout")
    values = table[codes] * decoded_scales.repeat_interleave(16, dim=-1)
    scale = torch.as_tensor(global_scale, device=packed.device).float()
    if global_scale_mode == "quantization":
        values = values / scale
    elif global_scale_mode == "dequantization":
        values = values * scale
    else:
        raise ValueError("unknown global scale mode")
    return values.to(dtype)
