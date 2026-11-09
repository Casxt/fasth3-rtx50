"""Experimental MXFP8 E4M3 / unsigned E8M0, 32 contiguous K elements per scale.

Version 1 scale policy is conservative ceil(log2(amax/448)), clamped to
[-127,127], with zero blocks encoded as scale byte 127 (1.0). This is an
explicit policy, not a claim of bitwise equality with every MX converter.
Scale byte 255 is NaN/reserved and rejected. K padding is zero-filled.
Both CuTe GEMM variants use native block-scaled MMA. The explicit oracle
uses decoded operands and FP32 math.
"""
from dataclasses import dataclass
import math
import torch

BLOCK_SIZE = 32
FORMAT_VERSION = 'mxfp8-e4m3-e8m0-k32-ceil-v1'
ROW_MAJOR_SCALE_LAYOUT = 'row-major [out,ceil(original_K/32)]'
PACKED_SCALE_LAYOUT = 'tile32-k128 [out/32,padded_K/128,32,4]'


@dataclass(frozen=True)
class MXFP8Tensor:
    values: torch.Tensor  # [rows, ceil(K/32)*32], E4M3
    scales: torch.Tensor  # [rows, ceil(K/32)], raw unsigned E8M0 bytes
    logical_k: int

    def validate(self):
        if self.values.ndim != 2 or self.values.dtype != torch.float8_e4m3fn:
            raise ValueError('values must be a rank-2 E4M3 tensor')
        rows, k = self.values.shape
        if rows < 1 or k < 32 or k % 32 or not 0 < self.logical_k <= k or k-self.logical_k >= 32:
            raise ValueError('invalid logical K or zero padding geometry')
        if self.scales.dtype != torch.uint8 or self.scales.shape != (rows, k//32):
            raise ValueError('scales must be raw uint8 [rows,padded_K/32]')
        if self.scales.device != self.values.device:
            raise ValueError('values and scales must share a device')
        if not self.values.is_contiguous() or not self.scales.is_contiguous():
            raise ValueError('values and scales must be contiguous')
        if bool((self.scales == 255).any()):
            raise ValueError('reserved/NaN E8M0 scale byte 255')
        if not bool(self.values.float().isfinite().all()):
            raise ValueError('nonfinite E4M3 payload')
        if self.logical_k < k and bool(self.values.float()[:, self.logical_k:].count_nonzero()):
            raise ValueError('K padding must contain zero values')
        return self


def decode_e8m0(raw):
    """Raw 0..254 encode exactly 2**(raw-127); no zero scale exists."""
    if raw.dtype != torch.uint8 or bool((raw == 255).any()):
        raise ValueError('E8M0 requires raw uint8 bytes 0..254')
    return torch.ldexp(torch.ones_like(raw, dtype=torch.float64), raw.int()-127)


def _validate_quantization_input(x):
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError('expected a nonempty [rows,K] matrix')
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError('input must be FP32, FP16 or BF16; do not requantize old FP8')


def _quantize_mxfp8_resident(x):
    """Device-preserving eager implementation, after finite-input validation.

    No tensor download, host scalar reads, allocation on another device, or
    stream changes. Generated exponents are known valid, so avoid running the
    public scale decoder's reserved-byte validation (and a second CUDA sync).
    """
    rows, k = x.shape
    padded = torch.nn.functional.pad(x.double(), (0, (-k) % 32))
    blocks = padded.reshape(rows, -1, 32)
    amax = blocks.abs().amax(-1)
    # Exact ceil(log2(amax/448)) without a transcendental boundary error:
    # amax = mantissa*2**binary_exp, mantissa in [0.5,1), 448=.875*2**9.
    # Thus exponent=binary_exp-9+(mantissa>.875). Equality MUST stay in the
    # smaller exponent bin. CPU/CUDA log2 can disagree by an ulp exactly at
    # 448*2**k (e.g. amax=1.75), changing both scale and E4M3 code bytes.
    mantissa, binary_exp = torch.frexp(amax)
    exponent = binary_exp - 9 + (mantissa > .875).int()
    exponent = torch.where(amax == 0, torch.zeros_like(exponent), exponent).clamp(-127, 127).int()
    scales = (exponent+127).to(torch.uint8).contiguous()
    scale_values = torch.ldexp(torch.ones_like(amax), exponent)
    normalized = blocks / scale_values[..., None]
    values = normalized.clamp(-448, 448).float().to(torch.float8_e4m3fn).reshape(rows, -1).contiguous()
    return MXFP8Tensor(values, scales, k)


def quantize_mxfp8_cuda(x, *, check_finite=True):
    """Quantize a CUDA-resident matrix; payload/scales stay on that CUDA device.

    Correctness-first eager CUDA, not a fused/production-throughput quantizer.
    All reduction, FP64 scale calculation, E4M3 conversion and padding run on
    PyTorch's current stream. The optional finite check synchronizes that
    stream to reject NaN/Inf. With the check disabled the caller must supply
    finite inputs.
    Temporary FP64 storage is material: budget conservatively 64 bytes per
    padded element plus metadata, and measure peak bytes in actual probes.
    """
    if x.device.type != 'cuda':
        raise ValueError('quantize_mxfp8_cuda requires CUDA-resident input')
    _validate_quantization_input(x)
    if check_finite and not bool(x.isfinite().all()):
        raise ValueError('nonfinite input cannot be quantized')
    return _quantize_mxfp8_resident(x)


def quantize_mxfp8(x):
    """Dispatch by input device without moving data; ceil-v1 codes on CPU/CUDA.

    FP64 intermediates preserve FP32 subnormal and maximal-value behavior.
    CUDA uses the explicit resident entry with one finite-validation scalar sync.
    """
    if x.device.type == 'cuda':
        return quantize_mxfp8_cuda(x)
    if x.device.type != 'cpu':
        raise ValueError('MXFP8 quantization supports CPU or CUDA tensors only')
    _validate_quantization_input(x)
    if not bool(x.isfinite().all()):
        raise ValueError('nonfinite input cannot be quantized')
    return _quantize_mxfp8_resident(x)


def quantize_mxfp8_cute(x, *, check_finite=True):
    """Pack BF16 on CUDA SM120. Disabling the check assumes finite inputs."""
    _validate_quantization_input(x)
    if x.dtype != torch.bfloat16 or x.device.type != 'cuda' or torch.cuda.get_device_capability(x.device) != (12, 0):
        raise ValueError('CuTe MXFP8 quantization requires BF16 on CUDA SM120')
    if check_finite and not bool(x.isfinite().all()):
        raise ValueError('nonfinite input cannot be quantized')
    from ._cute_mxfp8 import launch_quantizer
    values, scales = launch_quantizer(x)
    return MXFP8Tensor(values, scales, x.shape[1])


def dequantize_mxfp8(x, *, dtype=torch.float32):
    """Decode without hidden saturation; a narrow output dtype can overflow.

    For example maximal FP32 input rounds to MX value 2**128, finite in FP64
    but infinite on an explicitly requested FP32 decode. MMA FP32 accumulator
    overflow likewise remains the caller's numerical-range responsibility.
    """
    x.validate()
    if dtype not in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
        raise ValueError('invalid decoded dtype')
    return (x.values.double().reshape(x.values.shape[0], -1, 32)
            * decode_e8m0(x.scales)[..., None]).reshape(x.values.shape)[:, :x.logical_k].to(dtype)


def quantize_mxfp8_scalar_oracle(x):
    """CPU-only independently selected scales and nearest-even E4M3 codes.

    Exhaustive finite E4M3 codebook avoids reusing production float8 casting.
    """
    if x.device.type != 'cpu' or x.ndim != 2 or x.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError('scalar oracle requires a rank-2 CPU float matrix')
    if not x.numel() or not bool(x.isfinite().all()):
        raise ValueError('scalar oracle requires nonempty finite input')
    # Positive E4M3: exponent zero has no hidden bit; 0x7f is NaN.
    levels = [(code/512. if code < 8 else (1+(code % 8)/8.) * 2.**(code//8-7)) for code in range(127)]
    rows, k = x.shape
    kp = math.ceil(k/32)*32
    codes = torch.zeros((rows, kp), dtype=torch.uint8)
    scales = torch.empty((rows, kp//32), dtype=torch.uint8)
    for row in range(rows):
        for start in range(0, kp, 32):
            block = [float(v) for v in x[row, start:min(k, start+32)]]
            maximum = max(abs(v) for v in block)
            exponent = 0 if maximum == 0 else max(-127, min(127, math.ceil(math.log2(maximum/448.))))
            scales[row, start//32] = exponent+127
            for j, value in enumerate(block):
                normalized = min(448., abs(value) / 2.**exponent)
                nearest = min(range(127), key=lambda c: (abs(levels[c]-normalized), c % 2))
                codes[row, start+j] = nearest | (128 if math.copysign(1.,value)<0 else 0)
    return MXFP8Tensor(codes.view(torch.float8_e4m3fn), scales, k)


def mxfp8_gemm(a, b, *, backend='cute'):
    """A[M,K] @ B[N,K].T. Native candidate: FP32 accumulators/output.

    Use cute for the single-warp reference and cute-tiled for shared staging.
    Both are experimental and raise on missing APIs, compile or launch failures.
    """
    a.validate(); b.validate()
    if a.logical_k != b.logical_k or a.values.shape[1] != b.values.shape[1] or a.values.device != b.values.device:
        raise ValueError('incompatible K/device')
    if backend == 'oracle':
        return dequantize_mxfp8(a) @ dequantize_mxfp8(b).T
    if backend not in ('cute', 'cute-tiled'):
        raise ValueError('backend must be cute, cute-tiled or explicit oracle')
    if a.values.device.type != 'cuda':
        raise ValueError('native MXFP8 requires CUDA')
    from ._cute_mxfp8 import launch
    return launch(a, b, tiled=backend == 'cute-tiled')
