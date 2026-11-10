"""Fused VAE denormalization, finite check, and channel-last RGB8 export."""
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver as cuda
import torch


@dsl_user_op
def _denormalize(value, std, mean, *, loc=None, ip=None):
    # Match the two eager FP32 ops, including rounding before the addition.
    operands = [x.ir_value(loc=loc, ip=ip) for x in (value, std, mean)]
    result = llvm.inline_asm(
        cutlass.Float32.mlir_type, operands,
        '{ .reg .f32 scaled; mul.rn.f32 scaled, $1, $2; add.rn.f32 $0, scaled, $3; }',
        '=f,f,f,f', loc=loc, ip=ip)
    return cutlass.Float32(result)


@cute.kernel
def _rgb8(X, Mean, Std, Output, Flag, Pixels: cutlass.Constexpr,
          Frames: cutlass.Constexpr, Height: cutlass.Constexpr, Width: cutlass.Constexpr):
    lane, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    pixel = block * 256 + lane
    invalid = cutlass.Boolean(False)
    if pixel < Pixels:
        x = pixel % Width
        y = pixel // Width % Height
        frame = pixel // (Width * Height) % Frames
        batch = pixel // (Width * Height * Frames)
        for channel in cutlass.range_constexpr(3):
            value = _denormalize(X[batch, channel, frame, y, x].to(cutlass.Float32), Std[channel], Mean[channel])
            # Check the FP32 exponent before clamp can turn infinities into valid pixels.
            finite = (value.bitcast(cutlass.Uint32) & 0x7f800000) != 0x7f800000
            invalid = invalid | ~finite
            color = cutlass.Uint8(0)
            if finite:
                clamped = cute.arch.fmin(cute.arch.fmax(value, 0.0), 1.0)
                color = (clamped * 255.0).to(cutlass.Uint8)
            Output[pixel * 3 + channel] = color
    if cute.arch.vote_any_sync(invalid) and lane % 32 == 0:
        cute.arch.atomic_or(Flag.iterator, cutlass.Int32(1))


@cute.jit
def _run(X, Mean, Std, Output, Flag, Pixels: cutlass.Constexpr,
         Frames: cutlass.Constexpr, Height: cutlass.Constexpr, Width: cutlass.Constexpr,
         stream: cuda.CUstream):
    _rgb8(X, Mean, Std, Output, Flag, Pixels, Frames, Height, Width).launch(
        grid=((Pixels + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream)


_CACHE = {}


def rgb8_to_host(decoded, mean, std):
    """Return one THWC RGB chunk and reuse its mandatory D2H sync for validation."""
    batch, _, frames, height, width = decoded.shape
    pixels = batch * frames * height * width
    payload = pixels * 3
    flag_offset = (payload + 3) // 4 * 4
    # Carry the flag in the RGB buffer so one synchronous D2H copies both.
    storage = torch.empty(flag_offset + 4, dtype=torch.uint8, device=decoded.device)
    flag = storage[flag_offset:].view(torch.int32).zero_()
    views = tuple(from_dlpack(t, assumed_align=t.element_size())
                  for t in (decoded.detach(), mean.flatten(), std.flatten(), storage, flag))
    args = (pixels, frames, height, width)
    stream = cuda.CUstream(torch.cuda.current_stream(decoded.device).cuda_stream)
    key = (decoded.device.index, decoded.dtype, tuple(decoded.shape), decoded.stride())
    compiled = _CACHE.get(key)
    if compiled is None:
        compiled = _CACHE[key] = cute.compile(_run, *views, *args, stream)
    compiled(*views, stream)
    host = storage.cpu()
    if bool(host[flag_offset:].view(torch.int32)[0]):
        raise FloatingPointError('Non-finite decoded video before export')
    return host[:payload].numpy().reshape(batch, frames, height, width, 3)
