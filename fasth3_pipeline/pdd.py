"""Independent FastH3 PDD fusion with MiniMax-H3's native raw-head direction.

Contract reference: FastVideo ce970eb715215c79a169658d7aa6c8a96e64891e,
fastvideo/layers/pdd.py and models/schedulers/scheduling_minimax_h3.py
(Apache-2.0). No FastVideo runtime dependency.

The checkpoint emits raw x0-eps. Training adapters negate it when exposing
native sigma velocity; inference must not mistake that adapter's contract
for the checkpoint output. With negative sigma differences, subtract the
fused raw head, matching MiniMaxH3Scheduler's positive clean-time update.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PDDPlan:
    grid: torch.Tensor
    indices: tuple[int, ...]
    sigmas: dict[str, torch.Tensor]
    weights: dict[str, torch.Tensor]

    @classmethod
    def create(cls, steps=32, indices=tuple(range(0, 33, 4)), max_t=.999,
               video_shift=12., audio_shift=3.):
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
            raise ValueError("PDD steps must be an integer >= 2")
        if not indices or any(type(i) is not int for i in indices):
            raise ValueError("PDD indices must be integers")
        if indices[0] != 0 or indices[-1] != steps or any(a >= b for a, b in zip(indices, indices[1:])):
            raise ValueError("PDD indices must increase strictly from zero to steps")
        if not math.isfinite(max_t) or not 0 < max_t <= 1:
            raise ValueError("Invalid PDD max_t")
        grid = torch.linspace(max_t, 0, steps + 1, dtype=torch.float64)
        sigmas, weights = {}, {}
        for name, shift in (("video", video_shift), ("audio", audio_shift)):
            if not math.isfinite(shift) or shift <= 0:
                raise ValueError("PDD shift must be finite and positive")
            denominator = max_t + (shift - 1) * grid
            sigmas[name] = (shift * grid * max_t / denominator)[list(indices)]
            # Stable rational difference, rather than subtracting nearby sigmas.
            weights[name] = shift * max_t**2 * (grid[1:] - grid[:-1]) / (denominator[:-1] * denominator[1:])
        return cls(grid, tuple(indices), sigmas, weights)

    def scale_initial_noise(self, sample, modality, condition_rows=0):
        if not 0 <= condition_rows <= sample.shape[1]:
            raise ValueError("Invalid conditioning row count")
        result = sample.clone()
        result[:, condition_rows:] *= float(self.sigmas[modality][0])
        return result

    def clean_time(self, modality, step):
        """H3 embeds clean time, while PDD integration uses noise sigma."""
        return 1.0-float(self.sigmas[modality][step])

    def update(self, sample, prediction, modality, step, condition_rows=0):
        """Update targets with a normalized fused raw x0-eps head; freeze references.

        ``weights`` store negative sigma differences. H3's raw projection has
        the opposite direction, hence ``x -= delta_sigma * raw_prediction``.
        """
        start, end = self.indices[step:step+2]
        if not 0 <= condition_rows <= sample.shape[1] or prediction.shape != sample.shape:
            raise ValueError("Invalid packed prediction or conditioning row count")
        result = sample.clone()
        target = sample[:, condition_rows:].float()
        delta = self.weights[modality][start:end].sum().to(sample.device, torch.float32)
        result[:, condition_rows:] = (target - delta * prediction[:, condition_rows:].float()).to(sample.dtype)
        return result


class PDDProjection(nn.Linear):
    """32-head checkpoint projection, fused before GEMM to bound activations.

    H3 rows use channel-major (grid, C, patch) layout, patch_size=1 for
    audio. Fusion preserves the checkpoint parameter dtype after FP32 accumulation.
    """
    def __init__(self, in_features, head_features, grid_size=32, bias=True, *, device=None, dtype=None):
        super().__init__(in_features, grid_size * head_features, bias, device=device, dtype=dtype)
        self.grid_size = grid_size
        self.head_features = head_features
        self._fusion = None
        self._combined = None

    @contextmanager
    def fuse(self, start, end, weights):
        if not 0 <= start < end <= self.grid_size or weights.shape != (self.grid_size,):
            raise ValueError("Invalid fusion range/weights")
        previous, combined = self._fusion, self._combined
        self._fusion = (start, end, weights)
        self._combined = None
        try:
            yield self
        finally:
            self._fusion = previous
            self._combined = combined

    def forward(self, x):
        if self._fusion is None:
            raise RuntimeError("PDD projection requires an explicit fused block; full-grid output is disabled")
        if self._combined is not None and not torch.is_grad_enabled():
            return F.linear(x, *self._combined)
        start, end, weights = self._fusion
        selected = weights[start:end].to(self.weight.device)
        total = selected.sum()
        if not torch.isfinite(total) or total == 0:
            raise ValueError("Invalid fusion weights")
        normalized = (selected / total).float()
        def combine(parameter):
            heads = parameter.unflatten(0, (self.grid_size, self.head_features))[start:end].float()
            return torch.einsum('n,n...->...', normalized, heads).to(parameter.dtype)
        combined = (combine(self.weight), None if self.bias is None else combine(self.bias))
        if not torch.is_grad_enabled():
            self._combined = combined
        return F.linear(x, *combined)
