import json
import sys
from pathlib import Path
import pytest
import torch
from torch import nn
from fasth3_pipeline.pdd import PDDPlan, PDDProjection
from fasth3_pipeline.attention import TileLayout, select_blocks


def test_pdd_shift_deltas_and_reference_rows():
    plan = PDDPlan.create()
    assert plan.indices == tuple(range(0, 33, 4))
    for name, shift in (("video", 12.), ("audio", 3.)):
        sigma = shift*plan.grid*.999 / (plan.grid*(shift-1)+.999)
        torch.testing.assert_close(plan.weights[name], sigma.diff(), atol=3e-16, rtol=2e-13)
        sample = torch.randn(1, 9, 3)
        initial = sample.clone()
        for step in range(8):
            sample = plan.update(sample, torch.ones_like(sample), name, step, condition_rows=2)
        torch.testing.assert_close(sample[:, :2], initial[:, :2], atol=0, rtol=0)
        torch.testing.assert_close(sample[:, 2:], initial[:, 2:]+.999)


def test_h3_clean_clock_and_initial_noise_scaling():
    plan = PDDPlan.create()
    initial = torch.ones(1, 5, 3)
    for modality in ("video", "audio"):
        sample = plan.scale_initial_noise(initial, modality, condition_rows=2)
        torch.testing.assert_close(sample[:, :2], initial[:, :2], atol=0, rtol=0)
        torch.testing.assert_close(sample[:, 2:], initial[:, 2:]*.999)
        assert plan.clean_time(modality, 0) == pytest.approx(.001)
        assert all(plan.clean_time(modality, i) < plan.clean_time(modality, i+1) for i in range(7))
        for step in range(8):
            # Known x0=0, eps=1: the raw H3 head is x0-eps=-1.
            sample = plan.update(sample, -torch.ones_like(sample), modality, step, condition_rows=2)
        torch.testing.assert_close(sample[:, 2:], torch.zeros_like(sample[:, 2:]), atol=1e-7, rtol=0)


@pytest.mark.parametrize("modality", ["video", "audio"])
def test_h3_raw_head_denoises_known_zero_x0_and_freezes_references(modality):
    plan = PDDPlan.create()
    eps = torch.randn(1, 11, 5, generator=torch.Generator().manual_seed(17))
    sample = plan.scale_initial_noise(eps, modality, condition_rows=2)
    raw_prediction = -eps.clone()  # H3 checkpoint emits x0-eps, not eps-x0.
    raw_prediction[:, :2] = 10000  # Conditioning predictions must never be applied.
    previous_rms = sample[:, 2:].square().mean().sqrt()
    for step in range(8):
        sample = plan.update(sample, raw_prediction, modality, step, condition_rows=2)
        torch.testing.assert_close(sample[:, :2], eps[:, :2], atol=0, rtol=0)
        expected = eps[:, 2:]*float(plan.sigmas[modality][step+1])
        torch.testing.assert_close(sample[:, 2:], expected, atol=4e-7, rtol=2e-6)
        rms = sample[:, 2:].square().mean().sqrt()
        assert rms < previous_rms
        previous_rms = rms
    torch.testing.assert_close(sample[:, 2:], torch.zeros_like(sample[:, 2:]), atol=4e-7, rtol=0)


def test_pdd_projection_fuses_before_linear():
    torch.manual_seed(1)
    plan = PDDPlan.create()
    layer = PDDProjection(7, 12)
    x = torch.randn(2, 3, 7)
    full = nn.functional.linear(x, layer.weight, layer.bias).unflatten(-1, (32, 12))
    with pytest.raises(RuntimeError, match="explicit fused"):
        layer(x)
    for start, end in zip(plan.indices, plan.indices[1:]):
        weights = plan.weights["video"][start:end].float()
        reference = (full[..., start:end, :]*weights[None, None, :, None]).sum(-2)/weights.sum()
        with layer.fuse(start, end, plan.weights["video"]):
            torch.testing.assert_close(layer(x), reference, atol=4e-7, rtol=2e-5)
    assert layer._fusion is None


def test_geometry_interleaves_dense_and_multiple_video_regions():
    layout = TileLayout.build([5, (3, 5, 11), 7, (5, 4, 8)])
    assert layout.prefix_tiles == 2
    assert layout.video_spans == ((2, 6), (6, 8))
    assert layout.lengths.sum() == 5+3*5*11+7+5*4*8
    x = torch.randn(1, layout.inverse.numel(), 2, 4)
    tiled = layout.tile(x)
    torch.testing.assert_close(tiled[:, layout.inverse], x, atol=0, rtol=0)
    pooled = layout.pool(tiled)
    for tile, n in enumerate(layout.lengths.tolist()):
        torch.testing.assert_close(pooled[:, :, tile], tiled[:, tile*128:tile*128+n].float().mean(1))
    scores = torch.randn(1, 2, len(layout.lengths), len(layout.lengths))
    mask = select_blocks(scores, layout)
    assert mask[:, :, :2].all()
    assert mask[:, :, :, :2].all()
    assert (mask[:, :, 2:, 2:6].sum(-1) == 1).all()
    assert (mask[:, :, 2:, 6:8].sum(-1) == 1).all()


def test_diagnostics_stop_counts_and_metadata_without_cuda():
    from fasth3_pipeline.diagnostics import PipelineDiagnostics
    calls = []
    diag = PipelineDiagnostics(stop_after="first-denoise", guard=calls.append)
    noise = torch.ones(1, 3, 2, dtype=torch.bfloat16)
    assert not diag.checkpoint("noise-scaled", {"video": noise})
    diag.forward_count = 1
    assert diag.checkpoint("first-denoise")
    assert diag.stopped("first-denoise") == {"stopped_after": "first-denoise", "transformer_forwards": 1}
    assert diag.events[0]["tensors"]["video"] == {"shape": [1, 3, 2], "dtype": "torch.bfloat16"}
    assert calls == ["checkpoint:noise-scaled", "checkpoint:first-denoise"]
    assert 0 <= diag.events[0]["elapsed_seconds"] <= diag.events[1]["elapsed_seconds"]
    assert all(event["stage_elapsed_seconds"] >= 0 for event in diag.events)
    with pytest.raises(ValueError, match="Unknown stop"):
        PipelineDiagnostics(stop_after="fake-stage")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_diagnostics_rejects_nonfinite_even_without_snapshot(bad):
    from fasth3_pipeline.diagnostics import PipelineDiagnostics
    assert PipelineDiagnostics(stop_after="encoder").checkpoint("encoder", {"prompt_embeds": torch.tensor([1., bad])})
    diag = PipelineDiagnostics(stop_after="encoder", debug_checks=True)
    with pytest.raises(FloatingPointError, match="encoder.*prompt_embeds"):
        diag.checkpoint("encoder", {"prompt_embeds": torch.tensor([1., bad])})
    assert diag.events[-1]["tensors"]["prompt_embeds"]["finite"] is False


def test_diagnostics_saves_actual_tensor_snapshot(tmp_path):
    from fasth3_pipeline.diagnostics import PipelineDiagnostics
    from safetensors.torch import load_file
    diag = PipelineDiagnostics(tmp_path)
    original = torch.arange(12).reshape(3, 4).T  # deliberately non-contiguous
    diag.checkpoint("noise-unscaled", {"video": original})
    snapshot = load_file(tmp_path / "00-noise-unscaled.safetensors")
    torch.testing.assert_close(snapshot["video"], original)
    assert json.loads((tmp_path / "stages.json").read_text())[0]["stage"] == "noise-unscaled"


def test_no_forbidden_runtime_imports():
    assert not any(name.split('.')[0] in {"fastvideo", "flashinfer", "triton"} for name in sys.modules)
