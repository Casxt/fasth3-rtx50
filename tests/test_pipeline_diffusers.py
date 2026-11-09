"""Small CPU architecture/integration tests, never a large-model GPU run."""
import pytest
import torch
from torch import nn
from fasth3_pipeline.pdd import PDDPlan


@pytest.mark.parametrize("modality,shift", [("video", 12.), ("audio", 3.)])
@pytest.mark.parametrize("raw_dtype", [torch.float32, torch.bfloat16])
def test_pdd_raw_head_updates_match_independent_official_scheduler(modality, shift, raw_dtype):
    from diffusers import MiniMaxH3Scheduler
    plan = PDDPlan.create()
    # PDD's supplied sigmas are already shifted; do not shift them a second time.
    scheduler = MiniMaxH3Scheduler(shift=shift, shift_provided_sigmas=False)
    scheduler.set_timesteps(sigmas=plan.sigmas[modality].float(), device="cpu")
    torch.testing.assert_close(scheduler.sigmas, plan.sigmas[modality].float(), atol=0, rtol=0)
    generator = torch.Generator().manual_seed(23)
    sample = torch.randn(1, 11, 5, generator=generator)
    frozen = sample[:, :2].clone()
    oracle = sample[:, 2:].clone()
    for step, timestep in enumerate(scheduler.timesteps):
        raw = torch.randn(sample.shape, generator=generator).to(raw_dtype)
        raw[:, :2] = 10000
        oracle = scheduler.step(raw[:, 2:].float(), timestep, oracle, return_dict=False)[0]
        sample = plan.update(sample, raw, modality, step, condition_rows=2)
        torch.testing.assert_close(sample[:, :2], frozen, atol=0, rtol=0)
        torch.testing.assert_close(sample[:, 2:], oracle, atol=5e-7, rtol=2e-6)


def test_transformer_rejects_legacy_checkpoint(tmp_path):
    from fasth3_pipeline.transformer import load_transformer

    with pytest.raises(ValueError, match='packed MXFP8 checkpoint'):
        load_transformer(tmp_path, 'cpu')


def test_vae_offload_handles_weightnorm_and_direct_qkv_weight():
    import copy
    from torch.nn.utils import weight_norm
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import MiniMaxH3AudioCausalAttention
    from fasth3_pipeline.runner import offload_vae
    class TinyAudio(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = weight_norm(nn.Conv1d(16, 16, 3, padding=1))
            self.attn = MiniMaxH3AudioCausalAttention(16, 4, 2)
        def forward(self, x):
            return self.attn(self.conv(x).transpose(1, 2))
    model = TinyAudio().eval()
    x = torch.randn(1, 16, 7)
    with torch.no_grad():
        expected = model(x)
        calls = []
        handle = offload_vae(model, "cpu", calls.append)
        try:
            for _ in range(2):
                torch.testing.assert_close(model(x), expected)
        finally:
            handle.close()
    assert any(phase.startswith("block load:") for phase in calls)


def test_official_block_wiring_and_defaults():
    from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import MiniMaxH3Blocks
    blocks = MiniMaxH3Blocks()
    assert set(blocks.sub_blocks) == {"before_encode", "text_encoder", "vae_encoder", "denoise", "decode"}
    core = blocks.sub_blocks["denoise"].sub_blocks["ref2va"]
    for name in ("prepare_layout", "prepare_condition_latents", "prepare_latents", "prepare_latents_ref2va", "after_denoise"):
        assert name in core.sub_blocks
