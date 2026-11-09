from types import SimpleNamespace

import pytest
import torch
from diffusers.models.attention import FeedForward
from safetensors.torch import save_file

from fasth3_pipeline.mxfp8_linear import MXFP8Linear
from fasth3_pipeline.vae_mxfp8 import SCHEMA, ffn_linears, install_vae_mxfp8, source_digest


def test_overlay_rejects_wrong_source_before_replacing_any_linear(tmp_path, monkeypatch):
    ff = FeedForward(128, inner_dim=128, activation_fn='swiglu').bfloat16()
    vae = SimpleNamespace(decoder=SimpleNamespace(transformer_blocks=[SimpleNamespace(ff=ff)]))
    targets = list(ffn_linears(vae))
    tensors, metadata = {}, {'schema': SCHEMA}
    for name, _, _, _, source in targets:
        n, k = source.weight.shape
        tensors[name + '.weight'] = torch.zeros((n, k), dtype=torch.float8_e4m3fn)
        tensors[name + '.weight_scale'] = torch.zeros((n // 32, k // 128, 32, 4), dtype=torch.uint8)
        metadata[name] = source_digest(source)
    original = ff.net[0].proj
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda _: (12, 0))
    path = tmp_path / 'overlay.safetensors'
    save_file(tensors, path, metadata={**metadata, targets[-1][0]: 'wrong'})
    with pytest.raises(ValueError, match='source digest mismatch'):
        install_vae_mxfp8(vae, path, 'cuda:0', lambda _: None)
    assert ff.net[0].proj is original

    save_file(tensors, path, metadata=metadata)
    install_vae_mxfp8(vae, path, 'cuda:0', lambda _: None)
    assert isinstance(ff.net[0].proj, MXFP8Linear)
    assert ff.net[0].proj.bias is original.bias
