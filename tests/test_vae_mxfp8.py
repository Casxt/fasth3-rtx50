import json
import os
from types import SimpleNamespace

import pytest
import torch
from diffusers import AutoencoderKLMiniMaxH3
from diffusers.models.attention import FeedForward
from safetensors.torch import save_file

from fasth3_pipeline.mxfp8_linear import MXFP8Linear
from fasth3_pipeline.storage import INDEX, PersistentTensorStore
from fasth3_pipeline.vae_mxfp8 import (
    SCHEMA, ffn_linears, install_vae_mxfp8, load_vae_decoder, source_digest, verify_vae_overlay,
)


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


@pytest.fixture
def indexed_vae(tmp_path, monkeypatch):
    tmp_path = tmp_path / 'model'
    tmp_path.mkdir()
    vae = AutoencoderKLMiniMaxH3(block_out_channels=(32,), spatial_downsample_factors=(1,),
                                 temporal_downsample_factors=(1,), decoder_num_layers=1,
                                 decoder_num_attention_heads=2)
    vae.save_config(tmp_path)
    state = vae.state_dict()
    shard = tmp_path / 'diffusion_pytorch_model-00001-of-00001.safetensors'
    save_file(state, shard, metadata={'format': 'pt'})
    index = tmp_path / INDEX
    index.write_text(json.dumps({'weight_map': dict.fromkeys(state, shard.name),
                                 'metadata': {'total_size': sum(t.nbytes for t in state.values())}}), encoding='utf-8')
    vae.bfloat16()
    tensors, metadata = {}, {'schema': SCHEMA}
    for name, _, _, _, source in ffn_linears(vae):
        n, k = source.weight.shape
        tensors[name + '.weight'] = torch.zeros((n, k), dtype=torch.float8_e4m3fn)
        tensors[name + '.weight_scale'] = torch.zeros((n // 32, k // 128, 32, 4), dtype=torch.uint8)
        metadata[name] = source_digest(source)
    overlay = tmp_path / 'overlay.safetensors'
    save_file(tensors, overlay, metadata=metadata)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda _: (12, 0))
    return tmp_path, shard, index, overlay, set(tensors).intersection(state)


def test_verified_overlay_load_skips_original_ffn_payloads(indexed_vae, monkeypatch):
    directory, _, _, overlay, skipped = indexed_vae
    expected = AutoencoderKLMiniMaxH3.from_pretrained(directory, dtype=torch.bfloat16, local_files_only=True)
    install_vae_mxfp8(expected, overlay, 'cuda:0', lambda _: None)
    load_vae_decoder(directory, overlay, 'cuda:0', lambda _: None)  # legacy migration
    reads, tensor = set(), PersistentTensorStore.tensor

    def track(store, key):
        reads.add(key)
        return tensor(store, key)

    monkeypatch.setattr(PersistentTensorStore, 'tensor', track)
    actual = load_vae_decoder(directory, overlay, 'cuda:0', lambda _: None)
    assert skipped and not (skipped & reads)
    assert not actual.training
    for key, target in expected.state_dict().items():
        observed = actual.state_dict()[key]
        assert observed.dtype == target.dtype
        assert torch.equal(observed.flatten().view(torch.uint8), target.flatten().view(torch.uint8)), key
    expected_buffers, actual_buffers = dict(expected.named_buffers()), dict(actual.named_buffers())
    assert actual_buffers['decoder.rope.inv_freq'].dtype == torch.float32
    assert torch.equal(actual_buffers['decoder.rope.inv_freq'], expected_buffers['decoder.rope.inv_freq'])


def test_verified_overlay_rejects_source_change_until_prepared(indexed_vae):
    directory, shard, _, overlay, _ = indexed_vae
    verify_vae_overlay(directory, overlay)
    stat = shard.stat()
    os.utime(shard, ns=(stat.st_atime_ns, stat.st_mtime_ns + 100_000_000))
    with pytest.raises(ValueError, match='source changed'):
        load_vae_decoder(directory, overlay, 'cuda:0', lambda _: None)
    verify_vae_overlay(directory, overlay)
    load_vae_decoder(directory, overlay, 'cuda:0', lambda _: None)


def test_verified_overlay_rebinds_a_model_directory_move(indexed_vae):
    directory, _, _, overlay, _ = indexed_vae
    verify_vae_overlay(directory, overlay)
    relocated = directory.rename(directory.with_name('moved'))
    load_vae_decoder(relocated, relocated / overlay.name, 'cuda:0', lambda _: None)
    receipt = json.loads((relocated / (overlay.name + '.verified.json')).read_text(encoding='utf-8'))
    assert receipt['source'] == str(relocated)


def test_verified_overlay_rejects_payload_corruption(indexed_vae):
    directory, _, _, overlay, _ = indexed_vae
    verify_vae_overlay(directory, overlay)
    with overlay.open('r+b') as stream:
        stream.seek(-1, os.SEEK_END)
        byte = stream.read(1)
        stream.seek(-1, os.SEEK_END)
        stream.write(bytes([byte[0] ^ 1]))
    with pytest.raises(ValueError, match='failed checksum'):
        load_vae_decoder(directory, overlay, 'cuda:0', lambda _: None)


def test_overlay_loader_rejects_missing_nonoverlay_tensor(indexed_vae):
    directory, _, index, overlay, _ = indexed_vae
    data = json.loads(index.read_text(encoding='utf-8'))
    del data['weight_map']['post_quant_conv.weight']
    index.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='architecture mismatch.*post_quant_conv.weight'):
        load_vae_decoder(directory, overlay, 'cuda:0', lambda _: None)
