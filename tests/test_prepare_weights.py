import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fasth3_pipeline import prepare_weights as weights


def test_profile_paths_are_explicit(tmp_path):
    for task in ('f2va', 'fl2va', 'r2va'):
        paths = weights.generation_assets(tmp_path, task)
        assert paths['base_model'] == tmp_path / 'bf16'
        assert paths['transformer_path'] == tmp_path / 'mxfp8-packed' / 'transformer_ref'
        assert paths['text_encoder_path'] == tmp_path / 'encoder-nvfp4' / 'text_encoder'
    t2va = weights.generation_assets(tmp_path, 't2va')
    assert t2va['base_model'] == tmp_path / 't2va-v2-bf16'
    assert t2va['transformer_path'] == tmp_path / 't2va-v2-mxfp8-packed' / 'transformer'
    with pytest.raises(ValueError, match='Unknown generation task'):
        weights.generation_assets(tmp_path, 'ti2v')


def test_pinned_allowlist_rejects_unrelated_variants_and_escaped_paths():
    assert weights._allow(weights.T2VA, 'transformer/diffusion_pytorch_model-00001-of-00014.safetensors')
    assert not weights._allow(weights.OMNI, 'transformer/diffusion_pytorch_model-00001-of-00014.safetensors')
    assert weights._allow(weights.OMNI, 'transformer_ref/config.json')
    assert not weights._allow(weights.T2VA, 'text_encoder/model-00001-of-00014.safetensors')
    assert weights._allow(weights.SHARED, 'LICENSE')
    assert not weights._allow(weights.SHARED, '../LICENSE')
    assert weights._allow(weights.ENCODER, 'text_encoder/model-00001-of-00004.safetensors')


def test_inventory_requires_expected_revision_and_digest(monkeypatch):
    src = weights.Source(weights.T2VA.folder, weights.T2VA.repo, weights.T2VA.revision, 1)
    sibling = SimpleNamespace(rfilename='fastvideo_inference.json', lfs=None, blob_id='a' * 40, size=17)
    info = SimpleNamespace(sha=src.revision, siblings=[sibling])
    api = SimpleNamespace(model_info=lambda *_args, **_kwargs: info)
    # Source selection is a closed registry. Patch the singleton for this small fixture.
    monkeypatch.setattr(weights, 'T2VA', src)
    assert weights._assets(api, src) == [weights.Asset('fastvideo_inference.json', 17, 'a' * 40, 'git-sha1')]
    info.sha = 'b' * 40
    with pytest.raises(ValueError, match='Revision mismatch'):
        weights._assets(api, src)
    info.sha, sibling.blob_id = src.revision, None
    with pytest.raises(ValueError, match='checksum metadata'):
        weights._assets(api, src)


def test_download_resumes_staging_then_publishes_verified_asset(tmp_path, monkeypatch):
    source = weights.T2VA
    data = b'original weights'
    asset = weights.Asset('transformer/config.json', len(data), hashlib.sha256(data).hexdigest(), 'sha256')
    staging = tmp_path / '.downloads' / source.folder / source.revision / (asset.path + '.partial')
    staging.parent.mkdir(parents=True)
    staging.write_bytes(data[:3])
    calls = []

    def curl(argv, *, input, text, check):
        calls.append(argv)
        assert 'fake-hf-token' not in str(argv)
        assert 'fake-hf-token' in input and '--continue-at' in argv
        partial = Path(argv[argv.index('--output') + 1])
        assert partial.read_bytes() == data[:3]
        partial.write_bytes(data)

    monkeypatch.setattr(weights.subprocess, 'run', curl)
    weights._prepare_source('curl', tmp_path, source, [asset], 'fake-hf-token')
    target = tmp_path / source.folder / asset.path
    assert target.read_bytes() == data and not staging.exists()
    assert json.loads((tmp_path / source.folder / '.verified-source.json').read_text())['files'][asset.path]['digest'] == asset.digest
    weights._prepare_source('curl', tmp_path, source, [asset], 'fake-hf-token')
    assert len(calls) == 1
    target.write_bytes(b'altered')
    with pytest.raises(ValueError, match='failed verification'):
        weights._prepare_source('curl', tmp_path, source, [asset], 'fake-hf-token')


def test_download_corruption_leaves_no_final_asset(tmp_path, monkeypatch):
    source = weights.T2VA
    asset = weights.Asset('fastvideo_inference.json', 5, 'a' * 64, 'sha256')

    def bad_download(argv, **_kwargs):
        Path(argv[argv.index('--output') + 1]).write_bytes(b'wrong')

    monkeypatch.setattr(weights.subprocess, 'run', bad_download)
    with pytest.raises(ValueError, match='checksum'):
        weights._download('curl', tmp_path, source, asset, None)
    assert not (tmp_path / source.folder / asset.path).exists()


def test_converted_shards_are_adopted_only_with_matching_hashes(tmp_path):
    from fasth3_pipeline.storage import INDEX

    packed = tmp_path / 'packed'
    packed.mkdir()
    shard = packed / 'shard.safetensors'
    data = b'verified export'
    shard.write_bytes(data)
    (packed / INDEX).write_text(json.dumps({'weight_map': {'a': shard.name}}), encoding='utf-8')
    manifest = {'shard_sha256': {shard.name: hashlib.sha256(data).hexdigest()}}
    weights._verify_conversion(tmp_path, packed, manifest)
    shard.write_bytes(b'corrupted data!')
    with pytest.raises(ValueError, match='checksum verification'):
        weights._verify_conversion(tmp_path, packed, manifest)
