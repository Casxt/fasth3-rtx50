"""Offline packed MXFP8 overlay for the H3 video decoder FFNs."""
from contextlib import contextmanager
import hashlib
from functools import partial
import json
from pathlib import Path
from types import MethodType

import torch
from torch import nn

from .mxfp8_linear import MXFP8Linear
from .storage import INDEX, PersistentTensorStore

SCHEMA = 'fasth3-vae-ffn-mxfp8-v1'
RECEIPT_SCHEMA = 'fasth3-vae-ffn-verified-v1'


def ffn_linears(vae):
    for i, block in enumerate(vae.decoder.transformer_blocks):
        ff = block.ff
        if len(ff.net) != 3 or ff.net[1].p != 0 or type(ff.net[0]).__name__ != 'SwiGLU':
            raise ValueError(f'Unsupported VAE decoder FFN at block {i}')
        for name, owner, attr in (('net.0.proj', ff.net[0], 'proj'), ('net.2', ff.net, '2')):
            source = getattr(owner, attr)
            if not isinstance(source, nn.Linear) or source.weight.dtype != torch.bfloat16:
                raise ValueError(f'Expected a BF16 VAE decoder linear at block {i}, {name}')
            yield f'decoder.transformer_blocks.{i}.ff.{name}', ff, owner, attr, source


def source_digest(source):
    return _weight_digest(source.weight.detach())


def _weight_digest(weight):
    if weight.device.type != 'cpu' or not weight.is_contiguous():
        raise ValueError('VAE overlay requires the original contiguous CPU checkpoint')
    return hashlib.sha256(weight.view(torch.uint8).numpy()).hexdigest()


def _check_device(device):
    if torch.device(device).type != 'cuda' or torch.cuda.get_device_capability(device) != (12, 0):
        raise ValueError('VAE MXFP8 requires CUDA SM120')


def _overlay_targets(vae, file):
    targets = list(ffn_linears(vae))
    metadata = file.metadata() or {}
    keys = {name + suffix for name, *_ in targets for suffix in ('.weight', '.weight_scale')}
    if not keys or metadata.get('schema') != SCHEMA or set(file.keys()) != keys:
        raise ValueError('VAE MXFP8 overlay does not cover the video decoder FFNs')
    for name, _, _, _, source in targets:
        n, k = source.weight.shape
        weight, scale = (file.get_slice(name + suffix) for suffix in ('.weight', '.weight_scale'))
        if (n % 32 or k % 128 or weight.get_shape() != [n, k] or weight.get_dtype() != 'F8_E4M3'
                or scale.get_shape() != [n // 32, k // 128, 32, 4] or scale.get_dtype() != 'U8'):
            raise ValueError(f'Invalid VAE MXFP8 tensor metadata: {name}')
    return targets, metadata


def _install_targets(file, targets, guard, debug_checks):
    from .transformer import _pipelined_mxfp8_ffn, _streamed_fused_ffn

    replacements = []
    for name, ff, owner, attr, source in targets:
        guard(f'VAE MXFP8 load:{name}')
        layer = MXFP8Linear(source.in_features, source.out_features, bias=source.bias is not None,
                            device='meta', bias_dtype=source.bias.dtype if source.bias is not None else torch.bfloat16,
                            backend='cute-optimized', chunk_tokens=512, packed_scales=True,
                            debug_checks=debug_checks)
        layer.weight = file.get_tensor(name + '.weight')
        layer.weight_scale = file.get_tensor(name + '.weight_scale')
        layer.bias = source.bias
        replacements.append((ff, owner, attr, layer))
    for ff, owner, attr, layer in replacements:
        setattr(owner, attr, layer)
        if attr == '2':
            up = ff.net[0].proj
            pipelined = (not debug_checks and
                         (up.in_features, up.out_features, layer.in_features, layer.out_features)
                         == (2048, 16384, 8192, 2048))
            forward = partial(_pipelined_mxfp8_ffn, persistent_full=False) if pipelined else _streamed_fused_ffn
            ff.forward = MethodType(forward, ff)


def install_vae_mxfp8(vae, path, device, guard, *, debug_checks=False):
    """Install a legacy overlay on an already loaded BF16 video VAE."""
    from safetensors import safe_open

    _check_device(device)
    with safe_open(path, framework='pt', device='cpu') as file:
        targets, metadata = _overlay_targets(vae, file)
        for name, _, _, _, source in targets:
            if metadata.get(name) != source_digest(source):
                raise ValueError(f'VAE MXFP8 source digest mismatch: {name}')
        _install_targets(file, targets, guard, debug_checks)


@contextmanager
def _decoder_source(directory, overlay):
    from accelerate import init_empty_weights
    from diffusers import AutoencoderKLMiniMaxH3
    from safetensors import safe_open

    with PersistentTensorStore(directory) as store, safe_open(overlay, framework='pt', device='cpu') as file:
        config = AutoencoderKLMiniMaxH3.load_config(store.directory, local_files_only=True)
        dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            # Keep the nonpersistent rotary frequencies initialized in FP32 on CPU.
            with init_empty_weights(include_buffers=False):
                vae = AutoencoderKLMiniMaxH3.from_config(config)
        finally:
            torch.set_default_dtype(dtype)
        store.validate_keys(vae)
        targets, metadata = _overlay_targets(vae, file)
        yield vae, store, file, targets, metadata


def _file_stamp(path):
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns, stat.st_ino]


def _verify_overlay(store, overlay, targets, metadata, *, prepare=False, debug_checks=False):
    from .prepare_weights import _receipt

    # A local verification cache, like the verified conversion receipts. File
    # identities detect ordinary edits/replacements; debug mode rehashes content.
    overlay = Path(overlay).resolve(strict=True)
    receipt = overlay.with_name(overlay.name + '.verified.json')
    source_names = sorted({'config.json', INDEX, *store.weight_map.values()})
    def snapshot():
        return {
            'schema': RECEIPT_SCHEMA,
            'source': str(store.directory),
            'source_files': {name: _file_stamp(store.directory / name) for name in source_names},
            'overlay': str(overlay),
            'overlay_file': _file_stamp(overlay),
        }

    current = snapshot()
    previous = json.loads(receipt.read_text(encoding='utf-8')) if receipt.is_file() else None
    if previous is not None:
        digest = previous.get('overlay_sha256') if isinstance(previous, dict) else None
        if (not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)
                or previous.get('schema') != RECEIPT_SCHEMA):
            raise ValueError(f'Invalid VAE overlay verification receipt: {receipt}')
        if previous == {**current, 'overlay_sha256': digest} and not debug_checks:
            return
        if not prepare and current['source_files'] != previous.get('source_files'):
            raise ValueError('VAE overlay source changed; run fasth3 prepare to revalidate it')

    with overlay.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if previous is not None and digest != previous['overlay_sha256']:
        raise ValueError(f'VAE MXFP8 overlay failed checksum verification: {overlay}')
    for name, _, _, _, _ in targets:
        source = store.tensor(name + '.weight').to(torch.bfloat16)
        if metadata.get(name) != _weight_digest(source):
            raise ValueError(f'VAE MXFP8 source digest mismatch: {name}')
        del source
    if snapshot() != current:
        raise RuntimeError('VAE source or overlay changed during verification')
    try:
        _receipt(receipt, {**current, 'overlay_sha256': digest})
    except PermissionError:
        if prepare:
            raise
        # A read-only legacy model directory still works after verification.


def verify_vae_overlay(directory, overlay):
    """Bind a prepared overlay to the verified local VAE assets."""
    with _decoder_source(directory, overlay) as (_, store, _, targets, metadata):
        _verify_overlay(store, overlay, targets, metadata, prepare=True)


def load_vae_decoder(directory, overlay, device, guard, *, debug_checks=False):
    """Load all original non-overlay weights and attach verified MXFP8 buffers."""
    _check_device(device)
    guard('VAE decode:initialize')
    with _decoder_source(directory, overlay) as (vae, store, file, targets, metadata):
        _verify_overlay(store, overlay, targets, metadata, debug_checks=debug_checks)
        skipped = {name + '.weight' for name, *_ in targets}
        fp32_modules = set(vae._keep_in_fp32_modules or ())
        state = {}
        last_shard = None
        for key, shard in sorted(store.weight_map.items(), key=lambda entry: entry[1]):
            if key in skipped:
                continue
            if shard != last_shard:
                guard(f'VAE decode:load {shard}')
                last_shard = shard
            value = store.tensor(key)
            if value.is_floating_point():
                dtype = torch.float32 if fp32_modules.intersection(key.split('.')) else torch.bfloat16
                value = value.to(dtype)
            state[key] = value
        result = vae.load_state_dict(state, strict=False, assign=True)
        if set(result.missing_keys) != skipped or result.unexpected_keys:
            raise RuntimeError('VAE selective load did not cover all non-overlay tensors')
        _install_targets(file, targets, guard, debug_checks)
    return vae.eval()
