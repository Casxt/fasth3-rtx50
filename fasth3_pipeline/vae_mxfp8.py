"""Offline packed MXFP8 overlay for the H3 video decoder FFNs."""
import hashlib
from functools import partial
from types import MethodType

import torch
from torch import nn

from .mxfp8_linear import MXFP8Linear

SCHEMA = 'fasth3-vae-ffn-mxfp8-v1'


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
    weight = source.weight.detach()
    if weight.device.type != 'cpu' or not weight.is_contiguous():
        raise ValueError('VAE overlay requires the original contiguous CPU checkpoint')
    return hashlib.sha256(weight.view(torch.uint8).numpy()).hexdigest()


def install_vae_mxfp8(vae, path, device, guard, *, debug_checks=False):
    from safetensors import safe_open
    from .transformer import _pipelined_mxfp8_ffn, _streamed_fused_ffn

    if torch.device(device).type != 'cuda' or torch.cuda.get_device_capability(device) != (12, 0):
        raise ValueError('VAE MXFP8 requires CUDA SM120')
    targets = list(ffn_linears(vae))
    with safe_open(path, framework='pt', device='cpu') as file:
        metadata = file.metadata() or {}
        keys = {name + suffix for name, *_ in targets for suffix in ('.weight', '.weight_scale')}
        if not keys or metadata.get('schema') != SCHEMA or set(file.keys()) != keys:
            raise ValueError('VAE MXFP8 overlay does not cover the video decoder FFNs')
        replacements = []
        for name, ff, owner, attr, source in targets:
            guard(f'VAE MXFP8 load:{name}')
            if metadata.get(name) != source_digest(source):
                raise ValueError(f'VAE MXFP8 source digest mismatch: {name}')
            weight = file.get_tensor(name + '.weight')
            scales = file.get_tensor(name + '.weight_scale')
            layer = MXFP8Linear(source.in_features, source.out_features, bias=source.bias is not None,
                                 device='meta', bias_dtype=source.bias.dtype if source.bias is not None else torch.bfloat16,
                                 backend='cute-optimized', chunk_tokens=512, packed_scales=True,
                                 debug_checks=debug_checks)
            if (weight.shape != layer.weight.shape or weight.dtype != layer.weight.dtype or
                    scales.shape != layer.weight_scale.shape or scales.dtype != layer.weight_scale.dtype):
                raise ValueError(f'Invalid VAE MXFP8 tensor metadata: {name}')
            layer.weight, layer.weight_scale = weight, scales
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
