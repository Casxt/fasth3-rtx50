"""Exact per-step AdaLN tables for the fixed FastH3 PDD schedule."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .storage import INDEX


def checkpoint_identity(directory):
    directory = Path(directory)
    digest = hashlib.sha256()
    for name in (INDEX, 'mxfp8_manifest.json', 'config.json'):
        digest.update((directory / name).read_bytes())
    return digest.hexdigest()


class AdaLNTable:
    def __init__(self, path, directory, blocks):
        from safetensors import safe_open
        from diffusers.models.transformers.transformer_minimax_h3 import MINIMAX_H3_MODALITY_NUM

        with safe_open(path, framework='pt', device='cpu') as file:
            meta = file.metadata() or {}
            if meta.get('schema') != 'fasth3-adaln-v1' or meta.get('checkpoint') != checkpoint_identity(directory):
                raise ValueError('AdaLN table does not match the MXFP8 checkpoint')
            times = [tuple(row) for row in json.loads(meta['timesteps'])]
            if not times or len(times) != len(set(times)) or any(
                    not 1 <= len(row) <= 4 or tuple(sorted(set(row))) != row
                    or any(not 0 <= t <= 1 for t in row) for row in times):
                raise ValueError('Invalid AdaLN timestep table')
            self.ranges = {}
            offset = 0
            for row in times:
                stop = offset + len(row) * MINIMAX_H3_MODALITY_NUM
                self.ranges[row] = slice(offset, stop)
                offset = stop
            keys = [f'block-{i}' for i in range(len(blocks))]
            if set(file.keys()) != set(keys):
                raise ValueError('AdaLN table must cover every transformer block')
            self.rows = []
            for key, block in zip(keys, blocks, strict=True):
                width = block.adaln_proj.hidden_size
                view = file.get_slice(key)
                if view.get_shape() != [offset, 6 * width] or view.get_dtype() != 'BF16':
                    raise ValueError(f'Invalid AdaLN tensor metadata: {key}')
                self.rows.append(file.get_tensor(key))
        self.current = None

    @contextmanager
    def use(self, times):
        key = tuple(times.tolist())
        if key not in self.ranges:
            raise ValueError(f'AdaLN table does not cover timesteps {key}')
        if self.current is not None:
            raise RuntimeError('AdaLN table already has an active forward')
        self.current = self.ranges[key]
        try:
            yield
        finally:
            self.current = None


class TableProjection(nn.Module):
    def __init__(self, table, index):
        super().__init__()
        self.table, self.index = table, index

    def forward(self, temb):
        if self.table.current is None:
            raise RuntimeError('AdaLN table requires the explicit PDD timestep scope')
        return self.table.rows[self.index][self.table.current].to(temb.device).chunk(6, dim=-1)


def install_adaln_table(model, store, path):
    table = AdaLNTable(path, store.directory, model.transformer_blocks)
    for index, block in enumerate(model.transformer_blocks):
        prefix = f'transformer_blocks.{index}.adaln_proj.'
        for key, value in block.adaln_proj.state_dict().items():
            if tuple(store.shape(prefix + key)) != tuple(value.shape) or store.dtype(prefix + key) != torch.bfloat16:
                raise ValueError(f'Invalid source AdaLN tensor: {prefix}{key}')
            del store.weight_map[prefix + key]
        block.adaln_proj = TableProjection(table, index)
    model._adaln_table = table
