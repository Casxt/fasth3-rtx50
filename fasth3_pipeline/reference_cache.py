"""Bounded process-local cache for prompt-independent reference features."""
from collections import OrderedDict
import hashlib
from pathlib import Path

import numpy as np
import torch


def _tensor_digest(digest, value):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(repr((tuple(tensor.shape), str(tensor.dtype))).encode())
        array = tensor.reshape(-1).view(torch.uint8).numpy()
    else:
        array = np.ascontiguousarray(value)
        digest.update(repr((array.shape, array.dtype.str)).encode())
    digest.update(memoryview(array).cast('B'))


def _model_identity(directory):
    root = Path(directory).resolve(strict=True)
    return (root, tuple((path.relative_to(root), st.st_size, st.st_mtime_ns, st.st_ino)
                        for path in sorted(root.rglob('*')) if path.is_file()
                        for st in (path.stat(),)))


def vae_reference_key(pipe, state, directory):
    digest = hashlib.sha256()
    for ref in state.get('normalized_references'):
        digest.update(repr((ref.kind, getattr(ref, 'fps', None), getattr(ref, 'sample_rate', None),
                            ref.has_audio)).encode())
        if ref.kind == 'image':
            _tensor_digest(digest, ref.image)
        elif ref.kind == 'video':
            _tensor_digest(digest, ref.frames)
        if ref.has_audio:
            _tensor_digest(digest, ref.audio)
    return ('vae', _model_identity(Path(directory) / 'vae'), _model_identity(Path(directory) / 'audio_vae'),
            pipe.keyframe_encode_seed, tuple(pipe.pixel_mean), tuple(pipe.pixel_std), digest.digest())


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensors(item)


def _copy(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device, copy=True)
    if isinstance(value, list):
        return [_copy(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy(item, device) for item in value)
    return value


class ReferenceCache:
    def __init__(self, limit_bytes=512 << 20):
        if not 0 <= limit_bytes <= 2 << 30:
            raise ValueError('Reference cache budget must be between zero and 2 GiB')
        self.limit_bytes = limit_bytes
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = self.evictions = 0

    def get(self, key, device='cpu'):
        entry = self.entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        self.entries.move_to_end(key)
        return _copy(entry[1], device)

    def __contains__(self, key):
        return key in self.entries

    def put(self, key, value):
        size = sum(t.nbytes for t in _tensors(value))
        if not 0 < size <= self.limit_bytes:
            return
        old = self.entries.pop(key, None)
        if old is not None:
            self.bytes -= old[0]
        while self.bytes + size > self.limit_bytes:
            _, (removed, _) = self.entries.popitem(last=False)
            self.bytes -= removed
            self.evictions += 1
        self.entries[key] = size, _copy(value, 'cpu')
        self.bytes += size

    def snapshot(self):
        return {name: getattr(self, name) for name in ('bytes', 'hits', 'misses', 'evictions', 'limit_bytes')}


def memoize_qwen_vision(model, cache, directory, stream=None):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import BaseModelOutputWithDeepstackFeatures

    core = model.model
    original = core.get_image_features
    identity = _model_identity(directory)
    active_keys = {}

    def key_for(pixel_values, grid):
        digest = hashlib.sha256()
        _tensor_digest(digest, pixel_values)
        _tensor_digest(digest, grid)
        return ('qwen', identity, str(pixel_values.device), digest.digest())

    def forward(pixel_values, image_grid_thw=None, **kwargs):
        key = active_keys.get((id(pixel_values), id(image_grid_thw)))
        if key is None:
            key = key_for(pixel_values, image_grid_thw)
        features = cache.get(key, pixel_values.device)
        if features is not None:
            hidden, pooled, deepstack = features
            return BaseModelOutputWithDeepstackFeatures(last_hidden_state=hidden, pooler_output=pooled,
                                                       deepstack_features=deepstack)
        output = original(pixel_values, image_grid_thw, **kwargs)
        cache.put(key, (output.last_hidden_state, output.pooler_output, output.deepstack_features))
        return output

    core.get_image_features = forward
    if stream is not None:
        original_core = core.forward

        def cached_core(*args, **kwargs):
            for pixels, grid in (('pixel_values', 'image_grid_thw'), ('pixel_values_videos', 'video_grid_thw')):
                data, geometry = kwargs.get(pixels), kwargs.get(grid)
                if data is not None:
                    active_keys[id(data), id(geometry)] = key_for(data, geometry)
            omit = ('model.visual',) if active_keys and all(key in cache for key in active_keys.values()) else ()
            try:
                with stream.skip_prefetch_groups(omit):
                    return original_core(*args, **kwargs)
            finally:
                active_keys.clear()

        core.forward = cached_core
