"""H3 P2 multi-region VSA routing, separate from the fine-attention kernel.

Omni uses (4, 4, 8) logical video tiles and V2 uses (4, 4, 4).
"""
from dataclasses import dataclass
import math
import torch
from torch import nn


@dataclass
class TileLayout:
    source: torch.Tensor  # [tiles * tile_size], -1 denotes padding
    inverse: torch.Tensor  # packed row -> padded row
    lengths: torch.Tensor
    prefix_tiles: int
    video_spans: tuple
    tile_size: int = 128

    @classmethod
    def build(cls, segments, device="cpu", *, tile_size=128):
        if tile_size not in (64, 128):
            raise ValueError("VSA logical tiles must contain 64 or 128 tokens")
        tile_w = tile_size // 16
        dense, video, cursor = [], [], 0
        for segment in segments:
            if isinstance(segment, int):
                if segment < 0:
                    raise ValueError("Negative dense segment")
                dense.extend(list(range(i, min(i+tile_size, cursor+segment)))
                             for i in range(cursor, cursor+segment, tile_size))
                cursor += segment
            else:
                if len(segment) != 3 or any(type(x) is not int or x <= 0 for x in segment):
                    raise ValueError("Invalid video token grid")
                t, h, w = segment
                tiles = []
                for a in range(0, t, 4):
                    for b in range(0, h, 4):
                        for c in range(0, w, tile_w):
                            tiles.append([cursor+(i*h+j)*w+k for i in range(a, min(a+4, t))
                                          for j in range(b, min(b+4, h)) for k in range(c, min(c+tile_w, w))])
                video.append(tiles)
                cursor += t*h*w
        prefix = len(dense)
        spans = []
        for tiles in video:
            spans.append((len(dense), len(dense)+len(tiles)))
            dense.extend(tiles)
        if not dense:
            raise ValueError("Empty packed sequence")
        lengths = torch.tensor([len(x) for x in dense], dtype=torch.int32, device=device)
        source = torch.tensor([n for tile in dense for n in tile+[-1]*(tile_size-len(tile))], device=device)
        inverse = torch.empty(cursor, dtype=torch.long, device=device)
        valid = source >= 0
        inverse[source[valid]] = torch.arange(source.numel(), device=device)[valid]
        return cls(source, inverse, lengths, prefix, tuple(spans), tile_size)

    def tile(self, x):
        if x.shape[1] != self.inverse.numel():
            raise ValueError("VSA layout differs from packed sequence")
        result = x[:, self.source.clamp_min(0)]
        return result.masked_fill((self.source < 0)[None, :, None, None], 0)

    def pool(self, tiled):
        b, _, h, d = tiled.shape
        sums = tiled.reshape(b, -1, self.tile_size, h, d).sum(2, dtype=torch.float32)
        return (sums / self.lengths[None, :, None, None]).transpose(1, 2)


def _select_video_blocks(scores, layout, sparsity, reference_keep_rate):
    if not 0 <= sparsity <= 1 or not 0 < reference_keep_rate <= 1:
        raise ValueError("Invalid VSA sparsity/keep rate")
    for i, (start, end) in enumerate(layout.video_spans):
        region_sparsity = sparsity if i == len(layout.video_spans)-1 else 1-reference_keep_rate
        if sparsity <= 0:
            region_sparsity = sparsity
        count = max(1, min(math.ceil((1-region_sparsity)*(end-start)), end-start))
        yield scores[..., start:end].topk(count, dim=-1).indices + start


def select_blocks(scores, layout, sparsity=.9, reference_keep_rate=.1):
    mask = torch.zeros_like(scores, dtype=torch.bool)
    for selected in _select_video_blocks(scores, layout, sparsity, reference_keep_rate):
        mask.scatter_(-1, selected, True)
    mask[..., :layout.prefix_tiles] = True
    mask[..., :layout.prefix_tiles, :] = True
    return mask


def select_block_indices(scores, layout, sparsity=.9, reference_keep_rate=.1):
    """Ordered video indices; dense query rows and prefix keys are implicit."""
    parts = [part.sort(dim=-1).values.to(torch.int32) for part in
             _select_video_blocks(scores, layout, sparsity, reference_keep_rate)]
    if not parts:
        return torch.empty((*scores.shape[:-1], 0), device=scores.device, dtype=torch.int32)
    return torch.cat(parts, dim=-1)


class H3VSAProcessor:
    def __init__(self, layout, *, sparsity=.9, packed_output=False, cache_bytes=0, host_scratch=None,
                 reuse_input=False):
        self.layout = layout
        self.sparsity = sparsity
        self.packed_output, self.cache_bytes = packed_output, cache_bytes
        self.host_scratch = host_scratch
        self.reuse_input = reuse_input

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        if attention_mask is not None:
            raise ValueError("H3 packed VSA does not accept an additional attention mask")
        from .long_video import grouped_attention
        return grouped_attention(self, attn, hidden_states, rotary_emb)


def add_compression_gates(model):
    for block in model.transformer_blocks:
        attn = block.attn
        attn.to_gate_compress = nn.Linear(attn.to_q.in_features, attn.inner_dim, bias=False, device="meta")
