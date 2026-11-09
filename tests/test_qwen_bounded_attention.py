import pytest
import torch
from torch.nn import functional as F

from fasth3_pipeline.encoder import _qwen_padding_mask, bounded_qwen_attention


@pytest.mark.parametrize('mask_kind', ['none', 'padding', 'additive'])
def test_bounded_causal_gqa_matches_dense_through_chunk_boundary(mask_kind):
    length = 301
    g = torch.Generator().manual_seed(7)
    q = torch.randn(2, 6, length, 16, generator=g)
    k = torch.randn(2, 2, length, 16, generator=g)
    v = torch.randn(2, 2, length, 16, generator=g)
    causal = torch.ones(length, length, dtype=torch.bool).tril()
    mask = None
    expected_mask = causal
    if mask_kind == 'padding':
        mask = torch.ones(2, length, dtype=torch.bool)
        mask[0, 40:70] = False
        mask[1, -22:] = False
        expected_mask = causal & mask[:, None, None, :]
    if mask_kind == 'additive':
        mask = torch.zeros((2, 1, length, length))
        mask[:, :, :, 75:135] = -1.5
        expected_mask = mask.masked_fill(~causal, -torch.inf)
    expected = F.scaled_dot_product_attention(q, k.repeat_interleave(3, dim=1), v.repeat_interleave(3, dim=1),
                                              attn_mask=expected_mask)
    actual, _ = bounded_qwen_attention(None, q, k, v, mask)
    torch.testing.assert_close(actual, expected.transpose(1, 2), rtol=1e-5, atol=1e-6)


def test_padding_mask_does_not_expand_to_quadratic_size():
    full = torch.ones(1, 16, dtype=torch.bool)
    assert _qwen_padding_mask(attention_mask=full) is None
    full[0, -1] = False
    assert _qwen_padding_mask(attention_mask=full) is full
