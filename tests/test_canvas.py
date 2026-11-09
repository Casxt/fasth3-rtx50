import pytest

from fasth3_pipeline.canvas import released_canvas


@pytest.mark.parametrize('ratio,expected', [('16:9', (768, 1344)), ('9:16', (1344, 768)), ('1:1', (768, 768))])
def test_release_canvas_uses_pinned_rule(ratio, expected):
    height, width, contract = released_canvas(ratio)
    assert (height, width) == expected
    assert contract['canvas_max_pixels'] == 768 * 1344
    assert contract['canvas_multiple'] == 32


@pytest.mark.parametrize('ratio', ['0:1', '1:0', 'bad', '5:1'])
def test_release_canvas_rejects_bad_ratio(ratio):
    with pytest.raises(ValueError):
        released_canvas(ratio)
