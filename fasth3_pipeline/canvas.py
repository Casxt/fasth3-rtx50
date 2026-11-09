"""Resolve the pinned H3 release's canvas rule; never invent a 768p size."""
from fractions import Fraction


def released_canvas(aspect_ratio='16:9'):
    from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import resolve_canvas_size
    try:
        ratio = Fraction(str(aspect_ratio).replace(':', '/'))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError('Aspect ratio must be a positive ratio such as 16:9') from error
    if ratio <= 0:
        raise ValueError('Aspect ratio must be positive')
    defaults = {spec.name: spec.default for spec in MiniMaxH3Ref2VASetupStep().expected_configs}
    short = defaults['canvas_short_edge']
    cap = defaults['canvas_max_pixels']
    # Geometry of the pinned released VAE x DiT spatial patch. Fail closed if
    # the release changes rather than labeling a different preset as this one.
    if (short, cap) != (768, 768 * 1344):
        raise ValueError('Pinned H3 released canvas contract changed')
    height, width = resolve_canvas_size(ratio.numerator, ratio.denominator, 32, short, cap)
    return height, width, {
        'source': 'pinned Diffusers MiniMaxH3Ref2VASetupStep.expected_configs',
        'requested_aspect_ratio': str(aspect_ratio), 'canvas_multiple': 32,
        'canvas_short_edge': short, 'canvas_max_pixels': cap,
        'height': height, 'width': width,
    }
