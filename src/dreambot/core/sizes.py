"""Turning an aspect ratio into pixel dimensions a model can generate.

Midjourney renders roughly a megapixel at any aspect ratio: 1024x1024 for 1:1, 1456x816 for
16:9, and so on. Dreambot uses Midjourney's exact sizes for the ratios it documents and the same
pixel budget for every other ratio, then snaps to the multiple the model requires.
"""

from __future__ import annotations

import math

from dreambot.core.params import PromptError

#: Sizes Midjourney documents for its current default model (V8.2, standard definition).
MIDJOURNEY_SIZES: dict[tuple[int, int], tuple[int, int]] = {
    (1, 1): (1024, 1024),
    (4, 3): (1232, 928),
    (3, 4): (928, 1232),
    (3, 2): (1344, 896),
    (2, 3): (896, 1344),
    (16, 9): (1456, 816),
    (9, 16): (816, 1456),
}

#: Pixel budget for ratios outside the table (the documented non-square sizes average ~1.18 MP).
DEFAULT_PIXELS = 1_180_000
MIN_SIDE = 256
MAX_SIDE = 2560


def snap(value: float, multiple: int) -> int:
    """Round to the nearest multiple (halves up), never below one multiple."""
    return max(multiple, int(value / multiple + 0.5) * multiple)


def dimensions(
    aspect: tuple[int, int] | None,
    *,
    multiple: int = 16,
    scale: float = 1.0,
    native: int = 1024,
) -> tuple[int, int]:
    """Width and height for an aspect ratio.

    ``scale`` shrinks or grows both sides (0.5 for drafts, 2.0 for native HD). ``native`` is the
    model's square training resolution; sizes are scaled relative to 1024 so that a 512px or
    2048px model gets proportionally smaller or larger images.
    """
    ratio = aspect or (1, 1)
    factor = scale * native / 1024
    if ratio in MIDJOURNEY_SIZES:
        width, height = MIDJOURNEY_SIZES[ratio]
        width, height = width * factor, height * factor
    else:
        r = ratio[0] / ratio[1]
        pixels = DEFAULT_PIXELS * factor * factor
        width = math.sqrt(pixels * r)
        height = width / r

    w, h = snap(width, multiple), snap(height, multiple)
    if min(w, h) < MIN_SIDE or max(w, h) > MAX_SIDE:
        # Hold the ratio and fit inside the limits instead of distorting it.
        r = ratio[0] / ratio[1]
        long_side = min(MAX_SIDE, max(w, h))
        short_side = long_side / max(r, 1 / r)
        if short_side < MIN_SIDE:
            limit = MAX_SIDE // MIN_SIDE
            raise PromptError(
                f"`--ar {ratio[0]}:{ratio[1]}` is too extreme to render locally; "
                f"the widest supported ratio is about {limit}:1."
            )
        w, h = (long_side, short_side) if r >= 1 else (short_side, long_side)
        w, h = snap(w, multiple), snap(h, multiple)
    return w, h


def fit_within(width: int, height: int, *, max_pixels: int, multiple: int = 16) -> tuple[int, int]:
    """Scale a size down (never up) so it has at most ``max_pixels``, keeping its ratio."""
    if width * height <= max_pixels:
        return snap(width, multiple), snap(height, multiple)
    shrink = math.sqrt(max_pixels / (width * height))
    return snap(width * shrink, multiple), snap(height * shrink, multiple)


def working_size(width: int, height: int, *, native: int = 1024, multiple: int = 16) -> tuple[int, int]:
    """The size to run an image-to-image style operation at, given the source image's size.

    Sources near the model's pixel budget are used as they are (snapped). Much larger sources
    (a phone photo, an upscaled image) are brought down to the budget; tiny ones are brought up.
    """
    budget = DEFAULT_PIXELS * (native / 1024) ** 2
    pixels = width * height
    if 0.6 * budget <= pixels <= 1.35 * budget:
        return snap(width, multiple), snap(height, multiple)
    factor = math.sqrt(budget / pixels)
    return snap(width * factor, multiple), snap(height * factor, multiple)


def aspect_of(width: int, height: int) -> tuple[int, int]:
    """A human-friendly aspect ratio for a pixel size (1456x816 -> 16:9)."""
    target = width / height
    best: tuple[int, int] | None = None
    best_error = math.inf
    for (w, h), (pixel_w, pixel_h) in MIDJOURNEY_SIZES.items():
        if (pixel_w, pixel_h) == (width, height):
            return w, h
    for denominator in range(1, 33):
        numerator = round(target * denominator)
        if numerator < 1:
            continue
        error = abs(numerator / denominator - target) / target
        # Prefer small numbers: accept the first ratio that is within 1.5%.
        if error < 0.015:
            divisor = math.gcd(numerator, denominator)
            return numerator // divisor, denominator // divisor
        if error < best_error:
            best_error, best = error, (numerator, denominator)
    assert best is not None
    divisor = math.gcd(*best)
    return best[0] // divisor, best[1] // divisor
