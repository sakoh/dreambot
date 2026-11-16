"""Local stand-ins for the Midjourney parameters that live inside Midjourney's own model.

``--chaos``, ``--weird``, ``--exp``, high ``--stylize`` and numeric ``--sref`` codes steer
Midjourney's proprietary network; no open model has those dials. Dreambot reproduces their
*effect* by adding direction to the prompt each image is generated from:

* ``--chaos`` makes the images of one grid differ in framing, light, palette and (at the top
  of the range) medium, instead of differing by seed alone;
* ``--weird`` and ``--exp`` add offbeat or heightened aesthetic direction, scaled by the value;
* ``--stylize`` at 250 and above adds artistic emphasis (below that the prompt is left to the
  model, or to the optional LLM prompt enhancer);
* a style code deterministically selects a recipe (medium, palette, light, texture) from
  Dreambot's own style space. The codes are stable and shareable between Dreambot users, but
  they are not Midjourney's codes and do not reproduce Midjourney's styles.

Everything here is a pure function of its inputs, so the same prompt, parameters and seed
always produce the same directed prompt.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence

FRAMINGS = (
    "extreme close-up",
    "close-up",
    "medium shot",
    "wide shot",
    "sweeping establishing shot",
    "aerial view",
    "low-angle view",
    "view from directly above",
    "over-the-shoulder view",
    "symmetrical composition",
    "off-center composition with negative space",
    "dutch angle",
    "shot through a foreground frame",
    "telephoto compression",
    "fisheye perspective",
    "side profile view",
)
# Light is described by its quality, never by its source: "candlelight" and "softbox lighting"
# put a candle and a softbox in the picture.
LIGHTS = (
    "soft morning light",
    "golden hour light",
    "overcast diffuse light",
    "dramatic backlight",
    "hard midday sun with deep shadows",
    "blue hour twilight",
    "moonlight",
    "neon glow",
    "warm low amber light",
    "volumetric light through haze",
    "soft even studio light",
    "stormy low light",
)
PALETTES = (
    "muted earth tones",
    "vivid saturated colors",
    "monochrome palette",
    "pastel palette",
    "cool blue and teal tones",
    "warm amber and crimson tones",
    "high-contrast black and white",
    "complementary orange and teal",
    "desaturated film tones",
    "jewel tones",
)
MEDIUMS = (
    "35mm film photograph",
    "oil painting",
    "watercolor painting",
    "ink and wash drawing",
    "charcoal sketch",
    "3D render",
    "linocut print",
    "gouache illustration",
    "pencil drawing",
    "vintage poster illustration",
    "paper cut-out collage",
    "pixel art",
)
TEXTURES = (
    "visible film grain",
    "rough paper texture",
    "smooth glossy finish",
    "thick impasto brushwork",
    "halftone dots",
    "soft airbrushed gradients",
    "fine cross-hatching",
    "risograph misregistration",
)
WEIRD_MILD = (
    "slightly surreal",
    "quirky unexpected details",
    "an offbeat mood",
    "subtly strange proportions",
)
WEIRD_STRONG = (
    "surreal",
    "dreamlike logic",
    "uncanny atmosphere",
    "bizarre juxtapositions",
    "impossible geometry",
    "absurdist staging",
    "strange color choices",
)
WEIRD_EXTREME = (
    "hallucinatory imagery",
    "biomorphic distortions",
    "glitch aesthetics",
    "avant-garde experimental composition",
    "otherworldly and unsettling",
    "melting, warped forms",
    "fever-dream symbolism",
)


def _rng(*parts: object) -> random.Random:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _distinct(pool: tuple[str, ...], count: int, rng: random.Random) -> list[str]:
    """``count`` picks that only repeat once the pool is exhausted."""
    picks: list[str] = []
    while len(picks) < count:
        batch = list(pool)
        rng.shuffle(batch)
        picks.extend(batch)
    return picks[:count]


def chaos_directions(chaos: int, seed: int, count: int) -> list[list[str]]:
    """Per-image composition directions for a grid of ``count`` images."""
    if chaos < 10 or count < 1:
        return [[] for _ in range(count)]
    rng = _rng("chaos", seed, chaos // 10)
    layers = [FRAMINGS]
    if chaos >= 30:
        layers.append(LIGHTS)
    if chaos >= 60:
        layers.append(PALETTES)
    if chaos >= 85:
        layers.append(MEDIUMS)
    columns = [_distinct(pool, count, rng) for pool in layers]
    return [[column[i] for column in columns] for i in range(count)]


def weird_directions(weird: int, seed: int) -> list[str]:
    if weird <= 0:
        return []
    rng = _rng("weird", seed, weird)
    if weird < 250:
        return _distinct(WEIRD_MILD, 1, rng)
    if weird < 1000:
        return _distinct(WEIRD_STRONG, 2, rng)
    return _distinct(WEIRD_STRONG, 1, rng) + _distinct(WEIRD_EXTREME, 2 if weird < 2000 else 3, rng)


def exp_directions(exp: int) -> list[str]:
    if exp <= 0:
        return []
    if exp < 25:
        return ["finely detailed"]
    if exp < 50:
        return ["intricate detail", "dynamic composition"]
    return ["intricate detail", "dynamic composition", "dramatic depth", "rich tonal range"]


def stylize_directions(stylize: int) -> list[str]:
    """Artistic emphasis for high stylize values. Below 250 the prompt is left alone."""
    if stylize < 250:
        return []
    if stylize < 750:
        return ["artful composition", "expressive lighting"]
    return ["highly stylized", "artful composition", "expressive lighting", "painterly atmosphere"]


def random_style_code(rng: random.Random | None = None) -> str:
    return str((rng or random.SystemRandom()).randint(1, 4294967295))


def style_recipe(code: str, weight: int = 100) -> list[str]:
    """The style a numeric code stands for. ``weight`` is ``--sw`` (0-1000)."""
    if weight <= 0:
        return []
    rng = _rng("sref", int(code))
    medium = rng.choice(MEDIUMS)
    palette = rng.choice(PALETTES)
    light = rng.choice(LIGHTS)
    texture = rng.choice(TEXTURES)
    if weight < 50:
        return [f"with a hint of {medium} style"]
    recipe = [f"in the style of a {medium}", palette, light]
    if weight >= 200:
        recipe.append(texture)
    if weight >= 500:
        recipe.insert(0, "strongly stylized")
    return recipe


def direct(
    text: str,
    *,
    seed: int,
    index: int,
    count: int,
    chaos: int = 0,
    weird: int = 0,
    exp: int = 0,
    stylize: int = 100,
    raw: bool = False,
    style_codes: list[tuple[str, int]] | None = None,
    style: Sequence[str] = (),
) -> str:
    """The prompt for image ``index`` (0-based) of a grid, with all emulated direction applied.

    ``seed`` is the job's base seed, so the direction is identical on a re-run. ``raw`` switches
    off the stylize emphasis but keeps chaos, weird, exp and the style: those are things the
    person explicitly asked for. ``style`` is the wording of a blended style fingerprint
    (:mod:`dreambot.core.fingerprint`); ``style_codes`` are the original recipes (``--sv 1``).
    """
    directions: list[str] = list(style)
    for code, weight in style_codes or []:
        directions.extend(style_recipe(code, weight))
    directions.extend(chaos_directions(chaos, seed, count)[index] if count else [])
    directions.extend(weird_directions(weird, seed))
    directions.extend(exp_directions(exp))
    if not raw:
        directions.extend(stylize_directions(stylize))
    seen: set[str] = set()
    unique = [d for d in directions if not (d in seen or seen.add(d))]
    if not unique:
        return text
    base = text.rstrip().rstrip(",.")
    return f"{base}, {', '.join(unique)}" if base else ", ".join(unique)
