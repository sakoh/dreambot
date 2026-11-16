"""Style fingerprints: the one form every kind of style is reduced to.

Midjourney lets a prompt take its look from several places at once — reference images, style
codes, boards of images — and mixes them by weight inside its own model. Open models have no
such input, so Dreambot gives every source of style the same description and does the mixing
on that:

* **words** — the medium, how it is rendered, the palette, the light, the surface and the mood.
  For an image they come from the local vision model, asked about style and never content;
  for a style code they are decoded from the number.
* **numbers** — the dominant colours and the tonal balance, measured from the pixels.

A fingerprint never says what a picture shows, which is what lets it be applied to a prompt
about something else. The pictures themselves travel alongside it (see
:class:`dreambot.core.planner.StyleSource`) for models that can be shown a reference image.

Everything here is pure: the same inputs give the same fingerprint, the same blend and the
same code. Running a model to get the words is the service's job.
"""

from __future__ import annotations

import colorsys
import hashlib
import math
import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

#: The worded parts of a fingerprint, in the order they are asked for and written out.
WORD_FIELDS = ("medium", "rendering", "palette", "light", "texture", "mood")
MAX_CODE = 4294967295
#: From this share of the total weight up, one source sets the medium on its own.
DOMINANT_SHARE = 0.6
#: The share from which a style that is not the heaviest adds its surface and mood to a blend.
ACCENT_SHARE = 0.3

RGB = tuple[int, int, int]


@dataclass(frozen=True)
class Swatch:
    rgb: RGB
    #: Fraction of the picture this colour covers, 0..1.
    share: float

    @property
    def hex(self) -> str:
        return "#{:02x}{:02x}{:02x}".format(*self.rgb)


@dataclass(frozen=True)
class Tone:
    """The tonal balance of a picture, each 0..1 except ``warmth`` (-1 cool .. 1 warm)."""

    brightness: float
    contrast: float
    saturation: float
    warmth: float


@dataclass(frozen=True)
class StyleFingerprint:
    medium: str = ""
    rendering: str = ""
    palette: str = ""
    light: str = ""
    texture: str = ""
    mood: str = ""
    swatches: tuple[Swatch, ...] = ()
    tone: Tone | None = None
    #: A short display name ("Rust Gouache"). Never part of a prompt.
    name: str = ""

    @property
    def worded(self) -> bool:
        """Whether a model (or a style code) has said what kind of image this is."""
        return bool(self.medium)

    @property
    def empty(self) -> bool:
        return not self.worded and not self.swatches

    def words(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in WORD_FIELDS if getattr(self, name)}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StyleFingerprint:
        tone = data.get("tone")
        return cls(
            **{name: str(data.get(name) or "") for name in WORD_FIELDS},
            swatches=tuple(
                Swatch(tuple(item["rgb"]), float(item["share"]))  # type: ignore[arg-type]
                for item in data.get("swatches") or ()
            ),
            tone=Tone(**tone) if tone else None,
            name=str(data.get("name") or ""),
        )


# --------------------------------------------------------------------------------------
# Numbers: measured from the pixels
# --------------------------------------------------------------------------------------


def image_key(path: Path | str) -> str:
    """A key for an image's contents, so that a picture is only ever analysed once."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tone_of(image: Image.Image) -> Tone:
    grey = ImageStat.Stat(image.convert("L"))
    red, _green, blue = ImageStat.Stat(image).mean
    saturation = ImageStat.Stat(image.convert("HSV")).mean[1]
    return Tone(
        brightness=round(grey.mean[0] / 255, 3),
        # A standard deviation of 80 is about as contrasty as a photograph gets.
        contrast=round(min(1.0, grey.stddev[0] / 80), 3),
        saturation=round(saturation / 255, 3),
        warmth=round(max(-1.0, min(1.0, (red - blue) / 128)), 3),
    )


def measure(path: Path | str, colours: int = 5) -> StyleFingerprint:
    """The part of an image's fingerprint that needs no model: its colours and tonal balance."""
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    image.thumbnail((96, 96))
    quantized = image.quantize(colors=colours, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette() or []
    counted = sorted(quantized.getcolors() or [], reverse=True)
    total = sum(count for count, _ in counted) or 1
    swatches = tuple(
        Swatch((palette[index * 3], palette[index * 3 + 1], palette[index * 3 + 2]), round(count / total, 3))
        for count, index in counted
    )
    return StyleFingerprint(swatches=_merge_swatches(swatches), tone=_tone_of(image))


def _distance(a: RGB, b: RGB) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b, strict=True)) ** 0.5


def _merge_swatches(swatches: Iterable[Swatch], limit: int = 5, near: float = 42.0) -> tuple[Swatch, ...]:
    """Fold near-identical colours together and keep the ``limit`` largest, shares summing to 1."""
    merged: list[list[float]] = []  # [r, g, b, share], colour kept as a running weighted mean
    for swatch in sorted(swatches, key=lambda s: -s.share):
        if swatch.share <= 0:
            continue
        for entry in merged:
            if _distance((round(entry[0]), round(entry[1]), round(entry[2])), swatch.rgb) < near:
                total = entry[3] + swatch.share
                for channel in range(3):
                    entry[channel] = (entry[channel] * entry[3] + swatch.rgb[channel] * swatch.share) / total
                entry[3] = total
                break
        else:
            merged.append([*map(float, swatch.rgb), swatch.share])
    merged.sort(key=lambda entry: -entry[3])
    kept = merged[:limit]
    total = sum(entry[3] for entry in kept) or 1.0
    return tuple(
        Swatch((round(entry[0]), round(entry[1]), round(entry[2])), round(entry[3] / total, 3))
        for entry in kept
    )


def _tone_from_swatches(swatches: Sequence[Swatch]) -> Tone:
    """The tonal balance a palette implies, for styles that exist only as a recipe."""
    total = sum(s.share for s in swatches) or 1.0
    lumas = [(0.299 * s.rgb[0] + 0.587 * s.rgb[1] + 0.114 * s.rgb[2]) / 255 for s in swatches]
    mean = sum(luma * s.share for luma, s in zip(lumas, swatches, strict=True)) / total
    spread = (
        sum(s.share * (luma - mean) ** 2 for luma, s in zip(lumas, swatches, strict=True)) / total
    ) ** 0.5
    saturation = sum(colorsys.rgb_to_hsv(*(c / 255 for c in s.rgb))[1] * s.share for s in swatches) / total
    warmth = sum((s.rgb[0] - s.rgb[2]) / 128 * s.share for s in swatches) / total
    return Tone(
        brightness=round(mean, 3),
        contrast=round(min(1.0, spread / 0.31), 3),
        saturation=round(saturation, 3),
        warmth=round(max(-1.0, min(1.0, warmth)), 3),
    )


#: Hue names around the colour wheel: (upper bound in degrees, vivid, dark, pale, greyed).
_HUES = (
    (10, "red", "maroon", "blush pink", "dusty rose"),
    (40, "orange", "brown", "peach", "tan"),
    (52, "gold", "ochre", "cream", "khaki"),
    (66, "yellow", "olive", "cream", "khaki"),
    (95, "lime green", "olive", "pale green", "moss green"),
    (160, "green", "forest green", "mint", "sage"),
    (182, "teal", "deep teal", "pale aqua", "dusty teal"),
    (205, "cyan", "deep teal", "pale aqua", "slate blue"),
    (250, "blue", "navy", "powder blue", "slate blue"),
    (285, "violet", "indigo", "lavender", "dusty violet"),
    (330, "magenta", "plum", "lilac", "mauve"),
    (346, "pink", "burgundy", "pale pink", "dusty rose"),
    (361, "red", "maroon", "blush pink", "dusty rose"),
)
GREYS = frozenset({"black", "charcoal", "grey", "pale grey", "white"})


def colour_name(rgb: RGB) -> str:
    """An everyday name for a colour, the kind a person would use in a prompt."""
    hue, saturation, value = colorsys.rgb_to_hsv(*(channel / 255 for channel in rgb))
    if saturation < 0.12 or value < 0.1:
        for bound, name in ((0.14, "black"), (0.36, "charcoal"), (0.64, "grey"), (0.88, "pale grey")):
            if value < bound:
                return name
        return "white"
    vivid, dark, pale, greyed = next(names[1:] for names in _HUES if hue * 360 < names[0])
    if dark == "brown" and saturation >= 0.6 and value < 0.75:
        return "rust"  # a deep, saturated orange: darker than orange, redder than brown
    if value < 0.5:
        return dark
    if saturation < 0.45:
        if value < 0.82:
            return greyed
        return "cream" if pale == "peach" and saturation < 0.3 else pale
    return vivid


def _listing(items: Sequence[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def palette_words(swatches: Sequence[Swatch], tone: Tone | None = None, limit: int = 3) -> str:
    """A palette in words: "muted palette of rust, cream and deep teal"."""
    names: list[str] = []
    for swatch in swatches:
        name = colour_name(swatch.rgb)
        if name not in names and swatch.share >= 0.06:
            names.append(name)
    if not names:
        return ""
    if all(name in GREYS for name in names):
        return "monochrome black and white palette"
    # Lead with the colours, not the neutrals a picture always has some of.
    coloured = [name for name in names if name not in GREYS]
    chosen = (coloured + [name for name in names if name in GREYS])[:limit]
    quality = ""
    if tone is not None:
        if tone.saturation < 0.22:
            quality = "muted "
        elif tone.saturation > 0.66:
            quality = "vivid "
        elif tone.brightness > 0.7 and tone.saturation < 0.4:
            quality = "pastel "
    return f"{quality}palette of {_listing(chosen)}"


def tone_words(tone: Tone | None) -> list[str]:
    """What the tonal balance adds that a palette does not say."""
    if tone is None:
        return []
    words: list[str] = []
    if tone.contrast > 0.72:
        words.append("high contrast")
    elif tone.contrast < 0.3:
        words.append("soft low contrast")
    if tone.brightness < 0.28:
        words.append("dark and low-key")
    elif tone.brightness > 0.74:
        words.append("bright and airy")
    return words


# --------------------------------------------------------------------------------------
# Words: asked of a vision model
# --------------------------------------------------------------------------------------

#: Sent with an image. The answer comes back as numbered lines, which is the form the engine's
#: describe call already checks and returns.
#:
#: Only the medium is given examples. Tried on Qwen2.5-VL 7B: with an example for every line the
#: model answers with the examples ("crisp inked outlines" for a soft digital painting); with none
#: it writes sentences ("digital artwork created using 3D modeling and rendering software"). A
#: list of art forms works as a vocabulary to choose from; the rest it has to look for.
EXTRACT_INSTRUCTION = (
    "You are cataloguing the visual style of this image so that an artist can make a picture of something "
    "else entirely in the same style. Describe only how the image is made, never what it shows: no "
    "subject, person, animal, object, place, season or written word.\n"
    "\n"
    "Answer with exactly six numbered lines and nothing else. Every line starts with its label and a "
    "colon, followed by one phrase of two to six words. No sentences, and look closely: each answer must "
    "be true of this particular image.\n"
    "1. medium: the art form as a gallery label would name it, such as oil painting, watercolor, pencil "
    "drawing, manga illustration, comic book illustration, digital painting, 3D render, 35mm film "
    "photograph, black and white photograph, studio photograph, linocut print, collage\n"
    "2. rendering: the technique, how edges, lines, strokes or focus are handled\n"
    "3. palette: its actual colours\n"
    "4. light: the direction and quality of its light\n"
    "5. texture: its surface and grain\n"
    "6. mood: the feeling it gives"
)

_NUMBER = re.compile(r"^\W*\d{1,2}\s*[.):\-]\s*")
_LABEL = re.compile(
    r"^\W*(?P<label>medium|rendering|palette|light(?:ing)?|texture|mood)\W*?\s*[:\-–—=]\s*(?P<value>.*)$",
    re.IGNORECASE,
)
#: An answer that only repeats the question.
_ECHO = re.compile(
    r"gallery label|the art form|the technique|how edges|its actual colou?rs|quality of its light|"
    r"its surface and grain|the feeling it gives|two to six words|<|>",
    re.IGNORECASE,
)
MAX_PHRASE_WORDS = 8
MAX_MEDIUM_WORDS = 6


def _phrase(value: str, limit: int) -> str:
    """A model's answer cut down to a phrase: no sentence, no trailing list, no stray punctuation."""
    value = re.sub(r"\s+", " ", value).strip().strip("\"'`*.;:,“”‘’ ")
    value = re.split(r"(?<=\w)\.\s", value)[0]  # a second sentence
    chunks = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    while len(chunks) > 1 and sum(len(chunk.split()) for chunk in chunks) > limit:
        chunks.pop()  # a list that runs on loses its tail
    value = ", ".join(chunks)
    words = value.split()
    if not words or len(words) > limit + 4:
        return ""
    if len(words) > limit:
        value = " ".join(words[:limit])
    first = value.split()[0]
    if first[:1].isupper() and first[1:].islower():
        value = value[0].lower() + value[1:]  # "Oil painting", but not "CGI render" or "3D render"
    return value


def parse_words(answer: str | Iterable[str]) -> dict[str, str]:
    """The six phrases of a model's answer, by field. Missing or unusable ones are left out.

    Models drop the labels about as often as they keep them ("digital painting:" for the first
    line, or just "digital painting"), so a line without a label is read by its position —
    which is only safe when all six lines are there.
    """
    raw = answer.splitlines() if isinstance(answer, str) else list(answer)
    lines = [line.strip() for line in raw if line.strip()]
    positional = len(lines) == len(WORD_FIELDS)
    found: dict[str, str] = {}
    for position, line in enumerate(lines):
        text = _NUMBER.sub("", line)
        if match := _LABEL.match(text):
            label = match.group("label").lower()
            label = "light" if label.startswith("light") else label
            value = match.group("value")
        elif positional:
            label = WORD_FIELDS[position]
            # "studio photograph: highly detailed, sharp focus" — the answer is what comes first.
            value = text.split(":")[0]
        else:
            continue
        if label in found or _ECHO.search(value):
            continue
        if any(ord(char) > 0x024F and char.isalpha() for char in value):
            continue  # a model slipping out of English
        limit = MAX_MEDIUM_WORDS if label == "medium" else MAX_PHRASE_WORDS
        if label == "medium":
            value = value.split(",")[0]  # "digital illustration, digitally drawn"
        if phrase := _phrase(value, limit):
            found[label] = phrase
    return found


def with_words(print_: StyleFingerprint, words: dict[str, str]) -> StyleFingerprint:
    """``print_`` with a model's words filled in, and a name if it has none."""
    known = {name: value for name, value in words.items() if name in WORD_FIELDS and value}
    updated = replace(print_, **known)
    return updated if updated.name else replace(updated, name=name_of(updated))


def name_of(print_: StyleFingerprint) -> str:
    """A short display name: the leading colour and the medium ("Rust Gouache Illustration")."""
    colour = ""
    for swatch in print_.swatches:
        candidate = colour_name(swatch.rgb)
        if candidate not in ("grey", "pale grey", "white", "black", "charcoal"):
            colour = candidate
            break
    words = [*colour.split(), *print_.medium.split()[:3]]
    # Not str.title(): that turns "35mm" into "35Mm".
    return " ".join(word[:1].upper() + word[1:] for word in words) or "Untitled Style"


# --------------------------------------------------------------------------------------
# Style codes: a number that decodes to a fingerprint
# --------------------------------------------------------------------------------------

#: Media in families, so that a code never pairs film grain with brushwork. Every family has
#: the same number of media, renderings and textures, which keeps the code arithmetic uniform.
_FAMILIES: tuple[tuple[str, tuple[tuple[str, str], ...], tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "photograph",
        (
            ("35mm film photograph", "Film"),
            ("medium format photograph", "Medium Format"),
            ("instant film photograph", "Instant"),
            ("cinematic film still", "Cinema"),
            ("documentary photograph", "Documentary"),
            ("studio photograph", "Studio"),
            ("long-exposure photograph", "Long Exposure"),
            ("pinhole camera photograph", "Pinhole"),
        ),
        (
            "shallow depth of field",
            "deep focus throughout",
            "soft focus with gentle halation",
            "wide-angle perspective",
            "telephoto compression",
            "slight motion blur",
        ),
        (
            "visible film grain",
            "fine clean grain",
            "light leaks and dust specks",
            "smooth glossy finish",
            "soft bloom on the highlights",
        ),
    ),
    (
        "painting",
        (
            ("oil painting", "Oil"),
            ("watercolor painting", "Watercolor"),
            ("gouache illustration", "Gouache"),
            ("acrylic painting", "Acrylic"),
            ("ink wash painting", "Ink Wash"),
            ("soft pastel drawing", "Chalk Pastel"),
            ("fresco painting", "Fresco"),
            ("egg tempera painting", "Tempera"),
        ),
        (
            "loose expressive brushwork",
            "tight detailed brushwork",
            "flat simplified shapes",
            "soft blended edges",
            "bold palette-knife strokes",
            "thin translucent washes",
        ),
        (
            "thick impasto",
            "rough paper texture",
            "visible canvas weave",
            "dry-brush texture",
            "smooth matte surface",
        ),
    ),
    (
        "drawing",
        (
            ("pencil drawing", "Pencil"),
            ("charcoal sketch", "Charcoal"),
            ("ink line drawing", "Ink"),
            ("colored pencil illustration", "Colored Pencil"),
            ("comic book illustration", "Comic"),
            ("anime illustration", "Anime"),
            ("children's book illustration", "Storybook"),
            ("technical pen illustration", "Technical Pen"),
        ),
        (
            "crisp clean linework",
            "loose sketchy lines",
            "fine cross-hatching",
            "bold outlines with flat fills",
            "delicate stippling",
            "confident gestural marks",
        ),
        (
            "toothy paper grain",
            "smudged soft shading",
            "clean smooth paper",
            "screentone dots",
            "slightly yellowed paper",
        ),
    ),
    (
        "print",
        (
            ("linocut print", "Linocut"),
            ("woodblock print", "Woodblock"),
            ("risograph print", "Risograph"),
            ("screen print", "Screen Print"),
            ("vintage travel poster", "Poster"),
            ("etching", "Etching"),
            ("paper cut-out collage", "Collage"),
            ("mid-century editorial illustration", "Editorial"),
        ),
        (
            "bold simplified shapes",
            "limited flat colors",
            "carved angular lines",
            "layered overprinted shapes",
            "strong graphic silhouettes",
            "fine engraved hatching",
        ),
        (
            "halftone dots",
            "slight misregistration",
            "uneven ink coverage",
            "fibrous paper texture",
            "worn letterpress finish",
        ),
    ),
    (
        "digital",
        (
            ("3D render", "3D"),
            ("digital painting", "Digital Paint"),
            ("pixel art", "Pixel"),
            ("low-poly 3D illustration", "Low Poly"),
            ("vector illustration", "Vector"),
            ("claymation still", "Clay"),
            ("isometric illustration", "Isometric"),
            ("matte painting", "Matte"),
        ),
        (
            "smooth global illumination",
            "crisp hard edges",
            "soft airbrushed gradients",
            "stylized simplified forms",
            "intricate fine detail",
            "chunky exaggerated proportions",
        ),
        (
            "glossy smooth surfaces",
            "soft subsurface glow",
            "clean flat surfaces",
            "subtle noise grain",
            "handmade tactile surfaces",
        ),
    ),
)

#: (palette in words, short name, colours from most to least used).
_PALETTES: tuple[tuple[str, str, tuple[RGB, ...]], ...] = (
    (
        "muted earth tones",
        "Earthen",
        ((121, 85, 61), (189, 154, 122), (96, 108, 56), (221, 204, 170), (60, 47, 40)),
    ),
    (
        "vivid saturated colors",
        "Vivid",
        ((230, 57, 70), (255, 183, 3), (33, 158, 188), (131, 56, 236), (6, 214, 160)),
    ),
    (
        "soft pastel colors",
        "Pastel",
        ((255, 209, 220), (189, 224, 254), (205, 234, 192), (255, 236, 179), (216, 191, 216)),
    ),
    (
        "cool blues and teals",
        "Tidal",
        ((13, 59, 102), (38, 120, 140), (86, 180, 190), (190, 226, 230), (244, 247, 247)),
    ),
    (
        "warm amber and crimson",
        "Ember",
        ((120, 20, 25), (193, 39, 45), (244, 140, 6), (255, 200, 87), (51, 24, 20)),
    ),
    (
        "high-contrast black and white",
        "Noir",
        ((10, 10, 10), (250, 250, 250), (60, 60, 60), (215, 215, 215), (140, 140, 140)),
    ),
    (
        "teal and orange",
        "Tangerine Teal",
        ((0, 95, 115), (238, 155, 0), (10, 147, 150), (202, 103, 2), (233, 216, 166)),
    ),
    (
        "desaturated film tones",
        "Faded",
        ((92, 98, 92), (140, 138, 125), (178, 170, 150), (201, 196, 182), (64, 66, 70)),
    ),
    (
        "rich jewel tones",
        "Jewel",
        ((17, 75, 95), (123, 31, 162), (0, 121, 107), (173, 20, 87), (245, 197, 24)),
    ),
    (
        "neon pink and electric blue",
        "Neon",
        ((20, 10, 40), (255, 0, 128), (0, 200, 255), (120, 0, 255), (255, 240, 0)),
    ),
    (
        "sepia and cream",
        "Sepia",
        ((132, 94, 62), (229, 208, 174), (72, 49, 33), (184, 148, 108), (246, 238, 220)),
    ),
    (
        "forest greens and moss",
        "Moss",
        ((27, 67, 50), (45, 106, 79), (116, 148, 89), (183, 196, 142), (60, 50, 35)),
    ),
    (
        "dusty rose and sage",
        "Rosewood",
        ((181, 131, 141), (183, 196, 174), (229, 152, 155), (109, 104, 117), (255, 236, 229)),
    ),
    (
        "midnight blue and gold",
        "Midnight",
        ((10, 17, 40), (0, 31, 84), (254, 203, 0), (3, 64, 120), (255, 236, 160)),
    ),
    (
        "primary red, yellow and blue",
        "Primary",
        ((214, 40, 40), (247, 201, 0), (0, 70, 160), (250, 250, 245), (20, 20, 20)),
    ),
    (
        "lavender and mint",
        "Lilac",
        ((181, 160, 220), (170, 230, 205), (120, 90, 180), (240, 235, 250), (70, 60, 100)),
    ),
    (
        "rust and turquoise",
        "Rust",
        ((158, 61, 34), (52, 152, 152), (204, 102, 51), (120, 200, 190), (240, 225, 200)),
    ),
    (
        "monochrome indigo",
        "Indigo",
        ((15, 20, 60), (40, 50, 120), (80, 95, 170), (150, 165, 215), (225, 230, 245)),
    ),
    (
        "bleached sand and sky",
        "Shore",
        ((236, 223, 200), (160, 200, 220), (214, 188, 150), (100, 150, 190), (250, 248, 240)),
    ),
    (
        "acid green and violet",
        "Acid",
        ((40, 10, 60), (170, 255, 0), (110, 20, 160), (100, 220, 40), (240, 255, 200)),
    ),
)
#: Light is named by its quality, never its source: "candlelight" puts a candle in the picture.
_LIGHTS = (
    "soft morning light",
    "golden hour light",
    "overcast diffuse light",
    "dramatic backlight",
    "hard midday sun with deep shadows",
    "blue hour twilight",
    "cool moonlit glow",
    "neon glow",
    "warm low amber light",
    "volumetric light through haze",
    "soft even studio light",
    "stormy low light",
    "dappled light",
    "a single hard spotlight",
)
_MOODS = (
    "quiet and contemplative",
    "playful and whimsical",
    "tense and dramatic",
    "dreamy and nostalgic",
    "bold and energetic",
    "eerie and mysterious",
    "serene and airy",
    "gritty and raw",
    "elegant and refined",
    "wistful and melancholy",
    "joyful and bright",
    "ominous and brooding",
    "cozy and intimate",
    "surreal and uncanny",
    "epic and grand",
    "minimal and calm",
)
_SWATCH_SHARES = (0.34, 0.26, 0.18, 0.13, 0.09)

#: Sizes of each part of a code, least significant first.
_RADIX = (
    len(_MOODS),
    len(_LIGHTS),
    len(_PALETTES),
    len(_FAMILIES[0][3]),
    len(_FAMILIES[0][2]),
    len(_FAMILIES[0][1]),
    len(_FAMILIES),
)
STYLE_COUNT = math.prod(_RADIX)

# Codes are scrambled so that neighbouring numbers are unrelated styles, by a mapping that can
# be run backwards: that is what lets the explorer turn "anime, pastel" into codes directly.
_MASK = 0xFFFFFFFF
_MULTIPLIER = 0x9E3779B1
_INVERSE = pow(_MULTIPLIER, -1, 1 << 32)
_XOR = 0x5BD1E995


def _digits(code: int | str) -> tuple[int, ...]:
    """(mood, light, palette, texture, rendering, medium, family) for a code."""
    index = (((int(code) ^ _XOR) * _MULTIPLIER) & _MASK) % STYLE_COUNT
    digits = []
    for size in _RADIX:
        digits.append(index % size)
        index //= size
    return tuple(digits)


def code_for(digits: Sequence[int], rng: random.Random | None = None) -> str:
    """A code that decodes to ``digits``. Many codes share a style; ``rng`` picks which one."""
    index = 0
    for digit, size in zip(reversed(digits), reversed(_RADIX), strict=True):
        if not 0 <= digit < size:
            raise ValueError(f"style digit {digit} is out of range")
        index = index * size + digit
    rng = rng or random.Random(index)
    copies = ((_MASK + 1) - index + STYLE_COUNT - 1) // STYLE_COUNT
    while True:
        scrambled = index + rng.randrange(copies) * STYLE_COUNT
        code = ((scrambled * _INVERSE) & _MASK) ^ _XOR
        if code:
            return str(code)


def random_code(rng: random.Random | None = None) -> str:
    return str((rng or random.SystemRandom()).randint(1, MAX_CODE))


def from_code(code: int | str) -> StyleFingerprint:
    """The style a numeric code stands for (style version 2)."""
    mood, light, palette, texture, rendering, medium, family = _digits(code)
    _family, media, renderings, textures = _FAMILIES[family]
    words, short, colours = _PALETTES[palette]
    swatches = tuple(Swatch(rgb, share) for rgb, share in zip(colours, _SWATCH_SHARES, strict=True))
    return StyleFingerprint(
        medium=media[medium][0],
        rendering=renderings[rendering],
        palette=words,
        light=_LIGHTS[light],
        texture=textures[texture],
        mood=_MOODS[mood],
        swatches=swatches,
        tone=_tone_from_swatches(swatches),
        name=f"{short} {media[medium][1]}",
    )


def similar_codes(code: int | str, count: int = 6) -> list[str]:
    """Codes one step away from ``code``: the same style with one part changed."""
    digits = list(_digits(code))
    rng = random.Random(int(code))
    # Change the parts that matter least to the look first, so that neighbours stay recognisable.
    positions = [0, 1, 3, 4, 2, 5]
    found: list[str] = []
    for turn in range(count):
        position = positions[turn % len(positions)]
        changed = list(digits)
        changed[position] = (digits[position] + 1 + rng.randrange(_RADIX[position] - 1)) % _RADIX[position]
        candidate = code_for(changed, rng)
        if candidate not in found and candidate != str(code):
            found.append(candidate)
    return found


def _options(term: str) -> list[dict[int, set[int]]]:
    """Every way one search word can be read: as a mood, a light, a palette, a family or a medium."""
    options: list[dict[int, set[int]]] = []
    for position, texts in (
        (0, _MOODS),
        (1, _LIGHTS),
        (2, [f"{words} {short}" for words, short, _ in _PALETTES]),
    ):
        hits = {index for index, text in enumerate(texts) if term in text.lower()}
        if hits:
            options.append({position: hits})
    for family, (name, media, renderings, textures) in enumerate(_FAMILIES):
        if term in name:
            options.append({6: {family}})
        for position, texts in (
            (5, [f"{phrase} {short}" for phrase, short in media]),
            (4, renderings),
            (3, textures),
        ):
            hits = {index for index, text in enumerate(texts) if term in text.lower()}
            if hits:
                options.append({6: {family}, position: hits})
    return options


def search_codes(query: str, count: int = 24, rng: random.Random | None = None) -> list[str]:
    """Codes whose style matches every word of ``query`` ("anime pastel", "film grain")."""
    terms = [term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) > 1]
    readings = [_options(term) for term in terms]
    if not terms or not all(readings):
        return []
    rng = rng or random.Random(" ".join(terms))
    found: list[str] = []
    for _ in range(count * 20):
        allowed: dict[int, set[int]] = {}
        for options in readings:
            for position, choices in rng.choice(options).items():
                allowed[position] = allowed[position] & choices if position in allowed else set(choices)
        if not all(allowed.values()):
            continue  # two words read in ways that cannot both hold; try other readings
        digits = [
            rng.choice(sorted(allowed[position])) if position in allowed else rng.randrange(size)
            for position, size in enumerate(_RADIX)
        ]
        candidate = code_for(digits, rng)
        if candidate not in found:
            found.append(candidate)
        if len(found) >= count:
            break
    return found


# --------------------------------------------------------------------------------------
# Applying a fingerprint to a prompt
# --------------------------------------------------------------------------------------

_UNCOUNTABLE = re.compile(r"(?:\bart|\banime|photography|animation|collage|claymation|s)$", re.IGNORECASE)


def _with_article(medium: str) -> str:
    if _UNCOUNTABLE.search(medium) and not medium.lower().endswith(("ss", "us")):
        return medium
    first = medium.lstrip()[:1].lower()
    # "a 35mm photograph", "a 3D render", "an 8-bit sprite", "an oil painting".
    return f"{'an' if first in 'aeio8' else 'a'} {medium}"


_PALETTE_WORD = re.compile(r"palette|tone|colou?r|hue|monochrome|black and white", re.IGNORECASE)


def _palette_of(print_: StyleFingerprint) -> str:
    """The palette as a phrase that reads as a palette on its own ("muted teal and rust palette")."""
    palette = print_.palette or palette_words(print_.swatches, print_.tone)
    return palette if not palette or _PALETTE_WORD.search(palette) else f"{palette} palette"


def _hint(palette: str) -> str:
    article = "a " if re.search(r"palette$|^(?:\w+ )?palette of", palette, re.IGNORECASE) else ""
    return f"with a hint of {article}{palette}"


def phrases(print_: StyleFingerprint | None, weight: int = 100) -> list[str]:
    """The words that give a prompt this style. ``weight`` is 0-1000, as ``--sw``.

    Under 50 it is a hint; from 50 the medium, rendering, palette and light; from 200 the
    surface and the mood as well; from 500 it is insisted on.
    """
    if print_ is None or print_.empty or weight <= 0:
        return []
    palette = _palette_of(print_)
    if weight < 50:
        if print_.medium:
            return [f"with a hint of {print_.medium} style"]
        return [_hint(palette)] if palette else []
    wanted: list[str] = []
    if print_.medium:
        wanted.append(f"in the style of {_with_article(print_.medium)}")
    wanted += [part for part in (print_.rendering, palette, print_.light) if part]
    if not print_.worded:
        wanted += tone_words(print_.tone)
    if weight >= 200:
        wanted += [part for part in (print_.texture, print_.mood) if part]
    if weight >= 500:
        wanted.insert(0, "strongly stylized")
    seen: set[str] = set()
    return [part for part in wanted if not (part.lower() in seen or seen.add(part.lower()))]


def accents(print_: StyleFingerprint | None, weight: int = 100) -> list[str]:
    """What a fingerprint adds to a style that already has its medium: colour, light and mood."""
    if print_ is None or print_.empty or weight <= 0:
        return []
    palette = _palette_of(print_)
    if weight < 50:
        return [_hint(palette)] if palette else []
    wanted = [part for part in (palette, print_.light) if part]
    if weight >= 200:
        wanted += [part for part in (print_.mood, print_.texture) if part]
    return wanted


def summary(print_: StyleFingerprint | None) -> str:
    """One line a person can read to see what a style is."""
    if print_ is None or print_.empty:
        return ""
    palette = print_.palette or palette_words(print_.swatches, print_.tone)
    return " · ".join(
        part
        for part in (print_.medium, print_.rendering, palette, print_.light, print_.texture, print_.mood)
        if part
    )


# --------------------------------------------------------------------------------------
# Blending
# --------------------------------------------------------------------------------------

Part = tuple[StyleFingerprint, float]


def shares(parts: Sequence[Part]) -> list[Part]:
    """``parts`` with positive weight, heaviest first, weights scaled to sum to 1."""
    kept = [(print_, weight) for print_, weight in parts if weight > 0 and not print_.empty]
    total = sum(weight for _, weight in kept)
    if not kept or total <= 0:
        return []
    ordered = sorted(enumerate(kept), key=lambda item: (-item[1][1], item[0]))
    return [(print_, weight / total) for _, (print_, weight) in ordered]


def _distinct(values: Iterable[str], limit: int) -> list[str]:
    seen: set[str] = set()
    kept: list[str] = []
    for value in values:
        if value and value.lower() not in seen:
            seen.add(value.lower())
            kept.append(value)
    return kept[:limit]


def blend(parts: Sequence[Part]) -> StyleFingerprint | None:
    """Several styles as one, by weight.

    The colours and the tonal balance are a true weighted mix. Words cannot be averaged, so
    the heaviest style sets the structure — its medium, rendering and light — and a style
    holding at least ``ACCENT_SHARE`` of the weight adds its surface and mood. When no style
    clearly leads, the medium is named as a cross of the two heaviest.

    This is done by rule, not by a language model. Asked to fuse two styles into one, the
    7B model Dreambot runs joins their phrases with "and" or, given a worked example, copies
    the example's wording into the answer; neither beats a rule that is at least predictable.
    """
    ranked = shares(parts)
    if not ranked:
        return None
    if len(ranked) == 1:
        return ranked[0][0]
    lead = ranked[0][0]
    worded = [(print_, share) for print_, share in ranked if print_.worded]
    # The lead always speaks; the rest only once they are a real part of the mix.
    voiced = [print_ for index, (print_, share) in enumerate(ranked) if index == 0 or share >= ACCENT_SHARE]

    swatches = _merge_swatches(
        Swatch(swatch.rgb, swatch.share * share) for print_, share in ranked for swatch in print_.swatches
    )
    toned = [(print_.tone, share) for print_, share in ranked if print_.tone is not None]
    tone: Tone | None = None
    if toned:
        total = sum(share for _, share in toned)
        tone = Tone(
            **{
                name: round(sum(getattr(t, name) * share for t, share in toned) / total, 3)
                for name in ("brightness", "contrast", "saturation", "warmth")
            }
        )

    medium = rendering = light = ""
    if worded:
        first, first_share = worded[0]
        medium, rendering, light = first.medium, first.rendering, first.light
        rivals = [print_ for print_, _ in worded[1:] if print_.medium.lower() != first.medium.lower()]
        if rivals and first_share / sum(share for _, share in worded) < DOMINANT_SHARE:
            medium = f"{first.medium} crossed with {_with_article(rivals[0].medium)}"
            rendering = " and ".join(_distinct([first.rendering, rivals[0].rendering], 2))
    all_measured = all(print_.swatches for print_, _ in ranked)
    palette = (
        palette_words(swatches, tone)
        if all_measured and swatches
        else " with ".join(_distinct((print_.palette for print_, _ in ranked), 2))
    )
    blended = StyleFingerprint(
        medium=medium,
        rendering=rendering,
        palette=palette,
        light=light or next((print_.light for print_, _ in ranked if print_.light), ""),
        texture=" and ".join(_distinct((print_.texture for print_ in voiced), 2)),
        mood=", ".join(_distinct((print_.mood for print_ in voiced), 2)),
        swatches=swatches,
        tone=tone,
    )
    names = _distinct((print_.name for print_, _ in ranked), 2)
    return replace(blended, name=" × ".join(names) if len(names) > 1 else (lead.name or name_of(blended)))
