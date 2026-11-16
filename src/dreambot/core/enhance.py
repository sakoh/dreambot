"""The prompt enhancer: a local language model acting as art director.

Midjourney's look comes from inside its model. ``--stylize`` says how much of that house
aesthetic to apply, ``--chaos`` how different the images of one grid are, ``--weird`` and
``--exp`` how far from the ordinary to go. Open models have no such dials: they draw what the
prompt says, and a short prompt leaves most of the picture unsaid.

Dreambot fills in what a short prompt leaves open. For each image of a job, a small language
model running on the Mac writes one line of *art direction* — the shot, the light, the palette,
the finish — and that line is added after the prompt. The person's own words are never
rewritten or dropped. The model only ever adds to them, so nothing that was asked for can be
lost, and a quoted sign or a name comes through letter for letter.

This module is the pure part: deciding whether a job should be directed, wording the request
to the language model, and checking what comes back. Running the model is the service's job
(:meth:`dreambot.core.service.Dreambot._enhance`). :mod:`dreambot.core.styles` is the fallback
when no language model is installed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from dreambot.core.text import parse_numbered

#: A prompt this long already says how the picture should look; at default settings it is left alone.
DETAILED_WORDS = 45
#: Below this the dial is treated as "off", matching the prompt-level fallback in ``styles``.
CHAOS_THRESHOLD = 10
#: From here up, chaos asks for a different medium per image.
CHAOS_MEDIUMS = 75
#: Flat graphics want to be drawn exactly as described: a camera angle or a light source added to
#: a logo makes it worse, and small language models add them however they are told not to.
_GRAPHIC = re.compile(
    r"\b(?:logo|logotype|icon|icons|vector|sticker|emoji|pictogram|infographic|diagram|wireframe|"
    r"typography|lettering|monogram|clip ?art|seamless pattern|flat design|svg|qr code|floor ?plan|"
    r"pixel art|coloring page|colouring page)\b",
    re.IGNORECASE,
)

SYSTEM = (
    "You are the art director for a text-to-image model. You are given a prompt. For each image to be "
    "made from it you write one line of visual direction, which is added after the prompt. The prompt "
    "already says what the picture is of. Your line says how it is seen.\n"
    "\n"
    "A line is a few short descriptive phrases separated by commas, in this order: the shot and "
    "composition, the light, the colours and mood, the medium and its fine detail.\n"
    "\n"
    "Rules:\n"
    "- Each line is used on its own, for a different image. Never refer to another line, and never "
    "write 'the same'.\n"
    "- Do not restate the prompt and do not contradict it. If the prompt names a medium, a style, a "
    "colour or a kind of light, keep to it and build on it.\n"
    "- Suit the medium: lens, focus and film for a photograph; brushwork, line and paper for a painting "
    "or drawing.\n"
    "- Do not add subjects, people, animals, objects or written words that the prompt does not mention.\n"
    "- It is a single still image: nothing moves, and the camera does not move.\n"
    "- Be concrete. No empty praise such as stunning, beautiful or masterpiece, and no resolution terms "
    "such as 4k or 8k.\n"
    "- Write in English: phrases, not full sentences, and no quotation marks.\n"
    "\n"
    "Two examples of the form. Do not reuse their wording.\n"
    "\n"
    "Prompt: a bowl of ramen on a wooden counter\n"
    "1. overhead shot with the bowl centred, hard window light from the left casting long shadows, deep "
    "umber and amber tones, crisp 50mm photograph\n"
    "2. low angle close to the rim, soft overcast light, muted greens against the warm broth, shallow "
    "depth of field with the counter out of focus\n"
    "3. three-quarter view with empty space to the right, warm tungsten glow, rich reds and browns, fine "
    "film grain\n"
    "\n"
    "Prompt: a harbour at dawn, gouache\n"
    "1. wide view with the boats low in the frame, pale light spreading from the horizon, soft peach and "
    "slate blue, flat opaque washes on textured paper\n"
    "2. close view of two moored boats, cool early light with long reflections, dusty teal and coral, "
    "dry-brush edges and simple shapes"
)
#: Nucleus sampling for the request. Without it a 4-bit model at a raised temperature produces
#: word salad; with it, variety has to come from the instructions, which is where it belongs.
TOP_P = 0.9
#: Small models asked for long lines start looping on a phrase; this stops them.
REPETITION_PENALTY = 1.15


@dataclass(frozen=True)
class Brief:
    """What the art director is told about one job."""

    #: The person's prompt, without parameters.
    text: str
    #: How many images, and so how many lines of direction.
    count: int
    stylize: int = 100
    chaos: int = 0
    weird: int = 0
    exp: int = 0
    #: A look the images must have (from a style code). The direction has to agree with it.
    style: tuple[str, ...] = ()
    #: Things the person asked not to see (``--no``).
    avoid: tuple[str, ...] = ()

    @property
    def dials(self) -> bool:
        """Whether the person asked for variety or strangeness explicitly."""
        return self.chaos >= CHAOS_THRESHOLD or self.weird > 0 or self.exp > 0

    @property
    def words(self) -> int:
        """The most words one line of direction may have."""
        return _amount(self.stylize)[0] + (8 if self.weird or self.exp else 0)


def wanted(brief: Brief) -> bool:
    """Whether this job should be given art direction at all."""
    if not brief.text.strip() or brief.count < 1 or _GRAPHIC.search(brief.text):
        return False
    if brief.dials:
        return True
    if brief.stylize <= 0:
        return False
    # A long prompt at default stylize has already been art-directed by its author.
    return not (len(brief.text.split()) >= DETAILED_WORDS and brief.stylize <= 100)


def _amount(stylize: int) -> tuple[int, str]:
    """(word budget, how far to go) for a stylize value.

    The wording deliberately contains nothing that could be copied into a line: a small model
    told to use "dramatic cinematic light" writes exactly that, four times.
    """
    if stylize <= 0:
        return 14, "Add no style of your own. Write only what the variety asked for below needs."
    if stylize < 75:
        return 16, "Keep the direction minimal: two phrases, the shot and the light."
    if stylize < 200:
        return 30, "Make the direction balanced and considered, as a good photographer or painter would."
    if stylize < 600:
        return 42, (
            "Make the direction bold: more dramatic and more stylised than a straightforward picture, "
            "with a fifth phrase for atmosphere."
        )
    return 55, (
        "Make the direction as strong and opinionated as you can — daring, theatrical, far from a "
        "straightforward picture — with extra phrases for texture and atmosphere."
    )


def _variety(chaos: int, count: int) -> str:
    if count < 2:
        return ""
    if chaos < CHAOS_THRESHOLD:
        return (
            "The images are one series: keep the same look in every line and change only the shot — "
            "the framing, the camera angle or the moment. No two lines may describe the same shot."
        )
    if chaos < 40:
        return "Give each image a clearly different composition and different light."
    if chaos < 75:
        return (
            "Make the images very different from one another: composition, light, colour palette and setting."
        )
    return (
        "Make every image a completely different interpretation: each line names a different medium or "
        "art style, with its own palette, composition and mood."
    )


def _medium(chaos: int) -> str:
    """Keep the enhancer from inventing a medium: left alone, it turns every other cat into a watercolour."""
    if chaos >= CHAOS_MEDIUMS:
        return ""
    return (
        "Unless the prompt names or clearly implies another medium or style, every image is a photograph: "
        "direct it as one, and do not mention paint, brushwork or drawing."
    )


def _strangeness(weird: int) -> str:
    """Strangeness is the one dial that may change the scene itself, not only how it is seen."""
    if weird <= 0:
        return ""
    if weird < 250:
        return "Give each image a slightly odd, offbeat touch."
    if weird < 1000:
        return (
            "Make each image surreal: begin every line with one strange twist on the scene itself — an "
            "impossible setting, scale, material or juxtaposition — and then direct it."
        )
    return (
        "Push each image into the bizarre: begin every line with a hallucinatory, uncanny twist on the "
        "scene itself — impossible settings, materials and distortions — and then direct it."
    )


def _experiment(exp: int) -> str:
    if exp <= 0:
        return ""
    if exp < 25:
        return "Add a little more fine detail than you otherwise would."
    if exp < 50:
        return "Make the images more intricate and the compositions more dynamic."
    return "Make the images as intricate, deep and dynamic as you can, with experimental compositions."


def request(brief: Brief) -> str:
    """The message asking the language model for ``brief.count`` lines of direction."""
    words, amount = _amount(brief.stylize)
    words = brief.words
    lines = [f"Prompt: {brief.text.strip()}", "", amount]
    for extra in (
        # A style already names its medium; "every image is a photograph" would contradict it.
        "" if brief.style else _medium(brief.chaos),
        _variety(brief.chaos, brief.count),
        _strangeness(brief.weird),
        _experiment(brief.exp),
    ):
        if extra:
            lines.append(extra)
    if brief.style:
        lines.append(
            f"Every image is made in this style, which you build on without repeating it: {', '.join(brief.style)}."
        )
    if brief.avoid:
        lines.append(f"Never mention: {', '.join(brief.avoid)}.")
    lines.append("")
    if brief.count == 1:
        lines.append(f"Write exactly one line, numbered 1., of at most {words} words, and nothing else.")
    else:
        lines.append(
            f"Write exactly {brief.count} lines, numbered 1. to {brief.count}., one per image, each of at "
            f"most {words} words, and nothing else."
        )
    return "\n".join(lines)


def max_tokens(brief: Brief) -> int:
    """Room for every line (about 1.5 tokens a word, plus numbering) without letting a model ramble."""
    return min(1500, 40 + brief.count * round(brief.words * 1.8 + 12))


def temperature(brief: Brief) -> float:
    """A little more adventurous with more chaos, but never enough to lose coherence."""
    return round(0.6 + 0.2 * min(100, max(0, brief.chaos)) / 100, 2)


_LABEL = re.compile(r"^(?:image|shot|line|direction|frame|variation)\s*\d*\s*[:\-–—]\s*", re.IGNORECASE)
_META = re.compile(r"\b(?:prompt|as an ai|i cannot|i can't|i can not|here (?:is|are))\b", re.IGNORECASE)
#: Phrases that only make sense next to the other lines, or in a film rather than a picture.
_UNUSABLE = re.compile(
    r"\b(?:the same|as before|this time|once again|previous (?:image|line|shot)|"
    r"(?:camera|shot|view|frame) (?:slowly |gently |subtly |then )?(?:pans?|zooms?|tracks?|follows?|sweeps?|glides?|"
    r"pulls? (?:back|out|away)|moves?|pushes? in)|panning|zooming (?:in|out)|time-?lapse|slow motion|"
    r"(?:slow|fast|gentle|smooth|quick) (?:pan|zoom|tilt|dolly)|tracking shot|dolly (?:in|out|shot|zoom))\b",
    re.IGNORECASE,
)
_PHRASE = re.compile(r"\s*[,;.](?:\s+|$)")
#: Words that turn a picture into a painting or a drawing.
_PAINTED = re.compile(
    r"\b(?:brush\w*|painterly|paint\w*|watercolou?r\w*|gouache|impasto|canvas|sketch\w*|drawing|drawn|"
    r"illustrat\w*|charcoal|engraving|etching|woodcut|linocut|anime|cartoon\w*|comic)\b",
    re.IGNORECASE,
)


def _clean(line: str, brief: Brief) -> str:
    """One line of direction made safe to append, or ``""`` if it cannot be used."""
    text = re.sub(r"\s+", " ", line).strip().strip("\"'`*“”‘’ ")
    text = _LABEL.sub("", text).strip()
    # Small models like to restate the prompt before continuing it.
    said = brief.text.strip().rstrip(",.;: ")
    if said and text.lower().startswith(said.lower()):
        text = text[len(said) :].lstrip(",.;:—–- ")
    # Qwen-family models sometimes slip into Chinese mid-sentence. Keep what came before.
    foreign = next((i for i, char in enumerate(text) if ord(char) > 0x024F and char.isalpha()), None)
    if foreign is not None:
        text = text[:foreign].rpartition(",")[0]
    if _META.search(text) or '"' in text or "“" in text:
        return ""
    phrases = [phrase.strip() for phrase in _PHRASE.split(text) if phrase.strip()]
    said_words = _content_words(brief.text)
    # The model may only paint when the person, a style code or a high chaos value asked for it.
    asked_for_paint = bool(_PAINTED.search(brief.text))
    photographic = brief.chaos < CHAOS_MEDIUMS and not brief.style and not brief.weird and not asked_for_paint
    # ...and when the person asked for a painting, a lens or a film stock would undo it.
    painted = brief.chaos < CHAOS_MEDIUMS and asked_for_paint and not _PHOTOGRAPHIC.search(brief.text)
    kept: list[str] = []
    seen: set[frozenset[str]] = set()
    for phrase in phrases:
        if _UNUSABLE.search(phrase) or phrase.startswith(("'s ", "’s ")):
            continue
        if any(_mentions(phrase, term) for term in brief.avoid):
            continue
        if photographic and _PAINTED.search(phrase):
            continue
        if painted and _PHOTOGRAPHIC.search(phrase):
            continue
        meaning = frozenset(_content_words(phrase))
        if meaning <= said_words or meaning in seen:
            continue  # the prompt's own words again, or a phrase the line already has
        seen.add(meaning)
        if kept and phrase[0].isupper() and phrase.split()[0][1:].islower():
            phrase = phrase[0].lower() + phrase[1:]  # it began a sentence in the answer
        kept.append(phrase)
    limit = round(brief.words * 1.5) + 4
    while len(kept) > 1 and sum(len(phrase.split()) for phrase in kept) > limit:
        kept.pop()  # a rambling line loses its last phrases, not its first
    words = " ".join(kept).split()
    if len(words) < 3:
        return ""
    if len(words) > limit:
        return " ".join(words[:limit])
    return ", ".join(kept)


#: Words that turn a painting into a photograph.
_PHOTOGRAPHIC = re.compile(
    r"\b(?:\d+\s?mm|lens\w*|film|photo\w*|depth of field|bokeh|exposure|darkroom|dslr|telephoto|"
    r"shutter|aperture|(?:shallow|sharp|selective|crisp) focus|out of focus|blurred background)\b",
    re.IGNORECASE,
)

_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "in",
        "on",
        "at",
        "to",
        "with",
        "and",
        "or",
        "for",
        "from",
        "by",
        "as",
        "is",
        "are",
        "its",
        "it",
        "this",
        "that",
        "into",
        "over",
        "under",
    ]
)


def _content_words(text: str) -> set[str]:
    """The words of a phrase that carry meaning, crudely singularised."""
    words = re.findall(r"[^\W\d_]+", text.lower())
    return {word.removesuffix("s") for word in words if word not in _STOPWORDS}


def _mentions(phrase: str, term: str) -> bool:
    stem = re.escape(term.strip().rstrip("s"))
    return bool(stem) and re.search(rf"\b{stem}(?:s|es)?\b", phrase, re.IGNORECASE) is not None


def parse(answer: str, brief: Brief) -> list[str]:
    """One checked line of direction per image, in order.

    A line the model garbled is replaced by one of its good lines, so a grid is still directed
    throughout. If nothing usable came back every entry is empty, and the caller falls back to
    the prompt-level emulation.
    """
    lines = [_clean(line, brief) for line in parse_numbered(answer, brief.count)]
    good = [line for line in lines if line]
    if not good:
        return [""] * brief.count
    lines += [""] * (brief.count - len(lines))
    return [line or good[index % len(good)] for index, line in enumerate(lines[: brief.count])]


def compose(text: str, direction: str) -> str:
    """The prompt with a line of direction added after it."""
    base = text.strip()
    if not direction:
        return base
    direction = direction[0].upper() + direction[1:]
    base = base.rstrip(",.;: ")
    return f"{base}. {direction}" if base else direction
