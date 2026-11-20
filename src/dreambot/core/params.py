"""The prompt parameters Dreambot understands (``--ar 16:9``, ``--chaos 20``, ...).

The table follows Midjourney's parameter list (as of V8.2, plus the legacy parameters people
still type) so that prompts written for Midjourney work unchanged. Each parameter records how
faithfully it is reproduced locally:

``native``
    Same behaviour as Midjourney, done for real by the local model.
``emulated``
    Midjourney's version depends on its proprietary model; Dreambot approximates the effect
    by other means (documented per parameter).
``accepted``
    Parsed and validated for compatibility, but has nothing to do on a local, private,
    single-GPU setup.
``planned``
    Recognised, not implemented yet; using it yields a clear message rather than a parse error.

Parameters marked ``extension`` do not exist in Midjourney: they expose controls that only make
sense when you own the model (which model, how many steps, which LoRAs, how long a video).
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field, fields
from enum import StrEnum
from typing import Any

MAX_SEED = 4294967295


class Support(StrEnum):
    NATIVE = "native"
    EMULATED = "emulated"
    ACCEPTED = "accepted"
    PLANNED = "planned"


class Kind(StrEnum):
    FLAG = "flag"  # no value
    INT = "int"
    FLOAT = "float"
    RATIO = "ratio"  # W:H
    TEXT = "text"  # free text up to the next parameter
    LIST = "list"  # comma-separated free text
    CHOICE = "choice"
    REFS = "refs"  # image references: URLs, @attachments, codes, "random", each with optional ::weight
    OPTIONAL_VALUE = "optional_value"  # flag that may carry a value (--niji 7, --p code1 code2)
    NAME = "name"  # a single identifier (model name)
    LORAS = "loras"  # name[:scale] ...


class PromptError(ValueError):
    """A prompt could not be understood. The message is written for the person who typed it."""


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: Kind
    aliases: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()
    support: Support = Support.NATIVE
    extension: bool = False
    #: One-line description shown in help.
    summary: str = ""
    #: How Dreambot realises the parameter when that differs from Midjourney.
    how: str = ""
    #: Field of :class:`Params` this parameter fills (defaults to ``name``).
    target: str | None = None
    #: Value stored for a FLAG parameter (defaults to ``True``).
    flag_value: Any = True

    @property
    def dest(self) -> str:
        return self.target or self.name


@dataclass(frozen=True)
class Ref:
    """One image reference inside a parameter such as ``--sref`` or ``--edit``."""

    #: ``url`` | ``attachment`` (1-based index into the message's attachments) | ``code`` | ``random`` |
    #: ``path`` | ``saved`` (a style kept under a name, ``--sref`` only)
    kind: str
    value: str
    weight: float = 1.0


@dataclass(frozen=True)
class LoraRef:
    name: str
    scale: float = 1.0


@dataclass
class Params:
    """Parsed parameter values. ``None`` means "not given": the user's settings, then the
    model's defaults, fill it in later."""

    # -- composition
    aspect: tuple[int, int] | None = None
    seed: int | None = None
    sameseed: int | None = None
    repeat: int | None = None
    count: int | None = None
    hd: bool | None = None
    tile: bool | None = None
    # -- look
    stylize: int | None = None
    chaos: int | None = None
    weird: int | None = None
    exp: int | None = None
    raw: bool | None = None
    style: str | None = None
    quality: float | None = None
    stop: int | None = None
    draft: bool | None = None
    no: list[str] = field(default_factory=list)
    # -- references
    iw: float | None = None
    sref: list[Ref] = field(default_factory=list)
    sw: int | None = None
    sv: int | None = None
    cref: list[Ref] = field(default_factory=list)
    cw: int | None = None
    oref: list[Ref] = field(default_factory=list)
    ow: int | None = None
    edit: list[Ref] = field(default_factory=list)
    personalize: list[str] | None = None
    # -- model selection
    version: str | None = None
    niji: str | None = None
    model: str | None = None
    # -- video
    video: bool | None = None
    motion: str | None = None
    loop: bool | None = None
    end: list[Ref] = field(default_factory=list)
    bs: int | None = None
    seconds: float | None = None
    #: What a video made from words alone starts on: ``picture`` (one made from the words
    #: first) or ``words`` (nothing: the video model films the words itself). ``--from``.
    video_from: str | None = None
    # -- compatibility no-ops
    speed: str | None = None
    visibility: str | None = None
    preview: bool | None = None
    # -- extensions
    steps: int | None = None
    guidance: float | None = None
    strength: float | None = None
    loras: list[LoraRef] = field(default_factory=list)
    negative: str | None = None

    @property
    def wants_video(self) -> bool:
        """Whether these parameters ask for a video: ``--video``, or one that only a video has."""
        return bool(
            self.video or self.motion or self.loop or self.end or self.bs or self.seconds or self.video_from
        )

    def given(self) -> dict[str, Any]:
        """Only the parameters that were explicitly set."""
        return {
            f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) not in (None, [], ())
        }

    def merged_over(self, defaults: Params) -> Params:
        """These parameters, with unset ones filled from ``defaults``."""
        merged = Params()
        for f in fields(self):
            mine = getattr(self, f.name)
            setattr(merged, f.name, mine if mine not in (None, []) else getattr(defaults, f.name))
        return merged


_E = Support.EMULATED
_A = Support.ACCEPTED
_P = Support.PLANNED

PARAMS: tuple[ParamSpec, ...] = (
    # ------------------------------------------------------------- composition ----
    ParamSpec(
        "aspect",
        Kind.RATIO,
        aliases=("ar",),
        summary="Aspect ratio, width:height in whole numbers (default 1:1).",
    ),
    ParamSpec(
        "seed",
        Kind.INT,
        minimum=0,
        maximum=MAX_SEED,
        summary="Starting seed. The same seed reproduces the same images exactly.",
    ),
    ParamSpec(
        "sameseed",
        Kind.INT,
        minimum=0,
        maximum=MAX_SEED,
        summary="Legacy: give all images in the grid the same seed.",
    ),
    ParamSpec(
        "repeat", Kind.INT, aliases=("r",), minimum=1, maximum=40, summary="Run the prompt this many times."
    ),
    ParamSpec(
        "count",
        Kind.INT,
        aliases=("n",),
        minimum=1,
        maximum=16,
        extension=True,
        summary="Images per job (default 4, a 2x2 grid).",
    ),
    ParamSpec(
        "hd",
        Kind.FLAG,
        support=_E,
        summary="2K output (2048px base instead of 1024px).",
        how="Generates at standard size, then runs the upscaler on every image.",
    ),
    ParamSpec(
        "sd", Kind.FLAG, target="hd", flag_value=False, summary="Standard 1024px output (the default)."
    ),
    ParamSpec(
        "tile",
        Kind.FLAG,
        support=_P,
        summary="Generate a seamlessly repeating tile.",
    ),
    # -------------------------------------------------------------------- look ----
    ParamSpec(
        "stylize",
        Kind.INT,
        aliases=("s",),
        minimum=0,
        maximum=1000,
        support=_E,
        summary="How much artistic interpretation to add (0-1000, default 100).",
        how=(
            "Sets how boldly the local language model art-directs your prompt (0 uses it as written). "
            "Without a language model, 250 and up add artistic emphasis by rule."
        ),
    ),
    ParamSpec(
        "chaos",
        Kind.INT,
        aliases=("c",),
        minimum=0,
        maximum=100,
        support=_E,
        summary="How different the images in a grid are from each other (0-100).",
        how=(
            "Each image gets its own direction: different shots at 0, then different light, palette and "
            "setting, and from 75 a different medium. Written by the local language model, or picked by rule."
        ),
    ),
    ParamSpec(
        "weird",
        Kind.INT,
        aliases=("w",),
        minimum=0,
        maximum=3000,
        support=_E,
        summary="Push results toward the unconventional (0-3000).",
        how="Adds a strange twist to each image, scaled by the value (by the language model, or by rule).",
    ),
    ParamSpec(
        "exp",
        Kind.INT,
        minimum=0,
        maximum=100,
        support=_E,
        summary="Experimental aesthetics: more detail and dynamism (0-100).",
        how="Asks for more intricate detail and more dynamic composition, scaled by the value.",
    ),
    ParamSpec("raw", Kind.FLAG, summary="Use your prompt as written: no art direction is added."),
    ParamSpec(
        "style",
        Kind.CHOICE,
        choices=("raw", "cute", "expressive", "scenic", "original", "4a", "4b", "4c"),
        summary="Legacy style switch; `--style raw` is the same as `--raw`.",
        how="Only `raw` has an effect; the legacy Niji and V4 styles are accepted and ignored.",
    ),
    ParamSpec(
        "quality",
        Kind.FLOAT,
        aliases=("q",),
        minimum=0.25,
        maximum=4,
        support=_E,
        summary="Time spent per image: .25, .5, 1, 2 or 4 (default 1).",
        how="Scales the number of denoising steps (never below what a distilled model needs).",
    ),
    ParamSpec(
        "stop",
        Kind.INT,
        minimum=10,
        maximum=100,
        summary="Stop the job partway (10-100%) for a softer, less finished image.",
    ),
    ParamSpec(
        "draft",
        Kind.FLAG,
        support=_E,
        summary="Fast, rough drafts.",
        how="Renders at reduced resolution with the fastest installed model.",
    ),
    ParamSpec(
        "no",
        Kind.LIST,
        summary="Things to keep out of the image, comma-separated.",
        how="A true negative prompt on models with classifier-free guidance; otherwise folded into the prompt.",
    ),
    # -------------------------------------------------------------- references ----
    ParamSpec(
        "iw",
        Kind.FLOAT,
        minimum=0,
        maximum=3,
        support=_E,
        summary="Weight of image prompts relative to the text (0-3, default 1).",
        how="Maps onto how strongly the image-prompt model follows the reference images.",
    ),
    ParamSpec(
        "sref",
        Kind.REFS,
        support=_E,
        summary="Style reference: images, style codes, saved styles or `random`, each with an optional `::weight`.",
        how="Every reference becomes a style fingerprint (medium, rendering, palette, light, texture, mood) "
        "and they are blended by weight; an edit model is shown the pictures as well. Codes are "
        "Dreambot's own, not Midjourney's.",
    ),
    ParamSpec(
        "sw",
        Kind.INT,
        minimum=0,
        maximum=1000,
        support=_E,
        summary="Style reference weight (0-1000, default 100).",
        how="Scales how insistently the style instruction is worded and, where the model allows, the reference strength.",
    ),
    ParamSpec(
        "sv",
        Kind.INT,
        minimum=1,
        maximum=7,
        support=_E,
        summary="Style version: 1 original, 2 blended (default), 3 words only.",
        how="1 keeps the original recipes for codes; 2 blends style fingerprints and shows an edit model "
        "the pictures; 3 applies the fingerprint in words on your chosen model.",
    ),
    ParamSpec(
        "cref",
        Kind.REFS,
        support=_E,
        summary="Character reference image(s) (Midjourney V6).",
        how="Passed to a reference-capable edit model with an instruction to keep the character.",
    ),
    ParamSpec(
        "cw",
        Kind.INT,
        minimum=0,
        maximum=100,
        support=_E,
        summary="Character weight (0 = face only, 100 = face, hair and clothes).",
        how="Low values ask the edit model to keep only the face; high values also keep hair and clothing.",
    ),
    ParamSpec(
        "oref",
        Kind.REFS,
        support=_E,
        summary="Omni reference image (Midjourney V7): put this person or object in the image.",
        how="Passed to a reference-capable edit model with an instruction to keep the subject.",
    ),
    ParamSpec(
        "ow",
        Kind.INT,
        minimum=1,
        maximum=1000,
        support=_E,
        summary="Omni reference weight (1-1000, default 100).",
        how="Scales how strictly the edit model is told to preserve the referenced subject.",
    ),
    ParamSpec(
        "edit",
        Kind.REFS,
        summary="Edit or combine up to 4 images by instruction (Midjourney V8 Edit Model).",
        how="Runs a local instruction-edit model (FLUX.2 klein, Qwen-Image-Edit, ...).",
    ),
    ParamSpec(
        "personalize",
        Kind.OPTIONAL_VALUE,
        aliases=("p", "profile", "look"),
        support=_E,
        summary="Apply Lookbooks: your defaults, or ones named by name or code, each with an optional `::weight`.",
        how="A Lookbook is a set of pictures (Midjourney's moodboard). Each image of a grid leans on a "
        "different picture of it; `--stylize` sets how strongly it is applied.",
    ),
    # ---------------------------------------------------------- model selection ----
    ParamSpec(
        "version",
        Kind.NAME,
        aliases=("v",),
        support=_A,
        summary="Midjourney model version.",
        how="Midjourney version numbers are accepted and ignored; pick a local model with `--model`.",
    ),
    ParamSpec(
        "niji",
        Kind.OPTIONAL_VALUE,
        support=_E,
        summary="Anime and illustration style.",
        how="Switches to the installed anime model (Anima).",
    ),
    ParamSpec(
        "model",
        Kind.NAME,
        aliases=("m",),
        extension=True,
        summary="Which local model to use, e.g. `--model qwen-image` (see `/models`).",
    ),
    # ------------------------------------------------------------------- video ----
    ParamSpec(
        "video",
        Kind.FLAG,
        summary="Make a video: of the image at the start of the prompt, or from the words alone.",
        how="A local video model. Midjourney only animates an image. Here a prompt without one works too: "
        "a picture is made from the words first, and the video starts on it (see `--from`).",
    ),
    ParamSpec(
        "from",
        Kind.CHOICE,
        choices=("picture", "words"),
        target="video_from",
        extension=True,
        summary="What a video with no image starts on: `picture` (one made from your words first, the "
        "default) or `words` (the video model films the words itself).",
    ),
    ParamSpec(
        "motion",
        Kind.CHOICE,
        choices=("low", "high"),
        support=_E,
        summary="How much moves in a video: `low` (the default) or `high`.",
        how="A video that starts on a picture is begun from the picture held still (low: the frame stays "
        "where it is) or from nothing (high: the camera moves too), and the model is told the same in "
        "words. A video filmed from words alone is only told.",
    ),
    ParamSpec("loop", Kind.FLAG, summary="Make the video end on the frame it starts on, so that it loops."),
    ParamSpec(
        "end",
        Kind.REFS,
        # Midjourney's ending frame. No video model here can be led up to a picture yet: the
        # planner says so when it is asked for (see ``VideoTraits.ends``).
        support=_P,
        summary="An image for the video to end on. Not yet: Wan 2.2 cannot lead a video up to a picture.",
    ),
    ParamSpec(
        "bs",
        Kind.INT,
        minimum=1,
        maximum=4,
        summary="Videos per job, 1-4 (the default is 4, as in Midjourney).",
    ),
    ParamSpec(
        "seconds",
        Kind.FLOAT,
        aliases=("sec",),
        minimum=1,
        maximum=5,
        extension=True,
        summary="How long a video is, 1-5 seconds (default 5), or how much `extend` adds (default 4).",
    ),
    # ------------------------------------------------------ compatibility no-ops ----
    ParamSpec(
        "fast",
        Kind.FLAG,
        target="speed",
        flag_value="fast",
        support=_A,
        summary="No effect: local jobs always run at full speed.",
    ),
    ParamSpec(
        "relax",
        Kind.FLAG,
        target="speed",
        flag_value="relax",
        support=_A,
        summary="No effect: local jobs always run at full speed.",
    ),
    ParamSpec(
        "turbo",
        Kind.FLAG,
        target="speed",
        flag_value="turbo",
        support=_A,
        summary="No effect: local jobs always run at full speed.",
    ),
    ParamSpec(
        "stealth",
        Kind.FLAG,
        target="visibility",
        flag_value="stealth",
        support=_A,
        summary="No effect: everything Dreambot makes is private.",
    ),
    ParamSpec(
        "public",
        Kind.FLAG,
        target="visibility",
        flag_value="public",
        support=_A,
        summary="No effect: everything Dreambot makes is private.",
    ),
    ParamSpec(
        "preview", Kind.FLAG, support=_A, summary="No effect (Midjourney's opt-in to preview features)."
    ),
    # -------------------------------------------------------------- extensions ----
    ParamSpec(
        "steps", Kind.INT, minimum=1, maximum=200, extension=True, summary="Exact number of denoising steps."
    ),
    ParamSpec(
        "guidance",
        Kind.FLOAT,
        aliases=("cfg",),
        minimum=0,
        maximum=30,
        extension=True,
        summary="Guidance scale, for models that use one.",
    ),
    ParamSpec(
        "strength",
        Kind.FLOAT,
        minimum=0,
        maximum=1,
        extension=True,
        summary="For variations and image-to-image: how much of the source image to keep (0-1).",
    ),
    ParamSpec(
        # Named as people type it, so that a job's prompt reads back the way it was written.
        "lora",
        Kind.LORAS,
        aliases=("loras",),
        target="loras",
        extension=True,
        summary="LoRA(s) to apply: `--lora name` or `--lora name:0.8 other:0.5`; `--lora none` for none.",
        how="A LoRA is a small fine-tune of one model for one look. One added with `dreambot loras add` "
        "brings its model and its trigger words with it.",
    ),
    ParamSpec("negative", Kind.TEXT, aliases=("neg",), extension=True, summary="Free-text negative prompt."),
)

_BY_NAME: dict[str, ParamSpec] = {}
for _spec in PARAMS:
    for _name in (_spec.name, *_spec.aliases):
        if _name in _BY_NAME:
            raise RuntimeError(f"duplicate parameter name: {_name}")
        _BY_NAME[_name] = _spec


def find_param(name: str) -> ParamSpec | None:
    return _BY_NAME.get(name.lower())


def param_names() -> list[str]:
    return sorted(_BY_NAME)


def suggest_param(name: str) -> list[str]:
    return difflib.get_close_matches(name.lower(), param_names(), n=2, cutoff=0.6)


# --------------------------------------------------------------------------------------
# Value parsing
# --------------------------------------------------------------------------------------

_URL_RE = re.compile(r"^https?://\S+$", re.IGNORECASE)
_NUMBER_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)$")
_WEIGHT_SUFFIX_RE = re.compile(r"^(?P<body>.+?)::(?P<weight>[+-]?(?:\d+\.?\d*|\.\d+))$")
#: Midjourney only accepts these values for --quality; anything else is snapped to the nearest.
QUALITY_STEPS = (0.25, 0.5, 1.0, 2.0, 4.0)


def _number(spec: ParamSpec, raw: str) -> float:
    text = raw.strip()
    if not text:
        raise PromptError(f"`--{spec.name}` needs a value.")
    if not _NUMBER_RE.match(text):
        raise PromptError(f"`--{spec.name}` needs a number, not `{text}`.")
    value = float(text)
    if (spec.minimum is not None and value < spec.minimum) or (
        spec.maximum is not None and value > spec.maximum
    ):
        raise PromptError(
            f"`--{spec.name}` must be between {_fmt(spec.minimum)} and {_fmt(spec.maximum)} (got {_fmt(value)})."
        )
    return value


def _fmt(value: float | None) -> str:
    if value is None:
        return "?"
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def parse_ratio(raw: str) -> tuple[int, int]:
    text = raw.strip().lower().replace("x", ":").replace("/", ":")
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)", text)
    if not match:
        raise PromptError(f"`--ar` needs a ratio like `16:9`, not `{raw.strip()}`.")
    width, height = float(match.group(1)), float(match.group(2))
    if width <= 0 or height <= 0:
        raise PromptError("`--ar` values must be greater than zero.")
    # Midjourney only takes whole numbers; be kinder and scale decimals up (1.5:1 -> 3:2).
    for scale in (1, 2, 4, 5, 10, 100):
        if (width * scale).is_integer() and (height * scale).is_integer():
            width, height = width * scale, height * scale
            break
    else:
        raise PromptError(f"`--ar {raw.strip()}` has too many decimals; use whole numbers like `16:9`.")
    from math import gcd

    w, h = int(width), int(height)
    divisor = gcd(w, h)
    w, h = w // divisor, h // divisor
    if max(w, h) / min(w, h) > 14:
        raise PromptError("`--ar` can be at most 14:1 (or 1:14).")
    return w, h


def parse_refs(spec: ParamSpec, raw: str) -> list[Ref]:
    """Parse the value of a reference parameter.

    Accepted tokens: ``https://...`` URLs, ``@1`` (the first image attached to the message),
    file paths, ``random``, numeric style codes and, for ``--sref``, the name of a saved style;
    each may end in ``::weight``. An empty value means "use the images attached to this
    message" and yields no refs here.
    """
    refs: list[Ref] = []
    for token in raw.split():
        weight = 1.0
        if match := _WEIGHT_SUFFIX_RE.match(token):
            token, weight = match.group("body"), float(match.group("weight"))
        if _URL_RE.match(token):
            refs.append(Ref("url", token, weight))
        elif re.fullmatch(r"@\d+", token):
            index = int(token[1:])
            if index < 1:
                raise PromptError(f"`--{spec.name} {token}`: attachments are numbered from @1.")
            refs.append(Ref("attachment", str(index), weight))
        elif token.lower() == "random":
            refs.append(Ref("random", "random", weight))
        elif re.fullmatch(r"\d+", token):
            refs.append(Ref("code", token, weight))
        elif token.startswith(("/", "~/", "./")):
            refs.append(Ref("path", token, weight))
        elif spec.name == "sref" and re.fullmatch(r"[A-Za-z][\w.\-]*", token):
            refs.append(Ref("saved", token, weight))  # looked up when the prompt is run
        else:
            raise PromptError(
                f"`--{spec.name}` takes image links, `@1`-style attachment numbers, style codes or `random`; "
                f"I don't understand `{token}`."
            )
    return refs


def parse_loras(raw: str) -> list[LoraRef]:
    loras: list[LoraRef] = []
    for token in raw.replace(",", " ").split():
        name, _, scale_text = (
            token.rpartition(":") if ":" in token and not token.startswith("http") else (token, "", "")
        )
        if scale_text and _NUMBER_RE.match(scale_text):
            scale = float(scale_text)
        elif scale_text:
            name, scale = token, 1.0
        else:
            scale = 1.0
        if not name:
            raise PromptError(f"`--lora {token}`: missing LoRA name.")
        if not -4.0 <= scale <= 4.0:
            raise PromptError(f"`--lora {token}`: the scale must be between -4 and 4.")
        loras.append(LoraRef(name, scale))
    if not loras:
        raise PromptError("`--lora` needs a LoRA name, e.g. `--lora watercolor:0.8`.")
    return loras


def parse_value(spec: ParamSpec, raw: str) -> Any:
    """Convert the raw text following ``--name`` into the parameter's value."""
    text = raw.strip()
    kind = spec.kind
    if kind is Kind.FLAG:
        if text:
            raise PromptError(f"`--{spec.name}` doesn't take a value (found `{text}` after it).")
        return spec.flag_value
    if kind is Kind.INT:
        value = _number(spec, text)
        if not value.is_integer():
            raise PromptError(f"`--{spec.name}` needs a whole number, not `{text}`.")
        return int(value)
    if kind is Kind.FLOAT:
        value = _number(spec, text)
        if spec.name == "quality":
            value = min(QUALITY_STEPS, key=lambda step: abs(step - value))
        return value
    if kind is Kind.RATIO:
        return parse_ratio(text)
    if kind is Kind.TEXT:
        if not text:
            raise PromptError(f"`--{spec.name}` needs a value.")
        return text
    if kind is Kind.LIST:
        items = [item.strip() for item in text.split(",") if item.strip()]
        if not items:
            raise PromptError(
                f"`--{spec.name}` needs at least one item, e.g. `--{spec.name} text, watermark`."
            )
        return items
    if kind is Kind.CHOICE:
        choice = text.lower()
        if choice not in spec.choices:
            raise PromptError(
                f"`--{spec.name}` must be one of: {', '.join(spec.choices)} (got `{text or 'nothing'}`)."
            )
        return choice
    if kind is Kind.REFS:
        return parse_refs(spec, text)
    if kind is Kind.OPTIONAL_VALUE:
        if spec.name == "personalize":
            return text.split() or ["default"]
        return text or "default"
    if kind is Kind.NAME:
        if not text or len(text.split()) != 1:
            raise PromptError(f"`--{spec.name}` needs a single name, e.g. `--{spec.name} {_example(spec)}`.")
        return text
    if kind is Kind.LORAS:
        return parse_loras(text)
    raise AssertionError(f"unhandled parameter kind: {kind}")  # pragma: no cover


def _example(spec: ParamSpec) -> str:
    return {"model": "z-image-turbo", "version": "8.2"}.get(spec.name, "value")
