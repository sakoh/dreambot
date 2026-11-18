"""Planning: deciding how a request will be carried out on the models that are installed.

The planner turns "what the person asked for" (a parsed prompt, or an action on an earlier
image) into a :class:`Plan`: which model, which operation, what size, which seeds, and the
exact engine requests to run. It never touches MLX and performs no generation, so every
decision here is unit-testable.

When the ideal model for an operation is not installed, the planner falls back to the next
best way of doing it and records a note for the person, rather than failing — for example
outpainting through an instruction-edit model when no inpainting model is present.
"""

from __future__ import annotations

import math
import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from PIL import Image

from dreambot.config import Config
from dreambot.core import enhance, fingerprint, lookbook, styles
from dreambot.core.fingerprint import StyleFingerprint
from dreambot.core.imaging import Canvas, Tile
from dreambot.core.params import MAX_SEED, LoraRef, Params, parse_loras
from dreambot.core.prompt import ParsedPrompt
from dreambot.core.settings import ResolvedSettings
from dreambot.core.sizes import dimensions, snap, working_size
from dreambot.engine import loras as lora_library
from dreambot.engine import weights as weights_module
from dreambot.engine.protocol import GenerateRequest, Lora, Op
from dreambot.engine.registry import REGISTRY, ModelSpec, Registry, UnknownModel
from dreambot.engine.weights import Weights

#: The noise level a variation starts from: subtle changes details, strong keeps only the
#: composition. (The engine turns this into the right setting for each model and image size.)
VARIATION_NOISE = {"subtle": 0.6, "strong": 0.85}
#: Noise for re-rendering an enlarged image: enough to invent fine detail, not to change content.
CREATIVE_UPSCALE_NOISE = 0.45
#: The same, when standing in for a dedicated upscaler that is not installed.
FALLBACK_UPSCALE_NOISE = 0.3
MAX_UPSCALED_SIDE = 3072
#: Style versions (``--sv``): how a style reference becomes a look. 1 is the original (a code
#: is a fixed recipe, an image is only shown to an edit model); 2 blends fingerprints and shows
#: the pictures as well where a model can take them; 3 uses the fingerprint's words alone.
STYLE_LEGACY, STYLE_BLEND, STYLE_WORDS = 1, 2, 3
DEFAULT_STYLE_VERSION = STYLE_BLEND


class PlanError(Exception):
    """The request cannot be carried out. The message tells the person what to do about it."""


# --------------------------------------------------------------------------------------
# What is installed
# --------------------------------------------------------------------------------------


class Catalog:
    """Answers "which models can I use right now?"."""

    def __init__(
        self,
        config: Config,
        registry: Registry = REGISTRY,
        *,
        cache: Path | None = None,
        assume_installed: bool = False,
    ):
        self.config = config
        self.registry = registry
        self._cache = cache
        #: With the fake backend there are no weights; every model counts as installed.
        self.assume_installed = assume_installed

    def lookup(self, name: str) -> ModelSpec:
        try:
            return self.registry.get(name)
        except UnknownModel as exc:
            raise PlanError(str(exc)) from exc

    def weights(self, spec: ModelSpec) -> Weights | None:
        override = self.config.models.get(spec.id)
        return weights_module.resolve(
            spec, override=override.weights if override else None, cache=self._cache
        )

    def is_installed(self, spec: ModelSpec) -> bool:
        return self.assume_installed or self.weights(spec) is not None

    def pick(self, names: Iterable[str], op: Op) -> ModelSpec | None:
        """The first model in ``names`` that supports ``op`` and is installed."""
        for name in names:
            spec = self.registry.find(name)
            if spec is not None and spec.supports(op) and self.is_installed(spec):
                return spec
        return None

    def installed(self, op: Op | None = None) -> list[ModelSpec]:
        return [
            spec for spec in self.registry if (op is None or spec.supports(op)) and self.is_installed(spec)
        ]

    def any_with(self, op: Op, preferred: Iterable[str] = ()) -> ModelSpec | None:
        return self.pick(preferred, op) or next(iter(self.installed(op)), None)

    def vision_model(self) -> str | None:
        """Where the vision-language model for /describe lives, or ``None`` if there is none."""
        if self.assume_installed:
            return "fake"
        found = weights_module.find_vision_model(self.config.assist.vision_model, cache=self._cache)
        return str(found) if found else None


# --------------------------------------------------------------------------------------
# Plans
# --------------------------------------------------------------------------------------


#: Stands in a prompt's list of style images for a board that has not been cut yet.
BOARD = "<style board>"


@dataclass(frozen=True)
class Board:
    """A sheet of swatches cut from style pictures, for a model that takes reference images.

    Shown a style picture whole, an edit model puts what is in it into the result — a hand, a
    caption. A board of shuffled pieces keeps what makes the style (colour, line, texture,
    light) and loses what the pictures were of. Each picture gets pieces in proportion to its
    weight, so a blend of references is a blend on the board too, in one reference slot.
    """

    #: (picture, weight) for every picture that contributes.
    tiles: tuple[tuple[str, float], ...]
    #: Makes the cut reproducible.
    seed: int


@dataclass(frozen=True)
class Fitted:
    """One LoRA a job uses."""

    #: What it is called: its name in the library, or what the person typed.
    name: str
    #: Its file, or a Hugging Face repository that is already in the cache.
    path: str
    scale: float = 1.0
    #: Registry id of the model it was trained on, when that is known.
    base: str = ""
    #: The words it answers to, which go into the prompt.
    trigger: str = ""

    @property
    def lora(self) -> Lora:
        return Lora(path=self.path, scale=self.scale)

    def record(self) -> dict[str, Any]:
        """What a job keeps of it, so that variations and upscales of its pictures use it too."""
        return {"name": self.name, "path": self.path, "scale": self.scale, "base": self.base}


def triggered(prompt: str, loras: Sequence[Fitted]) -> str:
    """``prompt`` with the trigger words of every LoRA it is rendered with, the first one's first."""
    for lora in reversed(loras):
        prompt = lora_library.worded(lora.trigger, prompt)
    return prompt


@dataclass
class Render:
    """One engine request, producing one image per seed for the listed slots."""

    op: Op
    model: str
    prompt: str
    seeds: list[int]
    #: 1-based image numbers within the job, parallel to ``seeds``.
    slots: list[int]
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    guidance: float | None = None
    negative: str | None = None
    image: str | None = None
    strength: float | None = None
    noise: float | None = None
    images: list[str] = field(default_factory=list)
    image_weights: list[float] | None = None
    mask: str | None = None
    loras: list[Lora] = field(default_factory=list)
    weights: str | None = None
    quantize: int | None = None
    stop_at: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    #: A style board to cut and show the model after ``images`` (see :class:`Board`).
    board: Board | None = None

    def request(
        self,
        *,
        request_id: str,
        out_dir: str,
        basename: str,
        preview_every: int = 0,
        board: str | None = None,
    ) -> GenerateRequest:
        """The engine request. ``board`` is the file a style board was cut to, if the render has one."""
        return GenerateRequest(
            id=request_id,
            model=self.model,
            op=self.op,
            prompt=self.prompt,
            negative_prompt=self.negative,
            width=self.width,
            height=self.height,
            steps=self.steps,
            guidance=self.guidance,
            seeds=self.seeds,
            image=self.image,
            strength=self.strength,
            noise=self.noise,
            images=[*self.images, board] if board else self.images,
            image_weights=self.image_weights,
            mask=self.mask,
            loras=self.loras,
            weights=self.weights,
            quantize=self.quantize,
            out_dir=out_dir,
            basename=basename,
            stop_at=self.stop_at,
            preview_every=preview_every,
            extra=self.extra,
        )


@dataclass
class Plan:
    model: str
    op: Op
    #: The text handed to the model (before per-image direction such as chaos).
    text: str
    width: int
    height: int
    seeds: list[int]
    renders: list[Render]
    negative: str | None = None
    #: Run once per finished image, with ``image`` set to it (``--hd``).
    finish: Render | None = None
    #: Set for outpainting: the original is composited back over the result.
    canvas: Canvas | None = None
    #: Set for region edits: everything outside this mask is restored from ``source``.
    keep_outside_mask: str | None = None
    #: Set for tiled upscales: each render produces one tile, pasted back over ``tile_base``.
    tiles: list[Tile] | None = None
    tile_base: str | None = None
    source: str | None = None
    #: Set when a language model should write art direction for each image before rendering
    #: (see :mod:`dreambot.core.enhance`). The renders above already work without it.
    brief: enhance.Brief | None = None
    notes: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.seeds)

    @property
    def work(self) -> int:
        """How many model runs the renders add up to (for progress reporting)."""
        return sum(len(render.seeds) for render in self.renders)


def style_version(value: int | None, notes: list[str] | None = None) -> int:
    """The style version a ``--sv`` value selects."""
    if value is None:
        return DEFAULT_STYLE_VERSION
    if value in (STYLE_LEGACY, STYLE_BLEND, STYLE_WORDS):
        return value
    if notes is not None:
        notes.append(
            f"`--sv {value}` is a Midjourney style version. Dreambot has 1 (the original), 2 (blended, "
            f"the default) and 3 (words only); using {DEFAULT_STYLE_VERSION}."
        )
    return DEFAULT_STYLE_VERSION


@dataclass
class StyleSource:
    """One weighted ingredient of a prompt's style: a reference image, a style code, a saved style
    or a Lookbook."""

    #: ``image`` | ``code`` | ``saved`` | ``look``
    kind: str
    #: How it is named to the person: "image 1", "style 4821", "Lookbook Dunes".
    label: str
    weight: float = 1.0
    print: StyleFingerprint | None = None
    #: Pictures a style board may be cut from for this ingredient.
    pictures: list[str] = field(default_factory=list)
    #: Cache key of the picture, for ingredients whose words come from the vision model.
    key: str | None = None
    #: A style code, or the code of the Lookbook version used.
    code: str | None = None
    #: Whether ``code`` was drawn for ``--sref random``.
    drawn: bool = False
    #: The pictures of a Lookbook, each with its own fingerprint.
    members: list[lookbook.Member] = field(default_factory=list)


@dataclass
class Look:
    """The style one image is given."""

    print: StyleFingerprint | None
    #: The words added to the prompt.
    phrases: list[str]
    #: What a model that takes reference images is shown.
    board: Board | None = None


def _scaled(tiles: Sequence[tuple[str, float]], total: float) -> list[tuple[str, float]]:
    """``tiles`` with their weights rescaled to add up to ``total``."""
    have = sum(weight for _, weight in tiles)
    if have <= 0 or total <= 0:
        return []
    return [(path, weight * total / have) for path, weight in tiles]


@dataclass
class ResolvedRefs:
    """Reference images of a prompt, downloaded and normalised to local files."""

    #: Images given at the start of the prompt or attached without a role.
    image_prompts: list[str] = field(default_factory=list)
    #: ``--edit``: the images an instruction applies to.
    edit: list[str] = field(default_factory=list)
    #: ``--oref`` / ``--cref``: a person, character or object to carry over.
    subject: list[str] = field(default_factory=list)
    #: ``character`` for ``--cref``, ``subject`` for ``--oref``.
    subject_kind: str = "subject"
    #: ``--sref`` images.
    style: list[str] = field(default_factory=list)
    #: ``--sref`` codes, with the ``random`` ones already drawn.
    style_codes: list[str] = field(default_factory=list)
    #: Every style ingredient in the order given, with its weight (style version 2 and up).
    #: Left empty, it is derived from ``style`` and ``style_codes`` at equal weights.
    sources: list[StyleSource] = field(default_factory=list)

    @property
    def images(self) -> list[str]:
        return [*self.edit, *self.subject, *self.image_prompts, *self.style]

    @property
    def content_images(self) -> list[str]:
        """The images that say what the picture is of, as opposed to how it looks."""
        return [*self.edit, *self.subject, *self.image_prompts]

    def style_sources(self) -> list[StyleSource]:
        if not self.sources:
            self.sources = [
                *(
                    StyleSource("image", f"image {number}", pictures=[path])
                    for number, path in enumerate(self.style, start=1)
                ),
                *(StyleSource("code", f"style {code}", code=code) for code in self.style_codes),
            ]
        for source in self.sources:
            if source.kind == "look":
                source.print = lookbook.consensus(source.members)
            elif source.print is None:
                if source.code is not None:
                    source.print = fingerprint.from_code(source.code)
                elif source.pictures:
                    source.print = fingerprint.measure(source.pictures[0])
        return self.sources


def _join(numbers: Sequence[int], total: int) -> str:
    if total == 1:
        return "the reference image"
    if len(numbers) == 1:
        return f"image {numbers[0]}"
    return "images " + ", ".join(map(str, numbers[:-1])) + f" and {numbers[-1]}"


def reference_instruction(
    text: str,
    refs: ResolvedRefs,
    params: Params,
    *,
    strength: int | None = None,
    board: bool = False,
) -> str:
    """Word a prompt for a reference-capable edit model, naming each image's role.

    ``refs.style`` holds what the model is shown for style: the reference pictures themselves in
    the original style version, or — with ``board`` — one board of swatches cut from them.
    ``strength`` is how insistently the style is asked for (0-1000); it defaults to ``--sw``.
    """
    total = len(refs.images)
    cursor = 0
    clauses: list[str] = []

    def take(paths: list[str]) -> list[int]:
        nonlocal cursor
        numbers = list(range(cursor + 1, cursor + len(paths) + 1))
        cursor += len(paths)
        return numbers

    take(refs.edit)  # the instruction itself talks about these
    if refs.subject:
        which = _join(take(refs.subject), total)
        kind = refs.subject_kind
        if kind == "character" and params.cw is not None and params.cw < 50:
            clauses.append(
                f"Keep the face of the character from {which}; their hair and clothing may change."
            )
        elif params.ow is not None and params.ow >= 400:
            clauses.append(f"Reproduce the {kind} from {which} exactly, in every detail.")
        elif params.ow is not None and params.ow < 50:
            clauses.append(f"Loosely base the {kind} on {which}.")
        else:
            clauses.append(f"Keep the {kind} from {which}: the same identity, face and distinctive features.")
    if refs.image_prompts:
        which = _join(take(refs.image_prompts), total)
        weight = 1.0 if params.iw is None else params.iw
        if weight < 0.75:
            clauses.append(f"Loosely take inspiration from {which}.")
        elif weight > 1.5:
            clauses.append(f"Closely follow the composition, subject and mood of {which}.")
        else:
            clauses.append(f"Take inspiration for composition, subject and mood from {which}.")
    if refs.style:
        which = _join(take(refs.style), total)
        weight = strength if strength is not None else (100 if params.sw is None else params.sw)
        # Edit models copy what they are shown unless told otherwise, so every wording says
        # outright that the reference's content must not appear (verified on FLUX.2 klein).
        if weight <= 0:
            pass
        elif weight < 50:
            clauses.append(
                f"Borrow a little of the visual style of {which}, but show nothing that is depicted in it."
            )
        elif board:
            clauses.append(
                f"{which[0].upper()}{which[1:]} is a board of style swatches cut from other pictures, not part "
                "of the scene: show nothing that appears on it, and do not lay the picture out as a grid. "
                + (
                    "Reproduce its medium, color palette, line, texture, lighting and mood exactly."
                    if weight >= 400
                    else "Take only its visual look: color palette, line, texture, lighting and mood."
                )
            )
        elif weight >= 400:
            clauses.append(
                f"This is a completely new scene: do not show anything depicted in {which}. It is only a style "
                "guide; reproduce its medium, visual look, color palette, lighting, texture and mood exactly."
            )
        else:
            clauses.append(
                f"This is a completely new scene: do not show anything depicted in {which}. It is only a style "
                "guide, so copy just its visual look, color palette, lighting and mood."
            )
    base = text.strip()
    if base and base[-1] not in ".!?":
        base += "."
    return " ".join([base, *clauses]).strip()


def _closing(text: str, sentence: str) -> str:
    """``text`` ending with ``sentence`` (how a model without a negative prompt is told what to avoid)."""
    if not sentence:
        return text
    base = text.rstrip().rstrip(".")
    return f"{base}. {sentence}" if base else sentence


# --------------------------------------------------------------------------------------
# The planner
# --------------------------------------------------------------------------------------


class Planner:
    def __init__(self, catalog: Catalog, config: Config, rng: random.Random | None = None):
        self.catalog = catalog
        self.config = config
        self.loras = lora_library.Library(config.lora_dir, catalog.registry)
        self._rng = rng or random.SystemRandom()

    # ----------------------------------------------------------------- helpers ----

    def _seeds(self, params: Params, count: int) -> list[int]:
        if params.sameseed is not None:
            return [params.sameseed] * count
        base = params.seed if params.seed is not None else self._rng.randint(0, MAX_SEED)
        return [(base + offset) % (MAX_SEED + 1) for offset in range(count)]

    def _weights(self, spec: ModelSpec) -> tuple[str | None, int | None]:
        """(weights location, on-the-fly quantization) for a model."""
        override = self.config.models.get(spec.id)
        found = self.catalog.weights(spec)
        location = found.location if found else None
        quantize = override.quantize if override else None
        if found is not None and found.quant is not None:
            quantize = None  # already quantized on disk
        return location, quantize

    def _steps(self, spec: ModelSpec, params: Params, notes: list[str]) -> tuple[int, float]:
        """(steps, resolution scale) after ``--steps`` / ``--quality``."""
        override = self.config.models.get(spec.id)
        base = override.steps if override and override.steps else spec.steps
        if params.steps is not None:
            return params.steps, 1.0
        quality = params.quality
        if quality is None or quality == 1:
            return base, 1.0
        if not spec.distilled:
            return max(8, round(base * quality)), 1.0
        if quality > 1:
            return min(base * 2, round(base * quality)), 1.0
        # A distilled model needs its trained step count; trade resolution for speed instead.
        notes.append(f"`--quality {quality:g}` renders smaller on {spec.title} (its step count is fixed).")
        return base, math.sqrt(quality)

    def _guidance(self, spec: ModelSpec, params: Params) -> float | None:
        if params.guidance is not None:
            return params.guidance
        override = self.config.models.get(spec.id)
        return override.guidance if override and override.guidance is not None else None

    def _avoid(
        self,
        spec: ModelSpec,
        terms: list[str],
        free_text: str | None,
        guidance: float | None,
        notes: list[str],
    ) -> tuple[str | None, float | None, str]:
        """Apply things-to-avoid. Returns (negative prompt, guidance, closing sentence).

        A model with classifier-free guidance takes a real negative prompt. Any other model is
        told in the prompt itself, in a sentence that closes it, which is best-effort.
        """
        wanted = [*terms, *([free_text] if free_text else [])]
        if not wanted:
            return None, guidance, ""
        if spec.cfg:
            effective = guidance if guidance is not None else spec.guidance
            if effective is None or effective <= 1.0:
                # Classifier-free guidance only runs (and only then reads the negative prompt) above 1.
                guidance = 3.0
            return ", ".join(wanted), guidance, ""
        notes.append(
            f"{spec.title} has no negative prompt, so `--no` is best-effort. A model with guidance honours it fully."
        )
        return None, guidance, f"Without {', '.join(terms or wanted)}."

    def _loras(self, requested: Sequence[LoraRef], allow_paths: bool = False) -> list[Fitted]:
        """The LoRAs asked for with ``--lora``. ``--lora none`` asks for none at all."""
        if any(lora.name.lower() in lora_library.NONE for lora in requested):
            return []
        return [self._lora(lora, allow_paths) for lora in requested]

    def _lora(self, wanted: LoraRef, allow_paths: bool) -> Fitted:
        """One ``--lora`` name as a file.

        A name is a LoRA of the library, or a Hugging Face repository that is already in the
        cache. A path is only taken from the Mac itself: a prompt that arrives over chat must
        not be able to point Dreambot at any file on it.
        """
        name = wanted.name
        if allow_paths:
            direct = Path(name).expanduser()
            if direct.is_file():
                return Fitted(direct.stem, str(direct), wanted.scale)
        try:
            entry = self.loras.find(name)
        except lora_library.LoraError as exc:
            raise PlanError(f"The LoRA `{name}` can't be used: {exc}.") from exc
        if entry is not None:
            # The scale typed is a share of the strength the LoRA was added with.
            return Fitted(entry.name, entry.path, wanted.scale * entry.scale, entry.base, entry.trigger)
        if ".." not in name and not name.startswith(".") and re.fullmatch(r"[\w.-]+/[\w.-]+", name):
            return Fitted(name, name, wanted.scale)  # a Hugging Face repository; it must be in the cache
        if not allow_paths and not re.fullmatch(r"[\w-][\w.-]*", name):
            raise PlanError("`--lora` takes the name of a LoRA; file paths only work from the Mac itself.")
        available = [entry.name for entry in self.loras.entries()[0]]
        if available:
            listing = f" Available: {', '.join(available)}."
        elif allow_paths:
            listing = (
                f" Add one with `dreambot loras add`, or put .safetensors files in {self.config.lora_dir}."
            )
        else:
            listing = " None are installed on the Mac yet."
        raise PlanError(f"I can't find a LoRA called `{name}`.{listing}")

    def _lora_model(
        self, loras: Sequence[Fitted], typed: Params, settings: ResolvedSettings, notes: list[str]
    ) -> Params:
        """What the person typed, with the model their LoRA was trained on chosen for them.

        A LoRA is changes to one model's weights and means nothing on another. So naming a
        LoRA names its model too, unless the person names a model themselves — and then the two
        have to be of one family.
        """
        known = [
            (lora, spec) for lora in loras if lora.base and (spec := self.catalog.registry.find(lora.base))
        ]
        if not known:
            return typed
        if typed.model is not None:
            self._fitting(self.catalog.lookup(typed.model), loras, notes, strict=True)
            return typed
        (first, model), *others = known
        for lora, spec in others:
            if spec.family != model.family:
                raise PlanError(
                    f"`{first.name}` and `{lora.name}` were trained on different models ({model.title} and "
                    f"{spec.title}), so they can't be used together."
                )
        ready = next((spec for _, spec in known if self.catalog.is_installed(spec)), None)
        if ready is None:
            raise PlanError(
                f"`{first.name}` is a LoRA for {model.title}, which isn't installed. "
                f"Run `dreambot models pull {model.id}`."
            )
        theirs = self.catalog.registry.find(settings.model) if settings.model else None
        if theirs is not None and theirs.id != ready.id:
            notes.append(
                f"`{first.name}` is a LoRA for {ready.title}, so this ran on it rather than on {theirs.title}."
            )
        return replace(typed, model=ready.id)

    def _fitting(
        self, spec: ModelSpec, loras: Sequence[Fitted], notes: list[str], *, strict: bool
    ) -> list[Fitted]:
        """The LoRAs that can go on ``spec``: those trained on a model of its family, or on nobody knows what.

        One that cannot is refused when the person asked for it here (``strict``), and left out
        with a note when it only came along from an earlier job or from their settings.
        """
        kept: list[Fitted] = []
        for lora in loras:
            base = self.catalog.registry.find(lora.base) if lora.base else None
            if base is None or base.family == spec.family:
                kept.append(lora)
            elif strict:
                raise PlanError(
                    f"`{lora.name}` is a LoRA for {base.title}; it can't be used with {spec.title}. "
                    f"Leave `--model` out to run it on {base.title}."
                )
            else:
                notes.append(
                    f"`{lora.name}` is a LoRA for {base.title}, so {spec.title} made this without it."
                )
        return kept

    def _require(self, spec: ModelSpec, op: Op) -> None:
        if not spec.supports(op):
            capable = ", ".join(s.id for s in self.catalog.installed(op)) or "none installed"
            raise PlanError(f"{spec.title} can't do this ({op.value}). Models that can: {capable}.")
        if not self.catalog.is_installed(spec):
            raise PlanError(f"{spec.title} isn't installed. Run `dreambot models pull {spec.id}`.")

    def _grouped(
        self,
        base: Render,
        prompts: list[str],
        seeds: list[int],
        boards: Sequence[Board | None] | None = None,
    ) -> list[Render]:
        """Split one render into as few requests as possible when images differ in prompt or style board."""
        groups: dict[tuple[str, Board | None], Render] = {}
        for index, (prompt, seed) in enumerate(zip(prompts, seeds, strict=True)):
            board = boards[index] if boards else None
            if (prompt, board) not in groups:
                groups[prompt, board] = Render(
                    **{**base.__dict__, "prompt": prompt, "seeds": [], "slots": [], "board": board}
                )
            groups[prompt, board].seeds.append(seed)
            groups[prompt, board].slots.append(index + 1)
        return list(groups.values())

    # ----------------------------------------------------------------- imagine ----

    def imagine(
        self,
        parsed: ParsedPrompt,
        settings: ResolvedSettings,
        refs: ResolvedRefs | None = None,
        *,
        directions: Sequence[str] | None = None,
        seeds: Sequence[int] | None = None,
        allow_paths: bool = False,
    ) -> Plan:
        """Plan a prompt.

        The result can be rendered as it is. When it carries a ``brief``, a language model may
        first write a line of art direction for each image; planning again with those
        ``directions`` (and the same ``seeds``) then builds the prompts from them. An empty
        direction leaves that image with the rule-based emulation in :mod:`dreambot.core.styles`.

        ``allow_paths`` says the prompt was typed on the Mac itself, where ``--lora`` may name
        a file anywhere on it.
        """
        typed = parsed.params.loras
        if typed or not settings.loras:
            return self._imagine(parsed, settings, refs, typed, directions, seeds, allow_paths)
        # The person's default LoRAs go on every prompt they fit. One they do not fit — an edit,
        # when they were trained on a model that only draws — runs without them rather than not at all.
        try:
            usual = parse_loras(" ".join(settings.loras))
            return self._imagine(parsed, settings, refs, usual, directions, seeds, False)
        except PlanError as exc:
            plan = self._imagine(parsed, settings, refs, [], directions, seeds, allow_paths)
            plan.notes.append(f"Your default LoRA was left out of this one. ({exc})")
            return plan

    def _imagine(
        self,
        parsed: ParsedPrompt,
        settings: ResolvedSettings,
        refs: ResolvedRefs | None,
        loras: Sequence[LoraRef],
        directions: Sequence[str] | None,
        seeds: Sequence[int] | None,
        allow_paths: bool,
    ) -> Plan:
        refs = refs or ResolvedRefs()
        notes: list[str] = []
        fitted = self._loras(loras, allow_paths)
        typed = self._lora_model(fitted, parsed.params, settings, notes)
        params = typed.merged_over(settings.as_params())
        text = parsed.positive_text
        if not text and not refs.images:
            raise PlanError("Give me something to imagine: a description, an image, or both.")

        for name in parsed.planned:
            notes.append(f"`--{name}` isn't available yet and was ignored.")

        version = style_version(params.sv, notes)
        style_weight = 100 if params.sw is None else params.sw
        stylize = params.stylize if params.stylize is not None else settings.stylize
        count = params.count or settings.count
        seeds = list(seeds) if seeds else self._seeds(params, count)

        # From style version 2 every style ingredient is a fingerprint first. What a model that
        # takes reference images is shown is a board of swatches cut from the pictures, never
        # the pictures themselves: shown whole, their content leaks into the result.
        styled = version != STYLE_LEGACY and bool(refs.sources or refs.style or refs.style_codes)
        sources: list[StyleSource] = []
        looks: list[Look] = []
        overall = Look(None, [], None)
        shown, choosing = refs, typed
        if styled:
            sources = refs.style_sources()
            boarded, choosing = self._board_model(typed, settings, refs, sources, version, notes)
            looks, overall = self._looks(sources, seeds[0], count, style_weight, stylize, boarded)
            shown = replace(refs, style=[BOARD] if overall.board is not None else [])

        try:
            op, spec = self._imagine_model(choosing, params, settings, shown, notes)
        except PlanError as exc:
            if typed.model is not None and parsed.params.model is None:
                # The model was chosen for the LoRA, not by the person: say why it is in the way.
                held = self.catalog.lookup(typed.model)
                raise PlanError(
                    f"`{fitted[0].name}` only works on {held.title}, the model it was trained on. {exc}"
                ) from exc
            raise
        steps, quality_scale = self._steps(spec, params, notes)
        scale = quality_scale * (0.5 if params.draft else 1.0)

        # A style lends its look, not its shape (in the original style version it did both).
        shaping = shown.content_images if styled else shown.images
        if params.aspect is None and op is not Op.TXT2IMG and shaping:
            # Like Midjourney's edit model: follow the first image's shape unless told otherwise.
            with Image.open(shaping[0]) as first:
                width, height = working_size(
                    first.width, first.height, native=spec.native, multiple=spec.multiple
                )
            width, height = snap(width * scale, spec.multiple), snap(height * scale, spec.multiple)
        else:
            width, height = dimensions(params.aspect, multiple=spec.multiple, scale=scale, native=spec.native)

        guidance = self._guidance(spec, params)
        negative, guidance, without = self._avoid(
            spec, parsed.negative_terms, params.negative, guidance, notes
        )
        location, quantize = self._weights(spec)

        style_codes = [] if styled else [(code, style_weight) for code in refs.style_codes]
        raw = bool(params.raw)
        chaos, weird, exp = params.chaos or 0, params.weird or 0, params.exp or 0

        brief: enhance.Brief | None = None
        if settings.enhance and op is Op.TXT2IMG and not shown.images:
            # Only a plain text prompt is art-directed. With images attached the text is an
            # instruction, or a description that has to agree with a picture the model cannot see.
            candidate = enhance.Brief(
                # With a LoRA's trigger words: they usually name its look ("pixel art style"),
                # and the art director then keeps to it as to any other medium the prompt names.
                text=triggered(text, fitted),
                count=count,
                stylize=0 if raw else stylize,
                chaos=chaos,
                weird=weird,
                exp=exp,
                style=(
                    *(overall.phrases if styled else ()),
                    *(phrase for code, weight in style_codes for phrase in styles.style_recipe(code, weight)),
                ),
                avoid=tuple(parsed.negative_terms),
            )
            if enhance.wanted(candidate):
                brief = candidate

        directed: list[str] = []
        for index in range(count):
            direction = directions[index] if directions and index < len(directions) else ""
            if direction:
                # The language model has covered stylize, chaos, weird and exp; a style code's
                # recipe is still added by rule, so that a code always means the same look.
                prompt = styles.direct(
                    enhance.compose(text, direction),
                    seed=seeds[0],
                    index=index,
                    count=count,
                    raw=True,
                    style_codes=style_codes,
                    style=looks[index].phrases if looks else (),
                )
            else:
                prompt = styles.direct(
                    text,
                    seed=seeds[0],
                    index=index,
                    count=count,
                    chaos=chaos,
                    weird=weird,
                    exp=exp,
                    stylize=stylize,
                    raw=raw,
                    style_codes=style_codes,
                    style=looks[index].phrases if looks else (),
                )
            directed.append(_closing(prompt, without))
        text = _closing(text, without)

        base = Render(
            op=op,
            model=spec.id,
            prompt="",
            seeds=[],
            slots=[],
            width=width,
            height=height,
            steps=steps,
            guidance=guidance,
            negative=negative,
            loras=[lora.lora for lora in fitted],
            weights=location,
            quantize=quantize,
            stop_at=params.stop / 100 if params.stop is not None and params.stop < 100 else None,
        )
        boards: list[Board | None] | None = None
        if op is Op.EDIT and styled:
            base.images = shown.content_images
            boards = [look.board for look in looks]
            strength = style_weight if any(source.kind != "look" for source in sources) else stylize
            directed = [
                reference_instruction(prompt, shown, params, strength=strength, board=look.board is not None)
                for prompt, look in zip(directed, looks, strict=True)
            ]
        elif op is Op.EDIT:
            base.images = shown.images
            directed = [reference_instruction(prompt, shown, params) for prompt in directed]
        elif op is Op.REDUX:
            base.images = refs.image_prompts
            # Redux strengths run 0..1; --iw above 1 cannot push past "follow the image fully".
            weight = 1.0 if params.iw is None else params.iw
            base.image_weights = [min(1.0, weight)] * len(refs.image_prompts)
        elif op is Op.IMG2IMG:
            base.image = refs.image_prompts[0]
            if params.strength is not None:
                # Said outright: how much of the picture to keep.
                base.strength = params.strength
            else:
                weight = 1.0 if params.iw is None else params.iw
                # A heavier image weight keeps more of the picture, i.e. starts from less noise.
                base.noise = max(0.4, min(0.95, 0.95 - 0.15 * weight))
            if len(refs.image_prompts) > 1:
                notes.append(
                    "Only the first image prompt was used; install an edit model to combine several."
                )
        directed = [triggered(prompt, fitted) for prompt in directed]

        plan = Plan(
            model=spec.id,
            op=op,
            text=text,
            negative=negative,
            width=width,
            height=height,
            seeds=seeds,
            renders=self._grouped(base, directed, seeds, boards),
            brief=None if directions else brief,
            notes=notes,
        )
        if styled and (detail := self._style_detail(sources, overall, version, style_weight, stylize)):
            plan.detail["style"] = detail
        if fitted:
            plan.detail["loras"] = [lora.record() for lora in fitted]
        if params.hd:
            self._add_hd(plan)
        return plan

    # ------------------------------------------------------------------- style ----

    def _board_model(
        self,
        typed: Params,
        settings: ResolvedSettings,
        refs: ResolvedRefs,
        sources: list[StyleSource],
        version: int,
        notes: list[str],
    ) -> tuple[bool, Params]:
        """Whether the model that will run a prompt can be shown a style board.

        Returns that, and the typed parameters to choose the model with. A style goes along with
        the model a person chose — typed, or as their default — rather than moving their job to
        another one: where that model cannot take a reference image, the style is applied in
        words alone.
        """
        pictured = any(source.weight > 0 and (source.pictures or source.members) for source in sources)
        if version != STYLE_BLEND or not pictured:
            return False, typed
        content = refs.content_images
        pinned = typed
        chosen = self.catalog.registry.find(settings.model) if settings.model else None
        if typed.model is not None:
            spec = self.catalog.lookup(typed.model)
            if not spec.supports(Op.EDIT):
                notes.append(
                    f"{spec.title} can't be shown reference images, so the style is applied in words."
                )
                return False, typed
        elif not content and chosen is not None and self.catalog.is_installed(chosen):
            if not chosen.supports(Op.EDIT):
                return False, typed
            spec, pinned = chosen, replace(typed, model=chosen.id)
        else:
            found = self.catalog.pick(self.config.prefer.edit, Op.EDIT) or next(
                iter(self.catalog.installed(Op.EDIT)), None
            )
            if found is None:
                notes.append(
                    "The style is applied in words. An edit model can be shown its pictures as well: "
                    "`dreambot models pull flux2-klein-4b`."
                )
                return False, typed
            spec = found
        limit = spec.variants[Op.EDIT].max_refs
        if len(content) + 1 > limit:
            if len(content) <= limit:
                notes.append(
                    f"{spec.title} takes {limit} reference image(s) and the others fill them, so the style "
                    "is applied in words."
                )
            return False, typed
        return True, pinned

    def _looks(
        self,
        sources: list[StyleSource],
        seed: int,
        count: int,
        style_weight: int,
        stylize: int,
        boarded: bool,
    ) -> tuple[list[Look], Look]:
        """The look of each image of a grid, and the look of the job as a whole.

        Style references (``--sref``) give every image the same look, scaled by ``--sw``. A
        Lookbook gives each image its own: one of its pictures leads (see
        :func:`dreambot.core.lookbook.cast`), scaled by ``--stylize``. With both, the reference
        decides the medium and the Lookbook colours it.
        """
        references = [s for s in sources if s.kind != "look" and s.weight > 0 and s.print is not None]
        books = [s for s in sources if s.kind == "look" and s.weight > 0 and s.members]
        if style_weight <= 0:
            references = []
        if stylize <= 0:
            books = []
        reference_blend = fingerprint.blend([(source.print, source.weight) for source in references])
        reference_tiles = [
            (path, source.weight / len(source.pictures)) for source in references for path in source.pictures
        ]

        def look_of(book_parts: list[fingerprint.Part], book_tiles: list[tuple[str, float]]) -> Look:
            book_blend = fingerprint.blend(book_parts)
            if reference_blend is not None:
                phrases = [
                    *fingerprint.phrases(reference_blend, style_weight),
                    *fingerprint.accents(book_blend, stylize),
                ]
            else:
                phrases = fingerprint.phrases(book_blend, stylize)
            seen: set[str] = set()
            phrases = [
                phrase for phrase in phrases if not (phrase.lower() in seen or seen.add(phrase.lower()))
            ]
            board: Board | None = None
            # Between a style reference and a Lookbook, the stronger dial gets more of the board.
            tiles = [*_scaled(reference_tiles, style_weight), *_scaled(book_tiles, stylize)]
            if boarded and tiles:
                board = Board(tuple(tiles), seed)
            return Look(reference_blend if reference_blend is not None else book_blend, phrases, board)

        looks: list[Look] = []
        for index in range(count):
            parts: list[fingerprint.Part] = []
            tiles: list[tuple[str, float]] = []
            for book in books:
                weights = lookbook.cast(len(book.members), seed, index)
                scale = book.weight / sum(weights)
                for member, weight in zip(book.members, weights, strict=True):
                    parts.append((member.print, weight * scale))
                    tiles.append((member.path, weight * scale))
            looks.append(look_of(parts, tiles))
        overall = look_of(
            [(member.print, book.weight / len(book.members)) for book in books for member in book.members],
            [(member.path, book.weight / len(book.members)) for book in books for member in book.members],
        )
        return looks, overall

    @staticmethod
    def _style_detail(
        sources: list[StyleSource], overall: Look, version: int, style_weight: int, stylize: int
    ) -> dict[str, Any]:
        """What a job records about its style, for `info` and the web app."""
        if overall.print is None:
            return {}
        total = sum(source.weight for source in sources if source.weight > 0) or 1.0
        return {
            "version": version,
            "name": overall.print.name,
            "summary": fingerprint.summary(overall.print),
            "weight": style_weight,
            "stylize": stylize,
            "swatches": [swatch.hex for swatch in overall.print.swatches],
            "sources": [
                {
                    "kind": source.kind,
                    "label": source.label,
                    "share": round(max(0.0, source.weight) / total, 3),
                    **({"code": source.code} if source.code else {}),
                }
                for source in sources
            ],
            "board": overall.board is not None,
        }

    def _imagine_model(
        self, typed: Params, params: Params, settings: ResolvedSettings, refs: ResolvedRefs, notes: list[str]
    ) -> tuple[Op, ModelSpec]:
        prefer = self.config.prefer
        needs_edit = bool(refs.edit or refs.subject or refs.style)
        has_image_prompts = bool(refs.image_prompts)

        if typed.model is not None:
            spec = self.catalog.lookup(typed.model)
            if needs_edit or (has_image_prompts and spec.supports(Op.EDIT)):
                op = Op.EDIT
            elif has_image_prompts:
                op = Op.REDUX if spec.supports(Op.REDUX) else Op.IMG2IMG
            else:
                op = Op.TXT2IMG
            self._require(spec, op)
            self._check_reference_count(spec, op, refs)
            return op, spec

        if needs_edit or has_image_prompts:
            spec = self.catalog.pick(prefer.edit, Op.EDIT)
            if spec is not None and len(refs.images) <= spec.variants[Op.EDIT].max_refs:
                return Op.EDIT, spec
            if needs_edit:
                wanted = max(1, len(refs.images))
                capable = [
                    s for s in self.catalog.installed(Op.EDIT) if s.variants[Op.EDIT].max_refs >= wanted
                ]
                if capable:
                    return Op.EDIT, capable[0]
                if spec is not None:
                    limit = spec.variants[Op.EDIT].max_refs
                    raise PlanError(
                        f"{spec.title} takes at most {limit} reference image(s); you gave {len(refs.images)}."
                    )
                raise PlanError(
                    "Image references and edits need an edit model. "
                    "Install one with `dreambot models pull flux2-klein-4b`."
                )
            # Image prompts only: blending model, then plain image-to-image as a last resort.
            spec = self.catalog.pick(prefer.blend, Op.REDUX)
            if spec is not None:
                return Op.REDUX, spec
            spec = self._default_model(params, settings, Op.IMG2IMG, notes)
            notes.append(
                "No edit model is installed, so the image prompt only seeds the composition. "
                "`dreambot models pull flux2-klein-4b` gives real image prompts."
            )
            return Op.IMG2IMG, spec

        return Op.TXT2IMG, self._default_model(params, settings, Op.TXT2IMG, notes)

    def _check_reference_count(self, spec: ModelSpec, op: Op, refs: ResolvedRefs) -> None:
        if op is Op.EDIT:
            limit = spec.variants[Op.EDIT].max_refs
            if len(refs.images) > limit:
                raise PlanError(
                    f"{spec.title} takes at most {limit} reference image(s); you gave {len(refs.images)}."
                )

    def _default_model(
        self, params: Params, settings: ResolvedSettings, op: Op, notes: list[str]
    ) -> ModelSpec:
        prefer = self.config.prefer
        if params.niji is not None:
            spec = self.catalog.pick(prefer.niji, op)
            if spec is None:
                raise PlanError(
                    "`--niji` needs an anime model. Install one with `dreambot models pull anima-turbo`."
                )
            return spec
        if params.draft:
            spec = self.catalog.pick(prefer.draft, op)
            if spec is not None:
                return spec
        if settings.model:
            spec = self.catalog.registry.find(settings.model)
            if spec is not None and spec.supports(op) and self.catalog.is_installed(spec):
                return spec
            if spec is not None and op is Op.TXT2IMG:
                notes.append(f"Your default model {spec.title} isn't available; using another.")
        spec = self.catalog.any_with(op, prefer.imagine)
        if spec is None:
            raise PlanError(
                "No image model is installed yet. Run `dreambot models pull z-image-turbo` to get started."
            )
        return spec

    def default_model(self, settings: ResolvedSettings, op: Op = Op.TXT2IMG) -> ModelSpec:
        """The model a plain prompt from someone with these settings would use."""
        return self._default_model(settings.as_params(), settings, op, [])

    def _add_hd(self, plan: Plan) -> None:
        upscaler = self.catalog.pick(self.config.prefer.upscale, Op.UPSCALE)
        if upscaler is None:
            plan.notes.append(
                "`--hd` needs an upscaler: run `dreambot models pull seedvr2-3b`. Delivered at standard size."
            )
            return
        location, quantize = self._weights(upscaler)
        plan.finish = Render(
            op=Op.UPSCALE,
            model=upscaler.id,
            prompt="",
            seeds=[0],
            slots=[0],
            weights=location,
            quantize=quantize,
            extra={"scale": 2},
        )
        plan.detail["hd"] = True

    # --------------------------------------------------------------- variations ----

    def variation(
        self,
        *,
        source: str,
        text: str,
        model: str,
        width: int,
        height: int,
        mode: str,
        count: int,
        negative: str | None = None,
        params: Params | None = None,
        loras: Sequence[Fitted] = (),
    ) -> Plan:
        """New takes on one image: same prompt (unless remixed), keeps the composition.

        ``loras`` are the ones the image was made with. They are used again, unless a remixed
        prompt names LoRAs of its own.
        """
        params = params or Params()
        notes: list[str] = []
        spec = self.catalog.registry.find(model)
        if spec is None or not spec.supports(Op.IMG2IMG) or not self.catalog.is_installed(spec):
            fallback = self.catalog.any_with(Op.IMG2IMG, self.config.prefer.variation)
            if fallback is None:
                raise PlanError(
                    "No installed model can make variations. `dreambot models pull z-image-turbo` adds one."
                )
            if spec is not None:
                notes.append(f"{spec.title} can't make variations; {fallback.title} did these instead.")
            spec = fallback
        if params.model is not None:
            spec = self.catalog.lookup(params.model)
            self._require(spec, Op.IMG2IMG)

        if params.loras:
            fitted = self._fitting(spec, self._loras(params.loras), notes, strict=True)
        else:
            fitted = self._fitting(spec, loras, notes, strict=False)
        steps, _ = self._steps(spec, params, notes)
        guidance = self._guidance(spec, params)
        seeds = self._seeds(params, count)
        location, quantize = self._weights(spec)
        if params.aspect is not None:
            width, height = dimensions(params.aspect, multiple=spec.multiple, native=spec.native)
        render = Render(
            op=Op.IMG2IMG,
            model=spec.id,
            prompt=triggered(text, fitted),
            seeds=seeds,
            slots=list(range(1, count + 1)),
            width=snap(width, spec.multiple),
            height=snap(height, spec.multiple),
            steps=steps,
            guidance=guidance,
            negative=negative if spec.cfg else None,
            image=source,
            # `--strength` is the raw per-model control; without it, aim for a noise level.
            strength=params.strength,
            noise=None if params.strength is not None else VARIATION_NOISE[mode],
            loras=[lora.lora for lora in fitted],
            weights=location,
            quantize=quantize,
        )
        detail: dict[str, Any] = {"mode": mode, "noise": render.noise, "strength": render.strength}
        if fitted:
            detail["loras"] = [lora.record() for lora in fitted]
        return Plan(
            model=spec.id,
            op=Op.IMG2IMG,
            text=text,
            negative=render.negative,
            width=render.width or width,
            height=render.height or height,
            seeds=seeds,
            renders=[render],
            source=source,
            notes=notes,
            detail=detail,
        )

    # ------------------------------------------------------------------ upscale ----

    def upscale(
        self,
        *,
        source: str,
        text: str,
        model: str,
        creative: bool,
        enlarged: str | None = None,
        tiles: list[Tile] | None = None,
        loras: Sequence[Fitted] = (),
    ) -> Plan:
        """Double an image's size.

        Subtle uses a dedicated super-resolution model and stays faithful to the source.

        Creative re-renders the enlarged image so the model invents new fine detail. That is
        done tile by tile at the model's native resolution — rendering a 4-megapixel image in
        one piece pushes the noise schedule so far that the picture would be redrawn rather
        than refined. ``enlarged`` is the plainly enlarged source and ``tiles`` its tiles, and
        ``loras`` are the ones the image was made with: new detail is drawn in the same look.
        """
        notes: list[str] = []
        with Image.open(source) as image:
            width, height = image.size
        factor = min(2.0, MAX_UPSCALED_SIDE / max(width, height))
        if factor < 1.2:
            raise PlanError("This image is already as large as Dreambot upscales.")

        upscaler = self.catalog.pick(self.config.prefer.upscale, Op.UPSCALE)
        if not creative and upscaler is not None:
            location, quantize = self._weights(upscaler)
            render = Render(
                op=Op.UPSCALE,
                model=upscaler.id,
                prompt="",
                seeds=[self._rng.randint(0, MAX_SEED)],
                slots=[1],
                image=source,
                weights=location,
                quantize=quantize,
                extra={"scale": round(factor, 3)},
            )
            return Plan(
                model=upscaler.id,
                op=Op.UPSCALE,
                text=text,
                width=round(width * factor),
                height=round(height * factor),
                seeds=render.seeds,
                renders=[render],
                source=source,
                notes=notes,
                detail={"mode": "subtle", "factor": factor},
            )

        noise = CREATIVE_UPSCALE_NOISE
        if not creative:
            noise = FALLBACK_UPSCALE_NOISE
            notes.append(
                "No dedicated upscaler is installed, so fine detail was re-rendered rather than strictly "
                "preserved. `dreambot models pull seedvr2-3b` adds a faithful upscaler."
            )
        spec = self.catalog.registry.find(model)
        if spec is None or not spec.supports(Op.IMG2IMG) or not self.catalog.is_installed(spec):
            spec = self.catalog.any_with(Op.IMG2IMG, self.config.prefer.variation)
        if spec is None:
            raise PlanError(
                "No installed model can upscale. `dreambot models pull seedvr2-3b` adds a dedicated upscaler."
            )
        if enlarged is None or not tiles:
            raise PlanError("A re-rendered upscale needs the enlarged source image and its tiles.")
        location, quantize = self._weights(spec)
        fitted = self._fitting(spec, loras, notes, strict=False)
        with Image.open(enlarged) as big:
            target_w, target_h = big.size
        seed = self._rng.randint(0, MAX_SEED)
        renders = [
            Render(
                op=Op.IMG2IMG,
                model=spec.id,
                prompt=triggered(text, fitted),
                seeds=[seed],
                slots=[number],
                width=tile.box[2] - tile.box[0],
                height=tile.box[3] - tile.box[1],
                steps=spec.steps,
                image=str(tile.path),
                noise=noise,
                loras=[lora.lora for lora in fitted],
                weights=location,
                quantize=quantize,
            )
            for number, tile in enumerate(tiles, start=1)
        ]
        detail: dict[str, Any] = {
            "mode": "creative" if creative else "subtle",
            "factor": factor,
            "tiles": len(tiles),
        }
        if fitted:
            detail["loras"] = [lora.record() for lora in fitted]
        return Plan(
            model=spec.id,
            op=Op.IMG2IMG,
            text=text,
            width=target_w,
            height=target_h,
            seeds=[seed],
            renders=renders,
            source=source,
            tiles=list(tiles),
            tile_base=enlarged,
            notes=notes,
            detail=detail,
        )

    # ---------------------------------------------------------- repaint / extend ----

    def outpaint(
        self, *, source: str, canvas: Canvas, text: str, count: int, params: Params | None = None
    ) -> Plan:
        """Fill the empty part of a canvas around an image (zoom out, pan, make square)."""
        params = params or Params()
        notes: list[str] = []
        width, height = canvas.size
        seeds = self._seeds(params, count)
        slots = list(range(1, count + 1))

        spec = self.catalog.pick(self.config.prefer.inpaint, Op.INPAINT)
        if params.model is not None:
            spec = self.catalog.lookup(params.model)
            if not (spec.supports(Op.INPAINT) or spec.supports(Op.EDIT)):
                self._require(spec, Op.INPAINT)
        scene = f" The scene: {text.strip()}" if text.strip() else ""
        # Wording and the flat gray canvas were chosen by testing on real models: this form makes
        # an instruction-following model extend the picture and leave the original where it is.
        instruction = (
            "Fill the gray area around the photo: extend the scene naturally so the photo becomes "
            "part of a larger, seamless picture. No borders or frames." + scene
        )
        if spec is not None and spec.supports(Op.INPAINT) and self.catalog.is_installed(spec):
            location, quantize = self._weights(spec)
            render = Render(
                op=Op.INPAINT,
                model=spec.id,
                # Caption-style inpainters want the finished picture described; instructed ones
                # want to be told what to do.
                prompt=instruction if spec.variants[Op.INPAINT].instructed else text,
                seeds=seeds,
                slots=slots,
                width=width,
                height=height,
                steps=self._steps(spec, params, notes)[0],
                guidance=self._guidance(spec, params),
                image=str(canvas.image),
                mask=str(canvas.mask),
                weights=location,
                quantize=quantize,
            )
        else:
            editor = (
                spec
                if spec is not None and spec.supports(Op.EDIT)
                else self.catalog.pick(self.config.prefer.edit, Op.EDIT)
            )
            if editor is None or not self.catalog.is_installed(editor):
                raise PlanError(
                    "Extending an image needs an inpainting or edit model. "
                    "Install one with `dreambot models pull flux2-klein-4b`."
                )
            location, quantize = self._weights(editor)
            render = Render(
                op=Op.EDIT,
                model=editor.id,
                prompt=instruction,
                seeds=seeds,
                slots=slots,
                width=width,
                height=height,
                steps=self._steps(editor, params, notes)[0],
                guidance=self._guidance(editor, params),
                images=[str(canvas.image)],
                weights=location,
                quantize=quantize,
            )
            spec = editor
        return Plan(
            model=spec.id,
            op=render.op,
            text=text,
            width=width,
            height=height,
            seeds=seeds,
            renders=[render],
            canvas=canvas,
            source=source,
            notes=notes,
        )

    def region(
        self,
        *,
        source: str,
        mask: str | None,
        text: str,
        where: str | None,
        count: int,
        params: Params | None = None,
    ) -> Plan:
        """Repaint part of an image (Vary Region).

        With a ``mask`` and an inpainting model this is true inpainting. Otherwise an edit model
        is told in words which part to change (``where``), and — when a mask is known — the
        rest of the picture is restored from the source afterwards.
        """
        params = params or Params()
        notes: list[str] = []
        with Image.open(source) as image:
            source_w, source_h = image.size
        seeds = self._seeds(params, count)
        slots = list(range(1, count + 1))

        inpainter = self.catalog.pick(self.config.prefer.inpaint, Op.INPAINT) if mask else None
        if inpainter is not None:
            width, height = working_size(
                source_w, source_h, native=inpainter.native, multiple=inpainter.multiple
            )
            location, quantize = self._weights(inpainter)
            instructed = inpainter.variants[Op.INPAINT].instructed
            render = Render(
                op=Op.INPAINT,
                model=inpainter.id,
                prompt=f"Replace the masked area with: {text.strip()}" if instructed else text,
                seeds=seeds,
                slots=slots,
                width=width,
                height=height,
                steps=self._steps(inpainter, params, notes)[0],
                guidance=self._guidance(inpainter, params),
                image=source,
                mask=mask,
                weights=location,
                quantize=quantize,
            )
            spec = inpainter
        else:
            editor = self.catalog.pick(self.config.prefer.edit, Op.EDIT)
            if editor is None:
                raise PlanError(
                    "Changing part of an image needs an inpainting or edit model. "
                    "Install one with `dreambot models pull flux2-klein-4b`."
                )
            width, height = working_size(source_w, source_h, native=editor.native, multiple=editor.multiple)
            location, quantize = self._weights(editor)
            target = f" in {where}" if where else ""
            render = Render(
                op=Op.EDIT,
                model=editor.id,
                prompt=f"Change only this{target}: {text.strip()}. Keep everything else in the image exactly the same.",
                seeds=seeds,
                slots=slots,
                width=width,
                height=height,
                steps=self._steps(editor, params, notes)[0],
                guidance=self._guidance(editor, params),
                images=[source],
                weights=location,
                quantize=quantize,
            )
            spec = editor
        return Plan(
            model=spec.id,
            op=render.op,
            text=text,
            width=width,
            height=height,
            seeds=seeds,
            renders=[render],
            keep_outside_mask=mask,
            source=source,
            notes=notes,
        )

    # -------------------------------------------------------------------- edit ----

    def edit(self, *, sources: list[str], instruction: str, count: int, params: Params | None = None) -> Plan:
        """Apply a written instruction to one or more images."""
        refs = ResolvedRefs(edit=list(sources))
        params = params or Params()
        notes: list[str] = []
        if params.model is not None:
            spec = self.catalog.lookup(params.model)
            self._require(spec, Op.EDIT)
        else:
            spec = self.catalog.pick(self.config.prefer.edit, Op.EDIT)
            if spec is None:
                raise PlanError(
                    "Editing needs an edit model. Install one with `dreambot models pull flux2-klein-4b`."
                )
        self._check_reference_count(spec, Op.EDIT, refs)
        with Image.open(sources[0]) as first:
            width, height = working_size(
                first.width, first.height, native=spec.native, multiple=spec.multiple
            )
        if params.aspect is not None:
            width, height = dimensions(params.aspect, multiple=spec.multiple, native=spec.native)
        seeds = self._seeds(params, count)
        location, quantize = self._weights(spec)
        render = Render(
            op=Op.EDIT,
            model=spec.id,
            prompt=instruction.strip(),
            seeds=seeds,
            slots=list(range(1, count + 1)),
            width=width,
            height=height,
            steps=self._steps(spec, params, notes)[0],
            guidance=self._guidance(spec, params),
            images=list(sources),
            weights=location,
            quantize=quantize,
        )
        return Plan(
            model=spec.id,
            op=Op.EDIT,
            text=instruction.strip(),
            width=width,
            height=height,
            seeds=seeds,
            renders=[render],
            source=sources[0],
            notes=notes,
        )

    def blend(self, *, sources: list[str], aspect: tuple[int, int] | None, count: int) -> Plan:
        """Merge 2-5 images into new ones (Midjourney ``/blend``)."""
        if not 2 <= len(sources) <= 5:
            raise PlanError("Blend takes between 2 and 5 images.")
        notes: list[str] = []
        seeds = self._seeds(Params(), count)
        slots = list(range(1, count + 1))
        spec = self.catalog.pick(self.config.prefer.blend, Op.REDUX)
        if spec is not None:
            op, prompt = Op.REDUX, ""
        else:
            capable = [
                s
                for s in (self.catalog.registry.find(name) for name in self.config.prefer.edit)
                if s is not None
                and s.supports(Op.EDIT)
                and self.catalog.is_installed(s)
                and s.variants[Op.EDIT].max_refs >= len(sources)
            ]
            if not capable:
                raise PlanError(
                    f"Blending {len(sources)} images needs a model that accepts that many references. "
                    "`dreambot models pull flux2-klein-4b` handles up to 4."
                )
            spec = capable[0]
            op = Op.EDIT
            prompt = (
                "Blend all of the reference images into one new, cohesive image that combines their "
                "subjects, style, colours and mood."
            )
        width, height = dimensions(aspect, multiple=spec.multiple, native=spec.native)
        location, quantize = self._weights(spec)
        render = Render(
            op=op,
            model=spec.id,
            prompt=prompt,
            seeds=seeds,
            slots=slots,
            width=width,
            height=height,
            steps=spec.steps,
            images=list(sources),
            weights=location,
            quantize=quantize,
        )
        return Plan(
            model=spec.id,
            op=op,
            text=prompt,
            width=width,
            height=height,
            seeds=seeds,
            renders=[render],
            notes=notes,
        )
