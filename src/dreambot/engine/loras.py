"""LoRAs: one fine-tuned look each, kept in a library next to the models.

A model in the registry draws anything, in every style it has ever seen. A LoRA narrows it to
a single look — anime, pixel art, one illustrator's line. It is a small file (a few hundred
megabytes) of changes, trained on pictures of that look and laid over the model as it loads.
It is how the current generation of image models is fine-tuned: nobody passes around a whole
retrained copy of a 20 GB model when 200 MB of changes will do.

The library is a folder, ``<home>/loras``. Any ``.safetensors`` file in it can be used by name,
``--lora watercolor``. A LoRA added with ``dreambot loras add`` also has a few notes beside it,
``<name>.toml``, recording what a person would otherwise have to look up every time::

    title = "Pixel Art"
    base = "z-image-turbo"            # the model it was trained on
    trigger = "Pixel art style."      # words it answers to, put in front of the prompt
    scale = 1.0                       # how strongly it is applied
    source = "tarn59/pixel_art_style_lora_z_image_turbo"
    license = "apache-2.0"

With those, ``a fox --lora pixel-art`` runs on the right model with the right words, and the
person never has to know either.

Like the registry, this module never imports ``mlx`` or ``mflux`` and does not touch the
network: downloading is :mod:`dreambot.engine.prepare`'s job, handed in as a :class:`Hub`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from dreambot.engine.protocol import Op
from dreambot.engine.registry import REGISTRY, ModelSpec, Registry

#: Where a trigger says the person's own words go, when not simply after it.
PLACEHOLDER = "{prompt}"
_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,47}")
#: What ``--lora`` and ``/settings lora`` take to mean "no LoRA".
NONE = frozenset({"none", "off"})
#: Tensor names that only an adapter has (LoRA, LoKr and the kohya / ComfyUI spellings of both).
_ADAPTER_KEYS = ("lora_a", "lora_b", "lora_up", "lora_down", "lora.up", "lora.down", "lokr_w1", "lokr_w2")
#: A file larger than this is a whole model, whatever it is called.
_LARGEST = 3 * 1000**3
#: Base models named on Hugging Face by a repository the registry lists under a later release.
_BASE_REPOS = {"qwen/qwen-image": "qwen-image"}
_HF_URL = re.compile(
    r"https?://(?:www\.)?(?:huggingface\.co|hf\.co)/(?P<repo>[^/\s]+/[^/\s?#]+)"
    r"(?:/(?:blob|resolve|tree)/[^/\s]+/(?P<file>[^\s?#]+))?",
    re.IGNORECASE,
)
_REPO = re.compile(r"[\w.-]+/[\w.-]+")


class LoraError(Exception):
    """A LoRA cannot be read, added or used. The message is written for the person."""


@dataclass(frozen=True)
class Entry:
    """One LoRA of the library."""

    name: str
    #: The ``.safetensors`` file, as an absolute path.
    path: str
    #: Registry id of the model it was trained on, when that is known.
    base: str = ""
    title: str = ""
    #: Words it was trained to answer to. They go in front of the prompt, or — when they
    #: contain ``{prompt}`` — around it.
    trigger: str = ""
    #: How strongly it is applied; 1 is as trained.
    scale: float = 1.0
    #: Where it came from: a Hugging Face repository, or the name of a file.
    source: str = ""
    #: Its own license. The license of the model it runs on applies as well.
    license: str = ""


# --------------------------------------------------------------------------------------
# Names and trigger words
# --------------------------------------------------------------------------------------


def slug(text: str) -> str:
    """A LoRA name made from a title, a file name or a repository name."""
    name = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return name[:48].rstrip("-")


def check_name(name: str) -> str:
    """``name`` as a LoRA added to the library may be called, or :class:`LoraError` saying why not."""
    name = name.strip().lower()
    if not _NAME.fullmatch(name) or ".." in name:
        raise LoraError(
            f"`{name}` can't be a LoRA's name. Use lowercase letters, digits and hyphens, "
            "starting with a letter or a digit, e.g. `pixel-art`."
        )
    if name in NONE:
        raise LoraError(f'`{name}` already means "no LoRA". Pick another name.')
    return name


def worded(trigger: str, prompt: str) -> str:
    """``prompt`` with a LoRA's trigger words — once, however often it is asked for.

    A prompt comes back here when a job is varied or run again, already carrying the words.
    """
    text = prompt.strip()
    words = trigger.strip()
    if not words or not text:
        return prompt
    lowered = text.lower()
    if PLACEHOLDER in words:
        parts = [part.strip(" ,.;:").lower() for part in words.split(PLACEHOLDER)]
        if all(part in lowered for part in parts if part):
            return prompt
        return words.replace(PLACEHOLDER, text)
    if words.strip(" ,.;:").lower() in lowered:
        return prompt
    return f"{words} {text}" if words[-1] in ".!?:," else f"{words}, {text}"


# --------------------------------------------------------------------------------------
# The library
# --------------------------------------------------------------------------------------


class _Notes(BaseModel):
    """The notes kept beside a LoRA. Every one of them is optional."""

    model_config = ConfigDict(extra="forbid")

    #: The weights, when they are not ``<name>.safetensors`` beside the notes.
    file: str = ""
    base: str = ""
    title: str = ""
    trigger: str = ""
    scale: float = Field(default=1.0, ge=-4, le=4)
    source: str = ""
    license: str = ""


class Library:
    """The LoRAs in one folder."""

    def __init__(self, folder: Path, models: Registry = REGISTRY):
        self.folder = Path(folder)
        self._models = models

    def find(self, name: str) -> Entry | None:
        """The LoRA called ``name``: one with notes, or else a file of that name.

        Raises :class:`LoraError` when its notes cannot be used, rather than quietly applying
        the file without the model and the words the notes were written to supply.
        """
        if not re.fullmatch(r"[\w.-]+", name) or ".." in name:
            return None
        notes = self.folder / f"{name.lower()}.toml"
        if notes.is_file():
            return self._noted(notes)
        for file in (self.folder / name, self.folder / f"{name}.safetensors"):
            if file.is_file() and file.suffix == ".safetensors":
                return Entry(name=file.stem, path=str(file))
        return None

    def entries(self) -> tuple[list[Entry], dict[str, str]]:
        """Every LoRA in the folder by name, and the ones that could not be read with the reason."""
        found: dict[str, Entry] = {}
        problems: dict[str, str] = {}
        try:
            files = sorted(self.folder.iterdir())
        except OSError:
            return [], {}
        for path in files:
            if path.suffix == ".toml" and path.is_file():
                try:
                    found[path.stem] = self._noted(path)
                except LoraError as exc:
                    problems[path.stem] = str(exc)
        described = {entry.path for entry in found.values()}
        for path in files:
            claimed = str(path) in described or path.stem in found or path.stem in problems
            if path.suffix == ".safetensors" and path.is_file() and not claimed:
                found[path.stem] = Entry(name=path.stem, path=str(path))
        return [found[name] for name in sorted(found)], problems

    def _noted(self, path: Path) -> Entry:
        name = path.stem
        try:
            notes = _Notes.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError) as exc:
            raise LoraError(f"its notes can't be read ({exc})") from exc
        except tomllib.TOMLDecodeError as exc:
            raise LoraError(f"its notes aren't valid TOML ({exc})") from exc
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(map(str, error['loc']))}: {error['msg']}" for error in exc.errors()
            )
            raise LoraError(f"its notes have mistakes ({problems})") from exc
        file = Path(notes.file).expanduser() if notes.file else Path(f"{name}.safetensors")
        file = file if file.is_absolute() else path.parent / file
        if not file.is_file():
            raise LoraError(f"its file is gone ({file.name})")
        base = ""
        if notes.base.strip():
            spec = self._models.find(notes.base)
            if spec is None:
                raise LoraError(
                    f"its notes say it was trained on `{notes.base}`, which is not a model Dreambot has"
                )
            base = spec.id
        return Entry(
            name=name,
            path=str(file),
            base=base,
            title=notes.title.strip(),
            trigger=notes.trigger.strip(),
            scale=notes.scale,
            source=notes.source.strip(),
            license=notes.license.strip(),
        )


def save(folder: Path, entry: Entry) -> Path:
    """Write the notes of a LoRA whose file is in ``folder``, and return the notes file."""
    import tomli_w

    name = check_name(entry.name)
    file = Path(entry.path)
    data = {
        "title": entry.title,
        "base": entry.base,
        "trigger": entry.trigger,
        "scale": entry.scale,
        "source": entry.source,
        "license": entry.license,
        "file": "" if file == folder / f"{name}.safetensors" else str(file),
    }
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.toml"
    partial = path.with_suffix(".toml.part")
    partial.write_text(
        tomli_w.dumps({key: value for key, value in data.items() if value != ""}), encoding="utf-8"
    )
    os.replace(partial, path)  # never half-written notes for a running Dreambot to read
    return path


def remove(folder: Path, name: str) -> bool:
    """Take a LoRA out of the library: its notes, and its file when that is in the library."""
    name = check_name(name)
    found = False
    for path in (folder / f"{name}.toml", folder / f"{name}.safetensors"):
        found = found or path.is_file()
        path.unlink(missing_ok=True)
    return found


# --------------------------------------------------------------------------------------
# Looking inside a weights file
# --------------------------------------------------------------------------------------


def read_header(path: Path) -> dict[str, Any]:
    """The header of a ``.safetensors`` file: its tensors by name, and ``__metadata__``."""
    try:
        with path.open("rb") as file:
            length = int.from_bytes(file.read(8), "little")
            if not 2 <= length <= 256 * 1024 * 1024:
                raise ValueError("no header")
            header = json.loads(file.read(length))
    except (OSError, ValueError) as exc:
        raise LoraError(f"{path.name} is not a safetensors file.") from exc
    if not isinstance(header, dict):
        raise LoraError(f"{path.name} is not a safetensors file.")
    return header


def is_adapter(header: dict[str, Any]) -> bool:
    """Whether a weights file is a LoRA (or LoKr) rather than a whole model."""
    return any(
        marker in name.lower() for name in header if name != "__metadata__" for marker in _ADAPTER_KEYS
    )


def trigger_in(header: dict[str, Any]) -> str:
    """Trigger words a LoRA file carries about itself, if its trainer wrote them down."""
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict):
        return ""
    words = metadata.get("modelspec.trigger_phrase") or ""
    return words.strip() if isinstance(words, str) else ""


# --------------------------------------------------------------------------------------
# Adding to the library
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Card:
    """What a Hugging Face repository says about itself."""

    repo: str
    #: (path within the repository, bytes)
    files: tuple[tuple[str, int], ...] = ()
    #: Repositories of the model(s) it was trained on.
    base_models: tuple[str, ...] = ()
    trigger: str = ""
    license: str = ""


class Hub(Protocol):
    """What adding a LoRA needs from outside this module: the network, and mflux."""

    def card(self, repo: str) -> Card: ...

    def fetch_file(self, repo: str, filename: str, folder: Path) -> Path:
        """Download one file of a repository into ``folder`` and return where it is."""
        ...

    def takes_loras(self, base: ModelSpec) -> bool: ...


@dataclass
class Wanted:
    """What ``dreambot loras add`` was asked for."""

    #: A Hugging Face repository (its id or a link to it), or a ``.safetensors`` file on this Mac.
    source: str
    name: str | None = None
    base: str | None = None
    title: str | None = None
    trigger: str | None = None
    scale: float = 1.0
    #: Which file of a repository that holds several.
    file: str | None = None
    #: Replace the LoRA that already has this name.
    replace: bool = False


def parse_source(source: str) -> tuple[str, str, str | None]:
    """``(kind, where, file)`` — kind is ``file`` or ``repo``."""
    text = source.strip()
    if match := _HF_URL.match(text):
        return "repo", match.group("repo"), match.group("file")
    if text.startswith(("http://", "https://")):
        raise LoraError(
            "Only Hugging Face links can be downloaded. From anywhere else, download the file "
            "yourself and give its path."
        )
    local = Path(text).expanduser()
    if local.is_dir():
        raise LoraError(f"{local} is a folder. A LoRA is a single `.safetensors` file.")
    if local.is_file():
        return "file", str(local.resolve()), None
    repo, _, file = text.partition(":")
    if _REPO.fullmatch(repo) and not text.startswith((".", "~", "/")):
        return "repo", repo, file or None
    raise LoraError(f"`{text}` is neither a file on this Mac nor a Hugging Face repository (`owner/name`).")


def base_for(card: Card, models: Registry = REGISTRY) -> ModelSpec | None:
    """The model a repository says it was trained on, if Dreambot has it."""
    for repo in card.base_models:
        named = models.find(_BASE_REPOS.get(repo.lower(), ""))
        matches = [named] if named else [spec for spec in models if spec.repo.lower() == repo.lower()]
        # Several models can share a repository (a model and its ControlNet); the one that
        # draws from text is the one a LoRA was trained on.
        for spec in sorted(matches, key=lambda spec: not spec.supports(Op.TXT2IMG)):
            return spec
    return None


def _pick_file(card: Card, wanted: str | None) -> str:
    """The LoRA file of a repository: the one asked for, or the only one."""
    candidates = [(name, size) for name, size in card.files if name.endswith(".safetensors")]
    listing = (
        " It has: " + ", ".join(f"{name} ({size / 1000**2:.0f} MB)" for name, size in candidates[:12]) + "."
        if candidates
        else ""
    )
    if wanted:
        chosen = next(
            ((name, size) for name, size in candidates if wanted in (name, name.rsplit("/", 1)[-1])), None
        )
        if chosen is None:
            raise LoraError(f"{card.repo} has no file called `{wanted}`.{listing}")
    elif not candidates:
        raise LoraError(f"{card.repo} has no `.safetensors` file to use.")
    elif len(candidates) > 1:
        raise LoraError(f"{card.repo} has several files. Say which one with `--file`.{listing}")
    else:
        chosen = candidates[0]
    if chosen[1] > _LARGEST:
        raise LoraError(
            f"{chosen[0]} is {chosen[1] / 1000**3:.0f} GB: that is a whole model, not a LoRA. "
            "Dreambot's models come from `dreambot models pull`; a LoRA of the same look is a file of a few hundred MB."
        )
    return chosen[0]


def install(
    folder: Path,
    wanted: Wanted,
    models: Registry = REGISTRY,
    *,
    hub: Hub | None = None,
    say: Callable[[str], None] = print,
) -> Entry:
    """Add a LoRA to the library in ``folder`` and return it.

    Everything that can be learned from the source is: which model it was trained on, its
    trigger words, its license. What the person says outright wins over all of it.
    """
    kind, where, file_hint = parse_source(wanted.source)
    card: Card | None = None
    if kind == "repo":
        if hub is None:
            raise LoraError("Downloading from Hugging Face needs the network.")
        card = hub.card(where)

    # -- which model it is for
    base: ModelSpec | None = None
    if wanted.base:
        base = models.find(wanted.base)
        if base is None:
            raise LoraError(f"`{wanted.base}` is not a model Dreambot has. See `dreambot models list --all`.")
    elif card is not None:
        base = base_for(card, models)
        if base is not None:
            say(f"Trained on {base.title} ({base.id}), going by what {card.repo} says.")
    if base is None:
        trained_on = f" (its page names {', '.join(card.base_models)})" if card and card.base_models else ""
        say(
            f"I can't tell which model this was trained on{trained_on}. It will be tried on whichever model "
            "a prompt runs on; `--base` says which one it is for."
        )
    elif hub is not None and not hub.takes_loras(base):
        raise LoraError(
            f"{base.title} can't take a LoRA. Models that can include z-image-turbo, dev and qwen-image."
        )

    # -- what to call it
    stem = where.rstrip("/").rsplit("/", 1)[-1].removesuffix(".safetensors")
    name = check_name(wanted.name or slug(stem) or "lora")
    target, notes = folder / f"{name}.safetensors", folder / f"{name}.toml"
    if (target.exists() or notes.exists()) and not wanted.replace:
        raise LoraError(f"There is already a LoRA called `{name}`. Use `--name`, or `--replace`.")

    # -- its file
    folder.mkdir(parents=True, exist_ok=True)
    arriving = folder / f".adding-{name}"
    shutil.rmtree(arriving, ignore_errors=True)
    try:
        if card is not None:
            assert hub is not None
            chosen = _pick_file(card, wanted.file or file_hint)
            say(f"Downloading {chosen} from {card.repo} ...")
            arriving.mkdir()
            fetched = hub.fetch_file(card.repo, chosen, arriving)
        else:
            fetched = Path(where)
            if fetched.suffix != ".safetensors":
                raise LoraError(f"{fetched.name} is not a `.safetensors` file.")
        header = read_header(fetched)
        if not is_adapter(header):
            raise LoraError(f"{fetched.name} is a whole model in a single file, not a LoRA.")
        if card is not None:
            os.replace(fetched, target)
        elif fetched != target:
            shutil.copy2(fetched, target)  # a copy of its own: the original may live in Downloads
    finally:
        shutil.rmtree(arriving, ignore_errors=True)

    trigger = (
        wanted.trigger
        if wanted.trigger is not None
        else ((card.trigger if card else "") or trigger_in(header))
    )
    entry = Entry(
        name=name,
        path=str(target),
        base=base.id if base else "",
        title=(wanted.title or name.replace("-", " ").replace("_", " ").replace(".", " ").title())[:80],
        trigger=trigger.strip(),
        scale=wanted.scale,
        source=card.repo if card else Path(where).name,
        license=card.license if card else "",
    )
    save(folder, entry)
    return entry
