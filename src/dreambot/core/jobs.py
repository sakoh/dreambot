"""Jobs: one unit of work a person asked for, and the images it produced."""

from __future__ import annotations

import re
import secrets
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

_OLD_LORA_NAME = re.compile(r"(?<!\S)--loras(?!\S)")

# Crockford base32 without the look-alike characters, so ids survive being read aloud or typed
# on a phone: no I, L, O or U.
_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
JOB_ID_LENGTH = 6


def new_job_id() -> str:
    return "".join(secrets.choice(_ALPHABET) for _ in range(JOB_ID_LENGTH))


def normalize_job_id(text: str) -> str:
    """Forgive the usual typos when someone types an id: case, and look-alike letters."""
    table = str.maketrans({"I": "1", "L": "1", "O": "0", "U": "V"})
    return text.strip().lstrip("#").upper().translate(table)


class JobKind(StrEnum):
    IMAGINE = "imagine"
    VARIATION = "variation"
    UPSCALE = "upscale"
    ZOOM = "zoom"
    PAN = "pan"
    EDIT = "edit"
    RETEXTURE = "retexture"
    BLEND = "blend"
    DESCRIBE = "describe"
    SHORTEN = "shorten"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def finished(self) -> bool:
        return self in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED)


@dataclass
class JobImage:
    #: 1-based position in the grid, as people refer to it (U1..U4).
    index: int
    path: str
    seed: int
    width: int
    height: int
    favorite: bool = False
    #: The full prompt this image was rendered from, when it says more than the job's text
    #: (per-image art direction). Variations and upscales of the image start from it.
    prompt: str | None = None


@dataclass
class Job:
    id: str
    kind: JobKind
    #: ``<frontend>:<platform user id>``, e.g. ``discord:123456789``.
    user: str
    #: Where results go: ``<frontend>:<conversation id>``.
    channel: str
    #: The prompt as the person would retype it (text and parameters, normalised).
    prompt: str
    status: JobStatus = JobStatus.QUEUED
    #: The descriptive text handed to the model, after any enhancement.
    text: str = ""
    negative: str | None = None
    model: str = ""
    width: int = 0
    height: int = 0
    seeds: list[int] = field(default_factory=list)
    #: Parameters in effect, by name (JSON-friendly).
    params: dict[str, Any] = field(default_factory=dict)
    #: The job and image this one was derived from (variation, upscale, zoom...).
    parent: str | None = None
    parent_index: int | None = None
    #: Input image paths by role: ``init``, ``mask``, ``refs`` (list).
    inputs: dict[str, Any] = field(default_factory=dict)
    #: Kind-specific details: zoom factor, pan direction, upscale mode, variation strength...
    detail: dict[str, Any] = field(default_factory=dict)
    images: list[JobImage] = field(default_factory=list)
    #: Text results (describe, shorten).
    texts: list[str] = field(default_factory=list)
    grid: str | None = None
    #: Notes for the person: parameters that were approximated, ignored or fell back.
    notes: list[str] = field(default_factory=list)
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None

    def image(self, index: int) -> JobImage:
        for image in self.images:
            if image.index == index:
                return image
        raise KeyError(index)

    @property
    def is_grid(self) -> bool:
        return len(self.images) > 1

    @property
    def seconds(self) -> float | None:
        if self.started is None or self.finished is None:
            return None
        return self.finished - self.started

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Job:
        values = dict(data)
        # Prompts kept before `--lora` was written back the way it is typed say `--loras`.
        values["prompt"] = _OLD_LORA_NAME.sub("--lora", values.get("prompt") or "")
        values["kind"] = JobKind(values["kind"])
        values["status"] = JobStatus(values["status"])
        values["images"] = [JobImage(**image) for image in values.get("images", [])]
        return cls(**values)


class EventType(StrEnum):
    QUEUED = "queued"
    STARTED = "started"
    #: The model is being loaded into memory (can take a while the first time).
    LOADING = "loading"
    #: The language model is writing art direction for the job's images.
    THINKING = "thinking"
    PROGRESS = "progress"
    PREVIEW = "preview"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class JobEvent:
    """What frontends subscribe to in order to render a job's life."""

    type: EventType
    job: Job
    #: Overall completion, 0..1 (PROGRESS).
    fraction: float = 0.0
    #: Path of a preview image (PREVIEW).
    preview: str | None = None
    #: Position in the queue, 0 = next (QUEUED).
    position: int = 0
