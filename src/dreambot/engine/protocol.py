"""Wire protocol between the Dreambot process and its MLX worker subprocess.

One JSON object per line. Requests flow parent -> worker on the worker's stdin; events flow
worker -> parent on the worker's original stdout. Images never cross the pipe: the worker
writes files to disk and reports their paths.

The protocol is deliberately small and language-neutral so that an alternative worker (for
example a Swift one) could implement it later.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class Op(StrEnum):
    """What a generation request asks a model to do."""

    TXT2IMG = "txt2img"
    #: Start from an init image; ``strength`` is how much of it to keep (variations, creative upscale).
    IMG2IMG = "img2img"
    #: Instruction-driven edit conditioned on one or more reference images.
    EDIT = "edit"
    #: Repaint the white area of a mask (vary region, zoom out, pan).
    INPAINT = "inpaint"
    #: Blend image prompts into a new image (image prompts, /blend, style references).
    REDUX = "redux"
    #: Keep the structure (depth / edges) of an image and repaint it (retexture).
    CONTROL = "control"
    #: Super-resolution.
    UPSCALE = "upscale"


class _Message(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------------------
# Requests (parent -> worker)
# --------------------------------------------------------------------------------------


class Lora(_Message):
    #: Local path or Hugging Face repo id of a LoRA.
    path: str
    scale: float = 1.0


class GenerateRequest(_Message):
    type: Literal["generate"] = "generate"
    id: str
    #: Registry id of the model, e.g. ``z-image-turbo``.
    model: str
    op: Op = Op.TXT2IMG
    prompt: str = ""
    negative_prompt: str | None = None
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    guidance: float | None = None
    #: One image is produced per seed.
    seeds: list[int] = Field(min_length=1)
    #: Init image (img2img, inpaint, upscale, control).
    image: str | None = None
    #: 0..1 — how much of the init image survives. 0 ignores it, 1 returns it unchanged.
    #: This is the model's raw control (the fraction of denoising steps skipped); what it
    #: means visually depends on the model and the image size. Prefer ``noise``.
    strength: float | None = None
    #: 0..1 — the noise level to start denoising the init image from: ~0.3 adds detail,
    #: ~0.6 varies details, ~0.85 keeps only the composition. The backend translates it into
    #: the right ``strength`` for the model's schedule at this size. Wins over ``strength``.
    noise: float | None = None
    #: Reference images (edit, redux).
    images: list[str] = Field(default_factory=list)
    image_weights: list[float] | None = None
    #: Mask image; white pixels are regenerated, black pixels are kept.
    mask: str | None = None
    loras: list[Lora] = Field(default_factory=list)
    #: Resolved weights location: a local directory or a Hugging Face repo id. ``None`` lets
    #: the backend use the model's upstream default.
    weights: str | None = None
    #: On-the-fly quantization bits for non-quantized weights (3, 4, 5, 6, 8).
    quantize: int | None = None
    out_dir: str
    #: Output files are named ``{basename}_{index}.png``.
    basename: str
    #: Fraction (0..1] of the denoising schedule to run before stopping early (Midjourney ``--stop``).
    stop_at: float | None = None
    #: Emit a decoded preview every N steps; 0 disables previews.
    preview_every: int = 0
    #: Backend-specific knobs (scheduler, upscale target, softness, ...).
    extra: dict[str, Any] = Field(default_factory=dict)


class DescribeRequest(_Message):
    """Caption an image with a vision-language model (Midjourney ``/describe``)."""

    type: Literal["describe"] = "describe"
    id: str
    model: str | None = None
    weights: str | None = None
    image: str
    #: Number of distinct prompt suggestions to produce.
    count: int = 4
    instruction: str | None = None
    max_tokens: int = 220
    seed: int | None = None


class TextRequest(_Message):
    """Run a local text LLM (prompt enhancement, /shorten, conversation)."""

    type: Literal["text"] = "text"
    id: str
    model: str | None = None
    weights: str | None = None
    system: str | None = None
    prompt: str
    max_tokens: int = 400
    temperature: float = 0.7
    seed: int | None = None


class CancelRequest(_Message):
    type: Literal["cancel"] = "cancel"
    #: Id of the request to cancel.
    id: str


class UnloadRequest(_Message):
    """Drop loaded models to free memory. ``model=None`` unloads everything."""

    type: Literal["unload"] = "unload"
    id: str
    model: str | None = None


class StatusRequest(_Message):
    type: Literal["status"] = "status"
    id: str


class ShutdownRequest(_Message):
    type: Literal["shutdown"] = "shutdown"


Request = Annotated[
    GenerateRequest
    | DescribeRequest
    | TextRequest
    | CancelRequest
    | UnloadRequest
    | StatusRequest
    | ShutdownRequest,
    Field(discriminator="type"),
]

# --------------------------------------------------------------------------------------
# Events (worker -> parent)
# --------------------------------------------------------------------------------------


class Ready(_Message):
    type: Literal["ready"] = "ready"
    pid: int
    backend: str
    versions: dict[str, str] = Field(default_factory=dict)


class Loading(_Message):
    """The model for a request is not in memory yet and is being loaded."""

    type: Literal["loading"] = "loading"
    id: str
    model: str


class Loaded(_Message):
    type: Literal["loaded"] = "loaded"
    id: str
    model: str
    seconds: float


class Progress(_Message):
    type: Literal["progress"] = "progress"
    id: str
    #: Index of the image within the request (0-based).
    index: int
    step: int
    total: int


class Preview(_Message):
    type: Literal["preview"] = "preview"
    id: str
    index: int
    step: int
    total: int
    path: str


class ImageDone(_Message):
    type: Literal["image"] = "image"
    id: str
    index: int
    seed: int
    path: str
    width: int
    height: int
    seconds: float


class TextDone(_Message):
    type: Literal["text"] = "text"
    id: str
    texts: list[str]
    seconds: float


class LoadedModel(_Message):
    key: str
    model: str
    idle_seconds: float


class StatusReport(_Message):
    type: Literal["status"] = "status"
    id: str
    loaded: list[LoadedModel] = Field(default_factory=list)
    active_memory_gb: float = 0.0
    peak_memory_gb: float = 0.0
    cache_memory_gb: float = 0.0


class Done(_Message):
    """Terminal event of a successful request."""

    type: Literal["done"] = "done"
    id: str
    seconds: float = 0.0
    peak_memory_gb: float = 0.0


class Cancelled(_Message):
    """Terminal event of a request that was cancelled."""

    type: Literal["cancelled"] = "cancelled"
    id: str


class Failed(_Message):
    """Terminal event of a request that failed. ``id`` is ``None`` for protocol-level errors."""

    type: Literal["error"] = "error"
    id: str | None = None
    #: Machine-readable category, see :class:`ErrorKind`.
    kind: str = "internal"
    message: str
    detail: str | None = None


class ErrorKind(StrEnum):
    #: The request itself is invalid (unknown model, unsupported operation, bad parameter).
    INVALID = "invalid"
    #: The model's weights are not on disk.
    NOT_INSTALLED = "not_installed"
    #: An optional dependency (mlx-vlm, mlx-lm) is missing.
    MISSING_DEPENDENCY = "missing_dependency"
    OUT_OF_MEMORY = "out_of_memory"
    #: The worker process died.
    CRASHED = "crashed"
    INTERNAL = "internal"


Event = Annotated[
    Ready
    | Loading
    | Loaded
    | Progress
    | Preview
    | ImageDone
    | TextDone
    | StatusReport
    | Done
    | Cancelled
    | Failed,
    Field(discriminator="type"),
]

TERMINAL_EVENTS = (Done, Cancelled, Failed)

_request_adapter: TypeAdapter[Request] = TypeAdapter(Request)
_event_adapter: TypeAdapter[Event] = TypeAdapter(Event)


def encode(message: BaseModel) -> str:
    """Serialize a request or event to a single JSON line (no trailing newline)."""
    return message.model_dump_json(exclude_defaults=False)


def decode_request(line: str | bytes) -> Request:
    return _request_adapter.validate_json(line)


def decode_event(line: str | bytes) -> Event:
    return _event_adapter.validate_json(line)
