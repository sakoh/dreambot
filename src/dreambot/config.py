"""Configuration: one TOML file plus a few environment overrides.

Default location: ``~/Library/Application Support/Dreambot/config.toml``. Nothing in it is
required — Dreambot runs with zero configuration and picks the best installed model — so the
file only records choices that differ from the defaults. Bot tokens are kept out of the file
(see :mod:`dreambot.secrets`).
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

APP_NAME = "Dreambot"


def default_home() -> Path:
    """Where Dreambot keeps its config, database and logs."""
    if override := os.environ.get("DREAMBOT_HOME"):
        return Path(override).expanduser()
    return Path.home() / "Library" / "Application Support" / APP_NAME


def default_output_dir() -> Path:
    return Path.home() / "Pictures" / APP_NAME


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelOverride(_Section):
    """Per-model settings under ``[models.<id>]``."""

    #: Use these weights instead of auto-detection: a local folder or a cached Hugging Face repo id.
    weights: str | None = None
    #: Quantize on load (3, 4, 5, 6 or 8 bits). Only applies to non-quantized weights.
    quantize: Literal[3, 4, 5, 6, 8] | None = None
    steps: int | None = Field(default=None, ge=1, le=200)
    guidance: float | None = Field(default=None, ge=0, le=30)


class EngineConfig(_Section):
    #: ``mlx`` runs real models; ``fake`` draws placeholders (for trying the bots without weights).
    backend: Literal["mlx", "fake"] = "mlx"
    #: Seconds without work after which models are unloaded and their memory returned. 0 disables.
    idle_unload_seconds: float = Field(default=900, ge=0)
    #: Upper bound for memory held by loaded models, in GB. Default: half of the Mac's RAM.
    memory_budget_gb: float | None = Field(default=None, gt=0)
    #: Show a decoded preview every N denoising steps while a job runs (0 = off).
    preview_every: int = Field(default=0, ge=0)


class Preferences(_Section):
    """Which model to reach for, per operation. The first *installed* model in a list wins."""

    imagine: list[str] = Field(
        default_factory=lambda: [
            "z-image-turbo",
            "qwen-image-2.1-turbo",
            "flux2-klein-9b",
            "flux2-klein-4b",
            "krea-2",
            "flux3-image-klein",
            "schnell",
            "qwen-image",
            "qwen-image-2.1",
            "dev",
            "krea-dev",
            "z-image",
        ]
    )
    #: Fastest first: used by ``--draft``.
    draft: list[str] = Field(
        default_factory=lambda: [
            "flux2-klein-4b",
            "z-image-turbo",
            "schnell",
            "iris-3b",
            "lens-turbo",
            "boogu-image-turbo",
        ]
    )
    #: Anime and illustration: used by ``--niji``.
    niji: list[str] = Field(default_factory=lambda: ["anima-turbo", "anima-aesthetic", "anima-base"])
    #: Instruction edits and image references (``--edit``, ``--cref``, ``--oref``, ``--sref``).
    edit: list[str] = Field(
        default_factory=lambda: [
            "flux2-klein-9b-kv",
            "flux2-klein-9b",
            "qwen-image-edit-2511",
            "flux2-klein-4b",
            "qwen-image-2.1-turbo",
            "flux3-image-klein",
            "qwen-image-edit",
            "qwen-image-2.1",
            "dev-kontext",
            "fibo2-edit-turbo",
        ]
    )
    #: Mask-based repainting (vary region, zoom out, pan).
    inpaint: list[str] = Field(
        default_factory=lambda: [
            "dev-fill",
            "qwen-image-2.1-turbo",
            "qwen-image-2.1-turbo-controlnet-union",
            "qwen-image-2.1",
            "fibo2-edit-turbo",
        ]
    )
    upscale: list[str] = Field(default_factory=lambda: ["seedvr2-3b", "seedvr2-7b"])
    #: Blending image prompts.
    blend: list[str] = Field(default_factory=lambda: ["dev-redux"])
    #: Image-to-image fallbacks when a job's own model cannot do variations.
    variation: list[str] = Field(
        default_factory=lambda: [
            "z-image-turbo",
            "flux2-klein-4b",
            "flux2-klein-9b",
            "schnell",
            "qwen-image",
            "krea-2",
        ]
    )


class AssistConfig(_Section):
    """The local vision-language model behind ``/describe``, ``/shorten`` and the prompt enhancer."""

    #: Folder or cached Hugging Face repo id of an MLX vision model. Default: auto-detect.
    vision_model: str | None = None


class Defaults(_Section):
    """Defaults applied to every user until they change their own ``/settings``."""

    #: Force a model for /imagine instead of the first installed one in ``prefer.imagine``.
    model: str | None = None
    stylize: int = Field(default=100, ge=0, le=1000)
    #: Images per job.
    count: int = Field(default=4, ge=1, le=16)
    raw: bool = False
    #: ``strong`` or ``subtle``: what the V1-V4 buttons do.
    variation: Literal["strong", "subtle"] = "strong"
    remix: bool = False
    hd: bool = False
    #: Let a local language model add art direction to prompts. Unset: on when the Mac has the
    #: memory to keep that model loaded next to the image model (24 GB or more).
    enhance: bool | None = None


class AccessConfig(_Section):
    """Who may use the bot. Ids look like ``discord:1234``, ``telegram:5678``, ``imessage:+15551234567``."""

    #: Full control, including admin commands. The first person to pair becomes an owner.
    owners: list[str] = Field(default_factory=list)
    #: May generate images.
    allowed: list[str] = Field(default_factory=list)
    #: Let anyone who can reach the bot use it. Off by default: a local GPU is not a public service.
    open: bool = False
    #: Jobs a single person may have waiting at once.
    max_queued_per_user: int = Field(default=10, ge=1)


class DiscordConfig(_Section):
    enabled: bool = False
    #: Restrict slash commands to these server ids (empty = every server the bot is in).
    guilds: list[int] = Field(default_factory=list)


class TelegramConfig(_Section):
    enabled: bool = False


class IMessageConfig(_Section):
    enabled: bool = False
    #: ``self``: you text your own number and the bot answers in that thread.
    #: ``dedicated``: Messages on this Mac is signed into the bot's own Apple Account.
    mode: Literal["self", "dedicated"] = "self"
    #: In conversations with other people, only messages starting with this are treated as
    #: commands (ignored in your self-thread and in dedicated mode, where everything is for the bot).
    trigger: str = "/"
    #: Seconds between checks for new messages.
    poll_seconds: float = Field(default=1.0, gt=0)


class Config(_Section):
    home: Path = Field(default_factory=default_home)
    output_dir: Path = Field(default_factory=default_output_dir)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    defaults: Defaults = Field(default_factory=Defaults)
    prefer: Preferences = Field(default_factory=Preferences)
    models: dict[str, ModelOverride] = Field(default_factory=dict)
    assist: AssistConfig = Field(default_factory=AssistConfig)
    access: AccessConfig = Field(default_factory=AccessConfig)
    discord: DiscordConfig = Field(default_factory=DiscordConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    imessage: IMessageConfig = Field(default_factory=IMessageConfig)

    @property
    def config_path(self) -> Path:
        return self.home / "config.toml"

    @property
    def database_path(self) -> Path:
        return self.home / "dreambot.db"

    @property
    def log_dir(self) -> Path:
        return Path.home() / "Library" / "Logs" / APP_NAME

    @property
    def lora_dir(self) -> Path:
        return self.home / "loras"

    @property
    def scratch_dir(self) -> Path:
        """Working files (downloaded attachments, canvases, masks) that are safe to delete."""
        return self.home / "scratch"


class ConfigError(Exception):
    """The config file is invalid. The message names the offending keys."""


def load_config(path: Path | None = None, *, home: Path | None = None) -> Config:
    """Load ``config.toml`` (from ``path``, or from Dreambot's home directory)."""
    base = home or default_home()
    file = path or base / "config.toml"
    data: dict[str, Any] = {}
    if file.is_file():
        try:
            data = tomllib.loads(file.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{file}: {exc}") from exc
    data.setdefault("home", str(base))
    if output := os.environ.get("DREAMBOT_OUTPUT_DIR"):
        data["output_dir"] = output
    if backend := os.environ.get("DREAMBOT_BACKEND"):
        data.setdefault("engine", {})["backend"] = backend
    for key in ("home", "output_dir"):
        if key in data:
            data[key] = str(Path(str(data[key])).expanduser())
    try:
        return Config.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, error['loc']))}: {error['msg']}" for error in exc.errors())
        raise ConfigError(f"{file}: {problems}") from exc
