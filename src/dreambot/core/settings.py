"""Per-user settings: what Midjourney exposes through ``/settings`` and ``/prefer``."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

from dreambot.config import Defaults
from dreambot.core.params import Params
from dreambot.engine.weights import physical_memory_gb

#: Midjourney's stylize presets, as shown in its settings panel.
STYLIZE_PRESETS = {"low": 50, "med": 100, "high": 250, "very high": 750}
VARIATION_MODES = ("strong", "subtle")
MAX_OPTION_SETS = 40
#: The prompt enhancer keeps a language model loaded next to the image model. With less memory
#: than this the two take turns, every job pays for a reload, and it is off unless asked for.
ENHANCE_MIN_MEMORY_GB = 24


def auto_enhance() -> bool:
    """Whether the prompt enhancer is on when neither the person nor the config has said."""
    return physical_memory_gb() >= ENHANCE_MIN_MEMORY_GB


@dataclass
class UserSettings:
    #: Model for /imagine. ``None`` follows the bot-wide default.
    model: str | None = None
    stylize: int | None = None
    count: int | None = None
    raw: bool | None = None
    #: What V1-V4 do: ``strong`` or ``subtle``.
    variation: str | None = None
    #: When on, variations ask for an edited prompt first.
    remix: bool | None = None
    hd: bool | None = None
    #: Whether a local language model adds art direction to prompts.
    enhance: bool | None = None
    #: Text appended to every prompt (``/prefer suffix``).
    suffix: str = ""
    #: Named shortcuts for parameter text (``/prefer option set``), used as ``--name``.
    option_sets: dict[str, str] = field(default_factory=dict)
    #: One-time explanations this person has already been shown.
    hints: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserSettings:
        known = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value not in (None, "", {}, [])}

    def resolved(self, defaults: Defaults) -> ResolvedSettings:
        """These settings with unset ones filled from the bot-wide defaults."""

        def pick(mine: Any, theirs: Any) -> Any:
            return theirs if mine is None else mine

        return ResolvedSettings(
            model=pick(self.model, defaults.model),
            stylize=pick(self.stylize, defaults.stylize),
            count=pick(self.count, defaults.count),
            raw=pick(self.raw, defaults.raw),
            variation=pick(self.variation, defaults.variation),
            remix=pick(self.remix, defaults.remix),
            hd=pick(self.hd, defaults.hd),
            enhance=pick(self.enhance, auto_enhance() if defaults.enhance is None else defaults.enhance),
            suffix=self.suffix,
            option_sets=dict(self.option_sets),
        )


@dataclass(frozen=True)
class ResolvedSettings:
    model: str | None
    stylize: int
    count: int
    raw: bool
    variation: str
    remix: bool
    hd: bool
    enhance: bool = False
    suffix: str = ""
    option_sets: dict[str, str] = field(default_factory=dict)

    def as_params(self) -> Params:
        """The settings expressed as prompt parameters, to merge under what a person typed."""
        return Params(
            model=self.model,
            stylize=self.stylize,
            count=self.count,
            raw=True if self.raw else None,
            hd=True if self.hd else None,
        )
