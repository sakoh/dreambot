"""Actions on finished images: the buttons under a Midjourney grid, as data.

An action can arrive three ways, and all three produce the same :class:`Action`:

* a button press (Discord components, Telegram inline keyboards) carrying an encoded action;
* a typed shortcut such as ``U2``, ``V3``, ``zoom 2 1.5x`` or ``pan left`` (iMessage has no
  buttons, so text is its primary interface);
* a tapback or reaction on a specific image.

The typed grammar is deliberately strict. In a direct conversation any message that is not a
command is taken as a prompt, so ``zoom lens photo of a hawk`` must not be mistaken for a zoom.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class ActionKind(StrEnum):
    #: U1-U4: pull one image out of the grid at full size.
    SELECT = "select"
    #: V1-V4: variations using the person's variation mode (strong or subtle).
    VARY = "vary"
    VARY_SUBTLE = "vary_subtle"
    VARY_STRONG = "vary_strong"
    #: Repaint part of the image. ``value`` is the region, ``prompt`` what to put there.
    VARY_REGION = "vary_region"
    #: Apply a written instruction to the whole image. ``prompt`` is the instruction.
    EDIT = "edit"
    REROLL = "reroll"
    UPSCALE_SUBTLE = "upscale_subtle"
    UPSCALE_CREATIVE = "upscale_creative"
    #: ``value`` is the zoom-out factor, e.g. ``"2"`` or ``"1.5"``.
    ZOOM = "zoom"
    #: ``value`` is ``left``, ``right``, ``up`` or ``down``.
    PAN = "pan"
    SQUARE = "square"
    FAVORITE = "favorite"
    #: Add the image to a Lookbook. ``prompt`` is the Lookbook's name.
    LOOK = "look"
    INFO = "info"
    CANCEL = "cancel"
    ANIMATE = "animate"


#: Actions that start a new generation job (the rest answer immediately).
GENERATIVE = frozenset(
    {
        ActionKind.VARY,
        ActionKind.VARY_SUBTLE,
        ActionKind.VARY_STRONG,
        ActionKind.VARY_REGION,
        ActionKind.EDIT,
        ActionKind.REROLL,
        ActionKind.UPSCALE_SUBTLE,
        ActionKind.UPSCALE_CREATIVE,
        ActionKind.ZOOM,
        ActionKind.PAN,
        ActionKind.SQUARE,
    }
)

PAN_DIRECTIONS = ("left", "right", "up", "down")


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    #: Job id, or ``None`` for "the most recent job in this conversation".
    job: str | None = None
    #: 1-based image number; ``None`` when the job has one image or the action is job-wide.
    index: int | None = None
    value: str | None = None
    #: A replacement prompt (remix), an edit instruction, or what to paint into a region.
    prompt: str | None = None

    def on(self, job: str | None, index: int | None = None) -> Action:
        """This action aimed at a specific job (and image)."""
        return Action(
            self.kind, job or self.job, index if index is not None else self.index, self.value, self.prompt
        )


# --------------------------------------------------------------------------------------
# Compact encoding for button payloads (Telegram allows 64 bytes, Discord 100 characters)
# --------------------------------------------------------------------------------------

_CODES = {
    ActionKind.SELECT: "u",
    ActionKind.VARY: "v",
    ActionKind.VARY_SUBTLE: "vs",
    ActionKind.VARY_STRONG: "vg",
    ActionKind.VARY_REGION: "vr",
    ActionKind.EDIT: "e",
    ActionKind.REROLL: "r",
    ActionKind.UPSCALE_SUBTLE: "us",
    ActionKind.UPSCALE_CREATIVE: "uc",
    ActionKind.ZOOM: "z",
    ActionKind.PAN: "p",
    ActionKind.SQUARE: "sq",
    ActionKind.FAVORITE: "f",
    ActionKind.LOOK: "lb",
    ActionKind.INFO: "i",
    ActionKind.CANCEL: "x",
    ActionKind.ANIMATE: "a",
}
_KINDS = {code: kind for kind, code in _CODES.items()}
ENCODED_PREFIX = "db"


def encode_action(action: Action) -> str:
    """``db:<kind>:<job>:<index>:<value>`` — prompts are never encoded (they are asked for)."""
    if action.job is None:
        raise ValueError("only actions bound to a job can be encoded")
    fields = [ENCODED_PREFIX, _CODES[action.kind], action.job, str(action.index or ""), action.value or ""]
    return ":".join(fields).rstrip(":")


def decode_action(payload: str) -> Action | None:
    parts = payload.split(":")
    if len(parts) < 3 or parts[0] != ENCODED_PREFIX or parts[1] not in _KINDS:
        return None
    index = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else None
    value = ":".join(parts[4:]) if len(parts) > 4 and parts[4] else None
    return Action(_KINDS[parts[1]], parts[2], index, value)


# --------------------------------------------------------------------------------------
# Typed shortcuts
# --------------------------------------------------------------------------------------

_INDEX = r"(?P<index>\d{1,2})"
_FACTOR = r"(?P<factor>\d(?:\.\d+)?)\s*x?"
_ARROWS = {
    "⬅": "left",
    "←": "left",
    "➡": "right",
    "→": "right",
    "⬆": "up",
    "↑": "up",
    "⬇": "down",
    "↓": "down",
}

_PATTERNS: list[tuple[re.Pattern[str], ActionKind]] = [
    (re.compile(rf"u\s*{_INDEX}"), ActionKind.SELECT),
    (re.compile(rf"v\s*{_INDEX}(?:\s*:\s*(?P<prompt>.+))?"), ActionKind.VARY),
    (re.compile(r"(?:r|reroll|re-roll|redo|again|🔄)"), ActionKind.REROLL),
    (re.compile(rf"(?:upscale|up)\s*(?:{_INDEX})?\s*(?P<mode>subtle|creative)?"), ActionKind.UPSCALE_SUBTLE),
    (re.compile(rf"upscale\s+(?P<mode>subtle|creative)\s*(?:{_INDEX})?"), ActionKind.UPSCALE_SUBTLE),
    (
        re.compile(rf"vary\s*(?:{_INDEX})?\s*(?P<mode>subtle|strong)?(?:\s*:\s*(?P<prompt>.+))?"),
        ActionKind.VARY,
    ),
    (re.compile(rf"vary\s+(?P<mode>subtle|strong)\s*(?:{_INDEX})?"), ActionKind.VARY),
    (re.compile(rf"zoom(?:\s+out)?(?:\s+{_INDEX}(?!\s*x|\.))?(?:\s+{_FACTOR})?"), ActionKind.ZOOM),
    (re.compile(rf"pan\s+(?:{_INDEX}\s+)?(?P<direction>left|right|up|down)"), ActionKind.PAN),
    (re.compile(rf"pan\s+(?P<direction>left|right|up|down)\s+{_INDEX}"), ActionKind.PAN),
    (re.compile(rf"(?:make\s+)?square(?:\s+{_INDEX})?"), ActionKind.SQUARE),
    (re.compile(rf"(?:fav|fave|favorite|favourite|like|love|❤️?|♥️?)\s*(?:{_INDEX})?"), ActionKind.FAVORITE),
    (re.compile(rf"(?:info|seed|✉️?)\s*(?:{_INDEX})?"), ActionKind.INFO),
    (re.compile(r"(?:cancel|stop|❌)"), ActionKind.CANCEL),
    (re.compile(rf"animate\s*(?:{_INDEX})?\s*(?P<motion>low|high)?"), ActionKind.ANIMATE),
]

# Commands that take free text need an explicit marker so prompts are never mistaken for them:
#   region 2 B2-D4: a red balloon      (cells, then what to paint)
#   region 2 the sky: an aurora        (a description, then what to paint)
#   edit 2: make it night
#   look 2: autumn diner               (add the image to that Lookbook)
# They are matched against the text as typed, to keep a prompt's capitals, so they ignore case
# themselves: a phone capitalises the first word, and "Edit: make it night" is still an edit.
_REGION = re.compile(
    rf"(?:region|vary\s+region)\s+(?:{_INDEX}\s+)?(?P<where>[^:]+?)\s*:\s*(?P<prompt>.+)", re.S | re.I
)
_EDIT = re.compile(rf"edit\s*(?:{_INDEX})?\s*:\s*(?P<prompt>.+)", re.S | re.I)
_LOOK = re.compile(rf"(?:lookbook|look)\s*(?:{_INDEX})?\s*:\s*(?P<name>[^\n]+)", re.I)


def parse_action(text: str) -> Action | None:
    """Recognise a typed shortcut. Returns ``None`` for anything that is not exactly one."""
    raw = text.strip()
    if not raw or len(raw) > 2000:
        return None
    body = raw[1:].strip() if raw.startswith("/") else raw
    lowered = body.lower()

    for arrow, direction in _ARROWS.items():
        if lowered.rstrip("️").startswith(arrow):
            rest = lowered.rstrip("️")[len(arrow) :].strip("️ ").strip()
            if rest == "" or rest.isdigit():
                return Action(ActionKind.PAN, index=int(rest) if rest else None, value=direction)

    if match := _REGION.fullmatch(body):
        index = int(match.group("index")) if match.group("index") else None
        return Action(
            ActionKind.VARY_REGION,
            index=index,
            value=match.group("where").strip(),
            prompt=match.group("prompt").strip(),
        )
    if match := _EDIT.fullmatch(body):
        index = int(match.group("index")) if match.group("index") else None
        return Action(ActionKind.EDIT, index=index, prompt=match.group("prompt").strip())
    if match := _LOOK.fullmatch(body):
        index = int(match.group("index")) if match.group("index") else None
        return Action(ActionKind.LOOK, index=index, prompt=match.group("name").strip())

    for pattern, kind in _PATTERNS:
        match = pattern.fullmatch(lowered)
        if not match:
            continue
        groups = match.groupdict()
        index = int(groups["index"]) if groups.get("index") else None
        if index is not None and index < 1:
            return None
        prompt = None
        if groups.get("prompt"):
            # Take the prompt from the original text so its capitalisation survives.
            prompt = body[match.start("prompt") : match.end("prompt")].strip()
        value: str | None = None
        if kind is ActionKind.UPSCALE_SUBTLE and groups.get("mode") == "creative":
            kind = ActionKind.UPSCALE_CREATIVE
        elif kind is ActionKind.VARY and groups.get("mode"):
            kind = ActionKind.VARY_SUBTLE if groups["mode"] == "subtle" else ActionKind.VARY_STRONG
        elif kind is ActionKind.ZOOM:
            factor = float(groups["factor"]) if groups.get("factor") else 2.0
            if not 1.0 < factor <= 4.0:
                return None
            value = f"{factor:g}"
        elif kind is ActionKind.PAN:
            value = groups["direction"]
        elif kind is ActionKind.ANIMATE:
            value = groups.get("motion") or "low"
        return Action(kind, index=index, value=value, prompt=prompt)
    return None
