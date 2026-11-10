"""Parsing Midjourney-style prompts.

A prompt has up to three sections, in this order::

    [image links ...] descriptive text [--parameter value ...]

On top of that:

* **Permutations** — ``a {red, green} bird --ar {1:1, 16:9}`` expands into one prompt per
  combination (``expand_permutations``). Braces nest; ``\\,`` is a literal comma.
* **Multi-prompts** — ``space:: ship::2 --no stars`` weights parts of the text against each
  other; negative weights (``::-0.5``) ask for less of something.
* **Dashes** — iPhones autocorrect ``--`` to an em dash, so ``—ar 16:9`` is accepted too. That
  matters here: iMessage is a first-class way to talk to Dreambot.

Parsing is pure text work with no I/O, so it can be tested exhaustively.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from dreambot.core.params import (
    Kind,
    Params,
    ParamSpec,
    PromptError,
    Ref,
    Support,
    find_param,
    parse_value,
    suggest_param,
)

#: Midjourney caps permutation jobs at 40 on its largest plans; a local GPU deserves a cap too.
MAX_PERMUTATIONS = 40
MAX_PROMPT_CHARS = 6000

_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
# A parameter starts at "--name" (or an em/en dash from iOS autocorrect) preceded by whitespace
# or the start of the prompt. The name must start with a letter, so "-- " and "--5" are text.
_PARAM_RE = re.compile(r"(?:^|(?<=\s))(?:--|—|–)(?P<name>[A-Za-z][A-Za-z0-9_]*)")
# A weight must follow "::" immediately: in "space:: 2001 odyssey" the number is text.
_WEIGHT_RE = re.compile(r"::(?P<weight>[+-]?(?:\d+\.?\d*|\.\d+))?")


@dataclass(frozen=True)
class PromptPart:
    """One weighted section of a multi-prompt."""

    text: str
    weight: float = 1.0


@dataclass
class ParsedPrompt:
    #: The descriptive text with links and parameters removed.
    text: str
    #: Weighted parts of the text. A plain prompt is a single part of weight 1.
    parts: list[PromptPart]
    #: Image prompts: links found at the start of the prompt (attachments are added by the caller).
    image_prompts: list[Ref]
    params: Params
    #: Parameters that were recognised but are not implemented yet (``--video`` ...), by name.
    planned: list[str] = field(default_factory=list)
    #: Parameters accepted for compatibility that have no local effect (``--stealth`` ...), by name.
    ignored: list[str] = field(default_factory=list)
    #: Parameters in the order given, as normalised ``--name value`` strings.
    given: list[str] = field(default_factory=list)

    @property
    def is_multi(self) -> bool:
        return len(self.parts) > 1 or (len(self.parts) == 1 and self.parts[0].weight != 1.0)

    @property
    def positive_text(self) -> str:
        """The text to hand the model: positively weighted parts, strongest first."""
        if not self.is_multi:
            return self.text
        positive = [part for part in self.parts if part.weight > 0]
        positive.sort(key=lambda part: -part.weight)
        return ", ".join(part.text for part in positive)

    @property
    def negative_terms(self) -> list[str]:
        """Everything the prompt asks to avoid: ``--no`` items and negatively weighted parts."""
        return [*self.params.no, *(part.text for part in self.parts if part.weight < 0)]

    def canonical(self) -> str:
        """The prompt re-rendered in a normal form (used for display and for re-running jobs)."""
        pieces = [ref.value for ref in self.image_prompts]
        if self.is_multi:
            pieces.append(" ".join(f"{part.text}::{_trim(part.weight)}" for part in self.parts))
        elif self.text:
            pieces.append(self.text)
        pieces.extend(self.given)
        return " ".join(pieces)


def _trim(number: float) -> str:
    return str(int(number)) if float(number).is_integer() else f"{number:g}"


# --------------------------------------------------------------------------------------
# Permutations
# --------------------------------------------------------------------------------------


def expand_permutations(prompt: str, limit: int = MAX_PERMUTATIONS) -> list[str]:
    """Expand ``{a, b}`` groups into every combination.

    >>> expand_permutations("a {red, green} bird")
    ['a red bird', 'a green bird']
    """
    results = _expand(prompt, limit)
    cleaned: list[str] = []
    for result in results:
        text = re.sub(
            r"[ \t]+", " ", result.replace("\\,", ",").replace("\\{", "{").replace("\\}", "}")
        ).strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def _expand(prompt: str, limit: int) -> list[str]:
    start = _find_unescaped(prompt, "{", 0)
    if start < 0:
        return [prompt]
    depth = 0
    end = -1
    index = start
    while index < len(prompt):
        char = prompt[index]
        if char == "\\":
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index
                break
        index += 1
    if end < 0:
        raise PromptError("There's a `{` without a matching `}`. To use a literal brace, write `\\{`.")

    options = _split_options(prompt[start + 1 : end])
    if len(options) < 2:
        # A single option is not a permutation; keep it (and its braces' content) as text.
        options = options or [""]
    head, tail = prompt[:start], prompt[end + 1 :]
    results: list[str] = []
    tails = _expand(tail, limit)
    for option in options:
        for expanded_option in _expand(option, limit):
            for expanded_tail in tails:
                results.append(f"{head}{expanded_option}{expanded_tail}")
                if len(results) > limit:
                    raise PromptError(
                        f"That expands to more than {limit} prompts. Use fewer or smaller `{{...}}` groups."
                    )
    return results


def _find_unescaped(text: str, char: str, start: int) -> int:
    index = start
    while index < len(text):
        if text[index] == "\\":
            index += 2
            continue
        if text[index] == char:
            return index
        index += 1
    return -1


def _split_options(body: str) -> list[str]:
    """Split a permutation group on top-level commas, honouring nesting and ``\\,`` escapes."""
    options: list[str] = []
    current: list[str] = []
    depth = 0
    index = 0
    while index < len(body):
        char = body[index]
        if char == "\\" and index + 1 < len(body):
            current.append(body[index : index + 2])
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        if char == "," and depth == 0:
            options.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        index += 1
    options.append("".join(current).strip())
    return options


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------


def parse_prompt(raw: str, option_sets: dict[str, str] | None = None) -> ParsedPrompt:
    """Parse one prompt (after permutation expansion).

    ``option_sets`` maps a user's custom option names to the parameter text they stand for
    (Midjourney's ``/prefer option set``): with ``{"wide": "--ar 21:9 --stylize 300"}``, a
    prompt may say ``--wide``.
    """
    text = raw.replace(" ", " ").strip()
    if len(text) > MAX_PROMPT_CHARS:
        raise PromptError(f"That prompt is {len(text)} characters long; the limit is {MAX_PROMPT_CHARS}.")
    text = _expand_option_sets(text, option_sets or {})

    matches = _parameter_matches(text)
    body = text[: matches[0].start()] if matches else text

    params = Params()
    planned: list[str] = []
    ignored: list[str] = []
    given: list[str] = []
    seen: set[str] = set()

    for position, match in enumerate(matches):
        name = match.group("name").lower()
        value_end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        raw_value = text[match.end() : value_end].strip()
        spec = find_param(name)
        if spec is None:
            hint = suggest_param(name)
            suggestion = f" Did you mean `--{hint[0]}`?" if hint else ""
            raise PromptError(f"I don't know the parameter `--{name}`.{suggestion}")
        _apply(params, spec, raw_value, seen)
        if spec.support is Support.PLANNED:
            planned.append(spec.name)
        elif spec.support is Support.ACCEPTED:
            ignored.append(spec.name)
        given.append(f"--{spec.name} {raw_value}".strip())

    image_prompts, description = _split_image_prompts(body)
    description = re.sub(r"\s+", " ", description).strip()
    parts = _split_multi_prompt(description)

    if params.style == "raw":
        params.raw = True
    return ParsedPrompt(
        text=description,
        parts=parts,
        image_prompts=image_prompts,
        params=params,
        planned=planned,
        ignored=ignored,
        given=given,
    )


def _parameter_matches(text: str) -> list[re.Match[str]]:
    """Locate the ``--name`` markers that start parameters.

    A real double hyphen always starts a parameter (unknown names are reported as errors, as
    Midjourney does). An em or en dash only counts when it spells a known parameter, so prose
    like "a quiet night —cold and still" is left alone.
    """
    found: list[re.Match[str]] = []
    for match in _PARAM_RE.finditer(text):
        typed_dashes = match.group(0)[: -len(match.group("name"))]
        if typed_dashes != "--" and find_param(match.group("name")) is None:
            continue
        found.append(match)
    return found


def _expand_option_sets(text: str, option_sets: dict[str, str]) -> str:
    if not option_sets:
        return text
    lowered = {name.lower(): value for name, value in option_sets.items()}

    def replace(match: re.Match[str]) -> str:
        name = match.group("name").lower()
        if find_param(name) is None and name in lowered:
            return lowered[name]
        return match.group(0)

    # One pass only: option sets may not refer to other option sets, which rules out loops.
    return _PARAM_RE.sub(replace, text)


def _apply(params: Params, spec: ParamSpec, raw_value: str, seen: set[str]) -> None:
    value = parse_value(spec, raw_value)
    dest = spec.dest
    if spec.kind in (Kind.REFS, Kind.LIST, Kind.LORAS):
        # Repeating a list parameter adds to it: `--no text --no watermark`.
        getattr(params, dest).extend(value)
        if spec.kind is Kind.REFS and not value:
            # `--sref` with no value means "use the attached image(s)"; mark it as requested.
            getattr(params, dest).append(Ref("attachment", "*"))
    elif dest in seen and getattr(params, dest) != value:
        raise PromptError(f"`--{spec.name}` is given more than once with different values.")
    else:
        setattr(params, dest, value)
    seen.add(dest)


def _split_image_prompts(body: str) -> tuple[list[Ref], str]:
    """Peel image links off the front of the prompt; later links stay part of the text."""
    refs: list[Ref] = []
    rest = body.strip()
    while True:
        match = _URL_RE.match(rest)
        if not match:
            break
        refs.append(Ref("url", match.group(0)))
        rest = rest[match.end() :].lstrip()
    return refs, rest


def _split_multi_prompt(description: str) -> list[PromptPart]:
    text = description
    if "::" not in text:
        return [PromptPart(text)] if text else []
    parts: list[PromptPart] = []
    cursor = 0
    for match in _WEIGHT_RE.finditer(text):
        chunk = text[cursor : match.start()].strip()
        weight = float(match.group("weight")) if match.group("weight") else 1.0
        if chunk:
            parts.append(PromptPart(chunk, weight))
        elif match.group("weight") and parts:
            raise PromptError("A `::weight` has no text in front of it.")
        cursor = match.end()
    trailing = text[cursor:].strip()
    if trailing:
        parts.append(PromptPart(trailing, 1.0))
    parts = [part for part in parts if part.weight != 0]
    if parts and sum(part.weight for part in parts) <= 0:
        raise PromptError("Multi-prompt weights must add up to more than zero.")
    return parts
