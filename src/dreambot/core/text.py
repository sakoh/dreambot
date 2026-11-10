"""Small text utilities shared by the service and the worker (no heavy imports)."""

from __future__ import annotations

import re

SHORTEN_INSTRUCTION = (
    "You shorten prompts for a text-to-image model. Remove filler and anything that does not change the "
    "picture; keep the subject, setting, visual style and lighting. Write {count} versions of the prompt "
    "below, from longest to shortest; the last one is the bare essentials. Output exactly {count} lines "
    "numbered 1. to {count}. and nothing else.\n\nPrompt: {prompt}"
)

#: Phrases that pad prompts without changing what a modern model draws.
_FILLER = (
    "highly detailed",
    "ultra detailed",
    "extremely detailed",
    "intricate details",
    "best quality",
    "high quality",
    "masterpiece",
    "award winning",
    "award-winning",
    "trending on artstation",
    "artstation",
    "8k",
    "4k",
    "uhd",
    "hdr",
    "ultra realistic",
    "hyper realistic",
    "hyperrealistic",
    "photorealistic",
    "sharp focus",
    "professional",
    "stunning",
    "beautiful",
    "amazing",
    "epic",
    "very",
    "really",
    "extremely",
)


def parse_numbered(answer: str, count: int) -> list[str]:
    """Pull the numbered lines out of a model's answer, tolerating the usual formatting noise."""
    found: list[str] = []
    for line in answer.splitlines():
        match = re.match(r"^\s*(?:\*\*)?\s*(\d{1,2})\s*[.):\-]\s*(?:\*\*)?\s*(.+?)\s*$", line)
        if match:
            text = match.group(2).strip().strip("*").strip().strip('"').strip()
            if len(text) > 8:
                found.append(text)
    if not found:
        # No numbering at all: fall back to non-empty lines / paragraphs.
        found = [
            chunk.strip().strip('"') for chunk in re.split(r"\n\s*\n|\n", answer) if len(chunk.strip()) > 20
        ]
    return found[:count]


def heuristic_shorten(text: str, count: int = 5) -> list[str]:
    """Shorter versions of a prompt without any model: drop filler, then trailing phrases.

    Used when no language model is installed. Prompts are mostly comma-separated phrases in
    rough order of importance, so trimming from the end keeps what matters most.
    """
    cleaned = text
    for filler in sorted(_FILLER, key=len, reverse=True):
        cleaned = re.sub(rf"(?<![\w-]){re.escape(filler)}(?![\w-])", "", cleaned, flags=re.IGNORECASE)
    phrases = [re.sub(r"\s+", " ", phrase).strip() for phrase in re.split(r"[,;.]", cleaned)]
    phrases = [phrase for phrase in phrases if phrase]
    if not phrases:
        return [text.strip()] if text.strip() else []
    versions: list[str] = []
    for keep in range(len(phrases), 0, -1):
        candidate = ", ".join(phrases[:keep])
        if candidate not in versions:
            versions.append(candidate)
    if len(phrases) == 1:
        # One long phrase: shorten by words instead.
        words = phrases[0].split()
        for keep in (len(words) * 3 // 4, len(words) // 2, max(3, len(words) // 3)):
            candidate = " ".join(words[: max(1, keep)])
            if candidate not in versions:
                versions.append(candidate)
    if len(versions) > count:
        # Keep the longest, the shortest, and an even spread in between.
        step = (len(versions) - 1) / (count - 1)
        versions = [versions[round(i * step)] for i in range(count)]
    return versions


def split_text(text: str, limit: int) -> list[str]:
    """Cut a long message into parts of at most ``limit`` characters.

    Chat platforms refuse (Telegram) or silently cut (Discord) a message that is too long.
    Parts end at a blank line where possible, then at a line break, then at a space, so a list
    is not cut in the middle of an entry unless one entry is itself longer than a message.
    """
    limit = max(1, limit)
    parts: list[str] = []
    rest = text.strip("\n")
    while len(rest) > limit:
        window = rest[:limit]
        cut = -1
        for separator in ("\n\n", "\n", " "):
            position = window.rfind(separator)
            if position >= limit // 4:  # a nicer boundary is not worth a nearly empty message
                cut = position
                break
        if cut < 0:
            parts.append(window)  # one unbroken run of text
            rest = rest[limit:]
        else:
            parts.append(rest[:cut].rstrip())
            rest = rest[cut:].lstrip("\n ")
    if rest.strip():
        parts.append(rest)
    return parts


def split_caption(text: str, limit: int) -> tuple[str, str]:
    """(what fits under a picture, what has to follow as its own message).

    Whole lines are kept together: a caption's first lines name the job, and the notes after
    them are worth reading in full rather than half.
    """
    if len(text) <= limit:
        return text, ""
    lines = text.split("\n")
    head: list[str] = []
    for line in lines:
        if len("\n".join([*head, line])) > limit:
            break
        head.append(line)
    if not head:
        # Even the first line is too long: shorten it, and let the other lines follow whole.
        return lines[0][: max(1, limit - 1)].rstrip() + "…", "\n".join(lines[1:]).strip("\n")
    return "\n".join(head), "\n".join(lines[len(head) :]).strip("\n")
