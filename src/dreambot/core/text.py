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
