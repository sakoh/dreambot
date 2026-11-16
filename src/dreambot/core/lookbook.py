"""Lookbooks: a named set of pictures that stands for a look.

This is Dreambot's counterpart of Midjourney's moodboards. A person collects pictures — uploads,
links, their own results — and from then on ``--p <name>`` gives a prompt that look. Where a
style reference (``--sref``) is one specific look applied as it is, a Lookbook is a range: each
image of a grid leans on a different picture of the book, so a varied book gives varied results
and a tight one consistent results.

Like a moodboard, a Lookbook changes as pictures are added and removed, and every state of it has
a **code**. A code is a frozen version: it keeps working after the book is edited or deleted,
which is what makes an old prompt reproducible. The code is derived from the pictures themselves,
so going back to an earlier set of pictures gives back the earlier code.

This module is the pure part. Storage is :mod:`dreambot.core.store`; reading the pictures with
the vision model and applying the result is the service's and the planner's job.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from dreambot.core import fingerprint
from dreambot.core.fingerprint import StyleFingerprint

CODE_LENGTH = 7
#: No i, l or o, and no 1: a code survives being read aloud or retyped from a phone.
_LETTERS = "abcdefghjkmnpqrstuvwxyz"
_ALPHABET = _LETTERS + "023456789"
#: The token for "the Lookbooks I have chosen as my defaults" (a bare ``--p``).
DEFAULT = "default"
#: The token for the book Dreambot keeps of the pictures a person has favourited.
LIKED = "liked"
#: How many favourites the liked book looks at, newest first.
LIKED_LIMIT = 24
#: The share of an image's look that its lead picture holds. Above ``fingerprint.DOMINANT_SHARE``,
#: so that the lead sets the medium and the rest of the book colours it.
LEAD_SHARE = 0.6
MAX_NAME = 60


def version_code(lookbook_id: str, keys: Iterable[str]) -> str:
    """The code of a Lookbook holding exactly the pictures with these content keys."""
    digest = hashlib.sha256("|".join([lookbook_id, *sorted(set(keys))]).encode()).digest()
    number = int.from_bytes(digest[:8], "big")
    # A letter first, so that a code can never be mistaken for a numeric style code.
    chars = [_LETTERS[number % len(_LETTERS)]]
    number //= len(_LETTERS)
    for _ in range(CODE_LENGTH - 1):
        chars.append(_ALPHABET[number % len(_ALPHABET)])
        number //= len(_ALPHABET)
    return "".join(chars)


def looks_like_code(token: str) -> bool:
    return len(token) == CODE_LENGTH and token[0] in _LETTERS and all(char in _ALPHABET for char in token)


def slug(name: str) -> str:
    """A Lookbook's name as the one word it is typed as after ``--p``."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def clean_name(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip()[:MAX_NAME]


def parse_token(token: str) -> tuple[str, float]:
    """``dunes::2`` -> ("dunes", 2.0). A token without a weight counts once."""
    body, separator, weight = token.rpartition("::")
    if separator and body:
        try:
            return body, float(weight)
        except ValueError:
            pass
    return token, 1.0


@dataclass(frozen=True)
class Member:
    """One picture of a Lookbook, with what is known about its style."""

    key: str
    path: str
    print: StyleFingerprint


def consensus(members: Sequence[Member]) -> StyleFingerprint | None:
    """What a Lookbook's pictures have in common: all of them, blended at equal weight."""
    return fingerprint.blend([(member.print, 1.0) for member in members])


def cast(count: int, seed: int, index: int) -> list[float]:
    """The weight of each of a book's ``count`` pictures for image ``index`` of a grid.

    One picture leads and holds ``LEAD_SHARE`` of the look; the others share the rest equally.
    The lead rotates through the book in an order fixed by the job's seed, so the images of a
    grid — and a re-run with the same seed — are each led by a known picture.
    """
    if count <= 0:
        return []
    if count == 1:
        return [1.0]
    order = list(range(count))
    random.Random(f"lookbook|{seed}").shuffle(order)
    lead = order[index % count]
    # With every other picture at weight 1, this is the weight that gives the lead its share.
    lead_weight = LEAD_SHARE * (count - 1) / (1 - LEAD_SHARE)
    return [lead_weight if position == lead else 1.0 for position in range(count)]
