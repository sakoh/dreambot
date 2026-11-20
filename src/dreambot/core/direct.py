"""The director: a local model that says what happens in a video.

A picture says what is in it and nothing about what happens next. Midjourney's Animate has an
"auto" setting that writes the motion prompt itself. Open video models have no such thing: they
film what the prompt says, and a picture with no words, or with three, mostly stands still.

Dreambot fills that in the way it fills in a short picture prompt (:mod:`dreambot.core.enhance`).
For each video of a job, a model running on the Mac writes one line of *direction*: what moves,
in what order, and what the camera does. The model is one that can look at a picture: the
vision model that /describe uses. A video that starts on a picture is directed from
that picture, and so is one asked for in words, which starts on a picture made from them; a
video that is continued with new words, from its last frame (with none, it goes on from the
prompt it was made from, and nobody is asked); a video the video model films from words alone
(``--from words``), from the words. ``--motion`` is what the director is told about how much
should move. The person's own words are never rewritten or dropped: the line is added after
them.

This module is the pure part: deciding whether a video should be directed, wording the request,
checking what comes back and putting the prompt together. Running the model is the service's
job (:meth:`dreambot.core.service.Dreambot._direct`). Without a model to ask, ``--motion`` is
said to the video model in a fixed sentence (:data:`MOTION_WORDS`), which moves less.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

#: A prompt this long already says what happens; it is filmed as written.
DETAILED_WORDS = 45
#: How much of the language model's answer one line of direction may take, in tokens.
LINE_TOKENS = 120
#: A line longer than this many words has run away from what was asked; it is cut at a sentence.
MAX_WORDS = 80
#: How freely the language model writes when it has only words to go on.
TEMPERATURE = 0.7

#: How much should move in a video, as it is said to the video model when nothing better is
#: known (``--motion``). Midjourney's low motion is a camera that mostly holds still and a
#: subject that moves slowly; its high motion moves both.
MOTION_WORDS = {
    "low": "The camera holds still. The movement is slow, gentle and natural.",
    "high": "The camera moves through the scene, and everything in it is in lively motion.",
}
#: The same, after words of the person's own. What happens is then theirs to say (a fight is
#: not made gentle because the camera holds still), and only the camera is spoken for.
CAMERA_WORDS = {
    "low": "The camera holds still.",
    "high": "The camera moves with the action.",
}
#: Said after a line of direction that leaves the camera out, so that low motion keeps its promise.
STILL_CAMERA = "The camera stays still."

#: What the director is told about how much should move. Nothing here names a thing that could
#: move: a small model writes whatever it is given as an example into every line, whether or
#: not it is in the picture.
_LEVELS = {
    "low": (
        "The movements are small and slow, and everything stays where it is in the frame. The camera does "
        "not move: do not mention it."
    ),
    "high": (
        "The movement is large. If there are people or animals, one of them carries out one continuous "
        "action that takes it through the frame, and the camera moves with it, slowly, and keeps it in view "
        "the whole time. If there are none, the camera itself moves slowly forward through the scene. Say "
        "how the camera moves in the last sentence."
    ),
}
#: The same when the person has said what happens. Then that is what happens, as hard and as
#: fast as their words have it: told that movements are small, a model writes a fight as two
#: animals circling each other. Low and high are then about the camera, and about what else moves.
_LEVELS_SAID = {
    "low": (
        "What the person asked for happens with all the force and speed their words have: do not tone it "
        "down. Nothing else in the scene moves much. The camera does not move: do not mention it."
    ),
    "high": (
        "What the person asked for happens with all the force and speed their words have, and the camera "
        "moves with it and keeps it in view the whole time. Say how the camera moves in the last sentence."
    ),
}
#: A loop is filmed one way and then played backwards (see :mod:`dreambot.engine.backends.wan`).
_LOOP = (
    "The video will be played forwards and then backwards, as a loop. Describe one unhurried movement "
    "that looks natural played either way, such as a turn, a sway or a drift. Nothing leaves the frame, "
    "and nothing falls, pours or walks."
)


@dataclass(frozen=True)
class Shoot:
    """What the director is told about one job."""

    #: The person's words for what should happen, without parameters. May be empty.
    text: str
    #: How many videos, and so how many lines of direction.
    count: int
    #: ``low`` or ``high`` (``--motion``).
    motion: str = "low"
    #: The picture the video starts on, for the director to look at.
    start: str | None = None
    #: The prompt that picture was made from, when Dreambot made it: it says what the picture
    #: is of, which helps a model that is only shown it. (Midjourney's auto does the same.)
    about: str = ""
    #: The video that is being continued. The director looks at its last frame.
    clip: str | None = None
    #: What that video was made from, so that what comes next follows from it.
    before: str = ""
    #: How many seconds are being made.
    seconds: float = 5.0
    loop: bool = False
    #: What the person asked not to see (``--no``). The video model is told in the prompt.
    avoid: tuple[str, ...] = ()

    @property
    def seen(self) -> bool:
        """Whether there is a picture to show the director."""
        return bool(self.start or self.clip)


def wanted(shoot: Shoot) -> bool:
    """Whether this job should be given direction at all."""
    if shoot.count < 1:
        return False
    if not shoot.text.strip():
        if shoot.clip and _carried(shoot.before):
            # A video that is continued with nothing new said goes on from the prompt it was
            # made from, as Midjourney's Extend Auto does. (Asked for a new line instead, the
            # director wound a fight down, and wrote in a cloth that then appeared.)
            return False
        # Nothing was said: there has to be a picture to go on.
        return shoot.seen
    # A long prompt has already been directed by its author.
    return len(shoot.text.split()) < DETAILED_WORDS


def _level(shoot: Shoot) -> str:
    levels = _LEVELS_SAID if shoot.text.strip() else _LEVELS
    return levels.get(shoot.motion, levels["low"])


def _numbered(count: int) -> str:
    if count == 1:
        return "Output one line, numbered 1., and nothing else."
    return f"Output exactly {count} lines numbered 1. to {count}. and nothing else."


def instruction(shoot: Shoot) -> str:
    """What the vision model is asked, with the picture: one line for each video."""
    # Of a loop, half is filmed: the other half is that half, backwards.
    seconds = f"{max(1, round(shoot.seconds / 2 if shoot.loop else shoot.seconds))}"
    if shoot.clip:
        opening = (
            f"You write prompts for a video model. This picture is the last frame so far of a video, which "
            f"now goes on for {seconds} more seconds. "
        )
        if shoot.before.strip():
            opening += f'Up to here, this is what happened: "{_one_line(shoot.before, 400)}". '
        task = "what happens next"
    else:
        opening = (
            f"You write prompts for an image-to-video model. This picture is the first frame of a video "
            f"{seconds} seconds long. "
        )
        if shoot.about.strip():
            opening += f'The picture was made from the words "{_one_line(shoot.about, 400)}". '
        task = "what happens in the video"
    if shoot.count == 1:
        opening += f"Write one prompt that says {task}, as one line of 30 to 50 words."
    else:
        opening += (
            f"Write {shoot.count} different prompts. Each says {task}, as one line of 30 to 50 words, and is "
            "used on its own for a different video: never refer to another one."
        )
    rules = [
        "Say only what moves, and how: the people and animals in the picture, and the loose things around "
        "them that the air or the water moves. Do not describe the picture, and say nothing about what "
        "stays as it is.",
        "Name nothing that cannot be seen in the picture. Nothing new comes into it.",
        "The light, the weather and the time of day stay exactly as they are: do not mention them.",
        _level(shoot),
        'Use verbs in the present progressive ("is turning", "are swaying"), in the order things happen.',
        "Use plain words. Nothing about feelings, mood or atmosphere.",
    ]
    if shoot.loop:
        rules.append(_LOOP)
    if shoot.avoid:
        rules.append(f"Leave these out, and do not mention them: {', '.join(shoot.avoid)}.")
    asked = ""
    if shoot.text.strip():
        asked = (
            f'\nThe person asked for this: "{_one_line(shoot.text, 600)}". It is put before your line as it '
            "is, so do not repeat it. Your line says how it happens and what else moves, and never goes "
            "against it."
        )
    return (
        f"{opening}{asked}\nRules:\n"
        + "\n".join(f"- {rule}" for rule in rules)
        + f"\n{_numbered(shoot.count)}"
    )


#: For a video made from words alone: the same language model, with nothing to look at.
SYSTEM = (
    "You are the director for a text-to-video model. You are given a prompt that says what a video of a few "
    "seconds shows. For each video to be made from it you write one line of direction, which is added after "
    "the prompt. The prompt already says what the video is of. Your line says what happens in it.\n"
    "\n"
    "A line is two or three plain sentences: what moves and how, in the order it happens, with verbs in the "
    'present progressive ("is walking", "are drifting"); then the shot and what the camera does; then the '
    "light.\n"
    "\n"
    "Rules:\n"
    "- Each line is used on its own, for a different video. Never refer to another line.\n"
    "- Do not restate the prompt and do not go against it.\n"
    "- Do not add people, animals, objects or written words that the prompt does not mention, not even "
    "to watch.\n"
    "- Call everyone and everything by the name the prompt gives it, every time.\n"
    "- It is one continuous shot: no cuts, no titles, no time stamps.\n"
    "- Use plain words for what is seen. No feelings, no mood, and no praise such as stunning or beautiful.\n"
    "- Write in English, 30 to 50 words a line."
)


def request(shoot: Shoot) -> str:
    """What the language model is asked when there is no picture: the user turn after :data:`SYSTEM`."""
    lines = [f"Prompt: {_one_line(shoot.text, 900)}", "", f"How much moves: {_level(shoot)}"]
    if shoot.avoid:
        lines.append(f"Leave these out, and do not mention them: {', '.join(shoot.avoid)}.")
    many = "one line" if shoot.count == 1 else f"{shoot.count} lines"
    lines += ["", f"Write exactly {many}, numbered from 1. Nothing else."]
    return "\n".join(lines)


def parse(lines: Sequence[str], shoot: Shoot) -> list[str]:
    """The usable lines of what the model wrote, one for each video (``""`` where there is none)."""
    cleaned = [_clean(line, shoot.text) for line in lines]
    cleaned = [line for line in cleaned if len(line.split()) >= 5]
    return [cleaned[index] if index < len(cleaned) else "" for index in range(shoot.count)]


def compose(shoot: Shoot, line: str) -> str:
    """The prompt of one video: what its picture is of, the person's words, then the director's line.

    A picture Dreambot made is said to the video model in the words it was made from, before
    anything else: told only what moves, the model is free with the rest (asked for waves
    breaking at the foot of a lighthouse, it put a hand and a glass of water in front of it;
    told that it was a lighthouse on a cliff at dusk as well, it kept to that in three of four
    where it had in two). Without a line, a fixed sentence stands in for it: after words of the
    person's own it says only what the camera does, and with nothing said it says how much
    should move. A video that is continued with nothing new said goes on as it was going: what
    it was made from is said again, with what the camera does now. What to leave out is said last.
    """
    said = _sentence(shoot.text)
    scene = _sentence(_scene(shoot))
    level = MOTION_WORDS.get(shoot.motion, MOTION_WORDS["low"])
    if line:
        parts = [scene, said, _sentence(line)]
        if shoot.motion != "high" and not re.search(r"\bcamera\b", f"{shoot.text} {line}", re.IGNORECASE):
            parts.append(STILL_CAMERA)
    elif said:
        # What happens is theirs to say. Only the camera is spoken for, unless they spoke for it.
        parts = [scene, said]
        if not re.search(r"\bcamera\b", shoot.text, re.IGNORECASE):
            parts.append(CAMERA_WORDS.get(shoot.motion, CAMERA_WORDS["low"]))
    elif shoot.clip:
        # It goes on as it was going: what it was made from, and what the camera does now.
        carried = _sentence(_carried(shoot.before))
        if not carried:
            parts = [level]
        elif re.search(r"\bcamera\b", carried, re.IGNORECASE):
            parts = [carried]
        else:
            parts = [carried, CAMERA_WORDS.get(shoot.motion, CAMERA_WORDS["low"])]
    elif shoot.start:
        # The picture is all there is to go on: the model is told what it is of, where that is
        # known, and how it should move.
        parts = [scene, level]
    else:
        parts = []
    if shoot.avoid and parts:
        parts.append(f"Without {', '.join(shoot.avoid)}.")
    return " ".join(part for part in parts if part)


def _scene(shoot: Shoot) -> str:
    """The words the picture a video starts on was made from, unless the person's words say them already."""
    about, said = (
        re.sub(r"\s+", " ", text).strip().rstrip(".!?").lower() for text in (shoot.about, shoot.text)
    )
    if not shoot.start or not about or (said and (about in said or said in about)):
        return ""
    return re.sub(r"\s+", " ", shoot.about).strip()


def _carried(before: str) -> str:
    """What an earlier prompt said happens, without what it said about how much moves."""
    text = before
    for fixed in (*MOTION_WORDS.values(), *CAMERA_WORDS.values(), STILL_CAMERA):
        text = text.replace(fixed, " ")
    return re.sub(r"\s+", " ", text).strip()


def _sentence(text: str) -> str:
    base = text.strip()
    if base and base[-1] not in ".!?\"'”":
        base += "."
    return base


def _one_line(text: str, limit: int) -> str:
    return re.sub(r"\s+", " ", text).strip().replace('"', "'")[:limit]


def _unrepeated(line: str, said: str) -> str:
    """``line`` without the person's own words at its start.

    Those are put before the line as they are. Told so, a small model still opens with them
    now and then, and a prompt that says the same thing twice is a prompt with less room.
    """
    words = re.sub(r"\s+", " ", said).strip().rstrip(".!?").strip()
    if len(words.split()) < 2:
        return line
    while line.lower().startswith(words.lower()) and not line[len(words) : len(words) + 1].isalnum():
        line = line[len(words) :].lstrip(" .,;:!?—–-")
    return line


def _clean(line: str, said: str = "") -> str:
    """One line of the model's answer as it goes into a prompt. ``said`` is what the person wrote."""
    text = re.sub(r"\s+", " ", line).strip().strip('"“”').strip()
    # Small models like to label their lines ("Prompt 1:", "Video:").
    text = re.sub(r"^(?:prompt|video|line|direction)\s*\d*\s*[:\-–]\s*", "", text, flags=re.IGNORECASE)
    text = _unrepeated(text, said)
    words = text.split()
    if len(words) > MAX_WORDS:
        kept = " ".join(words[:MAX_WORDS])
        cut = max(kept.rfind(". "), kept.rfind("! "), kept.rfind("? "))
        text = kept[: cut + 1] if cut > 0 else kept
    text = text.strip()
    return text[:1].upper() + text[1:]
