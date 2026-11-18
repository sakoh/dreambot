"""Words and button sets shared by every frontend, so Dreambot reads the same everywhere."""

from __future__ import annotations

from dreambot.core.actions import Action, ActionKind
from dreambot.core.jobs import Job, JobImage, JobKind, JobStatus
from dreambot.core.params import PARAMS, Support

KIND_LABELS = {
    JobKind.IMAGINE: "",
    JobKind.VARIATION: "Variations",
    JobKind.UPSCALE: "Upscaled",
    JobKind.ZOOM: "Zoomed out",
    JobKind.PAN: "Panned",
    JobKind.EDIT: "Edited",
    JobKind.BLEND: "Blend",
    JobKind.DESCRIBE: "Describe",
    JobKind.SHORTEN: "Shorten",
}


def job_title(job: Job, limit: int = 300) -> str:
    """One line naming a job: its prompt, trimmed."""
    prompt = job.prompt.strip() or job.text.strip() or job.kind.value
    if len(prompt) > limit:
        prompt = prompt[: limit - 1].rstrip() + "…"
    return prompt


def job_caption(job: Job, *, markdown: bool = False) -> str:
    """The caption under a finished job, in the spirit of Midjourney's."""
    bold = "**" if markdown else ""
    label = KIND_LABELS.get(job.kind, "")
    detail = job.detail or {}
    if job.kind is JobKind.VARIATION and detail.get("mode"):
        label = f"Variations ({detail['mode'].title()})"
    elif job.kind is JobKind.UPSCALE and detail.get("mode"):
        label = f"Upscaled ({detail['mode'].title()})"
    elif job.kind is JobKind.ZOOM:
        label = "Made square" if detail.get("square") else f"Zoomed out {detail.get('factor', 2):g}x"
    elif job.kind is JobKind.PAN and detail.get("direction"):
        label = f"Panned {detail['direction']}"

    facts = [job.model] if job.model else []
    if job.images:
        facts.append(f"{job.images[0].width}×{job.images[0].height}")
    if job.seconds is not None:
        facts.append(f"{job.seconds:.0f}s")
    facts.append(f"#{job.id}")

    head = f"{bold}{job_title(job)}{bold}"
    if label:
        head = f"{head} — {label}"
    lines = [head, " · ".join(facts)]
    lines.extend(f"ⓘ {note}" for note in job.notes)
    return "\n".join(lines)


def failure_text(job: Job) -> str:
    if job.status is JobStatus.CANCELLED:
        return f"Cancelled: {job_title(job, 120)}"
    return f"Couldn't finish “{job_title(job, 120)}”: {job.error or 'unknown error'}"


def progress_text(job: Job, fraction: float, *, loading: bool = False, thinking: bool = False) -> str:
    title = job_title(job, 120)
    if thinking:
        # With a style reference or a Lookbook, the vision model reads the pictures before any
        # art direction is written: "working out the look" is true of both.
        styled = job.params.get("sref") or job.params.get("personalize") or job.detail.get("style")
        return f"{'Working out the look' if styled else 'Art-directing'}… — {title}"
    if loading:
        return f"Loading {job.model}… — {title}"
    return f"{round(fraction * 100)}% — {title}"


def queued_text(job: Job, position: int) -> str:
    if position <= 0:
        return f"Starting — {job_title(job, 120)}"
    return f"Queued (#{position + 1} in line) — {job_title(job, 120)}"


# --------------------------------------------------------------------------------------
# Buttons
# --------------------------------------------------------------------------------------


def grid_actions(job: Job) -> list[list[tuple[str, Action]]]:
    """Rows of (label, action) under a grid: U1-U4 / V1-V4 / reroll, like Midjourney."""
    numbers = [image.index for image in job.images]
    rows: list[list[tuple[str, Action]]] = []
    for start in range(0, len(numbers), 4):
        chunk = numbers[start : start + 4]
        rows.append([(f"U{n}", Action(ActionKind.SELECT, job.id, n)) for n in chunk])
    rows[0].append(("🔄", Action(ActionKind.REROLL, job.id)))
    for start in range(0, len(numbers), 4):
        chunk = numbers[start : start + 4]
        rows.append([(f"V{n}", Action(ActionKind.VARY, job.id, n)) for n in chunk])
    return rows


def image_actions(job: Job, image: JobImage) -> list[list[tuple[str, Action]]]:
    """Rows of (label, action) under a single image."""

    def a(kind: ActionKind, value: str | None = None) -> Action:
        return Action(kind, job.id, image.index, value)

    rows = [
        [
            ("Upscale (Subtle)", a(ActionKind.UPSCALE_SUBTLE)),
            ("Upscale (Creative)", a(ActionKind.UPSCALE_CREATIVE)),
        ],
        [
            ("Vary (Subtle)", a(ActionKind.VARY_SUBTLE)),
            ("Vary (Strong)", a(ActionKind.VARY_STRONG)),
            ("Vary (Region)", a(ActionKind.VARY_REGION)),
        ],
        [
            ("Zoom Out 2x", a(ActionKind.ZOOM, "2")),
            ("Zoom Out 1.5x", a(ActionKind.ZOOM, "1.5")),
            ("Custom Zoom", a(ActionKind.ZOOM, "custom")),
        ],
        [
            ("⬅️", a(ActionKind.PAN, "left")),
            ("➡️", a(ActionKind.PAN, "right")),
            ("⬆️", a(ActionKind.PAN, "up")),
            ("⬇️", a(ActionKind.PAN, "down")),
        ],
    ]
    last = [("❤️", a(ActionKind.FAVORITE)), ("Edit", a(ActionKind.EDIT))]
    if image.width != image.height:
        last.insert(0, ("Make Square", a(ActionKind.SQUARE)))
    rows.append(last)
    if job.kind is JobKind.UPSCALE:
        # An upscaled image is a finished piece: offer what still makes sense on it.
        rows = [rows[1][:2], rows[-1]]
    return rows


# --------------------------------------------------------------------------------------
# Help
# --------------------------------------------------------------------------------------

TEXT_SHORTCUTS = """\
After a grid arrives, reply with:
  U1–U4            pull one image out at full size
  V1–V4            variations of one image (V2: new prompt — to remix)
  R                reroll: same prompt, new images
With one image picked (add its number when replying to a grid):
  upscale [creative]        double its size
  vary subtle | vary strong
  zoom 2x | zoom 1.5x       zoom out
  pan left|right|up|down    extend the picture that way
  square                    fill it out to a square
  region B2-D4: a red balloon     repaint part of it (send `region` alone to see the cells)
  region the sky: an aurora       ...or describe the part instead
  edit: make it night       change it by instruction
  look: autumn diner        add it to that Lookbook (a set of pictures used with --p)
  fav · info · cancel"""


def help_text(*, direct: bool, slash: str = "/") -> str:
    intro = (
        "Send me a description and I'll make four images of it."
        if direct
        else f"Use {slash}imagine followed by a description and I'll make four images of it."
    )
    return f"""\
Dreambot — image generation that runs entirely on this Mac.

{intro}
Add Midjourney-style parameters at the end, for example:
  a red fox in the snow --ar 16:9 --stylize 250 --no people
Short prompts are art-directed for you (a different shot for each image); --raw uses yours as written.

Attach an image to use it as an image prompt, or steer with it:
  --edit (change it by instruction) · --oref (keep this person/object) · --sref (borrow its style)
Styles blend: --sref takes several images, style codes and saved styles, each with an optional ::weight.
A LoRA is one fine-tuned look (pixel art, anime): --lora <name>, or {slash}settings lora <name> for every prompt.

{TEXT_SHORTCUTS}

Commands:
  {slash}imagine <prompt>     {slash}describe (with an image)     {slash}blend (with 2–5 images)
  {slash}shorten <prompt>     {slash}edit (with an image)         {slash}models
  {slash}settings             {slash}prefer suffix|option ...       {slash}looks
  {slash}info                 {slash}show <job id>                 {slash}params
  {slash}cancel               {slash}help"""


def params_text() -> str:
    """The parameter list with how each one is realised locally."""
    marks = {Support.NATIVE: "●", Support.EMULATED: "◐", Support.ACCEPTED: "○", Support.PLANNED: "…"}
    lines = ["● native   ◐ approximated locally   ○ accepted, no effect   … planned", ""]
    for spec in PARAMS:
        names = ", ".join(f"--{name}" for name in (spec.name, *spec.aliases))
        suffix = " (Dreambot extension)" if spec.extension else ""
        lines.append(f"{marks[spec.support]} {names}{suffix}\n    {spec.summary}")
        if spec.how:
            lines.append(f"    ↳ {spec.how}")
    return "\n".join(lines)
