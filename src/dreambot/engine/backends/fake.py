"""A deterministic stand-in backend that draws placeholder images with Pillow.

It implements the full backend contract — progress, previews, early stop, cancellation, every
operation, videos included — without loading any model, which makes the rest of Dreambot
testable in milliseconds and lets people try the bots before downloading weights
(``DREAMBOT_BACKEND=fake``).
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from dreambot.core import video
from dreambot.engine.backends.base import BackendError, JobContext, input_files, require_inputs
from dreambot.engine.protocol import (
    VIDEO_OPS,
    DescribeRequest,
    ErrorKind,
    GenerateRequest,
    ImageDone,
    LoadedModel,
    Op,
    Preview,
    Progress,
    StatusReport,
    TextRequest,
)
from dreambot.engine.registry import REGISTRY, UnknownModel


def _color(*parts: object) -> tuple[int, int, int]:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return digest[0], digest[1], digest[2]


class FakeBackend:
    name = "fake"

    def __init__(self) -> None:
        #: Seconds to sleep per simulated denoising step.
        self.step_delay = float(os.environ.get("DREAMBOT_FAKE_STEP_DELAY", "0"))
        self._loaded: dict[str, float] = {}

    def versions(self) -> dict[str, str]:
        return {"backend": "fake"}

    # ------------------------------------------------------------------ generate ----

    def generate(self, req: GenerateRequest, ctx: JobContext) -> None:
        try:
            spec = REGISTRY.get(req.model)
        except UnknownModel as exc:
            raise BackendError(str(exc), ErrorKind.INVALID) from exc
        variant = spec.variants.get(req.op)
        if variant is None:
            raise BackendError(f"{spec.id} does not support '{req.op.value}'.", ErrorKind.INVALID)
        require_inputs(req, variant.max_refs)
        for path in input_files(req):
            if not Path(path).is_file():
                raise BackendError(f"Input file not found: {path}", ErrorKind.INVALID)
        self._loaded[spec.id] = time.monotonic()
        if os.environ.get("DREAMBOT_FAKE_NOISE"):
            # Real model code prints to stdout (progress bars, warnings). Tests use this to
            # prove that such output cannot corrupt the protocol stream.
            print("noise on stdout {not json}", flush=True)
            os.write(1, b"raw noise on fd 1\n")

        out_dir = Path(req.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        total = req.steps or spec.steps
        if req.op in VIDEO_OPS:
            self._videos(req, ctx, out_dir, total)
            return
        stop_step = max(1, round(total * req.stop_at)) if req.stop_at else total

        for index, seed in enumerate(req.seeds):
            started = time.perf_counter()
            image = self._render(req, seed)
            for step in range(1, stop_step + 1):
                ctx.check_cancelled()
                if self.step_delay:
                    time.sleep(self.step_delay)
                ctx.emit(Progress(id=req.id, index=index, step=step, total=total))
                if req.preview_every and step % req.preview_every == 0 and step < stop_step:
                    preview_path = out_dir / f"{req.basename}_{index}_preview.png"
                    image.filter(ImageFilter.GaussianBlur(24 * (1 - step / total))).save(preview_path)
                    ctx.emit(Preview(id=req.id, index=index, step=step, total=total, path=str(preview_path)))
            if stop_step < total:
                image = image.filter(ImageFilter.GaussianBlur(24 * (1 - stop_step / total)))
            path = out_dir / f"{req.basename}_{index}.png"
            image.save(path)
            ctx.emit(
                ImageDone(
                    id=req.id,
                    index=index,
                    seed=seed,
                    path=str(path),
                    width=image.width,
                    height=image.height,
                    seconds=time.perf_counter() - started,
                )
            )

    def _videos(self, req: GenerateRequest, ctx: JobContext, out_dir: Path, total: int) -> None:
        """One placeholder video per seed: the picture it starts on, with something moving across it."""
        assert req.frames and req.fps
        before = 0
        size: tuple[int, int] | None = None
        try:
            if req.op is Op.EXTEND:
                # The video as it was, then the frames that are added to it.
                assert req.video
                source = video.probe(req.video)
                before, size = source.frames, (source.width, source.height)
                opening = video.first_frame(req.video)
            else:
                opening = None
        except video.VideoError as exc:
            raise BackendError(str(exc), ErrorKind.INVALID) from exc
        for index, seed in enumerate(req.seeds):
            started = time.perf_counter()
            for step in range(1, total + 1):
                ctx.check_cancelled()
                if self.step_delay:
                    time.sleep(self.step_delay)
                ctx.emit(Progress(id=req.id, index=index, step=step, total=total))
            scene = self._render(req, seed)
            if size is None:
                size = scene.size
            first = opening or (Image.open(req.image).convert("RGB").resize(size) if req.image else scene)
            last = Image.open(req.end_image).convert("RGB").resize(size) if req.end_image else None
            count = before + req.frames
            path = out_dir / f"{req.basename}_{index}.mp4"
            try:
                video.write(
                    self._frames(first, scene.resize(size), last, before, count),
                    path,
                    size=size,
                    fps=req.fps,
                )
            except video.VideoError as exc:
                raise BackendError(str(exc), ErrorKind.INTERNAL) from exc
            still = out_dir / f"{req.basename}_{index}.png"
            first.save(still)
            ctx.emit(
                ImageDone(
                    id=req.id,
                    index=index,
                    seed=seed,
                    path=str(still),
                    width=size[0],
                    height=size[1],
                    seconds=time.perf_counter() - started,
                    video=str(path),
                    frames=count,
                    fps=req.fps,
                )
            )

    @staticmethod
    def _frames(first: Image.Image, scene: Image.Image, last: Image.Image | None, still: int, count: int):
        """``count`` frames: ``first`` held for ``still`` of them, then moving into ``scene`` (and on to ``last``)."""
        width, height = scene.size
        side = max(8, min(width, height) // 6)
        moving = max(1, count - still)
        for number in range(count):
            if number < still or count == 1:
                yield first
                continue
            along = (number - still) / max(1, moving - 1)
            if last is not None and number == count - 1:
                frame = last.copy()
            else:
                frame = Image.blend(first, scene, min(1.0, 0.25 + 0.75 * along)) if number else first.copy()
            # A square that crosses the picture, so that no two frames are alike.
            left = round(along * (width - side))
            ImageDraw.Draw(frame).rectangle((left, height - side, left + side, height), fill=(255, 255, 255))
            yield frame

    def _render(self, req: GenerateRequest, seed: int) -> Image.Image:
        if req.op is Op.UPSCALE:
            assert req.image
            source = Image.open(req.image).convert("RGB")
            scale = float(req.extra.get("scale", 2))
            return source.resize((round(source.width * scale), round(source.height * scale)), Image.LANCZOS)

        source: Image.Image | None = None
        first = req.image or (req.images[0] if req.images else None)
        if first:
            source = Image.open(first).convert("RGB")
        width = req.width or (source.width if source else 1024)
        height = req.height or (source.height if source else 1024)

        top, bottom = _color(req.prompt, seed, "a"), _color(req.prompt, seed, "b")
        image = Image.new("RGB", (width, height), top)
        draw = ImageDraw.Draw(image)
        for y in range(height):
            t = y / max(1, height - 1)
            draw.line(
                [(0, y), (width, y)],
                fill=tuple(round(top[c] * (1 - t) + bottom[c] * t) for c in range(3)),
            )
        if source is not None:
            keep = (
                1.0 - req.noise
                if req.noise is not None
                else (req.strength if req.strength is not None else 0.5)
            )
            image = Image.blend(image, source.resize((width, height)), keep)
            draw = ImageDraw.Draw(image)
        if req.op is Op.INPAINT and req.mask and source is not None:
            mask = Image.open(req.mask).convert("L").resize((width, height))
            image = Image.composite(image, source.resize((width, height)), mask)
            draw = ImageDraw.Draw(image)
        label = f"{req.model} · {req.op.value} · seed {seed}\n{req.prompt[:160]}"
        draw.multiline_text((24, 24), label, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        return image

    # ---------------------------------------------------------------- text tasks ----

    def describe(self, req: DescribeRequest, ctx: JobContext) -> list[str]:
        if not Path(req.image).is_file():
            raise BackendError(f"Input image not found: {req.image}", ErrorKind.INVALID)
        with Image.open(req.image) as image:
            width, height = image.size
            red, green, blue = image.convert("RGB").resize((1, 1)).getpixel((0, 0))
        if req.instruction and "medium:" in req.instruction:
            # A request for a style fingerprint: answer in kind, differently for different pictures.
            media = ("gouache illustration", "35mm film photograph", "linocut print", "pencil drawing")
            return [
                f"medium: {media[(red + green + blue) % len(media)]}",
                "rendering: placeholder rendering",
                "palette: placeholder colors",
                "light: soft placeholder light",
                "texture: placeholder texture",
                "mood: placeholder mood",
            ][: req.count]
        if req.instruction and "video" in req.instruction:
            # A request for direction: what happens in each video that starts (or goes on) from this picture.
            return [
                f"Placeholder movement {i + 1}: something in the {width}x{height} picture is slowly turning"
                for i in range(req.count)
            ]
        return [f"a placeholder description #{i + 1} of a {width}x{height} image" for i in range(req.count)]

    def text(self, req: TextRequest, ctx: JobContext) -> list[str]:
        asked = re.search(r"Write exactly (\d+|one) lines?", req.prompt)
        if req.system and asked:
            # A request for numbered lines (the prompt enhancer): answer in kind.
            count = 1 if asked.group(1) == "one" else int(asked.group(1))
            return [
                "\n".join(
                    f"{n}. placeholder shot {n} from a low angle, soft window light, muted earth tones"
                    for n in range(1, count + 1)
                )
            ]
        return [f"[fake llm] {req.prompt[: req.max_tokens]}"]

    # ------------------------------------------------------------------- memory ----

    def unload(self, model: str | None) -> None:
        if model is None:
            self._loaded.clear()
        else:
            self._loaded.pop(model, None)

    def status(self, request_id: str) -> StatusReport:
        now = time.monotonic()
        return StatusReport(
            id=request_id,
            loaded=[
                LoadedModel(key=name, model=name, idle_seconds=now - at) for name, at in self._loaded.items()
            ],
        )
