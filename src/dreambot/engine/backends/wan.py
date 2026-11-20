"""Video through Dreambot's own Wan 2.2 engine (:mod:`dreambot.engine.wan`).

mflux has no video models, so this is a second engine inside the worker. It runs one set of
weights, FastMetal-5B-QAD: a video is denoised in three passes and its frames are decoded by a
small autoencoder, which together take about nine seconds for five seconds of 832x480 on an
M5 Ultra. The same three passes make a video from words, start one on a picture, and continue
a finished one.

How each kind of video is made:

* **From words**: the prompt, and noise.
* **Started on a picture**: the picture is the first frame, and is kept clean through every
  pass. The frames after it come out in the model's own look, which has more contrast and
  colour than most pictures; they are brought back to the picture's (:mod:`.wan.look`).
* **Low motion and high.** At low motion the clip starts as the picture held still under the
  noise, not as noise alone, and the model then keeps the frame where it is and moves what is
  in it. At high motion it starts as noise, and the camera goes where the model takes it
  (:mod:`.wan.sampler`). Words alone do not decide this: told that the camera holds still,
  the model moved in on two hikers all the same, ten times of ten.
* **A loop** (a video that ends on the picture it starts on) is filmed for half its length and
  then played back the way it came. Played backwards the way it came, a loop always closes.
* **Continued**: a new clip is held to the last five frames of the video, so that it goes on in
  the direction things were moving, and what comes after those five is added to the video.
  The part that was already there is not encoded again (:func:`dreambot.core.video.extended`).

The model cannot end a video on a picture: held to one at its end, it cuts to it.

Three parts are read from disk: the transformer (5.3 GB, kept loaded), the small autoencoder
(22 MB, kept loaded) and the text encoder (11 GB). The text encoder is needed for about a
second per prompt, so it is loaded to read one and let go of again, and what it read is kept:
making four videos from one prompt reads it once.
"""

from __future__ import annotations

import gc
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from dreambot import __version__
from dreambot.core import video
from dreambot.engine import weights
from dreambot.engine.backends.base import BackendError, JobContext
from dreambot.engine.protocol import ErrorKind, GenerateRequest, ImageDone, Op, Progress
from dreambot.engine.registry import ModelSpec

#: Below this much memory a video has the Mac's to itself: every other model is unloaded first.
ALONE_MEMORY_GB = 32
#: Below this much, the transformer is let go of while the text encoder reads a prompt, and
#: loaded again after (a few seconds): the two together are 17 GB.
TOGETHER_MEMORY_GB = 32
#: How many prompts' readings are kept (4 MB each).
KEPT_PROMPTS = 16
#: Latent frames of a video's end that its continuation is held to: the last frame, and the
#: four before it. From the last frame alone the model starts again from rest; with these it
#: goes on moving. (With nine, the first new frame came out blurred.)
CARRIED = 2
#: What turning a clip's latents into a video file counts for in a job's progress, in passes.
FINISH_STEPS = 1
#: At low motion, how much of what a clip starts as is noise; the rest is the picture, still.
#: With less noise than this little moves at all (at 0.94, two bears that were to fight
#: touched noses); with more, the frame starts to wander (at 0.985, twice as far).
LOW_MOTION_NOISE = 0.97


def loops(req: GenerateRequest) -> bool:
    """Whether a request asks for a loop: a video that ends on the very picture it starts on."""
    return bool(
        req.op is Op.IMG2VID and req.image and req.end_image and Path(req.image) == Path(req.end_image)
    )


class Pipeline:
    """The parts of the model that stay loaded, and what has been read of the prompts so far."""

    def __init__(self, folder: Path, autoencoder: Path):
        from dreambot.engine.wan.taehv import TinyAutoencoder

        self.folder = folder
        self.small = TinyAutoencoder.load(autoencoder)
        self.model: Any = None
        self.read: OrderedDict[str, Any] = OrderedDict()

    def transformer(self) -> Any:
        if self.model is None:
            import mlx.core as mx

            from dreambot.engine.wan.dit import Transformer

            self.model = Transformer.load(self.folder)
            # MLX reads a file when its numbers are first needed. Now, so that "loading" is over
            # when it says it is.
            mx.eval(self.model.arrays)
        return self.model

    def release(self) -> None:
        import mlx.core as mx

        self.model = None
        gc.collect()
        mx.clear_cache()


class WanVideo:
    """The video operations of the Wan model, for the MLX backend to hand requests to."""

    def __init__(self, memory_gb: float):
        self.memory_gb = memory_gb

    # ------------------------------------------------------------------ loading ----

    def load(self, spec: ModelSpec, req: GenerateRequest) -> Pipeline:
        """The model's transformer and autoencoder, in memory.

        (On a Mac where the transformer makes way for the text encoder, it is not read before
        the first prompt has been.)
        """
        from dreambot.engine.wan.dit import WanError
        from dreambot.engine.wan.taehv import CheckpointError

        missing = BackendError(
            f"The weights for {spec.id} are not installed. Run `dreambot models pull {spec.id}`.",
            ErrorKind.NOT_INSTALLED,
        )
        if not req.weights or not Path(req.weights).is_dir():
            raise missing
        if req.loras:
            raise BackendError(f"{spec.id} does not take LoRAs.", ErrorKind.INVALID)
        autoencoder = next((weights.asset_path(asset) for asset in spec.assets), None)
        if autoencoder is None:
            raise missing
        try:
            pipeline = Pipeline(Path(req.weights), autoencoder)
            if self.memory_gb >= TOGETHER_MEMORY_GB:
                pipeline.transformer()
        except (WanError, CheckpointError, FileNotFoundError) as exc:
            raise BackendError(str(exc), ErrorKind.NOT_INSTALLED) from exc
        return pipeline

    def _text(self, pipeline: Pipeline, prompt: str) -> Any:
        """What the text encoder makes of a prompt, as the transformer takes it."""
        import mlx.core as mx

        from dreambot.engine.wan.umt5 import TextEncoder

        if prompt in pipeline.read:
            pipeline.read.move_to_end(prompt)
        else:
            if self.memory_gb < TOGETHER_MEMORY_GB:
                pipeline.release()
            try:
                encoder = TextEncoder.load(pipeline.folder)
            except (FileNotFoundError, KeyError, ValueError) as exc:
                raise BackendError(
                    f"The video model's text encoder could not be read ({exc}).", ErrorKind.NOT_INSTALLED
                ) from exc
            pipeline.read[prompt] = encoder.encode([prompt]).astype("float16")
            del encoder
            gc.collect()
            mx.clear_cache()
            while len(pipeline.read) > KEPT_PROMPTS:
                pipeline.read.popitem(last=False)
        return mx.array(pipeline.read[prompt])

    # ---------------------------------------------------------------- generating ----

    def generate(self, pipeline: Pipeline, spec: ModelSpec, req: GenerateRequest, ctx: JobContext) -> None:
        """Make one video per seed."""
        import mlx.core as mx
        import numpy as np

        from dreambot.engine.wan import look, sampler

        assert req.frames and req.fps
        traits = spec.video
        assert traits is not None
        if req.end_image and not loops(req):
            raise BackendError(f"{spec.id} cannot end a video on a picture.", ErrorKind.INVALID)
        out_dir = Path(req.out_dir)
        try:
            text = self._text(pipeline, req.prompt)
            ctx.check_cancelled()
            model = pipeline.transformer()
            small = pipeline.small
            start, held, fps = None, 0, float(req.fps)
            #: The frame a low-motion clip stands still on before it is denoised.
            rest: Any = None
            width, height, count = req.width or 832, req.height or 480, req.frames
            if req.op is Op.EXTEND:
                assert req.video
                source = video.probe(req.video)
                width, height, fps = source.width, source.height, source.fps
                held = 1 + small.TIME * (CARRIED - 1)
                if source.frames < held:
                    raise BackendError(
                        f"That video is too short to continue: it has {source.frames} frames, and the next "
                        f"ones follow from its last {held}.",
                        ErrorKind.INVALID,
                    )
                self._on_grid(spec, width, height)
                # The frames to add come in groups of four after the five that are held.
                count = held + small.TIME * max(1, -(-req.frames // small.TIME))
                tail = np.frombuffer(video.pixels(req.video, last=held), dtype=np.uint8)
                tail = tail.reshape(held, height, width, 3).astype(np.float32) / 255.0
                start, rest = small.latents(tail), tail[-1]
                del tail
            else:
                self._on_grid(spec, width, height)
                if loops(req):
                    # Half the frames, and as many more as make whole groups of four.
                    count = 1 + small.TIME * -(-((count + 1) // 2 - 1) // small.TIME)
                if req.op is Op.IMG2VID:
                    assert req.image
                    rest = _fitted(req.image, width, height)
                    start, held = small.picture_latent(rest), 1
            if (count - 1) % small.TIME or count > traits.max_frames:
                raise BackendError(
                    f"{spec.id} makes videos of one frame and groups of {small.TIME}, up to "
                    f"{traits.max_frames} frames; {count} were asked for.",
                    ErrorKind.INVALID,
                )
            shape = (1 + (count - 1) // small.TIME, height // small.SPACE, width // small.SPACE)
            total = len(sampler.STEPS) + FINISH_STEPS
            standing, noise_share = None, 1.0
            if rest is not None and req.extra.get("motion", "low") != "high":
                standing, noise_share = self._standing(small, rest, start, shape[0]), LOW_MOTION_NOISE
            del rest

            for index, seed in enumerate(req.seeds):
                ctx.check_cancelled()
                started = time.perf_counter()

                def on_step(done: int, _passes: int, index: int = index) -> None:
                    ctx.emit(Progress(id=req.id, index=index, step=done, total=total))
                    ctx.check_cancelled()

                latents = sampler.sample(
                    model,
                    text,
                    shape,
                    seed=seed,
                    start=start,
                    still=standing,
                    noise_share=noise_share,
                    on_step=on_step,
                )
                frames = small.frames(latents)
                del latents
                ctx.check_cancelled()
                if held:
                    frames = look.keep(frames, held)
                notes = self._notes(spec, req, seed)
                final = out_dir / f"{req.basename}_{index}.mp4"
                still = out_dir / f"{req.basename}_{index}.png"
                if req.op is Op.EXTEND:
                    assert req.video
                    video.extended(req.video, frames[held:], final, notes=notes)
                    first = video.first_frame(final)
                else:
                    if loops(req):
                        # Out, and back the way it came: the frame it turns round on is shown once.
                        frames = np.concatenate([frames, frames[-2::-1]], axis=0)
                    video.write(
                        iter(frames), final, size=(width, height), fps=fps, quality=video.QUALITY, notes=notes
                    )
                    first = Image.fromarray(frames[0])
                del frames
                info = PngInfo()
                info.add_text("dreambot", json.dumps(notes, ensure_ascii=False))
                first.save(still, pnginfo=info)
                made = video.probe(final)
                ctx.emit(
                    ImageDone(
                        id=req.id,
                        index=index,
                        seed=seed,
                        path=str(still),
                        width=made.width,
                        height=made.height,
                        seconds=time.perf_counter() - started,
                        video=str(final),
                        frames=made.frames,
                        fps=made.fps,
                        sound=made.sound,
                    )
                )
        except video.VideoError as exc:
            raise BackendError(str(exc), ErrorKind.INTERNAL) from exc
        finally:
            mx.clear_cache()

    @staticmethod
    def _standing(small: Any, frame: Any, start: Any, frames: int) -> Any:
        """A clip's latents if nothing in it moved: what it is held to, then ``frame`` for the rest of it.

        A latent frame after the first stands for four frames. For a frame that stays as it
        is, that is the third latent of nine copies of it: by then the first frame's own
        latent, which stands for one frame alone, is out of what the encoder remembers.
        """
        import numpy as np

        kept = small.latents(np.repeat(frame[None], 9, axis=0))[:, :, 2:3]
        return np.concatenate([start, np.repeat(kept, frames - start.shape[2], axis=2)], axis=2)

    @staticmethod
    def _on_grid(spec: ModelSpec, width: int, height: int) -> None:
        if width % spec.multiple or height % spec.multiple:
            raise BackendError(
                f"{spec.id} makes videos with sides in steps of {spec.multiple} pixels, and this one would be "
                f"{width}x{height}.",
                ErrorKind.INVALID,
            )

    @staticmethod
    def _notes(spec: ModelSpec, req: GenerateRequest, seed: int) -> dict[str, Any]:
        """What a video says about itself, as Dreambot's pictures do."""
        return {
            "software": f"dreambot {__version__}",
            "model": spec.id,
            "operation": req.op.value,
            "prompt": req.prompt,
            "seed": seed,
            "frames": req.frames,
            "fps": req.fps,
        }


def _fitted(path: str, width: int, height: int) -> Any:
    """A picture at a video's size, as numbers from 0 to 1: scaled to cover it, and cut to it about its middle."""
    import numpy as np

    with Image.open(path) as opened:
        picture = opened.convert("RGB")
    scale = max(width / picture.width, height / picture.height)
    if (picture.width, picture.height) != (width, height):
        size = (max(width, round(picture.width * scale)), max(height, round(picture.height * scale)))
        picture = picture.resize(size, Image.Resampling.LANCZOS)
        left, top = (picture.width - width) // 2, (picture.height - height) // 2
        picture = picture.crop((left, top, left + width, top + height))
    return np.asarray(picture, dtype=np.float32) / 255.0
