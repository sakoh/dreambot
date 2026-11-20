"""Video plumbing that is independent of any model: what a video file holds, its first frame,
and writing one.

All of it is done by ``ffmpeg``, and by the one that is installed with Dreambot (the
``imageio-ffmpeg`` package carries the program), so that nothing has to be set up on the Mac
and every Dreambot writes its videos the same way: H.264 in an MP4, which Messages, Telegram,
Discord and every browser play.
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

#: Seconds a single ffmpeg run on a short video may take before it is given up on.
TIMEOUT = 300.0
#: Quality of the videos Dreambot writes (x264's scale: lower is better and larger; 18 is
#: visually lossless, 23 the encoder's own default).
QUALITY = 18
#: Quality of the stand-in backend's placeholder videos, which are written as fast as can be.
DRAFT_QUALITY = 23

_STREAM = re.compile(r"Stream #\d+:\d+[^\n]*?: Video: [^\n]*?\b(\d{2,5})x(\d{2,5})\b[^\n]*")
_RATE = re.compile(r"\b([\d.]+) fps\b")
_FRAMES = re.compile(r"frame=\s*(\d+)")
_SOUND = re.compile(r"Stream #\d+:\d+[^\n]*?: Audio:")


class VideoError(Exception):
    """A video could not be read or written. The message is written for the person who asked."""


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    frames: int
    fps: float
    #: Whether it has a soundtrack.
    sound: bool = False

    @property
    def seconds(self) -> float:
        return self.frames / self.fps if self.fps else 0.0


def ffmpeg() -> str:
    """The ffmpeg program Dreambot uses."""
    import imageio_ffmpeg

    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except RuntimeError as exc:  # the package is there but its program is not
        raise VideoError("Dreambot's copy of ffmpeg is missing. Run `uv sync` in its folder.") from exc


def _run(arguments: list[str]) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            [ffmpeg(), "-hide_banner", "-nostdin", *arguments],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise VideoError("ffmpeg did not finish in time.") from exc


def probe(path: Path | str) -> VideoInfo:
    """What a video file holds: its size, how many frames at how many a second, and whether it has sound.

    The frames are counted, not read from the file's own header, which is allowed to be wrong.
    """
    source = Path(path)
    if not source.is_file():
        raise VideoError(f"I couldn't find the video {source.name}.")
    report = _run(["-i", str(source), "-map", "0:v:0", "-c", "copy", "-f", "null", "-"]).stderr.decode(
        errors="replace"
    )
    stream, counted = _STREAM.search(report), _FRAMES.findall(report)
    rate = _RATE.search(stream.group(0)) if stream else None
    if stream is None or rate is None or not counted or int(counted[-1]) < 1:
        raise VideoError(f"I couldn't read {source.name} as a video.")
    return VideoInfo(
        width=int(stream.group(1)),
        height=int(stream.group(2)),
        frames=int(counted[-1]),
        fps=float(rate.group(1)),
        sound=bool(_SOUND.search(report)),
    )


def first_frame(path: Path | str) -> Image.Image:
    """The picture a video starts on."""
    source = Path(path)
    result = _run(["-i", str(source), "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"])
    try:
        with Image.open(io.BytesIO(result.stdout)) as opened:
            return opened.convert("RGB")
    except (OSError, ValueError) as exc:
        raise VideoError(f"I couldn't read a frame of {source.name}.") from exc


def frame(path: Path | str, number: int) -> Image.Image:
    """Frame ``number`` of a video, counting from 0."""
    source = Path(path)
    picked = ["-vf", f"select=eq(n\\,{max(0, number)})", "-frames:v", "1"]
    result = _run(["-i", str(source), *picked, "-f", "image2pipe", "-vcodec", "png", "-"])
    try:
        with Image.open(io.BytesIO(result.stdout)) as opened:
            return opened.convert("RGB")
    except (OSError, ValueError) as exc:
        raise VideoError(f"I couldn't read a frame of {source.name}.") from exc


def pixels(path: Path | str, *, last: int | None = None) -> bytes:
    """The frames of a video as raw 8-bit RGB, one after another. ``last`` gives only that many, from its end."""
    source = Path(path)
    picked: list[str] = []
    if last is not None:
        # The frames that are kept are timed from nought again: left at their own times, the
        # gap before them would be filled with copies of the first.
        first = max(0, probe(source).frames - last)
        picked = ["-vf", f"select=gte(n\\,{first}),setpts=PTS-STARTPTS"]
    result = _run(["-i", str(source), *picked, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    if result.returncode != 0 or not result.stdout:
        raise VideoError(f"I couldn't read the frames of {source.name}.")
    return result.stdout


def last_frame(path: Path | str) -> Image.Image:
    """The picture a video ends on."""
    source = Path(path)
    with tempfile.TemporaryDirectory(prefix="dreambot-frame-") as folder:
        out = Path(folder) / "last.png"
        # Every frame of the last second is written over the one before; what is left is the last.
        _run(["-sseof", "-1", "-i", str(source), "-update", "1", "-y", str(out)])
        try:
            with Image.open(out) as opened:
                return opened.convert("RGB")
        except (OSError, ValueError) as exc:
            raise VideoError(f"I couldn't read a frame of {source.name}.") from exc


def write(
    frames: Iterable[Any],
    out: Path,
    *,
    size: tuple[int, int],
    fps: float,
    tone: float | None = None,
    quality: int = DRAFT_QUALITY,
    notes: Mapping[str, object] | None = None,
) -> Path:
    """Write pictures as the frames of an MP4.

    A frame is a picture, or its pixels as they lie in memory: anything that gives ``height *
    width * 3`` bytes of RGB (an array of 8-bit numbers, say). ``tone`` adds a soundtrack: a
    note of that pitch (Hz), which is what the stand-in backend's videos have for sound.
    ``quality`` is x264's (see :data:`QUALITY`). ``notes`` (the prompt, the seed, the model)
    are written into the file, the way they are into Dreambot's PNGs.
    """
    width, height = size
    if width % 2 or height % 2:
        raise VideoError("A video's width and height have to be even.")
    command = [ffmpeg(), "-hide_banner", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24"]
    command += ["-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-"]
    if tone is not None:
        # The note runs on for ever; `-shortest` ends it with the picture.
        command += ["-f", "lavfi", "-i", f"sine=frequency={tone:g}", "-c:a", "aac", "-shortest"]
    preset = "veryfast" if quality >= DRAFT_QUALITY else "medium"
    command += ["-c:v", "libx264", "-preset", preset, "-pix_fmt", "yuv420p", "-crf", str(quality)]
    if notes:
        command += ["-metadata", f"comment={json.dumps(dict(notes), ensure_ascii=False)}"]
    command += ["-movflags", "+faststart", str(out)]
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with tempfile.TemporaryFile() as report:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=report)
        assert process.stdin is not None
        try:
            for picture in frames:
                if isinstance(picture, Image.Image):
                    if picture.size != size or picture.mode != "RGB":
                        picture = picture.convert("RGB").resize(size)
                    pixels = picture.tobytes()
                else:
                    pixels = memoryview(picture).cast("B")
                    if len(pixels) != width * height * 3:
                        raise VideoError("A frame is not the size of the video it is for.")
                process.stdin.write(pixels)
                count += 1
            process.stdin.close()
            code = process.wait(timeout=TIMEOUT)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            code = -1
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        report.seek(0)
        said = report.read()
    if code != 0 or not count or not out.is_file():
        out.unlink(missing_ok=True)
        why = _last_line(said) if count else "there were no frames to make it of"
        raise VideoError(f"The video could not be written ({why}).")
    return out


def extended(
    source: Path | str, frames: Sequence[Any], out: Path, *, notes: Mapping[str, object] | None = None
) -> Path:
    """``source`` with ``frames`` after it, as one video.

    What is already there is not encoded again where that can be helped (every encoding
    loses a little, and a video may be continued four times). The new frames are written as a
    video of their own, the way Dreambot writes every video, and the two are put end to end as
    they are. That is done when ``source`` was written the same way, which is looked at and
    not assumed (:func:`_setup`), and has no sound. Otherwise the two are encoded again as
    one, and what sound there is stays with the part it belongs to.
    """
    source = Path(source)
    info = probe(source)
    wanted = info.frames + len(frames)
    out.parent.mkdir(parents=True, exist_ok=True)
    noted = ["-metadata", f"comment={json.dumps(dict(notes), ensure_ascii=False)}"] if notes else []
    with tempfile.TemporaryDirectory(prefix="dreambot-extend-") as folder:
        added = write(
            frames, Path(folder) / "added.mp4", size=(info.width, info.height), fps=info.fps, quality=QUALITY
        )
        setup = _setup(source)
        if not info.sound and setup is not None and setup == _setup(added):
            listing = Path(folder) / "parts.txt"
            parts = (str(source.resolve()), str(added))
            # (In such a list a quote inside a name is closed, escaped and opened again.)
            listing.write_text("".join("file '{}'\n".format(part.replace("'", "'\\''")) for part in parts))
            arguments = ["-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", *noted]
            result = _run([*arguments, "-movflags", "+faststart", str(out)])
            if result.returncode == 0 and out.is_file() and probe(out).frames == wanted:
                return out
            out.unlink(missing_ok=True)
        arguments = ["-y", "-i", str(source), "-i", str(added)]
        arguments += ["-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[picture]", "-map", "[picture]"]
        if info.sound:
            arguments += ["-map", "0:a", "-c:a", "copy"]
        arguments += ["-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p", "-crf", str(QUALITY)]
        result = _run([*arguments, "-r", f"{info.fps:g}", *noted, "-movflags", "+faststart", str(out)])
    if result.returncode != 0 or not out.is_file():
        out.unlink(missing_ok=True)
        raise VideoError(f"The video could not be made longer ({_last_line(result.stderr)}).")
    return out


def _setup(path: Path) -> bytes | None:
    """How a video's pictures are encoded: the H.264 parameter sets in its header (its ``avcC`` box).

    Two videos with the same ones can be put end to end without encoding either again. A
    player reads them once, at the start, and decodes everything after by them.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return None
    at = data.find(b"avcC")
    if at < 4:
        return None
    size = int.from_bytes(data[at - 4 : at], "big")
    return data[at + 4 : at - 4 + size] if 8 < size < 4096 else None


def gif(source: Path | str, out: Path, *, width: int = 480, fps: int = 12) -> Path:
    """A video as an animated GIF: smaller, slower and with far fewer colours, and it plays anywhere.

    The colours are chosen for this video (a GIF has 256 of them), which is what keeps it from
    looking like a newspaper photograph.
    """
    look = f"fps={fps},scale={width}:-2:flags=lanczos"
    graph = f"[0:v]{look},split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
    out.parent.mkdir(parents=True, exist_ok=True)
    result = _run(["-y", "-i", str(source), "-filter_complex", graph, "-loop", "0", str(out)])
    if result.returncode != 0 or not out.is_file():
        out.unlink(missing_ok=True)
        raise VideoError(f"The GIF could not be made ({_last_line(result.stderr)}).")
    return out


#: Bits a second the sound of a shrunk video is given, and the least its picture is worth sending at.
FIT_SOUND_RATE = 96_000
FIT_LEAST_RATE = 100_000


def fit(source: Path | str, out: Path, *, limit: int) -> Path:
    """A copy of a video that is at most ``limit`` bytes, for a chat app that takes no more.

    The picture is encoded again at the bit rate that fills the limit, so it loses quality;
    the video itself stays as it was on the Mac. An encoder aims at a bit rate and can miss
    it, most of all on a short video, so a copy that comes out too large is made again smaller.
    """
    info = probe(source)
    # A tenth is left for the file's own bookkeeping.
    rate = int(limit * 8 * 0.9 / max(info.seconds, 0.1)) - (FIT_SOUND_RATE if info.sound else 0)
    if rate < FIT_LEAST_RATE:
        raise VideoError("That video is too long to be made small enough to send here.")
    out.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(4):
        arguments = ["-y", "-i", str(source), "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p"]
        arguments += ["-b:v", str(rate), "-maxrate", str(rate), "-bufsize", str(rate)]
        arguments += ["-c:a", "aac", "-b:a", str(FIT_SOUND_RATE)] if info.sound else ["-an"]
        result = _run([*arguments, "-movflags", "+faststart", str(out)])
        if result.returncode != 0 or not out.is_file():
            break
        size = out.stat().st_size
        if size <= limit:
            return out
        rate = int(rate * limit / size * 0.85)
        if rate < FIT_LEAST_RATE // 2:
            break
    out.unlink(missing_ok=True)
    raise VideoError("That video could not be made small enough to send here.")


def _last_line(report: bytes) -> str:
    lines = [line.strip() for line in report.decode(errors="replace").splitlines() if line.strip()]
    return lines[-1][:200] if lines else "ffmpeg said nothing"
