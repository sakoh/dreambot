"""Image plumbing that is independent of any model: grids, outpainting canvases, masks, and
normalising whatever a chat app hands us into a clean PNG.

Masks follow one convention everywhere in Dreambot: **white is repainted, black is kept**.
"""

from __future__ import annotations

import math
import random
import subprocess
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter, ImageOps, ImageStat

from dreambot.core.sizes import snap

#: How far (in pixels) the repaint region reaches into the kept image, so seams can blend.
SEAM_OVERLAP = 48
#: Colour of the not-yet-painted part of an outpainting canvas.
EMPTY_FILL = (128, 128, 128)


class ImageError(Exception):
    """An input image could not be read. The message is written for the person who sent it."""


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------


def prepare_input(source: Path | str, out_dir: Path, *, max_side: int = 4096) -> Path:
    """Turn any incoming image into an upright RGB(A) PNG the models can read.

    Phone photos arrive as HEIC, rotated via EXIF and far larger than any model needs; this
    converts, applies the rotation and caps the size. HEIC is decoded by macOS's own ``sips``
    when Pillow cannot open it.
    """
    source = Path(source).expanduser()
    if not source.is_file():
        raise ImageError(f"I couldn't find the image {source.name}.")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"input-{uuid.uuid4().hex[:12]}.png"
    try:
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened)
            image.load()
    except Exception:
        image = _open_with_sips(source, out_dir)
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA" if "A" in image.getbands() or "transparency" in image.info else "RGB")
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.LANCZOS)
    image.save(target)
    return target


def _open_with_sips(source: Path, out_dir: Path) -> Image.Image:
    converted = out_dir / f"sips-{uuid.uuid4().hex[:12]}.png"
    try:
        subprocess.run(
            ["/usr/bin/sips", "-s", "format", "png", str(source), "--out", str(converted)],
            check=True,
            capture_output=True,
            timeout=60,
        )
        with Image.open(converted) as opened:
            image = ImageOps.exif_transpose(opened)
            image.load()
        return image
    except Exception as exc:
        raise ImageError(f"I couldn't read {source.name} as an image.") from exc
    finally:
        converted.unlink(missing_ok=True)


def flatten(image: Image.Image, background: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """Composite an image with transparency onto a solid background."""
    if image.mode != "RGBA":
        return image.convert("RGB")
    base = Image.new("RGB", image.size, background)
    base.paste(image, mask=image.getchannel("A"))
    return base


# --------------------------------------------------------------------------------------
# Grids
# --------------------------------------------------------------------------------------


def grid_shape(count: int) -> tuple[int, int]:
    """(columns, rows) for ``count`` images: 2x2 for four, as wide as it is tall otherwise."""
    if count <= 1:
        return 1, 1
    columns = math.ceil(math.sqrt(count))
    return columns, math.ceil(count / columns)


def compose_grid(paths: list[str | Path], out: Path, *, max_side: int = 2048, quality: int = 92) -> Path:
    """Tile images into one picture for chat previews (the full-size originals stay separate)."""
    if not paths:
        raise ValueError("compose_grid needs at least one image")
    images = [flatten(Image.open(path)) for path in paths]
    columns, rows = grid_shape(len(images))
    tile_w, tile_h = images[0].size
    scale = min(1.0, max_side / (tile_w * columns), max_side / (tile_h * rows))
    tile_w, tile_h = max(1, round(tile_w * scale)), max(1, round(tile_h * scale))
    sheet = Image.new("RGB", (tile_w * columns, tile_h * rows), (0, 0, 0))
    for position, image in enumerate(images):
        if image.size != (tile_w, tile_h):
            image = ImageOps.fit(image, (tile_w, tile_h), Image.LANCZOS)
        sheet.paste(image, ((position % columns) * tile_w, (position // columns) * tile_h))
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() in (".jpg", ".jpeg"):
        sheet.save(out, quality=quality, optimize=True)
    else:
        sheet.save(out)
    return out


# --------------------------------------------------------------------------------------
# Outpainting canvases (zoom out, pan, make square)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Canvas:
    """A source image placed on a larger canvas, ready to be outpainted."""

    #: The canvas: the source where it belongs, flat gray everywhere else.
    image: Path
    #: White where new content is needed.
    mask: Path
    #: Where the (possibly rescaled) source sits on the canvas: left, top, right, bottom.
    box: tuple[int, int, int, int]
    size: tuple[int, int]


def _build_canvas(
    source: Image.Image, size: tuple[int, int], box: tuple[int, int, int, int], out_dir: Path, tag: str
) -> Canvas:
    width, height = size
    left, top, right, bottom = box
    placed = source.resize((right - left, bottom - top), Image.LANCZOS)

    # The empty area is flat neutral gray. That is what edit models recognise as "nothing here
    # yet": tested on FLUX.2 klein, a blurred guess of the surroundings is taken for real
    # content and returned untouched, while a solid fill is painted over.
    canvas = Image.new("RGB", size, EMPTY_FILL)
    canvas.paste(placed, (left, top))

    mask = Image.new("L", size, 255)
    overlap = min(SEAM_OVERLAP, (right - left) // 8, (bottom - top) // 8)
    keep = [left, top, right, bottom]
    # Only pull the kept region in on sides that border new content.
    if left > 0:
        keep[0] += overlap
    if top > 0:
        keep[1] += overlap
    if right < width:
        keep[2] -= overlap
    if bottom < height:
        keep[3] -= overlap
    mask.paste(0, tuple(keep))

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{tag}-{uuid.uuid4().hex[:10]}"
    image_path, mask_path = out_dir / f"{stem}.png", out_dir / f"{stem}-mask.png"
    canvas.save(image_path)
    mask.save(mask_path)
    return Canvas(image=image_path, mask=mask_path, box=box, size=size)


def zoom_canvas(source_path: Path | str, factor: float, out_dir: Path, *, multiple: int = 16) -> Canvas:
    """Zoom out: the picture keeps its pixel size, its content shrinks by ``factor`` in the middle."""
    if not 1.0 < factor <= 4.0:
        raise ValueError("zoom factor must be greater than 1 and at most 4")
    source = flatten(Image.open(source_path))
    width, height = snap(source.width, multiple), snap(source.height, multiple)
    inner_w, inner_h = max(2, round(width / factor)), max(2, round(height / factor))
    left, top = (width - inner_w) // 2, (height - inner_h) // 2
    return _build_canvas(source, (width, height), (left, top, left + inner_w, top + inner_h), out_dir, "zoom")


def pan_canvas(
    source_path: Path | str,
    direction: str,
    out_dir: Path,
    *,
    amount: float = 0.5,
    max_side: int = 2560,
    multiple: int = 16,
) -> Canvas:
    """Pan: extend the canvas by ``amount`` of its size in one direction."""
    if direction not in ("left", "right", "up", "down"):
        raise ValueError(f"unknown pan direction: {direction}")
    source = flatten(Image.open(source_path))
    width, height = source.size
    if direction in ("left", "right"):
        new_w, new_h = snap(width * (1 + amount), multiple), snap(height, multiple)
    else:
        new_w, new_h = snap(width, multiple), snap(height * (1 + amount), multiple)
    # Past the size limit, shrink everything rather than refuse: the picture keeps growing in
    # field of view while staying a size the model can handle.
    shrink = min(1.0, max_side / max(new_w, new_h))
    if shrink < 1.0:
        new_w, new_h = snap(new_w * shrink, multiple), snap(new_h * shrink, multiple)
    # The source fills the canvas across the pan axis and keeps its proportions along it.
    if direction in ("left", "right"):
        placed_h = new_h
        placed_w = min(new_w - multiple, round(width * new_h / height))
    else:
        placed_w = new_w
        placed_h = min(new_h - multiple, round(height * new_w / width))
    left = new_w - placed_w if direction == "left" else 0
    top = new_h - placed_h if direction == "up" else 0
    return _build_canvas(source, (new_w, new_h), (left, top, left + placed_w, top + placed_h), out_dir, "pan")


def square_canvas(source_path: Path | str, out_dir: Path, *, multiple: int = 16) -> Canvas:
    """Make square: pad the short dimension equally on both sides."""
    source = flatten(Image.open(source_path))
    side = snap(max(source.size), multiple)
    scale = side / max(source.size)
    placed_w, placed_h = round(source.width * scale), round(source.height * scale)
    left, top = (side - placed_w) // 2, (side - placed_h) // 2
    return _build_canvas(
        source, (side, side), (left, top, left + placed_w, top + placed_h), out_dir, "square"
    )


def restore_source(result_path: Path | str, canvas: Canvas, source_path: Path | str) -> None:
    """Put the original pixels back over an outpainted result, feathering the seam.

    Outpainting models are asked to leave the kept region alone, but they re-encode it and
    drift slightly. Compositing the source back guarantees that the part of the picture the
    person already chose is bit-for-bit theirs.
    """
    result = flatten(Image.open(result_path))
    if result.size != canvas.size:
        result = result.resize(canvas.size, Image.LANCZOS)
    left, top, right, bottom = canvas.box
    source = flatten(Image.open(source_path)).resize((right - left, bottom - top), Image.LANCZOS)
    layer = Image.new("RGB", canvas.size)
    layer.paste(source, (left, top))

    keep = ImageChops.invert(Image.open(canvas.mask).convert("L"))
    result = _match_tone(result, layer, keep)
    # Blurring a hard-edged alpha would let the model's version bleed over the true edge of the
    # source, so shrink the opaque area first and blur inside the overlap band only.
    feather = max(1, SEAM_OVERLAP // 3)
    keep = keep.filter(ImageFilter.MinFilter(2 * feather + 1)).filter(ImageFilter.GaussianBlur(feather))
    Image.composite(layer, result, keep).save(result_path)


def _match_tone(
    result: Image.Image, reference: Image.Image, region: Image.Image, limit: float = 48.0
) -> Image.Image:
    """Shift ``result``'s colours so that, inside ``region``, they average the same as ``reference``.

    A model that extends a picture also re-renders the part it was given, usually a touch
    lighter, darker or warmer. Measuring that drift where both versions exist and removing it
    from the whole result is what keeps a seam from showing in skies and other flat areas.
    """
    if region.getextrema()[1] == 0:
        return result
    ours = ImageStat.Stat(result, region).mean
    theirs = ImageStat.Stat(reference, region).mean
    shifts = [max(-limit, min(limit, target - current)) for current, target in zip(ours, theirs, strict=True)]
    if all(abs(shift) < 0.5 for shift in shifts):
        return result
    channels = [
        channel.point(lambda value, shift=shift: max(0, min(255, round(value + shift))))
        for channel, shift in zip(result.split(), shifts, strict=True)
    ]
    return Image.merge("RGB", channels)


def restore_outside_mask(result_path: Path | str, source_path: Path | str, mask_path: Path | str) -> None:
    """After a region edit, put back everything outside the mask from the source image.

    Edit models asked to change "only this part" still nudge the rest of the picture. Restoring
    the untouched area makes a region edit behave like one: only the selected region changes.
    """
    result = flatten(Image.open(result_path))
    source = flatten(Image.open(source_path))
    if source.size != result.size:
        source = source.resize(result.size, Image.LANCZOS)
    mask = Image.open(mask_path).convert("L")
    if mask.size != result.size:
        mask = mask.resize(result.size, Image.BILINEAR)
    # Let the edit bleed slightly past the mask edge so the transition is not a hard line.
    feather = max(2, min(result.size) // 100)
    soft = mask.filter(ImageFilter.MaxFilter(2 * feather + 1)).filter(ImageFilter.GaussianBlur(feather))
    Image.composite(result, source, soft).save(result_path)


def enlarge(source_path: Path | str, factor: float, out: Path, *, multiple: int = 16) -> Path:
    """A plain (Lanczos) enlargement: the starting point for a creative upscale."""
    source = flatten(Image.open(source_path))
    size = (snap(source.width * factor, multiple), snap(source.height * factor, multiple))
    out.parent.mkdir(parents=True, exist_ok=True)
    source.resize(size, Image.LANCZOS).save(out)
    return out


# --------------------------------------------------------------------------------------
# Tiles (upscaling large images at the model's native resolution)
# --------------------------------------------------------------------------------------

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class Tile:
    #: Where the tile sits on the full image: left, top, right, bottom.
    box: Box
    path: Path


def _tile_spans(length: int, max_tile: int, overlap: int, multiple: int) -> list[tuple[int, int]]:
    """Cover ``length`` pixels with equally sized, evenly spaced, overlapping spans."""
    if length <= max_tile:
        return [(0, length)]
    count = math.ceil((length - overlap) / (max_tile - overlap))
    size = min(length, math.ceil((length + (count - 1) * overlap) / count / multiple) * multiple)
    step = (length - size) / (count - 1)
    return [(round(i * step), round(i * step) + size) for i in range(count)]


def tile_boxes(
    width: int, height: int, *, max_tile: int = 1280, overlap: int = 128, multiple: int = 16
) -> list[Box]:
    """Split an image into overlapping tiles no larger than ``max_tile`` on a side."""
    columns = _tile_spans(width, max_tile, overlap, multiple)
    rows = _tile_spans(height, max_tile, overlap, multiple)
    return [(left, top, right, bottom) for top, bottom in rows for left, right in columns]


def cut_tiles(source_path: Path | str, out_dir: Path, **layout: int) -> list[Tile]:
    """Cut an image into overlapping tiles and save each one."""
    source = flatten(Image.open(source_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles: list[Tile] = []
    for number, box in enumerate(tile_boxes(source.width, source.height, **layout)):
        path = out_dir / f"tile-{number:02d}.png"
        source.crop(box).save(path)
        tiles.append(Tile(box=box, path=path))
    return tiles


def blend_tiles(base_path: Path | str, tiles: list[tuple[Box, Path | str]], out: Path) -> Path:
    """Paste processed tiles back over ``base_path``, cross-fading where they overlap."""
    canvas = flatten(Image.open(base_path))
    for box, path in tiles:
        left, top, right, bottom = box
        tile = flatten(Image.open(path))
        if tile.size != (right - left, bottom - top):
            tile = tile.resize((right - left, bottom - top), Image.LANCZOS)
        ramp = max(8, min(right - left, bottom - top) // 8)
        alpha = Image.new("L", tile.size, 255)
        # Fade only the edges that border another tile; edges on the image border stay solid.
        if left > 0:
            alpha.paste(Image.linear_gradient("L").rotate(90).resize((ramp, tile.height)), (0, 0))
        if top > 0:
            fade = Image.linear_gradient("L").resize((tile.width, ramp))
            alpha.paste(ImageChops.darker(alpha.crop((0, 0, tile.width, ramp)), fade), (0, 0))
        canvas.paste(tile, (left, top), alpha)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


# --------------------------------------------------------------------------------------
# Region masks
# --------------------------------------------------------------------------------------

REGION_COLUMNS = "ABCDEF"
REGION_ROWS = 6


def parse_cells(
    spec: str, columns: int = len(REGION_COLUMNS), rows: int = REGION_ROWS
) -> set[tuple[int, int]]:
    """Parse a cell selection such as ``B2``, ``B2-D4`` or ``A1 C3-C5`` into (column, row) pairs."""
    cells: set[tuple[int, int]] = set()
    letters = REGION_COLUMNS[:columns]

    def cell(token: str) -> tuple[int, int]:
        token = token.strip().upper()
        if (
            len(token) < 2
            or token[0] not in letters
            or not token[1:].isdigit()
            or not 1 <= int(token[1:]) <= rows
        ):
            raise ValueError(f"`{token}` is not a cell between A1 and {letters[-1]}{rows}.")
        return letters.index(token[0]), int(token[1:]) - 1

    for part in spec.replace(",", " ").split():
        if "-" in part:
            first, _, last = part.partition("-")
            (c1, r1), (c2, r2) = cell(first), cell(last)
            for column in range(min(c1, c2), max(c1, c2) + 1):
                for row in range(min(r1, r2), max(r1, r2) + 1):
                    cells.add((column, row))
        else:
            cells.add(cell(part))
    if not cells:
        raise ValueError("No cells given. Use something like `B2-D4`.")
    return cells


def cells_mask(
    size: tuple[int, int],
    cells: set[tuple[int, int]],
    out: Path,
    *,
    columns: int = len(REGION_COLUMNS),
    rows: int = REGION_ROWS,
) -> Path:
    """A mask that is white over the selected cells of a ``columns`` x ``rows`` grid."""
    width, height = size
    mask = Image.new("L", size, 0)
    for column, row in cells:
        box = (
            round(column * width / columns),
            round(row * height / rows),
            round((column + 1) * width / columns),
            round((row + 1) * height / rows),
        )
        mask.paste(255, box)
    out.parent.mkdir(parents=True, exist_ok=True)
    mask.save(out)
    return out


def region_overlay(
    source_path: Path | str, out: Path, *, columns: int = len(REGION_COLUMNS), rows: int = REGION_ROWS
) -> Path:
    """The image with a labelled cell grid drawn over it, for choosing a region by text."""
    from PIL import ImageDraw, ImageFont

    image = flatten(Image.open(source_path))
    width, height = image.size
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", max(14, min(width, height) // 28))
    except OSError:
        font = ImageFont.load_default()
    line = max(1, min(width, height) // 400)
    for column in range(1, columns):
        x = round(column * width / columns)
        draw.line([(x, 0), (x, height)], fill=(255, 255, 255, 170), width=line)
    for row in range(1, rows):
        y = round(row * height / rows)
        draw.line([(0, y), (width, y)], fill=(255, 255, 255, 170), width=line)
    for column in range(columns):
        for row in range(rows):
            label = f"{REGION_COLUMNS[column]}{row + 1}"
            x = round(column * width / columns) + 6
            y = round(row * height / rows) + 4
            draw.text(
                (x, y),
                label,
                font=font,
                fill=(255, 255, 255, 235),
                stroke_width=2,
                stroke_fill=(0, 0, 0, 200),
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB").save(out, quality=90)
    return out


# --------------------------------------------------------------------------------------
# Style boards
# --------------------------------------------------------------------------------------


def style_board(
    tiles: Sequence[tuple[str | Path, float]], seed: int, out: Path, *, grid: int = 5, side: int = 1020
) -> Path:
    """Cut a board of style swatches from pictures, for a model that takes reference images.

    ``tiles`` are (picture, weight). The board is ``grid`` x ``grid`` square pieces, shuffled;
    each picture gets pieces in proportion to its weight and at least one, cut from random
    places in it. What a picture was *of* does not survive that, but its colour, line, texture
    and light do, which is the point: see :class:`dreambot.core.planner.Board`. The same tiles
    and seed always give the same board.
    """
    cells = grid * grid
    # Heaviest first, and no more pictures than there are pieces to give them.
    wanted = sorted(
        ((Path(path), weight) for path, weight in tiles if weight > 0), key=lambda item: -item[1]
    )[:cells]
    if not wanted:
        raise ImageError("There is no style picture to cut a board from.")
    total = sum(weight for _, weight in wanted)
    spare = cells - len(wanted)
    quotas = [weight / total * spare for _, weight in wanted]
    counts = [1 + int(quota) for quota in quotas]
    # The pieces left over by rounding down go to the largest remainders.
    by_remainder = sorted(range(len(quotas)), key=lambda index: (int(quotas[index]) - quotas[index], index))
    for index in by_remainder[: cells - sum(counts)]:
        counts[index] += 1

    rng = random.Random(f"board|{seed}")
    piece = side // grid
    pieces: list[Image.Image] = []
    for (path, _), count in zip(wanted, counts, strict=True):
        try:
            with Image.open(path) as opened:
                picture = flatten(opened)
        except (OSError, ValueError) as exc:
            raise ImageError(f"I couldn't read the style picture {path.name}.") from exc
        scale = side / min(picture.size)
        picture = picture.resize(
            (max(piece, round(picture.width * scale)), max(piece, round(picture.height * scale))),
            Image.LANCZOS,
        )
        for _ in range(count):
            left = rng.randrange(picture.width - piece + 1)
            top = rng.randrange(picture.height - piece + 1)
            pieces.append(picture.crop((left, top, left + piece, top + piece)))
    rng.shuffle(pieces)
    board = Image.new("RGB", (piece * grid, piece * grid))
    for position, cut in enumerate(pieces):
        board.paste(cut, ((position % grid) * piece, (position // grid) * piece))
    out.parent.mkdir(parents=True, exist_ok=True)
    board.save(out)
    return out
