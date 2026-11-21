"""Planning decisions: which model does what, and what happens when the ideal one is missing."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from PIL import Image

from dreambot.config import Config
from dreambot.core import imaging, styles
from dreambot.core.params import Params
from dreambot.core.planner import Catalog, PlanError, Planner, ResolvedRefs, reference_instruction
from dreambot.core.prompt import parse_prompt
from dreambot.core.settings import UserSettings
from dreambot.core.sizes import aspect_of, dimensions, fit_within, working_size
from dreambot.engine.protocol import Op


class Shelf(Catalog):
    """A catalog in which exactly the named models are installed."""

    def __init__(self, config: Config, *installed: str):
        super().__init__(config)
        self._installed = set(installed)

    def weights(self, spec):
        return None

    def is_installed(self, spec) -> bool:
        return spec.id in self._installed


def planner(tmp_path: Path, *installed: str, **config) -> Planner:
    cfg = Config.model_validate(
        {"home": str(tmp_path / "home"), "output_dir": str(tmp_path / "out"), **config}
    )
    return Planner(Shelf(cfg, *installed), cfg, rng=random.Random(1))


# The enhancer's default depends on the Mac's memory; planning tests pin it off.
SETTINGS = UserSettings(enhance=False).resolved(Config().defaults)


def imagine(p: Planner, prompt: str, refs: ResolvedRefs | None = None):
    return p.imagine(parse_prompt(prompt), SETTINGS, refs)


@pytest.fixture
def picture(tmp_path: Path) -> str:
    path = tmp_path / "picture.png"
    Image.new("RGB", (1200, 800), (90, 140, 200)).save(path)
    return str(path)


# --------------------------------------------------------------------------------------
# Sizes
# --------------------------------------------------------------------------------------


def test_midjourney_sizes_and_the_pixel_budget():
    assert dimensions(None) == (1024, 1024)
    assert dimensions((16, 9)) == (1456, 816)
    assert dimensions((2, 3)) == (896, 1344)
    assert dimensions((4, 3)) == (1232, 928)
    for ratio in [(21, 9), (5, 4), (1, 2), (3, 1), (7, 5)]:
        width, height = dimensions(ratio)
        assert width % 16 == 0 and height % 16 == 0
        assert width / height == pytest.approx(ratio[0] / ratio[1], rel=0.03)
        assert 0.9e6 < width * height < 1.35e6
    assert dimensions((1, 1), scale=0.5) == (512, 512)
    # Larger than the size limit: scaled to fit, keeping the ratio.
    assert dimensions((16, 9), scale=2.0) == (2560, 1440)
    assert dimensions((1, 1), multiple=32) == (1024, 1024)


def test_extreme_ratios_fit_within_limits_or_are_refused():
    width, height = dimensions((8, 1))
    assert max(width, height) <= 2560 and min(width, height) >= 256
    assert width / height == pytest.approx(8, rel=0.06)
    from dreambot.core.params import PromptError

    with pytest.raises(PromptError, match="too extreme"):
        dimensions((14, 1))


def test_working_size_and_aspect_helpers():
    assert working_size(1024, 1024) == (1024, 1024)
    assert working_size(1456, 816) == (1456, 816)
    big = working_size(4032, 3024)  # a phone photo
    assert big[0] * big[1] < 1.4e6 and big[0] / big[1] == pytest.approx(4 / 3, rel=0.03)
    small = working_size(320, 240)
    assert small[0] * small[1] > 0.9e6
    assert fit_within(4000, 3000, max_pixels=1_000_000)[0] <= 1168
    assert (
        aspect_of(1456, 816) == (16, 9) and aspect_of(1024, 1024) == (1, 1) and aspect_of(896, 1344) == (2, 3)
    )
    assert aspect_of(1200, 800) == (3, 2)


# --------------------------------------------------------------------------------------
# Choosing a model
# --------------------------------------------------------------------------------------


def test_nothing_installed_says_how_to_start(tmp_path: Path):
    with pytest.raises(PlanError, match="models pull z-image-turbo"):
        imagine(planner(tmp_path), "a fox")


def test_default_model_follows_the_preference_order(tmp_path: Path):
    assert imagine(planner(tmp_path, "schnell", "z-image-turbo"), "a fox").model == "z-image-turbo"
    assert imagine(planner(tmp_path, "schnell", "dev"), "a fox").model == "schnell"
    # A model outside the preference list still works when it is all there is.
    assert imagine(planner(tmp_path, "ming-image-design"), "a fox").model == "ming-image-design"
    custom = planner(tmp_path, "schnell", "z-image-turbo", defaults={"model": "schnell"})
    settings = UserSettings().resolved(custom.config.defaults)
    assert custom.imagine(parse_prompt("a fox"), settings).model == "schnell"


def test_explicit_model_must_be_installed_and_capable(tmp_path: Path):
    p = planner(tmp_path, "z-image-turbo")
    with pytest.raises(PlanError, match="isn't installed"):
        imagine(p, "a fox --model qwen-image")
    with pytest.raises(PlanError, match="can't do this"):
        imagine(planner(tmp_path, "z-image-turbo", "seedvr2-3b"), "a fox --model seedvr2-3b")


def test_niji_and_draft(tmp_path: Path):
    with pytest.raises(PlanError, match="anima-turbo"):
        imagine(planner(tmp_path, "z-image-turbo"), "a fox --niji")
    assert (
        imagine(planner(tmp_path, "z-image-turbo", "anima-aesthetic"), "a fox --niji 7").model
        == "anima-aesthetic"
    )
    draft = imagine(planner(tmp_path, "z-image-turbo", "qwen-image"), "a fox --draft --ar 16:9")
    assert draft.model == "z-image-turbo" and (draft.width, draft.height) == (736, 416)


# --------------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------------


def test_seeds_are_consecutive_and_reproducible(tmp_path: Path):
    p = planner(tmp_path, "z-image-turbo")
    assert imagine(p, "a fox --seed 4294967294").seeds == [4294967294, 4294967295, 0, 1]
    assert imagine(p, "a fox --sameseed 9").seeds == [9, 9, 9, 9]
    assert len(imagine(p, "a fox --n 6").seeds) == 6


def test_quality_scales_steps_or_size(tmp_path: Path):
    p = planner(tmp_path, "qwen-image", "z-image-turbo")
    assert imagine(p, "a fox --model qwen-image").renders[0].steps == 20
    assert imagine(p, "a fox --model qwen-image --q 2").renders[0].steps == 40
    assert imagine(p, "a fox --model qwen-image --q .5").renders[0].steps == 10
    assert imagine(p, "a fox --model qwen-image --steps 33 --q 4").renders[0].steps == 33
    # A distilled model keeps its step count and trades resolution instead.
    fast = imagine(p, "a fox --q .25")
    assert fast.renders[0].steps == 9 and (fast.width, fast.height) == (512, 512)
    assert any("renders smaller" in note for note in fast.notes)
    assert imagine(p, "a fox --q 2").renders[0].steps == 18


def test_negative_prompts(tmp_path: Path):
    p = planner(tmp_path, "qwen-image", "z-image-turbo", "qwen-image-2.1")
    cfg = imagine(
        p, "a street:: cars::-0.5 --no people --neg blurry, low quality --model qwen-image"
    ).renders[0]
    assert (
        cfg.negative == "people, cars, blurry, low quality"
        and cfg.prompt == "a street"
        and cfg.guidance is None
    )
    # Qwen 2.1 runs without guidance by default; a negative prompt needs it switched on.
    assert imagine(p, "a street --no cars --model qwen-image-2.1").renders[0].guidance == 3.0
    assert imagine(p, "a street --no cars --model qwen-image-2.1 --cfg 5").renders[0].guidance == 5.0
    plain = imagine(p, "a street --no cars, people")
    assert plain.renders[0].negative is None and plain.renders[0].prompt == "a street. Without cars, people."


def test_stop_stylize_raw_and_loras(tmp_path: Path):
    p = planner(tmp_path, "z-image-turbo")
    assert imagine(p, "a fox --stop 50").renders[0].stop_at == 0.5
    assert imagine(p, "a fox --stop 100").renders[0].stop_at is None
    assert imagine(p, "a fox").renders[0].prompt == "a fox"
    assert "artful composition" in imagine(p, "a fox --stylize 400").renders[0].prompt
    assert imagine(p, "a fox --stylize 400 --raw").renders[0].prompt == "a fox"

    with pytest.raises(PlanError, match="can't find a LoRA"):
        imagine(p, "a fox --lora watercolor")
    p.config.lora_dir.mkdir(parents=True)
    (p.config.lora_dir / "watercolor.safetensors").write_bytes(b"x")
    render = imagine(p, "a fox --lora watercolor:0.7 someone/repo").renders[0]
    assert [(Path(lora.path).name, lora.scale) for lora in render.loras] == [
        ("watercolor.safetensors", 0.7),
        ("repo", 1.0),
    ]


def test_chaos_weird_and_exp_direct_each_image(tmp_path: Path):
    p = planner(tmp_path, "z-image-turbo")
    calm = imagine(p, "a fox --seed 5")
    assert len(calm.renders) == 1 and calm.renders[0].seeds == [5, 6, 7, 8]

    wild = imagine(p, "a fox --seed 5 --chaos 70")
    prompts = [render.prompt for render in wild.renders]
    assert (
        len(prompts) == 4 and len(set(prompts)) == 4 and all(text.startswith("a fox, ") for text in prompts)
    )
    assert [render.slots for render in wild.renders] == [[1], [2], [3], [4]]
    assert [render.seeds for render in wild.renders] == [[5], [6], [7], [8]]
    assert imagine(p, "a fox --seed 5 --chaos 70").renders[0].prompt == prompts[0]  # deterministic
    assert imagine(p, "a fox --seed 6 --chaos 70").renders[0].prompt != prompts[0]

    weird = imagine(p, "a fox --seed 5 --weird 1500 --exp 60").renders[0].prompt
    assert "intricate detail" in weird and len(weird) > len("a fox") + 40


def test_style_direction_helpers_are_deterministic_and_scaled():
    assert styles.chaos_directions(0, 1, 4) == [[], [], [], []]
    assert [len(d) for d in styles.chaos_directions(20, 1, 4)] == [1, 1, 1, 1]
    assert [len(d) for d in styles.chaos_directions(95, 1, 4)] == [4, 4, 4, 4]
    assert len({tuple(d) for d in styles.chaos_directions(95, 1, 4)}) == 4
    assert styles.weird_directions(0, 1) == [] and len(styles.weird_directions(100, 1)) == 1
    assert len(styles.weird_directions(3000, 1)) == 4
    assert styles.style_recipe("12345") == styles.style_recipe("12345") != styles.style_recipe("12346")
    assert styles.style_recipe("12345", 0) == [] and len(styles.style_recipe("12345", 600)) > len(
        styles.style_recipe("12345", 100)
    )
    assert styles.direct("a fox", seed=1, index=0, count=4) == "a fox"
    assert styles.direct("a fox.", seed=1, index=0, count=4, exp=10) == "a fox, finely detailed"


def test_hd_needs_an_upscaler(tmp_path: Path):
    without = imagine(planner(tmp_path, "z-image-turbo"), "a fox --hd")
    assert without.finish is None and any("seedvr2-3b" in note for note in without.notes)
    with_upscaler = imagine(planner(tmp_path, "z-image-turbo", "seedvr2-3b"), "a fox --hd")
    assert with_upscaler.finish is not None and with_upscaler.finish.model == "seedvr2-3b"


# --------------------------------------------------------------------------------------
# References
# --------------------------------------------------------------------------------------


def test_references_need_an_edit_model(tmp_path: Path, picture: str):
    p = planner(tmp_path, "z-image-turbo")
    with pytest.raises(PlanError, match="flux2-klein-4b"):
        imagine(p, "her in a garden --oref", ResolvedRefs(subject=[picture]))
    # An image prompt alone degrades gracefully to image-to-image.
    plan = imagine(p, "a castle", ResolvedRefs(image_prompts=[picture]))
    assert (
        plan.op is Op.IMG2IMG
        and plan.renders[0].image == picture
        and plan.renders[0].noise == pytest.approx(0.8)
    )
    assert any("No edit model" in note for note in plan.notes)
    assert imagine(p, "a castle --iw 3", ResolvedRefs(image_prompts=[picture])).renders[
        0
    ].noise == pytest.approx(0.5)


def test_edit_model_handles_every_kind_of_reference(tmp_path: Path, picture: str):
    p = planner(tmp_path, "z-image-turbo", "flux2-klein-4b")
    plan = imagine(p, "her in a garden --oref", ResolvedRefs(subject=[picture]))
    assert plan.op is Op.EDIT and plan.model == "flux2-klein-4b" and plan.renders[0].images == [picture]
    # No --ar: the result takes the reference's 3:2 shape, at the model's pixel budget.
    assert plan.width / plan.height == pytest.approx(1.5, rel=0.03)
    assert imagine(p, "x --oref --ar 1:1", ResolvedRefs(subject=[picture])).width == 1024

    many = ResolvedRefs(subject=[picture], style=[picture] * 4)
    with pytest.raises(PlanError, match="at most 4"):
        imagine(p, "x --sv 1", many)
    # From style version 2 the four style pictures are cut into one board, which takes one slot.
    fitted = imagine(p, "x", ResolvedRefs(subject=[picture], style=[picture] * 4))
    assert fitted.renders[0].images == [picture] and len(fitted.renders[0].board.tiles) == 4

    redux = planner(tmp_path, "z-image-turbo", "dev-redux")
    blended = imagine(redux, "a castle --iw 0.5", ResolvedRefs(image_prompts=[picture, picture]))
    assert blended.op is Op.REDUX and blended.renders[0].image_weights == [0.5, 0.5]


def test_reference_instruction_wording(picture: str):
    assert reference_instruction("make it night", ResolvedRefs(edit=[picture]), Params()) == "make it night."
    subject = reference_instruction("a knight", ResolvedRefs(subject=[picture]), Params())
    assert (
        subject
        == "a knight. Keep the subject from the reference image: the same identity, face and distinctive features."
    )

    mixed = ResolvedRefs(
        subject=[picture], subject_kind="character", image_prompts=[picture], style=[picture, picture]
    )
    text = reference_instruction("a knight on a hill", mixed, Params(cw=20, sw=500, iw=2.0))
    assert "Keep the face of the character from image 1" in text
    assert "Closely follow the composition, subject and mood of image 2" in text
    assert "do not show anything depicted in images 3 and 4" in text and "exactly" in text

    assert "exactly, in every detail" in reference_instruction(
        "x", ResolvedRefs(subject=[picture]), Params(ow=800)
    )
    assert "Loosely" in reference_instruction("x", ResolvedRefs(subject=[picture]), Params(ow=10))
    assert "style" not in reference_instruction("x", ResolvedRefs(style=[picture]), Params(sw=0))
    plain_style = reference_instruction("a cottage on a cliff", ResolvedRefs(style=[picture]), Params())
    assert plain_style.startswith(
        "a cottage on a cliff. This is a completely new scene: do not show anything depicted in the reference image."
    )
    assert "only a style guide" in plain_style


# --------------------------------------------------------------------------------------
# Actions on an image
# --------------------------------------------------------------------------------------


def test_variation_falls_back_when_the_job_model_cannot_do_it(tmp_path: Path, picture: str):
    common = {"source": picture, "text": "a fox", "width": 1024, "height": 1024, "count": 4}
    same = planner(tmp_path, "z-image-turbo").variation(model="z-image-turbo", mode="subtle", **common)
    assert same.model == "z-image-turbo" and same.renders[0].noise == 0.6 and not same.notes

    other = planner(tmp_path, "z-image-turbo", "ming-image-design").variation(
        model="ming-image-design", mode="strong", **common
    )
    assert other.model == "z-image-turbo" and "can't make variations" in other.notes[0]

    with pytest.raises(PlanError, match="No installed model can make variations"):
        planner(tmp_path, "ming-image-design").variation(model="ming-image-design", mode="strong", **common)


def test_a_text_only_model_only_ever_draws_from_text(tmp_path: Path, picture: str):
    """MOSS-Image can only draw from a prompt. Asked for by name for anything else it is refused;
    as the default model, or as the model a picture was made with, another model does that work."""
    installed = ("moss-image-sft", "z-image-turbo", "flux2-klein-4b", "seedvr2-3b")
    p = planner(tmp_path, *installed)
    moss = Params(model="moss-image-sft")
    assert p.catalog.registry.find("moss").ops == (Op.TXT2IMG,)

    plan = imagine(p, "a poster that says HELLO --model moss")
    assert plan.op is Op.TXT2IMG and plan.model == "moss-image-sft"
    # A style reference it cannot be shown is put into words; the picture is still drawn from text.
    styled = imagine(p, "a fox --model moss", ResolvedRefs(style=[picture]))
    assert styled.op is Op.TXT2IMG and "applied in words" in styled.notes[0]

    for refs in (
        ResolvedRefs(image_prompts=[picture]),
        ResolvedRefs(subject=[picture]),
        ResolvedRefs(edit=[picture]),
    ):
        with pytest.raises(PlanError, match="MOSS-Image SFT can't do this"):
            imagine(p, "a fox --model moss", refs)
    common = {"source": picture, "text": "a fox", "width": 1024, "height": 1024, "count": 4}
    with pytest.raises(PlanError, match="can't do this"):
        p.variation(model="z-image-turbo", mode="strong", params=moss, **common)
    with pytest.raises(PlanError, match="can't do this"):
        p.edit(sources=[picture], instruction="make it night", count=1, params=moss)
    canvas = imaging.zoom_canvas(picture, 2.0, tmp_path / "canvas")
    with pytest.raises(PlanError, match="can't do this"):
        p.outpaint(source=picture, canvas=canvas, text="a fox", count=4, params=moss)
    with pytest.raises(PlanError, match="can't retexture"):
        p.retexture(source=picture, text="a fox", count=4, settings=SETTINGS, params=moss)

    # Chosen as the default model, it draws plain prompts and nothing else.
    default = UserSettings(enhance=False, model="moss-image-sft").resolved(Config().defaults)
    assert p.imagine(parse_prompt("a fox"), default).model == "moss-image-sft"
    blended = p.imagine(parse_prompt("a fox"), default, ResolvedRefs(image_prompts=[picture]))
    assert blended.model == "flux2-klein-4b"
    # Work on a picture MOSS made goes to a model that can do it.
    varied = p.variation(model="moss-image-sft", mode="subtle", **common)
    assert varied.model == "z-image-turbo" and "can't make variations" in varied.notes[0]
    enlarged = imaging.enlarge(picture, 2.0, tmp_path / "big.png")
    tiles = imaging.cut_tiles(enlarged, tmp_path / "tiles")
    for creative in (False, True):
        upscaled = p.upscale(
            source=picture,
            text="a fox",
            model="moss-image-sft",
            creative=creative,
            enlarged=str(enlarged),
            tiles=tiles,
        )
        assert upscaled.model != "moss-image-sft"


def test_upscale_paths(tmp_path: Path, picture: str):
    enlarged = imaging.enlarge(picture, 2.0, tmp_path / "big.png")
    tiles = imaging.cut_tiles(enlarged, tmp_path / "tiles")
    extra = {"enlarged": str(enlarged), "tiles": tiles}

    faithful = planner(tmp_path, "z-image-turbo", "seedvr2-3b").upscale(
        source=picture, text="a fox", model="z-image-turbo", creative=False, **extra
    )
    assert faithful.op is Op.UPSCALE and faithful.model == "seedvr2-3b" and faithful.tiles is None
    assert (faithful.width, faithful.height) == (2400, 1600)

    creative = planner(tmp_path, "z-image-turbo", "seedvr2-3b").upscale(
        source=picture, text="a fox", model="z-image-turbo", creative=True, **extra
    )
    assert creative.op is Op.IMG2IMG and len(creative.renders) == len(tiles) == 4
    assert all(
        render.noise == 0.45 and render.width <= 1280 and render.height <= 1280 for render in creative.renders
    )
    assert (
        len({render.seeds[0] for render in creative.renders}) == 1
    )  # one seed, so texture matches across tiles

    fallback = planner(tmp_path, "z-image-turbo").upscale(
        source=picture, text="a fox", model="z-image-turbo", creative=False, **extra
    )
    assert fallback.tiles and fallback.renders[0].noise == 0.3 and "seedvr2-3b" in fallback.notes[0]

    huge = tmp_path / "huge.png"
    Image.new("RGB", (3000, 2000)).save(huge)
    with pytest.raises(PlanError, match="already as large"):
        planner(tmp_path, "z-image-turbo").upscale(
            source=str(huge), text="", model="z-image-turbo", creative=True
        )


def test_outpaint_prefers_inpainting_then_falls_back_to_an_edit_model(tmp_path: Path, picture: str):
    canvas = imaging.zoom_canvas(picture, 2.0, tmp_path / "canvas")
    args = {"source": picture, "canvas": canvas, "text": "a fox", "count": 4}

    fill = planner(tmp_path, "z-image-turbo", "dev-fill", "flux2-klein-4b").outpaint(**args)
    assert fill.op is Op.INPAINT and fill.model == "dev-fill"
    assert fill.renders[0].image == str(canvas.image) and fill.renders[0].mask == str(canvas.mask)

    edit = planner(tmp_path, "z-image-turbo", "flux2-klein-4b").outpaint(**args)
    assert edit.op is Op.EDIT and edit.model == "flux2-klein-4b"
    assert edit.renders[0].prompt.startswith("Fill the gray area around the photo")
    assert edit.renders[0].prompt.endswith("The scene: a fox")
    assert edit.canvas is canvas and (edit.width, edit.height) == canvas.size

    with pytest.raises(PlanError, match="inpainting or edit model"):
        planner(tmp_path, "z-image-turbo").outpaint(**args)


def test_region_edit_with_and_without_a_mask(tmp_path: Path, picture: str):
    mask = str(imaging.cells_mask((1200, 800), imaging.parse_cells("B2-C3"), tmp_path / "mask.png"))
    p = planner(tmp_path, "dev-fill", "flux2-klein-4b")
    masked = p.region(source=picture, mask=mask, text="a red balloon", where=None, count=2)
    assert masked.op is Op.INPAINT and masked.keep_outside_mask == mask and len(masked.seeds) == 2

    described = p.region(source=picture, mask=None, text="an aurora", where="the sky", count=4)
    assert described.op is Op.EDIT and described.keep_outside_mask is None
    assert described.renders[0].prompt.startswith("Change only this in the sky: an aurora.")

    # A mask but no inpainting model: the edit model does it and the rest is restored afterwards.
    edit_only = planner(tmp_path, "flux2-klein-4b").region(
        source=picture, mask=mask, text="a balloon", where=None, count=1
    )
    assert edit_only.op is Op.EDIT and edit_only.keep_outside_mask == mask


def test_blend_and_edit(tmp_path: Path, picture: str):
    p = planner(tmp_path, "flux2-klein-4b")
    blend = p.blend(sources=[picture, picture, picture], aspect=(2, 3), count=4)
    assert (
        blend.op is Op.EDIT
        and (blend.width, blend.height) == (896, 1344)
        and len(blend.renders[0].images) == 3
    )
    with pytest.raises(PlanError, match="that many references"):
        p.blend(sources=[picture] * 5, aspect=None, count=4)
    assert planner(tmp_path, "dev-redux").blend(sources=[picture] * 5, aspect=None, count=4).op is Op.REDUX

    edit = p.edit(sources=[picture], instruction="  make it night ", count=1)
    assert edit.renders[0].prompt == "make it night" and edit.width / edit.height == pytest.approx(
        1.5, rel=0.03
    )
    with pytest.raises(PlanError, match="Editing needs an edit model"):
        planner(tmp_path, "z-image-turbo").edit(sources=[picture], instruction="x", count=1)


# --------------------------------------------------------------------------------------
# Canvases and tiles
# --------------------------------------------------------------------------------------


def test_canvases_place_the_source_and_mask_everything_else(tmp_path: Path, picture: str):
    zoom = imaging.zoom_canvas(picture, 2.0, tmp_path / "c")
    assert zoom.size == (1200, 800) and zoom.box == (300, 200, 900, 600)
    with Image.open(zoom.mask) as mask:
        assert mask.getpixel((10, 10)) == 255 and mask.getpixel((600, 400)) == 0
        # The mask reaches a little way into the picture on every side, to blend the seam.
        assert mask.getpixel((305, 400)) == 255 and mask.getpixel((360, 400)) == 0

    right = imaging.pan_canvas(picture, "right", tmp_path / "c")
    assert right.size == (1808, 800) and right.box == (0, 0, 1200, 800)
    with Image.open(right.mask) as mask:
        assert (
            mask.getpixel((5, 400)) == 0
            and mask.getpixel((1500, 400)) == 255
            and mask.getpixel((1190, 400)) == 255
        )
    up = imaging.pan_canvas(picture, "up", tmp_path / "c")
    assert up.size == (1200, 1200) and up.box == (0, 400, 1200, 1200)
    left = imaging.pan_canvas(picture, "left", tmp_path / "c")
    assert left.box == (608, 0, 1808, 800)

    square = imaging.square_canvas(picture, tmp_path / "c")
    assert square.size == (1200, 1200) and square.box == (0, 200, 1200, 1000)

    with pytest.raises(ValueError):
        imaging.pan_canvas(picture, "sideways", tmp_path / "c")
    with pytest.raises(ValueError):
        imaging.zoom_canvas(picture, 1.0, tmp_path / "c")


def test_pan_keeps_growing_images_within_the_size_limit(tmp_path: Path):
    wide = tmp_path / "wide.png"
    Image.new("RGB", (2400, 800), (10, 20, 30)).save(wide)
    canvas = imaging.pan_canvas(wide, "right", tmp_path / "c")
    assert max(canvas.size) <= 2560
    left, top, right, bottom = canvas.box
    assert (right - left) / (bottom - top) == pytest.approx(3.0, rel=0.02)  # the source keeps its shape


def test_tiles_cover_the_image_and_blend_back_seamlessly(tmp_path: Path):
    source = tmp_path / "gradient.png"
    Image.linear_gradient("L").resize((2688, 1792)).convert("RGB").save(source)
    tiles = imaging.cut_tiles(source, tmp_path / "tiles")
    assert len(tiles) == 6
    covered = Image.new("L", (2688, 1792), 0)
    for tile in tiles:
        covered.paste(255, tile.box)
        with Image.open(tile.path) as cut:
            assert cut.size == (tile.box[2] - tile.box[0], tile.box[3] - tile.box[1])
            assert cut.width % 16 == 0 and cut.height % 16 == 0 and max(cut.size) <= 1280
    assert covered.getextrema() == (255, 255)

    # Untouched tiles blended back must reproduce the image exactly.
    out = imaging.blend_tiles(source, [(tile.box, tile.path) for tile in tiles], tmp_path / "out.png")
    with Image.open(out) as result, Image.open(source) as original:
        assert list(result.convert("L").resize((64, 64)).get_flattened_data()) == list(
            original.convert("L").resize((64, 64)).get_flattened_data()
        )


def test_cells_and_overlay(tmp_path: Path, picture: str):
    assert imaging.parse_cells("b2") == {(1, 1)}
    assert imaging.parse_cells("B2-C3") == {(1, 1), (2, 1), (1, 2), (2, 2)}
    assert imaging.parse_cells("A1, F6") == {(0, 0), (5, 5)}
    for bad in ("", "Z9", "B7", "the sky", "B"):
        with pytest.raises(ValueError):
            imaging.parse_cells(bad)
    overlay = imaging.region_overlay(picture, tmp_path / "overlay.jpg")
    with Image.open(overlay) as image:
        assert image.size == (1200, 800)


def test_restore_helpers_keep_what_must_not_change(tmp_path: Path, picture: str):
    canvas = imaging.zoom_canvas(picture, 2.0, tmp_path / "c")
    # A realistic "model output": new surroundings, and the given picture re-rendered in place
    # but 12 levels too bright — the kind of drift that shows up as a seam.
    result = tmp_path / "result.png"
    output = Image.new("RGB", canvas.size, (200, 60, 60))
    left, top, right, bottom = canvas.box
    output.paste(Image.new("RGB", (right - left, bottom - top), (102, 152, 212)), (left, top))
    output.save(result)
    imaging.restore_source(result, canvas, picture)
    with Image.open(result) as image:
        assert image.getpixel((600, 400)) == (90, 140, 200)  # the original pixels are back in the middle
        # The surroundings stay the model's, minus the brightness drift measured in the middle.
        assert image.getpixel((20, 20)) == (188, 48, 48)
        # Across the seam the two meet without a jump.
        inside, outside = image.getpixel((left + 46, 400)), image.getpixel((left + 2, 400))
        assert outside == (90, 140, 200) and inside == (90, 140, 200)

    mask = imaging.cells_mask((1200, 800), imaging.parse_cells("C3"), tmp_path / "m.png")
    edited = tmp_path / "edited.png"
    Image.new("RGB", (1200, 800), (0, 255, 0)).save(edited)
    imaging.restore_outside_mask(edited, picture, mask)
    with Image.open(edited) as image:
        assert image.getpixel((500, 330)) == (0, 255, 0) and image.getpixel((50, 50)) == (90, 140, 200)


def test_strength_says_outright_how_much_of_an_image_prompt_to_keep(tmp_path: Path, picture: str):
    """Classic image-to-image: a model that cannot take references re-renders the picture."""
    p = planner(tmp_path, "z-image-turbo", "flux2-klein-4b")
    refs = ResolvedRefs(image_prompts=[picture])
    plain = imagine(p, "a castle --model z-image-turbo", refs).renders[0]
    assert plain.op is Op.IMG2IMG and plain.image == picture
    assert plain.strength is None and plain.noise == pytest.approx(0.8), "by default --iw 1 decides"
    assert imagine(p, "a castle --model z-image-turbo --iw 2", refs).renders[0].noise == pytest.approx(0.65)

    kept = imagine(p, "a castle --model z-image-turbo --strength 0.7", refs).renders[0]
    assert kept.strength == 0.7 and kept.noise is None, "--strength wins over the image weight"
    both = imagine(p, "a castle --model z-image-turbo --strength 0.3 --iw 3", refs).renders[0]
    assert both.strength == 0.3 and both.noise is None
    # With an edit model and no --model, the picture is a reference instead, and strength does not apply.
    assert imagine(p, "a castle --strength 0.7", refs).renders[0].op is Op.EDIT
