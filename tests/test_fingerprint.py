"""Style fingerprints: measuring, wording, style codes, blending by weight — and the planner and
service putting them to use for ``--sref``."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from PIL import Image, ImageStat

from dreambot.config import Config
from dreambot.core import fingerprint, imaging, styles
from dreambot.core.fingerprint import StyleFingerprint, Swatch
from dreambot.core.jobs import EventType, JobStatus
from dreambot.core.params import Params
from dreambot.core.planner import BOARD, Board, ResolvedRefs, StyleSource, reference_instruction
from dreambot.core.prompt import parse_prompt
from dreambot.core.service import DreambotError, describe_job
from dreambot.core.settings import UserSettings
from dreambot.engine.protocol import Op
from tests.test_planner import imagine as plan
from tests.test_planner import planner
from tests.test_service import CHANNEL, USER, bot, imagine  # noqa: F401  (bot is a fixture)

RUST = (170, 70, 30)
TEAL = (20, 110, 120)


def picture_of(tmp_path: Path, name: str, *colours: tuple[int, int, int]) -> str:
    """A picture made of equal vertical bands of the given colours."""
    image = Image.new("RGB", (240, 160), colours[0])
    band = 240 // len(colours)
    for position, colour in enumerate(colours):
        image.paste(colour, (position * band, 0, 240, 160))
    path = tmp_path / f"{name}.png"
    image.save(path)
    return str(path)


# --------------------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------------------


def test_measure_finds_the_colours_and_the_tone(tmp_path: Path):
    measured = fingerprint.measure(picture_of(tmp_path, "bands", RUST, RUST, RUST, TEAL))
    assert [swatch.rgb for swatch in measured.swatches] == [RUST, TEAL]
    assert measured.swatches[0].share == pytest.approx(0.75, abs=0.02)
    assert sum(swatch.share for swatch in measured.swatches) == pytest.approx(1.0, abs=0.01)
    assert not measured.worded and not measured.empty
    assert measured.tone is not None and measured.tone.warmth > 0

    dark = fingerprint.measure(picture_of(tmp_path, "dark", (12, 12, 14)))
    assert dark.tone.brightness < 0.1 and "dark and low-key" in fingerprint.tone_words(dark.tone)
    assert fingerprint.palette_words(dark.swatches, dark.tone) == "monochrome black and white palette"


@pytest.mark.parametrize(
    ("rgb", "name"),
    [
        ((230, 57, 70), "red"),
        ((121, 85, 61), "brown"),
        ((158, 61, 34), "rust"),
        ((189, 154, 122), "tan"),
        ((246, 238, 220), "white"),
        ((229, 208, 174), "cream"),
        ((13, 59, 102), "navy"),
        ((0, 200, 255), "cyan"),
        ((245, 197, 24), "gold"),
        ((96, 108, 56), "olive"),
        ((255, 209, 220), "pale pink"),
        ((10, 10, 10), "black"),
        ((128, 128, 128), "grey"),
    ],
)
def test_colours_get_everyday_names(rgb, name):
    assert fingerprint.colour_name(rgb) == name


def test_palette_words_lead_with_colour(tmp_path: Path):
    measured = fingerprint.measure(picture_of(tmp_path, "p", (250, 250, 250), (250, 250, 250), RUST, TEAL))
    words = fingerprint.palette_words(measured.swatches, measured.tone)
    assert "rust" in words and "deep teal" in words and words.endswith("and white"), words
    assert fingerprint.image_key(picture_of(tmp_path, "a", RUST)) == fingerprint.image_key(
        picture_of(tmp_path, "b", RUST)
    ), "the key follows the contents, not the name"


# --------------------------------------------------------------------------------------
# Words
# --------------------------------------------------------------------------------------


def test_parse_words_tolerates_how_models_answer():
    labelled = (
        "Here is the style:\n"
        "1. **Medium:** Oil painting, on canvas.\n"
        "2. rendering - loose, expressive brushwork\n"
        "3. Palette: muted teal & rust\n"
        "4. Lighting: soft overcast light. It falls from the left.\n"
        "5. texture: the technique, how edges are handled\n"
        "6. mood: 安静\n"
        "7. medium: a second answer is ignored\n"
    )
    assert fingerprint.parse_words(labelled) == {
        "medium": "oil painting",
        "rendering": "loose, expressive brushwork",
        "palette": "muted teal & rust",
        "light": "soft overcast light",
    }
    # Seen from Qwen2.5-VL: the labels dropped, or the answer put where the label goes.
    bare = ["digital painting:", "soft, blurred edges:", "warm autumnal hues:", "diffused, warm light:"]
    assert fingerprint.parse_words(bare) == {}, "without labels, fewer than six lines cannot be placed"
    bare += ["detailed, smooth textures:", "cozy, nostalgic atmosphere:"]
    assert fingerprint.parse_words(bare) == {
        "medium": "digital painting",
        "rendering": "soft, blurred edges",
        "palette": "warm autumnal hues",
        "light": "diffused, warm light",
        "texture": "detailed, smooth textures",
        "mood": "cozy, nostalgic atmosphere",
    }
    mixed = [
        "studio photograph: highly detailed, sharp focus",
        "rendering: crisp lines",
        "x",
        "y",
        "z",
        "CGI",
    ]
    assert fingerprint.parse_words(mixed)["medium"] == "studio photograph"
    assert (
        fingerprint.parse_words(mixed)["rendering"] == "crisp lines"
        and fingerprint.parse_words(mixed)["mood"] == "CGI"
    )
    runs_on = "palette: vibrant yellow, burnt orange, deep brown, metallic silver, pale sky blue"
    assert fingerprint.parse_words(runs_on) == {
        "palette": "vibrant yellow, burnt orange, deep brown, metallic silver"
    }
    assert fingerprint.parse_words("medium: " + "very " * 14 + "long") == {}
    assert fingerprint.parse_words("nothing useful here") == {}


def test_words_complete_a_measured_fingerprint(tmp_path: Path):
    measured = fingerprint.measure(picture_of(tmp_path, "p", RUST, RUST, TEAL))
    worded = fingerprint.with_words(measured, {"medium": "35mm film photograph", "bogus": "x", "mood": ""})
    assert worded.worded and worded.swatches == measured.swatches and worded.mood == ""
    assert worded.name == "Rust 35mm Film Photograph"
    assert StyleFingerprint.from_dict(worded.to_dict()) == worded


# --------------------------------------------------------------------------------------
# Applying
# --------------------------------------------------------------------------------------

GOUACHE = StyleFingerprint(
    medium="gouache illustration",
    rendering="flat simplified shapes",
    palette="muted teal and rust",
    light="soft overcast light",
    texture="rough paper texture",
    mood="quiet and contemplative",
    swatches=(Swatch(TEAL, 0.6), Swatch(RUST, 0.4)),
    name="Teal Gouache",
)
FILM = StyleFingerprint(
    medium="35mm film photograph",
    rendering="shallow depth of field",
    palette="warm amber tones",
    light="golden hour light",
    texture="visible film grain",
    mood="dreamy and nostalgic",
    swatches=(Swatch((230, 150, 40), 0.7), Swatch((60, 40, 30), 0.3)),
    name="Amber Film",
)


def test_phrases_scale_with_the_weight():
    assert fingerprint.phrases(GOUACHE, 0) == [] and fingerprint.phrases(None) == []
    assert fingerprint.phrases(GOUACHE, 20) == ["with a hint of gouache illustration style"]
    normal = fingerprint.phrases(GOUACHE, 100)
    assert normal == [
        "in the style of a gouache illustration",
        "flat simplified shapes",
        "muted teal and rust palette",
        "soft overcast light",
    ]
    strong = fingerprint.phrases(GOUACHE, 250)
    assert strong[: len(normal)] == normal and strong[-2:] == [
        "rough paper texture",
        "quiet and contemplative",
    ]
    assert fingerprint.phrases(GOUACHE, 600)[0] == "strongly stylized"
    assert fingerprint.phrases(FILM)[2] == "warm amber tones", "a palette that says so is not relabelled"
    assert (
        fingerprint.phrases(StyleFingerprint(medium="oil painting"))[0] == "in the style of an oil painting"
    )
    assert fingerprint.phrases(StyleFingerprint(medium="pixel art"))[0] == "in the style of pixel art"
    assert fingerprint.phrases(StyleFingerprint(medium="3D render"))[0] == "in the style of a 3D render"


def test_a_measured_fingerprint_is_applied_through_its_colours(tmp_path: Path):
    measured = fingerprint.measure(picture_of(tmp_path, "p", RUST, RUST, TEAL))
    palette = fingerprint.palette_words(measured.swatches, measured.tone)
    assert palette == "vivid palette of rust and deep teal"
    assert fingerprint.phrases(measured) == [palette, "soft low contrast"], "colours, then what the tone adds"
    assert fingerprint.phrases(measured, 20) == [f"with a hint of a {palette}"]
    assert fingerprint.accents(GOUACHE) == ["muted teal and rust palette", "soft overcast light"]
    assert fingerprint.summary(GOUACHE).startswith("gouache illustration · flat simplified shapes · ")


# --------------------------------------------------------------------------------------
# Style codes
# --------------------------------------------------------------------------------------


def test_a_code_always_decodes_to_the_same_coherent_style():
    assert fingerprint.from_code("4821") == fingerprint.from_code(4821) != fingerprint.from_code("4822")
    rng = random.Random(7)
    for _ in range(300):
        style = fingerprint.from_code(fingerprint.random_code(rng))
        assert all(style.words().get(name) for name in fingerprint.WORD_FIELDS)
        assert len(style.swatches) == 5 and style.name and style.tone is not None
        family = next(f for f in fingerprint._FAMILIES if style.medium in [m for m, _ in f[1]])
        assert style.rendering in family[2] and style.texture in family[3], "never film grain on a linocut"
    assert fingerprint.STYLE_COUNT > 5_000_000


def test_codes_can_be_built_from_a_style_and_back():
    rng = random.Random(11)
    for _ in range(500):
        code = fingerprint.random_code(rng)
        digits = fingerprint._digits(code)
        rebuilt = fingerprint.code_for(digits, rng)
        assert 1 <= int(rebuilt) <= fingerprint.MAX_CODE
        assert fingerprint.from_code(rebuilt) == fingerprint.from_code(code)
    with pytest.raises(ValueError):
        fingerprint.code_for([99, 0, 0, 0, 0, 0, 0])


def test_search_finds_codes_for_words():
    anime = fingerprint.search_codes("anime pastel", 8)
    assert len(anime) == 8 and len(set(anime)) == 8
    for code in anime:
        style = fingerprint.from_code(code)
        assert style.medium == "anime illustration" and style.palette == "soft pastel colors"
    assert all("linocut" in fingerprint.from_code(c).medium for c in fingerprint.search_codes("Linocut", 5))
    assert all(
        "photograph" in fingerprint.from_code(c).medium or "film still" in fingerprint.from_code(c).medium
        for c in fingerprint.search_codes("photograph", 12)
    )
    assert fingerprint.search_codes("zzzz") == [] and fingerprint.search_codes("") == []
    assert fingerprint.search_codes("anime pastel", 8) == anime, "the same search gives the same codes"


def test_similar_codes_differ_in_one_part():
    base = fingerprint._digits("4821")
    neighbours = fingerprint.similar_codes("4821", 6)
    assert len(neighbours) >= 5 and "4821" not in neighbours
    for code in neighbours:
        assert sum(a != b for a, b in zip(base, fingerprint._digits(code), strict=True)) == 1


# --------------------------------------------------------------------------------------
# Blending
# --------------------------------------------------------------------------------------


def test_blend_mixes_colour_by_weight_and_takes_structure_from_the_lead():
    assert fingerprint.blend([]) is None and fingerprint.blend([(GOUACHE, 0)]) is None
    assert fingerprint.blend([(GOUACHE, 3)]) is GOUACHE
    assert fingerprint.blend([(GOUACHE, 1), (FILM, -1)]) is GOUACHE, "a negative weight leaves a style out"

    led = fingerprint.blend([(GOUACHE, 3), (FILM, 1)])
    assert led.medium == "gouache illustration" and led.light == "soft overcast light"
    assert led.texture == "rough paper texture" and led.mood == "quiet and contemplative", (
        "a quarter share adds colour, not surface or mood"
    )
    by_share = {swatch.rgb: swatch.share for swatch in led.swatches}
    assert by_share[TEAL] == pytest.approx(0.45, abs=0.01) and by_share[(230, 150, 40)] == pytest.approx(
        0.175, abs=0.01
    )
    assert "deep teal" in led.palette and led.name == "Teal Gouache × Amber Film"

    even = fingerprint.blend([(FILM, 1), (GOUACHE, 1)])
    assert even.medium == "35mm film photograph crossed with a gouache illustration"
    assert even.texture == "visible film grain and rough paper texture"
    assert even.mood == "dreamy and nostalgic, quiet and contemplative"
    assert fingerprint.blend([(GOUACHE, 1), (FILM, 1)]).medium.startswith("gouache illustration crossed with")

    reversed_lead = fingerprint.blend([(GOUACHE, 1), (FILM, 9)])
    assert (
        reversed_lead.medium == "35mm film photograph" and reversed_lead.name == "Amber Film × Teal Gouache"
    )


def test_blend_handles_fingerprints_without_words(tmp_path: Path):
    measured = fingerprint.measure(picture_of(tmp_path, "p", RUST, RUST, TEAL))
    mixed = fingerprint.blend([(measured, 1), (GOUACHE, 1)])
    assert mixed.medium == "gouache illustration", "only a worded style can name the medium"
    only_colour = fingerprint.blend(
        [(measured, 1), (fingerprint.measure(picture_of(tmp_path, "q", TEAL)), 1)]
    )
    assert not only_colour.worded and "deep teal" in fingerprint.phrases(only_colour)[0]


# --------------------------------------------------------------------------------------
# Style boards
# --------------------------------------------------------------------------------------


def test_a_style_board_shares_its_pieces_out_by_weight(tmp_path: Path):
    rust, teal = picture_of(tmp_path, "rust", RUST), picture_of(tmp_path, "teal", TEAL)
    out = imaging.style_board([(rust, 3), (teal, 1)], 7, tmp_path / "board.png", grid=4, side=400)
    with Image.open(out) as board:
        assert board.size == (400, 400)
        pieces = [board.getpixel((x * 100 + 50, y * 100 + 50)) for y in range(4) for x in range(4)]
    assert pieces.count(RUST) == 12 and pieces.count(TEAL) == 4
    assert pieces != sorted(pieces), "the pieces are shuffled, not laid out picture by picture"

    again = imaging.style_board([(rust, 3), (teal, 1)], 7, tmp_path / "again.png", grid=4, side=400)
    assert Path(again).read_bytes() == Path(out).read_bytes(), "the same pictures and seed cut the same board"
    other = imaging.style_board([(rust, 3), (teal, 1)], 8, tmp_path / "other.png", grid=4, side=400)
    assert Path(other).read_bytes() != Path(out).read_bytes()

    # A picture with next to no weight still gets a piece; one with none is left out.
    faint = imaging.style_board(
        [(rust, 100), (teal, 1), (teal, 0)], 1, tmp_path / "faint.png", grid=4, side=400
    )
    with Image.open(faint) as board:
        pieces = [board.getpixel((x * 100 + 50, y * 100 + 50)) for y in range(4) for x in range(4)]
    assert pieces.count(TEAL) == 1
    with pytest.raises(imaging.ImageError):
        imaging.style_board([(rust, 0)], 1, tmp_path / "none.png")


def test_a_style_board_scrambles_what_a_picture_shows(tmp_path: Path):
    # Left half white, right half black: on the board no such half survives.
    halves = Image.new("RGB", (800, 800), (255, 255, 255))
    halves.paste((0, 0, 0), (400, 0, 800, 800))
    path = tmp_path / "halves.png"
    halves.save(path)
    with Image.open(imaging.style_board([(path, 1)], 3, tmp_path / "board.png")) as board:
        left = ImageStat.Stat(board.crop((0, 0, board.width // 2, board.height)).convert("L")).mean[0]
        right = ImageStat.Stat(
            board.crop((board.width // 2, 0, board.width, board.height)).convert("L")
        ).mean[0]
    assert abs(left - right) < 90, "black and white are mixed across the board"


# --------------------------------------------------------------------------------------
# The planner
# --------------------------------------------------------------------------------------


def test_style_images_work_without_an_edit_model(tmp_path: Path):
    reference = picture_of(tmp_path, "ref", RUST, RUST, TEAL)
    p = planner(tmp_path, "z-image-turbo")
    with pytest.raises(Exception, match="need an edit model"):
        plan(p, "a cottage --sref --sv 1", ResolvedRefs(style=[reference]))

    worded = plan(p, "a cottage --sref", ResolvedRefs(style=[reference]))
    assert worded.op is Op.TXT2IMG and worded.model == "z-image-turbo"
    assert worded.renders[0].prompt == "a cottage, vivid palette of rust and deep teal, soft low contrast"
    assert any("applied in words" in note for note in worded.notes)
    assert (worded.width, worded.height) == (1024, 1024), "a style lends its look, not its shape"
    style = worded.detail["style"]
    assert style["sources"] == [{"kind": "image", "label": "image 1", "share": 1.0}] and not style["board"]


def test_an_edit_model_is_shown_a_board_not_the_pictures(tmp_path: Path):
    first, second = picture_of(tmp_path, "one", RUST), picture_of(tmp_path, "two", TEAL)
    p = planner(tmp_path, "z-image-turbo", "flux2-klein-4b")
    refs = ResolvedRefs(
        style=[first, second],
        sources=[
            StyleSource("image", "image 1", 1.0, pictures=[first]),
            StyleSource("image", "image 2", 3.0, pictures=[second]),
        ],
    )
    shown = plan(p, "a cottage on a cliff --sref", refs)
    render = shown.renders[0]
    assert shown.op is Op.EDIT and shown.model == "flux2-klein-4b" and len(shown.renders) == 1
    assert render.images == [], "the pictures themselves are never shown: their content would leak"
    assert render.board == Board(((first, 25.0), (second, 75.0)), shown.seeds[0]), (
        "one board, weights literal"
    )
    assert "The reference image is a board of style swatches" in render.prompt
    assert "show nothing that appears on it" in render.prompt and "deep teal" in render.prompt
    assert [part["share"] for part in shown.detail["style"]["sources"]] == [0.25, 0.75]
    assert shown.detail["style"]["board"] and (shown.width, shown.height) == (1024, 1024)
    request = render.request(request_id="r", out_dir="o", basename="b", board="/tmp/board.png")
    assert request.images == ["/tmp/board.png"]

    words_only = plan(p, "a cottage on a cliff --sref --sv 3", refs)
    assert words_only.op is Op.TXT2IMG and words_only.model == "z-image-turbo"
    assert words_only.renders[0].board is None and not words_only.detail["style"]["board"]
    assert plan(p, "a cottage --sref --sw 0", refs).op is Op.TXT2IMG, "a style at weight 0 is not used at all"
    assert "exactly" in plan(p, "a cottage --sref --sw 500", refs).renders[0].prompt
    assert "Borrow a little" in plan(p, "a cottage --sref --sw 20", refs).renders[0].prompt

    pinned = plan(p, "a cottage --sref --model z-image-turbo", refs)
    assert pinned.op is Op.TXT2IMG and any("can't be shown reference images" in note for note in pinned.notes)


def test_a_style_goes_along_with_the_model_a_person_chose(tmp_path: Path):
    reference = picture_of(tmp_path, "ref", RUST)
    p = planner(tmp_path, "z-image-turbo", "flux2-klein-4b", "qwen-image-2.1")
    refs = ResolvedRefs(style=[reference])
    chose_text_model = UserSettings(enhance=False, model="z-image-turbo").resolved(Config().defaults)
    stayed = p.imagine(parse_prompt("a cottage --sref"), chose_text_model, refs)
    assert stayed.model == "z-image-turbo" and stayed.op is Op.TXT2IMG and stayed.renders[0].board is None
    assert not stayed.notes, "nothing to explain: their model, their style, in words"

    chose_edit_model = UserSettings(enhance=False, model="qwen-image-2.1").resolved(Config().defaults)
    boarded = p.imagine(parse_prompt("a cottage --sref"), chose_edit_model, ResolvedRefs(style=[reference]))
    assert (
        boarded.model == "qwen-image-2.1" and boarded.op is Op.EDIT and boarded.renders[0].board is not None
    )


def test_a_board_takes_one_slot_next_to_other_references(tmp_path: Path):
    reference, person = picture_of(tmp_path, "ref", RUST), picture_of(tmp_path, "person", TEAL)
    p = planner(tmp_path, "z-image-turbo", "flux2-klein-4b")
    styles_ = [picture_of(tmp_path, f"s{n}", (n * 40, 90, 200 - n * 30)) for n in range(6)]
    both = plan(p, "her in a garden --oref --sref", ResolvedRefs(subject=[person], style=styles_))
    render = both.renders[0]
    assert render.images == [person] and len(render.board.tiles) == 6, "six style pictures, one slot"
    assert (
        "Keep the subject from image 1" in render.prompt
        and "Image 2 is a board of style swatches" in render.prompt
    )
    assert both.width / both.height == pytest.approx(1.5, rel=0.03), (
        "the subject sets the shape, not the style"
    )

    full = plan(p, "x --oref --sref", ResolvedRefs(subject=[person] * 4, style=[reference]))
    assert full.renders[0].board is None and any("the others fill them" in note for note in full.notes)
    assert "rust" in full.renders[0].prompt, "the style still arrives in words"


def test_codes_and_images_blend_in_one_prompt(tmp_path: Path):
    reference = picture_of(tmp_path, "ref", RUST)
    p = planner(tmp_path, "z-image-turbo")
    code = fingerprint.from_code("4821")
    refs = ResolvedRefs(
        sources=[
            StyleSource("code", "style 4821", 2.0, code="4821"),
            StyleSource("image", "image 1", 1.0, pictures=[reference]),
        ],
        style=[reference],
        style_codes=["4821"],
    )
    blended = plan(p, "a fox --sref", refs)
    assert f"in the style of a {code.medium}" in blended.renders[0].prompt
    assert "rust" in blended.renders[0].prompt, "the image's colour reaches the palette"
    assert blended.detail["style"]["name"].startswith(code.name)

    legacy = plan(p, "a fox --sref 4821 --sv 1", ResolvedRefs(style_codes=["4821"]))
    assert legacy.renders[0].prompt == "a fox, " + ", ".join(styles.style_recipe("4821", 100))
    assert "style" not in legacy.detail
    other = plan(p, "a fox --sref 4821 --sv 6", ResolvedRefs(style_codes=["4821"]))
    assert any("Midjourney style version" in note for note in other.notes)
    assert (
        other.renders[0].prompt
        == plan(p, "a fox --sref 4821", ResolvedRefs(style_codes=["4821"])).renders[0].prompt
    )


def test_board_wording_in_the_reference_instruction(tmp_path: Path):
    refs = ResolvedRefs(style=[BOARD])
    assert "style" not in reference_instruction("x", refs, Params(sw=0), board=True)
    assert "Borrow a little" in reference_instruction("x", refs, Params(), strength=30, board=True)
    assert "Take only its visual look" in reference_instruction("x", refs, Params(), board=True)
    assert "exactly" in reference_instruction("x", refs, Params(), strength=400, board=True)
    assert "exactly" not in reference_instruction("x", refs, Params(sw=400), strength=100, board=True), (
        "a Lookbook's strength comes from --stylize, not --sw"
    )


# --------------------------------------------------------------------------------------
# The service
# --------------------------------------------------------------------------------------


async def test_the_vision_model_words_a_style_once(bot, tmp_path: Path):  # noqa: F811
    service, recorder = bot
    reference = picture_of(tmp_path, "ref", RUST, RUST, TEAL)

    job = await imagine(service, recorder, "a cottage --sref --n 1 --sv 3", attachments=[reference])
    assert job.status is JobStatus.DONE and job.model == "z-image-turbo"
    assert EventType.THINKING in recorder.types(job)
    style = job.detail["style"]
    assert style["summary"].split(" · ")[0] in (
        "gouache illustration",
        "35mm film photograph",
        "linocut print",
        "pencil drawing",
    )
    assert "in the style of" in job.images[0].prompt
    assert "Style: " in describe_job(job)

    again = await imagine(service, recorder, "a harbour --sref --n 1 --sv 3", attachments=[reference])
    assert EventType.THINKING not in recorder.types(again), "the same picture is not read twice"
    assert again.detail["style"]["summary"] == style["summary"]


async def test_weighted_references_blend_by_weight(bot, tmp_path: Path):  # noqa: F811
    service, recorder = bot
    # The stand-in vision model names a medium by a picture's colour: these two get different ones.
    one, two = picture_of(tmp_path, "one", (200, 40, 40)), picture_of(tmp_path, "two", (40, 40, 201))
    job = await imagine(service, recorder, "a cottage --sref @1::2 @2::1 --n 1", attachments=[one, two])
    assert job.status is JobStatus.DONE and job.model.startswith("flux2-klein"), (
        "an edit model is shown a board"
    )
    style = job.detail["style"]
    assert [part["share"] for part in style["sources"]] == [0.667, 0.333] and style["board"]
    assert style["summary"].startswith("gouache illustration · "), "two thirds of the weight sets the medium"
    assert "image 1 67%, image 2 33%" in describe_job(job)
    assert style["swatches"][0] == "#c82828" and "#2828c9" in style["swatches"]
    assert not list((service.config.scratch_dir / "jobs").glob("*/board*.png")), "the board is working data"

    even = await imagine(service, recorder, "a cottage --sref @1 @2 --n 1", attachments=[one, two])
    assert EventType.THINKING not in recorder.types(even), "the pictures are not read again"
    assert even.detail["style"]["summary"].startswith("gouache illustration crossed with a ")


async def test_random_styles_are_pinned_per_job(bot):  # noqa: F811
    service, recorder = bot
    jobs = await service.imagine(
        user=USER, channel=CHANNEL, prompt="a fox --sref 77 random::2 --n 1 --repeat 3"
    )
    await recorder.wait(*jobs)
    drawn = []
    for job in jobs:
        assert job.status is JobStatus.DONE and "random" not in job.prompt
        code = job.prompt.split("--sref 77 ")[1].split("::")[0]
        assert code.isdigit() and f"picked style `{code}`" in " ".join(job.notes)
        assert [part.get("code") for part in job.detail["style"]["sources"]] == ["77", code]
        drawn.append(code)
    assert len(set(drawn)) == 3, "every repeat draws its own style"


async def test_saved_styles_are_used_by_name(bot, tmp_path: Path):  # noqa: F811
    service, recorder = bot
    with pytest.raises(DreambotError, match="don't know a style called `banana`"):
        await service.imagine(user=USER, channel=CHANNEL, prompt="x --sref banana")
    with pytest.raises(DreambotError, match="starts with a letter"):
        service.save_style(user=USER, name="42", print_=GOUACHE)

    reference = picture_of(tmp_path, "ref", TEAL)
    saved = service.save_style(user=USER, name="Dusty  Noir", print_=GOUACHE, pictures=[reference])
    assert saved["name"] == "Dusty Noir" and saved["slug"] == "dusty-noir" and saved["code"].startswith("s")
    kept = Path(saved["data"]["pictures"][0])
    assert kept.is_file() and kept != Path(reference), "the picture is copied, so the style outlives it"
    assert service.save_style(user=USER, name="dusty noir", print_=FILM)["code"] == saved["code"], (
        "saving under the same name replaces the style"
    )
    service.save_style(user=USER, name="Dusty Noir", print_=GOUACHE, pictures=[reference])

    for token in ("dusty-noir", "Dusty-Noir", saved["code"]):
        job = await imagine(service, recorder, f"a fox --sref {token} --n 1 --sv 3")
        assert "in the style of a gouache illustration" in job.images[0].prompt, token
    assert job.detail["style"]["sources"] == [{"kind": "saved", "label": "style Dusty Noir", "share": 1.0}]
    assert [style["name"] for style in service.store.list_styles(USER, saved=True)] == ["Dusty Noir"]
    assert service.store.get_style("discord:someone-else", "dusty-noir")["name"] == "Dusty Noir", (
        "a style saved on the web is meant to work from the chat apps too"
    )
    boarded = await imagine(service, recorder, "a fox --sref dusty-noir --n 1")
    assert boarded.model.startswith("flux2-klein") and boarded.detail["style"]["board"], (
        "its picture feeds a board"
    )
