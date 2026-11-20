"""LoRAs: the library of fine-tuned looks, and what naming one changes about a job.

Everything here runs without weights or a network: LoRA files are a few bytes with a real
safetensors header, and Hugging Face is a stand-in that remembers what it was asked for.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from dreambot import cli
from dreambot.config import Config
from dreambot.core import imaging
from dreambot.core.actions import parse_action
from dreambot.core.jobs import Job
from dreambot.core.planner import Catalog, Fitted, PlanError, Planner, ResolvedRefs
from dreambot.core.prompt import parse_prompt
from dreambot.core.service import DreambotError
from dreambot.core.settings import UserSettings
from dreambot.engine import loras
from dreambot.engine.backends.mlx_backend import _lora_problem
from dreambot.engine.loras import Card, Entry, Library, LoraError, Wanted
from dreambot.engine.registry import REGISTRY
from dreambot.frontends.web.server import WebApp
from tests.test_adapters import make_repo
from tests.test_commands import Harness, chat  # noqa: F401  (a fixture)
from tests.test_service import USER, act, bot, imagine  # noqa: F401  (a fixture)

runner = CliRunner()
LORA_KEYS = ["layers.0.attention.to_q.lora_A.weight", "layers.0.attention.to_q.lora_B.weight"]
MODEL_KEYS = ["layers.0.attention.to_q.weight", "layers.0.attention.to_q.bias"]


def safetensors(path: Path, keys: list[str] = LORA_KEYS, metadata: dict | None = None) -> Path:
    """A file with a real safetensors header naming ``keys`` (and next to no data)."""
    header: dict = {key: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]} for key in keys}
    if metadata:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + b"\0\0\0\0")
    return path


def config(tmp_path: Path, **overrides) -> Config:
    return Config.model_validate(
        {"home": str(tmp_path / "home"), "output_dir": str(tmp_path / "out"), **overrides}
    )


def add(cfg: Config, name: str = "pixel", **fields) -> Entry:
    """A LoRA in the library with notes: for Z-Image Turbo, answering to "Pixel art style."."""
    values = {"base": "z-image-turbo", "title": "Pixel Art", "trigger": "Pixel art style.", "scale": 0.9}
    entry = Entry(
        name=name, path=str(safetensors(cfg.lora_dir / f"{name}.safetensors")), **{**values, **fields}
    )
    loras.save(cfg.lora_dir, entry)
    return entry


class Shelf(Catalog):
    """A catalog in which exactly the named models are installed."""

    def __init__(self, cfg: Config, *installed: str):
        super().__init__(cfg)
        self._installed = set(installed)

    def weights(self, spec):
        return None

    def is_installed(self, spec) -> bool:
        return spec.id in self._installed


def planner(cfg: Config, *installed: str) -> Planner:
    return Planner(Shelf(cfg, *installed), cfg, rng=random.Random(1))


def person(**settings):
    return UserSettings(enhance=False, **settings).resolved(Config().defaults)


def plan(p: Planner, prompt: str, settings=None, refs=None, **kwargs):
    return p.imagine(parse_prompt(prompt), settings or person(), refs, **kwargs)


def used(render) -> list[tuple[str, float]]:
    return [(Path(lora.path).name, lora.scale) for lora in render.loras]


class Hub:
    """Hugging Face, standing in: it knows some repositories and records what is fetched."""

    def __init__(self, *cards: Card, takes: bool = True, keys: list[str] = LORA_KEYS):
        self.cards = {card.repo: card for card in cards}
        self.takes, self.keys = takes, keys
        self.fetched: list[tuple[str, str]] = []

    def card(self, repo: str) -> Card:
        if repo not in self.cards:
            raise LoraError(f"There is no model called {repo} on Hugging Face.")
        return self.cards[repo]

    def fetch_file(self, repo: str, filename: str, folder: Path) -> Path:
        self.fetched.append((repo, filename))
        return safetensors(folder / filename, self.keys)

    def takes_loras(self, base) -> bool:
        return self.takes


PIXEL = Card(
    repo="tarn59/pixel_art_style_lora_z_image_turbo",
    files=(("pixel_art_style_z_image_turbo.safetensors", 170_000_000), ("README.md", 2000)),
    base_models=("Tongyi-MAI/Z-Image-Turbo",),
    trigger="Pixel art style.",
    license="apache-2.0",
)


# --------------------------------------------------------------------------------------
# The library
# --------------------------------------------------------------------------------------


def test_a_lora_is_a_file_by_name_and_better_with_notes(tmp_path: Path):
    cfg = config(tmp_path)
    library = Library(cfg.lora_dir)
    assert library.entries() == ([], {}) and library.find("pixel") is None  # no folder yet

    # A file dropped into the folder is a LoRA already: nothing is known about it but its name.
    loose = safetensors(cfg.lora_dir / "Grain_v2.safetensors")
    assert library.find("Grain_v2") == Entry(name="Grain_v2", path=str(loose))
    assert library.find("Grain_v2.safetensors") == library.find("Grain_v2")

    saved = add(cfg, source="tarn59/pixel", license="apache-2.0")
    notes = cfg.lora_dir / "pixel.toml"
    assert "file" not in notes.read_text()  # its weights are the file of the same name beside it
    assert library.find("pixel") == saved and library.find("PIXEL") == saved
    assert [entry.name for entry in library.entries()[0]] == ["Grain_v2", "pixel"]

    # Notes can describe a file that lives somewhere else, and name its model by any of its names.
    elsewhere = safetensors(tmp_path / "Downloads" / "anime.safetensors")
    loras.save(cfg.lora_dir, Entry(name="anime", path=str(elsewhere), base="z-image-turbo"))
    (cfg.lora_dir / "anime.toml").write_text(f'file = "{elsewhere}"\nbase = "zit"\n')
    assert library.find("anime") == Entry(name="anime", path=str(elsewhere), base="z-image-turbo")

    assert loras.remove(cfg.lora_dir, "pixel") and not notes.exists()
    assert not (cfg.lora_dir / "pixel.safetensors").exists() and library.find("pixel") is None
    assert not loras.remove(cfg.lora_dir, "pixel")
    assert loras.remove(cfg.lora_dir, "anime") and elsewhere.exists()  # only what is in the library goes


def test_names_are_plain(tmp_path: Path):
    assert loras.check_name(" Pixel-Art ") == "pixel-art"
    assert loras.slug("Pixel Art Style (LoRA) v2!") == "pixel-art-style-lora-v2"
    for bad in ("", "../x", "a/b", ".hidden", "-x", "pixel art", "a..b", "x" * 49, "none", "off"):
        with pytest.raises(LoraError):
            loras.check_name(bad)
    with pytest.raises(LoraError):
        loras.remove(tmp_path, "../outside")
    library = Library(tmp_path)
    safetensors(tmp_path.parent / "outside.safetensors")
    assert library.find("../outside") is None and library.find("a/b") is None


def test_notes_that_cannot_be_used_are_explained(tmp_path: Path):
    cfg = config(tmp_path)
    folder = cfg.lora_dir
    add(cfg)
    for name, notes in {
        "typo": 'tirgger = "x"\n',
        "garbled": "base = [",
        "nobase": 'base = "sdxl"\n',
        "strong": "scale = 9\n",
    }.items():
        safetensors(folder / f"{name}.safetensors")
        (folder / f"{name}.toml").write_text(notes)
    (folder / "orphan.toml").write_text('base = "dev"\n')
    library = Library(folder)
    found, problems = library.entries()
    assert [entry.name for entry in found] == ["pixel"]
    assert "tirgger" in problems["typo"] and "TOML" in problems["garbled"] and "scale" in problems["strong"]
    assert "not a model Dreambot has" in problems["nobase"] and "its file is gone" in problems["orphan"]
    # Asked for by name, it is refused with the reason — not quietly used without its model and words.
    with pytest.raises(LoraError, match="tirgger"):
        library.find("typo")
    with pytest.raises(PlanError, match="The LoRA `typo` can't be used: its notes have mistakes"):
        plan(planner(cfg, "z-image-turbo"), "a fox --lora typo")


def test_trigger_words_are_added_once():
    assert loras.worded("Pixel art style.", "a fox") == "Pixel art style. a fox"
    assert loras.worded("anime screencap", "a fox") == "anime screencap, a fox"
    assert loras.worded("pixel art of {prompt}, 16-bit", "a fox") == "pixel art of a fox, 16-bit"
    # A varied or rerun job brings its prompt back with the words already in it.
    for trigger, prompt in [
        ("Pixel art style.", "Pixel art style. a fox"),
        ("Pixel art style.", "a fox, PIXEL ART STYLE, at night"),
        ("pixel art of {prompt}, 16-bit", "pixel art of a fox, 16-bit"),
    ]:
        assert loras.worded(trigger, prompt) == prompt
    # Nothing to say, or nothing to add it to (a blend has no prompt).
    assert loras.worded("", "a fox") == "a fox" and loras.worded("Pixel art style.", "") == ""


# --------------------------------------------------------------------------------------
# Planning with a LoRA
# --------------------------------------------------------------------------------------


def test_naming_a_lora_brings_its_model_and_its_words(tmp_path: Path):
    cfg = config(tmp_path)
    add(cfg)
    p = planner(cfg, "qwen-image", "z-image-turbo", "z-image", "dev")
    qwen = person(model="qwen-image")
    assert plan(p, "a fox", qwen).model == "qwen-image"  # having a LoRA changes nothing by itself

    made = plan(p, "a fox in the snow --lora pixel --n 2", qwen)
    (render,) = made.renders
    assert made.model == "z-image-turbo" and render.model == "z-image-turbo"  # the model it was trained on
    assert made.notes == [
        "`pixel` is a LoRA for Z-Image Turbo, so this ran on it rather than on Qwen-Image 2512."
    ]
    assert made.text == "a fox in the snow" and render.prompt == "Pixel art style. a fox in the snow"
    assert used(render) == [("pixel.safetensors", 0.9)]
    assert made.detail["loras"] == [
        {"name": "pixel", "path": render.loras[0].path, "scale": 0.9, "base": "z-image-turbo"}
    ]
    assert plan(p, "a fox --lora pixel").notes == []  # nobody's choice of model was overridden

    # A scale typed is a share of the strength it was added with; typing the words does not double them.
    half = plan(p, "a fox, pixel art style --lora pixel:0.5").renders[0]
    assert used(half) == [("pixel.safetensors", 0.45)] and half.prompt == "a fox, pixel art style"

    # A model of the person's own choosing is theirs, as long as the LoRA can go on it at all.
    assert plan(p, "a fox --lora pixel --model z-image").model == "z-image"
    with pytest.raises(
        PlanError, match=r"`pixel` is a LoRA for Z-Image Turbo; it can't be used with FLUX.1 dev"
    ):
        plan(p, "a fox --lora pixel --model dev")
    with pytest.raises(
        PlanError, match=r"Z-Image Turbo, which isn't installed. Run `dreambot models pull z-image-turbo`"
    ):
        plan(planner(cfg, "dev"), "a fox --lora pixel")


def test_the_parameter_reads_back_the_way_it_is_typed():
    # A job's prompt is shown as it was parsed: one spelling, the one people type.
    for typed in (
        "a fox --lora pixel:0.5 grain",
        "a fox --loras pixel:0.5 grain",
        "a fox —lora pixel:0.5 grain",
    ):
        parsed = parse_prompt(typed)
        assert parsed.canonical() == "a fox --lora pixel:0.5 grain"
        assert [(lora.name, lora.scale) for lora in parsed.params.loras] == [("pixel", 0.5), ("grain", 1.0)]
    # A job kept under the old spelling reads back under the one spelling too, wherever it is shown.
    kept = {
        "id": "ABC123",
        "kind": "imagine",
        "user": "u",
        "channel": "c",
        "prompt": "a fox --loras pixel --seed 7",
    }
    assert Job.from_dict({**kept, "status": "done"}).prompt == "a fox --lora pixel --seed 7"
    assert (
        Job.from_dict({**kept, "status": "done", "prompt": "the --lorastic fox"}).prompt
        == "the --lorastic fox"
    )


def test_a_lora_whose_model_cannot_do_what_was_asked_says_so(tmp_path: Path, sample_image: Path):
    cfg = config(tmp_path)
    add(cfg)
    p = planner(cfg, "z-image-turbo", "flux2-klein-4b")
    refs = ResolvedRefs(edit=[str(sample_image)])
    with pytest.raises(PlanError) as refused:
        plan(p, "make it night --edit --lora pixel", refs=refs)
    assert str(refused.value).startswith(
        "`pixel` only works on Z-Image Turbo, the model it was trained on. Z-Image Turbo can't do this (edit)."
    )
    # Asking for a model oneself is refused in the usual words.
    with pytest.raises(PlanError, match=r"^Z-Image Turbo can't do this \(edit\)"):
        plan(p, "make it night --edit --model z-image-turbo", refs=refs)


def test_several_loras_and_ones_nothing_is_known_about(tmp_path: Path):
    cfg = config(tmp_path)
    add(cfg)
    add(cfg, "anime", trigger="anime screencap", scale=1.0)
    add(cfg, "for-dev", base="dev", trigger="")
    safetensors(cfg.lora_dir / "grain.safetensors")
    p = planner(cfg, "qwen-image", "z-image-turbo", "dev")
    qwen = person(model="qwen-image")

    both = plan(p, "a fox --lora pixel anime:0.6", qwen).renders[0]
    assert used(both) == [("pixel.safetensors", 0.9), ("anime.safetensors", 0.6)]
    assert both.prompt == "Pixel art style. anime screencap, a fox" and both.model == "z-image-turbo"
    with pytest.raises(PlanError, match="`pixel` and `for-dev` were trained on different models"):
        plan(p, "a fox --lora pixel for-dev")

    # A bare file says nothing about itself: it goes on whichever model the prompt runs on.
    loose = plan(p, "a fox --lora grain:0.5", qwen)
    assert loose.model == "qwen-image" and loose.renders[0].prompt == "a fox" and loose.notes == []
    assert used(loose.renders[0]) == [("grain.safetensors", 0.5)]
    assert plan(p, "a fox --lora grain pixel", qwen).model == "z-image-turbo"

    with pytest.raises(
        PlanError, match=r"I can't find a LoRA called `watercolour`. Available: anime, for-dev, grain, pixel"
    ):
        plan(p, "a fox --lora watercolour")
    assert plan(p, "a fox --lora someone/repo-name").renders[0].loras[0].path == "someone/repo-name"


def test_a_default_lora_goes_on_every_prompt_it_fits(tmp_path: Path, sample_image: Path):
    cfg = config(tmp_path)
    add(cfg)
    add(cfg, "anime", trigger="anime screencap", scale=1.0)
    p = planner(cfg, "qwen-image", "z-image-turbo", "flux2-klein-4b")
    mine = person(model="qwen-image", loras=["pixel:0.5"])

    made = plan(p, "a fox", mine)
    assert made.model == "z-image-turbo" and used(made.renders[0]) == [("pixel.safetensors", 0.45)]
    assert made.renders[0].prompt == "Pixel art style. a fox"
    # Naming a LoRA in the prompt is instead of the usual one; `--lora none` is none for once.
    assert used(plan(p, "a fox --lora anime", mine).renders[0]) == [("anime.safetensors", 1.0)]
    bare = plan(p, "a fox --lora none", mine)
    assert bare.model == "qwen-image" and bare.renders[0].loras == [] and bare.renders[0].prompt == "a fox"

    # What it does not fit runs without it, and says so: an edit, on a model trained only to draw...
    edited = plan(p, "make it night --edit", mine, ResolvedRefs(edit=[str(sample_image)]))
    assert edited.model == "flux2-klein-4b" and edited.renders[0].loras == [] and "loras" not in edited.detail
    assert edited.renders[0].prompt.startswith("make it night")
    assert len(edited.notes) == 1 and edited.notes[0].startswith(
        "Your default LoRA was left out of this one."
    )
    # ...a model of another family asked for by name, and a LoRA that has since been removed.
    other = plan(p, "a fox --model qwen-image", mine)
    assert other.model == "qwen-image" and other.renders[0].loras == [] and "left out" in other.notes[0]
    loras.remove(cfg.lora_dir, "pixel")
    gone = plan(p, "a fox", mine)
    assert gone.model == "qwen-image" and "can't find a LoRA called `pixel`" in gone.notes[0]
    with pytest.raises(PlanError, match="Give me something to imagine"):
        plan(p, "", mine)  # a prompt that is wrong anyway is still refused for its own reason


def test_the_art_director_is_told_the_look_and_the_words_stay_in_front(tmp_path: Path):
    cfg = config(tmp_path)
    add(cfg)
    add(cfg, "anime", trigger="Anime screencap.")
    p = planner(cfg, "z-image-turbo")
    directing = UserSettings(enhance=True).resolved(Config().defaults)

    first = plan(p, "a fox --lora anime --n 2", directing)
    assert first.brief.text == "Anime screencap. a fox"
    again = plan(
        p, "a fox --lora anime --n 2", directing, directions=["wide shot, soft light", ""], seeds=first.seeds
    )
    assert [render.prompt for render in again.renders] == [
        "Anime screencap. a fox. Wide shot, soft light",
        "Anime screencap. a fox",
    ]
    # "pixel art" is a flat graphic, which is drawn as described: no art direction, as when typed.
    assert plan(p, "a fox --lora pixel", directing).brief is None


def test_what_is_made_from_its_pictures_keeps_the_look(tmp_path: Path, sample_image: Path):
    cfg = config(tmp_path)
    saved = add(cfg)
    add(cfg, "anime", trigger="anime screencap", scale=1.0)
    add(cfg, "for-dev", base="dev", trigger="")
    p = planner(cfg, "z-image-turbo", "qwen-image")
    image = str(sample_image)
    carried = [Fitted(**record) for record in plan(p, "a fox --lora pixel").detail["loras"]]
    assert carried == [Fitted("pixel", saved.path, 0.9, "z-image-turbo")]

    def vary(**kwargs):
        text = "Pixel art style. a fox"
        return p.variation(source=image, text=text, width=1024, height=1024, mode="subtle", count=2, **kwargs)

    varied = vary(model="z-image-turbo", loras=carried)
    assert used(varied.renders[0]) == [("pixel.safetensors", 0.9)]
    assert varied.renders[0].prompt == "Pixel art style. a fox"  # the words once, not twice
    assert varied.detail["loras"] == [carried[0].record()]  # and on to variations of the variation

    # A remixed prompt that names LoRAs of its own gets those, with their words...
    remixed = vary(model="z-image-turbo", loras=carried, params=parse_prompt("x --lora anime").params)
    assert used(remixed.renders[0]) == [("anime.safetensors", 1.0)]
    assert remixed.renders[0].prompt == "anime screencap, Pixel art style. a fox"
    with pytest.raises(
        PlanError, match=r"`for-dev` is a LoRA for FLUX.1 dev; it can't be used with Z-Image Turbo"
    ):
        vary(model="z-image-turbo", params=parse_prompt("x --lora for-dev").params)
    # ...and a model the LoRA cannot go on makes its variations without it, and says so.
    fallback = vary(model="qwen-image", loras=carried)
    assert fallback.renders[0].loras == [] and "loras" not in fallback.detail
    assert fallback.notes == ["`pixel` is a LoRA for Z-Image Turbo, so Qwen-Image 2512 made this without it."]

    # A re-rendered upscale draws its new detail with the same LoRA, tile by tile.
    enlarged = imaging.enlarge(image, 2.0, tmp_path / "work" / "enlarged.png")
    tiles = imaging.cut_tiles(enlarged, tmp_path / "work" / "tiles")
    upscaled = p.upscale(
        source=image,
        text="a fox",
        model="z-image-turbo",
        creative=True,
        enlarged=str(enlarged),
        tiles=tiles,
        loras=carried,
    )
    assert all(used(tile) == [("pixel.safetensors", 0.9)] for tile in upscaled.renders)
    assert upscaled.detail["loras"] == [carried[0].record()]


def test_a_lora_path_is_only_taken_from_the_mac_itself(tmp_path: Path):
    cfg = config(tmp_path)
    cfg.lora_dir.mkdir(parents=True)
    outside = safetensors(tmp_path / "outside.safetensors")
    p = planner(cfg, "z-image-turbo")
    for name in (
        str(outside),
        "../../outside.safetensors",
        "~/outside.safetensors",
        "./outside.safetensors",
        "/nonexistent/x.safetensors",
    ):
        # Refused the same way whether or not the file exists, and without naming any folder.
        with pytest.raises(PlanError, match="file paths only work from the Mac itself") as refused:
            plan(p, f"a fox --lora {name}")
        assert str(tmp_path) not in str(refused.value)
    with pytest.raises(PlanError, match="None are installed on the Mac yet") as missing:
        plan(p, "a fox --lora watercolour")
    assert str(tmp_path) not in str(missing.value)

    local = plan(p, f"a fox --lora {outside}:0.6", allow_paths=True).renders[0]
    assert [(lora.path, lora.scale) for lora in local.loras] == [(str(outside), 0.6)]
    with pytest.raises(PlanError, match=str(cfg.lora_dir)):
        plan(p, "a fox --lora watercolour", allow_paths=True)


# --------------------------------------------------------------------------------------
# Adding to the library
# --------------------------------------------------------------------------------------


def test_sources_are_told_apart(tmp_path: Path):
    lora = safetensors(tmp_path / "pixel.safetensors")
    assert loras.parse_source(str(lora)) == ("file", str(lora.resolve()), None)
    assert loras.parse_source("tarn59/pixel_lora") == ("repo", "tarn59/pixel_lora", None)
    assert loras.parse_source("tarn59/pixel_lora:a.safetensors") == (
        "repo",
        "tarn59/pixel_lora",
        "a.safetensors",
    )
    assert loras.parse_source("https://huggingface.co/tarn59/pixel_lora") == (
        "repo",
        "tarn59/pixel_lora",
        None,
    )
    assert loras.parse_source(
        "https://huggingface.co/tarn59/pixel_lora/blob/main/sub/a.safetensors?download=true"
    ) == ("repo", "tarn59/pixel_lora", "sub/a.safetensors")
    with pytest.raises(LoraError, match="Only Hugging Face links"):
        loras.parse_source("https://civitai.com/models/123")
    with pytest.raises(LoraError, match="is a folder"):
        loras.parse_source(str(tmp_path))
    with pytest.raises(LoraError, match="neither a file"):
        loras.parse_source(str(tmp_path / "missing.safetensors"))


def test_a_lora_from_hugging_face_brings_its_model_and_trigger_words(tmp_path: Path):
    folder = config(tmp_path).lora_dir
    hub, said = Hub(PIXEL), []
    added = loras.install(folder, Wanted(PIXEL.repo, name="pixel-art"), hub=hub, say=said.append)

    assert hub.fetched == [(PIXEL.repo, "pixel_art_style_z_image_turbo.safetensors")]
    assert added == Entry(
        name="pixel-art",
        path=str(folder / "pixel-art.safetensors"),
        base="z-image-turbo",
        title="Pixel Art",
        trigger="Pixel art style.",
        source=PIXEL.repo,
        license="apache-2.0",
    )
    assert any("Trained on Z-Image Turbo" in line for line in said)
    assert Library(folder).find("pixel-art") == added  # and it is in the library
    assert sorted(path.name for path in folder.iterdir()) == ["pixel-art.safetensors", "pixel-art.toml"]

    # Left to itself, it is named after the repository; what is said outright wins over the page.
    other = loras.install(
        folder,
        Wanted(
            f"https://huggingface.co/{PIXEL.repo}", base="z-image", trigger="", scale=0.7, title="Sprites"
        ),
        hub=hub,
    )
    assert other.name == "pixel-art-style-lora-z-image-turbo" and other.base == "z-image"
    assert other.trigger == "" and other.scale == 0.7 and other.title == "Sprites"

    with pytest.raises(LoraError, match="already a LoRA called `pixel-art`"):
        loras.install(folder, Wanted(PIXEL.repo, name="pixel-art"), hub=hub)
    assert (
        loras.install(folder, Wanted(PIXEL.repo, name="pixel-art", scale=0.5, replace=True), hub=hub).scale
        == 0.5
    )
    with pytest.raises(LoraError, match="needs the network"):
        loras.install(folder, Wanted(PIXEL.repo, name="offline"))


def test_what_a_repository_holds_decides_how_it_is_added(tmp_path: Path):
    folder = config(tmp_path).lora_dir
    several = Card(
        repo="someone/klein-pixel",
        files=(("pytorch_lora_weights.safetensors", 3e8), ("pytorch_lora_weights.comfyui.safetensors", 3e8)),
        base_models=("black-forest-labs/FLUX.2-klein-4B",),
    )
    whole = Card(
        repo="someone/BEYOND",
        files=(("beyond_bf16.safetensors", 12e9),),
        base_models=("Tongyi-MAI/Z-Image-Turbo",),
    )
    orphan = Card(
        repo="someone/mystery", files=(("m.safetensors", 1e8),), base_models=("someone/unknown-model",)
    )
    for_qwen = Card(
        repo="someone/qwen-anime", files=(("a.safetensors", 1e8),), base_models=("Qwen/Qwen-Image",)
    )
    for_anima = Card(
        repo="someone/anima-x", files=(("a.safetensors", 1e8),), base_models=("circlestone-labs/Anima",)
    )
    empty = Card(repo="someone/empty", files=(("README.md", 10),))
    hub = Hub(several, whole, orphan, for_qwen, for_anima, empty)

    with pytest.raises(LoraError, match=r"several files. Say which one with `--file`.*comfyui"):
        loras.install(folder, Wanted(several.repo), hub=hub)
    picked = loras.install(folder, Wanted(several.repo, file="pytorch_lora_weights.safetensors"), hub=hub)
    assert picked.base == "flux2-klein-4b" and hub.fetched == [
        (several.repo, "pytorch_lora_weights.safetensors")
    ]
    with pytest.raises(LoraError, match=r"no file called `nope.safetensors`"):
        loras.install(folder, Wanted(several.repo, name="x", file="nope.safetensors"), hub=hub)
    with pytest.raises(LoraError, match=r"no `.safetensors` file"):
        loras.install(folder, Wanted(empty.repo), hub=hub)
    # A 12 GB file is a whole model: refused before anything is downloaded.
    with pytest.raises(LoraError, match="12 GB: that is a whole model, not a LoRA"):
        loras.install(folder, Wanted(whole.repo), hub=hub)
    assert len(hub.fetched) == 1

    # Trained on a model Dreambot does not have by that name: added all the same, and it says so.
    said: list[str] = []
    unknown = loras.install(folder, Wanted(orphan.repo), hub=hub, say=said.append)
    assert unknown.base == "" and any(
        "can't tell which model" in line and "someone/unknown-model" in line for line in said
    )
    assert loras.install(folder, Wanted(orphan.repo, name="on-dev", base="dev"), hub=hub).base == "dev"
    with pytest.raises(LoraError, match="not a model Dreambot has"):
        loras.install(folder, Wanted(orphan.repo, name="y", base="sdxl"), hub=hub)
    # Named on Hugging Face by an earlier release of the same model, and by a repository of three.
    assert loras.install(folder, Wanted(for_qwen.repo), hub=hub).base == "qwen-image"
    assert loras.install(folder, Wanted(for_anima.repo), hub=hub).base == "anima-turbo"

    with pytest.raises(LoraError, match="FIBO can't take a LoRA"):
        loras.install(folder, Wanted(orphan.repo, name="z", base="fibo"), hub=Hub(orphan, takes=False))
    # What came down is a whole model after all: nothing is kept of it.
    with pytest.raises(LoraError, match="whole model in a single file"):
        loras.install(folder, Wanted(orphan.repo, name="z", base="dev"), hub=Hub(orphan, keys=MODEL_KEYS))
    assert not any(path.name.startswith(("z.", ".adding")) for path in folder.iterdir())


def test_a_file_on_this_mac(tmp_path: Path):
    folder = config(tmp_path).lora_dir
    lora = safetensors(
        tmp_path / "Downloads" / "Ghibli_Style v2.safetensors",
        metadata={"modelspec.trigger_phrase": "ghibli style"},
    )
    added = loras.install(folder, Wanted(str(lora), base="zit"), hub=Hub())
    kept = folder / "ghibli-style-v2.safetensors"
    assert (
        added.name == "ghibli-style-v2" and added.title == "Ghibli Style V2" and added.base == "z-image-turbo"
    )
    assert added.path == str(kept) and kept.read_bytes() == lora.read_bytes()  # a copy of its own
    assert added.trigger == "ghibli style" and added.source == lora.name  # no path: only the file's name
    assert loras.is_adapter(loras.read_header(kept))
    assert (
        loras.install(folder, Wanted(str(lora), name="plain"), hub=Hub()).base == ""
    )  # which model: not known

    whole = safetensors(tmp_path / "Downloads" / "anime_model.safetensors", MODEL_KEYS)
    with pytest.raises(LoraError, match="whole model in a single file"):
        loras.install(folder, Wanted(str(whole), base="dev"), hub=Hub())
    notes = tmp_path / "Downloads" / "notes.txt"
    notes.write_text("not weights")
    with pytest.raises(LoraError, match=r"not a `.safetensors` file"):
        loras.install(folder, Wanted(str(notes), base="dev"), hub=Hub())
    with pytest.raises(LoraError, match="not a safetensors file"):
        loras.read_header(notes)
    assert sorted(path.stem for path in folder.glob("*.toml")) == ["ghibli-style-v2", "plain"]


# --------------------------------------------------------------------------------------
# The service, chat, the web app and the command line
# --------------------------------------------------------------------------------------


async def test_a_job_with_a_lora_from_prompt_to_variation(bot, monkeypatch: pytest.MonkeyPatch):  # noqa: F811
    service, recorder = bot
    saved = add(service.config)  # added while the service is running
    requests = []
    generate = service.engine.generate

    async def spy(request, **kwargs):
        requests.append(request)
        return await generate(request, **kwargs)

    monkeypatch.setattr(service.engine, "generate", spy)

    job = await imagine(service, recorder, "a fox in the snow --lora pixel --n 2")
    assert job.model == "z-image-turbo" and job.text == "a fox in the snow" and len(job.images) == 2
    assert job.detail["loras"] == [
        {"name": "pixel", "path": saved.path, "scale": 0.9, "base": "z-image-turbo"}
    ]
    assert all(image.prompt == "Pixel art style. a fox in the snow" for image in job.images)
    (request,) = requests
    assert [(lora.path, lora.scale) for lora in request.loras] == [(saved.path, 0.9)]
    assert request.prompt == "Pixel art style. a fox in the snow"
    info = (await service.act(user=USER, channel="console:main", action=parse_action("info 1"))).message
    assert "LoRA: pixel (0.9)" in info

    # Variations of its pictures are made with the same LoRA; the words are there once.
    varied = await act(service, recorder, "V1", job)
    assert requests[-1].loras == request.loras and requests[-1].prompt == "Pixel art style. a fox in the snow"
    assert varied.detail["loras"] == job.detail["loras"]

    # An edit takes its instruction as written. A parameter typed into one is not handed to the
    # edit model as words to draw from: it is left out, and the job says so.
    edited = await act(service, recorder, "edit 1: make it night --lora pixel --ar 16:9", job)
    assert requests[-1].prompt == "make it night" and requests[-1].loras == []
    assert edited.prompt == "make it night" and edited.kind.value == "edit"
    (note,) = edited.notes
    assert note.startswith(
        "An edit takes its instruction as written, so `--lora pixel`, `--aspect 16:9` was left"
    )
    assert "say the look in words instead" in note
    plain = await act(service, recorder, "edit 1: make the sky pink -- and add a few more stars", job)
    assert requests[-1].prompt == "make the sky pink -- and add a few more stars" and plain.notes == []
    with pytest.raises(DreambotError, match="Say what to change in words"):
        await service.act(
            user=USER, channel="console:main", action=parse_action("edit 1: --lora pixel").on(job.id)
        )

    # As a person's usual LoRA: every plain prompt, until they say otherwise.
    settings = service.settings(USER)
    settings.loras = ["PIXEL:0.5"]
    service.save_settings(USER, settings)
    assert service.settings(USER).loras == ["pixel:0.5"]
    usual = await imagine(service, recorder, "a heron --n 1")
    assert [(lora.path, lora.scale) for lora in requests[-1].loras] == [(saved.path, 0.45)]
    assert usual.detail["loras"][0]["name"] == "pixel"
    settings.loras = ["watercolour"]
    with pytest.raises(DreambotError, match=r"can't find a LoRA called `watercolour`. Available: pixel"):
        service.save_settings(USER, settings)
    settings.loras = ["/etc/passwd"]
    with pytest.raises(DreambotError, match="can't find a LoRA"):
        service.save_settings(USER, settings)


async def test_chat_lists_loras_sets_a_usual_one_and_keeps_paths_out(chat: Harness, tmp_path: Path):  # noqa: F811
    assert "LoRAs —" not in (await chat.say("/models"))[0].text
    add(chat.service.config)
    listing = (await chat.say("/models"))[0].text
    assert (
        "LoRAs — one fine-tuned look each" in listing and "• pixel — Pixel Art, for z-image-turbo" in listing
    )
    assert "dreambot loras add" in listing

    assert "lora        none" in (await chat.say("/settings"))[0].text
    assert "lora        pixel" in (await chat.say("/settings lora pixel"))[0].text
    job = await chat.job("a fox --n 1")
    assert job.detail["loras"][0]["name"] == "pixel" and job.images[0].prompt == "Pixel art style. a fox"
    assert "loras" not in (await chat.job("a fox --lora none --n 1")).detail
    assert "can't find a LoRA called `nope`" in (await chat.say("/settings lora nope"))[0].text
    assert "lora        none" in (await chat.say("/settings lora off"))[0].text
    assert "loras" not in (await chat.job("a fox --n 1")).detail

    outside = safetensors(tmp_path / "outside.safetensors")
    refused = (await chat.say(f"a fox --lora {outside}"))[0].text
    assert "file paths only work from the Mac itself" in refused and str(tmp_path) not in refused
    local = await chat.job(f"a fox --lora {outside} --n 1", allow_paths=True)
    assert local.detail["loras"][0]["path"] == str(outside)

    # Running a job again is no way around it: a path in one's suffix is refused there too...
    await chat.say(f"/prefer suffix --lora {outside}")
    again = (await chat.say("reroll"))[0].text
    assert "file paths only work from the Mac itself" in again and str(tmp_path) not in again
    # ...while on the Mac itself, running the job again still takes the path.
    (rerun,) = (await chat.say("reroll", allow_paths=True))[0].queued
    assert rerun.detail["loras"][0]["path"] == str(outside)


async def test_the_web_app_offers_the_library(bot):  # noqa: F811
    service, recorder = bot
    add(service.config)
    add(service.config, "anime", title="Anime", trigger="anime screencap", scale=1.0)
    add(service.config, "for-dev", title="For Dev", base="dev", trigger="")
    safetensors(service.config.lora_dir / "grain.safetensors")
    web = WebApp(service, hosts=frozenset({"testserver"}))
    origin = "http://testserver"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url=origin) as client:
        await client.get("/")
        state = (await client.get("/api/state")).json()
        # What the prompt bar's LoRA menu is drawn from: each LoRA, and the family of models it is for.
        # `retexture` says whether a picture can be drawn again in its look (tests/test_retexture.py).
        assert all(lora.pop("retexture") is (lora["family"] != "") for lora in state["loras"])
        assert state["loras"] == [
            {
                "name": "anime",
                "title": "Anime",
                "base": "z-image-turbo",
                "family": "z-image",
                "trigger": "anime screencap",
            },
            {"name": "for-dev", "title": "For Dev", "base": "dev", "family": "flux1", "trigger": ""},
            {"name": "grain", "title": "grain", "base": "", "family": "", "trigger": ""},
            {
                "name": "pixel",
                "title": "Pixel Art",
                "base": "z-image-turbo",
                "family": "z-image",
                "trigger": "Pixel art style.",
            },
        ]
        assert state["settings"]["loras"] == []
        # Several at once, each at a strength of its own, as the chips under the prompt set them...
        both = await client.put(
            "/api/settings", json={"loras": ["pixel:0.5", "anime", "grain"]}, headers={"Origin": origin}
        )
        assert both.json()["settings"]["loras"] == ["pixel:0.5", "anime", "grain"]
        # ...but not two that were trained on different models: refused, and nothing changes.
        mixed = await client.put(
            "/api/settings", json={"loras": ["pixel", "for-dev"]}, headers={"Origin": origin}
        )
        assert (
            mixed.status_code == 400
            and "trained on different models (Z-Image Turbo and FLUX.1 dev)" in mixed.text
        )
        assert (await client.get("/api/state")).json()["settings"]["loras"] == ["pixel:0.5", "anime", "grain"]
        saved = await client.put("/api/settings", json={"loras": ["pixel"]}, headers={"Origin": origin})
        assert saved.status_code == 200 and saved.json()["settings"]["loras"] == ["pixel"]
        started = await client.post(
            "/api/imagine", json={"prompt": "a fox --n 1"}, headers={"Origin": origin}
        )
        (queued,) = started.json()["jobs"]
        await recorder.wait(service.store.get_job(queued["id"]))
        shown = (await client.get(f"/api/jobs/{queued['id']}")).json()["job"]
        assert shown["loras"] == [{"name": "pixel", "scale": 0.9}] and shown["model"] == "z-image-turbo"
        # An update must show at once, even in a browser that was told nothing about how long to
        # keep the old scripts: the page (never kept) loads them from an address that names the build...
        page = (await client.get("/")).text
        assert (
            page.count(f"/static/{web.build}/") == 2 and '"/static/app' not in page and len(web.build) == 12
        )
        script = await client.get(f"/static/{web.build}/create.js")
        assert script.status_code == 200 and "lora-menu" in script.text
        assert (
            await client.get(f"/static/{web.build}/vendor/preact-htm.js")
        ).status_code == 200  # as imported
        # ...and from now on browsers are told to check with the server on every load.
        assert script.headers["cache-control"] == "no-cache"
        fresh = await client.get("/static/create.js", headers={"If-None-Match": script.headers["etag"]})
        assert fresh.status_code == 304
        cleared = await client.put("/api/settings", json={"loras": []}, headers={"Origin": origin})
        assert cleared.json()["settings"]["loras"] == []
        unknown = await client.put("/api/settings", json={"loras": ["nope"]}, headers={"Origin": origin})
        assert unknown.status_code == 400 and "can't find a LoRA" in unknown.text
    web.close()


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("DREAMBOT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    (tmp_path / "hub").mkdir()
    return tmp_path / "home"


def said(result) -> str:
    return " ".join(result.output.split())


def test_the_command_line_lists_shows_and_removes(home: Path, tmp_path: Path):
    empty = runner.invoke(cli.app, ["loras"])
    assert empty.exit_code == 0 and "No LoRAs yet" in said(empty) and "dreambot loras add" in said(empty)

    cfg = Config.model_validate({"home": str(home)})
    add(cfg, source="tarn59/pixel", license="apache-2.0")
    safetensors(cfg.lora_dir / "grain.safetensors")
    safetensors(cfg.lora_dir / "typo.safetensors")
    (cfg.lora_dir / "typo.toml").write_text('tirgger = "x"\n')
    listed = said(runner.invoke(cli.app, ["loras", "list"]))
    assert "pixel" in listed and "z-image-turbo (not installed)" in listed and "Pixel art style." in listed
    assert "grain" in listed and "not known" in listed and "typo is left out" in listed

    make_repo(tmp_path / "hub", "mflux-community/z-image-turbo-mflux-q8", {"transformer/0.safetensors": b"x"})
    assert "(not installed)" not in said(runner.invoke(cli.app, ["loras"]))
    shown = said(runner.invoke(cli.app, ["loras", "info", "pixel"]))
    for expected in (
        "Pixel Art (pixel)",
        "Trained on: Z-Image Turbo (z-image-turbo)",
        "Trigger words: Pixel art style.",
    ):
        assert expected in shown
    assert "Scale: 0.9" in shown and "Source: tarn59/pixel" in shown
    assert "License: apache-2.0; Z-Image Turbo itself is apache-2.0" in shown
    assert "not known" in said(runner.invoke(cli.app, ["loras", "info", "grain"]))
    missing = runner.invoke(cli.app, ["loras", "info", "nope"])
    assert missing.exit_code == 2 and "no LoRA called nope" in said(missing)
    broken = runner.invoke(cli.app, ["loras", "info", "typo"])
    assert broken.exit_code == 2 and "tirgger" in said(broken)

    kept = runner.invoke(cli.app, ["loras", "remove", "pixel"], input="n\n")
    assert kept.exit_code == 1 and (cfg.lora_dir / "pixel.toml").exists()
    removed = runner.invoke(cli.app, ["loras", "remove", "pixel", "--yes"])
    assert removed.exit_code == 0 and "Removed pixel" in said(removed)
    assert not (cfg.lora_dir / "pixel.toml").exists() and not (cfg.lora_dir / "pixel.safetensors").exists()
    assert (cfg.lora_dir / "grain.safetensors").exists()
    assert runner.invoke(cli.app, ["loras", "remove", "pixel", "--yes"]).exit_code == 2
    assert runner.invoke(cli.app, ["loras", "remove", "../../etc", "--yes"]).exit_code == 2


def test_what_a_lora_calls_itself_is_listed_as_it_is(tmp_path: Path, monkeypatch):
    # A LoRA's trigger words and title are written by whoever published it, and a file put in
    # the folder by hand keeps the name it came with. Square brackets in them were read as the
    # console's markup and left out: `[trigger], pixel art` was listed as `, pixel art`.
    from dreambot.engine import prepare

    home = tmp_path / "[lab]" / "home"
    monkeypatch.setenv("DREAMBOT_HOME", str(home))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    make_repo(tmp_path / "hub", "mflux-community/z-image-turbo-mflux-q8", {"transformer/0.safetensors": b"x"})
    cfg = Config.model_validate({"home": str(home)})
    add(cfg, title="Pixel Art [v2]", trigger="[trigger], pixel art")
    safetensors(cfg.lora_dir / "anime[v2].safetensors")
    (cfg.lora_dir / "sketch[old].toml").write_text('title = "Sketch"\n')  # notes whose file is gone

    listed = said(runner.invoke(cli.app, ["loras", "list"]))
    assert "anime[v2]" in listed and "[trigger], pixel art" in listed
    assert "sketch[old] is left out: its file is gone (sketch[old].safetensors)." in listed
    assert "Pixel Art [v2] (pixel)" in said(runner.invoke(cli.app, ["loras", "info", "pixel"]))

    # Adding one says its trigger words, or where to write them down when its page names none.
    worded = Card(
        repo="someone/inked", files=(("inked.safetensors", 1000),), trigger="[trigger], ink drawing"
    )
    wordless = Card(repo="someone/grain", files=(("grain.safetensors", 1000),))

    def run(args, env=None, check=False):
        monkeypatch.setattr(prepare, "HuggingFace", lambda: Hub(worded, wordless))
        return type("Done", (), {"returncode": prepare.main(args[3:])})()

    monkeypatch.setattr(cli.subprocess, "run", run)
    added = runner.invoke(cli.app, ["loras", "add", "someone/inked", "--name", "inked"])
    assert added.exit_code == 0, added.output
    assert "Trigger words, put in front of the prompt: [trigger], ink drawing" in said(added)
    added = runner.invoke(cli.app, ["loras", "add", "someone/grain", "--name", "grain"])
    assert added.exit_code == 0 and "No trigger words are known for it" in said(added), added.output
    # (The console breaks a long path wherever the line ends; put it back together.)
    assert f"{cfg.lora_dir}/grain.toml as `trigger" in added.output.replace("\n", "")


def test_adding_from_the_command_line_runs_the_downloader_and_reports(home: Path, monkeypatch):
    from dreambot.engine import prepare

    ran: list[list[str]] = []

    def run(args, env=None, check=False):
        """The downloader, in this process, with Hugging Face stood in for."""
        ran.append(args)
        assert "HF_HUB_OFFLINE" not in env and args[2:4] == ["dreambot.engine.prepare", "lora"]
        monkeypatch.setattr(prepare, "HuggingFace", lambda: Hub(PIXEL))
        return type("Done", (), {"returncode": prepare.main(args[3:])})()

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(cli.subprocess, "run", run)
    added = runner.invoke(cli.app, ["loras", "add", PIXEL.repo, "--name", "pixel-art", "--scale", "0.8"])
    assert added.exit_code == 0, added.output
    assert json.loads(ran[0][5])["scale"] == 0.8 and ran[0][4] == str(home / "loras")
    for expected in (
        "Added the LoRA pixel-art, for z-image-turbo.",
        "Trigger words, put in front of the prompt: Pixel art style.",
        "It was trained on z-image-turbo, which is not installed: dreambot models pull z-image-turbo",
        'dreambot imagine "a fox in a forest --lora pixel-art"',
        "/settings lora pixel-art",
    ):
        assert expected in said(added), expected
    assert Library(home / "loras").find("pixel-art").scale == 0.8

    again = runner.invoke(cli.app, ["loras", "add", PIXEL.repo, "--name", "pixel-art"])
    assert again.exit_code == 2 and "already a LoRA called `pixel-art`" in said(again)


def test_a_lora_that_does_not_fit_is_explained_in_words():
    spec = REGISTRY.get("z-image-turbo")
    fits_nothing = ValueError(
        "No LoRA layers were applied from flux_pixel.safetensors. Key endings seen in the file: ['lora_A.weight']."
    )
    assert _lora_problem(fits_nothing, spec) == (
        "flux_pixel.safetensors does not fit Z-Image Turbo: it was trained on a different model."
    )
    partly = ValueError(
        "3 of the 40 layers named by x.safetensors could not be applied (a, b); each reason is above."
    )
    assert "x.safetensors does not fit Z-Image Turbo" in _lora_problem(partly, spec)
    assert "not on this Mac" in _lora_problem(FileNotFoundError("LoRA file not found: 'someone/repo'."), spec)
    assert "DoRA" in _lora_problem(ValueError("DoRA with LoRA matrices is not supported: k"), spec)
    # A model whose own weights are missing is still a missing model.
    assert _lora_problem(FileNotFoundError("No safetensors files found in /models/x/vae"), spec) is None
