"""Retexture: a picture drawn again in another look, with its structure kept.

Everything here runs without weights. What the look comes out as on real weights is in the
README; these are the rules around it: which model draws, what it is told, and what is kept.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from dreambot.core import render
from dreambot.core.actions import Action, ActionKind, decode_action, encode_action, parse_action
from dreambot.core.jobs import JobKind
from dreambot.core.params import Params
from dreambot.core.planner import RETEXTURE_HOLD, PlanError, Planner
from dreambot.core.prompt import parse_prompt
from dreambot.core.service import Dreambot, DreambotError
from dreambot.engine.backends.mlx_backend import _structure_until
from dreambot.engine.loras import unworded, worded
from dreambot.engine.protocol import Op
from dreambot.frontends.web.server import WebApp
from tests.test_frontend_base import MemoryFrontend, service  # noqa: F401  (a fixture)
from tests.test_loras import add, config, person, planner, safetensors, used
from tests.test_service import USER, act, bot, imagine  # noqa: F401  (a fixture)

CONTROLNET = "z-image-turbo-controlnet-union-2.1"


def retexture(p: Planner, source: Path, typed: str = "", settings=None, text: str = "a fox in the snow"):
    """Plan a retexture the way the service does: ``typed`` is what the person wrote after it."""
    parsed = parse_prompt(typed) if typed else None
    return p.retexture(
        source=str(source),
        text=(parsed.positive_text if parsed else "") or text,
        count=1,
        settings=settings or person(),
        params=parsed.params if parsed else Params(),
    )


# --------------------------------------------------------------------------------------
# How it is asked for
# --------------------------------------------------------------------------------------


def test_the_typed_form_is_strict_about_what_follows_the_word():
    def typed(text: str):
        action = parse_action(text)
        return None if action is None else (action.kind, action.index, action.prompt)

    kind = ActionKind.RETEXTURE
    assert typed("retexture") == (kind, None, None)
    assert typed("Retexture 2") == (kind, 2, None)
    assert typed("/retexture 2") == (kind, 2, None)
    # Words come after a colon, capitals kept; parameters alone may follow directly.
    assert typed("retexture 2: An Oil Painting --lora inkwash") == (kind, 2, "An Oil Painting --lora inkwash")
    assert typed("retexture: watercolour") == (kind, None, "watercolour")
    assert typed("retexture 2 --lora pixel-art:0.8 --strength 0.5") == (
        kind,
        2,
        "--lora pixel-art:0.8 --strength 0.5",
    )
    assert typed("retexture --lora pixel-art") == (kind, None, "--lora pixel-art")
    # A prompt that happens to start with the word is still a prompt.
    assert typed("retexture the walls of a cave") is None
    assert typed("retexture 2 an oil painting") is None
    assert typed("retexture 0") is None

    action = parse_action("retexture 2").on("ABC123")
    assert encode_action(action) == "db:rt:ABC123:2"
    assert decode_action("db:rt:ABC123:2") == Action(kind, "ABC123", 2)


# --------------------------------------------------------------------------------------
# What draws it, and what it is told
# --------------------------------------------------------------------------------------


def test_a_lora_is_put_on_a_picture_by_its_own_models_controlnet(tmp_path: Path, sample_image: Path):
    cfg = config(tmp_path)
    pixel = add(cfg)
    p = planner(cfg, "z-image-turbo", CONTROLNET, "flux2-klein-4b")

    plan = retexture(p, sample_image, "--lora pixel --seed 5")
    (draw,) = plan.renders
    assert plan.model == CONTROLNET == draw.model and plan.op is Op.CONTROL is draw.op
    assert draw.image == str(sample_image) and draw.seeds == [5]
    # Its words go in front, as in a prompt, and the structure is followed for the first steps only.
    assert draw.prompt == "Pixel art style. a fox in the snow" and plan.text == "a fox in the snow"
    assert draw.extra == {"control": "depth", "until": RETEXTURE_HOLD} and draw.strength is None
    assert used(draw) == [("pixel.safetensors", 0.9)]
    assert plan.detail == {
        "hold": RETEXTURE_HOLD,
        "loras": [{"name": "pixel", "path": pixel.path, "scale": 0.9, "base": "z-image-turbo"}],
    }
    assert plan.notes == [] and (plan.width, plan.height) == (draw.width, draw.height)

    # `--strength` is how much of the picture is kept: the share of the steps its structure is held for.
    firm = retexture(p, sample_image, "--lora pixel --strength 0.6")
    assert firm.renders[0].extra["until"] == 0.6 and firm.detail["hold"] == 0.6

    # Words alone will do: the look is then whatever they say.
    plain = retexture(p, sample_image, "a fox in the snow, watercolour")
    assert plain.model == CONTROLNET and plain.renders[0].loras == [] and "loras" not in plain.detail
    assert plain.renders[0].prompt == "a fox in the snow, watercolour"
    with pytest.raises(PlanError, match="needs words for what the picture shows"):
        retexture(p, sample_image, text="  ")


def test_the_model_is_one_of_the_loras_family(tmp_path: Path, sample_image: Path):
    cfg = config(tmp_path)
    add(cfg)
    add(cfg, "blocks", base="flux2-klein-4b", trigger="")
    add(cfg, "grain", base="dev", trigger="film grain", scale=1.0)
    safetensors(cfg.lora_dir / "loose.safetensors")  # nothing is known about this one

    everything = planner(cfg, "z-image-turbo", CONTROLNET, "dev", "dev-depth", "flux2-klein-4b")
    assert retexture(everything, sample_image, "--lora pixel").model == CONTROLNET
    assert everything.retexture_loras() == ["grain", "pixel"]
    # Another family's own model for it. It is not steered by a share of the steps, so it is left
    # at its own strength unless one is asked for.
    grainy = retexture(everything, sample_image, "--lora grain")
    assert (
        grainy.model == "dev-depth" and grainy.renders[0].extra == {} and grainy.renders[0].strength is None
    )
    assert grainy.renders[0].prompt == "film grain, a fox in the snow"
    assert retexture(everything, sample_image, "--lora grain --strength 0.7").renders[0].strength == 0.7
    # One nothing is known about goes on the usual model, alone or along with another.
    assert used(retexture(everything, sample_image, "--lora loose").renders[0]) == [
        ("loose.safetensors", 1.0)
    ]
    assert retexture(everything, sample_image, "--lora loose pixel").model == CONTROLNET
    # `--model` may name such a model, or the model it is built on.
    assert retexture(everything, sample_image, "--model z-image-turbo").model == CONTROLNET
    assert retexture(everything, sample_image, f"--model {CONTROLNET} --lora pixel").model == CONTROLNET
    assert retexture(everything, sample_image, "--model dev").model == "dev-depth"

    with pytest.raises(PlanError, match="were trained on different models"):
        retexture(everything, sample_image, "--lora pixel grain")
    with pytest.raises(
        PlanError, match=r"`pixel` is a LoRA for Z-Image Turbo; it can't be used with FLUX.1 Depth"
    ):
        retexture(everything, sample_image, "--model dev --lora pixel")
    # A family in which nothing follows a picture's structure.
    with pytest.raises(PlanError) as refused:
        retexture(everything, sample_image, "--lora blocks")
    assert str(refused.value) == (
        "`blocks` is a LoRA for FLUX.2 klein 4B, and no model of its family follows a picture's structure, "
        "so it can't be used to retexture one."
    )
    with pytest.raises(PlanError, match=r"FLUX.2 klein 4B can't retexture: .* Models that can: .*dev-depth"):
        retexture(everything, sample_image, "--model flux2-klein-4b")

    # Nothing installed that can: what to install is said, for the LoRA's model where one is named.
    bare = planner(cfg, "z-image-turbo", "dev")
    assert bare.retexture_loras() == []
    with pytest.raises(PlanError) as refused:
        retexture(bare, sample_image)
    assert str(refused.value) == (
        "Retexturing needs a model that follows a picture's structure. Install one with "
        f"`dreambot models pull {CONTROLNET}`."
    )
    with pytest.raises(PlanError) as refused:
        retexture(bare, sample_image, "--lora pixel")
    assert str(refused.value) == (
        "`pixel` is a LoRA for Z-Image Turbo. To retexture with it, install "
        f"Z-Image Turbo ControlNet Union 2.1: `dreambot models pull {CONTROLNET}`."
    )
    with pytest.raises(PlanError, match=f"To retexture on Z-Image Turbo, install .*pull {CONTROLNET}"):
        retexture(bare, sample_image, "--model z-image-turbo")


def test_a_usual_lora_is_the_look_unless_one_is_named(tmp_path: Path, sample_image: Path):
    cfg = config(tmp_path)
    add(cfg)
    add(cfg, "anime", trigger="anime screencap", scale=1.0)
    add(cfg, "blocks", base="flux2-klein-4b", trigger="")
    p = planner(cfg, "z-image-turbo", CONTROLNET, "flux2-klein-4b")

    usual = retexture(p, sample_image, settings=person(loras=["pixel:0.5"]))
    assert used(usual.renders[0]) == [("pixel.safetensors", 0.45)] and usual.notes == []
    named = retexture(p, sample_image, "--lora anime", settings=person(loras=["pixel"]))
    assert used(named.renders[0]) == [("anime.safetensors", 1.0)]
    none = retexture(p, sample_image, "--lora none", settings=person(loras=["pixel"]))
    assert none.renders[0].loras == [] and none.renders[0].prompt == "a fox in the snow"
    # One that cannot go on a retexture is left out, and the job says so.
    without = retexture(p, sample_image, settings=person(loras=["blocks"]))
    assert without.renders[0].loras == [] and without.model == CONTROLNET
    (note,) = without.notes
    assert note.startswith(
        "Your default LoRA was left out of this one. (`blocks` is a LoRA for FLUX.2 klein 4B"
    )


def test_trigger_words_come_out_the_way_they_went_in():
    for trigger in (
        "Pixel art style.",
        "Clay animation, Clay.",
        "anime screencap",
        "a painting of {prompt} in oil",
    ):
        assert unworded(trigger, worded(trigger, "a fox in the snow")) == "a fox in the snow"
    stacked = worded("Pixel art style.", worded("Clay animation, Clay.", "a fox"))
    assert unworded("Clay animation, Clay.", unworded("Pixel art style.", stacked)) == "a fox"
    # Words that are not in front are part of the prompt, and a prompt is never emptied.
    assert unworded("Pixel art style.", "a fox, pixel art style") == "a fox, pixel art style"
    assert unworded("Pixel art style.", "Pixel art style.") == "Pixel art style."
    assert unworded("", "a fox") == "a fox"


# --------------------------------------------------------------------------------------
# The engine: structure for the first steps only
# --------------------------------------------------------------------------------------


class Net:
    """A ControlNet standing in: it remembers the steps it was asked about."""

    control_in_dim = 33

    def __init__(self):
        self.steps: list[int] = []

    def __call__(self, **kwargs):
        self.steps.append(kwargs["t"])
        return {0: "residual"}


def draw(model, steps: int) -> list[dict]:
    """What a picture's steps get from the model's ControlNet, as mflux asks for it."""
    return [model["controlnet"](t=step, x=None) for step in range(steps)]


def test_the_controlnet_steers_the_first_steps_and_is_put_back():
    net = Net()
    model = {"controlnet": net}

    with _structure_until(model, 0.3, 8):
        assert model["controlnet"].control_in_dim == 33  # everything else about it is as it was
        answers = draw(model, 8)
    assert net.steps == [0, 1] and answers[:2] == [{0: "residual"}] * 2 and answers[2:] == [{}] * 6
    assert model["controlnet"] is net

    # Never fewer than one step, and all of them when nothing less is asked for.
    for share, steered in ((0.01, 1), (0.5, 4), (1.0, 8), (None, 8), ("nonsense", 8)):
        net.steps.clear()
        with _structure_until(model, share, 8):
            draw(model, 8)
        assert len(net.steps) == steered, share
    # The model is kept for the next request: it gets its ControlNet back when a picture fails.
    with pytest.raises(RuntimeError), _structure_until(model, 0.3, 8):
        raise RuntimeError("out of memory")
    assert model["controlnet"] is net
    # A model that keeps no ControlNet of its own is left alone.
    other = {"transformer": object()}
    with _structure_until(other, 0.3, 8):
        assert list(other) == ["transformer"]
    with _structure_until(object(), 0.3, 8):
        pass


# --------------------------------------------------------------------------------------
# From a picture to its retexture, and on from there
# --------------------------------------------------------------------------------------


async def test_a_picture_is_retextured_and_what_follows_keeps_the_look(bot, monkeypatch: pytest.MonkeyPatch):  # noqa: F811
    bot_service, recorder = bot
    pixel = add(bot_service.config)
    add(bot_service.config, "clay", title="Clay", trigger="Clay animation, Clay.", scale=1.0)
    requests = []
    generate = bot_service.engine.generate

    async def spy(request, **kwargs):
        requests.append(request)
        return await generate(request, **kwargs)

    monkeypatch.setattr(bot_service.engine, "generate", spy)
    job = await imagine(bot_service, recorder, "a fox in the snow --lora clay --n 1 --seed 3")
    assert requests[-1].prompt == "Clay animation, Clay. a fox in the snow"

    # Another look: the words of the one it had are taken out, the new one's put in.
    again = await act(bot_service, recorder, "retexture 1 --lora pixel:0.5 --n 2", job)
    request = requests[-1]
    assert again.kind is JobKind.RETEXTURE and again.model == CONTROLNET and len(again.images) == 2
    assert request.op is Op.CONTROL and request.image == job.images[0].path
    assert request.prompt == "Pixel art style. a fox in the snow"
    assert request.extra == {"control": "depth", "until": RETEXTURE_HOLD}
    assert [(lora.path, lora.scale) for lora in request.loras] == [(pixel.path, 0.45)]
    assert (
        again.text == "a fox in the snow" and again.prompt == "a fox in the snow --lora pixel:0.5 --count 2"
    )
    assert (again.parent, again.parent_index) == (job.id, 1) and again.inputs == {
        "source": job.images[0].path
    }
    assert again.detail["loras"][0]["name"] == "pixel" and again.notes == []
    assert all(image.prompt == "Pixel art style. a fox in the snow" for image in again.images)
    info = (
        await bot_service.act(user=USER, channel="console:main", action=parse_action("info 1").on(again.id))
    ).message
    assert "LoRA: pixel (0.45)" in info
    assert render.KIND_LABELS[again.kind] == "Retextured"

    # Run again, it is the same retexture of the same picture.
    rerun = await act(bot_service, recorder, "reroll", again)
    assert rerun.kind is JobKind.RETEXTURE and requests[-1].image == job.images[0].path
    assert requests[-1].loras == request.loras and rerun.prompt == again.prompt

    # Variations of it are made by the model itself, with the LoRA, and nothing needs saying.
    varied = await act(bot_service, recorder, "V1", again)
    assert requests[-1].model == "z-image-turbo" and requests[-1].op is Op.IMG2IMG
    assert requests[-1].loras == request.loras and requests[-1].prompt == "Pixel art style. a fox in the snow"
    assert varied.notes == []

    # With words of its own, those are what it is drawn from; without a LoRA, in no LoRA's look.
    worded_one = await act(
        bot_service, recorder, "retexture 1: A fox, as a Woodcut --no snow --lora none", job
    )
    assert requests[-1].prompt == "A fox, as a Woodcut" and requests[-1].loras == []
    assert worded_one.prompt == "A fox, as a Woodcut --no snow --lora none"
    # Nothing typed, no usual LoRA: the picture's own words, in the model's own look.
    bare = await act(bot_service, recorder, "retexture 1", job)
    assert requests[-1].prompt == "a fox in the snow" and bare.prompt == "a fox in the snow"

    # A person's usual LoRA is the look when none is named.
    settings = bot_service.settings(USER)
    settings.loras = ["pixel"]
    bot_service.save_settings(USER, settings)
    await act(bot_service, recorder, "retexture 1", job)
    assert requests[-1].prompt == "Pixel art style. a fox in the snow"

    with pytest.raises(DreambotError, match=r"can't find a LoRA called `nope`"):
        await bot_service.act(
            user=USER, channel="console:main", action=parse_action("retexture 1 --lora nope").on(job.id)
        )
    # A file is only taken from the Mac itself, as in a prompt.
    with pytest.raises(DreambotError, match="file paths only work from the Mac itself"):
        await bot_service.act(
            user=USER,
            channel="console:main",
            action=parse_action(f"retexture 1 --lora {pixel.path}").on(job.id),
        )


async def test_the_button_asks_for_the_look(service: Dreambot):  # noqa: F811
    add(service.config)
    front = MemoryFrontend(service)
    await front.start()
    await front.say("a fox --n 1")
    single = front.visible[-1]
    assert "Retexture" in single.labels()

    await front.press(single, "Retexture")
    question = front.visible[-1]
    assert question.ask_reply and "`--lora pixel`" in question.text and "LoRAs: pixel" in question.text
    await front.say("--lora pixel")
    made = service.store.recent_jobs(limit=1)[0]
    assert made.kind is JobKind.RETEXTURE and made.detail["loras"][0]["name"] == "pixel"
    assert "retexture --lora <name>" in render.help_text(direct=True)


async def test_the_web_app_says_which_loras_can_retexture(bot):  # noqa: F811
    bot_service, recorder = bot
    add(bot_service.config)
    add(bot_service.config, "blocks", base="flux2-klein-4b", trigger="")
    job = await imagine(bot_service, recorder, "a fox in the snow --n 1")
    web = WebApp(bot_service, hosts=frozenset({"testserver"}))
    origin = "http://testserver"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url=origin) as client:
        await client.get("/")
        state = (await client.get("/api/state")).json()
        # What the lightbox's Retexture box offers: the LoRAs of a family with such a model.
        assert {lora["name"]: lora["retexture"] for lora in state["loras"]} == {
            "blocks": False,
            "pixel": True,
        }
        answer = await client.post(
            f"/api/jobs/{job.id}/act",
            json={"kind": "retexture", "index": 1, "prompt": "--lora pixel"},
            headers={"Origin": origin},
        )
        assert answer.status_code == 200, answer.text
        (made,) = answer.json()["jobs"]
        assert made["kind"] == "retexture" and made["model"] == CONTROLNET
        assert made["loras"] == [{"name": "pixel", "scale": 0.9}]
        await recorder.wait(*bot_service.store.recent_jobs(limit=1))
