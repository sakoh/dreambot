"""The web app's server: who it lets in, and every route, against the real service on the fake
backend. The pages themselves are checked by hand in a browser."""

from __future__ import annotations

import asyncio
import io
import json
import zipfile
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from PIL import Image

from dreambot.core import fingerprint, lookbook
from dreambot.core.jobs import JobStatus
from dreambot.frontends.web.server import COOKIE, WebApp, WebFrontend, session_token
from tests.test_lookbook import settle
from tests.test_service import bot  # noqa: F401  (a fixture)

HOST = "testserver"
ORIGIN = f"http://{HOST}"


def png(colour: tuple[int, int, int], size: tuple[int, int] = (320, 240)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, "PNG")
    return buffer.getvalue()


@pytest.fixture
async def site(bot) -> AsyncIterator[tuple]:  # noqa: F811
    service, recorder = bot
    web = WebApp(service, hosts=frozenset({HOST}))
    transport = httpx.ASGITransport(app=web.app)
    async with httpx.AsyncClient(transport=transport, base_url=ORIGIN) as client:
        page = await client.get("/")
        assert page.status_code == 200 and COOKIE in client.cookies
        yield service, recorder, web, client
    web.close()


async def post(client: httpx.AsyncClient, path: str, body: dict | None = None, **kwargs) -> httpx.Response:
    return await client.post(path, json=body or {}, headers={"Origin": ORIGIN}, **kwargs)


async def finished(service, recorder, client, response: httpx.Response) -> dict:
    """Wait for the single job a request started, and return it as the web app shows it."""
    assert response.status_code == 200, response.text
    (queued,) = response.json()["jobs"]
    job = service.store.get_job(queued["id"])
    await recorder.wait(job)
    return (await client.get(f"/api/jobs/{queued['id']}")).json()["job"]


async def upload(client: httpx.AsyncClient, colour: tuple[int, int, int], name: str = "picture.png") -> dict:
    response = await client.post(
        "/api/uploads", files={"file": (name, png(colour), "image/png")}, headers={"Origin": ORIGIN}
    )
    assert response.status_code == 200, response.text
    return response.json()["uploads"][0]


# --------------------------------------------------------------------------------------
# Who gets in
# --------------------------------------------------------------------------------------


async def test_only_the_apps_own_page_gets_in(bot):  # noqa: F811
    service, _ = bot
    web = WebApp(service, hosts=frozenset({HOST}))
    transport = httpx.ASGITransport(app=web.app)
    async with httpx.AsyncClient(transport=transport, base_url=ORIGIN) as stranger:
        refused = await stranger.get("/api/state")
        assert refused.status_code == 401 and "own page" in refused.json()["error"], "no cookie, no data"
        assert (await stranger.get("/media/job/ABC123/1")).status_code == 401
        assert (await stranger.get("/static/app.css")).status_code in (200, 404), (
            "the page's own files are public"
        )

        page = await stranger.get("/")
        cookie = page.headers["set-cookie"].lower()
        assert "httponly" in cookie and "samesite=strict" in cookie
        assert "script-src 'self'" in page.headers["content-security-policy"]
        assert (await stranger.get("/api/state")).status_code == 200, "the page's cookie lets it in"

        # Another site's name pointed at this Mac (DNS rebinding) is turned away whatever it sends.
        rebound = await stranger.get("/api/state", headers={"Host": "evil.example:7373"})
        assert rebound.status_code == 403 and "not the Dreambot web app" in rebound.json()["error"]
        assert (await stranger.get("/", headers={"Host": "evil.example"})).status_code == 403

        # A request that changes something must come from the app's own origin.
        body = {"prompt": "a fox"}
        cross = await stranger.post("/api/imagine", json=body, headers={"Origin": "https://evil.example"})
        assert cross.status_code == 403 and "did not come from" in cross.json()["error"]
        fetched = await stranger.post("/api/imagine", json=body, headers={"Sec-Fetch-Site": "cross-site"})
        assert fetched.status_code == 403
        assert service.queue_length == 0, "nothing was started"
    async with httpx.AsyncClient(transport=transport, base_url=ORIGIN, cookies={COOKIE: "guess"}) as guesser:
        assert (await guesser.get("/api/state")).status_code == 401
    web.close()


def test_the_session_token_is_made_once_and_kept_private(tmp_path: Path):
    token = session_token(tmp_path / "home")
    assert len(token) >= 32 and session_token(tmp_path / "home") == token
    assert (tmp_path / "home" / "web-token").stat().st_mode & 0o777 == 0o600


async def test_the_real_server_streams_events_and_refuses_a_taken_port(bot):  # noqa: F811
    service, _ = bot
    front = WebFrontend(service, host="127.0.0.1", port=0)
    await front.start()
    try:
        assert front.port != 0 and front.url == f"http://127.0.0.1:{front.port}/"
        assert front.port != service.config.web.port, "0 asks for any free port, not for the configured one"
        async with httpx.AsyncClient(base_url=front.url.rstrip("/")) as client:
            assert (await client.get("/api/state")).status_code == 401
            assert (await client.get("/")).status_code == 200
            assert (await client.get("/api/state", headers={"Host": "evil.example"})).status_code == 403
            async with client.stream("GET", "/api/events") as stream:
                assert stream.headers["content-type"].startswith("text/event-stream")
                lines = stream.aiter_lines()

                async def next_event() -> dict:
                    async for line in lines:
                        if line.startswith("data: "):
                            return json.loads(line[6:])
                    raise AssertionError("the stream ended")

                assert (await asyncio.wait_for(next_event(), 5))["type"] == "hello"
                started = await client.post(
                    "/api/imagine", json={"prompt": "a fox --n 1"}, headers={"Origin": front.url.rstrip("/")}
                )
                assert started.status_code == 200
                seen = []
                while "finished" not in seen:
                    event = await asyncio.wait_for(next_event(), 20)
                    seen.append(event["type"])
                assert seen[0] == "queued" and "started" in seen
                assert event["job"]["images"][0]["url"].startswith("/media/job/")
        second = WebFrontend(service, host="127.0.0.1", port=front.port)
        with pytest.raises(RuntimeError, match=f"port {front.port} is in use"):
            await second.start()
    finally:
        await front.stop()


# --------------------------------------------------------------------------------------
# Creating
# --------------------------------------------------------------------------------------


async def test_state_and_settings(site):
    _, _, _, client = site
    state = (await client.get("/api/state")).json()
    assert state["backend"] == "fake" and state["model"] == "z-image-turbo" and state["vision"]
    assert any(model["id"] == "flux2-klein-4b" and model["edits"] for model in state["models"])
    assert state["settings"]["count"] == 4 and not state["settings"]["personalize"]

    saved = await client.put(
        "/api/settings",
        json={"count": 2, "stylize": 250, "variation": "subtle", "personalize": True, "model": "qwen-image"},
        headers={"Origin": ORIGIN},
    )
    settings = saved.json()["settings"]
    assert (settings["count"], settings["stylize"], settings["variation"]) == (2, 250, "subtle")
    assert (
        settings["personalize"]
        and settings["model"] == "qwen-image"
        and saved.json()["model"] == "qwen-image"
    )
    bad = await client.put("/api/settings", json={"variation": "wild"}, headers={"Origin": ORIGIN})
    assert bad.status_code == 400 and "strong or subtle" in bad.json()["error"]
    unknown = await client.put("/api/settings", json={"model": "no-such-model"}, headers={"Origin": ORIGIN})
    assert unknown.status_code == 400


async def test_imagine_and_the_feed(site):
    service, recorder, _, client = site
    empty = await post(client, "/api/imagine", {"prompt": "  "})
    assert empty.status_code == 400 and empty.json()["error"] == "Type a prompt first."
    bad = await post(client, "/api/imagine", {"prompt": "a fox --bogus 3"})
    assert bad.status_code == 400 and "don't know the parameter" in bad.json()["error"]

    job = await finished(
        service, recorder, client, await post(client, "/api/imagine", {"prompt": "a fox --n 2 --seed 4"})
    )
    assert job["status"] == "done" and job["user"] == "web:local" and len(job["images"]) == 2
    assert job["images"][0] == {
        "index": 1,
        "seed": 4,
        "width": 1024,
        "height": 1024,
        "favorite": False,
        "trashed": False,
        "prompt": None,
        "url": f"/media/job/{job['id']}/1",
    }
    feed = (await client.get("/api/jobs")).json()
    assert [entry["id"] for entry in feed["jobs"]] == [job["id"]] and feed["next"] is None
    assert (await client.get("/api/jobs", params={"q": "fox"})).json()["jobs"]
    assert not (await client.get("/api/jobs", params={"q": "fox heron"})).json()["jobs"], (
        "every word must match"
    )
    assert not (await client.get("/api/jobs", params={"q": "100%"})).json()["jobs"]
    assert (await client.get("/api/jobs/NOPE99")).status_code == 404

    full = await client.get(job["images"][0]["url"])
    assert full.status_code == 200 and full.headers["content-type"] == "image/png"
    thumb = await client.get(job["images"][0]["url"], params={"w": 256})
    assert thumb.headers["content-type"] == "image/jpeg" and len(thumb.content) < len(full.content)
    with Image.open(io.BytesIO(thumb.content)) as small:
        assert small.width == 256
    odd = await client.get(job["images"][0]["url"], params={"w": 300})
    assert odd.headers["content-type"] == "image/png", "only the known sizes are made"
    saved = await client.get(job["images"][0]["url"], params={"download": 1})
    assert "attachment" in saved.headers["content-disposition"]
    assert (await client.get(f"/media/job/{job['id']}/9")).status_code == 404
    assert (await client.get(f"/media/grid/{job['id']}")).status_code == 200


async def test_actions_are_the_chat_buttons(site):
    service, recorder, _, client = site
    job = await finished(
        service, recorder, client, await post(client, "/api/imagine", {"prompt": "a fox --n 2"})
    )
    act = f"/api/jobs/{job['id']}/act"

    liked = await post(client, act, {"kind": "favorite", "index": 2})
    assert liked.json()["message"] == "Added to favorites." and liked.json()["job"]["images"][1]["favorite"]
    need_index = await post(client, act, {"kind": "favorite"})
    assert need_index.status_code == 400 and "Which image?" in need_index.json()["error"]
    assert (await post(client, act, {"kind": "teleport"})).status_code == 400

    varied = await finished(
        service, recorder, client, await post(client, act, {"kind": "vary_subtle", "index": 1})
    )
    assert varied["kind"] == "variation" and varied["parent"] == job["id"] and varied["parentIndex"] == 1
    detail = (await client.get(f"/api/jobs/{job['id']}")).json()
    assert [child["id"] for child in detail["children"]] == [varied["id"]]
    assert (await client.get(f"/api/jobs/{varied['id']}")).json()["parent"]["id"] == job["id"]

    zoomed = await finished(
        service, recorder, client, await post(client, act, {"kind": "zoom", "index": 1, "value": "1.5"})
    )
    assert zoomed["kind"] == "zoom"
    remixed = await finished(
        service,
        recorder,
        client,
        await post(client, act, {"kind": "vary_strong", "index": 1, "prompt": "an owl"}),
    )
    assert remixed["text"] == "an owl"
    edited = await finished(
        service,
        recorder,
        client,
        await post(client, act, {"kind": "edit", "index": 1, "prompt": "make it night"}),
    )
    assert edited["kind"] == "edit" and edited["instruction"]
    rerolled = await finished(service, recorder, client, await post(client, act, {"kind": "reroll"}))
    assert rerolled["kind"] == "imagine" and rerolled["id"] != job["id"]

    kept = await post(client, act, {"kind": "look", "index": 1, "prompt": "Foxes"})
    assert "Started the Lookbook `Foxes`" in kept.json()["message"]


async def test_uploads_and_references(site):
    service, recorder, web, client = site
    first = await upload(client, (200, 40, 40), "red.png")
    assert first["name"] == "red.png" and first["url"] == f"/media/upload/{first['id']}"
    assert (await upload(client, (200, 40, 40), "again.png"))["id"] == first["id"], (
        "the same picture is kept once"
    )
    second = await upload(client, (40, 40, 201))
    listed = (await client.get("/api/uploads")).json()["uploads"]
    assert [item["id"] for item in listed] == [second["id"], first["id"]]
    assert (await client.get(first["url"], params={"w": 256})).headers["content-type"] == "image/jpeg"

    not_a_picture = await client.post(
        "/api/uploads", files={"file": ("notes.txt", b"hello", "text/plain")}, headers={"Origin": ORIGIN}
    )
    assert not_a_picture.status_code == 400
    web.config.web.max_upload_mb = 1
    too_big = await client.post(
        "/api/uploads", files={"file": ("big.png", b"x" * (2 << 20), "image/png")}, headers={"Origin": ORIGIN}
    )
    assert too_big.status_code == 413 and "larger than 1 MB" in too_big.json()["error"]
    private = await post(client, "/api/uploads", {"url": "http://127.0.0.1:9/x.png"})
    assert private.status_code == 400 and "isn't a public address" in private.json()["error"]
    assert not list((web.config.scratch_dir / "incoming").glob("*")), "nothing is left behind"

    styled = await finished(
        service,
        recorder,
        client,
        await post(
            client,
            "/api/imagine",
            {
                "prompt": "a cottage --sref @1::2 @2 --n 1",
                "attachments": [{"upload": first["id"]}, {"upload": second["id"]}],
            },
        ),
    )
    assert [part["share"] for part in styled["style"]["sources"]] == [0.667, 0.333] and styled["style"][
        "board"
    ]
    from_gallery = await finished(
        service,
        recorder,
        client,
        await post(
            client,
            "/api/imagine",
            {
                "prompt": "a harbour --sref @1 --n 1 --sv 3",
                "attachments": [{"job": styled["id"], "index": 1}],
            },
        ),
    )
    assert from_gallery["style"]["sources"][0]["label"] == "image 1"
    gone = await post(client, "/api/imagine", {"prompt": "x", "attachments": [{"upload": "nope"}]})
    assert gone.status_code == 404

    described = await finished(
        service, recorder, client, await post(client, "/api/describe", {"picture": {"upload": first["id"]}})
    )
    assert described["kind"] == "describe" and len(described["texts"]) == 4
    assert (await post(client, "/api/blend", {"pictures": [{"upload": first["id"]}]})).status_code == 400

    removed = await client.delete(f"/api/uploads/{first['id']}", headers={"Origin": ORIGIN})
    assert removed.status_code == 200 and (await client.get(first["url"])).status_code == 404
    assert styled["status"] == "done", "the job that used it kept its own copy"


# --------------------------------------------------------------------------------------
# Organising
# --------------------------------------------------------------------------------------


async def test_the_gallery_filters_and_bulk_actions(site):
    service, recorder, _, client = site
    fox = await finished(
        service, recorder, client, await post(client, "/api/imagine", {"prompt": "a red fox --n 2"})
    )
    owl = await finished(
        service,
        recorder,
        client,
        await post(client, "/api/imagine", {"prompt": "a grey owl --n 1 --model qwen-image"}),
    )

    async def gallery(**params) -> list[tuple[str, int]]:
        found = (await client.get("/api/images", params=params)).json()
        return [(image["job"], image["index"]) for image in found["images"]]

    everything = [(owl["id"], 1), (fox["id"], 1), (fox["id"], 2)]
    assert await gallery() == everything
    assert (
        await gallery(q="owl") == [(owl["id"], 1)] and await gallery(model="z-image-turbo") == everything[1:]
    )
    assert await gallery(kind="variation") == [] and await gallery(liked="1") == []
    today = (await client.get("/api/facets")).json()["days"][0][0]
    assert await gallery(day=today) == everything and await gallery(day="2001-01-01") == []
    assert (await client.get("/api/images", params={"day": "yesterday"})).status_code == 400
    assert await gallery(limit=1) == everything[:1]

    async def bulk(action: str, items: list[tuple[str, int]], **more) -> httpx.Response:
        body = {"action": action, "items": [{"job": job, "index": index} for job, index in items], **more}
        return await post(client, "/api/images/bulk", body)

    assert (await bulk("like", [(fox["id"], 2), (owl["id"], 1)])).json()["count"] == 2
    assert await gallery(liked="1") == [(owl["id"], 1), (fox["id"], 2)]
    await bulk("unlike", [(owl["id"], 1)])
    assert await gallery(liked="1") == [(fox["id"], 2)]

    await bulk("trash", [(fox["id"], 1)])
    assert await gallery() == [(owl["id"], 1), (fox["id"], 2)] and await gallery(trashed="1") == [
        (fox["id"], 1)
    ]
    assert (await client.get(f"/api/jobs/{fox['id']}")).json()["job"]["images"][0]["trashed"]
    assert Path(service.store.get_job(fox["id"]).images[0].path).is_file(), "trash only hides"
    await bulk("restore", [(fox["id"], 1)])
    assert await gallery() == everything

    folder = (await post(client, "/api/folders", {"name": " Animals "})).json()["folder"]
    assert folder["name"] == "Animals"
    assert (await post(client, "/api/folders", {"name": "animals"})).status_code == 400
    assert (await post(client, "/api/folders", {"name": ""})).status_code == 400
    await bulk("folder-add", [(fox["id"], 1), (owl["id"], 1)], folder=folder["id"])
    assert await gallery(folder=folder["id"]) == [(owl["id"], 1), (fox["id"], 1)]
    assert (await client.get("/api/folders")).json()["folders"][0]["count"] == 2
    assert (await client.get(f"/api/jobs/{fox['id']}")).json()["folders"] == {"1": [folder["id"]]}
    await bulk("folder-remove", [(owl["id"], 1)], folder=folder["id"])
    assert await gallery(folder=folder["id"]) == [(fox["id"], 1)]
    assert (await bulk("folder-add", [(fox["id"], 1)], folder="gone")).status_code == 404
    renamed = await client.patch(
        f"/api/folders/{folder['id']}", json={"name": "Wildlife"}, headers={"Origin": ORIGIN}
    )
    assert renamed.json()["folders"][0]["name"] == "Wildlife"
    assert (await client.delete(f"/api/folders/{folder['id']}", headers={"Origin": ORIGIN})).json()[
        "folders"
    ] == []
    assert await gallery() == everything, "deleting a folder leaves its images where they were"
    assert (await bulk("explode", everything)).status_code == 400 and (
        await bulk("like", [])
    ).status_code == 400

    facets = (await client.get("/api/facets")).json()
    assert dict(facets["models"]) == {"z-image-turbo": 1, "qwen-image": 1} and dict(facets["kinds"]) == {
        "imagine": 2
    }

    body = {
        "items": [
            {"job": fox["id"], "index": 1},
            {"job": owl["id"], "index": 1},
            {"job": "GONE00", "index": 1},
        ]
    }
    archive = await post(client, "/api/images/download", body)
    assert archive.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
        assert len(bundle.namelist()) == 2 and all(name.endswith(".png") for name in bundle.namelist())
    assert (
        await post(client, "/api/images/download", {"items": [{"job": "GONE00", "index": 1}]})
    ).status_code == 404
    assert not list((service.config.scratch_dir / "downloads").glob("*.zip")), "the zip is removed once sent"


# --------------------------------------------------------------------------------------
# Lookbooks
# --------------------------------------------------------------------------------------


async def test_lookbooks_in_the_browser(site):
    service, recorder, _, client = site
    made = await post(client, "/api/lookbooks", {"name": "Autumn Diner"})
    book = made.json()["lookbook"]
    assert book["slug"] == "autumn-diner" and book["images"] == [] and book["code"] is None
    assert (await post(client, "/api/lookbooks", {"name": "autumn diner"})).status_code == 400

    red, blue = await upload(client, (200, 40, 40)), await upload(client, (40, 40, 201))
    job = await finished(
        service, recorder, client, await post(client, "/api/imagine", {"prompt": "a fox --n 1"})
    )
    images = f"/api/lookbooks/{book['id']}/images"
    added = await post(
        client,
        images,
        {"pictures": [{"upload": red["id"]}, {"upload": blue["id"]}, {"job": job["id"], "index": 1}]},
    )
    book = added.json()["lookbook"]
    assert book["count"] == 3 and lookbook.looks_like_code(book["code"])
    assert [image["source"] for image in book["images"]] == ["upload", "upload", f"job:{job['id']}:1"]
    picture = await client.get(book["images"][0]["url"], params={"w": 256})
    assert picture.status_code == 200 and picture.headers["content-type"] == "image/jpeg"
    assert (await post(client, images, {})).status_code == 400
    assert (await post(client, images, {"url": "http://127.0.0.1:9/x.png"})).status_code == 400

    await settle(service)
    read = (await client.get(f"/api/lookbooks/{book['id']}")).json()["lookbook"]
    assert read["unread"] == 0 and read["style"]["read"] and len(read["style"]["swatches"]) >= 2

    chosen = await client.put(
        "/api/lookbooks/defaults", json={"lookbooks": [book["id"]]}, headers={"Origin": ORIGIN}
    )
    assert chosen.json()["lookbooks"] == [book["id"]]
    assert (await client.get("/api/lookbooks")).json()["lookbooks"][0]["default"]
    assert (await client.get("/api/state")).json()["settings"]["looks"] == [book["id"]]

    styled = await finished(
        service,
        recorder,
        client,
        await post(client, "/api/imagine", {"prompt": "a harbour --p --n 1 --sv 3"}),
    )
    assert styled["style"]["sources"][0]["label"] == "Lookbook Autumn Diner"
    assert f"--personalize {book['code']}" in styled["prompt"]
    by_look = (await client.get("/api/images", params={"look": book["code"]})).json()["images"]
    assert [image["job"] for image in by_look] == [styled["id"]]
    assert (await client.get("/api/facets")).json()["looks"] == [[book["code"], "Lookbook Autumn Diner"]]

    smaller = await client.delete(f"{images}/{book['images'][2]['id']}", headers={"Origin": ORIGIN})
    assert smaller.json()["lookbook"]["count"] == 2 and smaller.json()["lookbook"]["code"] != book["code"]
    renamed = await client.patch(
        f"/api/lookbooks/{book['id']}", json={"name": "Diner"}, headers={"Origin": ORIGIN}
    )
    assert renamed.json()["lookbook"]["slug"] == "diner"
    assert (
        await client.delete(f"/api/lookbooks/{book['id']}", headers={"Origin": ORIGIN})
    ).status_code == 200
    assert (await client.get("/api/lookbooks")).json()["lookbooks"] == []
    assert (await client.get(f"/api/lookbooks/{book['id']}")).status_code == 404


# --------------------------------------------------------------------------------------
# Styles
# --------------------------------------------------------------------------------------


async def test_exploring_liking_blending_and_saving_styles(site):
    service, recorder, _, client = site
    page = (await client.get("/api/styles", params={"seed": "7", "count": 12})).json()["styles"]
    assert len(page) == 12 and len({card["code"] for card in page}) == 12
    assert (await client.get("/api/styles", params={"seed": "7", "count": 12})).json()["styles"] == page
    assert (await client.get("/api/styles", params={"seed": "8", "count": 12})).json()["styles"] != page
    card = page[0]
    assert card["kind"] == "code" and card["use"] == card["code"] and not card["liked"]
    assert card["name"] and card["summary"] and len(card["swatches"]) == 5 and card["words"]["medium"]

    anime = (await client.get("/api/styles", params={"q": "anime pastel", "count": 6})).json()["styles"]
    assert len(anime) == 6 and all(c["words"]["medium"] == "anime illustration" for c in anime)
    assert (await client.get("/api/styles", params={"q": "zzzz"})).json()["styles"] == []

    detail = (await client.get(f"/api/styles/{card['code']}")).json()
    assert detail["style"]["code"] == card["code"] and len(detail["similar"]) >= 5
    assert (await client.get("/api/styles/nothing-here")).status_code == 404

    assert (await post(client, "/api/styles/like", {"code": card["code"]})).json()["liked"]
    mine = (await client.get("/api/styles/mine")).json()
    assert (
        [c["code"] for c in mine["liked"]] == [card["code"]]
        and mine["liked"][0]["liked"]
        and mine["saved"] == []
    )
    assert (await client.get("/api/styles", params={"seed": "7", "count": 12})).json()["styles"][0]["liked"]
    await post(client, "/api/styles/like", {"code": card["code"], "liked": False})
    assert (await client.get("/api/styles/mine")).json()["liked"] == []

    # The bench: a code and a picture at 2:1, described without rendering anything.
    red = await upload(client, (200, 40, 40))
    parts = [{"code": "4821", "weight": 2}, {"upload": red["id"], "weight": 1}]
    blend = (await post(client, "/api/styles/blend", {"parts": parts})).json()
    assert (
        blend["sref"] == "4821::2 @1" and blend["attachments"] == [{"upload": red["id"]}] and blend["direct"]
    )
    assert [part["share"] for part in blend["parts"]] == [0.667, 0.333]
    assert blend["style"]["words"]["medium"] == fingerprint.from_code("4821").medium
    assert "#c82828" in [swatch["hex"] for swatch in blend["style"]["swatches"]], (
        "the picture's red is in the mix"
    )
    assert blend["phrases"][0].startswith("in the style of")
    assert (await post(client, "/api/styles/blend", {"parts": []})).status_code == 400
    assert (await post(client, "/api/styles/blend", {"parts": [{"code": "nope"}]})).status_code == 404

    tried = await finished(
        service,
        recorder,
        client,
        await post(
            client,
            "/api/imagine",
            {"prompt": f"a fox --n 1 --sref {blend['sref']}", "attachments": blend["attachments"]},
        ),
    )
    assert [part["share"] for part in tried["style"]["sources"]] == [0.667, 0.333]

    saved = (await post(client, "/api/styles", {"name": "Red Riso", "parts": parts})).json()["style"]
    assert saved["kind"] == "saved" and saved["use"] == "red-riso" and saved["code"].startswith("s")
    assert saved["recipe"] == [{"label": "style 4821", "share": 0.667}, {"label": "picture", "share": 0.333}]
    assert (await client.get(saved["pictures"][0], params={"w": 256})).status_code == 200
    assert (await post(client, "/api/styles", {"name": "9 lives", "parts": parts})).status_code == 400
    assert [c["name"] for c in (await client.get("/api/styles/mine")).json()["saved"]] == ["Red Riso"]
    assert (await client.get(f"/api/styles/{saved['code']}")).json()["style"]["name"] == "Red Riso"

    used = await finished(
        service,
        recorder,
        client,
        await post(client, "/api/imagine", {"prompt": "a fox --n 1 --sref red-riso --sv 3"}),
    )
    assert used["style"]["sources"] == [{"kind": "saved", "label": "style Red Riso", "share": 1.0}]

    # A saved style and a Lookbook can go into a blend too; one with a Lookbook is used by saving it.
    book = (await post(client, "/api/lookbooks", {"name": "Blues"})).json()["lookbook"]
    blue = await upload(client, (40, 40, 201))
    await post(client, f"/api/lookbooks/{book['id']}/images", {"pictures": [{"upload": blue["id"]}]})
    mixed = (
        await post(
            client,
            "/api/styles/blend",
            {"parts": [{"code": "red-riso"}, {"lookbook": book["id"], "weight": 3}]},
        )
    ).json()
    assert not mixed["direct"] and mixed["sref"] == "red-riso" and mixed["parts"][1]["label"] == "Blues"
    assert "#2828c9" in [swatch["hex"] for swatch in mixed["style"]["swatches"]]

    assert (
        await client.delete(f"/api/styles/{saved['code']}", headers={"Origin": ORIGIN})
    ).status_code == 200
    assert (await client.get("/api/styles/mine")).json()["saved"] == []
    assert (await post(client, "/api/imagine", {"prompt": "a fox --sref red-riso"})).status_code == 400


async def test_events_carry_the_job_as_the_page_shows_it(site):
    service, recorder, web, client = site
    queue = web.hub.open()
    job = await finished(
        service, recorder, client, await post(client, "/api/imagine", {"prompt": "a fox --n 1"})
    )
    seen = []
    while not queue.empty():
        seen.append(queue.get_nowait())
    types = [event["type"] for event in seen]
    assert types[0] == "queued" and types[-1] == "finished" and "started" in types and "progress" in types
    assert all(event["job"]["id"] == job["id"] for event in seen)
    assert seen[-1]["fraction"] == 1.0 and seen[-1]["job"]["images"][0]["url"] == job["images"][0]["url"]
    assert service.store.get_job(job["id"]).status is JobStatus.DONE

    # A tab that stops reading is told to reload rather than fed a backlog.
    for number in range(300):
        web.hub.publish({"type": "noise", "n": number})
    assert queue.qsize() < 256 and queue.get_nowait() == {"type": "resync"}
    web.hub.close(queue)
    web.hub.publish({"type": "noise"})
    assert queue.qsize() < 256
