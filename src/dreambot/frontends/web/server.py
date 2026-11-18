"""The web app: Midjourney's website, served from this Mac to this Mac.

Create (a prompt bar, a live feed, a lightbox with every action), Organize (search, filters,
folders, trash), Lookbooks, and a Styles explorer with a bench for blending styles by weight.

Like the chat frontends this is thin. It turns requests into calls on
:class:`~dreambot.core.service.Dreambot` and relays the job events it emits; an action pressed
in the browser is the same :class:`~dreambot.core.actions.Action` a Discord button sends, so the
two cannot drift apart. Everything made through the chat apps shows up here too: there is one
database.

**Who can reach it.** The server listens on 127.0.0.1 and the app has no accounts: whoever
opens it is at the Mac and is its owner. A local web server is still reachable by every web page
the browser has open, so three things keep those out:

* the ``Host`` header must be this server's own address, which stops another site pointing its
  own name at 127.0.0.1 (DNS rebinding);
* every API and image request needs a session cookie that only this app's own page is given,
  marked ``SameSite=Strict`` so that no other site's request carries it;
* a request that changes something must come from this app's own origin.

Images are served by job id, upload id or Lookbook id — never by file path.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import random
import secrets
import shutil
import socket
import time
import zipfile
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import uvicorn
from PIL import Image
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp, Receive, Scope, Send

from dreambot import __version__
from dreambot.core import fingerprint, imaging, lookbook
from dreambot.core.access import LOCAL_WEB_USER
from dreambot.core.actions import Action, ActionKind
from dreambot.core.fetch import FetchError, fetch_image
from dreambot.core.fingerprint import StyleFingerprint
from dreambot.core.imaging import ImageError
from dreambot.core.jobs import EventType, Job, JobEvent, JobStatus, new_job_id
from dreambot.core.params import PromptError
from dreambot.core.service import Dreambot, DreambotError, style_card
from dreambot.engine.protocol import Op

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"


def _build() -> str:
    """A short name for the page's own files as they are now. It changes exactly when one does."""
    digest = hashlib.sha256()
    for path in sorted(STATIC.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(STATIC).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


class _Static(StaticFiles):
    """The page's scripts and styles, checked with the server each time they are loaded.

    With no word on caching a browser guesses, and goes on running last week's script next to
    this week's server. Asked every time, it is told "not modified" unless Dreambot was updated.

    That only helps a browser that asks. One that already holds a script from before, with
    leave to keep it for hours, never does: so the page also loads its files from an address
    with :func:`_build` in it, which no browser can have seen before an update.
    """

    async def get_response(self, path: str, scope: Any) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


COOKIE = "dreambot"
CHANNEL = "web:main"
THUMB_WIDTHS = (256, 512, 1024)
MAX_BODY = 1 << 20
#: How many images one request may put in a download.
MAX_DOWNLOAD = 400
_UNSAFE = frozenset({"POST", "PUT", "PATCH", "DELETE"})
CSP = (
    "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
    "script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
)


class WebError(Exception):
    """A request that cannot be served. The message is shown to the person."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def session_token(home: Path) -> str:
    """The secret this install's web app shares with its own page, made once and kept."""
    path = home / "web-token"
    with contextlib.suppress(OSError):
        token = path.read_text().strip()
        if len(token) >= 32:
            return token
    token = secrets.token_urlsafe(32)
    home.mkdir(parents=True, exist_ok=True)
    path.write_text(token)
    path.chmod(0o600)
    return token


# --------------------------------------------------------------------------------------
# The guard
# --------------------------------------------------------------------------------------


class Guard:
    """Lets in the app's own page and nothing else. See the module docstring."""

    def __init__(self, app: ASGIApp, *, hosts: frozenset[str], token: str):
        self.app = app
        self.hosts = hosts
        self.token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        host = request.headers.get("host", "").lower()
        if host not in self.hosts:
            await _refuse(403, "This address is not the Dreambot web app.")(scope, receive, send)
            return
        path = scope["path"]
        if path.startswith(("/api/", "/media/")):
            given = request.cookies.get(COOKIE, "")
            if not secrets.compare_digest(given, self.token):
                await _refuse(401, "Open the Dreambot web app's own page first.")(scope, receive, send)
                return
        if request.method in _UNSAFE:
            origin = request.headers.get("origin")
            site = request.headers.get("sec-fetch-site")
            if (origin is not None and origin.lower() != f"http://{host}") or (
                site is not None and site not in ("same-origin", "none")
            ):
                await _refuse(403, "That request did not come from the Dreambot web app.")(
                    scope, receive, send
                )
                return
        await self.app(scope, receive, send)


def _refuse(status: int, message: str) -> Response:
    return JSONResponse({"error": message}, status_code=status)


# --------------------------------------------------------------------------------------
# Live events
# --------------------------------------------------------------------------------------


class EventHub:
    """Fans job events out to every open browser tab."""

    def __init__(self) -> None:
        self._listeners: set[asyncio.Queue[dict[str, Any]]] = set()

    def open(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        self._listeners.add(queue)
        return queue

    def close(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._listeners.discard(queue)

    def publish(self, payload: dict[str, Any]) -> None:
        for queue in list(self._listeners):
            if queue.full():
                # A tab that has stopped reading: drop what it missed and tell it to reload its feed.
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait({"type": "resync"})
            else:
                queue.put_nowait(payload)


# --------------------------------------------------------------------------------------
# The app
# --------------------------------------------------------------------------------------

Handler = Callable[[Request], Awaitable[Response]]


class WebApp:
    """The routes of the web app, bound to one running Dreambot."""

    def __init__(self, service: Dreambot, *, hosts: frozenset[str] | None = None):
        self.service = service
        self.config = service.config
        self.store = service.store
        self.user = LOCAL_WEB_USER
        self.token = session_token(self.config.home)
        self.hub = EventHub()
        self.cache = self.config.home / "cache"
        self.uploads = self.config.home / "uploads"
        self._previews: dict[str, int] = {}
        self._unsubscribe = service.subscribe(self._on_job_event)
        #: Part of the address the page loads its scripts and styles from (see :class:`_Static`).
        self.build = _build()
        web = self.config.web
        allowed = hosts or frozenset(
            {f"127.0.0.1:{web.port}", f"localhost:{web.port}", f"[::1]:{web.port}", f"{web.host}:{web.port}"}
        )
        get, post, put, patch, delete = (
            lambda path, handler, method=method: Route(path, self._wrap(handler), methods=[method])
            for method in ("GET", "POST", "PUT", "PATCH", "DELETE")
        )
        routes = [
            Route("/", self.index),
            get("/api/state", self.state),
            get("/api/events", self.events),
            get("/api/jobs", self.jobs),
            get("/api/jobs/{job}", self.job),
            post("/api/imagine", self.imagine),
            post("/api/describe", self.describe),
            post("/api/blend", self.blend),
            post("/api/jobs/{job}/act", self.act),
            put("/api/settings", self.save_settings),
            get("/api/uploads", self.list_uploads),
            post("/api/uploads", self.upload),
            delete("/api/uploads/{upload}", self.delete_upload),
            get("/api/images", self.images),
            get("/api/facets", self.facets),
            post("/api/images/bulk", self.bulk),
            post("/api/images/download", self.download),
            get("/api/folders", self.folders),
            post("/api/folders", self.create_folder),
            patch("/api/folders/{folder}", self.rename_folder),
            delete("/api/folders/{folder}", self.delete_folder),
            get("/api/lookbooks", self.lookbooks),
            post("/api/lookbooks", self.create_lookbook),
            put("/api/lookbooks/defaults", self.default_lookbooks),
            get("/api/lookbooks/{book}", self.lookbook),
            patch("/api/lookbooks/{book}", self.rename_lookbook),
            delete("/api/lookbooks/{book}", self.delete_lookbook),
            post("/api/lookbooks/{book}/images", self.add_lookbook_image),
            delete("/api/lookbooks/{book}/images/{image}", self.remove_lookbook_image),
            get("/api/styles", self.styles),
            get("/api/styles/mine", self.my_styles),
            post("/api/styles/blend", self.blend_styles),
            post("/api/styles", self.save_style),
            post("/api/styles/like", self.like_style),
            delete("/api/styles/{code}", self.delete_style),
            get("/api/styles/{code}", self.style),
            get("/media/job/{job}/{index:int}", self.job_image),
            get("/media/grid/{job}", self.job_grid),
            get("/media/preview/{job}", self.job_preview),
            get("/media/upload/{upload}", self.upload_image),
            get("/media/lookbook/{book}/{image}", self.lookbook_image),
            get("/media/style/{code}/{number:int}", self.style_image),
            Mount(f"/static/{self.build}", _Static(directory=STATIC), name="built"),
            Mount("/static", _Static(directory=STATIC), name="static"),
        ]
        self.app: ASGIApp = Guard(Starlette(routes=routes), hosts=allowed, token=self.token)

    def close(self) -> None:
        self._unsubscribe()

    # ------------------------------------------------------------------ plumbing ----

    def _wrap(self, handler: Handler) -> Handler:
        """Turn the errors a person should be told about into answers they can read."""

        async def wrapped(request: Request) -> Response:
            try:
                return await handler(request)
            except WebError as exc:
                return JSONResponse({"error": str(exc)}, status_code=exc.status)
            except (DreambotError, PromptError, ImageError, FetchError) as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
            except Exception:
                log.exception("the web app failed on %s %s", request.method, request.url.path)
                return JSONResponse({"error": "Something went wrong inside Dreambot."}, status_code=500)

        return wrapped

    @staticmethod
    async def _body(request: Request) -> dict[str, Any]:
        declared = request.headers.get("content-length", "0")
        if declared.isdigit() and int(declared) > MAX_BODY:
            raise WebError("That request is too large.", 413)
        raw = await request.body()
        if len(raw) > MAX_BODY:
            raise WebError("That request is too large.", 413)
        try:
            data = json.loads(raw or b"{}")
        except ValueError as exc:
            raise WebError("That request could not be read.") from exc
        if not isinstance(data, dict):
            raise WebError("That request could not be read.")
        return data

    def _job(self, job_id: str) -> Job:
        job = self.store.get_job(job_id)
        if job is None:
            raise WebError("I can't find that job.", 404)
        return job

    def _picture(self, ref: Any) -> str:
        """The file a picture named by the browser stands for: an upload, or an image of a job."""
        if isinstance(ref, dict) and ref.get("upload"):
            upload = self.store.get_upload(str(ref["upload"]))
            if upload is None or not Path(upload["path"]).is_file():
                raise WebError("That upload is gone.", 404)
            return upload["path"]
        if isinstance(ref, dict) and ref.get("job"):
            job = self._job(str(ref["job"]))
            try:
                image = job.image(int(ref.get("index") or 1))
            except (KeyError, ValueError, TypeError) as exc:
                raise WebError("That job has no such image.", 404) from exc
            if not Path(image.path).is_file():
                raise WebError("That image's file is no longer on disk.", 404)
            return image.path
        raise WebError("That picture could not be found.")

    @staticmethod
    def _items(data: dict[str, Any]) -> list[tuple[str, int]]:
        items = data.get("items")
        if not isinstance(items, list) or not items:
            raise WebError("Choose some images first.")
        try:
            return [(str(item["job"]).upper(), int(item["index"])) for item in items]
        except (KeyError, TypeError, ValueError) as exc:
            raise WebError("Those images could not be read.") from exc

    # --------------------------------------------------------------------- views ----

    def job_view(self, job: Job, trashed: set[tuple[str, int]] | None = None) -> dict[str, Any]:
        trashed = self.store.trashed() if trashed is None else trashed
        return {
            "id": job.id,
            "kind": job.kind.value,
            "status": job.status.value,
            "user": job.user,
            "prompt": job.prompt,
            "text": job.text,
            "model": job.model,
            "width": job.width,
            "height": job.height,
            "created": job.created,
            "seconds": job.seconds,
            "parent": job.parent,
            "parentIndex": job.parent_index,
            "notes": job.notes,
            "error": job.error,
            "style": job.detail.get("style"),
            "loras": [{"name": lora["name"], "scale": lora["scale"]} for lora in job.detail.get("loras", [])],
            "enhanced": bool(job.detail.get("enhanced")),
            "instruction": bool(job.detail.get("instruction")),
            "count": len(job.seeds) or len(job.images),
            "texts": job.texts,
            "images": [
                {
                    "index": image.index,
                    "seed": image.seed,
                    "width": image.width,
                    "height": image.height,
                    "favorite": image.favorite,
                    "trashed": (job.id, image.index) in trashed,
                    "prompt": image.prompt,
                    "url": f"/media/job/{job.id}/{image.index}",
                }
                for image in job.images
            ],
        }

    def _book_view(self, book: dict[str, Any]) -> dict[str, Any]:
        view = dict(book)
        view["images"] = [
            {
                "id": image["id"],
                "source": image["source"],
                "style": image["style"],
                "url": f"/media/lookbook/{book['id']}/{image['id']}",
            }
            for image in book["images"]
        ]
        return view

    def _code_card(self, code: str, liked: set[str]) -> dict[str, Any]:
        card = style_card(fingerprint.from_code(code)) or {}
        return {**card, "code": code, "kind": "code", "liked": code in liked, "use": code}

    def _saved_card(self, style: dict[str, Any]) -> dict[str, Any]:
        data = style["data"] or {}
        card = style_card(StyleFingerprint.from_dict(data.get("print") or {})) or {}
        return {
            **card,
            "code": style["code"],
            "kind": "saved",
            "name": style["name"],
            "liked": style["liked"],
            "use": lookbook.slug(style["name"]),
            "recipe": data.get("recipe") or [],
            "pictures": [
                f"/media/style/{style['code']}/{n}" for n in range(1, len(data.get("pictures", [])) + 1)
            ],
        }

    # ---------------------------------------------------------------------- page ----

    async def index(self, request: Request) -> Response:
        # The page itself is never kept by the browser, and names its files by this build.
        page = (STATIC / "index.html").read_text().replace('"/static/', f'"/static/{self.build}/')
        response = HTMLResponse(page)
        response.set_cookie(COOKIE, self.token, httponly=True, samesite="strict", max_age=365 * 24 * 3600)
        response.headers["Content-Security-Policy"] = CSP
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    async def state(self, request: Request) -> Response:
        service = self.service
        resolved = service.resolved_settings(self.user)
        try:
            model = service.planner.default_model(resolved).id
        except Exception:
            model = None
        models = [
            {
                "id": spec.id,
                "title": spec.title,
                "generates": spec.supports(Op.TXT2IMG),
                "edits": spec.supports(Op.EDIT),
            }
            for spec in service.catalog.installed()
        ]
        settings = service.settings(self.user)
        return JSONResponse(
            {
                "version": __version__,
                "backend": self.config.engine.backend,
                "model": model,
                "models": models,
                # The LoRA library: fine-tuned looks a prompt can be given by name.
                "loras": [
                    {
                        "name": entry.name,
                        "title": entry.title or entry.name,
                        "base": entry.base,
                        # LoRAs for one family of models can be used together; others cannot.
                        "family": base.family if (base := service.catalog.registry.find(entry.base)) else "",
                        "trigger": entry.trigger,
                    }
                    for entry in service.planner.loras.entries()[0]
                ],
                "vision": service.catalog.vision_model() is not None,
                "queue": service.queue_length,
                "settings": {
                    "model": settings.model,
                    "loras": list(resolved.loras),
                    "stylize": resolved.stylize,
                    "count": resolved.count,
                    "raw": resolved.raw,
                    "variation": resolved.variation,
                    "remix": resolved.remix,
                    "hd": resolved.hd,
                    "enhance": resolved.enhance,
                    "personalize": resolved.personalize,
                    "looks": list(resolved.looks),
                },
                "output": str(self.config.output_dir),
            }
        )

    async def save_settings(self, request: Request) -> Response:
        data = await self._body(request)
        settings = self.service.settings(self.user)
        for key in ("raw", "remix", "hd", "enhance", "personalize"):
            if key in data:
                setattr(settings, key, bool(data[key]))
        if "model" in data:
            settings.model = str(data["model"]) if data["model"] else None
        if "loras" in data:
            wanted = data["loras"] if isinstance(data["loras"], list) else []
            settings.loras = [str(name) for name in wanted if name]
        if "stylize" in data:
            settings.stylize = max(0, min(1000, int(data["stylize"])))
        if "count" in data:
            settings.count = max(1, min(16, int(data["count"])))
        if "variation" in data:
            if data["variation"] not in ("strong", "subtle"):
                raise WebError("Variation mode is strong or subtle.")
            settings.variation = data["variation"]
        self.service.save_settings(self.user, settings)
        return await self.state(request)

    # -------------------------------------------------------------------- events ----

    def _on_job_event(self, event: JobEvent) -> None:
        job = event.job
        payload: dict[str, Any] = {
            "type": event.type.value,
            "job": self.job_view(job),
            "fraction": round(event.fraction, 3),
            "position": event.position,
        }
        if event.type is EventType.PREVIEW and event.preview:
            # The worker's preview file is gone when the job ends; keep the latest where it can be served.
            target = self.cache / "previews" / f"{job.id}.png"
            with contextlib.suppress(OSError):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(event.preview, target)
                self._previews[job.id] = self._previews.get(job.id, 0) + 1
        if job.id in self._previews:
            payload["preview"] = f"/media/preview/{job.id}?v={self._previews[job.id]}"
        if job.status.finished:
            self._previews.pop(job.id, None)
            payload.pop("preview", None)
        self.hub.publish(payload)

    async def events(self, request: Request) -> Response:
        hub = self.hub

        async def stream() -> AsyncIterator[str]:
            queue = hub.open()
            try:
                yield "retry: 2000\n\n"
                yield f"data: {json.dumps({'type': 'hello', 'queue': self.service.queue_length})}\n\n"
                while True:
                    try:
                        payload = await asyncio.wait_for(queue.get(), timeout=15)
                    except TimeoutError:
                        yield ": still here\n\n"
                        continue
                    yield f"data: {json.dumps(payload)}\n\n"
            finally:
                hub.close(queue)

        headers = {"Cache-Control": "no-store", "X-Accel-Buffering": "no"}
        return StreamingResponse(stream(), media_type="text/event-stream", headers=headers)

    # ---------------------------------------------------------------------- jobs ----

    async def jobs(self, request: Request) -> Response:
        query = request.query_params
        before = float(query["before"]) if query.get("before") else None
        limit = max(1, min(100, int(query.get("limit", 30))))
        found = self.store.jobs_page(before=before, limit=limit, search=query.get("q") or None)
        trashed = self.store.trashed()
        return JSONResponse(
            {
                "jobs": [self.job_view(job, trashed) for job in found],
                "next": found[-1].created if len(found) == limit else None,
            }
        )

    async def job(self, request: Request) -> Response:
        job = self._job(request.path_params["job"])
        trashed = self.store.trashed()
        parent = self.store.get_job(job.parent) if job.parent else None
        return JSONResponse(
            {
                "job": self.job_view(job, trashed),
                "parent": self.job_view(parent, trashed) if parent else None,
                "children": [self.job_view(child, trashed) for child in self.store.children_of(job.id)],
                "folders": {
                    str(index): folders
                    for (owner, index), folders in self.store.folder_items().items()
                    if owner == job.id
                },
            }
        )

    async def imagine(self, request: Request) -> Response:
        data = await self._body(request)
        prompt = str(data.get("prompt") or "").strip()
        attachments = [self._picture(ref) for ref in data.get("attachments") or []]
        if not prompt and not attachments:
            raise WebError("Type a prompt first.")
        jobs = await self.service.imagine(
            user=self.user, channel=CHANNEL, prompt=prompt, attachments=attachments
        )
        return JSONResponse({"jobs": [self.job_view(job) for job in jobs]})

    async def describe(self, request: Request) -> Response:
        data = await self._body(request)
        job = await self.service.describe(
            user=self.user, channel=CHANNEL, image=self._picture(data.get("picture"))
        )
        return JSONResponse({"jobs": [self.job_view(job)]})

    async def blend(self, request: Request) -> Response:
        data = await self._body(request)
        pictures = [self._picture(ref) for ref in data.get("pictures") or []]
        if not 2 <= len(pictures) <= 5:
            raise WebError("Blend takes two to five pictures.")
        job = await self.service.blend(user=self.user, channel=CHANNEL, images=pictures)
        return JSONResponse({"jobs": [self.job_view(job)]})

    async def act(self, request: Request) -> Response:
        data = await self._body(request)
        try:
            kind = ActionKind(str(data.get("kind")))
        except ValueError as exc:
            raise WebError("I don't know that action.") from exc
        index = int(data["index"]) if data.get("index") else None
        action = Action(
            kind,
            request.path_params["job"],
            index,
            str(data["value"]) if data.get("value") not in (None, "") else None,
            str(data["prompt"]).strip() if data.get("prompt") else None,
        )
        result = await self.service.act(user=self.user, channel=CHANNEL, action=action)
        return JSONResponse(
            {
                "jobs": [self.job_view(job) for job in result.jobs],
                "message": result.message,
                "job": self.job_view(result.job) if result.job else None,
            }
        )

    # ------------------------------------------------------------------- uploads ----

    def _upload_view(self, upload: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": upload["id"],
            "name": upload["name"],
            "created": upload["created"],
            "url": f"/media/upload/{upload['id']}",
        }

    async def list_uploads(self, request: Request) -> Response:
        uploads = [u for u in self.store.list_uploads() if Path(u["path"]).is_file()]
        return JSONResponse({"uploads": [self._upload_view(upload) for upload in uploads]})

    def _keep_upload(self, source: Path, name: str) -> dict[str, Any]:
        """Normalise a picture into the uploads library, or find it there already."""
        prepared = imaging.prepare_input(source, self.uploads)
        key = fingerprint.image_key(prepared)
        known = self.store.find_upload(key)
        if known is not None and Path(known["path"]).is_file():
            prepared.unlink(missing_ok=True)
            return known
        upload_id = new_job_id().lower()
        target = self.uploads / f"{upload_id}.png"
        prepared.rename(target)
        return self.store.add_upload(upload_id, self.user, str(target), key, name[:120])

    async def upload(self, request: Request) -> Response:
        limit = self.config.web.max_upload_mb * 1024 * 1024
        scratch = self.config.scratch_dir / "incoming"
        scratch.mkdir(parents=True, exist_ok=True)
        kept: list[dict[str, Any]] = []
        if request.headers.get("content-type", "").startswith("application/json"):
            data = await self._body(request)
            url = str(data.get("url") or "").strip()
            if not url:
                raise WebError("Paste a link to an image.")
            downloaded = await asyncio.to_thread(fetch_image, url, scratch)
            try:
                kept.append(await asyncio.to_thread(self._keep_upload, downloaded, url.rsplit("/", 1)[-1]))
            finally:
                downloaded.unlink(missing_ok=True)
        else:
            declared = request.headers.get("content-length", "0")
            if declared.isdigit() and int(declared) > limit * 8:
                raise WebError("That is too much to upload at once.", 413)
            async with request.form(max_files=16, max_part_size=limit) as form:
                files = [value for _, value in form.multi_items() if hasattr(value, "filename")]
                if not files:
                    raise WebError("Choose an image to upload.")
                for item in files:
                    incoming = scratch / f"{new_job_id().lower()}.upload"
                    size = 0
                    with incoming.open("wb") as handle:
                        while chunk := await item.read(1 << 20):
                            size += len(chunk)
                            if size > limit:
                                break
                            handle.write(chunk)
                    try:
                        if size > limit:
                            raise WebError(
                                f"{item.filename} is larger than {self.config.web.max_upload_mb} MB.", 413
                            )
                        kept.append(
                            await asyncio.to_thread(self._keep_upload, incoming, item.filename or "upload")
                        )
                    finally:
                        incoming.unlink(missing_ok=True)
        for upload in kept:
            self.service.study_picture(upload["path"])
        return JSONResponse({"uploads": [self._upload_view(upload) for upload in kept]})

    async def delete_upload(self, request: Request) -> Response:
        upload = self.store.get_upload(request.path_params["upload"])
        if upload is None:
            raise WebError("That upload is gone.", 404)
        self.store.delete_upload(upload["id"])
        # Safe to remove: a prompt, a Lookbook or a saved style that used it made its own copy.
        with contextlib.suppress(OSError):
            Path(upload["path"]).unlink()
        return JSONResponse({"ok": True})

    # ------------------------------------------------------------------- gallery ----

    async def images(self, request: Request) -> Response:
        """The gallery: single images, newest first, filtered."""
        query = request.query_params
        limit = max(1, min(200, int(query.get("limit", 60))))
        before = float(query["before"]) if query.get("before") else None
        stop_at: float | None = None
        if day := query.get("day"):
            try:
                start = datetime.strptime(day, "%Y-%m-%d").timestamp()
            except ValueError as exc:
                raise WebError("That day could not be read.") from exc
            stop_at, end = start, start + 86400
            before = end if before is None else min(before, end)
        want_trash = query.get("trashed") == "1"
        liked = query.get("liked") == "1"
        model, kind, folder, look = (query.get(name) or None for name in ("model", "kind", "folder", "look"))
        trashed = self.store.trashed()
        in_folders = self.store.folder_items()
        only = {key for key, folders in in_folders.items() if folder in folders} if folder else None

        found: list[dict[str, Any]] = []
        cursor = before
        exhausted = False
        while len(found) < limit and not exhausted:
            page = self.store.jobs_page(before=cursor, limit=120, search=query.get("q") or None)
            if len(page) < 120:
                exhausted = True
            for job in page:
                cursor = job.created
                if stop_at is not None and job.created < stop_at:
                    exhausted = True
                    break
                if job.status is not JobStatus.DONE or not job.images:
                    continue
                if (model and job.model != model) or (kind and job.kind.value != kind):
                    continue
                if look and not any(
                    look in (part.get("code"), part.get("label"))
                    for part in (job.detail.get("style") or {}).get("sources", [])
                ):
                    continue
                for image in job.images:
                    key = (job.id, image.index)
                    if (key in trashed) != want_trash or (liked and not image.favorite):
                        continue
                    if only is not None and key not in only:
                        continue
                    found.append(
                        {
                            "job": job.id,
                            "index": image.index,
                            "url": f"/media/job/{job.id}/{image.index}",
                            "prompt": job.prompt,
                            "model": job.model,
                            "kind": job.kind.value,
                            "created": job.created,
                            "width": image.width,
                            "height": image.height,
                            "favorite": image.favorite,
                            "trashed": key in trashed,
                            "folders": in_folders.get(key, []),
                        }
                    )
                if len(found) >= limit:
                    break
        return JSONResponse({"images": found, "next": None if exhausted else cursor})

    async def facets(self, request: Request) -> Response:
        """What the gallery can be filtered by."""
        models: dict[str, int] = {}
        kinds: dict[str, int] = {}
        looks: dict[str, str] = {}
        cursor: float | None = None
        for _ in range(40):  # the newest few thousand jobs are plenty to build the filter lists from
            page = self.store.jobs_page(before=cursor, limit=200)
            for job in page:
                cursor = job.created
                if job.status is not JobStatus.DONE or not job.images:
                    continue
                models[job.model] = models.get(job.model, 0) + 1
                kinds[job.kind.value] = kinds.get(job.kind.value, 0) + 1
                for part in (job.detail.get("style") or {}).get("sources", []):
                    if part.get("kind") == "look" and part.get("code"):
                        looks[part["code"]] = part.get("label", part["code"])
            if len(page) < 200:
                break
        return JSONResponse(
            {
                "models": sorted(models.items(), key=lambda item: -item[1]),
                "kinds": sorted(kinds.items(), key=lambda item: -item[1]),
                "looks": sorted(looks.items(), key=lambda item: item[1]),
                "days": self.store.job_days(),
                "folders": self.store.list_folders(),
                "trash": len(self.store.trashed()),
            }
        )

    async def bulk(self, request: Request) -> Response:
        data = await self._body(request)
        items = self._items(data)
        action = data.get("action")
        if action in ("like", "unlike"):
            by_job: dict[str, list[int]] = {}
            for job_id, index in items:
                by_job.setdefault(job_id, []).append(index)
            for job_id, indexes in by_job.items():
                job = self._job(job_id)
                for image in job.images:
                    if image.index in indexes:
                        image.favorite = action == "like"
                self.store.save_job(job)
        elif action in ("trash", "restore"):
            self.store.set_trashed(items, action == "trash")
        elif action in ("folder-add", "folder-remove"):
            folder = str(data.get("folder") or "")
            if not any(known["id"] == folder for known in self.store.list_folders()):
                raise WebError("That folder is gone.", 404)
            self.store.set_in_folder(folder, items, action == "folder-add")
        else:
            raise WebError("I don't know that action.")
        return JSONResponse({"ok": True, "count": len(items)})

    async def download(self, request: Request) -> Response:
        """The originals of the chosen images, as one zip file."""
        items = self._items(await self._body(request))[:MAX_DOWNLOAD]
        folder = self.config.scratch_dir / "downloads"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"dreambot-{new_job_id().lower()}.zip"

        def build() -> int:
            added = 0
            # PNGs are already compressed: storing them is as small and far quicker.
            with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED) as archive:
                for job_id, index in items:
                    job = self.store.get_job(job_id)
                    if job is None:
                        continue
                    for image in job.images:
                        if image.index == index and Path(image.path).is_file():
                            archive.write(image.path, Path(image.path).name)
                            added += 1
            return added

        if not await asyncio.to_thread(build):
            target.unlink(missing_ok=True)
            raise WebError("None of those images are on disk any more.", 404)

        def tidy() -> None:
            target.unlink(missing_ok=True)

        return FileResponse(
            target,
            filename="dreambot-images.zip",
            media_type="application/zip",
            background=BackgroundTask(tidy),
        )

    # ------------------------------------------------------------------- folders ----

    async def folders(self, request: Request) -> Response:
        return JSONResponse({"folders": self.store.list_folders()})

    def _folder_name(self, raw: Any, keeping: str | None = None) -> str:
        name = lookbook.clean_name(str(raw or ""))
        if not name:
            raise WebError("Give the folder a name.")
        for folder in self.store.list_folders():
            if folder["name"].lower() == name.lower() and folder["id"] != keeping:
                raise WebError(f"There is already a folder called {folder['name']}.")
        return name

    async def create_folder(self, request: Request) -> Response:
        data = await self._body(request)
        folder = self.store.create_folder(
            new_job_id().lower(), self.user, self._folder_name(data.get("name"))
        )
        return JSONResponse({"folder": folder})

    async def rename_folder(self, request: Request) -> Response:
        folder = request.path_params["folder"]
        data = await self._body(request)
        if not self.store.rename_folder(folder, self._folder_name(data.get("name"), keeping=folder)):
            raise WebError("That folder is gone.", 404)
        return JSONResponse({"folders": self.store.list_folders()})

    async def delete_folder(self, request: Request) -> Response:
        """Remove a folder. The images that were in it stay in the gallery."""
        if not self.store.delete_folder(request.path_params["folder"]):
            raise WebError("That folder is gone.", 404)
        return JSONResponse({"folders": self.store.list_folders()})

    # ----------------------------------------------------------------- lookbooks ----

    async def lookbooks(self, request: Request) -> Response:
        books = await asyncio.to_thread(self.service.lookbooks, self.user)
        return JSONResponse({"lookbooks": [self._book_view(book) for book in books]})

    def _described(self, book_id: str) -> dict[str, Any]:
        book = self.store.get_lookbook(book_id)
        if book is None or book["deleted"]:
            raise WebError("I can't find that Lookbook.", 404)
        return self._book_view(self.service.describe_lookbook(book, self.user))

    async def lookbook(self, request: Request) -> Response:
        view = await asyncio.to_thread(self._described, request.path_params["book"])
        return JSONResponse({"lookbook": view})

    async def create_lookbook(self, request: Request) -> Response:
        data = await self._body(request)
        book = self.service.create_lookbook(user=self.user, name=str(data.get("name") or ""))
        return JSONResponse({"lookbook": self._book_view(book)})

    async def rename_lookbook(self, request: Request) -> Response:
        data = await self._body(request)
        book = self.service.rename_lookbook(
            user=self.user, lookbook_id=request.path_params["book"], name=str(data.get("name") or "")
        )
        return JSONResponse({"lookbook": self._book_view(book)})

    async def delete_lookbook(self, request: Request) -> Response:
        self.service.delete_lookbook(lookbook_id=request.path_params["book"])
        return JSONResponse({"ok": True})

    async def default_lookbooks(self, request: Request) -> Response:
        data = await self._body(request)
        chosen = self.service.set_default_lookbooks(
            user=self.user, lookbook_ids=[str(item) for item in data.get("lookbooks") or []]
        )
        return JSONResponse({"lookbooks": chosen})

    async def add_lookbook_image(self, request: Request) -> Response:
        data = await self._body(request)
        book_id = request.path_params["book"]
        added: dict[str, Any] | None = None
        if url := str(data.get("url") or "").strip():
            added = await self.service.add_to_lookbook(user=self.user, lookbook_id=book_id, source=url)
        for ref in data.get("pictures") or []:
            origin = f"job:{ref['job']}:{ref.get('index') or 1}" if ref.get("job") else "upload"
            added = await self.service.add_to_lookbook(
                user=self.user, lookbook_id=book_id, source=Path(self._picture(ref)), origin=origin
            )
        if added is None:
            raise WebError("Choose a picture to add.")
        return JSONResponse({"lookbook": self._book_view(added)})

    async def remove_lookbook_image(self, request: Request) -> Response:
        book = self.service.remove_from_lookbook(
            user=self.user, lookbook_id=request.path_params["book"], image_id=request.path_params["image"]
        )
        return JSONResponse({"lookbook": self._book_view(book)})

    # -------------------------------------------------------------------- styles ----

    def _liked_codes(self) -> set[str]:
        return {style["code"] for style in self.store.list_styles(self.user, liked=True)}

    async def styles(self, request: Request) -> Response:
        """Style codes to browse: a random page, or the codes that match a search."""
        query = request.query_params
        count = max(1, min(60, int(query.get("count", 24))))
        seed = query.get("seed") or "0"
        search = (query.get("q") or "").strip()
        rng = random.Random(f"explore|{seed}|{search}")
        if search:
            codes = fingerprint.search_codes(search, count, rng)
        else:
            codes = [fingerprint.random_code(rng) for _ in range(count)]
        liked = self._liked_codes()
        return JSONResponse({"styles": [self._code_card(code, liked) for code in codes]})

    async def style(self, request: Request) -> Response:
        code = request.path_params["code"]
        liked = self._liked_codes()
        if code.isdigit() and 1 <= int(code) <= fingerprint.MAX_CODE:
            similar = [self._code_card(other, liked) for other in fingerprint.similar_codes(code, 6)]
            return JSONResponse({"style": self._code_card(code, liked), "similar": similar})
        saved = self.store.get_style(self.user, code)
        if saved is None:
            raise WebError("I can't find that style.", 404)
        return JSONResponse({"style": self._saved_card(saved), "similar": []})

    async def my_styles(self, request: Request) -> Response:
        saved = [self._saved_card(style) for style in self.store.list_styles(self.user, saved=True)]
        liked = self._liked_codes()
        codes = [style["code"] for style in self.store.list_styles(self.user, liked=True, saved=False)]
        return JSONResponse(
            {"saved": saved, "liked": [self._code_card(code, liked) for code in codes if code.isdigit()]}
        )

    async def like_style(self, request: Request) -> Response:
        data = await self._body(request)
        code = str(data.get("code") or "")
        liked = bool(data.get("liked", True))
        if code.isdigit() and 1 <= int(code) <= fingerprint.MAX_CODE:
            if liked:
                self.store.save_style(self.user, code, name=fingerprint.from_code(code).name, liked=True)
            else:
                self.store.delete_style(self.user, code)
        else:
            saved = self.store.get_style(self.user, code)
            if saved is None:
                raise WebError("I can't find that style.", 404)
            self.store.save_style(self.user, saved["code"], name=saved["name"], liked=liked)
        return JSONResponse({"code": code, "liked": liked})

    async def delete_style(self, request: Request) -> Response:
        saved = self.store.get_style(self.user, request.path_params["code"])
        if saved is None or not self.store.delete_style(self.user, saved["code"]):
            raise WebError("I can't find that style.", 404)
        shutil.rmtree(self.config.home / "styles" / saved["code"], ignore_errors=True)
        return JSONResponse({"ok": True})

    def _ingredients(self, parts: Any) -> list[dict[str, Any]]:
        """The ingredients of a blend named by the browser, each resolved to a fingerprint.

        An ingredient is a style code, a saved style, a Lookbook, an upload or an image of a job.
        ``token`` is how it is written after ``--sref``; pictures have none and go in as
        attachments, which ``attach`` counts.
        """
        if not isinstance(parts, list) or not parts:
            raise WebError("Add at least one style to blend.")
        found: list[dict[str, Any]] = []
        for part in parts[:12]:
            if not isinstance(part, dict):
                raise WebError("That blend could not be read.")
            weight = max(0.0, min(100.0, float(part.get("weight", 1))))
            entry: dict[str, Any] = {"weight": weight, "pictures": [], "token": None, "ref": None}
            if code := str(part.get("code") or ""):
                if code.isdigit() and 1 <= int(code) <= fingerprint.MAX_CODE:
                    entry.update(print=fingerprint.from_code(code), token=code, label=f"style {code}")
                else:
                    saved = self.store.get_style(self.user, code)
                    if saved is None or not saved["data"]:
                        raise WebError(f"I can't find the style {code}.", 404)
                    entry.update(
                        print=StyleFingerprint.from_dict(saved["data"].get("print") or {}),
                        token=lookbook.slug(saved["name"]),
                        label=saved["name"],
                        pictures=[p for p in saved["data"].get("pictures", []) if Path(p).is_file()],
                    )
            elif book_id := str(part.get("lookbook") or ""):
                book = self.store.get_lookbook(book_id)
                if book is None or book["deleted"]:
                    raise WebError("I can't find that Lookbook.", 404)
                pictures = [
                    image["path"]
                    for image in self.store.lookbook_images(book_id)
                    if Path(image["path"]).is_file()
                ]
                if not pictures:
                    raise WebError(f"The Lookbook {book['name']} has no pictures in it yet.")
                prints = [(self.service.style_of(path)[1], 1.0) for path in pictures]
                entry.update(print=fingerprint.blend(prints), label=book["name"], pictures=pictures)
            else:
                if part.get("upload"):
                    ref = {"upload": part["upload"]}
                else:
                    ref = {"job": part.get("job"), "index": part.get("index")}
                path = self._picture(ref)
                entry.update(print=self.service.style_of(path)[1], label="picture", pictures=[path], ref=ref)
                entry["waiting"] = self.service.study_picture(path)
            found.append(entry)
        return found

    def _blend_view(self, parts: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        ingredients = self._ingredients(parts)
        blended = fingerprint.blend([(entry["print"], entry["weight"]) for entry in ingredients])
        total = sum(entry["weight"] for entry in ingredients) or 1.0
        # How to ask for the same blend in a prompt: codes and saved styles by name, pictures as attachments.
        tokens: list[str] = []
        attachments: list[Any] = []
        for entry in ingredients:
            if entry["weight"] <= 0:
                continue
            weight = "" if entry["weight"] == 1 else f"::{entry['weight']:g}"
            if entry["token"]:
                tokens.append(f"{entry['token']}{weight}")
            elif entry["ref"]:
                attachments.append(entry["ref"])
                tokens.append(f"@{len(attachments)}{weight}")
        view = {
            "style": style_card(blended),
            "phrases": fingerprint.phrases(blended, 100),
            "parts": [
                {
                    "style": style_card(entry["print"]),
                    "label": entry["label"],
                    "share": round(entry["weight"] / total, 3),
                    "waiting": bool(entry.get("waiting")),
                }
                for entry in ingredients
            ],
            "sref": " ".join(tokens),
            "attachments": attachments,
            # A Lookbook has no word after --sref: a blend that includes one is used by saving it.
            "direct": all(entry["token"] or entry["ref"] or entry["weight"] <= 0 for entry in ingredients),
        }
        return view, ingredients

    async def blend_styles(self, request: Request) -> Response:
        """What a blend of styles comes to, without rendering anything."""
        data = await self._body(request)
        view, _ = await asyncio.to_thread(self._blend_view, data.get("parts"))
        return JSONResponse(view)

    async def save_style(self, request: Request) -> Response:
        data = await self._body(request)
        view, ingredients = await asyncio.to_thread(self._blend_view, data.get("parts"))
        blended = fingerprint.blend([(entry["print"], entry["weight"]) for entry in ingredients])
        if blended is None:
            raise WebError("There is no style to save yet.")
        pictures = [path for entry in ingredients if entry["weight"] > 0 for path in entry["pictures"]][:12]
        recipe = [{"label": part["label"], "share": part["share"]} for part in view["parts"]]
        saved = self.service.save_style(
            user=self.user, name=str(data.get("name") or ""), print_=blended, pictures=pictures, recipe=recipe
        )
        kept = self.store.get_style(self.user, saved["code"])
        assert kept is not None
        return JSONResponse({"style": self._saved_card(kept)})

    # --------------------------------------------------------------------- media ----

    def _thumb(self, source: Path, width: int) -> Path:
        stamp = f"{source}|{source.stat().st_mtime_ns}|{width}".encode()
        target = self.cache / "thumbs" / f"{hashlib.sha256(stamp).hexdigest()[:24]}.jpg"
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as opened:
                image = imaging.flatten(opened)
            image.thumbnail((width, width * 3), Image.LANCZOS)
            partial = target.with_suffix(".part")
            image.save(partial, "JPEG", quality=86)
            partial.replace(target)
        return target

    async def _serve(self, request: Request, path: str | Path, *, name: str | None = None) -> Response:
        source = Path(path)
        if not source.is_file():
            raise WebError("That image's file is no longer on disk.", 404)
        headers = {"Cache-Control": "private, max-age=86400"}
        width = request.query_params.get("w")
        if width and width.isdigit() and int(width) in THUMB_WIDTHS:
            thumb = await asyncio.to_thread(self._thumb, source, int(width))
            return FileResponse(thumb, media_type="image/jpeg", headers=headers)
        if request.query_params.get("download"):
            return FileResponse(source, filename=name or source.name, headers=headers)
        return FileResponse(source, headers=headers)

    async def job_image(self, request: Request) -> Response:
        job = self._job(request.path_params["job"])
        try:
            image = job.image(request.path_params["index"])
        except KeyError as exc:
            raise WebError("That job has no such image.", 404) from exc
        return await self._serve(request, image.path)

    async def job_grid(self, request: Request) -> Response:
        job = self._job(request.path_params["job"])
        if not job.grid:
            raise WebError("That job has no grid.", 404)
        return await self._serve(request, job.grid)

    async def job_preview(self, request: Request) -> Response:
        job = self._job(request.path_params["job"])
        target = self.cache / "previews" / f"{job.id}.png"
        if not target.is_file():
            raise WebError("There is no preview yet.", 404)
        return FileResponse(target, headers={"Cache-Control": "no-store"})

    async def upload_image(self, request: Request) -> Response:
        upload = self.store.get_upload(request.path_params["upload"])
        if upload is None:
            raise WebError("That upload is gone.", 404)
        return await self._serve(request, upload["path"], name=upload["name"] or None)

    async def lookbook_image(self, request: Request) -> Response:
        wanted = request.path_params["image"]
        for image in self.store.lookbook_images(request.path_params["book"]):
            if image["id"] == wanted:
                return await self._serve(request, image["path"])
        raise WebError("That picture isn't in this Lookbook.", 404)

    async def style_image(self, request: Request) -> Response:
        saved = self.store.get_style(self.user, request.path_params["code"])
        pictures = (saved or {}).get("data", {}).get("pictures", []) if saved and saved["data"] else []
        number = request.path_params["number"]
        if not 1 <= number <= len(pictures):
            raise WebError("That style has no such picture.", 404)
        return await self._serve(request, pictures[number - 1])


# --------------------------------------------------------------------------------------
# Running it
# --------------------------------------------------------------------------------------


class _Server(uvicorn.Server):
    """uvicorn without its signal handlers: stopping Dreambot is the command line's job."""

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


class WebFrontend:
    """The web app as something ``dreambot run`` starts and stops next to the chat bots."""

    name = "web"

    def __init__(self, service: Dreambot, *, host: str | None = None, port: int | None = None):
        self.service = service
        self.host = host or service.config.web.host
        self.port = port or service.config.web.port
        self.web: WebApp | None = None
        self._server: _Server | None = None
        self._task: asyncio.Task[None] | None = None

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{host}:{self.port}/"

    async def start(self) -> None:
        # Bound here rather than by uvicorn, which answers a port in use by exiting the program.
        sock = socket.socket(socket.AF_INET6 if ":" in self.host else socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((self.host, self.port))
            self.port = sock.getsockname()[1]  # the port chosen, when asked for "any" (0)
        except OSError as exc:
            sock.close()
            raise RuntimeError(
                f"port {self.port} is in use ({exc.strerror or exc}). Stop what is using it, or choose "
                f"another with `[web] port` in the config file or `dreambot web --port`."
            ) from exc
        if self.host not in ("127.0.0.1", "localhost", "::1"):
            log.warning(
                "The web app is listening on %s. It has no accounts: anyone who can reach that address "
                "can use this Dreambot.",
                self.host,
            )
        hosts = frozenset(
            {
                f"127.0.0.1:{self.port}",
                f"localhost:{self.port}",
                f"[::1]:{self.port}",
                f"{self.host}:{self.port}",
            }
        )
        self.web = WebApp(self.service, hosts=hosts)
        config = uvicorn.Config(
            self.web.app, log_level="warning", access_log=False, lifespan="off", timeout_graceful_shutdown=2
        )
        self._server = _Server(config)
        self._task = asyncio.create_task(self._server.serve(sockets=[sock]), name="dreambot-web")
        for _ in range(200):
            if self._server.started or self._task.done():
                break
            await asyncio.sleep(0.01)
        if self._task.done():
            self._task.result()
            raise RuntimeError("the web server stopped as soon as it started")

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(self._task, timeout=5)
        if self.web is not None:
            self.web.close()
        self._server = self._task = None


def clean_cache(home: Path, *, days: float = 30) -> None:
    """Drop thumbnails nobody has looked at for a while. They are rebuilt when needed."""
    limit = time.time() - days * 86400
    for path in (home / "cache" / "thumbs").glob("*.jpg"):
        with contextlib.suppress(OSError):
            if path.stat().st_atime < limit:
                path.unlink()
