"""The Dreambot service: everything a chat frontend needs, behind one object.

Frontends (Discord, Telegram, iMessage, the console) are thin. They translate platform events
into calls on :class:`Dreambot` — ``imagine``, ``act``, ``describe`` — and render the
:class:`~dreambot.core.jobs.JobEvent` stream it emits. All behaviour that makes Dreambot feel
like Midjourney (prompt handling, variations, upscales, zooms, the queue) lives here, once.

There is one GPU, so jobs run strictly one at a time from a FIFO queue.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import shutil
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dreambot.config import Config
from dreambot.core import imaging, styles
from dreambot.core.access import Access
from dreambot.core.actions import GENERATIVE, PAN_DIRECTIONS, Action, ActionKind
from dreambot.core.fetch import FetchError, fetch_image
from dreambot.core.imaging import ImageError
from dreambot.core.jobs import EventType, Job, JobEvent, JobImage, JobKind, JobStatus
from dreambot.core.params import Params, PromptError, Ref
from dreambot.core.planner import Catalog, Plan, PlanError, Planner, Render, ResolvedRefs
from dreambot.core.prompt import MAX_PERMUTATIONS, ParsedPrompt, expand_permutations, parse_prompt
from dreambot.core.settings import ResolvedSettings, UserSettings
from dreambot.core.sizes import working_size
from dreambot.core.store import Store
from dreambot.core.text import SHORTEN_INSTRUCTION, heuristic_shorten, parse_numbered
from dreambot.engine.client import Engine, EngineCancelled, EngineError, new_request_id
from dreambot.engine.protocol import (
    DescribeRequest,
    ErrorKind,
    Event,
    Loading,
    Preview,
    Progress,
    TextRequest,
)

log = logging.getLogger(__name__)

Listener = Callable[[JobEvent], Awaitable[None] | None]
#: Seconds between progress events for one job (frontends edit a chat message for each).
PROGRESS_INTERVAL = 1.0
MAX_JOBS_PER_MESSAGE = MAX_PERMUTATIONS


class DreambotError(Exception):
    """Something the person should be told, in words meant for them."""


@dataclass
class ActionResult:
    """The outcome of an action: either newly queued jobs, or an immediate answer."""

    jobs: list[Job]
    #: For SELECT / FAVORITE / INFO: the job and image the action was about.
    job: Job | None = None
    image: JobImage | None = None
    message: str | None = None


@dataclass
class _Queued:
    job: Job
    run: Callable[[Job], Awaitable[None]]


def _slug(text: str, limit: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:limit].rstrip("-") or "image"


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


class Dreambot:
    def __init__(
        self,
        config: Config,
        *,
        engine: Engine | None = None,
        store: Store | None = None,
        catalog: Catalog | None = None,
    ):
        self.config = config
        fake = config.engine.backend == "fake"
        self.engine = engine or Engine(
            backend=config.engine.backend,
            idle_timeout=config.engine.idle_unload_seconds or None,
            env=(
                {"DREAMBOT_MEMORY_BUDGET_GB": str(config.engine.memory_budget_gb)}
                if config.engine.memory_budget_gb
                else None
            ),
        )
        self.store = store or Store(config.database_path)
        self.access = Access(config, self.store)
        self.catalog = catalog or Catalog(config, assume_installed=fake)
        self.planner = Planner(self.catalog, config)
        self._listeners: list[Listener] = []
        self._queue: list[_Queued] = []
        self._wake = asyncio.Event()
        self._worker: asyncio.Task[None] | None = None
        self._current: Job | None = None
        self._current_request: str | None = None
        self._closing = False

    # ----------------------------------------------------------------- lifecycle ----

    async def start(self) -> None:
        for job in self.store.interrupted_jobs():
            job.status = JobStatus.FAILED
            job.error = "Dreambot was restarted before this job finished."
            job.finished = time.time()
            self.store.save_job(job)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._closing = False
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._work(), name="dreambot-queue")

    async def close(self) -> None:
        self._closing = True
        self._wake.set()
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._worker
            self._worker = None
        await self.engine.close()
        self.store.close()

    async def __aenter__(self) -> Dreambot:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    # -------------------------------------------------------------------- events ----

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        """Receive every job event. Returns a function that unsubscribes."""
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener) if listener in self._listeners else None

    async def _emit(self, event: JobEvent) -> None:
        for listener in list(self._listeners):
            try:
                outcome = listener(event)
                if asyncio.iscoroutine(outcome):
                    await outcome
            except Exception:
                log.exception("a job listener failed on %s for job %s", event.type, event.job.id)

    # ------------------------------------------------------------------ settings ----

    def settings(self, user: str) -> UserSettings:
        return UserSettings.from_dict(self.store.get_settings(user))

    def resolved_settings(self, user: str) -> ResolvedSettings:
        return self.settings(user).resolved(self.config.defaults)

    def save_settings(self, user: str, settings: UserSettings) -> None:
        if settings.model is not None:
            try:
                settings.model = self.catalog.lookup(settings.model).id
            except PlanError as exc:
                raise DreambotError(str(exc)) from exc
        self.store.save_settings(user, settings.to_dict())

    # --------------------------------------------------------------------- queue ----

    @property
    def queue_length(self) -> int:
        return len(self._queue) + (1 if self._current is not None else 0)

    def queued_for(self, user: str) -> int:
        return sum(1 for item in self._queue if item.job.user == user) + (
            1 if self._current is not None and self._current.user == user else 0
        )

    def _check_room(self, user: str, adding: int) -> None:
        limit = self.config.access.max_queued_per_user
        waiting = self.queued_for(user)
        if waiting + adding > limit:
            raise DreambotError(
                f"You have {waiting} job(s) in progress and the limit is {limit}. "
                "Wait for some to finish (or `cancel` them) and try again."
            )

    async def _enqueue(self, job: Job, run: Callable[[Job], Awaitable[None]]) -> None:
        self.store.save_job(job)
        self._queue.append(_Queued(job, run))
        position = len(self._queue) - 1 + (1 if self._current is not None else 0)
        await self._emit(JobEvent(EventType.QUEUED, job, position=position))
        self._wake.set()

    async def _work(self) -> None:
        while not self._closing:
            if not self._queue:
                self._wake.clear()
                await self._wake.wait()
                continue
            item = self._queue.pop(0)
            job = item.job
            self._current = job
            job.status = JobStatus.RUNNING
            job.started = time.time()
            self.store.save_job(job)
            await self._emit(JobEvent(EventType.STARTED, job))
            try:
                await item.run(job)
            except EngineCancelled:
                job.status = JobStatus.CANCELLED
            except asyncio.CancelledError:
                job.status = JobStatus.CANCELLED
                job.finished = time.time()
                self.store.save_job(job)
                raise
            except EngineError as exc:
                job.status = JobStatus.FAILED
                job.error = _explain(exc)
                if exc.kind in (ErrorKind.INTERNAL, ErrorKind.CRASHED):
                    log.error("job %s failed: %s\n%s", job.id, exc, exc.detail or "")
            except (DreambotError, PlanError, ImageError) as exc:
                job.status = JobStatus.FAILED
                job.error = str(exc)
            except Exception as exc:
                log.exception("job %s crashed", job.id)
                job.status = JobStatus.FAILED
                job.error = f"Something went wrong inside Dreambot ({type(exc).__name__})."
            else:
                job.status = JobStatus.DONE
            finally:
                self._current = None
                self._current_request = None
            job.finished = time.time()
            self.store.save_job(job)
            kind = {
                JobStatus.DONE: EventType.FINISHED,
                JobStatus.CANCELLED: EventType.CANCELLED,
            }.get(job.status, EventType.FAILED)
            await self._emit(JobEvent(kind, job, fraction=1.0 if job.status is JobStatus.DONE else 0.0))

    async def cancel(
        self, job_id: str | None = None, *, user: str | None = None, channel: str | None = None
    ) -> list[Job]:
        """Cancel one job by id, or — without an id — everything ``user`` has running in ``channel``."""
        cancelled: list[Job] = []
        for item in list(self._queue):
            job = item.job
            if (job_id and job.id == job_id) or (
                not job_id
                and (user is None or job.user == user)
                and (channel is None or job.channel == channel)
            ):
                self._queue.remove(item)
                job.status = JobStatus.CANCELLED
                job.finished = time.time()
                self.store.save_job(job)
                cancelled.append(job)
                await self._emit(JobEvent(EventType.CANCELLED, job))
        current = self._current
        if current is not None and (
            (job_id and current.id == job_id)
            or (
                not job_id
                and (user is None or current.user == user)
                and (channel is None or current.channel == channel)
            )
        ):
            if self._current_request is not None:
                await self.engine.cancel(self._current_request)
            cancelled.append(current)
        return cancelled

    # ------------------------------------------------------------------- imagine ----

    async def imagine(
        self,
        *,
        user: str,
        channel: str,
        prompt: str,
        attachments: Sequence[str | Path] = (),
        allow_paths: bool = False,
    ) -> list[Job]:
        """Queue one job per prompt (permutations and ``--repeat`` make several)."""
        settings = self.resolved_settings(user)
        full_prompt = f"{prompt} {settings.suffix}".strip() if settings.suffix else prompt
        try:
            # An empty message still goes through planning, which explains what is missing.
            expanded = expand_permutations(full_prompt) or [""]
            parsed_prompts = [parse_prompt(text, option_sets=settings.option_sets) for text in expanded]
        except PromptError as exc:
            raise DreambotError(str(exc)) from exc
        total = sum(parsed.params.repeat or 1 for parsed in parsed_prompts)
        if total > MAX_JOBS_PER_MESSAGE:
            raise DreambotError(
                f"That would start {total} jobs; the most one message can start is {MAX_JOBS_PER_MESSAGE}."
            )
        self._check_room(user, total)

        scratch = self._scratch()
        try:
            prepared = [str(imaging.prepare_input(path, scratch)) for path in attachments]
        except ImageError as exc:
            raise DreambotError(str(exc)) from exc

        planned: list[tuple[Job, Plan]] = []
        for parsed in parsed_prompts:
            refs, ref_notes = await self._resolve_refs(parsed, prepared, scratch, allow_paths=allow_paths)
            for _ in range(parsed.params.repeat or 1):
                try:
                    plan = self.planner.imagine(parsed, settings, refs)
                except (PlanError, PromptError) as exc:
                    raise DreambotError(str(exc)) from exc
                plan.notes[:0] = ref_notes
                if refs.edit:
                    plan.detail["instruction"] = True  # `--edit`: the text says what to change
                job = self._job_from_plan(
                    JobKind.IMAGINE,
                    plan,
                    user=user,
                    channel=channel,
                    prompt=self._display_prompt(parsed, refs),
                    params=parsed.params,
                )
                job.inputs = {"attachments": prepared}
                planned.append((job, plan))
        for job, plan in planned:
            await self._enqueue(job, self._runner(plan))
        return [job for job, _ in planned]

    def _display_prompt(self, parsed: ParsedPrompt, refs: ResolvedRefs) -> str:
        """The prompt as it should be shown and re-run: `--sref random` pinned to the code drawn."""
        text = parsed.canonical()
        if refs.style_codes and "random" in text:
            for code in refs.style_codes:
                text = re.sub(r"(--sref[^-]*?)\brandom\b", rf"\g<1>{code}", text, count=1)
        return text

    async def _resolve_refs(
        self, parsed: ParsedPrompt, attachments: list[str], scratch: Path, *, allow_paths: bool
    ) -> tuple[ResolvedRefs, list[str]]:
        """Download and normalise every image a prompt refers to."""
        params = parsed.params
        notes: list[str] = []
        claimed: set[int] = set()

        async def one(ref: Ref, role: str) -> list[str]:
            if ref.kind == "url":
                try:
                    downloaded = await asyncio.to_thread(fetch_image, ref.value, scratch)
                    return [str(imaging.prepare_input(downloaded, scratch))]
                except (FetchError, ImageError) as exc:
                    raise DreambotError(str(exc)) from exc
            if ref.kind == "attachment":
                if ref.value == "*":
                    free = [i for i in range(len(attachments)) if i not in claimed]
                    if not free:
                        raise DreambotError(f"`--{role}` needs an image: attach one, or give a link.")
                    claimed.update(free)
                    return [attachments[i] for i in free]
                position = int(ref.value) - 1
                if position >= len(attachments):
                    raise DreambotError(
                        f"`--{role} @{ref.value}`: this message has {len(attachments)} attached image(s)."
                    )
                claimed.add(position)
                return [attachments[position]]
            if ref.kind == "path":
                if not allow_paths:
                    raise DreambotError(
                        "File paths only work from the Mac itself; attach the image or use a link."
                    )
                try:
                    return [str(imaging.prepare_input(ref.value, scratch))]
                except ImageError as exc:
                    raise DreambotError(str(exc)) from exc
            return []

        async def many(refs: list[Ref], role: str) -> list[str]:
            # Explicit references first, so that a bare `--edit` takes what is left over.
            ordered = sorted(refs, key=lambda ref: ref.kind == "attachment" and ref.value == "*")
            paths: list[str] = []
            for ref in ordered:
                paths.extend(await one(ref, role))
            return paths

        resolved = ResolvedRefs()
        resolved.edit = await many(params.edit, "edit")
        subject_refs = [*params.oref, *params.cref]
        resolved.subject = await many(subject_refs, "oref" if params.oref else "cref")
        resolved.subject_kind = "character" if params.cref and not params.oref else "subject"
        style_images = [ref for ref in params.sref if ref.kind not in ("code", "random")]
        resolved.style = await many(style_images, "sref")
        for ref in params.sref:
            if ref.kind == "code":
                resolved.style_codes.append(ref.value)
            elif ref.kind == "random":
                code = styles.random_style_code()
                resolved.style_codes.append(code)
                notes.append(f"`--sref random` picked style `{code}`. Reuse it with `--sref {code}`.")
        resolved.image_prompts = await many(parsed.image_prompts, "image prompt")
        # Attached images nobody claimed are image prompts, as in Midjourney.
        resolved.image_prompts.extend(path for i, path in enumerate(attachments) if i not in claimed)
        return resolved, notes

    # ------------------------------------------------------------------- actions ----

    def find_job(self, job_id: str) -> Job:
        job = self.store.get_job(job_id)
        if job is None:
            raise DreambotError(f"I can't find a job with the id `{job_id}`.")
        return job

    def _target(self, action: Action, channel: str, *, need_image: bool) -> tuple[Job, JobImage | None]:
        if action.job:
            job = self.find_job(action.job)
        else:
            found = self.store.last_job(channel=channel)
            if found is None:
                raise DreambotError("There's nothing here to apply that to yet. Send a prompt first.")
            job = found
        if not need_image:
            return job, None
        if not job.images:
            raise DreambotError(f"Job `{job.id}` has no images.")
        index = action.index
        if index is None:
            if len(job.images) > 1:
                raise DreambotError(f"Which image? Add its number (1-{len(job.images)}), for example `U2`.")
            index = job.images[0].index
        try:
            image = job.image(index)
        except KeyError:
            raise DreambotError(
                f"Job `{job.id}` has {len(job.images)} image(s); there is no number {index}."
            ) from None
        if not Path(image.path).is_file():
            raise DreambotError("That image's file is no longer on disk.")
        return job, image

    async def act(self, *, user: str, channel: str, action: Action) -> ActionResult:
        """Carry out an action on an earlier job."""
        kind = action.kind
        if kind is ActionKind.CANCEL:
            cancelled = await self.cancel(
                action.job, user=None if action.job else user, channel=None if action.job else channel
            )
            message = f"Cancelled {len(cancelled)} job(s)." if cancelled else "Nothing is running."
            return ActionResult(jobs=[], message=message)
        if kind is ActionKind.ANIMATE:
            raise DreambotError("Turning images into video isn't available yet.")
        if kind is ActionKind.REROLL:
            job, _ = self._target(action, channel, need_image=False)
            return ActionResult(jobs=await self._reroll(job, user=user, channel=channel))

        job, image = self._target(action, channel, need_image=True)
        assert image is not None
        if kind is ActionKind.SELECT:
            return ActionResult(jobs=[], job=job, image=image)
        if kind is ActionKind.FAVORITE:
            image.favorite = not image.favorite
            self.store.save_job(job)
            return ActionResult(
                jobs=[],
                job=job,
                image=image,
                message="Added to favorites." if image.favorite else "Removed from favorites.",
            )
        if kind is ActionKind.INFO:
            return ActionResult(jobs=[], job=job, image=image, message=describe_job(job, image))
        assert kind in GENERATIVE
        self._check_room(user, 1)
        try:
            new_job, plan = self._plan_action(action, job, image, user=user, channel=channel)
        except (PlanError, PromptError, ImageError, ValueError) as exc:
            raise DreambotError(str(exc)) from exc
        await self._enqueue(new_job, self._runner(plan))
        return ActionResult(jobs=[new_job])

    def _plan_action(
        self, action: Action, job: Job, image: JobImage, *, user: str, channel: str
    ) -> tuple[Job, Plan]:
        settings = self.resolved_settings(user)
        kind = action.kind
        scratch = self._scratch()
        text, negative, params = self._scene_text(job), job.negative, Params()
        # A derived job gets fresh seeds, so the parent's --seed must not appear in its prompt.
        prompt = re.sub(r"\s--(?:seed|sameseed)\s+\d+", "", f" {job.prompt}").strip()
        if action.prompt and kind in (
            ActionKind.VARY,
            ActionKind.VARY_SUBTLE,
            ActionKind.VARY_STRONG,
            ActionKind.ZOOM,
            ActionKind.PAN,
        ):
            # Remix: the variation follows a new prompt.
            parsed = parse_prompt(action.prompt, option_sets=settings.option_sets)
            text = parsed.positive_text or job.text
            negative = ", ".join(parsed.negative_terms) or None
            params, prompt = parsed.params, parsed.canonical()

        count = params.count or settings.count
        detail: dict[str, Any] = {
            "action": {"kind": kind.value, "value": action.value, "prompt": action.prompt}
        }

        if kind in (ActionKind.VARY, ActionKind.VARY_SUBTLE, ActionKind.VARY_STRONG):
            mode = (
                settings.variation
                if kind is ActionKind.VARY
                else ("subtle" if kind is ActionKind.VARY_SUBTLE else "strong")
            )
            spec = self.catalog.registry.find(job.model)
            native, multiple = (spec.native, spec.multiple) if spec else (1024, 16)
            width, height = working_size(image.width, image.height, native=native, multiple=multiple)
            plan = self.planner.variation(
                source=image.path,
                text=text,
                model=job.model,
                width=width,
                height=height,
                mode=mode,
                count=count,
                negative=negative,
                params=params,
            )
            new_kind = JobKind.VARIATION
        elif kind in (ActionKind.UPSCALE_SUBTLE, ActionKind.UPSCALE_CREATIVE):
            creative = kind is ActionKind.UPSCALE_CREATIVE
            factor = min(2.0, 3072 / max(image.width, image.height))
            work = scratch / f"upscale-{new_request_id()[:10]}"
            enlarged = imaging.enlarge(image.path, factor, work / "enlarged.png")
            tiles = imaging.cut_tiles(enlarged, work / "tiles")
            plan = self.planner.upscale(
                source=image.path,
                text=text,
                model=job.model,
                creative=creative,
                enlarged=str(enlarged),
                tiles=tiles,
            )
            new_kind = JobKind.UPSCALE
        elif kind is ActionKind.ZOOM:
            factor = float(action.value or 2)
            canvas = imaging.zoom_canvas(image.path, factor, scratch)
            plan = self.planner.outpaint(
                source=image.path, canvas=canvas, text=text, count=count, params=params
            )
            detail["factor"] = factor
            new_kind = JobKind.ZOOM
        elif kind is ActionKind.PAN:
            direction = action.value or ""
            if direction not in PAN_DIRECTIONS:
                raise DreambotError("Pan needs a direction: left, right, up or down.")
            canvas = imaging.pan_canvas(image.path, direction, scratch)
            plan = self.planner.outpaint(
                source=image.path, canvas=canvas, text=text, count=count, params=params
            )
            detail["direction"] = direction
            new_kind = JobKind.PAN
        elif kind is ActionKind.SQUARE:
            if image.width == image.height:
                raise DreambotError("That image is already square.")
            canvas = imaging.square_canvas(image.path, scratch)
            plan = self.planner.outpaint(
                source=image.path, canvas=canvas, text=text, count=count, params=params
            )
            detail["square"] = True
            new_kind = JobKind.ZOOM
        elif kind is ActionKind.VARY_REGION:
            if not action.prompt:
                raise DreambotError("Say what to put there, e.g. `region 2 B2-D4: a red balloon`.")
            where = (action.value or "").strip()
            mask: str | None = None
            try:
                cells = imaging.parse_cells(where)
            except ValueError:
                cells = None
            if cells is not None:
                mask = str(
                    imaging.cells_mask(
                        (image.width, image.height), cells, scratch / f"mask-{new_request_id()[:10]}.png"
                    )
                )
                where = ""
            elif not where:
                raise DreambotError(
                    "Say which part to change: cells like `B2-D4`, or a description like `the sky`."
                )
            plan = self.planner.region(
                source=image.path, mask=mask, text=action.prompt, where=where or None, count=count
            )
            prompt = action.prompt
            detail["instruction"] = True
            new_kind = JobKind.EDIT
        elif kind is ActionKind.EDIT:
            if not action.prompt:
                raise DreambotError("Say what to change, e.g. `edit 2: make it night`.")
            plan = self.planner.edit(sources=[image.path], instruction=action.prompt, count=count)
            prompt = action.prompt
            detail["instruction"] = True
            new_kind = JobKind.EDIT
        else:  # pragma: no cover - GENERATIVE is exhaustive above
            raise DreambotError("I don't know how to do that yet.")

        plan.detail.update(detail)
        new_job = self._job_from_plan(
            new_kind, plan, user=user, channel=channel, prompt=prompt, params=params
        )
        new_job.parent, new_job.parent_index = job.id, image.index
        new_job.inputs = {"source": image.path}
        return new_job, plan

    def _scene_text(self, job: Job, depth: int = 0) -> str:
        """A description of what a job's images show, for jobs derived from them.

        Usually that is the job's own text. But when the text was an instruction ("make it
        night"), it describes a change, not a scene: handing it to a zoom-out or a variation
        would be asking for the wrong thing. Then the nearest ancestor's description is used.
        """
        if not job.detail.get("instruction"):
            return job.text
        if job.parent and depth < 8:
            parent = self.store.get_job(job.parent)
            if parent is not None:
                return self._scene_text(parent, depth + 1)
        return ""

    async def _reroll(self, job: Job, *, user: str, channel: str) -> list[Job]:
        """Run a job again with fresh seeds."""
        if job.kind is JobKind.IMAGINE:
            prompt = re.sub(r"\s--(?:seed|sameseed)\s+\d+", "", f" {job.prompt}").strip()
            attachments = [p for p in job.inputs.get("attachments", []) if Path(p).is_file()]
            return await self.imagine(
                user=user, channel=channel, prompt=prompt, attachments=attachments, allow_paths=True
            )
        saved = job.detail.get("action")
        if not saved or job.parent is None:
            raise DreambotError("That job can't be rerolled.")
        again = Action(
            ActionKind(saved["kind"]), job.parent, job.parent_index, saved.get("value"), saved.get("prompt")
        )
        return (await self.act(user=user, channel=channel, action=again)).jobs

    # --------------------------------------------------------------------- blend ----

    async def blend(
        self, *, user: str, channel: str, images: Sequence[str | Path], aspect: tuple[int, int] | None = None
    ) -> Job:
        self._check_room(user, 1)
        scratch = self._scratch()
        try:
            prepared = [str(imaging.prepare_input(path, scratch)) for path in images]
            plan = self.planner.blend(
                sources=prepared, aspect=aspect, count=self.resolved_settings(user).count
            )
        except (PlanError, ImageError) as exc:
            raise DreambotError(str(exc)) from exc
        job = self._job_from_plan(
            JobKind.BLEND, plan, user=user, channel=channel, prompt="/blend", params=Params()
        )
        job.inputs = {"attachments": prepared}
        await self._enqueue(job, self._runner(plan))
        return job

    # ------------------------------------------------------------------ describe ----

    async def describe(self, *, user: str, channel: str, image: str | Path) -> Job:
        """Queue a job that writes four prompts for an image."""
        self._check_room(user, 1)
        vision = self.catalog.vision_model()
        if vision is None:
            raise DreambotError(
                "`/describe` needs a vision model, and none is installed. On the Mac, run: dreambot models pull describe"
            )
        try:
            prepared = str(imaging.prepare_input(image, self._scratch(), max_side=1536))
        except ImageError as exc:
            raise DreambotError(str(exc)) from exc
        job = Job(
            id=self.store.reserve_job_id(),
            kind=JobKind.DESCRIBE,
            user=user,
            channel=channel,
            prompt="/describe",
            inputs={"source": prepared},
        )

        async def run(running: Job) -> None:
            request = DescribeRequest(id=new_request_id(), image=prepared, count=4, weights=vision)
            self._current_request = request.id
            running.texts = await self.engine.describe(request)

        await self._enqueue(job, run)
        return job

    async def shorten(self, *, user: str, channel: str, prompt: str) -> Job:
        """Queue a job that proposes shorter versions of a prompt (Midjourney's /shorten)."""
        self._check_room(user, 1)
        settings = self.resolved_settings(user)
        try:
            parsed = parse_prompt(prompt, option_sets=settings.option_sets)
        except PromptError as exc:
            raise DreambotError(str(exc)) from exc
        if not parsed.text:
            raise DreambotError("Give me a prompt to shorten, e.g. `/shorten a very long prompt ...`.")
        suffix = " ".join(parsed.given)
        vision = self.catalog.vision_model()
        job = Job(
            id=self.store.reserve_job_id(), kind=JobKind.SHORTEN, user=user, channel=channel, prompt=prompt
        )

        async def run(running: Job) -> None:
            versions: list[str] = []
            if vision is not None:
                request = TextRequest(
                    id=new_request_id(),
                    weights=vision,
                    prompt=SHORTEN_INSTRUCTION.format(count=5, prompt=parsed.text),
                    max_tokens=600,
                    temperature=0.3,
                )
                self._current_request = request.id
                answers = await self.engine.text(request)
                versions = [
                    v
                    for v in parse_numbered(answers[0] if answers else "", 5)
                    if len(v) < len(parsed.text) + 20
                ]
            if len(versions) < 2:
                # No language model (or an unusable answer): trim mechanically instead.
                versions = heuristic_shorten(parsed.text, 5)
                running.notes.append(
                    "Shortened without a language model; `dreambot models pull describe` adds one."
                )
            running.texts = [f"{version} {suffix}".strip() for version in versions]

        await self._enqueue(job, run)
        return job

    # ----------------------------------------------------------------- execution ----

    def _scratch(self) -> Path:
        path = self.config.scratch_dir / datetime.now().strftime("%Y-%m-%d")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _job_from_plan(
        self, kind: JobKind, plan: Plan, *, user: str, channel: str, prompt: str, params: Params
    ) -> Job:
        return Job(
            id=self.store.reserve_job_id(),
            kind=kind,
            user=user,
            channel=channel,
            prompt=prompt,
            text=plan.text,
            negative=plan.negative,
            model=plan.model,
            width=plan.width,
            height=plan.height,
            seeds=list(plan.seeds),
            params=_jsonable(params.given()),
            notes=list(plan.notes),
            detail=_jsonable(plan.detail),
        )

    def _runner(self, plan: Plan) -> Callable[[Job], Awaitable[None]]:
        async def run(job: Job) -> None:
            await self._run_plan(job, plan)

        return run

    async def _run_plan(self, job: Job, plan: Plan) -> None:
        day = datetime.fromtimestamp(job.created).strftime("%Y-%m-%d")
        out_dir = self.config.output_dir / day
        out_dir.mkdir(parents=True, exist_ok=True)
        work_dir = self.config.scratch_dir / "jobs" / job.id
        stem = f"{_slug(job.text or job.kind.value)}_{job.id}"
        units = plan.work + (plan.count if plan.finish else 0)
        done = 0
        last_progress = 0.0

        async def on_event(event: Event, base: int) -> None:
            nonlocal last_progress
            if isinstance(event, Loading):
                await self._emit(JobEvent(EventType.LOADING, job, fraction=done / units))
            elif isinstance(event, Progress):
                now = time.monotonic()
                if now - last_progress < PROGRESS_INTERVAL:
                    return
                last_progress = now
                fraction = (base + event.index + event.step / max(1, event.total)) / units
                await self._emit(JobEvent(EventType.PROGRESS, job, fraction=min(0.99, fraction)))
            elif isinstance(event, Preview) and plan.tiles is None:
                await self._emit(JobEvent(EventType.PREVIEW, job, fraction=done / units, preview=event.path))

        try:
            pieces: dict[int, str] = {}
            for number, render in enumerate(plan.renders):
                request = render.request(
                    request_id=new_request_id(),
                    out_dir=str(work_dir),
                    basename=f"r{number}",
                    preview_every=self.config.engine.preview_every,
                )
                self._current_request = request.id
                base = done

                async def relay(event: Event, base: int = base) -> None:
                    await on_event(event, base)

                result = await self.engine.generate(request, on_event=relay)
                for produced in result.images:
                    slot = render.slots[produced.index]
                    done += 1
                    if plan.tiles is not None:
                        pieces[slot] = produced.path  # a tile, assembled below
                        continue
                    final = out_dir / f"{stem}_{slot}.png"
                    shutil.move(produced.path, final)
                    if plan.canvas is not None and plan.source:
                        await asyncio.to_thread(imaging.restore_source, final, plan.canvas, plan.source)
                    elif plan.keep_outside_mask and plan.source:
                        await asyncio.to_thread(
                            imaging.restore_outside_mask, final, plan.source, plan.keep_outside_mask
                        )
                    job.images.append(
                        JobImage(
                            index=slot,
                            path=str(final),
                            seed=produced.seed,
                            width=produced.width,
                            height=produced.height,
                        )
                    )

            if plan.tiles is not None and plan.tile_base:
                final = out_dir / f"{stem}_1.png"
                placed = [(tile.box, pieces[number]) for number, tile in enumerate(plan.tiles, start=1)]
                await asyncio.to_thread(imaging.blend_tiles, plan.tile_base, placed, final)
                job.images.append(
                    JobImage(
                        index=1, path=str(final), seed=plan.seeds[0], width=plan.width, height=plan.height
                    )
                )
            job.images.sort(key=lambda image: image.index)

            if plan.finish is not None:
                await self._finish_pass(job, plan.finish, work_dir, lambda: done, units, on_event)

            if len(job.images) > 1:
                grid = out_dir / f"{stem}_grid.jpg"
                await asyncio.to_thread(imaging.compose_grid, [image.path for image in job.images], grid)
                job.grid = str(grid)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            if plan.tile_base:
                shutil.rmtree(Path(plan.tile_base).parent, ignore_errors=True)

    async def _finish_pass(
        self,
        job: Job,
        template: Render,
        work_dir: Path,
        done: Callable[[], int],
        units: int,
        on_event: Callable[[Event, int], Awaitable[None]],
    ) -> None:
        """Run the per-image finishing step (the ``--hd`` upscale) over every image of a job."""
        for position, image in enumerate(job.images):
            render = Render(
                **{**template.__dict__, "image": image.path, "seeds": [image.seed], "slots": [image.index]}
            )
            request = render.request(
                request_id=new_request_id(), out_dir=str(work_dir), basename=f"hd{image.index}"
            )
            self._current_request = request.id
            base = len(job.images) + position

            async def relay(event: Event, base: int = base) -> None:
                await on_event(event, base)

            result = await self.engine.generate(request, on_event=relay)
            upscaled = result.images[0]
            shutil.move(upscaled.path, image.path)
            image.width, image.height = upscaled.width, upscaled.height
        if job.images:
            job.width, job.height = job.images[0].width, job.images[0].height

    # ---------------------------------------------------------------------- info ----

    def info(self, user: str) -> dict[str, Any]:
        """What Midjourney's /info shows, for a local setup."""
        counts = self.store.count_jobs(user=user)
        settings = self.resolved_settings(user)
        try:
            model = self.planner.default_model(settings).id
        except PlanError:
            model = None
        return {
            "user": user,
            "jobs_done": counts.get(JobStatus.DONE.value, 0),
            "jobs_failed": counts.get(JobStatus.FAILED.value, 0),
            "queued": len(self._queue),
            "running": self._current.id if self._current else None,
            "your_jobs_in_progress": self.queued_for(user),
            "model": model,
            "installed_models": [spec.id for spec in self.catalog.installed()],
            "backend": self.config.engine.backend,
        }


def _explain(error: EngineError) -> str:
    if error.kind == ErrorKind.OUT_OF_MEMORY:
        return "The Mac ran out of memory for that. Try a smaller size, or a smaller or more quantized model."
    return str(error)


def describe_job(job: Job, image: JobImage | None = None) -> str:
    """Seed, id and settings of a job (Midjourney's envelope reaction)."""
    lines = [f"Job {job.id} · {job.kind.value} · {job.model or 'n/a'}"]
    if job.prompt:
        lines.append(f"Prompt: {job.prompt}")
    if image is not None:
        lines.append(f"Image {image.index}: seed {image.seed} · {image.width}x{image.height}")
        lines.append(f"File: {image.path}")
    elif job.seeds:
        lines.append(f"Seeds: {', '.join(map(str, job.seeds))}")
    if job.seconds is not None:
        lines.append(f"Took {job.seconds:.0f}s")
    if job.parent:
        lines.append(f"From job {job.parent}" + (f", image {job.parent_index}" if job.parent_index else ""))
    return "\n".join(lines)
