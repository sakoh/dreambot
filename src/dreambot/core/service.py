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
import functools
import logging
import re
import shutil
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from dreambot.config import Config
from dreambot.core import enhance, fingerprint, imaging, lookbook
from dreambot.core.access import Access
from dreambot.core.actions import GENERATIVE, PAN_DIRECTIONS, Action, ActionKind
from dreambot.core.fetch import FetchError, fetch_image
from dreambot.core.fingerprint import StyleFingerprint
from dreambot.core.imaging import ImageError
from dreambot.core.jobs import EventType, Job, JobEvent, JobImage, JobKind, JobStatus, new_job_id
from dreambot.core.params import Params, PromptError, Ref
from dreambot.core.planner import (
    STYLE_LEGACY,
    Catalog,
    Plan,
    PlanError,
    Planner,
    Render,
    ResolvedRefs,
    StyleSource,
    style_version,
)
from dreambot.core.prompt import MAX_PERMUTATIONS, ParsedPrompt, expand_permutations, parse_prompt
from dreambot.core.settings import ResolvedSettings, UserSettings
from dreambot.core.sizes import working_size
from dreambot.core.store import Store, style_slug
from dreambot.core.text import SHORTEN_INSTRUCTION, heuristic_shorten, parse_numbered
from dreambot.engine.client import Engine, EngineCancelled, EngineError, new_request_id
from dreambot.engine.protocol import (
    DescribeRequest,
    ErrorKind,
    Event,
    Loading,
    Op,
    Preview,
    Progress,
    TextRequest,
)

log = logging.getLogger(__name__)

Listener = Callable[[JobEvent], Awaitable[None] | None]
#: Seconds between progress events for one job (frontends edit a chat message for each).
PROGRESS_INTERVAL = 1.0
MAX_JOBS_PER_MESSAGE = MAX_PERMUTATIONS
#: Shown once per person, on the first job a language model art-directed.
ENHANCE_HINT = (
    "A language model on this Mac added art direction to your prompt (`info` shows what it wrote). "
    "`--raw` skips that for one prompt and `/settings enhance off` turns it off. "
    "This note appears only once."
)
#: Operations whose prompt describes the picture (rather than instructing a change to one).
_SCENE_OPS = (Op.TXT2IMG, Op.IMG2IMG, Op.REDUX)


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
        #: Work that can wait for a quiet moment on the GPU (reading the style of a Lookbook's
        #: pictures). The queue runs one whenever no job is waiting.
        self._chores: list[Callable[[], Awaitable[None]]] = []
        self._chore_running = False

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

    @property
    def idle(self) -> bool:
        """Whether nothing is running or waiting to run, chores included."""
        return not (self._queue or self._chores or self._chore_running or self._current is not None)

    async def _work(self) -> None:
        while not self._closing:
            if not self._queue:
                if self._chores:
                    chore = self._chores.pop(0)
                    self._chore_running = True
                    try:
                        await chore()
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        log.exception("a background chore failed")
                    finally:
                        self._chore_running = False
                        self._current_request = None
                    continue
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

        planned: list[tuple[Job, Plan, Callable[..., Plan], ResolvedRefs | None]] = []
        for parsed in parsed_prompts:
            refs, ref_notes = await self._resolve_refs(
                parsed, prepared, scratch, user=user, settings=settings, allow_paths=allow_paths
            )
            for turn in range(parsed.params.repeat or 1):
                if turn:
                    # As in Midjourney, every repeat of `--sref random` draws a style of its own.
                    refs, ref_notes = self._redraw_styles(refs, ref_notes)
                try:
                    plan = self.planner.imagine(parsed, settings, refs)
                except (PlanError, PromptError) as exc:
                    raise DreambotError(str(exc)) from exc
                # The same request planned again, with the seeds already drawn, once the
                # language model has written its direction.
                replan = functools.partial(
                    self.planner.imagine, parsed, settings, refs, seeds=list(plan.seeds)
                )
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
                # Words for a style come from the vision model, which has to wait its turn on
                # the GPU like everything else: the job asks for them when it runs.
                waiting = self._style_pending(parsed, refs)
                planned.append((job, plan, replan, refs if waiting else None))
        for job, plan, replan, waiting_refs in planned:
            await self._enqueue(job, self._runner(plan, replan, waiting_refs))
        return [job for job, _, _, _ in planned]

    def _display_prompt(self, parsed: ParsedPrompt, refs: ResolvedRefs) -> str:
        """The prompt as it should be shown and re-run.

        `--sref random` is pinned to the code that was drawn, and a Lookbook named by name is
        pinned to the code of the version used — as Midjourney turns a moodboard's id into a
        code on submit — so that running the prompt again later gives the same look.
        """
        text = parsed.canonical()
        for source in refs.sources:
            if source.drawn and source.code:
                # The value of a parameter runs to the next " --"; names in it may contain hyphens.
                text = re.sub(r"(--sref\b(?:(?!\s--).)*?)\brandom\b", rf"\g<1>{source.code}", text, count=1)
        books = [source for source in refs.sources if source.kind == "look" and source.code]
        if books and parsed.params.personalize is not None:
            pinned = " ".join(
                source.code if source.weight == 1 else f"{source.code}::{source.weight:g}" for source in books
            )
            text = re.sub(
                r"--personalize\b(?:(?!\s--).)*", lambda _: f"--personalize {pinned}", text, count=1
            )
        return text

    async def _resolve_refs(
        self,
        parsed: ParsedPrompt,
        attachments: list[str],
        scratch: Path,
        *,
        user: str,
        settings: ResolvedSettings,
        allow_paths: bool,
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
        # Style ingredients keep the order and the weight they were given. Explicit ones are
        # resolved first, so that a bare `--sref` takes the attachments nobody else named.
        found: dict[int, list[StyleSource]] = {}
        for position, ref in sorted(
            enumerate(params.sref), key=lambda item: item[1].kind == "attachment" and item[1].value == "*"
        ):
            found[position] = await self._style_sources(ref, user, one, notes)
        resolved.sources = [source for position in sorted(found) for source in found[position]]
        pictured = 0
        for source in resolved.sources:
            if source.kind == "image":
                pictured += 1
                source.label = f"image {pictured}"
                resolved.style.extend(source.pictures)
            elif source.code is not None:
                resolved.style_codes.append(source.code)
        resolved.sources.extend(await asyncio.to_thread(self._look_sources, params, user, settings))
        resolved.image_prompts = await many(parsed.image_prompts, "image prompt")
        # Attached images nobody claimed are image prompts, as in Midjourney.
        resolved.image_prompts.extend(path for i, path in enumerate(attachments) if i not in claimed)
        return resolved, notes

    # -------------------------------------------------------------------- styles ----

    async def _style_sources(
        self,
        ref: Ref,
        user: str,
        fetch: Callable[[Ref, str], Awaitable[list[str]]],
        notes: list[str],
    ) -> list[StyleSource]:
        """The style ingredients one ``--sref`` token stands for."""
        if ref.kind == "code":
            return [StyleSource("code", f"style {ref.value}", ref.weight, code=ref.value)]
        if ref.kind == "random":
            code = fingerprint.random_code()
            notes.append(f"`--sref random` picked style `{code}`. Reuse it with `--sref {code}`.")
            return [StyleSource("code", f"style {code}", ref.weight, code=code, drawn=True)]
        if ref.kind == "saved":
            kept = self.store.get_style(user, ref.value)
            if kept is None or not kept["data"]:
                raise DreambotError(
                    f"I don't know a style called `{ref.value}`. `--sref` takes image links, `@1`-style "
                    "attachment numbers, style codes, `random`, or the name of a style you saved."
                )
            data = kept["data"]
            return [
                StyleSource(
                    "saved",
                    f"style {kept['name'] or kept['code']}",
                    ref.weight,
                    print=StyleFingerprint.from_dict(data.get("print") or {}),
                    pictures=[path for path in data.get("pictures", []) if Path(path).is_file()],
                )
            ]
        sources: list[StyleSource] = []
        for path in await fetch(ref, "sref"):
            key = await asyncio.to_thread(fingerprint.image_key, path)
            print_ = await asyncio.to_thread(self._print_of, path, key)
            sources.append(StyleSource("image", "image", ref.weight, print=print_, pictures=[path], key=key))
        return sources

    def style_of(self, path: str | Path) -> tuple[str, StyleFingerprint]:
        """A picture's content key and what is known about its style so far."""
        key = fingerprint.image_key(path)
        return key, self._print_of(path, key)

    def study_picture(self, path: str | Path) -> bool:
        """Have the vision model read a picture's style when the GPU is next free.

        Returns whether there is anything to wait for.
        """
        key, known = self.style_of(path)
        if known.worded or self.catalog.vision_model() is None:
            return False

        async def read() -> None:
            now = self.store.get_fingerprint(key)
            if not (now and now.get("medium")) and Path(path).is_file():
                await self._read_style(str(path), key, self._print_of(path, key))

        self._chores.append(read)
        self._wake.set()
        return True

    def _print_of(self, path: str | Path, key: str) -> StyleFingerprint:
        """What is known about a picture's style: its measured colours, and its words once read."""
        known = self.store.get_fingerprint(key)
        if known:
            return StyleFingerprint.from_dict(known)
        measured = fingerprint.measure(path)
        self.store.save_fingerprint(key, measured.to_dict())
        return measured

    async def _read_style(self, path: str, key: str | None, known: StyleFingerprint) -> StyleFingerprint:
        """Ask the vision model to put a picture's style into words, and remember the answer."""
        vision = self.catalog.vision_model()
        if vision is None:
            return known
        request = DescribeRequest(
            id=new_request_id(),
            image=path,
            count=len(fingerprint.WORD_FIELDS),
            weights=vision,
            instruction=fingerprint.EXTRACT_INSTRUCTION,
            max_tokens=40,
            seed=0,
        )
        self._current_request = request.id
        try:
            lines = await self.engine.describe(request)
        except EngineCancelled:
            raise
        except EngineError as exc:
            log.warning("could not read the style of %s (%s)", Path(path).name, exc)
            return known
        words = fingerprint.parse_words(lines)
        if not words.get("medium"):
            log.warning("the vision model's account of the style of %s was unusable", Path(path).name)
            return known
        read = fingerprint.with_words(known, words)
        if key:
            self.store.save_fingerprint(key, read.to_dict())
        return read

    def _style_pending(self, parsed: ParsedPrompt, refs: ResolvedRefs) -> bool:
        """Whether a job should ask the vision model about its style before it renders."""
        if not refs.sources or style_version(parsed.params.sv) == STYLE_LEGACY:
            return False
        if self.catalog.vision_model() is None:
            return False
        return any(
            (source.kind == "image" and source.print is not None and not source.print.worded)
            or any(not member.print.worded for member in source.members)
            for source in refs.sources
        )

    def _redraw_styles(self, refs: ResolvedRefs, notes: list[str]) -> tuple[ResolvedRefs, list[str]]:
        """``refs`` with every `--sref random` drawn afresh."""
        if not any(source.drawn for source in refs.sources):
            return refs, notes
        notes = [note for note in notes if not note.startswith("`--sref random` picked")]
        sources: list[StyleSource] = []
        for source in refs.sources:
            if source.drawn:
                code = fingerprint.random_code()
                notes.append(f"`--sref random` picked style `{code}`. Reuse it with `--sref {code}`.")
                source = StyleSource("code", f"style {code}", source.weight, code=code, drawn=True)
            sources.append(source)
        codes = [source.code for source in sources if source.code is not None]
        return replace(refs, sources=sources, style_codes=codes), notes

    async def _describe_styles(self, job: Job, refs: ResolvedRefs) -> None:
        """Have the vision model put a job's style into words (see :mod:`dreambot.core.fingerprint`).

        Like the prompt enhancer this refines a job and is never a reason for one to fail:
        without the words a style is still applied through its colours and, where a model
        takes them, its pictures.
        """
        if self.catalog.vision_model() is None:
            return
        announced = False
        for source in refs.sources:
            if source.kind != "image" or source.print is None or source.print.worded or not source.pictures:
                continue
            known = self.store.get_fingerprint(source.key) if source.key else None
            if known and known.get("medium"):
                source.print = StyleFingerprint.from_dict(known)  # another job read it in the meantime
                continue
            if not announced:
                announced = True
                await self._emit(JobEvent(EventType.THINKING, job))
            source.print = await self._read_style(source.pictures[0], source.key, source.print)

        # A Lookbook's pictures are usually read while the GPU is idle (see `study_lookbook`);
        # whatever is still unread when a job needs the book is read here.
        for source in refs.sources:
            for position, member in enumerate(source.members):
                if member.print.worded:
                    continue
                known = self.store.get_fingerprint(member.key)
                if known and known.get("medium"):
                    read = StyleFingerprint.from_dict(known)
                else:
                    if not announced:
                        announced = True
                        await self._emit(JobEvent(EventType.THINKING, job))
                    read = await self._read_style(member.path, member.key, member.print)
                source.members[position] = lookbook.Member(member.key, member.path, read)

    def save_style(
        self,
        *,
        user: str,
        name: str,
        print_: StyleFingerprint,
        pictures: Sequence[str | Path] = (),
        recipe: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Keep a style under a name, to be used as ``--sref <name>``.

        Midjourney cannot make a style code from your own image or blend; this is how Dreambot
        does. The style's pictures are copied, so it outlives the files it was made from.
        """
        name = re.sub(r"\s+", " ", name).strip()
        slug = style_slug(name)
        if not slug or not slug[0].isalpha():
            raise DreambotError("Give the style a name that starts with a letter, e.g. `dusty noir`.")
        if print_.empty:
            raise DreambotError("There is no style to save yet.")
        existing = self.store.get_style(user, slug)
        code = existing["code"] if existing and existing["data"] else f"s{new_job_id().lower()}"
        folder = self.config.home / "styles" / code
        folder.mkdir(parents=True, exist_ok=True)
        kept: list[str] = []
        for number, picture in enumerate(pictures, start=1):
            source = Path(picture)
            if source.is_file():
                target = folder / f"{number}{source.suffix.lower() or '.png'}"
                if source.resolve() != target.resolve():
                    shutil.copyfile(source, target)
                kept.append(str(target))
        named = replace(print_, name=name)
        data = {"print": named.to_dict(), "pictures": kept, "recipe": recipe or []}
        self.store.save_style(user, code, name=name, data=data, liked=bool(existing and existing["liked"]))
        return {"code": code, "name": name, "slug": slug, "data": data}

    # ----------------------------------------------------------------- lookbooks ----

    #: The most pictures one Lookbook holds.
    MAX_LOOKBOOK_IMAGES = 60

    def _book(self, lookbook_id: str) -> dict[str, Any]:
        book = self.store.get_lookbook(lookbook_id)
        if book is None or book["deleted"]:
            raise DreambotError("I can't find that Lookbook.")
        return book

    def find_lookbook(self, token: str) -> dict[str, Any] | None:
        """A Lookbook by its id, or by its name (as written, or as the one word typed after ``--p``)."""
        wanted = token.strip().lower()
        books = self.store.list_lookbooks()
        for book in books:
            if book["id"].lower() == wanted:
                return book
        for book in books:
            if lookbook.slug(book["name"]) == lookbook.slug(wanted):
                return book
        return None

    def _check_book_name(self, name: str, keeping: str | None = None) -> str:
        name = lookbook.clean_name(name)
        slug = lookbook.slug(name)
        if not slug:
            raise DreambotError("Give the Lookbook a name, e.g. `Autumn diner`.")
        if slug in (lookbook.DEFAULT, lookbook.LIKED):
            raise DreambotError(f"`{name}` already means something after `--p`; choose another name.")
        other = self.find_lookbook(slug)
        if other is not None and other["id"] != keeping:
            raise DreambotError(f"There is already a Lookbook called `{other['name']}`.")
        return name

    def _members(self, images: Sequence[dict[str, Any]]) -> list[lookbook.Member]:
        return [
            lookbook.Member(image["key"], image["path"], self._print_of(image["path"], image["key"]))
            for image in images
            if Path(image["path"]).is_file()
        ]

    def describe_lookbook(self, book: dict[str, Any], user: str | None = None) -> dict[str, Any]:
        """A Lookbook as frontends show it: its pictures, its code, and what Dreambot makes of it."""
        images = self.store.lookbook_images(book["id"])
        members = self._members(images)
        prints = {member.key: member.print for member in members}
        blend = lookbook.consensus(members)
        code = self._freeze(book["id"], book["name"], images) if images else None
        return {
            "id": book["id"],
            "name": book["name"],
            "slug": lookbook.slug(book["name"]),
            "code": code,
            "count": len(images),
            "updated": book["updated"],
            "default": bool(user and book["id"] in self.settings(user).looks),
            "unread": sum(1 for member in members if not member.print.worded),
            "style": style_card(blend),
            "images": [
                {
                    "id": image["id"],
                    "path": image["path"],
                    "source": image["source"],
                    "style": style_card(prints.get(image["key"])),
                }
                for image in images
            ],
        }

    def lookbooks(self, user: str | None = None) -> list[dict[str, Any]]:
        return [self.describe_lookbook(book, user) for book in self.store.list_lookbooks()]

    def create_lookbook(self, *, user: str, name: str) -> dict[str, Any]:
        book = self.store.create_lookbook(new_job_id().lower(), user, self._check_book_name(name))
        return self.describe_lookbook(book, user)

    def rename_lookbook(self, *, user: str, lookbook_id: str, name: str) -> dict[str, Any]:
        book = self._book(lookbook_id)
        self.store.update_lookbook(lookbook_id, name=self._check_book_name(name, keeping=book["id"]))
        return self.describe_lookbook(self._book(lookbook_id), user)

    def delete_lookbook(self, *, lookbook_id: str) -> None:
        """Remove a Lookbook from the list. The codes it has been used under keep working."""
        book = self._book(lookbook_id)
        images = self.store.lookbook_images(lookbook_id)
        if images:
            self._freeze(lookbook_id, book["name"], images)
        self.store.update_lookbook(lookbook_id, deleted=True)

    def _freeze(self, lookbook_id: str, name: str, images: Sequence[dict[str, Any]]) -> str:
        """The code of a Lookbook as it is now, recorded so that it can be used again later."""
        code = lookbook.version_code(lookbook_id, [image["key"] for image in images])
        self.store.save_lookbook_code(
            code,
            lookbook_id,
            {"name": name, "images": [{"key": image["key"], "path": image["path"]} for image in images]},
        )
        return code

    async def add_to_lookbook(
        self, *, user: str, lookbook_id: str, source: str | Path, origin: str = "upload"
    ) -> dict[str, Any]:
        """Add a picture to a Lookbook: a file on this Mac, or a link to one.

        The picture is copied into Dreambot's own folder, so the Lookbook does not depend on the
        file it came from. Adding a picture the book already holds changes nothing.
        """
        book = self._book(lookbook_id)
        images = self.store.lookbook_images(lookbook_id)
        if len(images) >= self.MAX_LOOKBOOK_IMAGES:
            raise DreambotError(f"A Lookbook holds at most {self.MAX_LOOKBOOK_IMAGES} pictures.")
        scratch = self._scratch()
        try:
            if isinstance(source, str) and re.match(r"https?://", source, re.IGNORECASE):
                origin = f"link:{source}"[:300]
                source = await asyncio.to_thread(fetch_image, source, scratch)
            prepared = await asyncio.to_thread(imaging.prepare_input, source, scratch, max_side=2048)
        except (FetchError, ImageError) as exc:
            raise DreambotError(str(exc)) from exc
        key = await asyncio.to_thread(fingerprint.image_key, prepared)
        if not any(image["key"] == key for image in images):
            image_id = new_job_id().lower()
            folder = self.config.home / "lookbooks" / lookbook_id
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / f"{image_id}{Path(prepared).suffix.lower() or '.png'}"
            shutil.copyfile(prepared, target)
            self.store.add_lookbook_image(image_id, lookbook_id, str(target), key, origin)
            await asyncio.to_thread(self._print_of, target, key)
            self.study_lookbook(lookbook_id)
        return self.describe_lookbook(self._book(book["id"]), user)

    def remove_from_lookbook(self, *, user: str, lookbook_id: str, image_id: str) -> dict[str, Any]:
        """Take a picture out of a Lookbook. Its file stays: an earlier code may still stand for it."""
        book = self._book(lookbook_id)
        images = self.store.lookbook_images(lookbook_id)
        if images:
            self._freeze(lookbook_id, book["name"], images)  # the state being left stays usable
        if not self.store.remove_lookbook_image(lookbook_id, image_id):
            raise DreambotError("That picture isn't in this Lookbook.")
        return self.describe_lookbook(self._book(lookbook_id), user)

    def _look_sources(self, params: Params, user: str, settings: ResolvedSettings) -> list[StyleSource]:
        """The Lookbooks a prompt asks for with ``--p``, each as a style ingredient.

        A token is the code of a version, the id or name of a Lookbook (its current version),
        ``default`` for the person's chosen Lookbooks, or ``liked`` for their favourites.
        """
        tokens = params.personalize
        # With the Personalize switch on, every prompt uses the defaults — but a person who has
        # not set any up yet is not told so on every single prompt.
        insist = tokens is not None
        if tokens is None:
            if not settings.personalize:
                return []
            tokens = [lookbook.DEFAULT]
        sources: list[StyleSource] = []
        for token in tokens:
            name, weight = lookbook.parse_token(token)
            if weight <= 0:
                continue
            wanted = name.lower()
            if wanted == lookbook.DEFAULT:
                chosen = [book for book in map(self.store.get_lookbook, settings.looks) if book]
                found = [
                    source
                    for book in chosen
                    if not book["deleted"] and (source := self._book_source(book, weight, quiet=True))
                ]
                liked = None if found else self._liked_source(weight)
                if not found and liked is None and insist:
                    raise DreambotError(
                        "You have no default Lookbook yet. Make one in the web app (`dreambot web`) and "
                        "choose it there or with `/looks default <name>`, or name one: `--p <name>`. "
                        "Once you have favourited some pictures, `--p` uses those."
                    )
                sources.extend(found or ([liked] if liked else []))
            elif wanted == lookbook.LIKED:
                liked = self._liked_source(weight)
                if liked is None:
                    raise DreambotError(
                        "You haven't favourited any pictures yet, so there is no `liked` look."
                    )
                sources.append(liked)
            elif frozen := (
                self.store.get_lookbook_code(wanted) if lookbook.looks_like_code(wanted) else None
            ):
                members = self._members(frozen.get("images", []))
                if not members:
                    raise DreambotError(f"The pictures behind the Lookbook code `{wanted}` are gone.")
                label = f"Lookbook {frozen.get('name') or wanted}"
                sources.append(StyleSource("look", label, weight, code=wanted, members=members))
            else:
                book = self.find_lookbook(name)
                if book is None:
                    known = ", ".join(f"`{lookbook.slug(b['name'])}`" for b in self.store.list_lookbooks())
                    raise DreambotError(
                        f"I don't know a Lookbook called `{name}`."
                        + (f" You have: {known}." if known else " Make one in the web app: `dreambot web`.")
                    )
                source = self._book_source(book, weight)
                assert source is not None
                sources.append(source)
        return sources

    def _book_source(self, book: dict[str, Any], weight: float, *, quiet: bool = False) -> StyleSource | None:
        images = self.store.lookbook_images(book["id"])
        members = self._members(images)
        if not members:
            if quiet:
                return None
            raise DreambotError(f"The Lookbook `{book['name']}` has no pictures in it yet.")
        code = self._freeze(book["id"], book["name"], images)
        return StyleSource("look", f"Lookbook {book['name']}", weight, code=code, members=members)

    def _liked_source(self, weight: float) -> StyleSource | None:
        """The pictures a person has favourited, as a Lookbook."""
        images = []
        for _job, image in self.store.favorite_images(lookbook.LIKED_LIMIT):
            if Path(image.path).is_file():
                images.append({"key": fingerprint.image_key(image.path), "path": image.path})
        if not images:
            return None
        code = self._freeze(lookbook.LIKED, "Liked", images)
        return StyleSource("look", "Lookbook Liked", weight, code=code, members=self._members(images))

    def set_default_lookbooks(self, *, user: str, lookbook_ids: Sequence[str]) -> list[str]:
        """Choose the Lookbooks a bare ``--p`` stands for."""
        known = {book["id"] for book in self.store.list_lookbooks()}
        chosen = [book_id for book_id in dict.fromkeys(lookbook_ids) if book_id in known]
        settings = self.settings(user)
        settings.looks = chosen
        self.store.save_settings(user, settings.to_dict())
        return chosen

    def study_lookbook(self, lookbook_id: str) -> int:
        """Have the vision model read a Lookbook's pictures when the GPU is next free.

        Returns how many are waiting to be read. A job that uses the book before then reads what
        it needs itself, so this only ever saves time.
        """
        if self.catalog.vision_model() is None:
            return 0
        waiting = 0
        for image in self.store.lookbook_images(lookbook_id):
            known = self.store.get_fingerprint(image["key"])
            if known and known.get("medium"):
                continue
            waiting += 1

            async def read(path: str = image["path"], key: str = image["key"]) -> None:
                now = self.store.get_fingerprint(key)
                if (now and now.get("medium")) or not Path(path).is_file():
                    return
                await self._read_style(path, key, self._print_of(path, key))

            self._chores.append(read)
        if waiting:
            self._wake.set()
        return waiting

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
        if kind is ActionKind.LOOK:
            name = lookbook.clean_name(action.prompt or "")
            if not name:
                raise DreambotError("Name the Lookbook to add this to, e.g. `look 2: autumn diner`.")
            book = self.find_lookbook(name)
            started = book is None
            if book is None:
                book = self.store.create_lookbook(new_job_id().lower(), user, self._check_book_name(name))
            kept = await self.add_to_lookbook(
                user=user, lookbook_id=book["id"], source=image.path, origin=f"job:{job.id}:{image.index}"
            )
            opening = (
                f"Started the Lookbook `{kept['name']}` with this picture."
                if started
                else f"Added to the Lookbook `{kept['name']}`, which now has {kept['count']} pictures."
            )
            return ActionResult(
                jobs=[], job=job, image=image, message=f"{opening} Use it with `--p {kept['slug']}`."
            )
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
        if image.prompt and action.kind in (
            ActionKind.VARY,
            ActionKind.VARY_SUBTLE,
            ActionKind.VARY_STRONG,
            ActionKind.UPSCALE_SUBTLE,
            ActionKind.UPSCALE_CREATIVE,
        ):
            # This image was art-directed on its own: a variation of the wide shot should stay
            # a wide shot. (Zooming and panning change the framing, so they use the plain text.)
            text = image.prompt
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
            # Fresh seeds, and one job: rerolling one result of a `--repeat 4` runs it once, as in Midjourney.
            prompt = re.sub(r"\s--(?:seed|sameseed|repeat)\s+\d+", "", f" {job.prompt}").strip()
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

    def _runner(
        self,
        plan: Plan,
        replan: Callable[..., Plan] | None = None,
        style_refs: ResolvedRefs | None = None,
    ) -> Callable[[Job], Awaitable[None]]:
        """What the queue calls to carry a plan out.

        ``style_refs`` are the references of a job whose style still has to be put into words:
        the vision model is asked first, and the job is planned again with what it said.
        """

        async def run(job: Job) -> None:
            final = plan
            if style_refs is not None and replan is not None:
                await self._describe_styles(job, style_refs)
                try:
                    final = replan()
                except (PlanError, PromptError) as exc:
                    log.warning("job %s: could not plan with the style's words (%s)", job.id, exc)
                else:
                    if "style" in final.detail:
                        job.detail["style"] = _jsonable(final.detail["style"])
            if final.brief is not None and replan is not None:
                final = await self._enhance(job, final, replan)
            await self._run_plan(job, final)

        return run

    async def _enhance(self, job: Job, plan: Plan, replan: Callable[..., Plan]) -> Plan:
        """Have the local language model art-direct each image, and plan again with what it wrote.

        Enhancement refines a job; it is never a reason for one to fail. Whatever goes wrong
        here — no language model, an unusable answer, a planning error — the job runs with the
        plan it already has.
        """
        brief = plan.brief
        vision = self.catalog.vision_model()
        if brief is None or vision is None:
            return plan
        await self._emit(JobEvent(EventType.THINKING, job))
        request = TextRequest(
            id=new_request_id(),
            weights=vision,
            system=enhance.SYSTEM,
            prompt=enhance.request(brief),
            max_tokens=enhance.max_tokens(brief),
            temperature=enhance.temperature(brief),
            top_p=enhance.TOP_P,
            repetition_penalty=enhance.REPETITION_PENALTY,
            # Seeded like the images, so the same prompt and seed give the same direction.
            seed=plan.seeds[0] if plan.seeds else None,
        )
        self._current_request = request.id
        try:
            answers = await self.engine.text(request)
        except EngineCancelled:
            raise
        except EngineError as exc:
            log.warning(
                "job %s: the prompt enhancer failed (%s); rendering the prompt as planned", job.id, exc
            )
            return plan
        directions = enhance.parse(answers[0] if answers else "", brief)
        if not any(directions):
            log.warning("job %s: the prompt enhancer returned nothing usable", job.id)
            return plan
        try:
            directed = replan(directions=directions)
        except (PlanError, PromptError) as exc:
            log.warning("job %s: could not plan with the enhancer's direction (%s)", job.id, exc)
            return plan
        job.detail["enhanced"] = True
        settings = self.settings(job.user)
        if "enhance" not in settings.hints:
            job.notes.append(ENHANCE_HINT)
            settings.hints.append("enhance")
            self.store.save_settings(job.user, settings.to_dict())
        return directed

    async def _run_plan(self, job: Job, plan: Plan) -> None:
        day = datetime.fromtimestamp(job.created).strftime("%Y-%m-%d")
        out_dir = self.config.output_dir / day
        out_dir.mkdir(parents=True, exist_ok=True)
        work_dir = self.config.scratch_dir / "jobs" / job.id
        stem = f"{_slug(job.text or job.kind.value)}_{job.id}"
        units = plan.work + (plan.count if plan.finish else 0)
        done = 0
        last_progress = 0.0
        # Whether each render's prompt describes its picture and so is worth keeping with it.
        scene = plan.op in _SCENE_OPS and plan.canvas is None and plan.keep_outside_mask is None

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
                board: str | None = None
                if render.board is not None:
                    cut = await asyncio.to_thread(
                        imaging.style_board,
                        render.board.tiles,
                        render.board.seed,
                        work_dir / f"board{number}.png",
                    )
                    board = str(cut)
                request = render.request(
                    request_id=new_request_id(),
                    out_dir=str(work_dir),
                    basename=f"r{number}",
                    preview_every=self.config.engine.preview_every,
                    board=board,
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
                            prompt=render.prompt if scene and render.prompt != job.text else None,
                        )
                    )

            if plan.tiles is not None and plan.tile_base:
                final = out_dir / f"{stem}_1.png"
                placed = [(tile.box, pieces[number]) for number, tile in enumerate(plan.tiles, start=1)]
                await asyncio.to_thread(imaging.blend_tiles, plan.tile_base, placed, final)
                # A tiled upscale keeps the prompt of the image it enlarged (that is its text).
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


def style_card(print_: StyleFingerprint | None) -> dict[str, Any] | None:
    """A fingerprint as frontends show it."""
    if print_ is None or print_.empty:
        return None
    return {
        "name": print_.name or fingerprint.name_of(print_),
        "summary": fingerprint.summary(print_),
        "words": print_.words(),
        "swatches": [{"hex": swatch.hex, "share": swatch.share} for swatch in print_.swatches],
        "read": print_.worded,
    }


def describe_job(job: Job, image: JobImage | None = None) -> str:
    """Seed, id and settings of a job (Midjourney's envelope reaction)."""
    lines = [f"Job {job.id} · {job.kind.value} · {job.model or 'n/a'}"]
    if job.prompt:
        lines.append(f"Prompt: {job.prompt}")
    if style := job.detail.get("style"):
        mix = ", ".join(f"{part['label']} {round(part['share'] * 100)}%" for part in style.get("sources", []))
        lines.append(f"Style: {style.get('summary', '')}" + (f" ({mix})" if "," in mix else ""))
    if image is not None:
        lines.append(f"Image {image.index}: seed {image.seed} · {image.width}x{image.height}")
        if image.prompt:
            lines.append(f"Rendered from: {image.prompt}")
        lines.append(f"File: {image.path}")
    elif job.seeds:
        lines.append(f"Seeds: {', '.join(map(str, job.seeds))}")
    if job.seconds is not None:
        lines.append(f"Took {job.seconds:.0f}s")
    if job.parent:
        lines.append(f"From job {job.parent}" + (f", image {job.parent_index}" if job.parent_index else ""))
    return "\n".join(lines)
