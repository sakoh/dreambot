"""Async client for the MLX worker subprocess.

The bot process never imports MLX. It talks to a worker (:mod:`dreambot.engine.worker`) that
owns the models, which buys three things:

* the asyncio loop that serves Discord, Telegram and iMessage never blocks on inference;
* memory is reclaimed for certain — stopping the worker returns every byte to the system,
  which the idle timeout uses to give a Mac its RAM back between sessions;
* a hard cancel is always possible: a job that ignores a polite cancel is stopped by
  restarting the worker.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import os
import sys
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import IO, Any

from pydantic import BaseModel, ValidationError

from dreambot.engine.protocol import (
    Cancelled,
    CancelRequest,
    DescribeRequest,
    Done,
    ErrorKind,
    Event,
    Failed,
    GenerateRequest,
    ImageDone,
    Loaded,
    Ready,
    ShutdownRequest,
    StatusReport,
    StatusRequest,
    TextDone,
    TextRequest,
    UnloadRequest,
    decode_event,
    encode,
)

log = logging.getLogger(__name__)

EventCallback = Callable[[Event], Awaitable[None] | None]


class EngineError(Exception):
    """A request failed. ``kind`` is an :class:`ErrorKind` value; ``str(error)`` is user-presentable."""

    def __init__(self, message: str, kind: str = ErrorKind.INTERNAL, detail: str | None = None):
        super().__init__(message)
        self.kind = str(kind)
        self.detail = detail


class EngineCancelled(Exception):
    """The request was cancelled before it finished."""


@dataclass(frozen=True)
class EngineImage:
    """One result of a request: a picture, or a video with the picture it starts on."""

    index: int
    seed: int
    path: str
    width: int
    height: int
    seconds: float
    #: Set by the video operations: the video file, of which ``path`` is the first frame.
    video: str | None = None
    frames: int | None = None
    fps: float | None = None
    sound: bool = False


@dataclass
class GenerateResult:
    images: list[EngineImage] = field(default_factory=list)
    #: Wall-clock seconds the worker spent on the request, including any model load.
    seconds: float = 0.0
    #: Seconds spent loading the model, or ``None`` if it was already in memory.
    load_seconds: float | None = None


def new_request_id() -> str:
    return uuid.uuid4().hex


class Engine:
    """Owns one worker subprocess and multiplexes requests onto it.

    The worker is started lazily on the first request and, when ``idle_timeout`` is set,
    stopped again after that many seconds without work.
    """

    def __init__(
        self,
        *,
        backend: str = "mlx",
        python: str | None = None,
        env: Mapping[str, str] | None = None,
        offline: bool = True,
        idle_timeout: float | None = 900.0,
        start_timeout: float = 180.0,
        cancel_grace: float = 20.0,
        stderr: int | IO[Any] | None = None,
    ):
        self.backend = backend
        self._python = python or sys.executable
        self._extra_env = dict(env or {})
        self._offline = offline
        self.idle_timeout = idle_timeout
        self._start_timeout = start_timeout
        self._cancel_grace = cancel_grace
        self._stderr = stderr

        self._proc: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task[None] | None = None
        self._ready: Ready | None = None
        self._lifecycle = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._streams: dict[str, asyncio.Queue[Event]] = {}
        #: Request ids in the order they were sent. The worker runs them strictly in this
        #: order, so the first entry is the one that is executing right now.
        self._inflight: list[str] = []
        #: Which worker process (by pid) each sent request went to. When a worker exits, only
        #: its own requests are failed — never ones waiting for, or sent to, its successor.
        self._owner: dict[str, int] = {}
        self._cancelling: set[str] = set()
        self._tasks: set[asyncio.Task[None]] = set()
        self._idle_handle: asyncio.TimerHandle | None = None
        self._closed = False

    # ----------------------------------------------------------------- lifecycle ----

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    @property
    def info(self) -> Ready | None:
        """The worker's hello message (backend and library versions), once it has started."""
        return self._ready

    def _environment(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "DREAMBOT_BACKEND": self.backend,
                "PYTHONUNBUFFERED": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "DO_NOT_TRACK": "1",
            }
        )
        if self._offline:
            # Generation must never reach the network: weights are fetched only by an
            # explicit `dreambot models pull`.
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
        env.update(self._extra_env)
        return env

    async def start(self) -> Ready:
        """Start the worker if it is not running. Safe to call repeatedly."""
        async with self._lifecycle:
            return await self._start_locked()

    async def _start_locked(self) -> Ready:
        if self._closed:
            raise EngineError("The engine has been closed.", ErrorKind.CRASHED)
        if self.running and self._ready is not None:
            return self._ready
        proc = await asyncio.create_subprocess_exec(
            self._python,
            "-m",
            "dreambot.engine.worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=self._stderr,
            env=self._environment(),
            limit=8 * 1024 * 1024,
            start_new_session=True,
        )
        assert proc.stdout is not None
        try:
            line = await asyncio.wait_for(proc.stdout.readline(), self._start_timeout)
            hello = decode_event(line) if line else None
        except (TimeoutError, ValidationError) as exc:
            await self._kill(proc)
            raise EngineError(
                f"The inference worker did not start ({type(exc).__name__}).", ErrorKind.CRASHED
            ) from exc
        if isinstance(hello, Failed):
            await self._kill(proc)
            raise EngineError(hello.message, hello.kind, hello.detail)
        if not isinstance(hello, Ready):
            await self._kill(proc)
            raise EngineError("The inference worker exited during startup.", ErrorKind.CRASHED)
        self._proc = proc
        self._ready = hello
        self._reader = asyncio.create_task(self._read_events(proc), name="dreambot-engine-reader")
        log.info("engine worker started (pid %s, backend %s)", hello.pid, hello.backend)
        return hello

    async def stop(self) -> None:
        """Stop the worker, releasing all model memory. A later request starts a fresh one."""
        async with self._lifecycle:
            await self._stop_locked()

    async def _stop_if_idle(self) -> None:
        async with self._lifecycle:
            # Re-checked under the lock: a request may have arrived since the timer fired.
            if not self._streams:
                await self._stop_locked()

    async def close(self) -> None:
        """Stop the worker for good."""
        async with self._lifecycle:
            self._closed = True
            await self._stop_locked()

    async def _stop_locked(self) -> None:
        self._cancel_idle_timer()
        proc, self._proc = self._proc, None
        reader, self._reader = self._reader, None
        self._ready = None
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(Exception):
                assert proc.stdin is not None
                proc.stdin.write((encode(ShutdownRequest()) + "\n").encode())
                await proc.stdin.drain()
                proc.stdin.close()
            try:
                await asyncio.wait_for(proc.wait(), 5.0)
            except TimeoutError:
                await self._kill(proc)
        if reader is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(reader, 5.0)

    @staticmethod
    async def _kill(proc: asyncio.subprocess.Process) -> None:
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), 3.0)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()

    async def __aenter__(self) -> Engine:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    # ------------------------------------------------------------ event plumbing ----

    async def _read_events(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stdout is not None
        while True:
            try:
                line = await proc.stdout.readline()
            except (asyncio.LimitOverrunError, ValueError):
                log.warning("engine worker sent an oversized line; dropping it")
                continue
            if not line:
                break
            try:
                event = decode_event(line)
            except ValidationError:
                log.warning("engine worker sent an unreadable line: %r", line[:200])
                continue
            request_id = getattr(event, "id", None)
            if request_id is None:
                if isinstance(event, Failed):
                    log.error("engine worker error: %s", event.message)
                continue
            stream = self._streams.get(request_id)
            if stream is not None:
                stream.put_nowait(event)

        code = await proc.wait()
        if self._proc is proc:
            self._proc = None
            self._ready = None
        # Requests this process was sent will never be answered now. Requests that belong to
        # (or are waiting for) a newer worker are left alone.
        orphaned = [request_id for request_id, pid in self._owner.items() if pid == proc.pid]
        for request_id in orphaned:
            stream = self._streams.get(request_id)
            if stream is None:
                continue
            if request_id in self._cancelling:
                stream.put_nowait(Cancelled(id=request_id))
            else:
                stream.put_nowait(
                    Failed(
                        id=request_id,
                        kind=ErrorKind.CRASHED,
                        message=f"The inference worker stopped unexpectedly (exit code {code}). "
                        "This usually means the Mac ran out of memory for this model and size.",
                    )
                )

    async def _send(self, message: BaseModel, *, track: str | None = None) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.returncode is not None:
            raise EngineError("The inference worker is not running.", ErrorKind.CRASHED)
        async with self._write_lock:
            try:
                proc.stdin.write((encode(message) + "\n").encode())
                if track is not None:
                    self._inflight.append(track)
                    self._owner[track] = proc.pid
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as exc:
                raise EngineError("The inference worker is not running.", ErrorKind.CRASHED) from exc

    def _spawn(self, coro: Any, name: str) -> None:
        task = asyncio.get_running_loop().create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._reap)

    def _reap(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            log.error("engine background task %s failed", task.get_name(), exc_info=task.exception())

    async def _events(self, request: BaseModel) -> AsyncIterator[Event]:
        """Send a request and yield its events up to and including the terminal one.

        Consumers must close the generator (``contextlib.aclosing``). If one stops early —
        typically because its task was cancelled — the request is cancelled in the worker
        too, so abandoned work never holds up the queue.
        """
        request_id: str = request.id  # type: ignore[attr-defined]
        self._cancel_idle_timer()
        stream: asyncio.Queue[Event] = asyncio.Queue()
        self._streams[request_id] = stream
        finished = False
        sent = False
        try:
            # Starting and sending happen under the lifecycle lock so that an idle shutdown
            # can never slip in between and take the worker away from this request.
            async with self._lifecycle:
                await self._start_locked()
                await self._send(request, track=request_id)
            sent = True
            while True:
                event = await stream.get()
                if isinstance(event, Done | Cancelled | Failed):
                    finished = True
                yield event
                if finished:
                    return
        finally:
            if sent and not finished and self.running:
                self._spawn(self._abandon(request_id, stream), f"dreambot-abandon-{request_id[:8]}")
            else:
                self._forget(request_id)

    def _forget(self, request_id: str) -> None:
        self._streams.pop(request_id, None)
        self._owner.pop(request_id, None)
        self._cancelling.discard(request_id)
        with contextlib.suppress(ValueError):
            self._inflight.remove(request_id)
        if not self._streams:
            self._arm_idle_timer()

    async def _abandon(self, request_id: str, stream: asyncio.Queue[Event]) -> None:
        """Cancel a request nobody is waiting for any more and see it through to its end."""
        self._cancelling.add(request_id)
        try:
            with contextlib.suppress(EngineError):
                await self._send(CancelRequest(id=request_id))
            watchdog = asyncio.ensure_future(self._enforce_cancel(request_id))
            try:
                while True:
                    event = await stream.get()
                    if isinstance(event, Done | Cancelled | Failed):
                        return
            finally:
                watchdog.cancel()
        finally:
            self._forget(request_id)

    async def _enforce_cancel(self, request_id: str) -> None:
        """Restart the worker if a cancelled request is still *running* after the grace period.

        A cancelled request that is merely queued behind another job is answered as soon as
        the worker reaches it, so the clock only runs while it is at the head of the line.
        """
        waited = 0.0
        tick = 0.25
        while request_id in self._streams and self.running:
            if self._inflight and self._inflight[0] == request_id:
                waited += tick
                if waited >= self._cancel_grace:
                    log.warning("request %s ignored cancellation; restarting the worker", request_id)
                    proc = self._proc
                    if proc is not None:
                        await self._kill(proc)
                    return
            await asyncio.sleep(tick)

    # ---------------------------------------------------------------- idle timer ----

    def _cancel_idle_timer(self) -> None:
        if self._idle_handle is not None:
            self._idle_handle.cancel()
            self._idle_handle = None

    def _arm_idle_timer(self) -> None:
        self._cancel_idle_timer()
        if self.idle_timeout is None or not self.running or self._closed:
            return
        loop = asyncio.get_running_loop()
        self._idle_handle = loop.call_later(self.idle_timeout, self._on_idle)

    def _on_idle(self) -> None:
        self._idle_handle = None
        if self._streams or not self.running:
            return
        log.info("engine idle for %.0fs; stopping the worker to release memory", self.idle_timeout)
        self._spawn(self._stop_if_idle(), "dreambot-engine-idle-stop")

    # --------------------------------------------------------------- public API ----

    async def generate(
        self, request: GenerateRequest, on_event: EventCallback | None = None
    ) -> GenerateResult:
        """Run a generation request to completion.

        ``on_event`` (sync or async) receives every event — loading, progress, previews and
        each finished image or video — as it happens.

        Raises :class:`EngineError` on failure and :class:`EngineCancelled` if cancelled.
        """
        result = GenerateResult()
        async with contextlib.aclosing(self._events(request)) as events:
            async for event in events:
                if on_event is not None:
                    outcome = on_event(event)
                    if inspect.isawaitable(outcome):
                        await outcome
                if isinstance(event, ImageDone):
                    result.images.append(
                        EngineImage(
                            index=event.index,
                            seed=event.seed,
                            path=event.path,
                            width=event.width,
                            height=event.height,
                            seconds=event.seconds,
                            video=event.video,
                            frames=event.frames,
                            fps=event.fps,
                            sound=event.sound,
                        )
                    )
                elif isinstance(event, Loaded):
                    result.load_seconds = event.seconds
                elif isinstance(event, Done):
                    result.seconds = event.seconds
                elif isinstance(event, Cancelled):
                    raise EngineCancelled(request.id)
                elif isinstance(event, Failed):
                    raise EngineError(event.message, event.kind, event.detail)
        result.images.sort(key=lambda image: image.index)
        return result

    async def _texts(self, request: DescribeRequest | TextRequest) -> list[str]:
        texts: list[str] = []
        async with contextlib.aclosing(self._events(request)) as events:
            async for event in events:
                if isinstance(event, TextDone):
                    texts = event.texts
                elif isinstance(event, Cancelled):
                    raise EngineCancelled(request.id)
                elif isinstance(event, Failed):
                    raise EngineError(event.message, event.kind, event.detail)
        return texts

    async def describe(self, request: DescribeRequest) -> list[str]:
        return await self._texts(request)

    async def text(self, request: TextRequest) -> list[str]:
        return await self._texts(request)

    async def status(self, timeout: float = 2.0) -> StatusReport | None:
        """Memory and loaded-model report.

        Returns ``None`` when no worker is running, or when the worker is busy with a job and
        cannot answer within ``timeout`` seconds (status requests queue behind running work).
        """
        if not self.running:
            return None

        async def ask() -> StatusReport | None:
            report: StatusReport | None = None
            async with contextlib.aclosing(self._events(StatusRequest(id=new_request_id()))) as events:
                async for event in events:
                    if isinstance(event, StatusReport):
                        report = event
                    elif isinstance(event, Failed):
                        raise EngineError(event.message, event.kind, event.detail)
            return report

        try:
            return await asyncio.wait_for(ask(), timeout)
        except TimeoutError:
            return None

    async def unload(self, model: str | None = None) -> None:
        """Drop one model (or all of them) from memory without stopping the worker."""
        if not self.running:
            return
        async with contextlib.aclosing(
            self._events(UnloadRequest(id=new_request_id(), model=model))
        ) as events:
            async for event in events:
                if isinstance(event, Failed):
                    raise EngineError(event.message, event.kind, event.detail)

    async def cancel(self, request_id: str) -> None:
        """Cancel a queued or running request.

        The worker is asked politely first. If the request is still running after the grace
        period (a model stuck outside its denoising loop), the worker is restarted.
        """
        if request_id not in self._streams or not self.running or request_id in self._cancelling:
            return
        self._cancelling.add(request_id)
        with contextlib.suppress(EngineError):
            await self._send(CancelRequest(id=request_id))
        self._spawn(self._enforce_cancel(request_id), f"dreambot-cancel-{request_id[:8]}")
