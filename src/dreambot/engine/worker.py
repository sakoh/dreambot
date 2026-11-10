"""The worker subprocess: owns the MLX runtime and every loaded model.

Run as ``python -m dreambot.engine.worker``. It speaks the JSON-lines protocol from
:mod:`dreambot.engine.protocol`: requests arrive on stdin, events leave on the *original*
stdout. Libraries print freely (progress bars, warnings), so file descriptor 1 is re-pointed
at stderr on startup and the protocol keeps a private duplicate of the real stdout.

Requests run strictly one at a time on the main thread. A small reader thread only parses
incoming lines, so a ``cancel`` can reach a job that is in the middle of denoising.
"""

from __future__ import annotations

import os
import queue
import signal
import sys
import threading
import time
import traceback
from typing import TextIO

from pydantic import BaseModel, ValidationError

from dreambot.engine.backends import load_backend
from dreambot.engine.backends.base import Backend, BackendError, JobCancelled, JobContext
from dreambot.engine.protocol import (
    Cancelled,
    CancelRequest,
    DescribeRequest,
    Done,
    ErrorKind,
    Failed,
    GenerateRequest,
    Ready,
    ShutdownRequest,
    StatusRequest,
    TextDone,
    TextRequest,
    UnloadRequest,
    decode_request,
    encode,
)


class _Cancellations:
    """Tracks which request ids have been cancelled, including ones that have not started yet."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: dict[str, threading.Event] = {}
        self._pending: set[str] = set()
        self._closed = False

    def begin(self, request_id: str) -> threading.Event:
        with self._lock:
            event = threading.Event()
            if self._closed or request_id in self._pending:
                self._pending.discard(request_id)
                event.set()
            self._events[request_id] = event
            return event

    def end(self, request_id: str) -> None:
        with self._lock:
            self._events.pop(request_id, None)

    def cancel(self, request_id: str) -> None:
        with self._lock:
            event = self._events.get(request_id)
            if event is not None:
                event.set()
            else:
                self._pending.add(request_id)

    def cancel_everything(self) -> None:
        """The parent went away: abort the running job and refuse the queued ones."""
        with self._lock:
            self._closed = True
            for event in self._events.values():
                event.set()


class Worker:
    def __init__(self, backend: Backend, out: TextIO, source: TextIO):
        self.backend = backend
        self._out = out
        self._source = source
        self._write_lock = threading.Lock()
        self._inbox: queue.Queue[BaseModel | None] = queue.Queue()
        self._cancellations = _Cancellations()

    def emit(self, event: BaseModel) -> None:
        line = encode(event)
        with self._write_lock:
            try:
                self._out.write(line + "\n")
                self._out.flush()
            except (BrokenPipeError, ValueError):
                # The parent closed the pipe; the reader thread will see EOF and stop us.
                pass

    # ------------------------------------------------------------------- reading ----

    def _read_requests(self) -> None:
        for raw in self._source:
            line = raw.strip()
            if not line:
                continue
            try:
                request = decode_request(line)
            except ValidationError as exc:
                self.emit(
                    Failed(
                        id=None,
                        kind=ErrorKind.INVALID,
                        message="Malformed request.",
                        detail=str(exc),
                    )
                )
                continue
            if isinstance(request, CancelRequest):
                self._cancellations.cancel(request.id)
            else:
                self._inbox.put(request)
        self._cancellations.cancel_everything()
        self._inbox.put(None)

    # ------------------------------------------------------------------- running ----

    def run(self) -> int:
        self.emit(Ready(pid=os.getpid(), backend=self.backend.name, versions=self.backend.versions()))
        threading.Thread(target=self._read_requests, name="dreambot-requests", daemon=True).start()
        while True:
            request = self._inbox.get()
            if request is None or isinstance(request, ShutdownRequest):
                return 0
            self._handle(request)

    def _handle(self, request: BaseModel) -> None:
        request_id: str = request.id  # type: ignore[attr-defined]
        cancel_event = self._cancellations.begin(request_id)
        ctx = JobContext(request_id, self.emit, cancel_event)
        started = time.perf_counter()
        try:
            ctx.check_cancelled()
            if isinstance(request, GenerateRequest):
                self.backend.generate(request, ctx)
            elif isinstance(request, DescribeRequest):
                texts = self.backend.describe(request, ctx)
                self.emit(TextDone(id=request_id, texts=texts, seconds=time.perf_counter() - started))
            elif isinstance(request, TextRequest):
                texts = self.backend.text(request, ctx)
                self.emit(TextDone(id=request_id, texts=texts, seconds=time.perf_counter() - started))
            elif isinstance(request, UnloadRequest):
                self.backend.unload(request.model)
            elif isinstance(request, StatusRequest):
                self.emit(self.backend.status(request_id))
            else:  # pragma: no cover - the protocol union is closed
                raise BackendError(f"Unsupported request type: {type(request).__name__}", ErrorKind.INVALID)
        except JobCancelled:
            self.emit(Cancelled(id=request_id))
        except BackendError as exc:
            self.emit(Failed(id=request_id, kind=exc.kind, message=str(exc), detail=exc.detail))
        except MemoryError as exc:
            self.emit(Failed(id=request_id, kind=ErrorKind.OUT_OF_MEMORY, message=f"Out of memory: {exc}"))
        except Exception as exc:
            traceback.print_exc()
            self.emit(
                Failed(
                    id=request_id,
                    kind=ErrorKind.INTERNAL,
                    message=f"{type(exc).__name__}: {exc}",
                    detail=traceback.format_exc(),
                )
            )
        else:
            self.emit(Done(id=request_id, seconds=time.perf_counter() - started))
        finally:
            self._cancellations.end(request_id)


def main() -> int:
    # Keep a private handle on the real stdout for the protocol, then send every other write
    # to file descriptor 1 (print(), tqdm, native libraries) to stderr.
    protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1, encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    protocol_in = os.fdopen(os.dup(sys.stdin.fileno()), "r", encoding="utf-8")

    # The parent decides when we stop (by closing stdin). A Ctrl-C in the terminal must not
    # tear a denoising loop apart from underneath it.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    name = os.environ.get("DREAMBOT_BACKEND", "mlx")
    try:
        backend = load_backend(name)
    except Exception as exc:
        traceback.print_exc()
        protocol_out.write(
            encode(
                Failed(
                    id=None,
                    kind=ErrorKind.CRASHED,
                    message=f"Could not start the '{name}' backend: {type(exc).__name__}: {exc}",
                    detail=traceback.format_exc(),
                )
            )
            + "\n"
        )
        protocol_out.flush()
        return 1
    return Worker(backend, protocol_out, protocol_in).run()


if __name__ == "__main__":
    sys.exit(main())
