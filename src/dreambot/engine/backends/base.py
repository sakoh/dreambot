"""Shared plumbing for worker backends.

A backend runs inside the worker subprocess and turns protocol requests into work. Backends
never touch the pipe directly: they report through a :class:`JobContext`.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Protocol

from pydantic import BaseModel

from dreambot.engine.protocol import (
    DescribeRequest,
    ErrorKind,
    GenerateRequest,
    LoadedModel,
    Op,
    StatusReport,
    TextRequest,
)


class BackendError(Exception):
    """A request failed in a way worth reporting to the user verbatim."""

    def __init__(self, message: str, kind: ErrorKind = ErrorKind.INTERNAL, detail: str | None = None):
        super().__init__(message)
        self.kind = kind
        self.detail = detail


class JobCancelled(BaseException):
    """Raised inside a running job when the parent cancelled it.

    Derives from ``BaseException`` so that it passes through ``except Exception`` blocks inside
    model code and unwinds the whole denoising loop.
    """


class JobContext:
    """Handle a backend uses to report on, and observe, the request it is running."""

    def __init__(self, request_id: str, emit: Callable[[BaseModel], None], cancel_event: threading.Event):
        self.id = request_id
        self._emit = emit
        self._cancel = cancel_event

    def emit(self, event: BaseModel) -> None:
        self._emit(event)

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def check_cancelled(self) -> None:
        if self._cancel.is_set():
            raise JobCancelled()


class Backend(Protocol):
    name: str

    def versions(self) -> dict[str, str]: ...

    def generate(self, req: GenerateRequest, ctx: JobContext) -> None: ...

    def describe(self, req: DescribeRequest, ctx: JobContext) -> list[str]: ...

    def text(self, req: TextRequest, ctx: JobContext) -> list[str]: ...

    def unload(self, model: str | None) -> None: ...

    def status(self, request_id: str) -> StatusReport: ...


def require_inputs(req: GenerateRequest, max_refs: int = 0) -> None:
    """Validate that a request carries the inputs its operation needs."""
    op = req.op
    if op in (Op.IMG2IMG, Op.UPSCALE, Op.CONTROL, Op.INPAINT) and not req.image:
        raise BackendError(f"'{op.value}' needs an input image.", ErrorKind.INVALID)
    if op is Op.INPAINT and not req.mask:
        raise BackendError("'inpaint' needs a mask image.", ErrorKind.INVALID)
    if op in (Op.EDIT, Op.REDUX):
        if not req.images:
            raise BackendError(f"'{op.value}' needs at least one reference image.", ErrorKind.INVALID)
        if max_refs and len(req.images) > max_refs:
            raise BackendError(
                f"{req.model} accepts at most {max_refs} reference image(s) for '{op.value}', got {len(req.images)}.",
                ErrorKind.INVALID,
            )
    if req.strength is not None and not 0.0 <= req.strength <= 1.0:
        raise BackendError("strength must be between 0 and 1.", ErrorKind.INVALID)
    if req.noise is not None and not 0.0 < req.noise <= 1.0:
        raise BackendError("noise must be greater than 0 and at most 1.", ErrorKind.INVALID)
    if req.stop_at is not None and not 0.0 < req.stop_at <= 1.0:
        raise BackendError("stop_at must be in (0, 1].", ErrorKind.INVALID)


__all__ = [
    "Backend",
    "BackendError",
    "JobCancelled",
    "JobContext",
    "LoadedModel",
    "require_inputs",
]
