"""What every chat frontend has in common.

A platform adapter only has to know how to *send*, *edit* and *delete* a message and how to turn
its incoming events into :class:`Incoming`, a button payload or a reaction. Everything else —
access control, interpreting text, tracking a job's status message, delivering results with the
right buttons, remembering which message shows which image — is done here, once, so the three
platforms cannot drift apart.

Capabilities differ per platform and are declared as class attributes: iMessage has no buttons
and cannot edit messages, so it gets typed shortcuts and one message per image instead.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from dreambot.core import render
from dreambot.core.actions import ENCODED_PREFIX, Action, ActionKind, decode_action, encode_action
from dreambot.core.commands import Commands, Reply, job_reply
from dreambot.core.jobs import EventType, Job, JobEvent, JobStatus
from dreambot.core.service import Dreambot, DreambotError
from dreambot.core.store import MessageLink
from dreambot.core.text import split_caption, split_text

log = logging.getLogger(__name__)

#: Rows of (label, payload).
Buttons = list[list[tuple[str, str]]]

#: Seconds between edits of a job's status message.
STATUS_INTERVAL = 2.5
#: How long a "reply with the details" question stays open.
PENDING_TTL = 10 * 60
PICK_PREFIX = "pick"

REACTION_ACTIONS = {
    "❤": ActionKind.FAVORITE,
    "♥": ActionKind.FAVORITE,
    "😍": ActionKind.FAVORITE,
    "🔥": ActionKind.FAVORITE,
    "👍": ActionKind.UPSCALE_SUBTLE,
    "🔁": ActionKind.VARY,
    "🔄": ActionKind.VARY,
    "‼": ActionKind.VARY,
    "✉": ActionKind.INFO,
    "❓": ActionKind.INFO,
    "❌": ActionKind.CANCEL,
    "👎": ActionKind.CANCEL,
}


@dataclass
class Incoming:
    """A message from a person, in platform-neutral form."""

    #: Platform user id (without the frontend prefix).
    user: str
    #: Platform conversation id (without the frontend prefix).
    conversation: str
    text: str = ""
    #: Local paths of images sent with the message.
    attachments: list[str] = field(default_factory=list)
    #: The conversation is with the bot alone, so plain text is a prompt.
    direct: bool = True
    #: Platform reference of the message this one replies to, if any.
    reply_to: str | None = None


@dataclass
class _Status:
    ref: str | None = None
    text: str = ""
    edited: float = 0.0


@dataclass
class _Pending:
    action: Action
    asked: float


class ChatFrontend:
    """Base class for platform adapters. Subclasses implement the three transport methods."""

    #: Prefix of user and channel ids, e.g. ``telegram``.
    name: str = "chat"
    #: The platform can show buttons under a message.
    has_buttons: bool = True
    #: The platform can edit a message after sending it (for live progress).
    can_edit: bool = True
    #: Send a job's images as one grid picture (True) or as separate pictures (False).
    grid_as_one_image: bool = True
    #: Acknowledge every job with a short message (for platforms without live progress).
    acknowledge: bool = False
    slash: str = "/"
    #: The longest caption the platform accepts under a picture, and the longest text message.
    #: Longer replies are sent in parts rather than refused or cut off by the platform.
    max_caption: int = 1000
    max_text: int = 4000

    def __init__(self, service: Dreambot):
        self.service = service
        self.commands = Commands(service)
        self._status: dict[str, _Status] = {}
        self._pending: dict[tuple[str, str], _Pending] = {}
        self._unsubscribe: Callable[[], None] | None = None

    # ------------------------------------------------------------- transport ----

    async def send(
        self,
        conversation: str,
        text: str = "",
        images: Sequence[str] = (),
        buttons: Buttons | None = None,
        *,
        reply_to: str | None = None,
        ask_reply: bool = False,
    ) -> list[str]:
        """Send a message. Returns one reference per message sent (one per image when images
        are sent separately), usable later with :meth:`edit`, :meth:`delete` and as a reply target."""
        raise NotImplementedError

    async def edit(self, conversation: str, ref: str, text: str, buttons: Buttons | None = None) -> None:
        raise NotImplementedError

    async def delete(self, conversation: str, ref: str) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------- lifecycle ----

    async def start(self) -> None:
        self._unsubscribe = self.service.subscribe(self._on_job_event)

    async def stop(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    def user_id(self, platform_user: str) -> str:
        return f"{self.name}:{platform_user}"

    def channel_id(self, conversation: str) -> str:
        return f"{self.name}:{conversation}"

    def _conversation_of(self, job: Job) -> str | None:
        prefix = f"{self.name}:"
        return job.channel[len(prefix) :] if job.channel.startswith(prefix) else None

    def _link(self, conversation: str, ref: str, job: Job, image: int | None) -> None:
        self.service.store.link_message(self.name, f"{conversation}/{ref}", job.id, image)

    def _lookup(self, conversation: str, ref: str | None) -> MessageLink | None:
        return self.service.store.find_message(self.name, f"{conversation}/{ref}") if ref else None

    # -------------------------------------------------------------- incoming ----

    async def handle_message(self, message: Incoming) -> None:
        """Entry point for an incoming message."""
        user, channel = self.user_id(message.user), self.channel_id(message.conversation)
        text = message.text.strip()
        key = (message.conversation, message.user)

        pending = self._pending.get(key)
        if pending is not None and time.monotonic() - pending.asked > PENDING_TTL:
            pending = None
            self._pending.pop(key, None)
        if pending is not None and text and not text.startswith(self.slash):
            # This message answers a question a button asked ("what should go there?").
            self._pending.pop(key, None)
            await self._complete(message, pending.action, text)
            return

        target = self._lookup(message.conversation, message.reply_to)
        replies = await self.commands.handle(
            user=user,
            channel=channel,
            text=text,
            attachments=message.attachments,
            direct=message.direct,
            target=target,
            slash=self.slash,
        )
        for reply in replies:
            await self._send_reply(message.conversation, reply)

    async def handle_button(self, user: str, conversation: str, payload: str) -> str | None:
        """Entry point for a button press. Returns a short notice to flash at the person, if any."""
        full_user, channel = self.user_id(user), self.channel_id(conversation)
        if not self.service.access.is_allowed(full_user):
            return self.service.access.refusal(full_user)

        if payload.startswith(f"{PICK_PREFIX}:"):
            # A numbered prompt under a /describe result.
            _, job_id, number = payload.split(":", 2)
            try:
                job = self.service.find_job(job_id)
                prompt = job.texts[int(number) - 1]
                await self.service.imagine(user=full_user, channel=channel, prompt=prompt)
            except (DreambotError, IndexError, ValueError) as exc:
                return str(exc) or "That prompt is no longer available."
            return None

        action = decode_action(payload)
        if action is None:
            return "That button is no longer valid."
        question = self._question_for(full_user, action)
        if question is not None:
            self._pending[(conversation, user)] = _Pending(action, time.monotonic())
            await self.send(conversation, question, ask_reply=True)
            return None
        return await self._run_action(full_user, channel, conversation, action)

    async def handle_reaction(self, user: str, conversation: str, ref: str, emoji: str) -> None:
        """Entry point for a reaction / tapback added to one of the bot's messages."""
        kind = REACTION_ACTIONS.get(emoji.replace("️", "")[:1]) or REACTION_ACTIONS.get(emoji.replace("️", ""))
        link = self._lookup(conversation, ref)
        full_user = self.user_id(user)
        if kind is None or link is None or not self.service.access.is_allowed(full_user):
            return
        action = Action(kind, link.job, link.image)
        notice = await self._run_action(full_user, self.channel_id(conversation), conversation, action)
        if notice:
            await self.send(conversation, notice)

    def _question_for(self, user: str, action: Action) -> str | None:
        """Some buttons need words before they can run. Returns the question to ask, or None."""
        kind = action.kind
        if kind is ActionKind.VARY_REGION:
            return (
                "Which part should change, and into what? Reply like:\n"
                "B2-D4: a red balloon   (cells — send `region` to see them)\n"
                "the sky: an aurora   (or describe the part)"
            )
        if kind is ActionKind.EDIT:
            return "What should change? Reply with an instruction, like: make it night"
        if kind is ActionKind.RETEXTURE:
            names = self.service.planner.retexture_loras()
            example = names[0] if names else "pixel-art"
            listing = f"\nLoRAs: {', '.join(names)}" if names else ""
            return (
                "Retexture draws it again in another look and keeps its layout. Reply with the look: "
                f"a LoRA like `--lora {example}`, a new prompt, or both.{listing}"
            )
        if kind is ActionKind.ZOOM and action.value == "custom":
            return "How far out? Reply with a number between 1 and 2 (for example 1.3). You can add a new prompt after it."
        if (
            kind in (ActionKind.VARY, ActionKind.VARY_SUBTLE, ActionKind.VARY_STRONG)
            and self.service.resolved_settings(user).remix
        ):
            try:
                current = self.service.find_job(action.job or "").prompt
            except DreambotError:
                current = ""
            return f"Remix: reply with the prompt for these variations.\nCurrent: {current}"
        return None

    async def _complete(self, message: Incoming, action: Action, text: str) -> None:
        """Finish a button action with the words the person replied with."""
        kind = action.kind
        if kind is ActionKind.VARY_REGION:
            where, separator, what = text.partition(":")
            if not separator or not what.strip():
                await self.send(
                    message.conversation,
                    "⚠️ I need both parts, like `B2-D4: a red balloon`. Press the button to try again.",
                )
                return
            action = Action(kind, action.job, action.index, where.strip(), what.strip())
        elif kind is ActionKind.ZOOM:
            number, _, prompt = text.partition(" ")
            try:
                factor = float(number.lower().rstrip("x"))
            except ValueError:
                factor = 0.0
            if not 1.0 < factor <= 2.0:
                await self.send(
                    message.conversation, "⚠️ Custom zoom takes a number above 1 and up to 2, like 1.5."
                )
                return
            action = Action(kind, action.job, action.index, f"{factor:g}", prompt.strip() or None)
        else:
            action = Action(kind, action.job, action.index, action.value, text)
        notice = await self._run_action(
            self.user_id(message.user), self.channel_id(message.conversation), message.conversation, action
        )
        if notice:
            await self.send(message.conversation, notice)

    async def _run_action(self, user: str, channel: str, conversation: str, action: Action) -> str | None:
        try:
            result = await self.service.act(user=user, channel=channel, action=action)
        except DreambotError as exc:
            return f"⚠️ {exc}"
        if result.jobs:
            return None
        if action.kind is ActionKind.SELECT and result.job and result.image:
            caption = f"Image {result.image.index} of #{result.job.id} — {render.job_title(result.job, 160)}"
            await self._send_reply(
                conversation,
                Reply(text=caption, images=[result.image.path], job=result.job, image=result.image),
            )
            return None
        if action.kind is ActionKind.INFO and result.message:
            await self.send_text(conversation, result.message)
            return None
        return result.message

    # -------------------------------------------------------------- outgoing ----

    def _buttons_for(self, reply: Reply) -> Buttons | None:
        if not self.has_buttons or reply.job is None or reply.job.status is not JobStatus.DONE:
            return None
        job = reply.job
        if job.texts:
            row = [(str(n), f"{PICK_PREFIX}:{job.id}:{n}") for n in range(1, len(job.texts) + 1)]
            return [row]
        if not job.images or not reply.images:
            return None
        rows = render.image_actions(job, reply.image) if reply.image is not None else render.grid_actions(job)
        return [[(label, encode_action(action)) for label, action in row] for row in rows]

    async def _send_reply(self, conversation: str, reply: Reply) -> None:
        if not reply.text and not reply.images:
            return  # nothing to say: the jobs it queued report through job events
        buttons = self._buttons_for(reply)
        text, overflow = reply.text, ""
        if reply.images:
            # What does not fit under the picture follows as a message of its own.
            text, overflow = split_caption(text, self.max_caption)
        elif len(text) > self.max_text:
            *earlier, text = split_text(text, self.max_text)
            for part in earlier:
                await self.send(conversation, part)
        refs = await self.send(conversation, text, reply.images, buttons)
        if reply.job is not None and reply.images:
            index = reply.image.index if reply.image else None
            for ref in refs:
                self._link(conversation, ref, reply.job, index)
        if overflow:
            await self.send_text(conversation, overflow)

    async def send_text(self, conversation: str, text: str) -> None:
        """Send text of any length, in as many messages as the platform needs."""
        for part in split_text(text, self.max_text):
            await self.send(conversation, part)

    async def deliver(self, job: Job) -> None:
        """Send a finished job to its conversation."""
        conversation = self._conversation_of(job)
        if conversation is None:
            return
        reply = job_reply(job)
        if job.status is JobStatus.DONE and len(job.images) > 1 and not self.grid_as_one_image:
            # No buttons and no grid: one message per image, so each can be replied to or
            # reacted to on its own, followed by the caption.
            for image in job.images:
                for ref in await self.send(conversation, "", [image.path]):
                    self._link(conversation, ref, job, image.index)
            hint = "Reply to a picture to change it, or send U1–U4 · V1–V4 · R (more: /help)"
            await self.send_text(conversation, f"{render.job_caption(job)}\n{hint}")
            return
        await self._send_reply(conversation, reply)

    # ------------------------------------------------------------ job events ----

    async def _on_job_event(self, event: JobEvent) -> None:
        job = event.job
        conversation = self._conversation_of(job)
        if conversation is None:
            return
        try:
            await self._render_event(event, conversation)
        except Exception:
            log.exception("%s: could not render %s for job %s", self.name, event.type, job.id)

    async def _render_event(self, event: JobEvent, conversation: str) -> None:
        job = event.job
        kind = event.type
        if kind is EventType.QUEUED:
            text = render.queued_text(job, event.position)
            if self.can_edit:
                refs = await self.send(conversation, f"⏳ {text}", buttons=self._cancel_button(job))
                self._status[job.id] = _Status(
                    ref=refs[0] if refs else None, text=text, edited=time.monotonic()
                )
            elif self.acknowledge or event.position > 0:
                await self.send(conversation, f"🎨 {text}")
        elif kind in (EventType.STARTED, EventType.LOADING, EventType.THINKING, EventType.PROGRESS):
            if not self.can_edit:
                return
            status = self._status.get(job.id)
            if status is None or status.ref is None:
                return
            text = render.progress_text(
                job, event.fraction, loading=kind is EventType.LOADING, thinking=kind is EventType.THINKING
            )
            now = time.monotonic()
            if text == status.text or (kind is EventType.PROGRESS and now - status.edited < STATUS_INTERVAL):
                return
            status.text, status.edited = text, now
            await self.edit(conversation, status.ref, f"⏳ {text}", self._cancel_button(job))
        elif kind is EventType.FINISHED:
            status = self._status.pop(job.id, None)
            if status is not None and status.ref is not None:
                with _suppress_errors():
                    await self.delete(conversation, status.ref)
            await self.deliver(job)
        elif kind in (EventType.FAILED, EventType.CANCELLED):
            status = self._status.pop(job.id, None)
            text = render.failure_text(job)
            if status is not None and status.ref is not None and self.can_edit:
                await self.edit(conversation, status.ref, f"⚠️ {text}")
            else:
                await self.send(conversation, f"⚠️ {text}")

    def _cancel_button(self, job: Job) -> Buttons | None:
        if not self.has_buttons:
            return None
        return [[("Cancel", encode_action(Action(ActionKind.CANCEL, job.id)))]]


class _suppress_errors:
    """Swallow and log transport errors for housekeeping calls that must not break delivery."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None and not isinstance(exc, asyncio.CancelledError):
            log.debug("ignored transport error: %s", exc)
            return True
        return False


__all__ = ["ENCODED_PREFIX", "PICK_PREFIX", "Buttons", "ChatFrontend", "Incoming"]
