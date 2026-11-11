"""iMessage frontend: text your Mac from your iPhone.

iMessage has no bots, buttons or message editing, so this frontend leans on what it does have:

* any text in the conversation is a prompt;
* every image is sent as its own message, so you can **reply to a picture** ("make it night",
  ``upscale``, ``V``) or **tapback** it (❤ favourite, 👍 upscale, ‼ variations);
* typed shortcuts (``U2``, ``V3``, ``zoom 2x``, ``pan left`` ...) cover everything else.

Two ways to run it (``imessage.mode`` in the config):

``self``
    Messages on this Mac uses your own Apple Account. You text *yourself*; the note-to-self
    thread is your conversation with Dreambot. You can also summon it inside any other
    conversation by starting a message with the trigger (``/imagine ...``), and people you have
    allowed can do the same when they text you. Dreambot never answers anyone else.

``dedicated``
    Messages on this Mac is signed into an Apple Account that belongs to the bot. People text
    that account like any contact, and everything they send is for the bot.

In ``self`` mode the bot's replies appear in the database looking just like messages you typed,
so every outgoing message is recorded in a ledger and recognised when it comes back around.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from dreambot.core.access import LOCAL_IMESSAGE_USER, normalize_user
from dreambot.core.service import Dreambot
from dreambot.frontends.base import Buttons, ChatFrontend, Incoming
from dreambot.frontends.imessage.chatdb import ChatDB, ChatDBError, Row
from dreambot.frontends.imessage.sender import Sender, SendError

log = logging.getLogger(__name__)

#: How long an outgoing message is remembered so its copies in the database are recognised.
LEDGER_TTL = 10 * 60
#: A note-to-self message can show up twice (sent copy and received copy) within this window.
ECHO_WINDOW = 20.0
#: How many polls to wait for a message row to be attached to its chat.
JOIN_RETRIES = 20
ATTACHMENT_WAIT = 30.0
FILE_REF = "file:"
#: Where the place reached in the database is kept between runs, and for how long it stays
#: worth going back to: a restart picks up what arrived in the meantime, but a bot that was
#: off for an hour does not wake up and answer an hour of old messages.
CURSOR_KEY = "imessage.cursor"
RESUME_WINDOW = 10 * 60
CURSOR_SAVE_INTERVAL = 30.0


def _normalize_text(text: str) -> str:
    return " ".join(text.split()).casefold()


def _handle_key(handle: str | None) -> str:
    """A handle reduced to a comparable form (the id part of ``normalize_user``)."""
    return normalize_user(f"imessage:{handle or ''}").partition(":")[2]


@dataclass
class _Outgoing:
    chat: str
    text: str
    at: float
    #: A sent text can surface twice in a note-to-self thread (sent copy, received copy).
    remaining: int = 2


@dataclass
class _Handled:
    key: tuple
    from_me: bool
    at: float


class IMessageFrontend(ChatFrontend):
    name = "imessage"
    has_buttons = False
    can_edit = False
    grid_as_one_image = False
    acknowledge = True
    max_caption = 3000
    max_text = 6000

    def __init__(
        self,
        service: Dreambot,
        *,
        db: ChatDB | None = None,
        sender: Sender | None = None,
        mode: str | None = None,
        trigger: str | None = None,
        poll_seconds: float | None = None,
    ):
        super().__init__(service)
        settings = service.config.imessage
        self.mode = mode or settings.mode
        self.trigger = trigger if trigger is not None else settings.trigger
        self.poll_seconds = poll_seconds if poll_seconds is not None else settings.poll_seconds
        self.db = db or ChatDB()
        self.sender = sender or Sender()
        self._cursor = 0
        #: (cursor, monotonic time) as last written to the store.
        self._saved_cursor: tuple[int, float] = (0, 0.0)
        self._task: asyncio.Task[None] | None = None
        self._retry: dict[int, int] = {}
        self._outgoing: deque[_Outgoing] = deque()
        self._staged: dict[str, float] = {}
        self._handled: deque[_Handled] = deque(maxlen=200)
        #: Our own pictures seen in the database before the base class had tied them to a job:
        #: staged name -> (chat guid, message guid).
        self._seen_files: dict[str, tuple[str, str]] = {}
        #: The other party of each one-to-one chat, for addressing it when its id is not enough.
        self._partners: dict[str, str] = {}
        #: Addresses of this Mac's own account, learned from the database as messages flow.
        self._own: set[str] = set()

    # ------------------------------------------------------------- lifecycle ----

    async def start(self) -> None:
        self.db.open()  # raises ChatDBError with instructions when Full Disk Access is missing
        self._cursor = self._starting_point(self.db.max_rowid())
        self._saved_cursor = (self._cursor, time.monotonic())
        self.db.changed()
        await super().start()
        with contextlib.suppress(Exception):
            self.sender.cleanup()
        self._task = asyncio.create_task(self._run(), name="dreambot-imessage")
        log.info("imessage: watching for messages (%s mode)", self.mode)
        if self.db.missing_optional:
            log.warning(
                "imessage: this macOS has no %s; tapbacks or replies to pictures may not be recognised",
                ", ".join(self.db.missing_optional),
            )
        self._permission = asyncio.create_task(self._check_permission(), name="dreambot-imessage-permission")

    def _starting_point(self, newest: int) -> int:
        """Where to start reading: the newest message, or where a recent run left off.

        Only the dedicated mode resumes. There, messages this Mac sent are never acted on; in a
        note-to-self thread the bot's own last words, read back without the ledger that
        recognises them, would look like the owner talking.
        """
        saved = self.service.store.get_state(CURSOR_KEY, None)
        if self.mode != "dedicated" or not isinstance(saved, dict):
            return newest
        try:
            rowid, at = int(saved["rowid"]), float(saved["at"])
        except (KeyError, TypeError, ValueError):
            return newest
        if time.time() - at > RESUME_WINDOW or not 0 < rowid <= newest:
            return newest
        if rowid < newest:
            log.info(
                "imessage: picking up %d message row(s) that arrived while Dreambot was restarting",
                newest - rowid,
            )
        return rowid

    def _save_cursor(self, *, force: bool = False) -> None:
        cursor, at = self._saved_cursor
        if not force and cursor == self._cursor and time.monotonic() - at < CURSOR_SAVE_INTERVAL:
            return
        self.service.store.set_state(CURSOR_KEY, {"rowid": self._cursor, "at": time.time()})
        self._saved_cursor = (self._cursor, time.monotonic())

    async def _check_permission(self) -> None:
        """Ask for the right to control Messages now, not in the middle of the first reply."""
        try:
            accounts = await self.sender.check()
        except SendError as exc:
            log.error("imessage: cannot send replies yet. %s", exc)
            return
        except Exception:
            log.exception("imessage: could not check whether Messages can be controlled")
            return
        if accounts is None:
            return
        if accounts == 0:
            log.error("imessage: Messages is not signed in to iMessage, so nothing can be sent or received")
        else:
            log.info(
                "imessage: allowed to send through Messages (%s iMessage account(s))",
                accounts if accounts > 0 else "?",
            )

    async def stop(self) -> None:
        for task in (self._task, getattr(self, "_permission", None)):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        self._task = None
        with contextlib.suppress(Exception):
            self._save_cursor(force=True)
        self.db.close()
        await super().stop()

    async def _run(self) -> None:
        failures = 0
        while True:
            try:
                await self.poll_once()
                failures = 0
                self._save_cursor()
            except asyncio.CancelledError:
                raise
            except ChatDBError as exc:
                failures += 1
                log.error("imessage: %s", exc)
            except Exception:
                failures += 1
                log.exception("imessage: polling failed")
            await asyncio.sleep(min(60.0, self.poll_seconds * (2 ** min(failures, 6))))

    # --------------------------------------------------------------- polling ----

    async def poll_once(self) -> int:
        """Process messages that arrived since the last poll. Returns how many rows were read."""
        retry_ids = list(self._retry)
        if not self.db.changed() and not retry_ids:
            return 0
        rows = self.db.rows_by_id(retry_ids) + self.db.rows_after(self._cursor)
        for row in rows:
            self._cursor = max(self._cursor, row.rowid)
            if row.chat_guid is None:
                # The row linking this message to its chat has not been written yet.
                attempts = self._retry.get(row.rowid, 0) + 1
                if attempts <= JOIN_RETRIES:
                    self._retry[row.rowid] = attempts
                else:
                    self._retry.pop(row.rowid, None)
                continue
            self._retry.pop(row.rowid, None)
            try:
                await self._process(row)
            except Exception:
                log.exception("imessage: could not handle message %s", row.rowid)
        self._expire()
        return len(rows)

    def _expire(self) -> None:
        now = time.monotonic()
        while self._outgoing and now - self._outgoing[0].at > LEDGER_TTL:
            self._outgoing.popleft()
        for name in [name for name, at in self._staged.items() if now - at > LEDGER_TTL]:
            del self._staged[name]

    def _is_ours(self, row: Row) -> bool:
        """Recognise a message Dreambot itself sent, and remember which image it carries."""
        ours = False
        for attachment in row.attachments:
            if attachment.name and attachment.name in self._staged:
                ours = True
                link = self._lookup(row.chat_guid or "", f"{FILE_REF}{attachment.name}")
                if link is not None:
                    # From now on a reply or tapback on this message finds its image.
                    self.service.store.link_message(
                        self.name, f"{row.chat_guid}/{row.guid}", link.job, link.image
                    )
                elif row.chat_guid:
                    # Read back before the picture was tied to its job (Messages writes the row
                    # while the send is still in progress): tie the two together when it is.
                    self._seen_files[attachment.name] = (row.chat_guid, row.guid)
        if ours:
            return True
        if row.text:
            wanted = _normalize_text(row.text)
            for entry in self._outgoing:
                if entry.remaining > 0 and entry.chat == row.chat_guid and entry.text == wanted:
                    entry.remaining -= 1
                    return True
        return False

    def _is_echo(self, row: Row, key: tuple) -> bool:
        """In a note-to-self thread the same message arrives as a sent and a received copy."""
        now = time.monotonic()
        for seen in list(self._handled):
            if now - seen.at > ECHO_WINDOW:
                continue
            if seen.key == key and seen.from_me != row.is_from_me:
                self._handled.remove(seen)
                return True
        self._handled.append(_Handled(key, row.is_from_me, now))
        return False

    def _is_self_chat(self, row: Row) -> bool:
        identifier = _handle_key(row.chat_identifier)
        if row.own_address:
            self._own.add(_handle_key(row.own_address))
        return bool(identifier) and not row.is_group and identifier in self._own

    async def _process(self, row: Row) -> None:
        chat = row.chat_guid
        assert chat is not None
        if not row.is_group and row.handle:
            self._partners[chat] = row.handle
        if self._is_ours(row):
            return
        self_chat = self._is_self_chat(row)

        if self.mode == "dedicated":
            if row.is_from_me:
                return
            user, direct = _handle_key(row.handle), not row.is_group
        elif self_chat:
            user, direct = LOCAL_IMESSAGE_USER, True
        elif row.is_from_me:
            user, direct = LOCAL_IMESSAGE_USER, False  # the owner, inside someone else's conversation
        else:
            user, direct = _handle_key(row.handle), False
        if not user:
            return

        if row.is_tapback:
            if (
                row.reaction
                and row.reaction_target
                and not self._is_echo(row, (chat, "tapback", row.reaction_target, row.reaction))
            ) and self.service.access.is_allowed(self.user_id(user)):
                log.info("imessage: a tapback")
                await self.handle_reaction(user, chat, row.reaction_target, row.reaction)
            return

        text = row.text
        images = [attachment for attachment in row.attachments if attachment.is_image]
        if not text and not images:
            return
        if self_chat and self._is_echo(
            row, (chat, _normalize_text(text), tuple(sorted(a.name or "" for a in images)))
        ):
            return
        if not direct and not text.startswith(self.trigger):
            return  # ordinary conversation with someone: none of the bot's business
        full_user = self.user_id(user)
        if not self.service.access.is_allowed(full_user) and not text.lower().startswith("/pair"):
            # Never answer strangers: in self mode the reply would come from the owner's own
            # account, and in any mode it would confirm a live number to a spammer.
            log.info(
                "imessage: ignoring a message from someone who is not allowed (send `/pair <code>` to claim the bot)"
            )
            return

        # The log says that something arrived and what kind of thing, never what it said.
        log.info(
            "imessage: a %s%s%s",
            "message" if text else "picture",
            f" with {len(images)} picture(s)" if text and images else "",
            " (a reply)" if row.reply_to else "",
        )
        paths = await self._collect(images)
        if images and not paths:
            log.warning("imessage: the attached picture never finished arriving from Messages")
            await self.send(chat, "⚠️ I couldn't get that image from Messages. Try sending it again.")
            return
        await self.handle_message(
            Incoming(
                user=user,
                conversation=chat,
                text=text,
                attachments=paths,
                direct=direct,
                reply_to=row.reply_to,
            )
        )

    def _link(self, conversation: str, ref: str, job, image: int | None) -> None:
        super()._link(conversation, ref, job, image)
        if ref.startswith(FILE_REF):
            seen = self._seen_files.pop(ref[len(FILE_REF) :], None)
            if seen is not None:
                self.service.store.link_message(self.name, f"{seen[0]}/{seen[1]}", job.id, image)

    async def _guids_of_sent(self, name: str) -> list[str]:
        """The ids Messages gave the picture just sent, which replies and tapbacks refer to."""
        for _ in range(10):
            guids = self.db.guids_of_attachment(name)
            if guids:
                return guids
            await asyncio.sleep(0.2)
        return []

    async def _collect(self, attachments: Sequence) -> list[str]:
        """Wait for incoming image files to finish downloading and return their paths."""
        paths: list[str] = []
        for attachment in attachments:
            if not attachment.path:
                continue
            path = Path(attachment.path)
            deadline = time.monotonic() + ATTACHMENT_WAIT
            last_size = -1
            while time.monotonic() < deadline:
                try:
                    size = path.stat().st_size
                except OSError:
                    size = -1
                complete = size > 0 and (attachment.total_bytes is None or size >= attachment.total_bytes)
                if complete and size == last_size:
                    paths.append(str(path))
                    break
                last_size = size
                await asyncio.sleep(0.25)
        return paths

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
        partner = self._partners.get(conversation)
        refs: list[str] = []
        try:
            # Pictures first, words after: the caption then sits underneath what it describes.
            for image in images:
                staged = self.sender.stage(image)
                # Registered before the send: Messages writes its row while the send is still
                # running, and the row has to be recognised as ours whenever it is read.
                self._staged[staged.name] = time.monotonic()
                await self.sender.send_staged(conversation, partner, staged)
                # A reply or a tapback names the message it is on, so the picture is recorded under
                # the message ids Messages gave it. The file name is kept as well: a copy that
                # shows up later (the received half of a note-to-self) is matched through it.
                refs.append(f"{FILE_REF}{staged.name}")
                refs.extend(await self._guids_of_sent(staged.name))
            if text:
                self._outgoing.append(_Outgoing(conversation, _normalize_text(text), time.monotonic()))
                await self.sender.send_text(conversation, partner, text)
            log.info(
                "imessage: sent %s",
                " and ".join(
                    part
                    for part in (f"{len(images)} picture(s)" if images else "", "a message" if text else "")
                    if part
                )
                or "nothing",
            )
        except SendError as exc:
            log.error("imessage: could not send (%s)", exc)
        return refs

    async def edit(self, conversation: str, ref: str, text: str, buttons: Buttons | None = None) -> None:
        return None  # iMessage cannot edit; the base class never asks (can_edit is False)

    async def delete(self, conversation: str, ref: str) -> None:
        return None
