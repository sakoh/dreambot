"""Telegram frontend.

Connects with long polling, so it works from a Mac behind a home router: there is no webhook,
no public address and no port to open. In a private chat with the bot, any message is a
prompt. In groups, use ``/imagine`` (Telegram only shows bots the commands and replies there).

Everything interesting happens in :class:`~dreambot.frontends.base.ChatFrontend`; this file
only translates between Telegram's objects and that base class.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image
from telegram import (
    BotCommand,
    ForceReply,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaDocument,
    InputMediaPhoto,
    Message,
    Update,
)
from telegram.constants import ChatType
from telegram.error import BadRequest, TelegramError
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    MessageReactionHandler,
    filters,
)
from telegram.request import BaseRequest

from dreambot.core.commands import Reply
from dreambot.core.imaging import flatten
from dreambot.core.service import Dreambot, DreambotError
from dreambot.frontends.base import Buttons, ChatFrontend, Incoming

log = logging.getLogger(__name__)

#: Telegram shows a photo at up to 1280px and rejects photos over 10 MB or 10,000px combined.
PHOTO_MAX_BYTES = 9 * 1024 * 1024
PHOTO_MAX_DIMENSIONS = 9500
#: Messages of one album arrive separately; wait this long for the rest before acting.
ALBUM_WAIT = 0.8
FILE_PREFIX = "tg:file"

COMMANDS = [
    ("imagine", "Create images from a prompt"),
    ("describe", "Write prompts for an image you attach"),
    ("blend", "Blend 2–5 attached images"),
    ("edit", "Change an attached image by instruction"),
    ("shorten", "Suggest shorter versions of a prompt"),
    ("settings", "View or change your settings"),
    ("models", "List the installed models"),
    ("show", "Show a job again"),
    ("info", "Queue and usage"),
    ("cancel", "Cancel your running jobs"),
    ("help", "How to use Dreambot"),
]


@dataclass
class _Album:
    messages: list[Message] = field(default_factory=list)
    timer: asyncio.TimerHandle | None = None


class TelegramFrontend(ChatFrontend):
    name = "telegram"
    has_buttons = True
    can_edit = True
    grid_as_one_image = True
    # Telegram's own limits are 1024 characters for a caption and 4096 for a message.
    max_caption = 1000
    max_text = 4000

    def __init__(
        self, service: Dreambot, token: str, *, request: BaseRequest | None = None, polling: bool = True
    ):
        super().__init__(service)
        builder = ApplicationBuilder().token(token).concurrent_updates(True)
        if request is not None:
            # Tests inject an offline transport here.
            builder = builder.request(request).get_updates_request(request)
        self.app: Application = builder.build()
        self._polling = polling
        self._albums: dict[str, _Album] = {}
        self._downloads = service.config.scratch_dir / "telegram"
        self.app.add_handler(CallbackQueryHandler(self._on_callback))
        self.app.add_handler(MessageReactionHandler(self._on_reaction))
        self.app.add_handler(MessageHandler(filters.ALL & ~filters.StatusUpdate.ALL, self._on_message))
        self.app.add_error_handler(self._on_error)

    # ------------------------------------------------------------- lifecycle ----

    async def start(self) -> None:
        await super().start()
        await self.app.initialize()
        await self.app.start()
        with contextlib.suppress(TelegramError):
            await self.app.bot.set_my_commands([BotCommand(name, text) for name, text in COMMANDS])
        if self._polling and self.app.updater is not None:
            await self.app.updater.start_polling(
                allowed_updates=[Update.MESSAGE, Update.CALLBACK_QUERY, Update.MESSAGE_REACTION],
                drop_pending_updates=False,
            )
        log.info("telegram: connected as @%s", self.app.bot.username)

    async def stop(self) -> None:
        if self.app.updater is not None and self.app.updater.running:
            await self.app.updater.stop()
        if self.app.running:
            await self.app.stop()
        await self.app.shutdown()
        await super().stop()

    @staticmethod
    async def _on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        log.error("telegram: handler error", exc_info=context.error)

    # ------------------------------------------------------------- addresses ----

    @staticmethod
    def _conversation(message: Message) -> str:
        """Chat id, plus the topic id inside forum groups so each topic is its own conversation."""
        if message.is_topic_message and message.message_thread_id:
            return f"{message.chat_id}:{message.message_thread_id}"
        return str(message.chat_id)

    @staticmethod
    def _address(conversation: str) -> tuple[int, int | None]:
        chat, _, thread = conversation.partition(":")
        return int(chat), int(thread) if thread else None

    # ------------------------------------------------------------- transport ----

    @staticmethod
    def _keyboard(buttons: Buttons | None) -> InlineKeyboardMarkup | None:
        if not buttons:
            return None
        return InlineKeyboardMarkup(
            [
                [InlineKeyboardButton(label, callback_data=payload) for label, payload in row]
                for row in buttons
            ]
        )

    def _photo(self, path: str) -> Path:
        """A version of an image Telegram will accept as a photo (the original is offered as a file)."""
        source = Path(path)
        with Image.open(source) as image:
            width, height = image.size
            fits = source.stat().st_size <= PHOTO_MAX_BYTES and width + height <= PHOTO_MAX_DIMENSIONS
            if fits and image.mode in ("RGB", "L"):
                return source
            preview = flatten(image)
        preview.thumbnail((2560, 2560), Image.LANCZOS)
        self._downloads.mkdir(parents=True, exist_ok=True)
        target = self._downloads / f"preview-{uuid.uuid4().hex[:12]}.jpg"
        preview.save(target, quality=90)
        return target

    def _buttons_for(self, reply: Reply) -> Buttons | None:
        """Add an "original file" button: Telegram recompresses photos, and people want the PNG."""
        buttons = super()._buttons_for(reply)
        if buttons and reply.job is not None and reply.job.images and reply.images:
            label = "⬇️ Original" if reply.image else "⬇️ Originals"
            index = reply.image.index if reply.image else ""
            buttons = [*buttons, [(label, f"{FILE_PREFIX}:{reply.job.id}:{index}")]]
        return buttons

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
        bot = self.app.bot
        chat_id, thread = self._address(conversation)
        common = {"chat_id": chat_id, "message_thread_id": thread}
        if reply_to:
            common["reply_to_message_id"] = int(reply_to)
        if not images:
            markup = ForceReply(selective=True) if ask_reply else self._keyboard(buttons)
            # The base class already splits long replies; the slices here are a last line of
            # defence, because Telegram rejects an over-long message outright.
            message = await bot.send_message(text=(text or "…")[:4096], reply_markup=markup, **common)
            return [str(message.message_id)]
        caption = text[:1024] or None
        if len(images) == 1:
            markup = self._keyboard(buttons)
            with self._photo(images[0]).open("rb") as handle:
                message = await bot.send_photo(photo=handle, caption=caption, reply_markup=markup, **common)
            return [str(message.message_id)]
        media = []
        handles = [self._photo(path).open("rb") for path in images[:10]]
        try:
            for position, handle in enumerate(handles):
                media.append(InputMediaPhoto(handle, caption=caption if position == 0 else None))
            messages = await bot.send_media_group(media=media, **common)
        finally:
            for handle in handles:
                handle.close()
        return [str(message.message_id) for message in messages]

    async def edit(self, conversation: str, ref: str, text: str, buttons: Buttons | None = None) -> None:
        chat_id, _ = self._address(conversation)
        try:
            await self.app.bot.edit_message_text(
                text=text[:4096], chat_id=chat_id, message_id=int(ref), reply_markup=self._keyboard(buttons)
            )
        except BadRequest as exc:
            if "not modified" not in str(exc).lower():
                raise

    async def delete(self, conversation: str, ref: str) -> None:
        chat_id, _ = self._address(conversation)
        await self.app.bot.delete_message(chat_id=chat_id, message_id=int(ref))

    # -------------------------------------------------------------- incoming ----

    async def _on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is None or update.effective_user is None or update.effective_user.is_bot:
            return
        if message.media_group_id:
            # An album: collect its parts, then handle them as one message.
            album = self._albums.setdefault(message.media_group_id, _Album())
            album.messages.append(message)
            if album.timer is not None:
                album.timer.cancel()
            loop = asyncio.get_running_loop()
            album.timer = loop.call_later(
                ALBUM_WAIT, lambda group=message.media_group_id: loop.create_task(self._flush_album(group))
            )
            return
        await self._dispatch([message])

    async def _flush_album(self, group: str) -> None:
        album = self._albums.pop(group, None)
        if album is not None and album.messages:
            await self._dispatch(sorted(album.messages, key=lambda m: m.message_id))

    async def _dispatch(self, messages: list[Message]) -> None:
        first = messages[0]
        user = first.from_user
        if user is None:
            return
        text = next((m.text or m.caption for m in messages if m.text or m.caption), "") or ""
        attachments: list[str] = []
        for message in messages:
            try:
                path = await self._download(message)
            except TelegramError as exc:
                log.warning("telegram: could not download an attachment: %s", exc)
                await self.send(
                    self._conversation(first),
                    "⚠️ I couldn't download that image (Telegram limits bots to 20 MB files).",
                )
                return
            if path:
                attachments.append(path)
        reply = first.reply_to_message
        await self.handle_message(
            Incoming(
                user=str(user.id),
                conversation=self._conversation(first),
                text=text,
                attachments=attachments,
                direct=first.chat.type == ChatType.PRIVATE,
                # In forum topics every message "replies" to the topic's root; that is not a reply.
                reply_to=str(reply.message_id)
                if reply and reply.message_id != first.message_thread_id
                else None,
            )
        )

    async def _download(self, message: Message) -> str | None:
        source = None
        suffix = ".jpg"
        if message.photo:
            source = message.photo[-1]
        elif message.document and (message.document.mime_type or "").startswith("image/"):
            source = message.document
            suffix = Path(message.document.file_name or "image").suffix or ".img"
        if source is None:
            return None
        self._downloads.mkdir(parents=True, exist_ok=True)
        target = self._downloads / f"in-{uuid.uuid4().hex[:12]}{suffix}"
        file = await source.get_file()
        await file.download_to_drive(target)
        return str(target)

    async def _on_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None or query.message is None or query.data is None:
            return
        conversation = self._conversation(query.message)  # type: ignore[arg-type]
        if query.data.startswith(f"{FILE_PREFIX}:"):
            notice = await self._send_originals(str(query.from_user.id), conversation, query.data)
        else:
            notice = await self.handle_button(str(query.from_user.id), conversation, query.data)
        with contextlib.suppress(TelegramError):
            if notice:
                await query.answer(text=notice[:190], show_alert=notice.startswith("⚠️") or len(notice) > 60)
            else:
                await query.answer()

    async def _send_originals(self, user: str, conversation: str, payload: str) -> str | None:
        """Send the untouched PNG(s) of a job as files."""
        if not self.service.access.is_allowed(self.user_id(user)):
            return self.service.access.refusal(self.user_id(user))
        _, _, job_id, index = ([*payload.split(":"), ""])[:4]
        try:
            job = self.service.find_job(job_id)
            images = [job.image(int(index))] if index else job.images
        except (DreambotError, KeyError, ValueError):
            return "⚠️ Those files are no longer available."
        paths = [Path(image.path) for image in images if Path(image.path).is_file()]
        if not paths:
            return "⚠️ Those files are no longer on disk."
        chat_id, thread = self._address(conversation)
        handles = [path.open("rb") for path in paths[:10]]
        try:
            if len(handles) == 1:
                await self.app.bot.send_document(
                    chat_id=chat_id, document=handles[0], filename=paths[0].name, message_thread_id=thread
                )
            else:
                media = [
                    InputMediaDocument(handle, filename=path.name)
                    for handle, path in zip(handles, paths, strict=False)
                ]
                await self.app.bot.send_media_group(chat_id=chat_id, media=media, message_thread_id=thread)
        finally:
            for handle in handles:
                handle.close()
        return None

    async def _on_reaction(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        reaction = update.message_reaction
        if reaction is None or reaction.user is None:
            return
        before = {getattr(item, "emoji", None) for item in reaction.old_reaction}
        for item in reaction.new_reaction:
            emoji = getattr(item, "emoji", None)
            if emoji and emoji not in before:
                await self.handle_reaction(
                    str(reaction.user.id), str(reaction.chat.id), str(reaction.message_id), emoji
                )
