"""The iMessage frontend against a synthetic Messages database.

Nothing here touches the real ``~/Library/Messages`` or the Messages app: the database is built
in a temp directory with the same tables and columns, and "sending" is a stand-in that writes
the rows Messages would write.
"""

from __future__ import annotations

import asyncio
import itertools
import shutil
import sqlite3
import subprocess
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from PIL import Image

from dreambot.core.jobs import JobKind, JobStatus
from dreambot.core.service import Dreambot
from dreambot.engine.client import Engine
from dreambot.frontends.imessage.bot import IMessageFrontend
from dreambot.frontends.imessage.chatdb import (
    ChatDB,
    ChatDBError,
    apple_time,
    clean_associated_guid,
    decode_attributed_body,
)
from dreambot.frontends.imessage.sender import SCRIPT, Sender, SendError
from tests.test_service import make_config

ME = "+15550001111"
FRIEND = "+15550002222"
STRANGER = "+15550003333"

SCHEMA = """
CREATE TABLE handle (ROWID INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT, service TEXT);
CREATE TABLE chat (ROWID INTEGER PRIMARY KEY AUTOINCREMENT, guid TEXT, chat_identifier TEXT, style INTEGER, service_name TEXT);
CREATE TABLE message (
    ROWID INTEGER PRIMARY KEY AUTOINCREMENT, guid TEXT, text TEXT, attributedBody BLOB, handle_id INTEGER DEFAULT 0,
    service TEXT, date INTEGER, is_from_me INTEGER DEFAULT 0, cache_has_attachments INTEGER DEFAULT 0,
    associated_message_guid TEXT, associated_message_type INTEGER DEFAULT 0, associated_message_emoji TEXT,
    thread_originator_guid TEXT, destination_caller_id TEXT
);
CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER);
CREATE TABLE attachment (
    ROWID INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, mime_type TEXT, transfer_name TEXT,
    transfer_state INTEGER DEFAULT 5, total_bytes INTEGER, hide_attachment INTEGER DEFAULT 0
);
CREATE TABLE message_attachment_join (message_id INTEGER, attachment_id INTEGER);
"""


def typedstream(text: str) -> bytes:
    """An ``attributedBody`` blob shaped like the real thing, carrying ``text``."""
    data = text.encode()
    length = bytes([len(data)]) if len(data) < 128 else b"\x81" + len(data).to_bytes(2, "little")
    return (
        b"\x04\x0bstreamtyped\x81\xe8\x03\x84\x01@\x84\x84\x84\x12NSAttributedString\x00\x84\x84\x08NSObject\x00\x85\x92\x84\x84\x84\x08NSString\x01\x94\x84\x01+"
        + length
        + data
        + b"\x86\x84\x02iI\x01"
    )


class FakeMessages:
    """Plays the part of the Messages app and its database."""

    def __init__(self, root: Path, *, self_chat_echo: bool = True):
        self.path = root / "chat.db"
        self.db = sqlite3.connect(self.path, isolation_level=None)
        self.db.executescript(SCHEMA)
        self._clock = itertools.count(700_000_000_000_000_000, 1_000_000_000)
        self.self_chat_echo = self_chat_echo
        self.sent: list[tuple[str, str, str]] = []  # (chat guid, text, file name)
        self.fail_with: str | None = None

    # -- building blocks --------------------------------------------------------

    def handle(self, ident: str) -> int:
        row = self.db.execute("SELECT ROWID FROM handle WHERE id = ?", (ident,)).fetchone()
        return (
            row[0]
            if row
            else self.db.execute(
                "INSERT INTO handle (id, service) VALUES (?, 'iMessage')", (ident,)
            ).lastrowid
        )

    def chat(self, identifier: str, *, group: bool = False) -> str:
        guid = f"any;{'+' if group else '-'};{identifier}"
        if not self.db.execute("SELECT 1 FROM chat WHERE guid = ?", (guid,)).fetchone():
            self.db.execute(
                "INSERT INTO chat (guid, chat_identifier, style, service_name) VALUES (?, ?, ?, 'iMessage')",
                (guid, identifier, 43 if group else 45),
            )
        return guid

    def insert(
        self,
        chat_guid: str | None,
        *,
        text: str | None = None,
        sender: str | None = None,
        from_me: bool = False,
        attachments: list[tuple[str, str]] | None = None,
        reply_to: str | None = None,
        tapback: tuple[int, str] | None = None,
        as_body: bool = False,
        join: bool = True,
        own: str = ME,
    ) -> tuple[int, str]:
        guid = str(uuid.uuid4()).upper()
        handle_id = self.handle(sender) if sender else 0
        rowid = self.db.execute(
            """INSERT INTO message (guid, text, attributedBody, handle_id, service, date, is_from_me, cache_has_attachments,
                                    associated_message_guid, associated_message_type, thread_originator_guid, destination_caller_id)
               VALUES (?, ?, ?, ?, 'iMessage', ?, ?, ?, ?, ?, ?, ?)""",
            (
                guid,
                None if as_body else text,
                typedstream(text) if (as_body and text) else None,
                handle_id,
                next(self._clock),
                int(from_me),
                int(bool(attachments)),
                f"p:0/{tapback[1]}" if tapback else None,
                tapback[0] if tapback else 0,
                reply_to,
                own,
            ),
        ).lastrowid
        if chat_guid and join:
            self.join(rowid, chat_guid)
        for path, name in attachments or []:
            attachment_id = self.db.execute(
                "INSERT INTO attachment (filename, mime_type, transfer_name, total_bytes) VALUES (?, 'image/png', ?, ?)",
                (path, name, Path(path).stat().st_size if Path(path).exists() else None),
            ).lastrowid
            self.db.execute("INSERT INTO message_attachment_join VALUES (?, ?)", (rowid, attachment_id))
        return rowid, guid

    def join(self, rowid: int, chat_guid: str) -> None:
        chat_id = self.db.execute("SELECT ROWID FROM chat WHERE guid = ?", (chat_guid,)).fetchone()[0]
        self.db.execute("INSERT INTO chat_message_join VALUES (?, ?)", (chat_id, rowid))

    def guid_of_sent_file(self, name: str, *, from_me: bool = True) -> str:
        return self.db.execute(
            """SELECT m.guid FROM message m JOIN message_attachment_join j ON j.message_id = m.ROWID
               JOIN attachment a ON a.ROWID = j.attachment_id WHERE a.transfer_name = ? AND m.is_from_me = ?""",
            (name, int(from_me)),
        ).fetchone()[0]

    # -- the "Messages app": what AppleScript sends ends up as rows ---------------

    async def run(self, arguments: list[str]) -> str:
        text, file_path, chat_guid, _handle = arguments
        if self.fail_with:
            return self.fail_with
        name = Path(file_path).name if file_path else ""
        self.sent.append((chat_guid, text, name))
        identifier = chat_guid.split(";")[-1]
        is_self = identifier == ME
        attachments = [(file_path, name)] if file_path else None
        self.insert(chat_guid, text=text or None, sender=identifier, from_me=True, attachments=attachments)
        if is_self and self.self_chat_echo:
            self.insert(
                chat_guid, text=text or None, sender=identifier, from_me=False, attachments=attachments
            )
        return "ok"

    def texts(self) -> list[str]:
        return [text for _, text, _ in self.sent if text]

    def files(self) -> list[str]:
        return [name for _, _, name in self.sent if name]


class Phone:
    """A test harness: the fake Messages app plus a running iMessage frontend."""

    def __init__(self, service: Dreambot, messages: FakeMessages, front: IMessageFrontend):
        self.service, self.messages, self.front = service, messages, front

    async def settle(self) -> None:
        """Poll until all jobs are finished and every message has been read back."""
        for _ in range(600):
            read = await self.front.poll_once()
            jobs = self.service.store.recent_jobs(limit=50)
            idle = self.service.queue_length == 0 and all(job.status.finished for job in jobs)
            if idle and read == 0 and not self.front._retry:
                await asyncio.sleep(0.03)
                if await self.front.poll_once() == 0:
                    return
            await asyncio.sleep(0.01)
        raise AssertionError("did not settle")

    def latest(self):
        return self.service.store.recent_jobs(limit=1)[0]


@pytest.fixture
def messages(tmp_path: Path) -> FakeMessages:
    return FakeMessages(tmp_path)


async def start_phone(tmp_path: Path, messages: FakeMessages, mode: str, **config) -> AsyncIterator[Phone]:
    engine = Engine(backend="fake", idle_timeout=None, stderr=subprocess.DEVNULL)
    async with Dreambot(make_config(tmp_path, **config), engine=engine) as service:
        sender = Sender(
            stage_dir=tmp_path / "stage", fallback_dir=tmp_path / "fallback", run=messages.run, pause=0
        )
        front = IMessageFrontend(
            service, db=ChatDB(messages.path), sender=sender, mode=mode, poll_seconds=0.01
        )
        front.db.open()
        front._cursor = front.db.max_rowid()
        front.db.changed()
        from dreambot.frontends.base import ChatFrontend

        await ChatFrontend.start(front)  # subscribe to job events without starting the polling task
        try:
            yield Phone(service, messages, front)
        finally:
            await ChatFrontend.stop(front)
            front.db.close()


@pytest.fixture
async def dedicated(tmp_path: Path, messages: FakeMessages) -> AsyncIterator[Phone]:
    async for phone in start_phone(tmp_path, messages, "dedicated"):
        phone.service.access.allow(f"imessage:{FRIEND}")
        yield phone


@pytest.fixture
async def myself(tmp_path: Path, messages: FakeMessages) -> AsyncIterator[Phone]:
    async for phone in start_phone(tmp_path, messages, "self"):
        yield phone


# --------------------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------------------


def test_attributed_body_decoding():
    assert decode_attributed_body(typedstream("a red fox —ar 16:9")) == "a red fox —ar 16:9"
    long_text = "detailed " * 40
    assert decode_attributed_body(typedstream(long_text)) == long_text
    assert decode_attributed_body(typedstream("emoji 🎨 ok")) == "emoji 🎨 ok"
    assert decode_attributed_body(None) is None
    assert decode_attributed_body(b"not a typedstream") is None
    assert decode_attributed_body(b"\x04\x0bstreamtyped but no string marker") is None


def test_associated_guid_forms_and_timestamps():
    guid = "A1B2C3D4-0000-1111-2222-333344445555"
    assert clean_associated_guid(f"p:2/{guid}") == (2, guid)
    assert clean_associated_guid(f"bp:{guid}") == (0, guid)
    assert clean_associated_guid(guid) == (0, guid)
    assert clean_associated_guid(None) == (0, None)
    assert apple_time(700_000_000_000_000_000) == pytest.approx(700_000_000 + 978307200)
    assert apple_time(500_000_000) == 500_000_000 + 978307200  # very old rows count in seconds
    assert apple_time(None) == 0.0


def test_reading_rows(messages: FakeMessages, tmp_path: Path):
    picture = tmp_path / "IMG_0001.png"
    Image.new("RGB", (8, 8)).save(picture)
    chat = messages.chat(FRIEND)
    group = messages.chat("chat123456", group=True)
    _, first = messages.insert(chat, text="plain text", sender=FRIEND)
    messages.insert(chat, text="from the body blob", sender=FRIEND, as_body=True)
    messages.insert(chat, text="￼ look at this", sender=FRIEND, attachments=[(str(picture), "IMG_0001.png")])
    messages.insert(chat, sender=FRIEND, tapback=(2000, first))
    messages.insert(chat, sender=FRIEND, tapback=(3000, first))
    messages.insert(chat, text="a reply", sender=FRIEND, reply_to=first)
    messages.insert(group, text="in a group", sender=FRIEND)
    messages.db.execute("UPDATE attachment SET hide_attachment = 0")

    db = ChatDB(messages.path)
    db.open()
    rows = db.rows_after(0)
    assert [row.text for row in rows] == [
        "plain text",
        "from the body blob",
        "look at this",
        "",
        "",
        "a reply",
        "in a group",
    ]
    assert (
        rows[0].handle == FRIEND
        and rows[0].chat_guid == chat
        and not rows[0].is_group
        and rows[0].own_address == ME
    )
    assert rows[2].attachments[0].path == str(picture) and rows[2].attachments[0].is_image
    assert (rows[3].reaction, rows[3].reaction_target, rows[3].is_tapback) == ("❤", first, True)
    assert rows[4].reaction is None and rows[4].reaction_removed and rows[4].is_tapback
    assert rows[5].reply_to == first and not rows[5].is_tapback
    assert rows[6].is_group
    assert db.rows_after(rows[-1].rowid) == []
    assert [row.rowid for row in db.rows_by_id([rows[1].rowid, rows[5].rowid])] == [
        rows[1].rowid,
        rows[5].rowid,
    ]

    assert db.changed() is True and db.changed() is False
    messages.insert(chat, text="new", sender=FRIEND)
    assert db.changed() is True
    db.close()


def test_older_schemas_without_newer_columns_still_read(tmp_path: Path):
    path = tmp_path / "old.db"
    db = sqlite3.connect(path)
    db.executescript(
        """
        CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT);
        CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, guid TEXT, chat_identifier TEXT, style INTEGER);
        CREATE TABLE message (ROWID INTEGER PRIMARY KEY, guid TEXT, text TEXT, attributedBody BLOB, handle_id INTEGER,
                              service TEXT, date INTEGER, is_from_me INTEGER, cache_has_attachments INTEGER);
        CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER);
        INSERT INTO handle VALUES (1, '+15550002222');
        INSERT INTO chat VALUES (1, 'iMessage;-;+15550002222', '+15550002222', 45);
        INSERT INTO message VALUES (1, 'G1', 'hello', NULL, 1, 'iMessage', 600000000, 0, 0);
        INSERT INTO chat_message_join VALUES (1, 1);
        """
    )
    db.commit()
    db.close()
    reader = ChatDB(path)
    reader.open()
    (row,) = reader.rows_after(0)
    assert row.text == "hello" and row.reply_to is None and row.reaction is None and row.own_address is None
    reader.close()


def test_unreadable_database_explains_what_to_do(tmp_path: Path):
    with pytest.raises(ChatDBError, match="sign in to iMessage"):
        ChatDB(tmp_path / "missing.db").open()
    not_a_db = tmp_path / "chat.db"
    not_a_db.write_bytes(b"this is not sqlite" * 100)
    with pytest.raises(ChatDBError, match="Full Disk Access"):
        ChatDB(not_a_db).open()


# --------------------------------------------------------------------------------------
# Sending
# --------------------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("osacompile") is None, reason="needs macOS osacompile")
def test_the_applescript_compiles(tmp_path: Path):
    source = tmp_path / "send.applescript"
    source.write_text(SCRIPT)
    result = subprocess.run(
        ["osacompile", "-o", str(tmp_path / "send.scpt"), str(source)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


async def test_sender_stages_files_and_explains_failures(tmp_path: Path, sample_image: Path):
    calls: list[list[str]] = []
    answer = "ok"

    async def run(arguments: list[str]) -> str:
        calls.append(arguments)
        return answer

    sender = Sender(stage_dir=tmp_path / "stage", fallback_dir=tmp_path / "fallback", run=run, pause=0)
    name = await sender.send_file("any;-;+1555", "+1555", sample_image)
    text, staged, chat, handle = calls[-1]
    assert (text, chat, handle) == ("", "any;-;+1555", "+1555")
    assert Path(staged).name == name and Path(staged).read_bytes() == sample_image.read_bytes()
    assert (
        (tmp_path / "stage") in Path(staged).parents and name.startswith("sample-") and name.endswith(".png")
    )
    # Two sends of the same picture get different names, so each can be recognised later.
    assert await sender.send_file("any;-;+1555", None, sample_image) != name

    await sender.send_text("any;-;+1555", None, "")  # nothing to send
    assert len(calls) == 2

    answer = "fail\tpre\t-1743\tNot authorized to send Apple events to Messages."
    with pytest.raises(SendError) as excinfo:
        await sender.send_text("any;-;+1555", "+1555", "hi")
    assert excinfo.value.code == -1743 and "Automation" in str(excinfo.value) and not excinfo.value.maybe_sent

    answer = "fail\tsent\t-1712\tAppleEvent timed out."
    with pytest.raises(SendError) as excinfo:
        await sender.send_text("any;-;+1555", "+1555", "hi")
    assert excinfo.value.maybe_sent

    answer = "fail\tpre\t-9999\tSomething odd"
    with pytest.raises(SendError, match="Something odd"):
        await sender.send_text("any;-;+1555", "+1555", "hi")


async def test_sender_falls_back_when_the_messages_folder_is_not_writable(tmp_path: Path, sample_image: Path):
    blocked = tmp_path / "blocked"
    blocked.write_text("a file where a folder should be")

    async def run(arguments: list[str]) -> str:
        return "ok"

    sender = Sender(stage_dir=blocked, fallback_dir=tmp_path / "fallback", run=run, pause=0)
    staged = sender.stage(sample_image)
    assert (tmp_path / "fallback") in staged.parents
    sender.cleanup(max_age=-1)
    assert not staged.exists()


# --------------------------------------------------------------------------------------
# Dedicated account: people text the bot
# --------------------------------------------------------------------------------------


async def test_a_text_becomes_four_pictures_and_a_caption(dedicated: Phone):
    chat = dedicated.messages.chat(FRIEND)
    dedicated.messages.insert(chat, text="a lighthouse at dusk —ar 16:9", sender=FRIEND, as_body=True)
    await dedicated.settle()

    job = dedicated.latest()
    assert job.status is JobStatus.DONE and (job.width, job.height) == (1456, 816)
    assert job.user == f"imessage:{FRIEND}" and job.channel == f"imessage:{chat}"
    texts, files = dedicated.messages.texts(), dedicated.messages.files()
    assert texts[0].startswith("🎨") and "a lighthouse at dusk" in texts[0]
    assert len(files) == 4 and "U1–U4" in texts[-1] and f"#{job.id}" in texts[-1]
    # The bot's own messages came back through the database and started nothing.
    assert len(dedicated.service.store.recent_jobs(limit=10)) == 1


async def test_reply_to_a_picture_and_tapbacks_target_that_picture(dedicated: Phone):
    chat = dedicated.messages.chat(FRIEND)
    dedicated.messages.insert(chat, text="a fox", sender=FRIEND)
    await dedicated.settle()
    job = dedicated.latest()
    third = dedicated.messages.guid_of_sent_file(dedicated.messages.files()[2])
    first = dedicated.messages.guid_of_sent_file(dedicated.messages.files()[0])

    dedicated.messages.insert(chat, text="upscale", sender=FRIEND, reply_to=third)
    await dedicated.settle()
    upscaled = dedicated.latest()
    assert upscaled.kind is JobKind.UPSCALE and (upscaled.parent, upscaled.parent_index) == (job.id, 3)

    dedicated.messages.insert(chat, text="make it snow", sender=FRIEND, reply_to=first)
    await dedicated.settle()
    edited = dedicated.latest()
    assert edited.kind is JobKind.EDIT and edited.text == "make it snow" and edited.parent_index == 1

    dedicated.messages.insert(chat, sender=FRIEND, tapback=(2000, third))  # ❤
    await dedicated.settle()
    assert dedicated.service.store.get_job(job.id).image(3).favorite
    assert dedicated.messages.texts()[-1] == "Added to favorites."

    dedicated.messages.insert(chat, sender=FRIEND, tapback=(2004, first))  # ‼ = variations
    await dedicated.settle()
    varied = dedicated.latest()
    assert varied.kind is JobKind.VARIATION and varied.parent_index == 1

    dedicated.messages.insert(chat, sender=FRIEND, tapback=(3000, third))  # removing a tapback does nothing
    await dedicated.settle()
    assert dedicated.service.store.get_job(job.id).image(3).favorite


async def test_shortcuts_and_attachments(dedicated: Phone, sample_image: Path):
    chat = dedicated.messages.chat(FRIEND)
    dedicated.messages.insert(chat, text="a fox", sender=FRIEND)
    await dedicated.settle()
    job = dedicated.latest()
    dedicated.messages.insert(chat, text="V2", sender=FRIEND)
    await dedicated.settle()
    assert dedicated.latest().kind is JobKind.VARIATION and dedicated.latest().parent == job.id

    # A bare shortcut applies to the most recent job in the conversation: here, the variations.
    variations = dedicated.latest()
    dedicated.messages.insert(chat, text="U3", sender=FRIEND)
    await dedicated.settle()
    assert dedicated.messages.files()[-1].startswith(Path(variations.image(3).path).stem[:40])

    dedicated.messages.insert(
        chat, text="make it night --edit --n 1", sender=FRIEND, attachments=[(str(sample_image), "IMG_1.png")]
    )
    await dedicated.settle()
    assert dedicated.latest().status is JobStatus.DONE and dedicated.latest().inputs["attachments"]


async def test_strangers_and_group_chatter_are_never_answered(dedicated: Phone):
    stranger_chat = dedicated.messages.chat(STRANGER)
    dedicated.messages.insert(stranger_chat, text="a fox", sender=STRANGER)
    dedicated.messages.insert(stranger_chat, text="/imagine a fox", sender=STRANGER)
    group = dedicated.messages.chat("chat777", group=True)
    dedicated.messages.insert(group, text="anyone up for lunch?", sender=FRIEND)
    await dedicated.settle()
    assert dedicated.messages.sent == [] and dedicated.service.store.recent_jobs(limit=5) == []

    dedicated.messages.insert(group, text="/imagine a fox --n 1", sender=FRIEND)
    await dedicated.settle()
    assert dedicated.latest().channel == f"imessage:{group}"

    code = dedicated.service.access.new_pairing_code()
    dedicated.messages.insert(stranger_chat, text=f"/pair {code}", sender=STRANGER)
    await dedicated.settle()
    assert dedicated.service.access.is_owner(f"imessage:{STRANGER}")


async def test_message_that_joins_its_chat_late_is_still_handled(dedicated: Phone):
    chat = dedicated.messages.chat(FRIEND)
    rowid, _ = dedicated.messages.insert(chat, text="a fox --n 1", sender=FRIEND, join=False)
    await dedicated.front.poll_once()
    await dedicated.front.poll_once()
    assert dedicated.service.store.recent_jobs(limit=5) == [] and rowid in dedicated.front._retry
    dedicated.messages.join(rowid, chat)
    await dedicated.settle()
    assert dedicated.latest().text == "a fox"


async def test_send_failures_do_not_crash_the_bot(dedicated: Phone):
    chat = dedicated.messages.chat(FRIEND)
    dedicated.messages.fail_with = "fail\tpre\t-1743\tNot authorized"
    dedicated.messages.insert(chat, text="a fox --n 1", sender=FRIEND)
    await dedicated.settle()
    assert dedicated.latest().status is JobStatus.DONE  # the image exists even though it could not be sent
    dedicated.messages.fail_with = None
    dedicated.messages.insert(chat, text="/show", sender=FRIEND)
    await dedicated.settle()
    assert "Recent jobs" in dedicated.messages.texts()[-1]


# --------------------------------------------------------------------------------------
# Self mode: you text yourself
# --------------------------------------------------------------------------------------


def text_myself(phone: Phone, text: str, **kwargs) -> None:
    """A note-to-self message as it lands in the database: a sent copy and a received copy."""
    chat = phone.messages.chat(ME)
    phone.messages.insert(chat, text=text, sender=ME, from_me=True, **kwargs)
    if phone.messages.self_chat_echo:
        phone.messages.insert(chat, text=text, sender=ME, from_me=False, **kwargs)


async def test_note_to_self_is_handled_once_and_the_bot_does_not_talk_to_itself(myself: Phone):
    text_myself(myself, "a fox")
    await myself.settle()
    jobs = myself.service.store.recent_jobs(limit=10)
    assert len(jobs) == 1 and jobs[0].user == "imessage:me" and jobs[0].status is JobStatus.DONE
    assert len(myself.messages.files()) == 4

    # Sending the very same prompt again is a new request, not an echo.
    text_myself(myself, "a fox")
    await myself.settle()
    assert len(myself.service.store.recent_jobs(limit=10)) == 2


async def test_reply_and_tapback_work_on_either_copy_of_a_picture(myself: Phone):
    text_myself(myself, "a fox")
    await myself.settle()
    job = myself.latest()
    name = myself.messages.files()[1]
    sent_copy = myself.messages.guid_of_sent_file(name, from_me=True)
    received_copy = myself.messages.guid_of_sent_file(name, from_me=False)

    text_myself(myself, "upscale", reply_to=received_copy)
    await myself.settle()
    assert (myself.latest().kind, myself.latest().parent_index) == (JobKind.UPSCALE, 2)

    chat = myself.messages.chat(ME)
    myself.messages.insert(chat, sender=ME, from_me=True, tapback=(2000, sent_copy))
    myself.messages.insert(chat, sender=ME, from_me=False, tapback=(2000, sent_copy))
    await myself.settle()
    # One tapback, seen twice, toggles the favourite exactly once.
    assert myself.service.store.get_job(job.id).image(2).favorite


async def test_email_addressed_note_to_self_has_no_echo(tmp_path: Path):
    messages = FakeMessages(tmp_path, self_chat_echo=False)
    async for phone in start_phone(tmp_path, messages, "self"):
        text_myself(phone, "a fox --n 1")
        await phone.settle()
        assert len(phone.service.store.recent_jobs(limit=10)) == 1 and len(messages.files()) == 1


async def test_summoning_the_bot_inside_another_conversation(myself: Phone):
    chat = myself.messages.chat(FRIEND)
    # Ordinary conversation in both directions is left alone.
    myself.messages.insert(chat, text="see you at 8?", sender=FRIEND, from_me=True)
    myself.messages.insert(chat, text="a fox would be a great logo", sender=FRIEND)
    await myself.settle()
    assert myself.messages.sent == []

    # The owner summons the bot with the trigger...
    myself.messages.insert(chat, text="/imagine a fox logo --n 1", sender=FRIEND, from_me=True)
    await myself.settle()
    assert myself.latest().user == "imessage:me" and myself.latest().channel == f"imessage:{chat}"

    # ...a friend can only do so once allowed, and is otherwise ignored without a reply.
    before = len(myself.messages.sent)
    myself.messages.insert(chat, text="/imagine a wolf --n 1", sender=FRIEND)
    await myself.settle()
    assert len(myself.messages.sent) == before
    myself.service.access.allow(f"imessage:{FRIEND}")
    myself.messages.insert(chat, text="/imagine a wolf --n 1", sender=FRIEND)
    await myself.settle()
    assert myself.latest().text == "a wolf" and myself.latest().user == f"imessage:{FRIEND}"
