"""Reading new messages from the Messages database.

macOS keeps every iMessage in ``~/Library/Messages/chat.db``, a SQLite database. Dreambot only
ever *reads* it (opened read-only), and only rows that arrive after it starts. Reading requires
the Full Disk Access permission for whichever program runs Dreambot.

Things worth knowing about this database, all handled here:

* ``message.text`` is often NULL on recent macOS; the body is then inside ``attributedBody``,
  an archived ``NSAttributedString`` (:func:`decode_attributed_body`).
* A tapback ("loved an image") is its own row whose ``associated_message_guid`` points at the
  message it reacts to; an inline reply names its target in ``thread_originator_guid``.
* A message row can appear a moment before the row that ties it to a chat, so a row without a
  chat is retried rather than trusted.
* Column sets differ between macOS versions, so columns are selected by name after probing.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PATH = Path.home() / "Library" / "Messages" / "chat.db"
#: Seconds between 1970-01-01 and 2001-01-01, the epoch Apple timestamps count from.
APPLE_EPOCH = 978307200

TAPBACKS = {0: "❤", 1: "👍", 2: "👎", 3: "😂", 4: "‼", 5: "❓"}

#: Columns Dreambot cannot work without, by table. They have been stable across macOS releases;
#: if one ever goes, it is better to say so once at start than to fail on every poll.
REQUIRED = {
    "message": {
        "guid",
        "text",
        "attributedBody",
        "date",
        "is_from_me",
        "service",
        "handle_id",
        "cache_has_attachments",
    },
    "handle": {"id"},
    "chat": {"guid", "chat_identifier", "style"},
    "chat_message_join": {"chat_id", "message_id"},
    "attachment": {
        "filename",
        "mime_type",
        "transfer_name",
        "transfer_state",
        "total_bytes",
        "hide_attachment",
    },
    "message_attachment_join": {"message_id", "attachment_id"},
}
#: Columns that are used when present (tapbacks, inline replies, which address was written to).
OPTIONAL = (
    "associated_message_guid",
    "associated_message_type",
    "associated_message_emoji",
    "thread_originator_guid",
    "destination_caller_id",
)


class ChatDBError(Exception):
    """The Messages database cannot be read. The message says how to fix it."""


def decode_attributed_body(blob: bytes | None) -> str | None:
    """Extract the plain text from a ``message.attributedBody`` typedstream blob."""
    if not blob or blob[:2] != b"\x04\x0b" or blob[2:13] != b"streamtyped":
        return None
    start = blob.find(b"\x01+", 13)
    if start < 0:
        return None
    index = start + 2
    if index >= len(blob):
        return None
    length = blob[index]
    index += 1
    if length == 0x81:
        length = int.from_bytes(blob[index : index + 2], "little")
        index += 2
    elif length == 0x82:
        length = int.from_bytes(blob[index : index + 4], "little")
        index += 4
    elif length >= 0x80:
        return None
    if index + length > len(blob):
        return None
    return blob[index : index + length].decode("utf-8", "replace")


def clean_associated_guid(value: str | None) -> tuple[int, str | None]:
    """Split ``associated_message_guid`` into (part index, message guid).

    Forms: ``p:2/<guid>`` (third part of a message), ``bp:<guid>`` (a bubble such as a link
    preview), or a bare guid.
    """
    if not value:
        return 0, None
    if value.startswith("p:"):
        part, _, rest = value[2:].partition("/")
        return (int(part) if part.isdigit() else 0), rest[:36] or None
    if value.startswith("bp:"):
        return 0, value[3:39] or None
    return 0, value[:36]


def apple_time(value: int | float | None) -> float:
    """Convert a Messages timestamp (nanoseconds, or seconds on very old rows) to Unix time."""
    if not value:
        return 0.0
    seconds = value / 1e9 if value > 1e11 else float(value)
    return seconds + APPLE_EPOCH


@dataclass(frozen=True)
class Attachment:
    #: Absolute path of the file, with ``~`` expanded.
    path: str | None
    mime: str | None
    #: The file's name as sent (Dreambot uses this to recognise its own outgoing images).
    name: str | None
    transfer_state: int | None = None
    total_bytes: int | None = None

    @property
    def is_image(self) -> bool:
        if self.mime:
            return self.mime.lower().startswith("image/")
        suffix = Path(self.path or self.name or "").suffix.lower()
        return suffix in {".png", ".jpg", ".jpeg", ".heic", ".heif", ".webp", ".gif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class Row:
    rowid: int
    guid: str
    text: str
    is_from_me: bool
    service: str
    #: The other party: the sender of an incoming message, the recipient of an outgoing 1:1 one.
    handle: str | None
    chat_guid: str | None
    chat_identifier: str | None
    is_group: bool
    #: The address of this Mac's own account that the message was sent to or from.
    own_address: str | None
    timestamp: float
    #: Guid of the message this one is an inline reply to.
    reply_to: str | None = None
    #: Tapback emoji when this row is a reaction being added, else ``None``.
    reaction: str | None = None
    #: Guid of the message a tapback reacts to.
    reaction_target: str | None = None
    #: True when this row removes a tapback instead of adding one.
    reaction_removed: bool = False
    attachments: list[Attachment] = field(default_factory=list)

    @property
    def is_tapback(self) -> bool:
        return self.reaction is not None or self.reaction_removed


def _expand(path: str | None) -> str | None:
    return str(Path(path).expanduser()) if path else None


class ChatDB:
    """A read-only, forward-only view of the Messages database."""

    def __init__(self, path: Path | str = DEFAULT_PATH):
        self.path = Path(path)
        self._db: sqlite3.Connection | None = None
        self._columns: set[str] = set()
        self._data_version: int | None = None
        #: Optional columns this macOS does not have (the features that use them are off).
        self.missing_optional: list[str] = []

    def open(self) -> None:
        if not self.path.exists():
            raise ChatDBError(
                f"There is no Messages database at {self.path}. Open the Messages app and sign in to iMessage first."
            )
        try:
            # Read-only, but NOT immutable: immutable connections never see new rows.
            db = sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True, timeout=5, isolation_level=None)
            db.row_factory = sqlite3.Row
            layout = {
                table: {row["name"] for row in db.execute(f"PRAGMA table_info({table})").fetchall()}
                for table in REQUIRED
            }
            self._columns = layout["message"]
            if self._columns:
                db.execute("SELECT MAX(ROWID) FROM message").fetchone()
        except sqlite3.Error as exc:
            raise ChatDBError(
                "macOS is blocking access to your Messages database. Grant Full Disk Access to the program "
                "that runs Dreambot: System Settings → Privacy & Security → Full Disk Access. "
                f"(`dreambot doctor` shows exactly which program to add.) [{exc}]"
            ) from exc
        if not self._columns:
            db.close()
            raise ChatDBError(f"{self.path} does not look like a Messages database.")
        missing = [
            f"{table}.{column}"
            for table, columns in REQUIRED.items()
            for column in sorted(columns - layout[table])
        ]
        if missing:
            db.close()
            raise ChatDBError(
                "This version of macOS stores messages differently from what Dreambot expects "
                f"(missing: {', '.join(missing)}). Please report it."
            )
        self.missing_optional = [name for name in OPTIONAL if name not in self._columns]
        self._db = db

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    @property
    def _connection(self) -> sqlite3.Connection:
        if self._db is None:
            self.open()
        assert self._db is not None
        return self._db

    def max_rowid(self) -> int:
        row = self._connection.execute("SELECT MAX(ROWID) AS n FROM message").fetchone()
        return int(row["n"] or 0)

    def changed(self) -> bool:
        """Cheap check for whether anything was written since the last call."""
        version = self._connection.execute("PRAGMA data_version").fetchone()[0]
        if version == self._data_version:
            return False
        self._data_version = version
        return True

    def _column(self, name: str) -> str:
        return f"m.{name}" if name in self._columns else "NULL"

    def rows_after(self, rowid: int, limit: int = 200) -> list[Row]:
        """Messages with a ROWID greater than ``rowid``, oldest first."""
        return self._rows("m.ROWID > ?", (rowid,), limit)

    def rows_by_id(self, rowids: list[int]) -> list[Row]:
        """Re-read specific messages (used to retry rows that had no chat yet)."""
        if not rowids:
            return []
        marks = ",".join("?" * len(rowids))
        return self._rows(f"m.ROWID IN ({marks})", tuple(rowids), len(rowids) * 4)

    def _rows(self, where: str, values: tuple, limit: int) -> list[Row]:
        query = f"""
            SELECT m.ROWID AS rowid, m.guid, m.text, m.attributedBody AS body, m.date, m.is_from_me,
                   m.service, m.cache_has_attachments,
                   {self._column("associated_message_guid")} AS associated_guid,
                   {self._column("associated_message_type")} AS associated_type,
                   {self._column("associated_message_emoji")} AS associated_emoji,
                   {self._column("thread_originator_guid")} AS thread_guid,
                   {self._column("destination_caller_id")} AS own_address,
                   h.id AS handle, c.guid AS chat_guid, c.chat_identifier, c.style AS chat_style
            FROM message m
            LEFT JOIN handle h ON h.ROWID = m.handle_id
            LEFT JOIN chat_message_join j ON j.message_id = m.ROWID
            LEFT JOIN chat c ON c.ROWID = j.chat_id
            WHERE {where}
            ORDER BY m.ROWID
            LIMIT ?
        """
        db = self._connection
        records = db.execute(query, (*values, limit)).fetchall()
        rows: list[Row] = []
        seen: set[int] = set()
        for record in records:
            if record["rowid"] in seen:  # a message joined to more than one chat row
                continue
            seen.add(record["rowid"])
            text = record["text"] or decode_attributed_body(record["body"]) or ""
            # U+FFFC is the placeholder Messages puts in the text where an attachment sits.
            text = text.replace("￼", "").strip()
            kind = record["associated_type"] or 0
            reaction: str | None = None
            removed = False
            target: str | None = None
            if 2000 <= kind <= 2007 or 3000 <= kind <= 3007:
                _, target = clean_associated_guid(record["associated_guid"])
                removed = kind >= 3000
                if not removed:
                    reaction = TAPBACKS.get(kind - 2000) or record["associated_emoji"] or "❔"
            chat_guid = record["chat_guid"]
            rows.append(
                Row(
                    rowid=record["rowid"],
                    guid=record["guid"],
                    text=text,
                    is_from_me=bool(record["is_from_me"]),
                    service=record["service"] or "",
                    handle=record["handle"],
                    chat_guid=chat_guid,
                    chat_identifier=record["chat_identifier"],
                    # The guid's middle field is "+" for groups and "-" for one-to-one chats;
                    # it is more dependable than the numeric style column.
                    is_group=";+;" in chat_guid if chat_guid else record["chat_style"] == 43,
                    own_address=record["own_address"],
                    timestamp=apple_time(record["date"]),
                    reply_to=record["thread_guid"],
                    reaction=reaction,
                    reaction_target=target,
                    reaction_removed=removed,
                    attachments=self._attachments(record["rowid"]) if record["cache_has_attachments"] else [],
                )
            )
        return rows

    def guids_of_attachment(self, transfer_name: str) -> list[str]:
        """The guids of the messages carrying the attachment sent under this (unique) name.

        Usually one. In a note-to-self thread there are two: the copy sent and the copy received.
        """
        try:
            rows = self._connection.execute(
                """
                SELECT m.guid FROM attachment a
                JOIN message_attachment_join j ON j.attachment_id = a.ROWID
                JOIN message m ON m.ROWID = j.message_id
                WHERE a.transfer_name = ?
                ORDER BY m.ROWID
                """,
                (transfer_name,),
            ).fetchall()
        except sqlite3.Error:
            return []
        return [row["guid"] for row in rows if row["guid"]]

    def _attachments(self, message_rowid: int) -> list[Attachment]:
        records = self._connection.execute(
            """
            SELECT a.filename, a.mime_type, a.transfer_name, a.transfer_state, a.total_bytes, a.hide_attachment
            FROM message_attachment_join j
            JOIN attachment a ON a.ROWID = j.attachment_id
            WHERE j.message_id = ?
            ORDER BY a.ROWID
            """,
            (message_rowid,),
        ).fetchall()
        return [
            Attachment(
                path=_expand(record["filename"]),
                mime=record["mime_type"],
                name=record["transfer_name"],
                transfer_state=record["transfer_state"],
                total_bytes=record["total_bytes"],
            )
            for record in records
            if not record["hide_attachment"]  # link previews and other non-user attachments
        ]
