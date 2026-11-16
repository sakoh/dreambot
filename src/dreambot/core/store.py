"""Persistence: jobs, per-user settings and the mapping from chat messages back to images.

Everything lives in one small SQLite file on the Mac. Calls are synchronous — each is a
sub-millisecond local write — and guarded by a lock so background threads may use the store too.

The schema only ever grows by whole tables, and nothing is added to the JSON of a job that an
older build would trip over. So an older Dreambot can still open a database a newer one has
used: it simply does not see the newer tables. That is why they do not raise ``SCHEMA_VERSION``.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dreambot.core.jobs import Job, JobImage, JobStatus, new_job_id, normalize_job_id

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id        TEXT PRIMARY KEY,
    user      TEXT NOT NULL,
    channel   TEXT NOT NULL,
    kind      TEXT NOT NULL,
    status    TEXT NOT NULL,
    parent    TEXT,
    created   REAL NOT NULL,
    data      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_by_user ON jobs (user, created);
CREATE INDEX IF NOT EXISTS jobs_by_channel ON jobs (channel, created);
CREATE INDEX IF NOT EXISTS jobs_by_status ON jobs (status);

CREATE TABLE IF NOT EXISTS users (
    id        TEXT PRIMARY KEY,
    settings  TEXT NOT NULL DEFAULT '{}',
    created   REAL NOT NULL
);

-- Which job (and which image of it) a chat message shows. This is what lets a button press,
-- an inline reply or a tapback on a particular message find its way back to an image.
CREATE TABLE IF NOT EXISTS messages (
    frontend  TEXT NOT NULL,
    ref       TEXT NOT NULL,
    job       TEXT NOT NULL,
    image     INTEGER,
    created   REAL NOT NULL,
    PRIMARY KEY (frontend, ref)
);
CREATE INDEX IF NOT EXISTS messages_by_job ON messages (job);

CREATE TABLE IF NOT EXISTS state (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);

-- What has been worked out about a style, so that it is only ever worked out once: the
-- fingerprint of an image (keyed by its contents) and the wording of a blend (keyed by its recipe).
CREATE TABLE IF NOT EXISTS fingerprints (
    key      TEXT PRIMARY KEY,
    data     TEXT NOT NULL,
    created  REAL NOT NULL
);

-- Styles a person keeps: ones they saved under a name (with the fingerprint in ``data``), and
-- numeric style codes they liked (``data`` is NULL: a code decodes to its style by itself).
CREATE TABLE IF NOT EXISTS styles (
    user     TEXT NOT NULL,
    code     TEXT NOT NULL,
    name     TEXT NOT NULL DEFAULT '',
    liked    INTEGER NOT NULL DEFAULT 0,
    data     TEXT,
    created  REAL NOT NULL,
    PRIMARY KEY (user, code)
);

-- Lookbooks: named sets of pictures that stand for a look (see dreambot.core.lookbook).
CREATE TABLE IF NOT EXISTS lookbooks (
    id       TEXT PRIMARY KEY,
    user     TEXT NOT NULL,
    name     TEXT NOT NULL,
    created  REAL NOT NULL,
    updated  REAL NOT NULL,
    deleted  REAL
);
CREATE TABLE IF NOT EXISTS lookbook_images (
    id       TEXT PRIMARY KEY,
    lookbook TEXT NOT NULL,
    path     TEXT NOT NULL,
    key      TEXT NOT NULL,
    source   TEXT NOT NULL DEFAULT '',
    added    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS lookbook_images_by_book ON lookbook_images (lookbook, added);
-- Every state a Lookbook has been used in, frozen: a code outlives edits to its book, and the book.
CREATE TABLE IF NOT EXISTS lookbook_codes (
    code     TEXT PRIMARY KEY,
    lookbook TEXT NOT NULL,
    data     TEXT NOT NULL,
    created  REAL NOT NULL
);

-- The web app's library of pictures a person has uploaded to use in prompts.
CREATE TABLE IF NOT EXISTS uploads (
    id       TEXT PRIMARY KEY,
    user     TEXT NOT NULL,
    path     TEXT NOT NULL,
    key      TEXT NOT NULL,
    name     TEXT NOT NULL DEFAULT '',
    created  REAL NOT NULL
);
-- Images put aside in the gallery. Nothing is deleted: a trashed image is only hidden.
CREATE TABLE IF NOT EXISTS trash (
    job      TEXT NOT NULL,
    image    INTEGER NOT NULL,
    created  REAL NOT NULL,
    PRIMARY KEY (job, image)
);
CREATE TABLE IF NOT EXISTS folders (
    id       TEXT PRIMARY KEY,
    user     TEXT NOT NULL,
    name     TEXT NOT NULL,
    created  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS folder_items (
    folder   TEXT NOT NULL,
    job      TEXT NOT NULL,
    image    INTEGER NOT NULL,
    added    REAL NOT NULL,
    PRIMARY KEY (folder, job, image)
);
"""


def style_slug(name: str) -> str:
    """A style's name as the one word it is typed as after ``--sref``."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


@dataclass(frozen=True)
class MessageLink:
    job: str
    #: 1-based image index, or ``None`` when the message shows the whole job (a grid).
    image: int | None


class Store:
    def __init__(self, path: Path | str):
        self.path = Path(path) if str(path) != ":memory:" else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.executescript(_SCHEMA)
            version = self._db.execute("PRAGMA user_version").fetchone()[0]
            if version == 0:
                self._db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            elif version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"{path} was written by a newer Dreambot (schema {version}); this build understands {SCHEMA_VERSION}."
                )

    def close(self) -> None:
        with self._lock:
            self._db.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            else:
                self._db.execute("COMMIT")

    # ---------------------------------------------------------------------- jobs ----

    def reserve_job_id(self) -> str:
        """A fresh job id that is not in use."""
        with self._lock:
            while True:
                candidate = new_job_id()
                if self._db.execute("SELECT 1 FROM jobs WHERE id = ?", (candidate,)).fetchone() is None:
                    return candidate

    def save_job(self, job: Job) -> None:
        payload = json.dumps(job.to_dict(), ensure_ascii=False)
        with self._lock:
            self._db.execute(
                """INSERT INTO jobs (id, user, channel, kind, status, parent, created, data)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET status = excluded.status, data = excluded.data""",
                (
                    job.id,
                    job.user,
                    job.channel,
                    job.kind.value,
                    job.status.value,
                    job.parent,
                    job.created,
                    payload,
                ),
            )

    def get_job(self, job_id: str) -> Job | None:
        with self._lock:
            row = self._db.execute(
                "SELECT data FROM jobs WHERE id = ?", (normalize_job_id(job_id),)
            ).fetchone()
        return Job.from_dict(json.loads(row["data"])) if row else None

    def recent_jobs(
        self, *, user: str | None = None, channel: str | None = None, limit: int = 20, only_done: bool = False
    ) -> list[Job]:
        """Newest first."""
        clauses, values = [], []
        if user is not None:
            clauses.append("user = ?")
            values.append(user)
        if channel is not None:
            clauses.append("channel = ?")
            values.append(channel)
        if only_done:
            clauses.append("status = ?")
            values.append(JobStatus.DONE.value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._db.execute(
                f"SELECT data FROM jobs {where} ORDER BY created DESC, rowid DESC LIMIT ?", (*values, limit)
            ).fetchall()
        return [Job.from_dict(json.loads(row["data"])) for row in rows]

    def last_job(self, *, channel: str, with_images: bool = True) -> Job | None:
        """The most recent finished job in a conversation — what a bare "U2" refers to."""
        for job in self.recent_jobs(channel=channel, limit=25, only_done=True):
            if job.images or not with_images:
                return job
        return None

    def interrupted_jobs(self) -> list[Job]:
        """Jobs left queued or running by a previous run of the program."""
        with self._lock:
            rows = self._db.execute(
                "SELECT data FROM jobs WHERE status IN (?, ?) ORDER BY created",
                (JobStatus.QUEUED.value, JobStatus.RUNNING.value),
            ).fetchall()
        return [Job.from_dict(json.loads(row["data"])) for row in rows]

    def count_jobs(self, *, user: str | None = None) -> dict[str, int]:
        where, values = ("WHERE user = ?", (user,)) if user else ("", ())
        with self._lock:
            rows = self._db.execute(
                f"SELECT status, COUNT(*) AS n FROM jobs {where} GROUP BY status", values
            ).fetchall()
        return {row["status"]: row["n"] for row in rows}

    # --------------------------------------------------------------------- users ----

    def get_settings(self, user: str) -> dict[str, Any]:
        with self._lock:
            row = self._db.execute("SELECT settings FROM users WHERE id = ?", (user,)).fetchone()
        return json.loads(row["settings"]) if row else {}

    def save_settings(self, user: str, settings: dict[str, Any]) -> None:
        with self._lock:
            self._db.execute(
                """INSERT INTO users (id, settings, created) VALUES (?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET settings = excluded.settings""",
                (user, json.dumps(settings, ensure_ascii=False), time.time()),
            )

    # ------------------------------------------------------------------ messages ----

    def link_message(self, frontend: str, ref: str, job: str, image: int | None = None) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO messages (frontend, ref, job, image, created) VALUES (?, ?, ?, ?, ?)",
                (frontend, str(ref), job, image, time.time()),
            )

    def find_message(self, frontend: str, ref: str) -> MessageLink | None:
        with self._lock:
            row = self._db.execute(
                "SELECT job, image FROM messages WHERE frontend = ? AND ref = ?", (frontend, str(ref))
            ).fetchone()
        return MessageLink(row["job"], row["image"]) if row else None

    def messages_for_job(self, frontend: str, job: str) -> list[tuple[str, int | None]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT ref, image FROM messages WHERE frontend = ? AND job = ? ORDER BY created",
                (frontend, job),
            ).fetchall()
        return [(row["ref"], row["image"]) for row in rows]

    # -------------------------------------------------------------------- styles ----

    def get_fingerprint(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT data FROM fingerprints WHERE key = ?", (key,)).fetchone()
        return json.loads(row["data"]) if row else None

    def save_fingerprint(self, key: str, data: dict[str, Any]) -> None:
        with self._lock:
            self._db.execute(
                """INSERT INTO fingerprints (key, data, created) VALUES (?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET data = excluded.data""",
                (key, json.dumps(data, ensure_ascii=False), time.time()),
            )

    def save_style(
        self, user: str, code: str, *, name: str = "", data: dict[str, Any] | None = None, liked: bool = False
    ) -> None:
        with self._lock:
            self._db.execute(
                """INSERT INTO styles (user, code, name, liked, data, created) VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(user, code) DO UPDATE SET
                       name = excluded.name, liked = excluded.liked, data = COALESCE(excluded.data, styles.data)""",
                (
                    user,
                    code,
                    name,
                    int(liked),
                    json.dumps(data, ensure_ascii=False) if data is not None else None,
                    time.time(),
                ),
            )

    @staticmethod
    def _style(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "code": row["code"],
            "name": row["name"],
            "liked": bool(row["liked"]),
            "data": json.loads(row["data"]) if row["data"] else None,
            "created": row["created"],
        }

    def get_style(self, user: str, token: str) -> dict[str, Any] | None:
        """A kept style by its code, or by its name written as one word ("dusty-noir").

        A Dreambot belongs to one household: a style saved from the web app is meant to work from
        iMessage too. So every allowed person can use a saved style, and ``user``'s own wins
        when two share a name.
        """
        wanted = token.strip().lower()
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM styles WHERE data IS NOT NULL ORDER BY (user = ?) DESC, created DESC", (user,)
            ).fetchall()
        for row in rows:
            if row["code"].lower() == wanted:
                return self._style(row)
        for row in rows:
            if row["name"] and style_slug(row["name"]) == style_slug(wanted):
                return self._style(row)
        return None

    def list_styles(
        self, user: str, *, liked: bool | None = None, saved: bool | None = None
    ) -> list[dict[str, Any]]:
        """Newest first. ``saved`` selects named styles (True) or liked codes (False)."""
        clauses, values = ["user = ?"], [user]
        if liked is not None:
            clauses.append("liked = ?")
            values.append(int(liked))
        if saved is not None:
            clauses.append("data IS NOT NULL" if saved else "data IS NULL")
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM styles WHERE {' AND '.join(clauses)} ORDER BY created DESC, rowid DESC",
                values,
            ).fetchall()
        return [self._style(row) for row in rows]

    def delete_style(self, user: str, code: str) -> bool:
        with self._lock:
            done = self._db.execute("DELETE FROM styles WHERE user = ? AND code = ?", (user, code))
        return done.rowcount > 0

    # ----------------------------------------------------------------- lookbooks ----

    @staticmethod
    def _lookbook(row: sqlite3.Row) -> dict[str, Any]:
        return {key: row[key] for key in ("id", "user", "name", "created", "updated", "deleted")}

    def create_lookbook(self, lookbook_id: str, user: str, name: str) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            self._db.execute(
                "INSERT INTO lookbooks (id, user, name, created, updated) VALUES (?, ?, ?, ?, ?)",
                (lookbook_id, user, name, now, now),
            )
        return {
            "id": lookbook_id,
            "user": user,
            "name": name,
            "created": now,
            "updated": now,
            "deleted": None,
        }

    def get_lookbook(self, lookbook_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM lookbooks WHERE id = ?", (lookbook_id,)).fetchone()
        return self._lookbook(row) if row else None

    def list_lookbooks(self) -> list[dict[str, Any]]:
        """Every Lookbook that has not been deleted, most recently changed first."""
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM lookbooks WHERE deleted IS NULL ORDER BY updated DESC, rowid DESC"
            ).fetchall()
        return [self._lookbook(row) for row in rows]

    def update_lookbook(self, lookbook_id: str, *, name: str | None = None, deleted: bool = False) -> None:
        now = time.time()
        with self._lock:
            if name is not None:
                self._db.execute(
                    "UPDATE lookbooks SET name = ?, updated = ? WHERE id = ?", (name, now, lookbook_id)
                )
            if deleted:
                self._db.execute("UPDATE lookbooks SET deleted = ? WHERE id = ?", (now, lookbook_id))

    def add_lookbook_image(
        self, image_id: str, lookbook_id: str, path: str, key: str, source: str = ""
    ) -> None:
        now = time.time()
        with self._transaction() as db:
            db.execute(
                "INSERT INTO lookbook_images (id, lookbook, path, key, source, added) VALUES (?, ?, ?, ?, ?, ?)",
                (image_id, lookbook_id, path, key, source, now),
            )
            db.execute("UPDATE lookbooks SET updated = ? WHERE id = ?", (now, lookbook_id))

    def remove_lookbook_image(self, lookbook_id: str, image_id: str) -> bool:
        with self._transaction() as db:
            done = db.execute(
                "DELETE FROM lookbook_images WHERE lookbook = ? AND id = ?", (lookbook_id, image_id)
            )
            if done.rowcount:
                db.execute("UPDATE lookbooks SET updated = ? WHERE id = ?", (time.time(), lookbook_id))
        return done.rowcount > 0

    def lookbook_images(self, lookbook_id: str) -> list[dict[str, Any]]:
        """Oldest first: the order the pictures were added in."""
        with self._lock:
            rows = self._db.execute(
                "SELECT id, path, key, source, added FROM lookbook_images WHERE lookbook = ? ORDER BY added, rowid",
                (lookbook_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_lookbook_code(self, code: str, lookbook_id: str, data: dict[str, Any]) -> None:
        """Record a version of a Lookbook. A code never changes what it stands for."""
        with self._lock:
            self._db.execute(
                "INSERT OR IGNORE INTO lookbook_codes (code, lookbook, data, created) VALUES (?, ?, ?, ?)",
                (code, lookbook_id, json.dumps(data, ensure_ascii=False), time.time()),
            )

    def get_lookbook_code(self, code: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT code, lookbook, data, created FROM lookbook_codes WHERE code = ?", (code.lower(),)
            ).fetchone()
        if row is None:
            return None
        return {
            "code": row["code"],
            "lookbook": row["lookbook"],
            "created": row["created"],
            **json.loads(row["data"]),
        }

    def favorite_images(self, limit: int = 24) -> list[tuple[Job, JobImage]]:
        """The pictures people have favourited, newest first."""
        with self._lock:
            rows = self._db.execute(
                """SELECT data FROM jobs WHERE status = ? AND data LIKE '%"favorite": true%'
                   ORDER BY created DESC, rowid DESC LIMIT ?""",
                (JobStatus.DONE.value, limit * 4),
            ).fetchall()
        found: list[tuple[Job, JobImage]] = []
        for row in rows:
            job = Job.from_dict(json.loads(row["data"]))
            found.extend((job, image) for image in job.images if image.favorite)
        return found[:limit]

    # ------------------------------------------------------------------- gallery ----

    def jobs_page(
        self,
        *,
        before: float | None = None,
        limit: int = 40,
        search: str | None = None,
        ids: Sequence[str] | None = None,
    ) -> list[Job]:
        """Jobs for a gallery, newest first: those created before ``before``, matching ``search``.

        ``search`` is words that must all appear in the prompt, in any order.
        """
        clauses: list[str] = []
        values: list[Any] = []
        if before is not None:
            clauses.append("created < ?")
            values.append(before)
        for word in (search or "").split():
            clauses.append("json_extract(data, '$.prompt') LIKE ? ESCAPE '\\'")
            escaped = word.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            values.append(f"%{escaped}%")
        if ids is not None:
            if not ids:
                return []
            clauses.append(f"id IN ({', '.join('?' * len(ids))})")
            values.extend(ids)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._db.execute(
                f"SELECT data FROM jobs {where} ORDER BY created DESC, rowid DESC LIMIT ?", (*values, limit)
            ).fetchall()
        return [Job.from_dict(json.loads(row["data"])) for row in rows]

    def job_days(self) -> list[tuple[str, int]]:
        """(day, number of jobs) for every day something was made, newest first — a gallery's timeline."""
        with self._lock:
            rows = self._db.execute(
                """SELECT date(created, 'unixepoch', 'localtime') AS day, COUNT(*) AS n FROM jobs
                   WHERE status = ? GROUP BY day ORDER BY day DESC""",
                (JobStatus.DONE.value,),
            ).fetchall()
        return [(row["day"], row["n"]) for row in rows]

    def children_of(self, job_id: str) -> list[Job]:
        with self._lock:
            rows = self._db.execute(
                "SELECT data FROM jobs WHERE parent = ? ORDER BY created", (normalize_job_id(job_id),)
            ).fetchall()
        return [Job.from_dict(json.loads(row["data"])) for row in rows]

    def set_trashed(self, items: Sequence[tuple[str, int]], trashed: bool) -> None:
        now = time.time()
        with self._transaction() as db:
            for job, image in items:
                if trashed:
                    db.execute(
                        "INSERT OR IGNORE INTO trash (job, image, created) VALUES (?, ?, ?)",
                        (job, image, now),
                    )
                else:
                    db.execute("DELETE FROM trash WHERE job = ? AND image = ?", (job, image))

    def trashed(self) -> set[tuple[str, int]]:
        with self._lock:
            return {(row["job"], row["image"]) for row in self._db.execute("SELECT job, image FROM trash")}

    def add_upload(self, upload_id: str, user: str, path: str, key: str, name: str = "") -> dict[str, Any]:
        now = time.time()
        with self._lock:
            self._db.execute(
                "INSERT INTO uploads (id, user, path, key, name, created) VALUES (?, ?, ?, ?, ?, ?)",
                (upload_id, user, path, key, name, now),
            )
        return {"id": upload_id, "path": path, "key": key, "name": name, "created": now}

    def get_upload(self, upload_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT id, path, key, name, created FROM uploads WHERE id = ?", (upload_id,)
            ).fetchone()
        return dict(row) if row else None

    def find_upload(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT id, path, key, name, created FROM uploads WHERE key = ?", (key,)
            ).fetchone()
        return dict(row) if row else None

    def list_uploads(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, path, key, name, created FROM uploads ORDER BY created DESC, rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_upload(self, upload_id: str) -> bool:
        with self._lock:
            return self._db.execute("DELETE FROM uploads WHERE id = ?", (upload_id,)).rowcount > 0

    def create_folder(self, folder_id: str, user: str, name: str) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            self._db.execute(
                "INSERT INTO folders (id, user, name, created) VALUES (?, ?, ?, ?)",
                (folder_id, user, name, now),
            )
        return {"id": folder_id, "name": name, "created": now, "count": 0}

    def list_folders(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                """SELECT f.id, f.name, f.created, COUNT(i.job) AS count FROM folders f
                   LEFT JOIN folder_items i ON i.folder = f.id GROUP BY f.id ORDER BY f.name COLLATE NOCASE"""
            ).fetchall()
        return [dict(row) for row in rows]

    def rename_folder(self, folder_id: str, name: str) -> bool:
        with self._lock:
            return (
                self._db.execute("UPDATE folders SET name = ? WHERE id = ?", (name, folder_id)).rowcount > 0
            )

    def delete_folder(self, folder_id: str) -> bool:
        with self._transaction() as db:
            db.execute("DELETE FROM folder_items WHERE folder = ?", (folder_id,))
            return db.execute("DELETE FROM folders WHERE id = ?", (folder_id,)).rowcount > 0

    def set_in_folder(self, folder_id: str, items: Sequence[tuple[str, int]], present: bool) -> None:
        now = time.time()
        with self._transaction() as db:
            for job, image in items:
                if present:
                    db.execute(
                        "INSERT OR IGNORE INTO folder_items (folder, job, image, added) VALUES (?, ?, ?, ?)",
                        (folder_id, job, image, now),
                    )
                else:
                    db.execute(
                        "DELETE FROM folder_items WHERE folder = ? AND job = ? AND image = ?",
                        (folder_id, job, image),
                    )

    def folder_items(self, folder_id: str | None = None) -> dict[tuple[str, int], list[str]]:
        """Which folders each image is in. With ``folder_id``, only that folder's images."""
        where, values = ("WHERE folder = ?", (folder_id,)) if folder_id else ("", ())
        found: dict[tuple[str, int], list[str]] = {}
        with self._lock:
            for row in self._db.execute(f"SELECT folder, job, image FROM folder_items {where}", values):
                found.setdefault((row["job"], row["image"]), []).append(row["folder"])
        return found

    # --------------------------------------------------------------------- state ----

    def get_state(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._db.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set_state(self, key: str, value: Any) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO state (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, json.dumps(value)),
            )
