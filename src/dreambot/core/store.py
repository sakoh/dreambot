"""Persistence: jobs, per-user settings and the mapping from chat messages back to images.

Everything lives in one small SQLite file on the Mac. Calls are synchronous — each is a
sub-millisecond local write — and guarded by a lock so background threads may use the store too.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dreambot.core.jobs import Job, JobStatus, new_job_id, normalize_job_id

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
"""


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
