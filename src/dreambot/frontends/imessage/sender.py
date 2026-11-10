"""Sending iMessages through the Messages app.

Apple offers no API for sending iMessages, so — like every local iMessage tool — Dreambot
asks the Messages app to do it via AppleScript. That needs the Automation permission (macOS
prompts for it the first time) and a logged-in desktop session.

Hard-won details baked in here:

* Messages reads outgoing files most reliably from inside its own attachments folder, so files
  are staged under ``~/Library/Messages/Attachments/dreambot/`` first (falling back to
  ``~/Pictures``).
* A chat that exists in the database is sometimes unknown to the running app; the script then
  addresses the participant directly.
* AppleScript can send one attachment per message, so a grid of four goes out as four messages.
* Message text and paths travel as script *arguments*, never spliced into the script source.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

log = logging.getLogger(__name__)

STAGE_DIR = Path.home() / "Library" / "Messages" / "Attachments" / "dreambot"
FALLBACK_STAGE_DIR = Path.home() / "Pictures" / "Dreambot" / ".outbox"
STAGE_MAX_AGE = 2 * 24 * 3600
SEND_TIMEOUT = 60.0

SCRIPT = """\
on run argv
    set {theText, filePath, chatGuid, theHandle} to argv
    set phase to "pre"
    try
        set theFile to missing value
        if filePath is not "" then set theFile to (POSIX file filePath) as alias
        tell application "Messages"
            try
                set target to chat id chatGuid
            on error number errorNumber
                if errorNumber is not -1728 or theHandle is "" then error number errorNumber
                set target to participant theHandle of (1st account whose service type = iMessage)
            end try
            if theText is not "" then
                set phase to "sent"
                send theText to target
            end if
            if theFile is not missing value then
                set phase to "sent"
                send theFile to target
            end if
        end tell
        return "ok"
    on error errorMessage number errorNumber
        return "fail" & tab & phase & tab & errorNumber & tab & errorMessage
    end try
end run
"""

_EXPLANATIONS = {
    -1743: (
        "macOS has not allowed Dreambot to control Messages. Allow it in System Settings → Privacy & Security → "
        "Automation (enable Messages for the program running Dreambot)."
    ),
    -1728: "Messages does not know that conversation. Send it a message by hand once, then try again.",
    -1712: "Messages did not answer in time. Make sure the Messages app is open and signed in.",
    -600: "The Messages app is not running. Open it and sign in to iMessage.",
    -609: "The Messages app quit unexpectedly. Open it again.",
}


class SendError(Exception):
    """A message could not be handed to Messages."""

    def __init__(self, message: str, code: int | None = None, maybe_sent: bool = False):
        super().__init__(message)
        self.code = code
        #: True when the failure happened after the send was issued; resending could duplicate it.
        self.maybe_sent = maybe_sent


Runner = Callable[[list[str]], Awaitable[str]]


async def run_osascript(arguments: list[str]) -> str:
    """Run :data:`SCRIPT` with ``arguments`` and return what it printed."""
    process = await asyncio.create_subprocess_exec(
        "/usr/bin/osascript",
        "-",
        *arguments,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(SCRIPT.encode()), SEND_TIMEOUT)
    except TimeoutError as exc:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        raise SendError(_EXPLANATIONS[-1712], -1712, maybe_sent=True) from exc
    if process.returncode != 0:
        detail = stderr.decode(errors="replace").strip()
        code = next((number for number in _EXPLANATIONS if str(number) in detail), None)
        raise SendError(_EXPLANATIONS.get(code, f"AppleScript failed: {detail}"), code)
    return stdout.decode(errors="replace").strip()


class Sender:
    def __init__(
        self,
        *,
        stage_dir: Path = STAGE_DIR,
        fallback_dir: Path = FALLBACK_STAGE_DIR,
        run: Runner = run_osascript,
        pause: float = 1.0,
    ):
        self.stage_dir = stage_dir
        self.fallback_dir = fallback_dir
        self._run = run
        #: Seconds to wait after each attachment so consecutive images arrive in order.
        self.pause = pause
        self._lock = asyncio.Lock()

    def stage(self, source: Path | str) -> Path:
        """Copy a file to where Messages can read it, under a name unique to this send."""
        source = Path(source)
        name = f"{source.stem[:40]}-{uuid.uuid4().hex[:8]}{source.suffix.lower() or '.png'}"
        for base in (self.stage_dir, self.fallback_dir):
            try:
                folder = base / uuid.uuid4().hex[:12]
                folder.mkdir(parents=True, exist_ok=True)
                target = folder / name
                shutil.copyfile(source, target)
                return target
            except OSError as exc:
                log.debug("imessage: cannot stage in %s: %s", base, exc)
        raise SendError(
            "Dreambot could not copy the image to a folder Messages can read. "
            "Check that it has Full Disk Access (System Settings → Privacy & Security)."
        )

    def cleanup(self, max_age: float = STAGE_MAX_AGE) -> None:
        """Delete staged copies old enough that Messages has certainly uploaded them."""
        cutoff = time.time() - max_age
        for base in (self.stage_dir, self.fallback_dir):
            try:
                folders = list(base.iterdir())
            except OSError:
                continue
            for folder in folders:
                with contextlib.suppress(OSError):
                    if folder.is_dir() and folder.stat().st_mtime < cutoff:
                        shutil.rmtree(folder, ignore_errors=True)

    async def _send(self, text: str, file_path: str, chat_guid: str, handle: str) -> None:
        # One send at a time: Messages processes Apple events serially, and order matters.
        async with self._lock:
            result = await self._run([text, file_path, chat_guid, handle])
        if result == "ok":
            return
        parts = result.split("\t", 3)
        if parts[0] != "fail":
            raise SendError(f"Unexpected answer from Messages: {result[:200]}")
        phase = parts[1] if len(parts) > 1 else "pre"
        try:
            code: int | None = int(parts[2])
        except (IndexError, ValueError):
            code = None
        detail = parts[3] if len(parts) > 3 else ""
        message = _EXPLANATIONS.get(code or 0, f"Messages refused to send ({code}): {detail}".strip())
        raise SendError(message, code, maybe_sent=phase == "sent")

    async def send_text(self, chat_guid: str, handle: str | None, text: str) -> None:
        if text:
            await self._send(text, "", chat_guid, handle or "")

    async def send_file(self, chat_guid: str, handle: str | None, path: Path | str) -> str:
        """Send one file. Returns the name it was sent under, which identifies it in the database."""
        staged = self.stage(path)
        await self._send("", str(staged), chat_guid, handle or "")
        if self.pause:
            await asyncio.sleep(self.pause)
        return staged.name
