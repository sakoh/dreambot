"""Who may use this Dreambot.

A Dreambot is one person's Mac and GPU, so it is private by default: nobody can use it until
its owner says so. Ownership is claimed by *pairing* — ``dreambot pair`` on the Mac prints a
short one-time code, and whoever sends ``/pair <code>`` to the bot becomes an owner. Owners can
then let other people in with ``/allow``.

People are identified as ``<frontend>:<id>`` (``discord:123``, ``telegram:456``,
``imessage:+15551234567``). Lists from the config file and lists changed at runtime are merged;
runtime changes live in the database so they survive restarts.
"""

from __future__ import annotations

import secrets
import time

from dreambot.config import Config
from dreambot.core.store import Store

PAIRING_TTL_SECONDS = 15 * 60
#: The Mac's own iMessage account: messages its owner sends from their own devices.
LOCAL_IMESSAGE_USER = "me"
#: Who the web app acts as. It listens on this Mac only, so whoever reaches it is at the Mac.
LOCAL_WEB_USER = "web:local"
OWNER = "owner"
MEMBER = "member"


def normalize_user(user: str) -> str:
    """Canonical form of a user id: frontend lowercased, phone numbers without punctuation."""
    frontend, _, ident = user.strip().partition(":")
    frontend = frontend.lower()
    ident = ident.strip()
    if frontend == "imessage" and ident != LOCAL_IMESSAGE_USER:
        if "@" in ident:
            ident = ident.lower()
        else:
            digits = "".join(ch for ch in ident if ch.isdigit())
            ident = ("+" if ident.startswith("+") else "") + digits
    return f"{frontend}:{ident}"


class Access:
    def __init__(self, config: Config, store: Store):
        self._config = config
        self._store = store

    # -------------------------------------------------------------------- roles ----

    def owners(self) -> set[str]:
        configured = {normalize_user(user) for user in self._config.access.owners}
        return configured | set(self._store.get_state("access.owners", []))

    def members(self) -> set[str]:
        configured = {normalize_user(user) for user in self._config.access.allowed}
        return configured | set(self._store.get_state("access.allowed", []))

    def role(self, user: str) -> str | None:
        """``"owner"``, ``"member"`` or ``None`` (not allowed)."""
        user = normalize_user(user)
        # The terminal on the Mac, its own iMessage account and its web app (which only answers
        # this Mac) are the owner by definition.
        if (
            user.startswith("console:")
            or user in (f"imessage:{LOCAL_IMESSAGE_USER}", LOCAL_WEB_USER)
            or user in self.owners()
        ):
            return OWNER
        if user in self.members() or self._config.access.open:
            return MEMBER
        return None

    def is_allowed(self, user: str) -> bool:
        return self.role(user) is not None

    def is_owner(self, user: str) -> bool:
        return self.role(user) == OWNER

    @property
    def has_owner(self) -> bool:
        return bool(self.owners())

    # ------------------------------------------------------------------ changes ----

    def _add(self, key: str, user: str) -> None:
        current = set(self._store.get_state(key, []))
        current.add(normalize_user(user))
        self._store.set_state(key, sorted(current))

    def add_owner(self, user: str) -> None:
        self._add("access.owners", user)

    def allow(self, user: str) -> None:
        self._add("access.allowed", user)

    def revoke(self, user: str) -> bool:
        """Remove someone added at runtime. People listed in the config file stay."""
        user = normalize_user(user)
        removed = False
        for key in ("access.allowed", "access.owners"):
            current = set(self._store.get_state(key, []))
            if user in current:
                current.discard(user)
                self._store.set_state(key, sorted(current))
                removed = True
        return removed

    # ------------------------------------------------------------------ pairing ----

    def new_pairing_code(self) -> str:
        """Create a one-time code. Whoever sends it to the bot within 15 minutes becomes an owner."""
        code = f"{secrets.randbelow(10**6):06d}"
        self._store.set_state("access.pairing", {"code": code, "expires": time.time() + PAIRING_TTL_SECONDS})
        return code

    def pair(self, user: str, code: str) -> bool:
        pending = self._store.get_state("access.pairing")
        if not pending or time.time() > pending["expires"]:
            return False
        if not secrets.compare_digest(str(pending["code"]), code.strip()):
            return False
        self._store.set_state("access.pairing", None)
        self.add_owner(user)
        return True

    def refusal(self, user: str) -> str:
        """What to tell someone who is not allowed in."""
        user = normalize_user(user)
        if not self.has_owner:
            return (
                "This Dreambot hasn't been claimed yet. On the Mac it runs on, run `dreambot pair` "
                "and send me the code it prints: /pair 123456"
            )
        return f"This Dreambot is private. Ask its owner to let you in with: /allow {user}"
