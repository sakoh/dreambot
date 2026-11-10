"""Bot tokens live in the macOS Keychain, not in a config file.

A token can also be supplied through the environment (``DREAMBOT_DISCORD_TOKEN``,
``DREAMBOT_TELEGRAM_TOKEN``), which takes precedence and is what a launchd service or CI would
use. Nothing here ever prints a token.
"""

from __future__ import annotations

import os

SERVICE = "dreambot"
NAMES = ("discord", "telegram")


def _env_name(name: str) -> str:
    return f"DREAMBOT_{name.upper()}_TOKEN"


def get_token(name: str) -> str | None:
    """The token for a frontend, or ``None`` if none is stored."""
    if value := os.environ.get(_env_name(name), "").strip():
        return value
    try:
        import keyring

        return keyring.get_password(SERVICE, name) or None
    except Exception:
        return None


def set_token(name: str, value: str) -> None:
    import keyring

    keyring.set_password(SERVICE, name, value.strip())


def delete_token(name: str) -> bool:
    import keyring
    from keyring.errors import PasswordDeleteError

    try:
        keyring.delete_password(SERVICE, name)
    except PasswordDeleteError:
        return False
    return True


def token_source(name: str) -> str | None:
    """Where a token would come from (``"environment"`` or ``"keychain"``), without revealing it."""
    if os.environ.get(_env_name(name), "").strip():
        return "environment"
    try:
        import keyring

        return "keychain" if keyring.get_password(SERVICE, name) else None
    except Exception:
        return None
