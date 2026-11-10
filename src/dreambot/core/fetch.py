"""Downloading images that people reference by link.

Links come from chat messages, i.e. from other people, and the download happens on the owner's
Mac inside their home network. So a link is only followed when it resolves to a public address:
otherwise a prompt could be used to pull images off devices on the local network (cameras,
NAS photo libraries) and have the bot post them back.
"""

from __future__ import annotations

import ipaddress
import socket
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import urlsplit

MAX_IMAGE_BYTES = 25 * 1024 * 1024
_TIMEOUT = 20
_SUFFIXES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/heic": ".heic",
    "image/heif": ".heic",
    "image/avif": ".avif",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
}


class FetchError(Exception):
    """A linked image could not be downloaded. The message is written for the person who sent it."""


def _is_public(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError:
        return False
    addresses = {info[4][0] for info in infos}
    if not addresses:
        return False
    for address in addresses:
        ip = ipaddress.ip_address(address.split("%")[0])
        if not ip.is_global:
            return False
    return True


class _NoPrivateRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parts = urlsplit(newurl)
        if parts.scheme not in ("http", "https") or not parts.hostname or not _is_public(parts.hostname):
            raise FetchError("That link redirects somewhere I won't follow.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_image(
    url: str, out_dir: Path, *, allow_private: bool = False, max_bytes: int = MAX_IMAGE_BYTES
) -> Path:
    """Download ``url`` into ``out_dir`` and return the file. Blocking: call it from a thread."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise FetchError("Image links must start with http:// or https://.")
    if not allow_private and not _is_public(parts.hostname):
        raise FetchError(f"I can't download from {parts.hostname}: it isn't a public address.")

    opener = (
        urllib.request.build_opener() if allow_private else urllib.request.build_opener(_NoPrivateRedirects())
    )
    request = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (Macintosh) image-fetch", "Accept": "image/*"}
    )
    try:
        with opener.open(request, timeout=_TIMEOUT) as response:
            content_type = (response.headers.get_content_type() or "").lower()
            declared = response.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > max_bytes:
                raise FetchError(f"That image is larger than {max_bytes // (1024 * 1024)} MB.")
            data = response.read(max_bytes + 1)
    except FetchError:
        raise
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        reason = getattr(exc, "reason", exc)
        raise FetchError(f"I couldn't download {parts.hostname}{parts.path[:40]}: {reason}") from exc
    if len(data) > max_bytes:
        raise FetchError(f"That image is larger than {max_bytes // (1024 * 1024)} MB.")
    if not data:
        raise FetchError("That link returned nothing.")
    if content_type.startswith("text/") or content_type in ("application/json", "application/xml"):
        raise FetchError("That link is a web page, not an image. Use a direct link to the image file.")

    suffix = _SUFFIXES.get(content_type) or Path(parts.path).suffix.lower() or ".img"
    if len(suffix) > 6 or not suffix[1:].isalnum():
        suffix = ".img"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"download-{uuid.uuid4().hex[:12]}{suffix}"
    target.write_bytes(data)
    return target
