"""Finding model weights that are already on disk.

Dreambot shares the standard Hugging Face cache with mflux and every other HF-based tool, so
weights are never stored twice and models a user already downloaded just work.

Generation runs with the network disabled, so a model is only usable if a *complete* snapshot
is present locally. This module finds those snapshots without importing ``huggingface_hub``,
``mflux`` or ``mlx`` — it is plain filesystem inspection, cheap enough to run on every request.

Three kinds of weights are recognised for a model, in order of preference:

1. pre-quantized exports published by the mflux community
   (``mflux-community/<slug>-mflux-<quant>``),
2. third-party mflux exports of the same upstream model (``<anyone>/<Upstream-Name>-mflux-...``),
3. the original upstream repository (quantized on the fly when requested).

A model that mflux does not run (the video model) names the repos its weights are in itself,
one per quantization (``ModelSpec.exports``), and only those count. The few files a model
needs that are not on Hugging Face at all (``ModelSpec.assets``) are kept in a folder of
Dreambot's own (:func:`assets_dir`), and the model is installed only when they are there too.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from dreambot.engine.registry import Asset, ModelSpec

COMMUNITY_ORG = "mflux-community"
QUANTS = ("q8", "q6", "q5", "q4", "q3", "bf16", "fp8")
_QUANT_RE = re.compile(r"-mflux-(q\d|bf16|fp8)\b", re.IGNORECASE)
_SHARD_RE = re.compile(r"(?P<stem>.+)-(?P<number>\d{5})-of-(?P<total>\d{5})\.safetensors")


def hf_cache_dir() -> Path:
    """The Hugging Face hub cache directory, honouring the standard environment overrides."""
    if cache := os.environ.get("HF_HUB_CACHE"):
        return Path(cache).expanduser()
    if home := os.environ.get("HF_HOME"):
        return Path(home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def assets_dir() -> Path:
    """Where the model files that do not come from Hugging Face are kept (``DREAMBOT_ASSETS_DIR`` moves it)."""
    if folder := os.environ.get("DREAMBOT_ASSETS_DIR"):
        return Path(folder).expanduser()
    return Path.home() / ".cache" / "dreambot"


def asset_path(asset: Asset, folder: Path | None = None) -> Path | None:
    """Where one of those files is on disk, if it is."""
    path = (folder or assets_dir()) / asset.name
    try:
        return path if path.is_file() and path.stat().st_size > 0 else None
    except OSError:
        return None


def physical_memory_gb() -> float:
    """This Mac's memory, in the units Apple sells it in (a "96 GB" Mac reports 96)."""
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3
    except (ValueError, OSError):
        return 16.0


def quant_preference(memory_gb: float | None = None) -> tuple[str, ...]:
    """Quantization levels from most to least preferred for a machine with this much memory."""
    memory = physical_memory_gb() if memory_gb is None else memory_gb
    if memory >= 48:
        return ("q8", "q6", "q5", "q4", "bf16", "q3", "fp8")
    if memory >= 24:
        return ("q6", "q5", "q4", "q8", "q3", "bf16", "fp8")
    return ("q4", "q3", "q5", "q6", "q8", "bf16", "fp8")


@dataclass(frozen=True)
class Weights:
    """One usable set of weights for a model."""

    #: Hugging Face repo id the files belong to.
    repo: str
    #: Quantization of the stored files (``q8`` ... ``bf16``), or ``None`` for original upstream weights.
    quant: str | None
    #: Local snapshot directory.
    path: Path
    size_bytes: int
    #: True for the model's original upstream repository.
    upstream: bool = False

    @property
    def location(self) -> str | None:
        """What to hand the worker as ``GenerateRequest.weights``.

        Exports are addressed by local path, which keeps loading fully offline. Upstream
        weights are addressed by leaving this unset: mflux then loads its pinned revision of
        the model's own repository from the cache.
        """
        return None if self.upstream else str(self.path)

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1000**3

    @property
    def label(self) -> str:
        return self.quant or "original"


def community_repo(spec: ModelSpec, quant: str) -> str | None:
    """Repo id of the community pre-quantized export of a model, if one is published."""
    return f"{COMMUNITY_ORG}/{spec.slug}-mflux-{quant}" if spec.slug else None


def _repo_dir(cache: Path, repo_id: str) -> Path:
    return cache / f"models--{repo_id.replace('/', '--')}"


def _snapshots(cache: Path, repo_id: str) -> list[Path]:
    root = _repo_dir(cache, repo_id) / "snapshots"
    try:
        found = [path for path in root.iterdir() if path.is_dir()]
    except OSError:
        return []
    return sorted(found, key=lambda path: path.stat().st_mtime, reverse=True)


def snapshot_is_complete(snapshot: Path) -> bool:
    """Whether a cached snapshot can be loaded without downloading anything.

    A snapshot counts as complete when it holds weight files, none of its entries is a dangling
    symlink (an interrupted download), every shard named by a shard index is present, and the
    repo has no partially downloaded blobs.
    """
    weight_files = 0
    try:
        for path in snapshot.rglob("*"):
            if path.is_dir():
                continue
            if not path.exists():  # dangling symlink into blobs/
                return False
            if path.suffix == ".safetensors":
                weight_files += 1
            elif path.name.endswith(".safetensors.index.json") and not _index_satisfied(path):
                return False
    except OSError:
        return False
    if weight_files == 0:
        return False
    blobs = snapshot.parent.parent / "blobs"
    return not (blobs.is_dir() and any(blobs.glob("*.incomplete")))


def _index_satisfied(index_path: Path) -> bool:
    try:
        weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map")
    except (OSError, ValueError, AttributeError):
        return False
    if not isinstance(weight_map, dict) or not weight_map:
        return False
    names = {name for name in weight_map.values() if isinstance(name, str)}
    present = [name for name in names if (index_path.parent / name).is_file()]
    if len(present) == len(names) == len(set(weight_map.values())):
        return True
    # An index left over from another upload of the model names none of the files that are
    # there (mlx-community/gemma-3-12b-it-4bit: two shards, and the index of the original
    # five). The shards' own numbering then says whether they are all there.
    return not present and _shards_all_present(index_path.parent)


def _shards_all_present(folder: Path) -> bool:
    """Whether the numbered shards in a folder (``model-00001-of-00002.safetensors``) form whole sets."""
    sets: dict[tuple[str, int], set[int]] = {}
    try:
        names = [path.name for path in folder.iterdir()]
    except OSError:
        return False
    for name in names:
        if match := _SHARD_RE.fullmatch(name):
            sets.setdefault((match["stem"], int(match["total"])), set()).add(int(match["number"]))
    return bool(sets) and all(numbers == set(range(1, total + 1)) for (_, total), numbers in sets.items())


def _size(snapshot: Path) -> int:
    total = 0
    for path in snapshot.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def _first_complete(cache: Path, repo_id: str) -> Path | None:
    for snapshot in _snapshots(cache, repo_id):
        if snapshot_is_complete(snapshot):
            return snapshot
    return None


def _third_party_exports(cache: Path, spec: ModelSpec) -> list[tuple[str, str | None]]:
    """Cached mflux exports of this model published outside the community org.

    They are recognised by the upstream repo's name followed by ``-mflux-``, e.g.
    ``someone/Ming-Image-0.1-Design-mflux-q8`` for ``inclusionAI/Ming-Image-0.1-Design``.
    """
    upstream_name = spec.repo.rsplit("/", 1)[-1].lower()
    prefix = f"{upstream_name}-mflux-"
    found: list[tuple[str, str | None]] = []
    try:
        entries = sorted(cache.iterdir())
    except OSError:
        return found
    for entry in entries:
        if not entry.name.startswith("models--"):
            continue
        org, _, name = entry.name[len("models--") :].partition("--")
        if not name or org == COMMUNITY_ORG or not name.lower().startswith(prefix):
            continue
        match = _QUANT_RE.search(name)
        found.append((f"{org}/{name}", match.group(1).lower() if match else None))
    return found


def _has_files(snapshot: Path, names: Sequence[str]) -> bool:
    """Whether a snapshot holds every file a model reads from it. A name may be a pattern."""
    for name in names:
        if "*" in name:
            if not any(path.is_file() for path in snapshot.glob(name)):
                return False
        elif not (snapshot / name).is_file():
            return False
    return True


def _export(cache: Path, repo_id: str, files: Sequence[str]) -> Path | None:
    """The snapshot of one of a model's own exports, if what the model reads from it is there."""
    for snapshot in _snapshots(cache, repo_id):
        if snapshot_is_complete(snapshot) and _has_files(snapshot, files):
            return snapshot
    return None


def installed(spec: ModelSpec, cache: Path | None = None) -> list[Weights]:
    """Every complete set of weights for ``spec`` found in the cache (unordered)."""
    cache = cache or hf_cache_dir()
    found: list[Weights] = []
    if not all(_first_complete(cache, repo) for repo in spec.helper_repos):
        return found  # e.g. FIBO without its prompt-writing model cannot run at all
    kept = [asset_path(asset) for asset in spec.assets]
    if not all(kept):
        return found
    if spec.exports:
        for quant, repo in spec.exports.items():
            if snapshot := _export(cache, repo, spec.files):
                own = sum(_file_size(path) for name in spec.files for path in snapshot.glob(name))
                own += sum(_file_size(path) for path in kept if path)
                found.append(Weights(repo=repo, quant=quant, path=snapshot, size_bytes=own))
        return found  # the original weights are in a form its library does not read
    for quant in QUANTS:
        repo = community_repo(spec, quant)
        if repo and (snapshot := _first_complete(cache, repo)):
            found.append(Weights(repo=repo, quant=quant, path=snapshot, size_bytes=_size(snapshot)))
    for repo, quant in _third_party_exports(cache, spec):
        if snapshot := _first_complete(cache, repo):
            found.append(Weights(repo=repo, quant=quant, path=snapshot, size_bytes=_size(snapshot)))
    # Original weights only count when every extra repository the model needs (a ControlNet,
    # an adapter) is present too; community exports already bundle those.
    extras = [_first_complete(cache, repo) for repo in spec.extra_repos]
    if all(extras) and (snapshot := _upstream(cache, spec)):
        own = sum(_file_size(snapshot / name) for name in spec.files) if spec.files else _size(snapshot)
        size = own + sum(_size(extra) for extra in extras if extra)
        found.append(Weights(repo=spec.repo, quant=None, path=snapshot, size_bytes=size, upstream=True))
    return found


def _upstream(cache: Path, spec: ModelSpec) -> Path | None:
    """The snapshot of the original repo holding this model's weights, if they are all there.

    Some repos hold several models (SeedVR2 3B and 7B share one): having the repo is then not
    the same as having the model, and the files the model reads are checked by name.
    """
    for snapshot in _snapshots(cache, spec.repo):
        if snapshot_is_complete(snapshot) and _has_files(snapshot, spec.files):
            return snapshot
    return None


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def resolve(
    spec: ModelSpec,
    *,
    prefer: Sequence[str] | None = None,
    override: str | None = None,
    cache: Path | None = None,
) -> Weights | None:
    """Pick the weights to use for ``spec``, or ``None`` if the model is not installed.

    ``override`` forces a specific location: a local directory or a repo id that must already
    be in the cache. ``prefer`` orders quantization levels (default: by this Mac's memory).
    """
    cache = cache or hf_cache_dir()
    if override:
        local = Path(override).expanduser()
        if local.is_dir():
            match = _QUANT_RE.search(local.name)
            return Weights(
                repo=override,
                quant=match.group(1).lower() if match else None,
                path=local,
                size_bytes=_size(local),
            )
        snapshot = _first_complete(cache, override)
        if snapshot is None:
            return None
        match = _QUANT_RE.search(override)
        return Weights(
            repo=override,
            quant=match.group(1).lower() if match else None,
            path=snapshot,
            size_bytes=_size(snapshot),
        )

    candidates = installed(spec, cache)
    if not candidates:
        return None
    order = list(prefer if prefer is not None else quant_preference())

    def rank(weights: Weights) -> tuple[int, int, int]:
        # Quantized exports before original weights (they load faster and need less memory);
        # then by preferred quantization; community exports before third-party ones.
        quant_rank = order.index(weights.quant) if weights.quant in order else len(order)
        return (1 if weights.upstream else 0, quant_rank, 0 if weights.repo.startswith(COMMUNITY_ORG) else 1)

    return min(candidates, key=rank)


def recommended_repo(spec: ModelSpec, prefer: Sequence[str] | None = None) -> tuple[str, str | None]:
    """The repo (and quantization) ``dreambot models pull`` should download for ``spec``."""
    order = prefer if prefer is not None else quant_preference()
    for quant in order:
        if spec.exports:
            if quant in spec.exports:
                return spec.exports[quant], quant
        elif repo := community_repo(spec, quant):
            return repo, quant
    if spec.exports:
        # None of the wanted quantizations is published: the smallest one that is.
        quant = next(level for level in ("q4", "q8", "bf16", *spec.exports) if level in spec.exports)
        return spec.exports[quant], quant
    return spec.repo, None


# --------------------------------------------------------------------------------------
# Vision-language models (for /describe and prompt help)
# --------------------------------------------------------------------------------------

#: Downloaded by ``dreambot models pull describe``: small, fast and good at captioning.
DEFAULT_VISION_MODEL = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
#: Name fragments that identify an MLX vision-language model in the cache, best first.
_VISION_HINTS = (
    "qwen3-vl",
    "qwen2.5-vl",
    "qwen3.5",
    "gemma-4",
    "gemma-3",
    "internvl",
    "pixtral",
    "llava",
    "lfm2-vl",
    "smolvlm",
    "fastvlm",
    "qwen2-vl",
)


def find_vision_model(override: str | None = None, cache: Path | None = None) -> Path | None:
    """A locally installed MLX vision-language model, or ``None``.

    ``override`` names a specific one (a folder, or a repo id that is in the cache). Otherwise
    the cache is searched for MLX conversions of well-known vision models.
    """
    cache = cache or hf_cache_dir()
    if override:
        local = Path(override).expanduser()
        if local.is_dir():
            return local
        return _first_complete(cache, override)
    try:
        entries = sorted(entry.name for entry in cache.iterdir() if entry.name.startswith("models--"))
    except OSError:
        return None
    candidates: list[tuple[int, str]] = []
    for entry in entries:
        org, _, name = entry[len("models--") :].partition("--")
        lowered = name.lower()
        # MLX conversions live in mlx-community or say so in their name.
        if org != "mlx-community" and "mlx" not in lowered:
            continue
        for rank, hint in enumerate(_VISION_HINTS):
            if hint in lowered:
                candidates.append((rank, f"{org}/{name}"))
                break
    for _, repo in sorted(candidates):
        snapshot = _first_complete(cache, repo)
        if snapshot is not None and (snapshot / "config.json").exists():
            return snapshot
    return None
