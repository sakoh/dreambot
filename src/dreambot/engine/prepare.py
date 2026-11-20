"""Downloading a model's weights. Run as ``python -m dreambot.engine.prepare <model> [quant]``.

This is the one place Dreambot touches the network on purpose. It runs as its own short-lived
process (started by ``dreambot models pull``) with downloads enabled, while the long-lived
inference worker always runs offline.

Three cases:

* The mflux community publishes a pre-quantized export of the model: download that repository.
* The model names the repos of its weights itself (the video model, which mflux does not
  run): download the files it reads from the one that suits this Mac. What it needs that is
  not on Hugging Face is fetched from where its author publishes it, and kept only if it is
  the very file that is expected (:func:`fetch_assets`).
* Otherwise the weights live in the original upstream repository, which usually contains far
  more than is needed (several variants, other frameworks). mflux knows exactly which files it
  wants, so the model is constructed once here and mflux fetches just those.

``python -m dreambot.engine.prepare lora <library folder> <request as JSON>`` adds a LoRA to
the library the same way (see :mod:`dreambot.engine.loras`).
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from dreambot.engine import loras, weights
from dreambot.engine.loras import Card, LoraError
from dreambot.engine.registry import REGISTRY, Asset, ModelSpec, UnknownModel

#: Seconds a download from outside Hugging Face may take to answer.
FETCH_TIMEOUT = 60.0


def download(spec: ModelSpec, quant: str | None = None) -> weights.Weights:
    """Fetch weights for ``spec`` and return what is now installed."""
    from huggingface_hub import snapshot_download

    if spec.exports:
        if quant is not None and quant not in spec.exports:
            raise SystemExit(f"{spec.id} is published as {', '.join(spec.exports)}; there is no {quant}.")
        repo, _ = weights.recommended_repo(spec, [quant] if quant else None)
        for wanted, patterns in (
            (repo, list(spec.files) or None),
            *((helper, None) for helper in spec.helper_repos),
        ):
            try:
                print(f"Downloading {wanted} ...", flush=True)
                snapshot_download(repo_id=wanted, allow_patterns=patterns)
            except Exception as exc:
                if type(exc).__name__ != "GatedRepoError":
                    raise
                raise SystemExit(
                    f"{wanted} is gated. Accept its license at https://huggingface.co/{wanted} "
                    "and sign in with `hf auth login`, then try again."
                ) from exc
        fetch_assets(spec)
    elif spec.slug:
        preference = [quant] if quant else list(weights.quant_preference())
        last_error: Exception | None = None
        for level in preference:
            repo = weights.community_repo(spec, level)
            assert repo is not None
            try:
                print(f"Downloading {repo} ...", flush=True)
                snapshot_download(repo_id=repo)
                break
            except Exception as exc:  # try the next quantization level if this one is not published
                if type(exc).__name__ not in ("RepositoryNotFoundError", "GatedRepoError"):
                    raise
                last_error = exc
                if type(exc).__name__ == "GatedRepoError":
                    raise SystemExit(
                        f"{repo} is gated. Accept its license at https://huggingface.co/{repo} "
                        "and sign in with `hf auth login`, then try again."
                    ) from exc
        else:
            raise SystemExit(f"No community export of {spec.id} could be downloaded: {last_error}")
    else:
        _construct(spec)
    if "briaai/FIBO-vlm" in spec.helper_repos:
        # FIBO takes structured JSON prompts; this model writes them from plain text.
        from mflux.models.fibo_vlm.model.fibo_vlm import FiboVLM

        print("Fetching FIBO's prompt-writing model (briaai/FIBO-vlm) ...", flush=True)
        FiboVLM()

    found = weights.resolve(spec)
    if found is None:
        raise SystemExit(
            f"The download finished but {spec.id} still looks incomplete in the Hugging Face cache."
        )
    return found


def fetch_assets(spec: ModelSpec, folder: Path | None = None) -> None:
    """Fetch what a model needs that is not on Hugging Face, into Dreambot's own folder for such files.

    A file is kept only if its SHA-256 is the one the registry names. A PyTorch checkpoint is
    kept as safetensors: the same arrays, read out of it without PyTorch and without running
    anything the file says to run (:mod:`dreambot.engine.wan.taehv`).
    """
    folder = folder or weights.assets_dir()
    for asset in spec.assets:
        if weights.asset_path(asset, folder):
            continue
        print(f"Downloading {asset.url} ...", flush=True)
        with tempfile.TemporaryDirectory(prefix="dreambot-asset-") as scratch:
            fetched = Path(scratch) / (Path(urlparse(asset.url).path).name or "download")
            try:
                with (
                    urllib.request.urlopen(asset.url, timeout=FETCH_TIMEOUT) as answer,
                    open(fetched, "wb") as out,
                ):
                    shutil.copyfileobj(answer, out)
            except OSError as exc:
                raise SystemExit(f"{asset.url} could not be downloaded ({exc}).") from exc
            _keep(asset, fetched, folder / asset.name)


def _keep(asset: Asset, fetched: Path, target: Path) -> None:
    """Put a downloaded file in its place if it is what was expected."""
    digest = hashlib.sha256()
    with open(fetched, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != asset.sha256:
        raise SystemExit(
            f"What came from {asset.url} is not the file that was expected (its SHA-256 is "
            f"{digest.hexdigest()}), so it was not kept."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    if fetched.suffix == ".pth" and target.suffix == ".safetensors":
        from dreambot.engine.wan import taehv

        try:
            taehv.convert(fetched, target)
        except taehv.CheckpointError as exc:
            raise SystemExit(str(exc)) from exc
    else:
        shutil.move(str(fetched), target)


def _construct(spec: ModelSpec) -> None:
    """Instantiate the model with the network on, so mflux downloads exactly what it needs."""
    from mflux.models.common.config.model_config import AVAILABLE_MODELS

    if not spec.variants:
        raise SystemExit(
            f"{spec.id} has no runnable operation in Dreambot yet, so there is nothing to prepare."
        )
    variant = next(iter(spec.variants.values()))
    module_name, _, class_name = variant.cls.partition(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    kwargs = dict(variant.init)
    if "model_config" in inspect.signature(cls).parameters:
        kwargs["model_config"] = AVAILABLE_MODELS[spec.mflux_key]
    print(f"Fetching the files {spec.title} needs from {spec.repo} ...", flush=True)
    try:
        cls(**kwargs)
    except Exception as exc:
        if type(exc).__name__ == "GatedRepoError" or "gated" in str(exc).lower():
            raise SystemExit(
                f"{spec.repo} is gated. Accept its license at https://huggingface.co/{spec.repo} "
                "and sign in with `hf auth login`, then try again."
            ) from exc
        raise


class HuggingFace:
    """Hugging Face and mflux, as adding a LoRA needs them (a :class:`dreambot.engine.loras.Hub`)."""

    def card(self, repo: str) -> Card:
        from huggingface_hub import HfApi

        try:
            info = HfApi().model_info(repo, files_metadata=True)
        except Exception as exc:
            kind = type(exc).__name__
            if kind == "GatedRepoError":
                raise LoraError(
                    f"{repo} is gated. Accept its terms at https://huggingface.co/{repo} and sign in "
                    "with `hf auth login`, then try again."
                ) from exc
            if kind in ("RepositoryNotFoundError", "HFValidationError"):
                raise LoraError(f"There is no model called {repo} on Hugging Face.") from exc
            raise LoraError(f"Hugging Face could not be reached for {repo} ({kind}: {exc}).") from exc
        data = info.card_data.to_dict() if info.card_data else {}
        base = data.get("base_model") or []
        return Card(
            repo=info.id or repo,
            files=tuple((item.rfilename, item.size or 0) for item in info.siblings or ()),
            base_models=tuple(map(str, [base] if isinstance(base, str) else base)),
            trigger=str(data.get("instance_prompt") or "").strip(),
            license=str(data.get("license") or ""),
        )

    def fetch_file(self, repo: str, filename: str, folder: Path) -> Path:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo_id=repo, filename=filename, local_dir=str(folder)))

    def takes_loras(self, base: ModelSpec) -> bool:
        """Whether mflux's class for the model accepts LoRAs at all (a few families do not)."""
        if not base.variants:
            return False
        module_name, _, class_name = next(iter(base.variants.values())).cls.partition(":")
        cls = getattr(importlib.import_module(module_name), class_name)
        return "lora_paths" in inspect.signature(cls).parameters


def add_lora(folder: Path, request: dict) -> int:
    """Add a LoRA to the library and say what it is now called."""
    try:
        entry = loras.install(folder, loras.Wanted(**request), hub=HuggingFace())
    except LoraError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(f"Added the LoRA {entry.name}" + (f", for {entry.base}." if entry.base else "."))
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: python -m dreambot.engine.prepare <model> [q3|q4|q5|q6|q8|bf16]", file=sys.stderr)
        return 2
    if argv[0] == "lora" and len(argv) == 3:
        return add_lora(Path(argv[1]), json.loads(argv[2]))
    if argv[0].lower() in ("describe", "vision"):
        from huggingface_hub import snapshot_download

        repo = argv[1] if len(argv) > 1 else weights.DEFAULT_VISION_MODEL
        print(f"Downloading the vision model {repo} ...", flush=True)
        snapshot_download(repo_id=repo)
        found_model = weights.find_vision_model(repo)
        print(
            f"Installed: {found_model}"
            if found_model
            else "The download finished but the model looks incomplete."
        )
        return 0 if found_model else 1
    try:
        spec = REGISTRY.get(argv[0])
    except UnknownModel as exc:
        print(exc, file=sys.stderr)
        return 2
    quant = argv[1] if len(argv) > 1 else None
    if quant is not None and quant not in weights.QUANTS:
        print(f"Unknown quantization '{quant}'. Choose one of: {', '.join(weights.QUANTS)}", file=sys.stderr)
        return 2
    found = download(spec, quant)
    print(f"Installed {spec.id}: {found.repo} ({found.label}, {found.size_gb:.1f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
