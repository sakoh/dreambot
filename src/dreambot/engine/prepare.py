"""Downloading a model's weights. Run as ``python -m dreambot.engine.prepare <model> [quant]``.

This is the one place Dreambot touches the network on purpose. It runs as its own short-lived
process (started by ``dreambot models pull``) with downloads enabled, while the long-lived
inference worker always runs offline.

Two cases:

* The mflux community publishes a pre-quantized export of the model: download that repository.
* Otherwise the weights live in the original upstream repository, which usually contains far
  more than is needed (several variants, other frameworks). mflux knows exactly which files it
  wants, so the model is constructed once here and mflux fetches just those.

``python -m dreambot.engine.prepare lora <library folder> <request as JSON>`` adds a LoRA to
the library the same way (see :mod:`dreambot.engine.loras`).
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys
from pathlib import Path

from dreambot.engine import loras, weights
from dreambot.engine.loras import Card, LoraError
from dreambot.engine.registry import REGISTRY, ModelSpec, UnknownModel


def download(spec: ModelSpec, quant: str | None = None) -> weights.Weights:
    """Fetch weights for ``spec`` and return what is now installed."""
    from huggingface_hub import snapshot_download

    if spec.slug:
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
