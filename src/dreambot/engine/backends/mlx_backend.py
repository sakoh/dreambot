"""The real backend: image models through mflux, captions through mlx-vlm, text through mlx-lm.

Everything here runs on MLX. Models stay loaded between requests (loading a 10-30 GB model
per job would dominate latency) in a small least-recently-used cache that is trimmed against
a memory budget.

All MLX work happens on the worker's main thread: MLX streams are thread-local, so models
must be created and evaluated on the same thread.
"""

from __future__ import annotations

import gc
import importlib
import inspect
import json
import math
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from dreambot import __version__
from dreambot.core.text import parse_numbered
from dreambot.engine.backends.base import BackendError, JobCancelled, JobContext, require_inputs
from dreambot.engine.protocol import (
    DescribeRequest,
    ErrorKind,
    GenerateRequest,
    ImageDone,
    Loaded,
    LoadedModel,
    Loading,
    Op,
    Preview,
    Progress,
    StatusReport,
    TextRequest,
)
from dreambot.engine.registry import REGISTRY, ModelSpec, UnknownModel, Variant

GB = 1000**3


def _import(path: str) -> Any:
    module_name, _, attr = path.partition(":")
    return getattr(importlib.import_module(module_name), attr)


def _snap(value: float, multiple: int) -> int:
    """Round to the nearest multiple, halves up (Python's round() would round halves to even)."""
    return max(multiple, int(value / multiple + 0.5) * multiple)


def _physical_memory_gb() -> float:
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / GB
    except (ValueError, OSError):
        return 16.0


class _StopEarly(BaseException):
    """Unwinds a denoising loop at the ``--stop`` step, carrying the partially denoised image."""

    def __init__(self, image: Image.Image):
        self.image = image


class _Probed(BaseException):
    """Unwinds a generation right before its first denoising step (used to read the schedule)."""


@dataclass
class _ImageJob:
    """Per-image state the callback hub reads while one image is being denoised."""

    req: GenerateRequest
    ctx: JobContext
    index: int
    preview_path: Path
    previews_ok: bool = True
    #: Stop before denoising: the caller only wants to learn the noise schedule.
    probe: bool = False


class _Hub:
    """Registered once on each loaded model; relays mflux's loop callbacks to the running job."""

    def __init__(self, model: Any, latent_cls: Any | None):
        self.model = model
        self.latent_cls = latent_cls
        self.job: _ImageJob | None = None
        #: The noise schedule (sigma per step, ending at 0) of the most recent generation.
        self.sigmas: list[float] | None = None

    def call_before_loop(
        self, seed, prompt, latents, config, canny_image=None, depth_image=None, control_images=None
    ):
        try:
            self.sigmas = [float(value) for value in config.scheduler.sigmas.tolist()]
        except Exception:
            self.sigmas = None
        if self.job is not None:
            self.job.ctx.check_cancelled()
            if self.job.probe:
                raise _Probed()

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
        job = self.job
        if job is None:
            return
        total = int(config.num_inference_steps)
        step = int(t) + 1
        job.ctx.emit(Progress(id=job.req.id, index=job.index, step=step, total=total))
        job.ctx.check_cancelled()

        if job.req.stop_at is not None:
            stop_step = max(1, math.ceil(total * job.req.stop_at))
            if stop_step < total and step >= stop_step:
                raise _StopEarly(self.decode(latents, config, seed, prompt))

        every = job.req.preview_every
        if every and job.previews_ok and step < total and step % every == 0 and self.latent_cls is not None:
            try:
                self.decode(latents, config, seed, prompt).save(job.preview_path)
            except Exception as exc:  # a preview must never fail the job
                job.previews_ok = False
                print(f"[dreambot] previews disabled for {job.req.model}: {type(exc).__name__}: {exc}")
            else:
                job.ctx.emit(
                    Preview(
                        id=job.req.id, index=job.index, step=step, total=total, path=str(job.preview_path)
                    )
                )

    def decode(self, latents, config, seed: int, prompt: str) -> Image.Image:
        """Decode intermediate latents to an image (same recipe as mflux's StepwiseHandler)."""
        from mflux.utils.image_util import ImageUtil

        if self.latent_cls is None:
            raise BackendError("This model cannot decode intermediate steps.", ErrorKind.INVALID)
        unpacked = self.latent_cls.unpack_latents(latents=latents, height=config.height, width=config.width)
        vae = getattr(self.model, "vae", None)
        if vae is None:  # pixel-space models: the sample already is the image
            decoded = unpacked
        elif hasattr(vae, "decode_packed_latents") and unpacked.shape[1] > getattr(
            vae, "latent_channels", 32
        ):
            decoded = vae.decode_packed_latents(unpacked)
        else:
            decoded = vae.decode(unpacked)
        generated = ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=getattr(self.model, "bits", None),
            generation_time=0,
        )
        return generated.image


@dataclass
class _Entry:
    key: tuple
    spec_id: str
    model: Any
    hub: _Hub | None = None
    last_used: float = field(default_factory=time.monotonic)
    #: Noise schedules seen so far, by (width, height, steps).
    schedules: dict[tuple, list[float]] = field(default_factory=dict)


class MlxBackend:
    name = "mlx"

    def __init__(self) -> None:
        import mlx.core as mx

        self._mx = mx
        self._models: OrderedDict[tuple, _Entry] = OrderedDict()
        ram = _physical_memory_gb()
        #: Loaded models are evicted (least recently used first) once MLX holds more than this.
        self.budget_gb = float(os.environ.get("DREAMBOT_MEMORY_BUDGET_GB", ram * 0.5))
        # How many models may stay loaded (the memory budget still applies). Two lets the small
        # language model that art-directs prompts sit next to the image model.
        default_models = 3 if ram >= 64 else 2 if ram >= 24 else 1
        self.max_models = int(os.environ.get("DREAMBOT_MAX_MODELS", default_models))
        # MLX keeps freed buffers in a cache that otherwise grows without bound across jobs.
        cache_gb = float(os.environ.get("DREAMBOT_MLX_CACHE_GB", min(8.0, ram * 0.1)))
        mx.set_cache_limit(int(cache_gb * GB))

    def versions(self) -> dict[str, str]:
        from importlib.metadata import PackageNotFoundError, version

        found = {"dreambot": __version__}
        for package in ("mlx", "mflux", "mlx-vlm", "mlx-lm"):
            try:
                found[package] = version(package)
            except PackageNotFoundError:
                continue
        return found

    # ------------------------------------------------------------------ generate ----

    def generate(self, req: GenerateRequest, ctx: JobContext) -> None:
        try:
            spec = REGISTRY.get(req.model)
        except UnknownModel as exc:
            raise BackendError(str(exc), ErrorKind.INVALID) from exc
        variant = spec.variants.get(req.op)
        if variant is None:
            supported = ", ".join(op.value for op in spec.ops) or "nothing yet"
            raise BackendError(
                f"{spec.id} does not support '{req.op.value}' (it supports: {supported}).", ErrorKind.INVALID
            )
        require_inputs(req, variant.max_refs)
        for path in [req.image, req.mask, *req.images]:
            if path and not Path(path).is_file():
                raise BackendError(f"Input image not found: {path}", ErrorKind.INVALID)
        if req.stop_at is not None and req.stop_at < 1.0 and variant.latent is None:
            raise BackendError(f"{spec.id} cannot stop early (--stop).", ErrorKind.INVALID)

        entry = self._load(spec, variant, req, ctx)
        out_dir = Path(req.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            strength = self._resolve_strength(entry, spec, variant, req, ctx)
            for index, seed in enumerate(req.seeds):
                ctx.check_cancelled()
                started = time.perf_counter()
                job = _ImageJob(
                    req=req,
                    ctx=ctx,
                    index=index,
                    preview_path=out_dir / f"{req.basename}_{index}_preview.png",
                )
                if entry.hub is not None:
                    entry.hub.job = job
                try:
                    image = self._invoke(entry.model, spec, variant, req, seed, strength).image
                except _StopEarly as stop:
                    image = stop.image
                finally:
                    if entry.hub is not None:
                        entry.hub.job = None

                path = out_dir / f"{req.basename}_{index}.png"
                image.save(path, pnginfo=self._metadata(spec, req, seed, image))
                job.preview_path.unlink(missing_ok=True)
                ctx.emit(
                    ImageDone(
                        id=req.id,
                        index=index,
                        seed=seed,
                        path=str(path),
                        width=image.width,
                        height=image.height,
                        seconds=time.perf_counter() - started,
                    )
                )
        except JobCancelled:
            # Buffers of the abandoned denoising pass are only reclaimed when asked.
            gc.collect()
            self._mx.clear_cache()
            raise
        finally:
            entry.last_used = time.monotonic()
        self._trim(keep=entry.key)

    def _resolve_strength(
        self, entry: _Entry, spec: ModelSpec, variant: Variant, req: GenerateRequest, ctx: JobContext
    ) -> float | None:
        """Turn the requested noise level into this model's ``strength`` for this image size.

        mflux expresses image-to-image as "skip this fraction of the steps". How much of the
        source survives that depends on the model's noise schedule, which shifts with resolution:
        the same fraction that gives a subtle variation at 1 MP redraws the whole picture at 4 MP.
        So Dreambot asks for a noise level instead and picks the step whose noise is closest.
        """
        if req.noise is None or "strength" not in variant.args:
            return req.strength
        fallback = max(0.0, min(1.0, 1.0 - req.noise))
        hub = entry.hub
        if hub is None:
            return fallback
        accepted = inspect.signature(entry.model.generate_image).parameters
        width, height = self._dimensions(spec, req, accepted)
        steps = req.steps if req.steps is not None else spec.steps
        key = (width, height, steps)
        sigmas = entry.schedules.get(key)
        if sigmas is None:
            # Run the generation up to (not into) its first step to learn the schedule.
            hub.job = _ImageJob(
                req=req, ctx=ctx, index=0, preview_path=Path(req.out_dir) / "probe.png", probe=True
            )
            hub.sigmas = None
            try:
                self._invoke(entry.model, spec, variant, req, req.seeds[0], 0.5)
            except _Probed:
                pass
            finally:
                hub.job = None
            sigmas = hub.sigmas
            if sigmas:
                entry.schedules[key] = sigmas
        if not sigmas or len(sigmas) < 3:
            return fallback
        last = len(sigmas) - 1  # sigmas has one more entry than there are steps
        # Starting at step k noises the image to sigmas[k]. Step 0 would ignore the image.
        start = min(range(1, last), key=lambda index: abs(sigmas[index] - req.noise))
        return (start + 0.5) / last

    def _invoke(
        self,
        model: Any,
        spec: ModelSpec,
        variant: Variant,
        req: GenerateRequest,
        seed: int,
        strength: float | None = None,
    ) -> Any:
        fn = model.generate_image
        accepted = inspect.signature(fn).parameters
        kwargs: dict[str, Any] = {}

        def put(name: str, value: Any) -> None:
            if value is not None and name in accepted:
                kwargs[name] = value

        put("seed", seed)
        if "prompt" in accepted:
            kwargs["prompt"] = req.prompt
        width, height = self._dimensions(spec, req, accepted)
        put("width", width)
        put("height", height)
        put("num_inference_steps", req.steps if req.steps is not None else spec.steps)
        put("guidance", req.guidance if req.guidance is not None else spec.guidance)
        put("negative_prompt", req.negative_prompt)
        put("scheduler", req.extra.get("scheduler"))
        for request_field, keyword in variant.args.items():
            value = (
                strength
                if request_field == "strength" and strength is not None
                else getattr(req, request_field)
            )
            put(keyword, value)
        # Power-user escape hatch: extra keyword arguments the model's generate_image understands.
        for name, value in (req.extra.get("kwargs") or {}).items():
            put(name, value)

        finish = None
        if variant.adapter is not None:
            adapter = _ADAPTERS.get(variant.adapter)
            if adapter is None:
                raise BackendError(
                    f"{spec.id} '{req.op.value}' is not wired up yet (adapter '{variant.adapter}').",
                    ErrorKind.INVALID,
                )
            finish = adapter(kwargs, req, spec, accepted)
        result = fn(**kwargs)
        if finish is not None:
            result.image = finish(result.image)
        return result

    @staticmethod
    def _dimensions(spec: ModelSpec, req: GenerateRequest, accepted: Any) -> tuple[int | None, int | None]:
        """Use the requested size, or else follow the source image so aspect ratios survive."""
        if req.width and req.height:
            return _snap(req.width, spec.multiple), _snap(req.height, spec.multiple)
        if req.op is Op.TXT2IMG or req.op is Op.UPSCALE:
            return req.width, req.height
        source = req.image or (req.images[0] if req.images else None)
        if source is None:
            return req.width, req.height
        # Models whose size parameters default to None derive the output size themselves.
        for name in ("width", "height"):
            parameter = accepted.get(name)
            if parameter is not None and parameter.default is None:
                return None, None
        with Image.open(source) as image:
            return _snap(image.width, spec.multiple), _snap(image.height, spec.multiple)

    @staticmethod
    def _metadata(spec: ModelSpec, req: GenerateRequest, seed: int, image: Image.Image) -> PngInfo:
        info = PngInfo()
        info.add_text(
            "dreambot",
            json.dumps(
                {
                    "software": f"dreambot {__version__}",
                    "model": spec.id,
                    "operation": req.op.value,
                    "prompt": req.prompt,
                    "negative_prompt": req.negative_prompt,
                    "seed": seed,
                    "steps": req.steps if req.steps is not None else spec.steps,
                    "guidance": req.guidance if req.guidance is not None else spec.guidance,
                    "width": image.width,
                    "height": image.height,
                    "strength": req.strength,
                    "loras": [lora.model_dump() for lora in req.loras],
                },
                ensure_ascii=False,
            ),
        )
        return info

    # --------------------------------------------------------------- model cache ----

    def _load(self, spec: ModelSpec, variant: Variant, req: GenerateRequest, ctx: JobContext) -> _Entry:
        key = (
            variant.cls,
            spec.mflux_key,
            req.weights,
            req.quantize,
            tuple((lora.path, lora.scale) for lora in req.loras),
            tuple(sorted(variant.init.items())),
        )
        entry = self._models.get(key)
        if entry is not None:
            self._models.move_to_end(key)
            return entry

        ctx.emit(Loading(id=req.id, model=spec.id))
        started = time.perf_counter()
        while len(self._models) >= self.max_models:
            self._evict_oldest()

        from mflux.models.common.config.model_config import AVAILABLE_MODELS

        cls = _import(variant.cls)
        accepted = inspect.signature(cls).parameters
        kwargs: dict[str, Any] = dict(variant.init)
        if "model_config" in accepted:
            kwargs["model_config"] = AVAILABLE_MODELS[spec.mflux_key]
        if req.weights is not None:
            kwargs["model_path"] = req.weights
        if req.quantize is not None:
            if "quantize" not in accepted:
                raise BackendError(f"{spec.id} cannot be quantized on the fly.", ErrorKind.INVALID)
            kwargs["quantize"] = req.quantize
        if req.loras:
            if "lora_paths" not in accepted:
                raise BackendError(f"{spec.id} does not support LoRAs.", ErrorKind.INVALID)
            kwargs["lora_paths"] = [lora.path for lora in req.loras]
            kwargs["lora_scales"] = [lora.scale for lora in req.loras]

        try:
            model = cls(**kwargs)
        except Exception as exc:
            if req.loras and (problem := _lora_problem(exc, spec)):
                # mflux refuses an adapter it cannot apply in full rather than drawing without it.
                raise BackendError(problem, ErrorKind.INVALID, detail=f"{type(exc).__name__}: {exc}") from exc
            if isinstance(exc, FileNotFoundError) or _is_offline_miss(exc):
                raise BackendError(
                    f"The weights for {spec.id} are not installed. Run `dreambot models pull {spec.id}`.",
                    ErrorKind.NOT_INSTALLED,
                    detail=f"{type(exc).__name__}: {exc}",
                ) from exc
            raise

        hub: _Hub | None = None
        callbacks = getattr(model, "callbacks", None)
        if callbacks is not None and hasattr(callbacks, "register"):
            hub = _Hub(model, _import(variant.latent) if variant.latent else None)
            callbacks.register(hub)

        entry = _Entry(key=key, spec_id=spec.id, model=model, hub=hub)
        self._models[key] = entry
        ctx.emit(Loaded(id=req.id, model=spec.id, seconds=time.perf_counter() - started))
        return entry

    def _evict_oldest(self) -> None:
        if not self._models:
            return
        _, entry = self._models.popitem(last=False)
        if entry.hub is not None:
            entry.hub.model = None
        del entry
        gc.collect()
        self._mx.clear_cache()

    def _trim(self, keep: tuple) -> None:
        """Evict idle models while MLX holds more than the memory budget."""
        while len(self._models) > 1 and self._mx.get_active_memory() / GB > self.budget_gb:
            oldest = next(iter(self._models))
            if oldest == keep:
                self._models.move_to_end(oldest)
                continue
            self._evict_oldest()

    def unload(self, model: str | None) -> None:
        for key in [k for k, e in self._models.items() if model is None or e.spec_id == model]:
            entry = self._models.pop(key)
            if entry.hub is not None:
                entry.hub.model = None
            del entry
        gc.collect()
        self._mx.clear_cache()

    def status(self, request_id: str) -> StatusReport:
        mx = self._mx
        now = time.monotonic()
        return StatusReport(
            id=request_id,
            loaded=[
                LoadedModel(key=repr(entry.key[:2]), model=entry.spec_id, idle_seconds=now - entry.last_used)
                for entry in self._models.values()
            ],
            active_memory_gb=mx.get_active_memory() / GB,
            peak_memory_gb=mx.get_peak_memory() / GB,
            cache_memory_gb=mx.get_cache_memory() / GB,
        )

    # ---------------------------------------------------------------- text tasks ----

    def _vision(self, weights: str | None, request_id: str, ctx: JobContext) -> tuple[Any, Any, Any]:
        """Load (or reuse) the vision-language model. Returns (model, processor, config)."""
        if not weights:
            raise BackendError(
                "No vision model is installed. Run `dreambot models pull describe`.", ErrorKind.NOT_INSTALLED
            )
        key = ("vision", weights)
        entry = self._models.get(key)
        if entry is not None:
            self._models.move_to_end(key)
            entry.last_used = time.monotonic()
            return entry.model
        try:
            from mlx_vlm import load
            from mlx_vlm.utils import load_config
        except ImportError as exc:
            raise BackendError("mlx-vlm is not installed.", ErrorKind.MISSING_DEPENDENCY) from exc
        name = Path(weights).parent.parent.name.removeprefix("models--").replace("--", "/") or weights
        ctx.emit(Loading(id=request_id, model=name))
        started = time.perf_counter()
        while len(self._models) >= self.max_models:
            self._evict_oldest()
        try:
            model, processor = load(weights)
            config = load_config(weights)
        except Exception as exc:
            if _is_offline_miss(exc) or isinstance(exc, FileNotFoundError):
                raise BackendError(
                    "The vision model is not installed. Run `dreambot models pull describe`.",
                    ErrorKind.NOT_INSTALLED,
                    detail=f"{type(exc).__name__}: {exc}",
                ) from exc
            raise
        entry = _Entry(key=key, spec_id=name, model=(model, processor, config))
        self._models[key] = entry
        ctx.emit(Loaded(id=request_id, model=name, seconds=time.perf_counter() - started))
        return entry.model

    def _chat(
        self,
        loaded: tuple[Any, Any, Any],
        prompt: str,
        *,
        system: str | None = None,
        image: str | None = None,
        max_tokens: int,
        temperature: float,
        seed: int | None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
    ) -> str:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        model, processor, config = loaded
        messages: Any = prompt
        if system:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        formatted = apply_chat_template(processor, config, messages, num_images=1 if image else 0)
        options: dict[str, Any] = {"max_tokens": max_tokens, "temperature": temperature, "verbose": False}
        if seed is not None:
            options["seed"] = seed
        if top_p is not None:
            options["top_p"] = top_p
        if repetition_penalty is not None:
            options["repetition_penalty"] = repetition_penalty
        result = generate(model, processor, formatted, image=[image] if image else None, **options)
        return str(getattr(result, "text", result)).strip()

    def describe(self, req: DescribeRequest, ctx: JobContext) -> list[str]:
        if not Path(req.image).is_file():
            raise BackendError(f"Input image not found: {req.image}", ErrorKind.INVALID)
        loaded = self._vision(req.weights, req.id, ctx)
        ctx.check_cancelled()
        instruction = req.instruction or DESCRIBE_INSTRUCTION.format(count=req.count)
        answer = self._chat(
            loaded,
            instruction,
            image=req.image,
            max_tokens=req.max_tokens * req.count,
            temperature=0.6,
            seed=req.seed,
        )
        prompts = parse_numbered(answer, req.count)
        if not prompts:
            raise BackendError(
                "The vision model did not return usable prompts. Try again.",
                ErrorKind.INTERNAL,
                detail=answer,
            )
        self._trim(keep=("vision", req.weights))
        return prompts

    def text(self, req: TextRequest, ctx: JobContext) -> list[str]:
        loaded = self._vision(req.weights, req.id, ctx)
        ctx.check_cancelled()
        answer = self._chat(
            loaded,
            req.prompt,
            system=req.system,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            seed=req.seed,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
        )
        return [answer]


DESCRIBE_INSTRUCTION = (
    "You write prompts for a text-to-image model. Study the image and write {count} different prompts that "
    "would each recreate it. Each prompt is a single line of at most 60 words describing the subject, the "
    "setting, the medium or visual style, the lighting and the mood. Make the prompts differ in emphasis: "
    "one on the subject's details, one on artistic style, one on composition and camera, one on atmosphere. "
    "Do not mention that it is an image or a prompt. Output exactly {count} lines numbered 1. to {count}. "
    "and nothing else."
)


def _lora_problem(exc: BaseException, spec: ModelSpec) -> str | None:
    """What to tell the person when a model failed to load because of a LoRA, else ``None``."""
    text = str(exc)
    match = re.search(r"[\w.\-]+\.safetensors", text)
    file = match.group(0) if match else "A LoRA"
    if "No LoRA layers were applied" in text or "could not be applied" in text:
        return f"{file} does not fit {spec.title}: it was trained on a different model."
    if "DoRA" in text:
        return f"{file} is a DoRA adapter, which is not supported."
    if "LoRA" in text or "LoKr" in text:
        if isinstance(exc, FileNotFoundError):
            return "A LoRA this job needs is not on this Mac (pictures are made offline)."
        return f"{file} could not be used as a LoRA on {spec.title}."
    return None


def _is_offline_miss(exc: BaseException) -> bool:
    """True when an exception chain is Hugging Face refusing to download because we are offline."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        name = type(current).__name__
        if name in {
            "LocalEntryNotFoundError",
            "OfflineModeIsEnabled",
        } or "outgoing traffic has been disabled" in str(current):
            return True
        current = current.__cause__ or current.__context__
    return False


# --------------------------------------------------------------------------------------
# Adapters: per-model fix-ups where the generic request -> keyword mapping is not enough.
#
# Signature: adapter(kwargs, request, spec, accepted_parameters). It edits ``kwargs`` in place
# and may return a function that post-processes the generated PIL image.
#
# The FLUX upscaler, Z-Image ControlNet and FIBO adapters follow what mflux's own command-line
# tools do for those models; they have not been run on real weights by Dreambot's tests yet.
# --------------------------------------------------------------------------------------


def _adapt_single_ref(kwargs: dict, req: GenerateRequest, spec: ModelSpec, accepted: Any) -> None:
    """Models that take exactly one reference image as a plain path rather than a list."""
    if isinstance(kwargs.get("image_path"), list):
        kwargs["image_path"] = kwargs["image_path"][0]


def _adapt_list_image(kwargs: dict, req: GenerateRequest, spec: ModelSpec, accepted: Any) -> None:
    """Models whose inpainting source goes in the ``image_paths`` list."""
    if isinstance(kwargs.get("image_paths"), str):
        kwargs["image_paths"] = [kwargs["image_paths"]]


def _adapt_seedvr2(kwargs: dict, req: GenerateRequest, spec: ModelSpec, accepted: Any) -> None:
    """SeedVR2 takes no prompt; ``resolution`` is a scale factor ("2x") or a target short side."""
    from mflux.utils.scale_factor import ScaleFactor

    kwargs.pop("prompt", None)
    target = req.extra.get("resolution")
    if target is not None:
        kwargs["resolution"] = int(target)
    else:
        scale = req.extra.get("scale", 2)
        kwargs["resolution"] = ScaleFactor.parse(f"{scale:g}x")
    kwargs["softness"] = float(req.extra.get("softness", 0.0))


def _adapt_flux_upscale(kwargs: dict, req: GenerateRequest, spec: ModelSpec, accepted: Any) -> None:
    """The FLUX ControlNet upscaler renders at the target size, guided by the small image."""
    if req.width is None or req.height is None:
        scale = float(req.extra.get("scale", 2))
        with Image.open(req.image) as source:  # type: ignore[arg-type]
            kwargs["width"] = _snap(source.width * scale, spec.multiple)
            kwargs["height"] = _snap(source.height * scale, spec.multiple)
    kwargs.setdefault("controlnet_strength", 0.4)  # mflux's default for this model


def _adapt_zimage_control(kwargs: dict, req: GenerateRequest, spec: ModelSpec, accepted: Any) -> None:
    """Z-Image's union ControlNet takes typed controls; mflux derives the control map itself."""
    from mflux.models.z_image.variants.controlnet.control_types import ControlSpec, ControlType

    kind = str(req.extra.get("control", "depth"))
    try:
        control_type = ControlType(kind)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in ControlType)
        raise BackendError(
            f"Unknown control type '{kind}'. Choose one of: {allowed}.", ErrorKind.INVALID
        ) from exc
    strength = req.strength if req.strength is not None else 1.0
    kwargs["controls"] = [ControlSpec(type=control_type, image_path=Path(req.image), strength=strength)]  # type: ignore[arg-type]
    if req.width is None or req.height is None:
        with Image.open(req.image) as source:  # type: ignore[arg-type]
            kwargs["width"], kwargs["height"] = (
                _snap(source.width, spec.multiple),
                _snap(source.height, spec.multiple),
            )


_FIBO_PROMPTS: OrderedDict[tuple, str] = OrderedDict()


def _remember(key: tuple, value: str) -> str:
    _FIBO_PROMPTS[key] = value
    while len(_FIBO_PROMPTS) > 32:
        _FIBO_PROMPTS.popitem(last=False)
    return value


def _is_json_object(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except (TypeError, ValueError):
        return False


def _fibo_vlm(quantize: int | None) -> Any:
    from mflux.models.fibo_vlm.model.fibo_vlm import FiboVLM

    try:
        return FiboVLM(quantize=quantize)
    except Exception as exc:
        if _is_offline_miss(exc) or isinstance(exc, FileNotFoundError):
            raise BackendError(
                "FIBO needs its prompt-writing model (briaai/FIBO-vlm). Run `dreambot models pull fibo`.",
                ErrorKind.NOT_INSTALLED,
            ) from exc
        raise


def _adapt_fibo(kwargs: dict, req: GenerateRequest, spec: ModelSpec, accepted: Any) -> None:
    """FIBO is prompted with structured JSON; its own small VLM writes that from plain text."""
    prompt = req.prompt
    if not _is_json_object(prompt):
        key = ("generate", prompt)
        if key not in _FIBO_PROMPTS:
            import mlx.core as mx

            vlm = _fibo_vlm(req.quantize)
            try:
                _remember(key, vlm.generate(prompt=prompt, seed=42))
            finally:
                del vlm
                gc.collect()
                mx.clear_cache()
        prompt = _FIBO_PROMPTS[key]
    kwargs["prompt"] = prompt
    kwargs.setdefault("scheduler", "flow_match_euler_discrete")


def _adapt_fibo_edit(kwargs: dict, req: GenerateRequest, spec: ModelSpec, accepted: Any) -> Any:
    """FIBO Edit: one image, a JSON prompt carrying the edit instruction, and optionally a mask."""
    from mflux.models.fibo.variants.edit.util import FIBO_EDIT_RMBG_DEFAULT_JSON_PROMPT, FiboEditUtil

    if isinstance(kwargs.get("image_path"), list):
        kwargs["image_path"] = kwargs["image_path"][0]
    source = kwargs.get("image_path")
    removes_background = "rmbg" in spec.id
    prompt = req.prompt.strip()

    if not prompt and removes_background:
        json_prompt = FiboEditUtil.ensure_edit_instruction(FIBO_EDIT_RMBG_DEFAULT_JSON_PROMPT)
    elif _is_json_object(prompt):
        json_prompt = FiboEditUtil.ensure_edit_instruction(prompt, edit_instruction=prompt)
    else:
        key = ("edit", prompt, str(source), req.mask)
        if key not in _FIBO_PROMPTS:
            import mlx.core as mx
            from mflux.utils.exif_orientation import open_oriented
            from mflux.utils.image_util import ImageUtil

            image = ImageUtil.load_image(source)
            if req.mask:
                mask = open_oriented(req.mask).convert("L")
                if mask.size != image.size:
                    mask = mask.resize(image.size)
                image = FiboEditUtil._composite_mask_on_image(mask=mask, image=image)
            vlm = _fibo_vlm(req.quantize)
            try:
                _remember(
                    key, vlm.edit(image=image, edit_instruction=prompt, use_mask=bool(req.mask), seed=42)
                )
            finally:
                del vlm
                gc.collect()
                mx.clear_cache()
        json_prompt = _FIBO_PROMPTS[key]
    kwargs["prompt"] = json_prompt
    kwargs.setdefault("scheduler", "flow_match_euler_discrete")

    if removes_background:
        # The model outputs a matte; combine it with the source into a transparent cut-out.
        return lambda matte: FiboEditUtil.build_rgba_composite_image(source, matte)
    return None


_ADAPTERS = {
    "single_ref": _adapt_single_ref,
    "list_image": _adapt_list_image,
    "qwen21_inpaint": _adapt_list_image,
    "seedvr2": _adapt_seedvr2,
    "flux_upscale": _adapt_flux_upscale,
    "zimage_control": _adapt_zimage_control,
    "fibo": _adapt_fibo,
    "fibo_edit": _adapt_fibo_edit,
}
