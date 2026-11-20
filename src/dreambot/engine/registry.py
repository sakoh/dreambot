"""The model registry: every MLX image and video model Dreambot can drive, and what each one can do.

This module is pure data — it never imports ``mlx`` or ``mflux`` — so the (lightweight) bot
process can plan jobs without loading the inference stack. The worker process resolves the
dotted class paths below against the installed mflux at load time.

Because the table mirrors mflux's own model list, ``tests/test_registry.py`` checks it against
the installed mflux (every mflux model is covered, default step counts match, every class path
imports). An mflux upgrade that adds or moves a model therefore fails a test instead of
silently going unsupported.

mflux has no video models. The one there is, Wan 2.2, is run by Dreambot's own engine
(:mod:`dreambot.engine.wan`, ``runtime="wan"``).

Operations map onto Midjourney features like this:

==========  =====================================================================
``txt2img``  /imagine
``img2img``  variations (V1-V4, Vary Subtle/Strong), creative upscale
``edit``     instruction edits, character / omni / style references, vary region
``inpaint``  vary region with a mask, zoom out, pan, make square
``redux``    image prompts and /blend
``control``  retexture (keep structure, repaint)
``upscale``  Upscale (Subtle)
``txt2vid``  /imagine with ``--video``
``img2vid``  Animate, a start frame, loops
``extend``   Extend
==========  =====================================================================
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from dreambot.engine.protocol import Op

_M = "mflux.models"


@dataclass(frozen=True)
class Variant:
    """How one operation of a model maps onto an mflux class."""

    #: ``module:Class`` of the mflux model class.
    cls: str
    #: Request field -> ``generate_image`` keyword. Fields: image, strength, images, image_weights, mask.
    args: Mapping[str, str] = field(default_factory=dict)
    #: Extra constructor keywords.
    init: Mapping[str, Any] = field(default_factory=dict)
    #: ``module:Class`` of the latent creator, used to decode previews and ``--stop`` images.
    latent: str | None = None
    #: Maximum number of reference images (edit / redux).
    max_refs: int = 0
    #: Name of a special-case handler in the mflux backend, when the generic mapping is not enough.
    adapter: str | None = None
    #: The model takes its prompt as an instruction ("fill the gray area...") rather than as a
    #: description of the finished picture. Matters for inpainting, where both kinds exist.
    instructed: bool = False


@dataclass(frozen=True)
class VideoTraits:
    """What the videos of a model are like."""

    #: Frames per second.
    fps: int = 24
    #: A video has ``1 + step * n`` frames: one, then whole groups of this many.
    step: int = 4
    #: The longest video made in one go, in frames.
    max_frames: int = 121
    #: The frames an extension adds follow from this many at the video's end, so a video needs
    #: at least this many to be extended, and one go adds at most ``max_frames`` less these.
    carry: int = 5
    #: The model can lead a video up to a picture it is given to end on.
    ends: bool = False

    @property
    def added(self) -> int:
        """The most frames one extension adds."""
        return self.max_frames - self.carry


@dataclass(frozen=True)
class Asset:
    """A file a model reads that is published somewhere other than Hugging Face."""

    #: The name it is kept under on this Mac (:func:`dreambot.engine.weights.assets_dir`).
    name: str
    #: Where it is published. The address names one state of the file for good (a commit).
    url: str
    #: The SHA-256 of what is there. Anything else is not kept.
    sha256: str
    #: Its licence, as its publisher states it.
    license: str = ""


@dataclass(frozen=True)
class ModelSpec:
    id: str
    title: str
    family: str
    summary: str
    #: Key in mflux's ``AVAILABLE_MODELS`` (selects the ``ModelConfig``). Empty for a model
    #: that mflux does not run.
    mflux_key: str
    #: Upstream Hugging Face repo with the original weights.
    repo: str
    variants: Mapping[Op, Variant]
    #: ``mflux-community/<slug>-mflux-<quant>`` holds pre-quantized weights when set.
    slug: str | None = None
    #: Additional upstream repos the model needs (ControlNets, adapters).
    extra_repos: tuple[str, ...] = ()
    #: The files this model reads from a repo that holds more than it needs: the upstream repo
    #: of several models (SeedVR2 3B and 7B, the Anima variants), or each of its ``exports``.
    #: The weights then count as installed only if these are present, and only these are
    #: downloaded. A name may be a pattern (``*.json``). Empty: the whole repo is the model.
    files: tuple[str, ...] = ()
    #: Separate models the model cannot run without, whatever export of it is installed
    #: (FIBO's prompt-writing VLM).
    helper_repos: tuple[str, ...] = ()
    #: Files the model cannot run without that are not on Hugging Face.
    assets: tuple[Asset, ...] = ()
    #: Default number of denoising steps.
    steps: int = 25
    #: Default guidance. ``None`` leaves the model's own default in place.
    guidance: float | None = None
    #: The model runs true classifier-free guidance, so a negative prompt has an effect.
    cfg: bool = False
    #: Step-distilled model: the step count is fixed by training and must not be scaled down.
    distilled: bool = False
    #: Width/height must be multiples of this.
    multiple: int = 16
    #: Native square resolution (pixels per side).
    native: int = 1024
    #: License identifier of the upstream weights, as reported by Hugging Face.
    license: str = ""
    #: The upstream repo is gated: fetching the original weights needs an accepted license and a HF login.
    gated: bool = False
    tags: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    #: Exercised end-to-end by Dreambot's own test-suite on real weights.
    verified: bool = False
    #: What runs it: ``mflux``, or ``wan`` (Dreambot's own engine) for the video model.
    runtime: str = "mflux"
    #: The weights as that library reads them, when they are published per quantization in
    #: repos of their own rather than by the mflux community: ``{"q8": "<repo id>", ...}``.
    exports: Mapping[str, str] = field(default_factory=dict)
    #: Set for a video model: what its videos are like.
    video: VideoTraits | None = None

    def supports(self, op: Op) -> bool:
        return op in self.variants

    @property
    def permissive(self) -> bool:
        """True when the weights carry a permissive (Apache/MIT) license."""
        return self.license in PERMISSIVE_LICENSES

    @property
    def ops(self) -> tuple[Op, ...]:
        return tuple(self.variants)


def _v(cls: str, latent: str | None = None, **kw: Any) -> Variant:
    args = kw.pop("args", {})
    init = kw.pop("init", {})
    return Variant(
        cls=f"{_M}.{cls}",
        latent=f"{_M}.{latent}" if latent else None,
        args=MappingProxyType(dict(args)),
        init=MappingProxyType(dict(init)),
        **kw,
    )


_IMG2IMG = {"image": "image_path", "strength": "image_strength"}

# -- latent creators -------------------------------------------------------------------
_L_FLUX = "flux.latent_creator.flux_latent_creator:FluxLatentCreator"
_L_FLUX2 = "flux2.latent_creator.flux2_latent_creator:Flux2LatentCreator"
_L_FLUX3 = "flux3.latent_creator.flux3_latent_creator:Flux3LatentCreator"
_L_ZIMAGE = "z_image.latent_creator:ZImageLatentCreator"
_L_QWEN = "qwen.latent_creator.qwen_latent_creator:QwenLatentCreator"
_L_QWEN21 = "qwen21.latent_creator.qwen21_latent_creator:Qwen21LatentCreator"
_L_QWEN21_EDIT = "qwen21.latent_creator.qwen_image21_latent_creator:QwenImage21LatentCreator"
_L_KREA2 = "krea2.latent_creator:Krea2LatentCreator"
_L_ERNIE = "ernie_image.latent_creator:ErnieLatentCreator"
_L_FIBO = "fibo.latent_creator.fibo_latent_creator:FiboLatentCreator"
_L_FIBO2 = "fibo2.latent_creator.fibo2_latent_creator:Fibo2LatentCreator"
_L_IDEOGRAM = "ideogram4.latent_creator:Ideogram4LatentCreator"
_L_IRIS = "iris.latent_creator.iris_latent_creator:IrisLatentCreator"
_L_MING = "ming_image.latent_creator.ming_latent_creator:MingLatentCreator"
_L_MOSS = "moss_image.latent_creator:MOSSImageLatentCreator"
_L_ANIMA = "anima.latent_creator.anima_latent_creator:AnimaLatentCreator"

# -- model classes ---------------------------------------------------------------------
_FLUX1 = "flux.variants.txt2img.flux:Flux1"
_KONTEXT = "flux.variants.kontext.flux_kontext:Flux1Kontext"
_FILL = "flux.variants.fill.flux_fill:Flux1Fill"
_REDUX = "flux.variants.redux.flux_redux:Flux1Redux"
_DEPTH = "flux.variants.depth.flux_depth:Flux1Depth"
_FLUX_CN = "flux.variants.controlnet.flux_controlnet:Flux1Controlnet"
_FLUX2 = "flux2.variants.txt2img.flux2_klein:Flux2Klein"
_FLUX2_EDIT = "flux2.variants.edit.flux2_klein_edit:Flux2KleinEdit"
_FLUX3 = "flux3.variants.txt2img.flux3_klein:Flux3Klein"
_FLUX3_EDIT = "flux3.variants.edit.flux3_klein_edit:Flux3KleinEdit"
_ZIMAGE = "z_image.variants.z_image:ZImage"
_ZIMAGE_CN = "z_image.variants.controlnet.z_image_turbo_controlnet:ZImageTurboControlnet"
_QWEN = "qwen.variants.txt2img.qwen_image:QwenImage"
_QWEN_EDIT = "qwen.variants.edit.qwen_image_edit:QwenImageEdit"
_QWEN21 = "qwen21.variants.txt2img.qwen_image_21:QwenImage21"
_QWEN21_EDIT = "qwen21.variants.edit.qwen_image_21_edit:QwenImage21Edit"
_QWEN21_CN = "qwen21.variants.controlnet.qwen_image_21_controlnet:QwenImage21Controlnet"
_KREA2 = "krea2.variants.txt2img.krea2:Krea2"
_KREA2_CN = "krea2.variants.controlnet.krea2_controlnet:Krea2Controlnet"
_ERNIE = "ernie_image.variants.txt2img.ernie_image:ErnieImage"
_FIBO = "fibo.variants.txt2img.fibo:FIBO"
_FIBO_EDIT = "fibo.variants.edit.fibo_edit:FIBOEdit"
_FIBO2 = "fibo2.variants.txt2img.fibo2:Fibo2"
_FIBO2_EDIT = "fibo2.variants.edit.fibo2_edit:Fibo2Edit"
_IDEOGRAM = "ideogram4.variants.txt2img.ideogram4:Ideogram4"
_IRIS = "iris.variants.txt2img.iris:Iris"
_MING = "ming_image.variants.ming_image:MingImage"
_LENS = "lens.variants.txt2img.lens_image:LensImage"
_MOSS = "moss_image.variants.txt2img.moss_image:MOSSImage"
_BOOGU = "boogu.variants.txt2img.boogu_image:BooguImage"
_ANIMA = "anima.variants.txt2img.anima:Anima"
_SEEDVR2 = "seedvr2.variants.upscale.seedvr2:SeedVR2"
# The video model is run by Dreambot's own engine, the same way for every operation.
_WAN = Variant(cls="dreambot.engine.backends.wan:WanVideo")

# -- licenses --------------------------------------------------------------------------
# License identifiers exactly as each upstream Hugging Face repo reports them.
_LIC_APACHE = "apache-2.0"
_LIC_MIT = "mit"
_LIC_FLUX1 = "flux-1-dev-non-commercial-license"
_LIC_FLUX2 = "flux-non-commercial-license"
_LIC_FLUX3 = "flux-kommunity-license"
_LIC_KREA2 = "krea-2-community-license"
_LIC_QWEN = "qwen-research"
_LIC_BRIA = "bria-fibo"
_LIC_BRIA_EDIT = "bria-fibo-edit"
_LIC_ANIMA = "circlestone-labs-non-commercial-license"
_LIC_IDEOGRAM = "ideogram-4-non-commercial"
_LIC_OTHER = "other"

#: Licenses that place no field-of-use restrictions on the weights or on generated images.
PERMISSIVE_LICENSES = frozenset({"apache-2.0", "mit"})


def _txt_img(cls: str, latent: str | None) -> dict[Op, Variant]:
    return {Op.TXT2IMG: _v(cls, latent), Op.IMG2IMG: _v(cls, latent, args=_IMG2IMG)}


def _flux2(key: str, title: str, repo: str, slug: str, **kw: Any) -> ModelSpec:
    return ModelSpec(
        id=key,
        title=title,
        family="flux2",
        mflux_key=key,
        repo=repo,
        slug=slug,
        variants={
            **_txt_img(_FLUX2, _L_FLUX2),
            Op.EDIT: _v(_FLUX2_EDIT, _L_FLUX2, args={"images": "image_paths"}, max_refs=4),
        },
        **kw,
    )


def _specs() -> list[ModelSpec]:
    return [
        # ---------------------------------------------------------------- Z-Image ----
        ModelSpec(
            id="z-image-turbo",
            title="Z-Image Turbo",
            family="z-image",
            summary="Fast, photorealistic 6B model from Tongyi; 9 steps. Dreambot's default.",
            mflux_key="z-image-turbo",
            repo="Tongyi-MAI/Z-Image-Turbo",
            slug="z-image-turbo",
            variants=_txt_img(_ZIMAGE, _L_ZIMAGE),
            steps=9,
            distilled=True,
            license=_LIC_APACHE,
            tags=("fast", "photo", "default"),
            aliases=("zimage-turbo", "zit", "turbo"),
            verified=True,
        ),
        ModelSpec(
            id="z-image",
            title="Z-Image",
            family="z-image",
            summary="The undistilled Z-Image base model: slower, honours negative prompts.",
            mflux_key="z-image",
            repo="Tongyi-MAI/Z-Image",
            slug="z-image-base",
            variants=_txt_img(_ZIMAGE, _L_ZIMAGE),
            steps=50,
            guidance=4.0,
            cfg=True,
            license=_LIC_APACHE,
            tags=("quality", "photo"),
            aliases=("zimage", "z-image-base"),
        ),
        ModelSpec(
            id="z-image-turbo-controlnet-union-2.1",
            title="Z-Image Turbo ControlNet Union 2.1",
            family="z-image",
            summary="Z-Image Turbo guided by pose, canny, HED, depth or MLSD control images.",
            mflux_key="z-image-turbo-controlnet-union-2.1",
            repo="Tongyi-MAI/Z-Image-Turbo",
            slug="z-image-turbo-controlnet-union-2-1",
            extra_repos=("alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1",),
            variants={Op.CONTROL: _v(_ZIMAGE_CN, _L_ZIMAGE, adapter="zimage_control")},
            steps=8,
            distilled=True,
            license=_LIC_APACHE,
            tags=("control",),
            aliases=("z-image-controlnet", "z-image-turbo-controlnet", "z-image-controlnet-union-2.1"),
        ),
        # ----------------------------------------------------------------- FLUX.1 ----
        ModelSpec(
            id="schnell",
            title="FLUX.1 schnell",
            family="flux1",
            summary="Black Forest Labs' 4-step FLUX.1 model.",
            mflux_key="schnell",
            repo="black-forest-labs/FLUX.1-schnell",
            slug="flux-1-schnell",
            variants=_txt_img(_FLUX1, _L_FLUX),
            steps=4,
            distilled=True,
            license=_LIC_APACHE,
            gated=True,
            tags=("fast",),
            aliases=("flux-schnell", "flux1-schnell"),
        ),
        ModelSpec(
            id="dev",
            title="FLUX.1 dev",
            family="flux1",
            summary="The guidance-distilled 12B FLUX.1 model; huge LoRA ecosystem.",
            mflux_key="dev",
            repo="black-forest-labs/FLUX.1-dev",
            slug="flux-1-dev",
            variants=_txt_img(_FLUX1, _L_FLUX),
            steps=25,
            guidance=3.5,
            license=_LIC_FLUX1,
            gated=True,
            tags=("quality", "lora"),
            aliases=("flux-dev", "flux1-dev", "flux"),
        ),
        ModelSpec(
            id="krea-dev",
            title="FLUX.1 Krea dev",
            family="flux1",
            summary="FLUX.1 dev tuned by Krea for a natural, less 'AI' photographic look.",
            mflux_key="krea-dev",
            repo="black-forest-labs/FLUX.1-Krea-dev",
            slug="flux-1-krea-dev",
            variants=_txt_img(_FLUX1, _L_FLUX),
            steps=25,
            guidance=3.5,
            license=_LIC_FLUX1,
            gated=True,
            tags=("quality", "photo", "aesthetic"),
            aliases=("dev-krea", "flux-krea"),
        ),
        ModelSpec(
            id="dev-kontext",
            title="FLUX.1 Kontext dev",
            family="flux1",
            summary="Instruction-based editing of a single image.",
            mflux_key="dev-kontext",
            repo="black-forest-labs/FLUX.1-Kontext-dev",
            slug="flux-1-dev-kontext",
            variants={
                Op.EDIT: _v(
                    _KONTEXT, _L_FLUX, args={"images": "image_path"}, max_refs=1, adapter="single_ref"
                )
            },
            steps=25,
            guidance=2.5,
            license=_LIC_FLUX1,
            gated=True,
            tags=("edit",),
            aliases=("kontext", "flux-kontext"),
        ),
        ModelSpec(
            id="dev-fill",
            title="FLUX.1 Fill dev",
            family="flux1",
            summary="Mask-based inpainting and outpainting.",
            mflux_key="dev-fill",
            repo="black-forest-labs/FLUX.1-Fill-dev",
            slug="flux-1-dev-fill",
            variants={
                Op.INPAINT: _v(_FILL, _L_FLUX, args={"image": "image_path", "mask": "masked_image_path"})
            },
            steps=25,
            guidance=30.0,
            license=_LIC_FLUX1,
            gated=True,
            tags=("inpaint",),
            aliases=("fill", "flux-fill"),
        ),
        ModelSpec(
            id="dev-redux",
            title="FLUX.1 Redux dev",
            family="flux1",
            summary="Blends one or more image prompts (with optional text) into new images.",
            mflux_key="dev-redux",
            repo="black-forest-labs/FLUX.1-Redux-dev",
            extra_repos=("black-forest-labs/FLUX.1-dev",),
            variants={
                Op.REDUX: _v(
                    _REDUX,
                    _L_FLUX,
                    args={"images": "redux_image_paths", "image_weights": "redux_image_strengths"},
                    max_refs=5,
                )
            },
            steps=25,
            guidance=3.5,
            license=_LIC_FLUX1,
            gated=True,
            tags=("blend", "image-prompt"),
            aliases=("redux", "flux-redux"),
        ),
        ModelSpec(
            id="dev-depth",
            title="FLUX.1 Depth dev",
            family="flux1",
            summary="Repaints an image while keeping its depth structure.",
            mflux_key="dev-depth",
            repo="black-forest-labs/FLUX.1-Depth-dev",
            slug="flux-1-dev-depth",
            variants={Op.CONTROL: _v(_DEPTH, _L_FLUX, args={"image": "image_path"})},
            steps=25,
            guidance=10.0,
            license=_LIC_FLUX1,
            gated=True,
            tags=("control",),
            aliases=("flux-depth",),
        ),
        ModelSpec(
            id="dev-controlnet-canny",
            title="FLUX.1 dev ControlNet Canny",
            family="flux1",
            summary="FLUX.1 dev guided by the edges of a control image.",
            mflux_key="dev-controlnet-canny",
            repo="black-forest-labs/FLUX.1-dev",
            slug="flux-1-dev-controlnet-canny",
            extra_repos=("InstantX/FLUX.1-dev-Controlnet-Canny",),
            variants={
                Op.CONTROL: _v(
                    _FLUX_CN,
                    _L_FLUX,
                    args={"image": "controlnet_image_path", "strength": "controlnet_strength"},
                )
            },
            steps=25,
            guidance=3.5,
            license=_LIC_FLUX1,
            gated=True,
            tags=("control",),
            aliases=("flux-canny",),
        ),
        ModelSpec(
            id="schnell-controlnet-canny",
            title="FLUX.1 schnell ControlNet Canny",
            family="flux1",
            summary="FLUX.1 schnell guided by the edges of a control image.",
            mflux_key="schnell-controlnet-canny",
            repo="black-forest-labs/FLUX.1-schnell",
            slug="flux-1-schnell-controlnet-canny",
            extra_repos=("InstantX/FLUX.1-dev-Controlnet-Canny",),
            variants={
                Op.CONTROL: _v(
                    _FLUX_CN,
                    _L_FLUX,
                    args={"image": "controlnet_image_path", "strength": "controlnet_strength"},
                )
            },
            steps=4,
            distilled=True,
            license=_LIC_FLUX1,
            gated=True,
            tags=("control", "fast"),
        ),
        ModelSpec(
            id="dev-controlnet-upscaler",
            title="FLUX.1 dev ControlNet Upscaler",
            family="flux1",
            summary="Prompt-guided, detail-inventing upscaler built on FLUX.1 dev.",
            mflux_key="dev-controlnet-upscaler",
            repo="black-forest-labs/FLUX.1-dev",
            slug="flux-1-dev-controlnet-upscaler",
            extra_repos=("jasperai/Flux.1-dev-Controlnet-Upscaler",),
            variants={
                Op.UPSCALE: _v(
                    _FLUX_CN,
                    _L_FLUX,
                    args={"image": "controlnet_image_path", "strength": "controlnet_strength"},
                    adapter="flux_upscale",
                )
            },
            steps=25,
            guidance=4.0,
            license=_LIC_FLUX1,
            gated=True,
            tags=("upscale",),
            aliases=("flux-upscaler",),
        ),
        ModelSpec(
            id="dev-fill-catvton",
            title="FLUX.1 Fill CatVTON",
            family="flux1",
            summary="Virtual try-on (specialty model; not mapped to a Dreambot command yet).",
            mflux_key="dev-fill-catvton",
            repo="black-forest-labs/FLUX.1-Fill-dev",
            slug="flux-1-dev-fill-catvton",
            extra_repos=("xiaozaa/catvton-flux-beta",),
            variants={},
            steps=25,
            guidance=30.0,
            license=_LIC_FLUX1,
            gated=True,
            tags=("specialty",),
        ),
        # ----------------------------------------------------------------- FLUX.2 ----
        _flux2(
            "flux2-klein-4b",
            "FLUX.2 klein 4B",
            "black-forest-labs/FLUX.2-klein-4B",
            "flux2-klein-4b",
            summary="Small, fast 4-step model that both generates and edits with reference images.",
            steps=4,
            guidance=1.0,
            distilled=True,
            license=_LIC_APACHE,
            tags=("fast", "edit", "references"),
            aliases=("flux2-klein", "klein-4b", "klein", "flux2"),
            verified=True,
        ),
        _flux2(
            "flux2-klein-9b",
            "FLUX.2 klein 9B",
            "black-forest-labs/FLUX.2-klein-9B",
            "flux2-klein-9b",
            summary="Larger FLUX.2 klein: better quality generation and multi-reference editing.",
            steps=4,
            guidance=1.0,
            distilled=True,
            license=_LIC_FLUX2,
            gated=True,
            tags=("fast", "edit", "references", "quality"),
            aliases=("klein-9b",),
        ),
        _flux2(
            "flux2-klein-9b-kv",
            "FLUX.2 klein 9B KV",
            "black-forest-labs/FLUX.2-klein-9b-kv",
            "flux2-klein-9b-kv",
            summary="FLUX.2 klein 9B tuned for KV-cached (faster) reference editing.",
            steps=4,
            guidance=1.0,
            distilled=True,
            license=_LIC_FLUX2,
            gated=True,
            tags=("fast", "edit", "references"),
            aliases=("klein-9b-kv",),
        ),
        _flux2(
            "flux2-klein-base-4b",
            "FLUX.2 klein base 4B",
            "black-forest-labs/FLUX.2-klein-base-4B",
            "flux-2-klein-base-4b",
            summary="Undistilled FLUX.2 klein 4B (50 steps); mainly a base for LoRA training.",
            steps=50,
            guidance=4.0,
            license=_LIC_APACHE,
            tags=("training",),
            aliases=("klein-base-4b", "flux2-base-4b"),
        ),
        _flux2(
            "flux2-klein-base-9b",
            "FLUX.2 klein base 9B",
            "black-forest-labs/FLUX.2-klein-base-9B",
            "flux-2-klein-base-9b",
            summary="Undistilled FLUX.2 klein 9B (50 steps); mainly a base for LoRA training.",
            steps=50,
            guidance=4.0,
            license=_LIC_FLUX2,
            gated=True,
            tags=("training",),
            aliases=("klein-base-9b", "flux2-base-9b"),
        ),
        # ----------------------------------------------------------------- FLUX 3 ----
        ModelSpec(
            id="flux3-image-klein",
            title="FLUX 3 Image klein",
            family="flux3",
            summary="FLUX 3 klein: 16-step generation plus editing with up to 10 reference images.",
            mflux_key="flux3-image-klein",
            repo="black-forest-labs/flux-3-image-klein",
            slug="flux-3-image-klein",
            variants={
                Op.TXT2IMG: _v(_FLUX3, _L_FLUX3),
                Op.EDIT: _v(_FLUX3_EDIT, _L_FLUX3, args={"images": "image_paths"}, max_refs=10),
            },
            steps=16,
            distilled=True,
            license=_LIC_FLUX3,
            gated=True,
            tags=("quality", "edit", "references"),
            aliases=("flux3-klein", "flux3", "flux-3", "flux-3-image-klein"),
        ),
        ModelSpec(
            id="flux3-image-klein-base",
            title="FLUX 3 Image klein base",
            family="flux3",
            summary="Undistilled FLUX 3 klein (50 steps).",
            mflux_key="flux3-image-klein-base",
            repo="black-forest-labs/flux-3-image-klein-base",
            slug="flux-3-image-klein-base",
            variants={
                Op.TXT2IMG: _v(_FLUX3, _L_FLUX3),
                Op.EDIT: _v(_FLUX3_EDIT, _L_FLUX3, args={"images": "image_paths"}, max_refs=10),
            },
            steps=50,
            license=_LIC_FLUX3,
            gated=True,
            tags=("quality",),
            aliases=("flux3-klein-base", "flux3-base", "flux-3-image-klein-base"),
        ),
        # ------------------------------------------------------------------- Iris ----
        ModelSpec(
            id="iris-3b",
            title="Iris 3B Turbo",
            family="iris",
            summary="Sperid Labs' 3B pixel-space model; 4 steps.",
            mflux_key="iris-3b",
            repo="speridlabs/iris-3b",
            slug="iris-3b-turbo",
            variants={Op.TXT2IMG: _v(_IRIS, _L_IRIS)},
            steps=4,
            distilled=True,
            license=_LIC_APACHE,
            tags=("fast",),
            aliases=("iris", "iris-turbo", "iris-3b-turbo"),
        ),
        ModelSpec(
            id="iris-3b-base",
            title="Iris 3B Base",
            family="iris",
            summary="Undistilled Iris 3B (100 steps).",
            mflux_key="iris-3b-base",
            repo="speridlabs/iris-3b",
            slug="iris-3b",
            variants={Op.TXT2IMG: _v(_IRIS, _L_IRIS)},
            steps=100,
            cfg=True,
            license=_LIC_APACHE,
            aliases=("iris-base",),
        ),
        # ----------------------------------------------------------------- Krea 2 ----
        ModelSpec(
            id="krea-2",
            title="Krea 2 Turbo",
            family="krea2",
            summary="Krea's aesthetics-first model; 8 steps. The closest thing to a 'house style'.",
            mflux_key="krea-2",
            repo="krea/Krea-2-Turbo",
            slug="krea-2-turbo",
            variants=_txt_img(_KREA2, _L_KREA2),
            steps=8,
            guidance=1.0,
            distilled=True,
            license=_LIC_KREA2,
            gated=True,
            tags=("aesthetic", "quality"),
            aliases=("krea2", "krea", "krea-2-turbo"),
        ),
        ModelSpec(
            id="krea-2-raw",
            title="Krea 2 Raw",
            family="krea2",
            summary="Undistilled Krea 2; the base Krea recommends for fine-tuning.",
            mflux_key="krea-2-raw",
            repo="krea/Krea-2-Raw",
            slug="krea-2-raw",
            variants=_txt_img(_KREA2, _L_KREA2),
            steps=25,
            guidance=4.0,
            cfg=True,
            license=_LIC_KREA2,
            gated=True,
            tags=("training",),
            aliases=("krea2-raw",),
        ),
        ModelSpec(
            id="krea-2-controlnet-depth",
            title="Krea 2 Depth ControlNet",
            family="krea2",
            summary="Krea 2 Turbo guided by the depth map of a source image.",
            mflux_key="krea-2-controlnet-depth",
            repo="krea/Krea-2-Turbo",
            extra_repos=("Patil/Krea-2-depth-controlnet",),
            variants={Op.CONTROL: _v(_KREA2_CN, _L_KREA2, args={"image": "image_path"})},
            steps=8,
            guidance=1.0,
            distilled=True,
            license=_LIC_KREA2,
            gated=True,
            tags=("control",),
            aliases=("krea-2-controlnet", "krea2-controlnet"),
        ),
        # ------------------------------------------------------------- Qwen-Image ----
        ModelSpec(
            id="qwen-image",
            title="Qwen-Image 2512",
            family="qwen",
            summary="Alibaba's 20B model: strong prompt adherence and text rendering.",
            mflux_key="qwen-image",
            repo="Qwen/Qwen-Image-2512",
            slug="qwen-image-2512",
            variants=_txt_img(_QWEN, _L_QWEN),
            steps=20,
            guidance=3.5,
            cfg=True,
            license=_LIC_APACHE,
            tags=("quality", "text"),
            aliases=("qwen", "qwen-image-2512", "qwen-2512"),
            verified=True,
        ),
        ModelSpec(
            id="qwen-image-edit",
            title="Qwen-Image-Edit 2509",
            family="qwen",
            summary="Instruction-based editing with multiple reference images.",
            mflux_key="qwen-image-edit",
            repo="Qwen/Qwen-Image-Edit-2509",
            slug="qwen-image-edit-2509",
            variants={Op.EDIT: _v(_QWEN_EDIT, _L_QWEN, args={"images": "image_paths"}, max_refs=3)},
            steps=20,
            guidance=2.5,
            cfg=True,
            license=_LIC_APACHE,
            tags=("edit", "references"),
            aliases=("qwen-edit", "qwen-edit-plus", "qwen-edit-2509"),
        ),
        ModelSpec(
            id="qwen-image-edit-2511",
            title="Qwen-Image-Edit 2511",
            family="qwen",
            summary="The most popular local edit model: consistent multi-reference instruction edits.",
            mflux_key="qwen-image-edit-2511",
            repo="Qwen/Qwen-Image-Edit-2511",
            slug="qwen-image-edit-2511",
            variants={Op.EDIT: _v(_QWEN_EDIT, _L_QWEN, args={"images": "image_paths"}, max_refs=3)},
            steps=20,
            guidance=2.5,
            cfg=True,
            license=_LIC_APACHE,
            tags=("edit", "references", "quality"),
            aliases=("qwen-edit-2511",),
        ),
        # --------------------------------------------------------- Qwen-Image 2.1 ----
        ModelSpec(
            id="qwen-image-2.1",
            title="Qwen-Image 2.1",
            family="qwen21",
            multiple=32,
            summary="Unified generate + edit + inpaint model (40 steps).",
            mflux_key="qwen-image-2.1",
            repo="Qwen/Qwen-Image-2.1",
            slug="qwen-image-2-1",
            variants={
                Op.TXT2IMG: _v(_QWEN21_EDIT, _L_QWEN21_EDIT),
                Op.IMG2IMG: _v(_QWEN21, _L_QWEN21, args=_IMG2IMG),
                Op.EDIT: _v(_QWEN21_EDIT, _L_QWEN21_EDIT, args={"images": "image_paths"}, max_refs=4),
                Op.INPAINT: _v(
                    _QWEN21_EDIT,
                    _L_QWEN21_EDIT,
                    args={"image": "image_paths", "mask": "mask_image"},
                    adapter="qwen21_inpaint",
                    instructed=True,
                ),
            },
            steps=40,
            guidance=1.0,
            cfg=True,
            license=_LIC_QWEN,
            tags=("quality", "edit", "inpaint", "references", "text"),
            aliases=("qwen-2.1", "qwen-image-21"),
            verified=True,
        ),
        ModelSpec(
            id="qwen-image-2.1-turbo",
            title="Qwen-Image 2.1 Turbo",
            family="qwen21",
            multiple=32,
            summary="8-step Qwen-Image 2.1: one model for generation, edits, references and inpainting.",
            mflux_key="qwen-image-2.1-turbo",
            repo="Qwen/Qwen-Image-2.1-Turbo",
            slug="qwen-image-2-1-turbo",
            variants={
                Op.TXT2IMG: _v(_QWEN21_EDIT, _L_QWEN21_EDIT),
                Op.IMG2IMG: _v(_QWEN21, _L_QWEN21, args=_IMG2IMG),
                Op.EDIT: _v(_QWEN21_EDIT, _L_QWEN21_EDIT, args={"images": "image_paths"}, max_refs=4),
                Op.INPAINT: _v(
                    _QWEN21_EDIT,
                    _L_QWEN21_EDIT,
                    args={"image": "image_paths", "mask": "mask_image"},
                    adapter="qwen21_inpaint",
                    instructed=True,
                ),
            },
            steps=8,
            guidance=1.0,
            distilled=True,
            license=_LIC_QWEN,
            tags=("fast", "quality", "edit", "inpaint", "references", "text"),
            aliases=("qwen-2.1-turbo", "qwen-image-21-turbo", "qwen-image-turbo-21"),
        ),
        ModelSpec(
            id="qwen-image-2.1-controlnet-union",
            title="Qwen-Image 2.1 ControlNet Union",
            family="qwen21",
            multiple=32,
            summary="Qwen-Image 2.1 with structure control (canny, depth, pose, ...) and mask inpainting.",
            mflux_key="qwen-image-2.1-controlnet-union",
            repo="Qwen/Qwen-Image-2.1",
            slug="qwen-image-2-1-controlnet-union",
            extra_repos=("alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union",),
            variants={
                Op.CONTROL: _v(
                    _QWEN21_CN,
                    _L_QWEN21_EDIT,
                    args={"image": "controlnet_image_path", "strength": "controlnet_strength"},
                ),
                Op.INPAINT: _v(
                    _QWEN21_CN, _L_QWEN21_EDIT, args={"image": "image_path", "mask": "mask_image"}
                ),
            },
            steps=40,
            guidance=1.0,
            cfg=True,
            license=_LIC_QWEN,
            tags=("control", "inpaint"),
            aliases=("qwen-image-2.1-controlnet", "qwen-2.1-controlnet"),
        ),
        ModelSpec(
            id="qwen-image-2.1-turbo-controlnet-union",
            title="Qwen-Image 2.1 Turbo ControlNet Union",
            family="qwen21",
            multiple=32,
            summary="8-step structure control and mask inpainting on Qwen-Image 2.1 Turbo.",
            mflux_key="qwen-image-2.1-turbo-controlnet-union",
            repo="Qwen/Qwen-Image-2.1-Turbo",
            extra_repos=("alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union",),
            variants={
                Op.CONTROL: _v(
                    _QWEN21_CN,
                    _L_QWEN21_EDIT,
                    args={"image": "controlnet_image_path", "strength": "controlnet_strength"},
                ),
                Op.INPAINT: _v(
                    _QWEN21_CN, _L_QWEN21_EDIT, args={"image": "image_path", "mask": "mask_image"}
                ),
            },
            steps=8,
            guidance=1.0,
            distilled=True,
            license=_LIC_QWEN,
            tags=("control", "inpaint", "fast"),
            aliases=("qwen-image-2.1-turbo-controlnet", "qwen-2.1-turbo-controlnet"),
        ),
        # ------------------------------------------------------------------- FIBO ----
        ModelSpec(
            id="fibo",
            helper_repos=("briaai/FIBO-vlm",),
            title="FIBO",
            family="fibo",
            summary="Bria's JSON-prompted model trained on licensed data.",
            mflux_key="fibo",
            repo="briaai/FIBO",
            slug="fibo",
            variants={
                Op.TXT2IMG: _v(_FIBO, _L_FIBO, adapter="fibo"),
                Op.IMG2IMG: _v(_FIBO, _L_FIBO, args=_IMG2IMG, adapter="fibo"),
            },
            steps=50,
            guidance=5.0,
            cfg=True,
            license=_LIC_BRIA,
            gated=True,
            tags=("structured",),
        ),
        ModelSpec(
            id="fibo-lite",
            helper_repos=("briaai/FIBO-vlm",),
            title="FIBO Lite",
            family="fibo",
            summary="Distilled 8-step FIBO.",
            mflux_key="fibo-lite",
            repo="briaai/Fibo-lite",
            slug="fibo-lite",
            variants={
                Op.TXT2IMG: _v(_FIBO, _L_FIBO, adapter="fibo"),
                Op.IMG2IMG: _v(_FIBO, _L_FIBO, args=_IMG2IMG, adapter="fibo"),
            },
            steps=8,
            guidance=1.0,
            distilled=True,
            license="cc-by-nc-4.0",
            gated=True,
            tags=("structured", "fast"),
            aliases=("fibo_lite",),
        ),
        ModelSpec(
            id="fibo-edit",
            helper_repos=("briaai/FIBO-vlm",),
            title="FIBO Edit",
            family="fibo",
            summary="Bria's instruction and mask-based editor.",
            mflux_key="fibo-edit",
            repo="briaai/Fibo-Edit",
            slug="fibo-edit",
            variants={
                Op.EDIT: _v(
                    _FIBO_EDIT, _L_FIBO, args={"images": "image_path"}, max_refs=1, adapter="fibo_edit"
                ),
                Op.INPAINT: _v(
                    _FIBO_EDIT,
                    _L_FIBO,
                    args={"image": "image_path", "mask": "mask_path"},
                    adapter="fibo_edit",
                    instructed=True,
                ),
            },
            steps=50,
            guidance=3.5,
            cfg=True,
            license=_LIC_BRIA_EDIT,
            gated=True,
            tags=("edit", "inpaint", "structured"),
            aliases=("fiboedit",),
        ),
        ModelSpec(
            id="fibo-edit-rmbg",
            title="FIBO Edit RMBG",
            family="fibo",
            summary="Background removal built on FIBO Edit.",
            mflux_key="fibo-edit-rmbg",
            repo="briaai/Fibo-Edit-RMBG",
            slug="fibo-edit-rmbg",
            variants={
                Op.EDIT: _v(
                    _FIBO_EDIT, _L_FIBO, args={"images": "image_path"}, max_refs=1, adapter="fibo_edit"
                ),
            },
            steps=10,
            guidance=1.0,
            license=_LIC_BRIA_EDIT,
            gated=True,
            tags=("specialty", "background-removal"),
            aliases=("fiboedit-rmbg",),
        ),
        ModelSpec(
            id="fibo2",
            title="FIBO 2",
            family="fibo2",
            summary="Second-generation FIBO (50 steps).",
            mflux_key="fibo2",
            repo="briaai/Fibo-2.0-dev",
            slug="fibo-2-0-dev",
            variants={Op.TXT2IMG: _v(_FIBO2, _L_FIBO2)},
            steps=50,
            cfg=True,
            license=_LIC_BRIA,
            gated=True,
            tags=("structured", "quality"),
            aliases=("fibo-2.0-dev",),
        ),
        ModelSpec(
            id="fibo2-edit-turbo",
            title="FIBO 2 Edit Turbo",
            family="fibo2",
            summary="4-step FIBO 2 editor: 1-5 reference images, optional mask.",
            mflux_key="fibo2-edit-turbo",
            repo="briaai/Fibo-2.0-edit-turbo",
            slug="fibo-2-0-edit-turbo",
            variants={
                Op.EDIT: _v(_FIBO2_EDIT, _L_FIBO2, args={"images": "image_paths"}, max_refs=5),
                Op.INPAINT: _v(
                    _FIBO2_EDIT,
                    _L_FIBO2,
                    args={"image": "image_paths", "mask": "mask_path"},
                    adapter="list_image",
                    instructed=True,
                ),
            },
            steps=4,
            distilled=True,
            license=_LIC_BRIA,
            gated=True,
            tags=("edit", "inpaint", "fast", "references"),
            aliases=("fibo-2-edit-turbo", "fibo-2.0-edit-turbo"),
        ),
        # ------------------------------------------------------------ Other models ----
        ModelSpec(
            id="ming-image-design",
            title="Ming-Image 0.1 Design",
            family="ming",
            summary="Ant Group's graphic-design model; outputs RGBA images with transparency.",
            mflux_key="ming-image-design",
            repo="inclusionAI/Ming-Image-0.1-Design",
            slug="ming-image-0-1",
            variants={Op.TXT2IMG: _v(_MING, _L_MING)},
            steps=12,
            guidance=1.0,
            license=_LIC_MIT,
            tags=("design", "transparent", "text"),
            aliases=("ming-image", "ming-design", "ming"),
            verified=True,
        ),
        ModelSpec(
            id="lens-turbo",
            title="Lens Turbo",
            family="lens",
            summary="Microsoft's 4-step Lens model.",
            mflux_key="lens-turbo",
            repo="Comfy-Org/Lens",
            variants={Op.TXT2IMG: _v(_LENS, _L_FLUX2)},
            steps=4,
            distilled=True,
            license=_LIC_OTHER,
            tags=("fast",),
            aliases=("lens",),
        ),
        ModelSpec(
            id="ernie-image",
            title="ERNIE-Image",
            family="ernie",
            summary="Baidu's ERNIE-Image (50 steps, true CFG).",
            mflux_key="ernie-image",
            repo="baidu/ERNIE-Image",
            slug="ernie-image-base",
            variants=_txt_img(_ERNIE, _L_ERNIE),
            steps=50,
            guidance=4.0,
            cfg=True,
            license=_LIC_APACHE,
            tags=("quality", "text"),
        ),
        ModelSpec(
            id="ernie-image-turbo",
            title="ERNIE-Image Turbo",
            family="ernie",
            summary="Distilled 8-step ERNIE-Image.",
            mflux_key="ernie-image-turbo",
            repo="baidu/ERNIE-Image-Turbo",
            slug="ernie-image-turbo",
            variants=_txt_img(_ERNIE, _L_ERNIE),
            steps=8,
            guidance=1.0,
            distilled=True,
            license=_LIC_APACHE,
            tags=("fast", "text"),
        ),
        ModelSpec(
            id="ideogram-4-fp8",
            title="Ideogram 4",
            family="ideogram",
            summary="Ideogram's open-weights model: best-in-class typography and layout.",
            mflux_key="ideogram-4-fp8",
            repo="ideogram-ai/ideogram-4-fp8",
            slug="ideogram-4",
            variants={Op.TXT2IMG: _v(_IDEOGRAM, _L_IDEOGRAM)},
            steps=20,
            cfg=True,
            license=_LIC_IDEOGRAM,
            gated=True,
            tags=("text", "design", "quality"),
            aliases=("ideogram4", "ideogram-4", "ideogram", "ideogram4-fp8"),
        ),
        ModelSpec(
            id="boogu-image-turbo",
            title="Boogu-Image Turbo",
            family="boogu",
            summary="4-step DMD-distilled model (use 8 steps at 1024px).",
            mflux_key="boogu-image-turbo",
            repo="Boogu/Boogu-Image-0.1-Turbo",
            slug="boogu-image-turbo",
            variants={Op.TXT2IMG: _v(_BOOGU)},
            steps=8,
            distilled=True,
            license=_LIC_APACHE,
            tags=("fast",),
            aliases=("boogu-turbo", "boogu-image", "boogu"),
        ),
        ModelSpec(
            id="moss-image-sft",
            title="MOSS-Image SFT",
            family="moss",
            summary="OpenMOSS's fine-tuned image model (50 steps, CFG 3).",
            mflux_key="moss-image-sft",
            repo="OpenMOSS-Team/MOSS-Image-SFT",
            slug="moss-image-sft",
            variants={Op.TXT2IMG: _v(_MOSS, _L_MOSS)},
            steps=50,
            guidance=3.0,
            cfg=True,
            license=_LIC_APACHE,
            tags=("quality",),
            aliases=("moss-image", "moss"),
        ),
        ModelSpec(
            id="moss-image-base",
            title="MOSS-Image Base",
            family="moss",
            summary="The MOSS-Image base model.",
            mflux_key="moss-image-base",
            repo="OpenMOSS-Team/MOSS-Image-Base",
            variants={Op.TXT2IMG: _v(_MOSS, _L_MOSS)},
            steps=50,
            guidance=3.0,
            cfg=True,
            license=_LIC_APACHE,
            aliases=("moss-base",),
        ),
        ModelSpec(
            id="anima-turbo",
            title="Anima Turbo",
            family="anima",
            summary="Anime and illustration model; 10 steps. Powers --niji.",
            mflux_key="anima-turbo",
            repo="circlestone-labs/Anima",
            files=(
                "split_files/diffusion_models/anima-turbo-v1.1.safetensors",
                "split_files/text_encoders/qwen_3_06b_base.safetensors",
                "split_files/vae/qwen_image_vae.safetensors",
            ),
            slug="anima-turbo",
            variants={Op.TXT2IMG: _v(_ANIMA, _L_ANIMA)},
            steps=10,
            guidance=1.0,
            distilled=True,
            license=_LIC_ANIMA,
            tags=("anime", "fast", "niji"),
            aliases=("anima",),
        ),
        ModelSpec(
            id="anima-aesthetic",
            title="Anima Aesthetic",
            family="anima",
            summary="Anima's aesthetic-tuned anime model (30 steps, CFG).",
            mflux_key="anima-aesthetic",
            repo="circlestone-labs/Anima",
            files=(
                "split_files/diffusion_models/anima-aesthetic-v1.1.safetensors",
                "split_files/text_encoders/qwen_3_06b_base.safetensors",
                "split_files/vae/qwen_image_vae.safetensors",
            ),
            slug="anima-aesthetic",
            variants={Op.TXT2IMG: _v(_ANIMA, _L_ANIMA)},
            steps=30,
            guidance=4.0,
            cfg=True,
            license=_LIC_ANIMA,
            tags=("anime", "quality", "niji"),
        ),
        ModelSpec(
            id="anima-base",
            title="Anima Base",
            family="anima",
            summary="The Anima base anime model (30 steps, CFG).",
            mflux_key="anima-base",
            repo="circlestone-labs/Anima",
            files=(
                "split_files/diffusion_models/anima-base-v1.0.safetensors",
                "split_files/text_encoders/qwen_3_06b_base.safetensors",
                "split_files/vae/qwen_image_vae.safetensors",
            ),
            slug="anima-base",
            variants={Op.TXT2IMG: _v(_ANIMA, _L_ANIMA)},
            steps=30,
            guidance=4.0,
            cfg=True,
            license=_LIC_ANIMA,
            tags=("anime",),
        ),
        # --------------------------------------------------------------- Upscalers ----
        ModelSpec(
            id="seedvr2-3b",
            title="SeedVR2 3B",
            family="seedvr2",
            summary="One-step, highly faithful diffusion upscaler. Powers Upscale (Subtle).",
            mflux_key="seedvr2-3b",
            repo="numz/SeedVR2_comfyUI",
            files=(
                "seedvr2_ema_3b_fp16.safetensors",
                "ema_vae_fp16.safetensors",
            ),
            variants={Op.UPSCALE: _v(_SEEDVR2, args={"image": "image_path"}, adapter="seedvr2")},
            steps=1,
            distilled=True,
            license=_LIC_APACHE,
            tags=("upscale",),
            aliases=("seedvr2",),
            verified=True,
        ),
        ModelSpec(
            id="seedvr2-7b",
            title="SeedVR2 7B",
            family="seedvr2",
            summary="The larger SeedVR2 upscaler.",
            mflux_key="seedvr2-7b",
            repo="numz/SeedVR2_comfyUI",
            files=(
                "seedvr2_ema_7b_fp16.safetensors",
                "ema_vae_fp16.safetensors",
            ),
            variants={Op.UPSCALE: _v(_SEEDVR2, args={"image": "image_path"}, adapter="seedvr2")},
            steps=1,
            distilled=True,
            license=_LIC_APACHE,
            tags=("upscale", "quality"),
        ),
        # ------------------------------------------------------------------- Video ----
        ModelSpec(
            id="wan-2.2",
            title="Wan 2.2",
            family="wan",
            summary=(
                "Alibaba's 5B video model in FastVideo's three-step build for Apple Silicon: "
                "video from words or from a picture."
            ),
            mflux_key="",
            repo="Wan-AI/Wan2.2-TI2V-5B",
            runtime="wan",
            # FastMetal: distilled to three steps and trained to run packed to 8 bits. It is the
            # one form these weights are published in for MLX.
            exports=MappingProxyType({"q8": "FastVideo/FastMetal-5B-QAD"}),
            # (The repo also holds Wan's own 2.8 GB autoencoder, which is not read.)
            files=("mlx_dit.safetensors", "mlx_dit.json", "text_encoder/*", "tokenizer/tokenizer.json"),
            assets=(
                # TAEHV, the small autoencoder the frames are decoded by. Its author publishes
                # it on GitHub; this is the file FastVideo's own runtime fetches.
                Asset(
                    name="taew2_2.safetensors",
                    url=(
                        "https://raw.githubusercontent.com/madebyollin/taehv/"
                        "563f40bdc820ed86bcad72ea515ee48f06bd22ec/taew2_2.pth"
                    ),
                    sha256="d053e216ca50e2bb837bbcd79b85f0366bea00e5938025572382a773b74c559a",
                    license=_LIC_MIT,
                ),
            ),
            variants={Op.TXT2VID: _WAN, Op.IMG2VID: _WAN, Op.EXTEND: _WAN},
            # The model is distilled to exactly these three passes.
            steps=3,
            distilled=True,
            # Sides are multiples of 32, and the size is the one closest to 832x480 in area
            # (which is what a native side of 624 gives for 16:9; 1248x704 at 1.5 times that).
            multiple=32,
            native=624,
            video=VideoTraits(fps=24, step=4, max_frames=121, carry=5, ends=False),
            license=_LIC_APACHE,
            tags=("video", "fast", "default"),
            aliases=("wan", "wan2.2", "wan-2.2-5b", "wan-video", "fastmetal"),
            verified=True,
        ),
    ]


class UnknownModel(KeyError):
    def __init__(self, name: str, suggestions: list[str]):
        self.name = name
        self.suggestions = suggestions
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        super().__init__(f"Unknown model '{name}'.{hint}")

    def __str__(self) -> str:
        return self.args[0]


class Registry:
    """Lookup of :class:`ModelSpec` by id or alias."""

    def __init__(self, specs: list[ModelSpec]):
        self._specs: dict[str, ModelSpec] = {}
        self._aliases: dict[str, str] = {}
        for spec in specs:
            if spec.id in self._specs:
                raise ValueError(f"duplicate model id: {spec.id}")
            self._specs[spec.id] = spec
        for spec in specs:
            for alias in spec.aliases:
                key = alias.lower()
                if key in self._specs or key in self._aliases:
                    raise ValueError(f"alias '{alias}' of {spec.id} collides with another model")
                self._aliases[key] = spec.id

    def __iter__(self):
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, name: str) -> bool:
        return self.find(name) is not None

    def find(self, name: str) -> ModelSpec | None:
        key = name.strip().lower()
        if key in self._specs:
            return self._specs[key]
        if key in self._aliases:
            return self._specs[self._aliases[key]]
        return None

    def get(self, name: str) -> ModelSpec:
        spec = self.find(name)
        if spec is None:
            import difflib

            names = list(self._specs) + list(self._aliases)
            raise UnknownModel(name, difflib.get_close_matches(name.strip().lower(), names, n=3, cutoff=0.5))
        return spec

    def with_op(self, op: Op) -> list[ModelSpec]:
        return [spec for spec in self if spec.supports(op)]

    def ids(self) -> list[str]:
        return list(self._specs)


REGISTRY = Registry(_specs())
