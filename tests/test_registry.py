"""The registry is hand-written data that mirrors mflux. These tests pin it to the installed
mflux so that drift (a renamed class, a new model, a changed default) fails loudly instead of
surfacing as a broken command at generation time. Nothing here loads weights."""

from __future__ import annotations

import importlib
import inspect

import pytest

from dreambot.engine.protocol import GenerateRequest, Op
from dreambot.engine.registry import REGISTRY, ModelSpec, Registry, UnknownModel, Variant

#: Places where Dreambot deliberately departs from mflux's default step count.
STEP_OVERRIDES = {
    # mflux defaults to 4 steps but its own help says to use 8 at 1024x1024, Dreambot's default size.
    "boogu-image-turbo": 8,
}


def load(path: str):
    module, _, name = path.partition(":")
    return getattr(importlib.import_module(module), name)


@pytest.fixture(scope="module")
def mflux_models():
    from mflux.models.common.config.model_config import AVAILABLE_MODELS

    return AVAILABLE_MODELS


@pytest.fixture(scope="module")
def mflux_steps():
    from mflux.cli.defaults.defaults import MODEL_INFERENCE_STEPS

    return MODEL_INFERENCE_STEPS


def all_variants() -> list[tuple[ModelSpec, Op, Variant]]:
    return [(spec, op, variant) for spec in REGISTRY for op, variant in spec.variants.items()]


def variant_id(item: tuple[ModelSpec, Op, Variant]) -> str:
    return f"{item[0].id}:{item[1].value}"


def test_every_mflux_model_is_covered_exactly_once(mflux_models):
    covered = [spec.mflux_key for spec in REGISTRY]
    assert sorted(covered) == sorted(set(covered)), "an mflux model is registered twice"
    missing = set(mflux_models) - set(covered)
    assert not missing, f"mflux models missing from the registry: {sorted(missing)}"
    unknown = set(covered) - set(mflux_models)
    assert not unknown, f"registry refers to models mflux does not have: {sorted(unknown)}"


def test_ids_match_mflux_keys_so_mflux_users_feel_at_home():
    for spec in REGISTRY:
        assert spec.id == spec.mflux_key


def test_upstream_repos_match_mflux(mflux_models):
    for spec in REGISTRY:
        assert spec.repo == mflux_models[spec.mflux_key].model_name, spec.id


def test_controlnet_repos_are_listed_as_extra_downloads(mflux_models):
    for spec in REGISTRY:
        controlnet = mflux_models[spec.mflux_key].controlnet_model
        if controlnet:
            assert controlnet in spec.extra_repos, f"{spec.id} must list {controlnet}"


def test_default_steps_match_mflux(mflux_steps):
    for spec in REGISTRY:
        expected = STEP_OVERRIDES.get(spec.id, mflux_steps.get(spec.mflux_key))
        if expected is not None:
            assert spec.steps == expected, f"{spec.id}: registry says {spec.steps}, mflux says {expected}"


def test_guidance_support_matches_mflux(mflux_models):
    """A model flagged ``cfg`` must be one mflux says supports guidance."""
    for spec in REGISTRY:
        if spec.cfg:
            assert mflux_models[spec.mflux_key].supports_guidance is not False, spec.id


def test_every_mflux_alias_resolves(mflux_models):
    for key, config in mflux_models.items():
        for alias in config.aliases:
            found = REGISTRY.find(alias)
            assert found is not None, f"mflux alias '{alias}' (of {key}) is unknown to Dreambot"
            assert found.mflux_key == key, f"alias '{alias}' resolves to {found.id}, mflux says {key}"


@pytest.mark.parametrize("item", all_variants(), ids=variant_id)
def test_variant_class_exists_and_accepts_the_mapped_arguments(item):
    _spec, _op, variant = item
    cls = load(variant.cls)
    assert inspect.isclass(cls)
    accepted = inspect.signature(cls.generate_image).parameters
    for request_field, keyword in variant.args.items():
        assert request_field in GenerateRequest.model_fields, f"{request_field} is not a request field"
        assert keyword in accepted, f"{cls.__name__}.generate_image has no '{keyword}' parameter"
    constructor = inspect.signature(cls).parameters
    for keyword in variant.init:
        assert keyword in constructor, f"{cls.__name__}() has no '{keyword}' parameter"
    if variant.latent is not None:
        assert hasattr(load(variant.latent), "unpack_latents")


@pytest.mark.parametrize("item", all_variants(), ids=variant_id)
def test_variant_operations_have_the_inputs_they_need(item):
    _spec, op, variant = item
    fields = set(variant.args)
    if op is Op.IMG2IMG:
        assert {"image", "strength"} <= fields
    if op is Op.INPAINT:
        assert {"image", "mask"} <= fields
    if op in (Op.EDIT, Op.REDUX):
        assert "images" in fields and variant.max_refs >= 1
    if op in (Op.UPSCALE, Op.CONTROL) and variant.adapter is None:
        assert "image" in fields


def test_every_adapter_named_in_the_registry_is_known():
    from dreambot.engine.backends.mlx_backend import _ADAPTERS

    pending = {
        f"{spec.id}:{op.value}"
        for spec, op, variant in all_variants()
        if variant.adapter is not None and variant.adapter not in _ADAPTERS
    }
    assert not pending, f"operations whose adapter is not implemented: {sorted(pending)}"


def test_lookup_by_id_alias_and_case():
    assert REGISTRY.get("z-image-turbo").id == "z-image-turbo"
    assert REGISTRY.get("Kontext").id == "dev-kontext"
    assert REGISTRY.get("  FLUX2  ").id == "flux2-klein-4b"
    assert "klein" in REGISTRY and "nope" not in REGISTRY


def test_unknown_model_suggests_close_matches():
    with pytest.raises(UnknownModel) as excinfo:
        REGISTRY.get("z-image-turbi")
    assert "z-image-turbo" in excinfo.value.suggestions
    assert "Did you mean" in str(excinfo.value)


def test_registry_rejects_colliding_names():
    spec = REGISTRY.get("schnell")
    with pytest.raises(ValueError, match="duplicate"):
        Registry([spec, spec])
    from dataclasses import replace

    with pytest.raises(ValueError, match="collides"):
        Registry([spec, replace(REGISTRY.get("dev"), aliases=("schnell",))])


def test_every_core_operation_has_at_least_one_permissively_licensed_model():
    """Dreambot must be fully usable without accepting a non-commercial license."""
    for op in (Op.TXT2IMG, Op.IMG2IMG, Op.EDIT, Op.UPSCALE, Op.CONTROL):
        assert any(spec.permissive for spec in REGISTRY.with_op(op)), op


def test_registry_module_does_not_import_the_inference_stack():
    import subprocess
    import sys

    code = (
        "import sys; import dreambot.engine.registry, dreambot.engine.weights, dreambot.engine.client; "
        "heavy = [m for m in ('mlx', 'mflux', 'torch', 'transformers', 'numpy') if m in sys.modules]; "
        "print(heavy); sys.exit(1 if heavy else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, f"the bot process would import: {result.stdout.strip()}"


def test_files_of_shared_repos_match_what_mflux_reads():
    """Where several models share an upstream repo, ``files`` names exactly what mflux loads."""
    from mflux.models.anima.weights.anima_weight_definition import AnimaWeightDefinition as Anima
    from mflux.models.seedvr2.weights.seedvr2_weight_definition import (
        SeedVR2WeightDefinition3B,
        SeedVR2WeightDefinition7B,
    )

    assert REGISTRY.get("seedvr2-3b").files == tuple(SeedVR2WeightDefinition3B.get_download_patterns())
    assert REGISTRY.get("seedvr2-7b").files == tuple(SeedVR2WeightDefinition7B.get_download_patterns())
    for variant, filename in Anima.TRANSFORMER_FILES.items():
        spec = REGISTRY.get(variant)
        assert spec.repo == Anima.REPO
        assert spec.files == (
            f"{Anima.TRANSFORMER_DIR}/{filename}",
            f"{Anima.TEXT_ENCODER_DIR}/{Anima.TEXT_ENCODER_FILE}",
            f"{Anima.VAE_DIR}/{Anima.VAE_FILE}",
        )


def test_models_that_share_everything_upstream_declare_their_files():
    """A new mflux model that reuses an existing repo must say which files are its own."""
    seen: dict[tuple, list] = {}
    for spec in REGISTRY:
        seen.setdefault((spec.repo, spec.extra_repos), []).append(spec)
    # Iris turbo and base are one set of weights run two ways (mflux gives both the same repo and file).
    same_weights = {frozenset({"iris-3b", "iris-3b-base"})}
    for (repo, _), specs in seen.items():
        if len(specs) < 2 or frozenset(spec.id for spec in specs) in same_weights:
            continue
        for spec in specs:
            assert spec.files, f"{spec.id} shares {repo} with {[s.id for s in specs]} but names no files"
        assert len({spec.files for spec in specs}) == len(specs), f"models in {repo} have the same files"
