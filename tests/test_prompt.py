from __future__ import annotations

import pytest

from dreambot.core.params import PARAMS, Kind, LoraRef, PromptError, Ref, Support, find_param
from dreambot.core.prompt import PromptPart, expand_permutations, parse_prompt

# --------------------------------------------------------------------------------------
# Plain prompts and parameters
# --------------------------------------------------------------------------------------


def test_plain_prompt():
    parsed = parse_prompt("a lighthouse on a cliff at golden hour")
    assert parsed.text == "a lighthouse on a cliff at golden hour"
    assert parsed.parts == [PromptPart("a lighthouse on a cliff at golden hour")]
    assert parsed.params.given() == {}
    assert not parsed.is_multi and not parsed.image_prompts


def test_typical_midjourney_prompt():
    parsed = parse_prompt(
        "cinematic photo of a red fox in snow --ar 16:9 --stylize 250 --chaos 20 --seed 42 --no text, watermark"
    )
    params = parsed.params
    assert parsed.text == "cinematic photo of a red fox in snow"
    assert params.aspect == (16, 9)
    assert params.stylize == 250 and params.chaos == 20 and params.seed == 42
    assert params.no == ["text", "watermark"]
    assert parsed.negative_terms == ["text", "watermark"]


def test_aliases_and_case_insensitivity():
    params = parse_prompt("x --AR 3:2 --s 50 --c 10 --w 250 --q 2 --r 3 --v 8.2").params
    assert params.aspect == (3, 2)
    assert (params.stylize, params.chaos, params.weird, params.quality, params.repeat) == (
        50,
        10,
        250,
        2.0,
        3,
    )
    assert params.version == "8.2"


@pytest.mark.parametrize("dash", ["--", "—", "–"])
def test_ios_autocorrected_dashes_are_parameters(dash: str):
    # iOS turns "--" into an em dash as you type; iMessage users will send exactly this.
    parsed = parse_prompt(f"a cat {dash}ar 16:9 {dash}chaos 30")
    assert parsed.text == "a cat"
    assert parsed.params.aspect == (16, 9) and parsed.params.chaos == 30


def test_em_dash_in_prose_is_not_a_parameter():
    parsed = parse_prompt("a quiet night —cold and still — over the harbour —ar 2:3")
    assert parsed.text == "a quiet night —cold and still — over the harbour"
    assert parsed.params.aspect == (2, 3)


def test_double_hyphen_inside_text_without_a_name_is_text():
    parsed = parse_prompt("black -- white contrast, 50--50 split --ar 1:1")
    assert parsed.text == "black -- white contrast, 50--50 split"
    assert parsed.params.aspect == (1, 1)


def test_hyphenated_words_are_not_parameters():
    parsed = parse_prompt("a well-lit, state-of-the-art lab")
    assert parsed.text == "a well-lit, state-of-the-art lab" and parsed.params.given() == {}


def test_whitespace_and_newlines_are_normalised():
    parsed = parse_prompt("  a   cat\n\nwith  a hat  --ar   4:3  ")
    assert parsed.text == "a cat with a hat"
    assert parsed.params.aspect == (4, 3)


def test_flags():
    params = parse_prompt("x --raw --tile --hd --draft").params
    assert params.raw is True and params.tile is True and params.hd is True and params.draft is True
    assert parse_prompt("x --sd").params.hd is False


def test_style_raw_is_the_same_as_raw():
    assert parse_prompt("x --style raw").params.raw is True
    assert parse_prompt("x --style cute").params.raw is None


def test_repeated_list_parameters_accumulate():
    assert parse_prompt("x --no text --no watermark, logo").params.no == ["text", "watermark", "logo"]


def test_repeating_a_parameter_with_the_same_value_is_fine_but_conflicts_are_not():
    assert parse_prompt("x --ar 1:1 --ar 1:1").params.aspect == (1, 1)
    with pytest.raises(PromptError, match="more than once"):
        parse_prompt("x --ar 1:1 --ar 16:9")
    with pytest.raises(PromptError, match="more than once"):
        parse_prompt("x --hd --sd")


def test_canonical_form_round_trips():
    parsed = parse_prompt("https://example.com/a.png a cat —ar 16:9 --s 250 --no dogs")
    canonical = parsed.canonical()
    assert canonical == "https://example.com/a.png a cat --aspect 16:9 --stylize 250 --no dogs"
    again = parse_prompt(canonical)
    assert again.params == parsed.params and again.text == parsed.text
    assert again.image_prompts == parsed.image_prompts


# --------------------------------------------------------------------------------------
# Validation and error messages
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prompt", "fragment"),
    [
        ("x --chaos 101", "between 0 and 100"),
        ("x --stylize -1", "between 0 and 1000"),
        ("x --weird 3001", "between 0 and 3000"),
        ("x --seed 4294967296", "between 0 and 4294967295"),
        ("x --stop 5", "between 10 and 100"),
        ("x --iw 3.5", "between 0 and 3"),
        ("x --chaos lots", "needs a number"),
        ("x --chaos", "needs a value"),
        ("x --seed 1.5", "whole number"),
        ("x --ar wide", "ratio like `16:9`"),
        ("x --ar 16:9 of a dog", "ratio like `16:9`"),
        ("x --ar 15:1", "at most 14:1"),
        ("x --ar 0:1", "greater than zero"),
        ("x --tile yes", "doesn't take a value"),
        ("x --style fancy", "must be one of"),
        ("x --motion fast", "must be one of"),
        ("x --model", "single name"),
        ("x --no", "at least one item"),
        ("x --oref banana", "don't understand `banana`"),
        ("x --lora", "needs a LoRA name"),
    ],
)
def test_invalid_values_are_explained(prompt: str, fragment: str):
    with pytest.raises(PromptError) as excinfo:
        parse_prompt(prompt)
    assert fragment in str(excinfo.value)


def test_unknown_parameter_suggests_a_fix():
    with pytest.raises(PromptError) as excinfo:
        parse_prompt("a cat --stylise 100")
    assert "`--stylise`" in str(excinfo.value) and "`--stylize`" in str(excinfo.value)
    with pytest.raises(PromptError, match="don't know the parameter `--zzzqq`"):
        parse_prompt("a cat --zzzqq 1")


def test_overlong_prompt_is_rejected():
    with pytest.raises(PromptError, match="limit"):
        parse_prompt("word " * 2000)


# --------------------------------------------------------------------------------------
# Aspect ratio
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("16:9", (16, 9)),
        ("9:16", (9, 16)),
        ("1920:1080", (16, 9)),
        ("4x3", (4, 3)),
        ("3/2", (3, 2)),
        ("1.5:1", (3, 2)),
        ("2.39:1", (239, 100)),
        ("14:1", (14, 1)),
        ("1:1", (1, 1)),
    ],
)
def test_aspect_ratio_forms(value: str, expected: tuple[int, int]):
    assert parse_prompt(f"x --ar {value}").params.aspect == expected


def test_quality_snaps_to_midjourney_values():
    assert parse_prompt("x --q 3").params.quality in (2.0, 4.0)
    assert parse_prompt("x --q .25").params.quality == 0.25
    assert parse_prompt("x --q 0.6").params.quality == 0.5


# --------------------------------------------------------------------------------------
# Image prompts and references
# --------------------------------------------------------------------------------------


def test_leading_links_are_image_prompts():
    parsed = parse_prompt("https://a.example/1.png https://b.example/2.jpg a cat in their style --iw 2")
    assert parsed.image_prompts == [
        Ref("url", "https://a.example/1.png"),
        Ref("url", "https://b.example/2.jpg"),
    ]
    assert parsed.text == "a cat in their style"
    assert parsed.params.iw == 2.0


def test_links_later_in_the_text_stay_in_the_text():
    parsed = parse_prompt("a poster that says visit https://example.com today")
    assert parsed.image_prompts == []
    assert "https://example.com" in parsed.text


def test_image_prompts_only():
    parsed = parse_prompt("https://a.example/1.png https://b.example/2.png")
    assert len(parsed.image_prompts) == 2 and parsed.text == "" and parsed.parts == []


def test_style_references():
    params = parse_prompt("x --sref https://a.example/s.png::2 12345 random @2 --sw 300 --sv 4").params
    assert params.sref == [
        Ref("url", "https://a.example/s.png", 2.0),
        Ref("code", "12345"),
        Ref("random", "random"),
        Ref("attachment", "2"),
    ]
    assert params.sw == 300 and params.sv == 4


def test_reference_parameter_without_a_value_means_the_attached_images():
    params = parse_prompt("make it night time --edit").params
    assert params.edit == [Ref("attachment", "*")]
    assert parse_prompt("x --sref --cref @1").params.sref == [Ref("attachment", "*")]


def test_edit_cref_and_oref():
    params = parse_prompt(
        "put her on a beach --edit https://a.example/p.png @1 --cref https://a.example/c.png --cw 50"
    ).params
    assert [ref.kind for ref in params.edit] == ["url", "attachment"]
    assert params.cref == [Ref("url", "https://a.example/c.png")] and params.cw == 50
    assert parse_prompt("x --oref ~/Pictures/me.png --ow 400").params.oref == [
        Ref("path", "~/Pictures/me.png")
    ]


def test_attachment_numbers_start_at_one():
    with pytest.raises(PromptError, match="numbered from @1"):
        parse_prompt("x --sref @0")


# --------------------------------------------------------------------------------------
# Multi-prompts
# --------------------------------------------------------------------------------------


def test_multi_prompt_default_weights():
    parsed = parse_prompt("space:: ship")
    assert parsed.parts == [PromptPart("space", 1.0), PromptPart("ship", 1.0)]
    assert parsed.is_multi and parsed.positive_text == "space, ship"


def test_multi_prompt_weights_order_the_positive_text():
    parsed = parse_prompt("lighthouse:: storm::3 seagulls::0.5")
    assert parsed.parts == [
        PromptPart("lighthouse", 1.0),
        PromptPart("storm", 3.0),
        PromptPart("seagulls", 0.5),
    ]
    assert parsed.positive_text == "storm, lighthouse, seagulls"


def test_negative_weights_become_things_to_avoid():
    parsed = parse_prompt("still life painting:: fruit::-0.5 --no flies")
    assert parsed.positive_text == "still life painting"
    assert parsed.negative_terms == ["flies", "fruit"]


def test_a_number_after_a_space_is_text_not_a_weight():
    parsed = parse_prompt("space:: 2001 a space odyssey")
    assert parsed.parts == [PromptPart("space", 1.0), PromptPart("2001 a space odyssey", 1.0)]


def test_zero_weight_parts_are_dropped_and_weights_must_be_positive_overall():
    assert parse_prompt("cat::0 dog::1").parts == [PromptPart("dog", 1.0)]
    with pytest.raises(PromptError, match="add up to more than zero"):
        parse_prompt("cat::1 dog::-2")


def test_plain_prompt_with_single_weight_counts_as_multi():
    parsed = parse_prompt("cat::2")
    assert parsed.is_multi and parsed.positive_text == "cat"


# --------------------------------------------------------------------------------------
# Permutations
# --------------------------------------------------------------------------------------


def test_permutations_in_text():
    assert expand_permutations("a {red, green, blue} bird") == ["a red bird", "a green bird", "a blue bird"]


def test_permutations_multiply_and_reach_into_parameters():
    prompts = expand_permutations("a {cat, dog} --ar {1:1, 16:9}")
    assert prompts == ["a cat --ar 1:1", "a cat --ar 16:9", "a dog --ar 1:1", "a dog --ar 16:9"]
    assert [parse_prompt(p).params.aspect for p in prompts] == [(1, 1), (16, 9), (1, 1), (16, 9)]


def test_nested_permutations():
    assert expand_permutations("a {sculpture, painting} of a {seagull {on a pier, on a beach}, poodle}") == [
        "a sculpture of a seagull on a pier",
        "a sculpture of a seagull on a beach",
        "a sculpture of a poodle",
        "a painting of a seagull on a pier",
        "a painting of a seagull on a beach",
        "a painting of a poodle",
    ]


def test_escaped_commas_and_braces():
    assert expand_permutations(r"{red\, white, blue} flag") == ["red, white flag", "blue flag"]
    assert expand_permutations(r"a \{literal\} brace") == ["a {literal} brace"]


def test_no_braces_is_a_single_prompt():
    assert expand_permutations("just a prompt --ar 1:1") == ["just a prompt --ar 1:1"]


def test_duplicate_expansions_are_collapsed():
    assert expand_permutations("{cat, cat} photo") == ["cat photo"]


def test_empty_option_is_allowed():
    assert expand_permutations("a {, fluffy} cat") == ["a cat", "a fluffy cat"]


def test_unbalanced_brace_is_an_error():
    with pytest.raises(PromptError, match="without a matching"):
        expand_permutations("a {red, green bird")


def test_permutation_limit():
    with pytest.raises(PromptError, match="more than 40"):
        expand_permutations("{a,b,c,d,e,f,g} {1,2,3,4,5,6}")
    assert len(expand_permutations("{a,b,c,d,e} {1,2,3,4,5,6,7,8}")) == 40


# --------------------------------------------------------------------------------------
# Option sets, extensions, support levels
# --------------------------------------------------------------------------------------


def test_option_sets_expand_like_prefer_option_set():
    sets = {"wide": "--ar 21:9 --stylize 300", "mine": "--model qwen-image"}
    parsed = parse_prompt("a desert road --wide —mine --chaos 5", option_sets=sets)
    # Ratios are stored in lowest terms: 21:9 is 7:3.
    assert parsed.params.aspect == (7, 3) and parsed.params.stylize == 300
    assert parsed.params.model == "qwen-image" and parsed.params.chaos == 5


def test_option_sets_cannot_shadow_real_parameters():
    parsed = parse_prompt("x --chaos 5", option_sets={"chaos": "--seed 1"})
    assert parsed.params.chaos == 5 and parsed.params.seed is None


def test_extensions():
    params = parse_prompt(
        "x --model flux2-klein-4b --steps 8 --cfg 3.5 --strength 0.6 --n 2 --neg blurry, low quality"
    ).params
    assert params.model == "flux2-klein-4b" and params.steps == 8 and params.guidance == 3.5
    assert params.strength == 0.6 and params.count == 2 and params.negative == "blurry, low quality"


def test_loras():
    params = parse_prompt("x --lora watercolor:0.8 org/repo-name another:-0.5").params
    assert params.loras == [
        LoraRef("watercolor", 0.8),
        LoraRef("org/repo-name", 1.0),
        LoraRef("another", -0.5),
    ]
    with pytest.raises(PromptError, match="between -4 and 4"):
        parse_prompt("x --lora big:9")


def test_niji_and_personalize_take_optional_values():
    assert parse_prompt("x --niji").params.niji == "default"
    assert parse_prompt("x --niji 7").params.niji == "7"
    assert parse_prompt("x --p").params.personalize == ["default"]
    assert parse_prompt("x --profile abc123 m789").params.personalize == ["abc123", "m789"]


def test_planned_and_ignored_parameters_are_reported_not_rejected():
    parsed = parse_prompt("x --video --motion high --stealth --turbo --tile")
    assert parsed.planned == ["video", "motion", "tile"]
    assert parsed.ignored == ["stealth", "turbo"]
    assert parsed.params.speed == "turbo" and parsed.params.visibility == "stealth"


def test_given_and_merge():
    typed = parse_prompt("x --ar 16:9 --no cats").params
    defaults = parse_prompt("x --ar 1:1 --stylize 250 --no dogs").params
    merged = typed.merged_over(defaults)
    assert merged.aspect == (16, 9) and merged.stylize == 250 and merged.no == ["cats"]
    assert set(typed.given()) == {"aspect", "no"}


# --------------------------------------------------------------------------------------
# The parameter table itself
# --------------------------------------------------------------------------------------


def test_every_parameter_is_documented_and_targets_a_real_field():
    from dataclasses import fields

    from dreambot.core.params import Params

    names = {f.name for f in fields(Params)}
    for spec in PARAMS:
        assert spec.summary, f"--{spec.name} has no summary"
        assert spec.dest in names, f"--{spec.name} targets unknown field {spec.dest}"
        if spec.support is Support.EMULATED:
            assert spec.how, f"--{spec.name} is emulated but does not say how"
        if spec.kind is Kind.CHOICE:
            assert spec.choices


def test_midjourney_parameter_list_is_covered():
    """Every parameter in Midjourney's current list (plus the legacy ones people still type)."""
    midjourney = [
        "aspect",
        "ar",
        "chaos",
        "c",
        "no",
        "quality",
        "q",
        "seed",
        "sameseed",
        "stop",
        "raw",
        "style",
        "stylize",
        "s",
        "tile",
        "weird",
        "w",
        "iw",
        "sref",
        "sw",
        "sv",
        "cref",
        "cw",
        "oref",
        "ow",
        "edit",
        "p",
        "profile",
        "personalize",
        "niji",
        "v",
        "version",
        "repeat",
        "r",
        "draft",
        "exp",
        "hd",
        "sd",
        "preview",
        "video",
        "motion",
        "loop",
        "end",
        "bs",
        "fast",
        "relax",
        "turbo",
        "stealth",
        "public",
    ]
    missing = [name for name in midjourney if find_param(name) is None]
    assert not missing
