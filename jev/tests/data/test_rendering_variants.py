"""Rendering-variant contracts: golden v1 bytes, phrase banks, noise profiles.

The released pilot rows pin every rule's v1 surface by hash. The phrase bank
path must reproduce those bytes exactly when the phrase index is 0 (or the
rendering stays a plain style string), while variant dicts select different
wordings and transcript-noise profiles without ever touching the frozen
decision semantics or the curate fidelity-flag gates.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from jev_data.arabic import fidelity_flags
from jev_data.generate import oracle_candidate
from jev_data.oracle import check_oracle
from jev_data.registry import Registry
from jev_data.rules import RULES, rule_for
from jev_data.rules._shared import STYLES
from jev_data.rules.phrasings import NOISE_PROFILES, phrase_count, phrase_slots

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden_v1_surfaces.jsonl"


def _golden():
    rows = [json.loads(line) for line in
            FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_kind = {kind: [row for row in rows if row["world"]["kind"] == kind]
               for kind in RULES}
    return rows, by_kind


GOLDEN, BY_KIND = _golden()


def fixture_registry():
    return Registry.from_dict({"schema_version": 1, "sources": [{
        "id": "commissioned", "status": "approved", "url": "internal://fixture-source",
        "revision": "fixture-v1", "sha256": "a" * 64, "license_ref": "fixture-rights",
        "attribution": "Test authors", "intended_uses": [
            "training", "evaluation", "redistribution", "derivatives"],
        "training": True, "redistribution": True, "derivatives": True, "consent": True,
        "approval_ref": "test-only-approval", "data_class": "commissioned",
        "original_language": "ar", "original_split": "unassigned",
        "collected_at": "2026-01-01", "related_sources": [],
    }]})


# ------------------------------------------------------------- golden bytes


@pytest.mark.parametrize("kind", sorted(RULES))
def test_golden_v1_bytes_are_reproducible_per_rule(kind):
    assert BY_KIND[kind], f"golden fixture has no {kind} rows"
    for row in BY_KIND[kind]:
        rendered = RULES[kind].surface(row["world"])
        assert rendered["state"] == row["state"], row["world"]["family_id"]
        assert rendered["question"] == row["question"], row["world"]["family_id"]


def test_golden_fixture_covers_every_kind_style_and_scenario():
    combos = {(row["world"]["kind"], row["world"]["rendering"]) for row in GOLDEN}
    for kind in RULES:
        for style in STYLES:
            assert (kind, style) in combos, f"fixture misses {kind}/{style}"
    scenarios = {row["world"].get("scenario") for row in GOLDEN
                 if "scenario" in row["world"]}
    assert scenarios == {"single", "installments", "business_hours", "sla"}


@pytest.mark.parametrize("kind", sorted(RULES))
@pytest.mark.parametrize("style", sorted(STYLES))
def test_variant_dict_defaults_reproduce_plain_style_bytes(kind, style):
    world = dict(BY_KIND[kind][0]["world"], rendering=style)
    plain = RULES[kind].surface(world)
    variant = RULES[kind].surface(dict(world, rendering={"style": style}))
    assert variant == plain


# --------------------------------------------------------- phrase variation


@pytest.mark.parametrize("kind", sorted(RULES))
@pytest.mark.parametrize("style", sorted(STYLES))
def test_phrase_indices_keep_decisions_options_and_vary_bytes(kind, style):
    world = dict(BY_KIND[kind][0]["world"], rendering=style)
    rule = RULES[kind]
    plain = rule.surface(world)
    decision = rule.solve(world)
    count = phrase_count(kind, style)
    for phrase in sorted({0, 1, 2, count - 1}):
        varied = dict(world, rendering={"style": style, "phrase": phrase})
        rendered = rule.surface(varied)
        assert rule.solve(varied) == decision
        assert rule.independent_solve(varied) == decision
        assert rendered["question"]["options"] == plain["question"]["options"]
        assert rendered["question"]["type"] == plain["question"]["type"]
    # The top index takes the last choice of every slot, so at least the final
    # question slot wording must differ from the v1 bytes.
    last = rule.surface(dict(world, rendering={"style": style, "phrase": count - 1}))
    assert json.dumps(last, sort_keys=True, ensure_ascii=False) != json.dumps(
        plain, sort_keys=True, ensure_ascii=False), "top phrase must change the bytes"


def test_phrase_slots_decompose_every_index_uniquely():
    for kind in sorted(RULES):
        for style in STYLES:
            count = phrase_count(kind, style)
            sampled = range(count) if count <= 100 else [0, 1, count // 2, count - 1]
            picks = {json.dumps(phrase_slots(kind, style, index), sort_keys=True,
                                ensure_ascii=False) for index in sampled}
            assert len(picks) == len(list(sampled)), f"{kind}/{style}"


# ------------------------------------------------------------ noise profiles


@pytest.mark.parametrize("noise", range(len(NOISE_PROFILES)))
def test_noise_profiles_are_flag_safe_on_every_golden_row(noise):
    for row in GOLDEN:
        world = dict(row["world"], rendering={"style": row["world"]["rendering"],
                                              "noise": noise})
        rendered = rule_for(world).surface(world)
        assert set(fidelity_flags(rendered["state"])) <= {"negation_present"}, (
            row["world"]["family_id"], NOISE_PROFILES[noise])


@pytest.mark.parametrize("noise", range(1, len(NOISE_PROFILES)))
def test_every_active_noise_profile_changes_some_state(noise):
    changed = 0
    for row in GOLDEN:
        world = dict(row["world"], rendering={"style": row["world"]["rendering"],
                                              "noise": noise})
        if rule_for(world).surface(world)["state"] != row["state"]:
            changed += 1
    assert changed, f"profile {NOISE_PROFILES[noise]} never changed a state"


@pytest.mark.parametrize("kind", sorted(RULES))
def test_noise_never_touches_questions_or_decisions(kind):
    world = BY_KIND[kind][0]["world"]
    rule = RULES[kind]
    decision = rule.solve(world)
    baseline = rule.surface(world)["question"]
    for style in STYLES:
        for noise in range(len(NOISE_PROFILES)):
            noised_world = dict(world, rendering={"style": style, "noise": noise})
            rendered = rule.surface(noised_world)
            assert rule.solve(noised_world) == decision
            assert rule.independent_solve(noised_world) == decision
            assert rendered["question"] == rule.surface(
                dict(world, rendering=style))["question"]


# -------------------------------------------------------- variant validation


def _budget_world():
    return copy.deepcopy(BY_KIND["budget_check"][0]["world"])


@pytest.mark.parametrize("rendering, message", [
    ({"style": "msa", "phrase": 10**9}, "phrase index outside the bank"),
    ({"style": "msa", "phrase": -1}, "phrase index outside the bank"),
    ({"style": "msa", "noise": 99}, "noise index outside the profiles"),
    ({"style": "msa", "extra": 1}, "unknown rendering variant keys"),
    ({"phrase": 0}, "rendering variant requires a style"),
    ({"style": "klingon"}, "unknown rendering style"),
    ({"style": True}, "unknown rendering style"),
    ({"style": "msa", "phrase": True}, "phrase index must be an int"),
    ({"style": "msa", "noise": False}, "noise index must be an int"),
    (["msa"], "rendering must be a style string or a variant object"),
])
def test_invalid_variant_renderings_are_rejected(rendering, message):
    world = _budget_world()
    world["rendering"] = rendering
    with pytest.raises(Exception, match=message):
        RULES["budget_check"].surface(world)


def test_variant_renderings_round_trip_through_the_oracle_pipeline():
    world = copy.deepcopy(BY_KIND["budget_check"][0]["world"])
    world.update(family_id="variant-budget-0", template_id="variant-budget-tpl00",
                 rendering={"style": "egyptian", "phrase": 5, "noise": 2})
    row = oracle_candidate(world, identifier="variant-000000",
                           source_id="commissioned", registry=fixture_registry())
    check_oracle(row)
    assert (row["language"], row["variety"]) == ("ar", "Egyptian")
    plain = RULES["budget_check"].surface(
        dict(world, rendering="egyptian"))["state"]
    assert row["state"] != plain, "phrase index must change the rendered bytes"
    # noise profile 2 is ta_marbuta: final ta/alif maqsura softened independently
    # of the rule code under test.
    phrase_only = RULES["budget_check"].surface(
        dict(world, rendering={"style": "egyptian", "phrase": 5}))["state"]
    softened = re.sub(r"ى(?=[\s.,;:]|$)", "ي",
                      re.sub(r"ة(?=[\s.,;:]|$)", "ه", phrase_only))
    assert row["state"] == softened


def test_attributes_for_reads_the_style_of_a_variant_dict():
    for kind in sorted(RULES):
        world = dict(BY_KIND[kind][0]["world"],
                     rendering={"style": "gulf", "phrase": 1, "noise": 1})
        attributes = RULES[kind].attributes_for(world)
        plain = RULES[kind].attributes_for(
            dict(BY_KIND[kind][0]["world"], rendering="gulf"))
        assert attributes == plain, kind
        assert (attributes["language"], attributes["variety"]) == ("ar", "Gulf")
