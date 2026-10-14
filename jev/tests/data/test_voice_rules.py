"""The nine executable oracle rules: frozen semantics, contracts and coverage.

tools/worlds_voice_v1.py self-checks every generated world at generation time;
these tests pin the contracts it relies on: the uniform module interface, the
frozen rule versions, boundary semantics per family, style invariance of the
decision, audit-envelope tamper detection and generator determinism plus full
kind/style/scenario coverage.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import random
from pathlib import Path

import pytest

from jev_contracts import ValidationError
from jev_data.generate import oracle_candidate
from jev_data.oracle import check_oracle
from jev_data.registry import Registry
from jev_data.rules import RULES, rule_for
from jev_data.rules._shared import ORIGINS, STYLES

ROOT = Path(__file__).resolve().parents[2]


def load_generator():
    path = ROOT / "tools/worlds_voice_v1.py"
    spec = importlib.util.spec_from_file_location("worlds_voice_v1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _world(kind, marker, style="msa", **body):
    return {"schema_version": 1, "kind": kind, "family_id": f"voice-{kind}-{marker}",
            "template_id": f"voice-{kind}-tpl-{marker}", "rendering": style, **body}


def decisions(kind, world, expected):
    rule = RULES[kind]
    rule.validate(world)
    assert rule.solve(world) == expected
    assert rule.independent_solve(world) == expected


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


def test_rule_registry_pins_frozen_versions_and_uniform_interface():
    expected = {
        "comparison": ("rational-comparison-v1", "numeric_policy"),
        "temporal_window": ("temporal-window-v1", "temporal_policy"),
        "policy_logic": ("policy-logic-v1", "policy_logic"),
        "budget_check": ("budget-check-v1", "budget_check"),
        "intent_routing": ("intent-routing-v1", "intent_routing"),
        "safety_gate": ("safety-gate-v1", "safety_gate"),
        "triage_rubric": ("triage-rubric-v1", "triage_rubric"),
        "readback_match": ("readback-match-v1", "readback_match"),
        "slot_assertion": ("slot-assertion-v1", "slot_assertion"),
    }
    assert {kind: (module.RULE_VERSION, module.TASK) for kind, module in RULES.items()} == expected
    for module in RULES.values():
        for name in ("validate", "solve", "independent_solve", "surface", "attributes_for"):
            assert callable(getattr(module, name))


def test_rule_for_rejects_unknown_and_malformed_worlds():
    with pytest.raises(ValidationError, match="object"):
        rule_for([])
    with pytest.raises(ValidationError, match="unsupported symbolic world"):
        rule_for({})
    with pytest.raises(ValidationError, match="unsupported symbolic world"):
        rule_for({"kind": "no_such_kind"})


# ---------------------------------------------------------------- per family


def test_comparison_boundaries_and_cross_unit_equality():
    def world(quantity, threshold, operator):
        return _world("comparison", "cmp", entity="طلب رقم 1", quantity=quantity,
                      threshold=threshold, operator=operator)

    decisions("comparison", world({"value": "1.5", "unit": "kg"},
                                  {"value": "1500", "unit": "g"}, "=="), "eligible")
    decisions("comparison", world({"value": "100", "unit": "SAR"},
                                  {"value": "100", "unit": "SAR"}, "<"), "ineligible")
    decisions("comparison", world({"value": "100", "unit": "SAR"},
                                  {"value": "100", "unit": "SAR"}, "<="), "eligible")
    decisions("comparison", world({"value": "3", "unit": "m"},
                                  {"value": "300", "unit": "cm"}, "!="), "ineligible")
    decisions("comparison", world(None, {"value": "100", "unit": "SAR"}, ">"), "insufficient")


def test_temporal_window_boundaries_weekend_and_sla():
    def business(day, opens, closes, event, minutes, weekend):
        return _world("temporal_window", "tmp", scenario="business_hours", date=day,
                      opens=opens, closes=closes, event=event, duration_minutes=minutes,
                      weekend=weekend)

    # 2026-11-02 is a Monday; a 60-minute appointment ending exactly at closing fits.
    decisions("temporal_window", business("2026-11-02", "09:00", "17:00", "16:00", 60,
                                          ["friday", "saturday"]), "eligible")
    decisions("temporal_window", business("2026-11-02", "09:00", "17:00", "16:30", 45,
                                          ["friday", "saturday"]), "ineligible")
    decisions("temporal_window", business("2026-11-02", "22:00", "02:00", "23:30", 60,
                                          ["friday", "saturday"]), "eligible")
    decisions("temporal_window", business("2026-11-07", "09:00", "17:00", "11:00", 60,
                                          ["saturday"]), "ineligible")
    decisions("temporal_window", business("2026-11-02", "09:00", "17:00", None, 45,
                                          ["friday", "saturday"]), "insufficient")

    def sla(replied, sla_minutes):
        return _world("temporal_window", "sla", scenario="sla", opened="2026-11-02T09:00",
                      replied=replied, sla_minutes=sla_minutes)

    decisions("temporal_window", sla("2026-11-02T13:00", 240), "true")
    decisions("temporal_window", sla("2026-11-02T13:00", 239), "false")


def policy_world(combine, clauses, facts, marker="pol"):
    labels = {name: {"ar": f"شرط {name}", "en": f"requirement {name}"}
              for name, _, _ in clauses}
    return _world("policy_logic", marker, entity="طلب", combine=combine,
                  clauses=[{"fact": name, "op": op, "value": value}
                           for name, op, value in clauses],
                  labels=labels, facts=facts)


def test_policy_logic_kleene_all_any_and_missing_facts():
    refund = [("days", "<=", 14), ("opened", "==", False), ("receipt", "==", True)]
    decisions("policy_logic", policy_world("all", refund,
                                           {"days": 7, "opened": False, "receipt": True}),
              "eligible")
    # A false requirement dominates even when another is unstated.
    decisions("policy_logic", policy_world("all", refund, {"days": 25}), "ineligible")
    # All stated requirements hold but one is unstated.
    decisions("policy_logic", policy_world("all", refund, {"days": 7, "opened": False}),
              "insufficient")

    enrollment = [("age", ">=", 18), ("consent", "==", True)]
    decisions("policy_logic", policy_world("any", enrollment,
                                           {"age": 21, "consent": False}), "eligible")
    decisions("policy_logic", policy_world("any", enrollment, {"age": 16}), "insufficient")
    decisions("policy_logic", policy_world("any", enrollment,
                                           {"age": 16, "consent": False}), "ineligible")


def budget_world(scenario, marker="bud", **amounts):
    unit = {"ar": "ريال", "en": "riyals"}
    return _world("budget_check", marker, scenario=scenario, entity="طلب صرف", unit=unit,
                  **amounts)


def test_budget_check_exact_boundaries_and_abstentions():
    decisions("budget_check", budget_world("single", total=1000, spent=400, requested=600),
              "eligible")
    decisions("budget_check", budget_world("single", total=1000, spent=400, requested=601),
              "ineligible")
    decisions("budget_check", budget_world("single", total=1000, spent=None, requested=100),
              "insufficient")
    decisions("budget_check",
              budget_world("installments", price=1200, installment=100, months=12), "eligible")
    decisions("budget_check",
              budget_world("installments", price=1201, installment=100, months=12), "ineligible")
    decisions("budget_check",
              budget_world("installments", price=1200, installment=100, months=None),
              "insufficient")


def routing_world(intents, utterance, marker="rte"):
    tokens = {token for _, keywords in intents for token in keywords} | set(utterance)
    lexicon = {token: {"ar": f"كلمة {token}", "en": f"word {token}"} for token in tokens}
    return _world("intent_routing", marker, domain={"ar": "المجال", "en": "domain"},
                  intents=[{"id": identifier,
                            "label": {"ar": f"نية {identifier}", "en": f"intent {identifier}"},
                            "keywords": keywords} for identifier, keywords in intents],
                  utterance=utterance, lexicon=lexicon)


def test_intent_routing_ties_shared_keywords_and_fallback():
    three = [("a", ["a1", "a2"]), ("b", ["b1", "b2"]), ("c", ["c1"])]
    # Tie on distinct matches resolves to the intent listed first.
    decisions("intent_routing", routing_world(three, ["a1", "b1"]), "a")
    decisions("intent_routing", routing_world(three, ["a1", "a2", "b1"]), "a")
    # No keyword matches route to the explicit fallback.
    decisions("intent_routing", routing_world(three, ["f1"]), "other")
    # A shared keyword scores once for every intent that lists it.
    shared = [("a", ["a1", "s1"]), ("b", ["b1", "s1"])]
    decisions("intent_routing", routing_world(shared, ["s1"]), "a")
    # Distinct matches for one intent beat a single match elsewhere.
    decisions("intent_routing", routing_world(three, ["a1", "b1", "b2"]), "b")

    question = RULES["intent_routing"].surface(routing_world(three, ["a1"]))["question"]
    assert question["options"][-1]["id"] == "other"


def safety_world(rules, facts, default, marker="saf"):
    labels = {}
    for _, clauses in rules:
        for fact, _, _ in clauses:
            labels.setdefault(fact, {"ar": f"حالة {fact}", "en": f"fact {fact}"})
    return _world("safety_gate", marker, request={"ar": "طلب", "en": "request"},
                  facts=facts, fact_labels=labels,
                  rules=[{"when": [{"fact": fact, "op": op, "value": value}
                                   for fact, op, value in clauses], "outcome": outcome}
                         for outcome, clauses in rules], default=default)


def test_safety_gate_first_match_gaps_and_defaults():
    transfer = [("allow", [("amount", "<=", 1000)]), ("escalate", [("amount", ">", 1000)])]
    decisions("safety_gate", safety_world(transfer, {"amount": 350}, "block"), "allow")
    decisions("safety_gate", safety_world(transfer, {"amount": 1500}, "block"), "escalate")
    # A missing fact escalates at its own rule instead of falling through.
    gap = [("allow", [("dose_known", "==", True)]), ("block", [("other", "==", True)])]
    decisions("safety_gate", safety_world(gap, {}, "escalate"), "escalate")
    # A present clause that fails shields the rule's other missing facts.
    shielded = [("block", [("age", "<", 18), ("third", "==", True)]),
                ("allow", [("age", ">=", 18)])]
    decisions("safety_gate", safety_world(shielded, {"age": 25}, "escalate"), "allow")
    # No rule applies -> the stated default outcome.
    fallback = [("block", [("consent", "==", False)])]
    decisions("safety_gate", safety_world(fallback, {"consent": True}, "escalate"), "escalate")


def triage_world(items, thresholds, marker="tri"):
    levels = [{"label": {"ar": f"مستوى {index}", "en": f"level {index}"}}
              for index in range(len(thresholds))]
    return _world("triage_rubric", marker, entity="تذكرة",
                  items=[{"id": f"it{index}",
                          "label": {"ar": f"بند {index}", "en": f"item {index}"},
                          "points": points, "met": met}
                         for index, (points, met) in enumerate(items)],
                  levels=levels, thresholds=thresholds)


def test_triage_rubric_threshold_edges():
    decisions("triage_rubric", triage_world([(3, True), (5, True), (4, True)], [0, 9]), "1")
    decisions("triage_rubric", triage_world([(3, True), (5, False)], [0, 9]), "0")
    decisions("triage_rubric", triage_world([(5, True), (1, False)], [0, 5, 10]), "1")
    decisions("triage_rubric", triage_world([(5, True), (5, True)], [0, 5, 10]), "2")
    question = RULES["triage_rubric"].surface(
        triage_world([(5, True), (5, True)], [0, 5, 10]))["question"]
    assert question["type"] == "score"
    assert [option["id"] for option in question["options"]] == ["0", "1", "2"]


def readback_world(expected, readback, marker="rdb"):
    return _world("readback_match", marker, expected=expected, readback=readback)


def test_readback_units_dimensions_and_absence():
    decisions("readback_match", readback_world({"value": "1.5", "unit": "kg"},
                                               {"value": "1500", "unit": "g"}), "true")
    decisions("readback_match", readback_world({"value": "3", "unit": "m"},
                                               {"value": "3", "unit": "kg"}), "false")
    decisions("readback_match", readback_world({"value": "120", "unit": "SAR"}, None), "false")
    decisions("readback_match", readback_world({"value": "2", "unit": "hour"},
                                               {"value": "120", "unit": "minute"}), "true")
    decisions("readback_match", readback_world({"value": "90", "unit": "minute"},
                                               {"value": "2", "unit": "hour"}), "false")
    decisions("readback_match", readback_world({"value": "0.5", "unit": "SAR"},
                                               {"value": "50", "unit": "halala"}), "true")


def slots_world(form, claim, marker="slt"):
    references = sorted({field for field, _ in form} | {claim[0]})
    labels = {field: {"ar": f"حقل {field}", "en": f"field {field}"} for field in references}
    return _world("slot_assertion", marker,
                  form=[{"field": field, "value": value} for field, value in form],
                  claim={"field": claim[0], "value": claim[1]}, field_labels=labels)


def test_slot_assertion_literal_equality_and_absence():
    decisions("slot_assertion", slots_world([("name", "سارة")], ("name", "سارة")), "true")
    decisions("slot_assertion", slots_world([("name", "سارة")], ("name", "نورة")), "false")
    decisions("slot_assertion", slots_world([("phone", "0501234567")],
                                            ("address", "الرياض")), "false")
    decisions("slot_assertion", slots_world([("phone", "0501234567")],
                                            ("address", None)), "true")
    decisions("slot_assertion", slots_world([("phone", "0501234567")],
                                            ("phone", None)), "false")
    decisions("slot_assertion", slots_world([("email", "Ali@Example.com")],
                                            ("email", "ali@example.com")), "false")


# ------------------------------------------------- styles, audit and generator


def test_styles_never_change_the_decision_and_keep_option_contracts():
    generator = load_generator()
    question_types = {"triage_rubric": "score", "readback_match": "noul",
                      "slot_assertion": "noul"}
    for kind in generator.KINDS:
        rng = random.Random(f"style-invariance:{kind}")
        body, expected = generator.BUILDERS[kind](0, "msa", rng)
        world = {"schema_version": 1, "kind": kind, "family_id": f"inv-{kind}",
                 "template_id": f"inv-{kind}-tpl00", **body}
        rule = rule_for(world)
        assert rule.solve(world) == expected
        msa = rule.surface(world)["question"]
        assert msa["type"] == question_types.get(kind, "choice")
        for style in STYLES:
            styled = {**world, "rendering": style}
            assert rule.solve(styled) == expected
            assert rule.independent_solve(styled) == expected
            attributes = rule.attributes_for(styled)
            assert (attributes["language"], attributes["variety"]) == STYLES[style]
            assert attributes["origin"] == ORIGINS[attributes["language"]]
            assert attributes["task"] == rule.TASK
            assert rule.surface(styled)["question"]["type"] == msa["type"]
        english = rule.surface({**world, "rendering": "english"})["question"]["options"]
        assert english != msa["options"]
        options = [option["id"] for option in msa["options"]]
        if kind in ("readback_match", "slot_assertion"):
            assert options == ["false", "true"]
        elif kind == "triage_rubric":
            assert options == [str(index) for index in range(len(world["levels"]))]
        elif kind == "intent_routing":
            assert options[-1] == "other" and len(set(options)) == len(options)
        elif kind == "safety_gate":
            assert sorted(options) == ["allow", "block", "escalate"]
        else:
            assert options == ["eligible", "ineligible", "insufficient"]


def test_check_oracle_detects_target_surface_and_world_tampering():
    generator = load_generator()
    for index, kind in enumerate(generator.KINDS):
        rng = random.Random(f"tamper:{kind}")
        body, _ = generator.BUILDERS[kind](0, "msa", rng)
        world = {"schema_version": 1, "kind": kind, "family_id": f"tamper-{kind}-0",
                 "template_id": f"tamper-{kind}-tpl00", **body}
        row = oracle_candidate(world, identifier=f"tamper-{index:06d}",
                               source_id="commissioned", registry=fixture_registry())

        option_ids = [option["id"] for option in row["question"]["options"]]
        wrong = next(option for option in option_ids if option != row["target"]["option_id"])
        retargeted = copy.deepcopy(row)
        retargeted["target"]["option_id"] = wrong
        with pytest.raises(ValidationError, match="target"):
            check_oracle(retargeted)

        tweaked = copy.deepcopy(row)
        tweaked["state"] = row["state"] + " "
        with pytest.raises(ValidationError, match="surface"):
            check_oracle(tweaked)

        mutated = copy.deepcopy(row)
        mutated["provenance"]["metadata"]["oracle"]["world"] = {
            **world, "family_id": f"tamper-{kind}-other"}
        with pytest.raises(ValidationError, match="hash"):
            check_oracle(mutated)


def test_voice_generator_is_deterministic_and_fully_covered(tmp_path):
    generator = load_generator()
    for root_name in ("first", "second"):
        assert generator.main(["--root", str(tmp_path / root_name)]) == 0
    first = tmp_path / "first/runs"
    second = tmp_path / "second/runs"
    for name in ("worlds/worlds-voice-v1-train.jsonl", "worlds/worlds-voice-v1-dev.jsonl",
                 "worlds/worlds-voice-v1-cal.jsonl", "worlds/worlds-voice-v1-test.jsonl",
                 "reports/reservations-voice-v1.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name

    counts = {"train": 648, "dev": 216, "cal": 108, "test": 216}
    coverage = set()
    for stem, count in counts.items():
        lines = (first / f"worlds/worlds-voice-v1-{stem}.jsonl").read_text(
            encoding="utf-8").splitlines()
        assert len(lines) == count
        for line in lines:
            world = json.loads(line)
            scenario = int(world["template_id"].rsplit("-tpl", 1)[1])
            coverage.add((world["kind"], world["rendering"], scenario))
            rule = rule_for(world)
            assert rule.solve(world) == rule.independent_solve(world)
    expected = {(kind, style, scenario) for kind in generator.KINDS for style in generator.STYLES
                for scenario in range(12)}
    assert coverage == expected

    reservations = json.loads((first / "reports/reservations-voice-v1.json").read_text(
        encoding="utf-8"))
    assert len(reservations) == 1188
    per_split = {"train": 0, "development": 0, "calibration": 0, "test": 0}
    for split in reservations.values():
        per_split[split] += 1
    assert per_split == {"train": 648, "development": 216, "calibration": 108, "test": 216}
