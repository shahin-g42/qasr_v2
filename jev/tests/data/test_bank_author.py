"""Phase-1 bank authoring: rejection sampling, validators, seed draft, apply.

tools/bank_author.py drafts the prod-v1 scenario banks by greedy rejection
sampling - structural validity, unique keys against the pinned voice-v1
tables, and pairwise similarity ratio < 0.85 against every pinned and kept
entry - and commits reviewed drafts as code constants. These tests pin the
selection engine, the validators, the pinned-exclusion wiring, the full seed
draft pipeline and both apply paths, including the append-safe
phrasings_llm.py merge into rules/phrasings.py.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"tools/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BA = _load_tool("bank_author")


@pytest.fixture(scope="module")
def seed_draft(tmp_path_factory) -> Path:
    output = tmp_path_factory.mktemp("seed-bank") / "draft"
    assert BA.main(["draft", "--source", "seed", "--output", str(output)]) == 0
    return output


# ---------------------------------------------------------------- selection


def test_select_keeps_valid_entries_up_to_target_in_order():
    words = (("عنبر", "amber"), ("ياقوت", "ruby"), ("فيروز", "turquoise"),
             ("مرجان", "coral"), ("عقيق", "agate"))
    candidates = [({"ar": ar, "en": en}, {"source": "t"}) for ar, en in words]
    kept, rejected = BA.select(
        candidates, target=3, validate=BA._bilingual, key_of=lambda e: e["ar"],
        ratio_of=lambda e: e["ar"], existing_keys=[], existing_ratios=[])
    assert [record["entry"]["en"] for record in kept] == ["amber", "ruby", "turquoise"]
    assert rejected == []
    assert kept[0]["provenance"] == {"source": "t"}


def test_select_rejects_invalid_duplicate_and_too_close_entries():
    pinned_entity = BA.PINNED["entities"]["comparison"][0]
    close_string = pinned_entity + "ه"
    candidates = [
        ({"ar": "", "en": "empty arabic"}, {"source": "t"}),          # invalid
        ({"ar": pinned_entity, "en": "the financing request"}, {"source": "t"}),  # dup key
        ({"ar": close_string, "en": "monthly bill variant"}, {"source": "t"}),    # ratio
        ({"ar": "خطة صيانة المصاعد", "en": "the elevator maintenance plan"}, {"source": "t"}),
    ]
    kept, rejected = BA.select(
        candidates, target=4, validate=BA._bilingual, key_of=lambda e: e["ar"],
        ratio_of=lambda e: e["ar"], existing_keys=[pinned_entity],
        existing_ratios=[pinned_entity])
    assert [record["entry"]["en"] for record in kept] == ["the elevator maintenance plan"]
    assert [record["reason"].split(":")[0] for record in rejected[:2]] == [
        "invalid", "duplicates a pinned or kept entry"]
    assert rejected[2]["reason"].startswith("too close")


def test_select_without_ratio_check_only_enforces_keys():
    candidates = [({"opens": "08:00", "closes": "22:00"}, {"source": "t"}),
                  ({"opens": "08:00", "closes": "22:00"}, {"source": "t"})]
    kept, rejected = BA.select(
        candidates, target=2, validate=BA.validate_profile,
        key_of=lambda e: f"{e['opens']}|{e['closes']}", ratio_of=None,
        existing_keys=[], existing_ratios=[])
    assert len(kept) == 1
    assert "duplicates a pinned or kept entry" in rejected[0]["reason"]


# ---------------------------------------------------------------- validators


def _domain():
    return {
        "id": "insurance",
        "label": {"ar": "خدمات التأمين", "en": "insurance services"},
        "intents": [
            {"id": "ins_claim", "label": {"ar": "تقديم مطالبة", "en": "filing a claim"},
             "keywords": [{"ar": "مطالبة", "en": "claim"}, {"ar": "تعويض", "en": "compensation"},
                          {"ar": "ضرر", "en": "damage"}]},
            {"id": "ins_policy", "label": {"ar": "استعلام عن وثيقة", "en": "policy inquiry"},
             "keywords": [{"ar": "وثيقة", "en": "policy"}, {"ar": "تغطية", "en": "coverage"},
                          {"ar": "عقد", "en": "contract"}]},
            {"id": "ins_renew", "label": {"ar": "تجديد الوثيقة", "en": "renewing a policy"},
             "keywords": [{"ar": "تجديد", "en": "renewal"}, {"ar": "استمرار", "en": "continuation"},
                          {"ar": "مدة", "en": "term"}]},
            {"id": "ins_agent", "label": {"ar": "التواصل مع وكيل", "en": "reaching an agent"},
             "keywords": [{"ar": "وكيل", "en": "agent"},
                          {"ar": "مندوب", "en": "representative"},
                          {"ar": "فرع", "en": "branch"}]},
        ],
        "fillers": [{"ar": "استفسار", "en": "inquiry"}, {"ar": "شكرًا", "en": "thanks"},
                    {"ar": "رجاء", "en": "please"}],
    }


def test_validate_domain_accepts_a_well_formed_domain():
    BA.validate_domain(_domain())


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d.update(id="Bad-ID"), "bad domain id"),
    (lambda d: d["intents"].pop(), "a domain needs exactly 4 intents"),
    (lambda d: d["intents"][0]["keywords"].pop(), "an intent needs exactly 3 keywords"),
    (lambda d: d["intents"][0]["keywords"][1].update(
        ar=d["intents"][0]["keywords"][0]["ar"]), "keyword words must be pairwise distinct"),
    (lambda d: d["intents"][0]["label"].update(ar="not arabic"),
     "must contain Arabic"),
    (lambda d: d["intents"][0]["label"].update(
        en="None of the intents"), "routing fallback label"),
    (lambda d: d["fillers"][0].update(ar=d["intents"][0]["keywords"][0]["ar"]),
     "a filler collides with a keyword"),
    (lambda d: d["fillers"].pop(), "a domain needs exactly 3 fillers"),
])
def test_validate_domain_rejects_malformed_domains(mutate, message):
    domain = _domain()
    mutate(domain)
    with pytest.raises(Exception, match=message):
        BA.validate_domain(domain)


def test_validate_persona_accepts_and_rejects():
    persona = {"id": "persona-01", "region": "egypt", "formality": "casual",
               "urgency": "high", "note_ar": "متصل مستعجل من القاهرة",
               "note_en": "a hurried caller from Cairo"}
    BA.validate_persona(persona)
    for mutate, message in [
            (lambda p: p.update(region="atlantis"), "unknown persona region"),
            (lambda p: p.update(id="X"), "bad persona id"),
            (lambda p: p.update(note_ar="english note"), "note_ar must contain Arabic")]:
        broken = dict(persona)
        mutate(broken)
        with pytest.raises(Exception, match=message):
            BA.validate_persona(broken)


def test_validate_policy_fact_types_and_bounds():
    BA.validate_policy_fact({"fact": "min_days", "ar": "أقل مدة", "en": "minimum days",
                             "type": "number", "op": ">=", "value": 3})
    BA.validate_policy_fact({"fact": "verified", "ar": "توثيق", "en": "verified",
                             "type": "bool", "op": "==", "value": True})
    with pytest.raises(Exception, match="bool facts"):
        BA.validate_policy_fact({"fact": "verified", "ar": "توثيق", "en": "verified",
                                 "type": "bool", "op": ">", "value": True})
    with pytest.raises(Exception, match="integer outside bounds"):
        BA.validate_policy_fact({"fact": "huge", "ar": "ضخم", "en": "huge",
                                 "type": "number", "op": "<", "value": 10 ** 10})


def test_validate_triage_item_accepts_id_ar_en():
    BA.validate_triage_item({"id": "production", "ar": "أثر على بيئة الإنتاج",
                             "en": "production environment affected"})


def test_question_validator_enforces_style_rules():
    validate = BA.question_validator("msa", ("هل الشرط متحقق؟",))
    validate("هل تنطبق القاعدة على الحالة؟")
    for bad, message in [
            ("Does this end correctly?", "must end with"),
            ("هل الشرط متحقق؟", "duplicates a bank choice"),
            ("سؤال\nمتعدد الأسطر؟", "one line of text"),
            ("هل " + "طويل " * 60 + "؟", "at most 160 characters")]:
        with pytest.raises(Exception, match=message):
            validate(bad)


# -------------------------------------------------- pinned voice-v1 exclusions


def test_pinned_tables_cover_the_voice_v1_sources():
    assert len(BA.PINNED["domain_ids"]) == 6
    assert "banking" in BA.PINNED["domain_ids"]
    assert BA.PINNED["entities"]["comparison"]
    assert BA.PINNED["fact_names"]["safety_gate"]
    assert BA.PINNED["names"] and BA.PINNED["cities"]


def test_seed_candidates_cannot_duplicate_pinned_voice_v1_entities():
    spec = {s["name"]: s for s in BA.section_specs()}["entities.comparison"]
    candidates = [({"ar": BA.PINNED["entities"]["comparison"][0], "en": "dup"},
                   {"source": "t"})]
    kept, rejected = BA.select(
        candidates, target=1, validate=spec["validate"], key_of=spec["key"],
        ratio_of=spec["ratio"], existing_keys=spec["existing"],
        existing_ratios=spec["existing_ratios"])
    assert kept == []
    assert "duplicates a pinned" in rejected[0]["reason"]


# ------------------------------------------------------------------ sections


def test_section_specs_shape_and_flags():
    specs = {spec["name"]: spec for spec in BA.section_specs()}
    procedural = {name for name, spec in specs.items() if spec["procedural"]}
    assert procedural == {"values.phones", "values.dates", "values.amounts",
                          "values.emails", "values.references"}
    for name in procedural:
        assert specs[name]["prompt"] is None and specs[name]["seed"] is not None
    phrasing = {name: spec for name, spec in specs.items()
                if name.startswith("phrasings.")}
    assert phrasing
    for name, spec in phrasing.items():
        _, kind, style = name.split(".")
        final = BA.SLOTS[kind][-1]
        existing = tuple(BA.VOCAB[kind][style][final])
        assert spec["llm_only"] is True and spec["seed"] is None
        assert spec["target"] == 12 - len(existing)
        assert tuple(spec["existing"]) == existing
        assert spec["target"] >= 1
    for required in ("domains", "personas", "entities.comparison", "facts.policy_logic",
                     "triage_items", "temporal.profiles", "temporal.weekends",
                     "values.names", "values.cities"):
        assert required in specs


# -------------------------------------------------------------- seed draft


def test_seed_draft_writes_the_full_review_package(seed_draft):
    draft = json.loads((seed_draft / "banks-draft.json").read_text(encoding="utf-8"))
    assert draft["bank_version"] == BA.BANK_VERSION
    assert draft["provenance"]["source"] == "seed"
    assert "phrasings" not in draft
    specs = {spec["name"]: spec for spec in BA.section_specs()}
    for name, spec in specs.items():
        if spec["llm_only"]:
            continue
        node = draft
        for part in spec["draft_key"]:
            node = node[part]
        assert len(node) == spec["target"], name
        assert (seed_draft / f"review/{name}.review.jsonl").exists(), name
    summary = (seed_draft / "summary.txt").read_text(encoding="utf-8")
    assert "SHORT" not in summary
    assert "skipped (LLM-only" in summary


# ------------------------------------------------------------------- apply


def test_apply_commits_seed_banks_and_removes_stale_module(tmp_path, seed_draft):
    root = tmp_path / "repo"
    stale = root / "packages/data/src/jev_data/rules/phrasings_llm.py"
    stale.parent.mkdir(parents=True)
    stale.write_text("QUESTION_EXTENSIONS = {}\n", encoding="utf-8")
    rc = BA.main(["apply", "--draft", str(seed_draft / "banks-draft.json"),
                  "--approved-by", "unit-reviewer", "--root", str(root)])
    assert rc == 0
    committed = json.loads((root / "configs/data/banks_prod_v1.json")
                           .read_text(encoding="utf-8"))
    assert committed["bank_version"] == BA.BANK_VERSION
    assert "phrasings" not in committed
    assert committed["provenance"]["approved_by"] == "unit-reviewer"
    assert committed["provenance"]["applied_at"]
    assert not stale.exists(), "a phrasing-free apply must remove stale modules"


def test_apply_rejects_drafts_with_missing_or_unknown_sections(tmp_path, seed_draft):
    draft = json.loads((seed_draft / "banks-draft.json").read_text(encoding="utf-8"))
    del draft["personas"]
    draft.setdefault("phrasings", {})["nope"] = {"msa": ["سؤال؟"]}
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(draft, ensure_ascii=False), encoding="utf-8")
    rc = BA.main(["apply", "--draft", str(path), "--approved-by", "x",
                  "--root", str(tmp_path / "repo")])
    assert rc == 2
    assert not (tmp_path / "repo/configs/data/banks_prod_v1.json").exists()


LLM_EXTENSIONS = (
    "هل تنطبق القاعدة المعروضة على هذه الحالة؟",
    "هل يستقيم تطبيق الشرط على ما هو معروض أمامك؟",
    "هل تُعد هذه الحالة مطابقة للشرط المذكور؟",
    "هل القاعدة سارية على المعطيات المتاحة؟",
    "هل يستوفي الموقف المعروض مطلب القاعدة؟",
    "هل تنطبق الشروط المنصوص عليها هنا؟",
    "هل الحالة المعروضة خاضعة لهذه القاعدة؟",
    "هل يصدق الشرط على المعلومات المتاحة أمامك؟",
    "هل القاعدة قابلة للتطبيق على هذه المعطيات؟",
)


def test_apply_with_llm_phrasings_merges_the_bank(tmp_path, seed_draft):
    from jev_data.rules.phrasings import SLOTS, VOCAB, phrase_count, phrase_slots
    final = SLOTS["comparison"][-1]
    existing = tuple(VOCAB["comparison"]["msa"][final])
    if len(existing) >= 12:
        pytest.skip("comparison/msa already carries committed LLM extensions")
    assert len(LLM_EXTENSIONS) == 12 - len(existing)

    root = tmp_path / "repo"
    (root / "packages/contracts").mkdir(parents=True)
    shutil.copytree(ROOT / "packages/contracts/src", root / "packages/contracts/src",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "packages/data/src", root / "packages/data/src",
                    ignore=shutil.ignore_patterns("__pycache__"))

    draft = json.loads((seed_draft / "banks-draft.json").read_text(encoding="utf-8"))
    draft["phrasings"] = {"comparison": {"msa": list(LLM_EXTENSIONS)}}
    draft_path = tmp_path / "with-extensions.json"
    draft_path.write_text(json.dumps(draft, ensure_ascii=False), encoding="utf-8")
    rc = BA.main(["apply", "--draft", str(draft_path), "--approved-by", "unit-reviewer",
                  "--root", str(root)])
    assert rc == 0  # includes the fresh-interpreter verification inside apply

    module = root / "packages/data/src/jev_data/rules/phrasings_llm.py"
    assert module.exists()
    generated = module.read_text(encoding="utf-8")
    assert "QUESTION_EXTENSIONS" in generated
    assert LLM_EXTENSIONS[0] in generated
    committed = json.loads((root / "configs/data/banks_prod_v1.json")
                           .read_text(encoding="utf-8"))
    assert "phrasings" not in committed

    old_count = phrase_count("comparison", "msa")
    old_words = [phrase_slots("comparison", "msa", index) for index in range(old_count)]
    code = "\n".join([
        "import json, sys",
        "sys.path[:0] = [sys.argv[1], sys.argv[2]]",
        "from jev_data.rules.phrasings import SLOTS, VOCAB, phrase_count, phrase_slots",
        "final = SLOTS['comparison'][-1]",
        "choices = VOCAB['comparison']['msa'][final]",
        "assert len(choices) == 12, len(choices)",
        "assert choices[:len(json.loads(sys.argv[3]))] == tuple(json.loads(sys.argv[3]))",
        "assert choices[len(json.loads(sys.argv[3])):] == tuple(json.loads(sys.argv[4]))",
        "count = phrase_count('comparison', 'msa')",
        "assert count == int(sys.argv[5]), count",
        "for index, words in enumerate(json.loads(sys.argv[6])):",
        "    assert phrase_slots('comparison', 'msa', index) == words, index",
        "assert phrase_slots('comparison', 'msa', count - 1)['question'] "
        "== json.loads(sys.argv[4])[-1], phrase_slots('comparison', 'msa', count - 1)",
        "print('merged')",
    ])
    subprocess.run([sys.executable, "-c", code,
                    str(root / "packages/contracts/src"), str(root / "packages/data/src"),
                    json.dumps(list(existing), ensure_ascii=False),
                    json.dumps(list(LLM_EXTENSIONS), ensure_ascii=False),
                    str(old_count * 4),  # 3 pinned final choices grow to 4 radices
                    json.dumps(old_words, ensure_ascii=False)], check=True)


def test_apply_rejects_phrasings_duplicating_bank_choices(tmp_path, seed_draft):
    from jev_data.rules.phrasings import SLOTS, VOCAB
    final = SLOTS["comparison"][-1]
    existing = VOCAB["comparison"]["msa"][final]
    if len(existing) >= 12:
        pytest.skip("comparison/msa already carries committed LLM extensions")
    draft = json.loads((seed_draft / "banks-draft.json").read_text(encoding="utf-8"))
    draft["phrasings"] = {"comparison": {"msa": [existing[0], *LLM_EXTENSIONS[1:] ]}}
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(draft, ensure_ascii=False), encoding="utf-8")
    root = tmp_path / "repo"
    rc = BA.main(["apply", "--draft", str(path), "--approved-by", "x", "--root", str(root)])
    assert rc == 2
    assert not (root / "configs/data/banks_prod_v1.json").exists()
    assert not (root / "packages/data/src/jev_data/rules/phrasings_llm.py").exists()


# ------------------------------------------------------- phrasings merge hook


def test_load_extensions_is_none_without_a_generated_module():
    from jev_data.rules import phrasings
    assert phrasings._load_extensions() is None


def test_merge_extensions_appends_to_final_slot_and_preserves_indices():
    from jev_data.rules import phrasings
    kind, style = "comparison", "msa"
    final = phrasings.SLOTS[kind][-1]
    before = phrasings.VOCAB[kind][style][final]
    extension = "هل تنطبق القاعدة على المعطيات المعروضة أمامك الآن؟"
    old_count = phrasings.phrase_count(kind, style)
    old_words = [phrasings.phrase_slots(kind, style, index)
                 for index in range(old_count)]
    try:
        phrasings._merge_extensions({kind: {style: (extension,)}})
        after = phrasings.VOCAB[kind][style][final]
        assert after == before + (extension,)
        assert phrasings.phrase_count(kind, style) > old_count
        for index, words in enumerate(old_words):
            assert phrasings.phrase_slots(kind, style, index) == words, index
        assert phrasings.phrase_slots(kind, style, old_count)[final] == extension
        assert phrasings.phrase_slots(kind, style, old_count - 1)[final] == before[-1]
    finally:
        phrasings.VOCAB[kind][style][final] = before


@pytest.mark.parametrize("extensions, message", [
    ({"nope": {"msa": ("سؤال؟",)}}, "unknown kind"),
    ({"comparison": {"klingon": ("سؤال؟",)}}, "unknown style"),
    ({"comparison": {"msa": (42,)}}, "invalid choices"),
    ({"comparison": {"msa": ("",)}}, "invalid choices"),
])
def test_merge_extensions_fails_loudly_on_malformed_input(extensions, message):
    from jev_data.rules import phrasings
    kind, style = "comparison", "msa"
    final = phrasings.SLOTS[kind][-1]
    before = phrasings.VOCAB[kind][style][final]
    try:
        with pytest.raises(ValueError, match=message):
            phrasings._merge_extensions(extensions)
    finally:
        phrasings.VOCAB[kind][style][final] = before


def test_merge_extensions_rejects_duplicates_of_pinned_choices():
    from jev_data.rules import phrasings
    kind, style = "comparison", "msa"
    final = phrasings.SLOTS[kind][-1]
    before = phrasings.VOCAB[kind][style][final]
    try:
        with pytest.raises(ValueError, match="duplicate choices"):
            phrasings._merge_extensions({kind: {style: (before[0],)}})
    finally:
        phrasings.VOCAB[kind][style][final] = before


# --------------------------------------------------------------- LLM source


def test_llm_candidates_retries_across_distinct_layouts():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(500)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"candidates": [1, 2]}'}}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = BA.llm_candidates(client, "http://mock/v1", "datagen", "sys", "user",
                                   max_tokens=64, timeout=5.0)
    assert result == [1, 2]
    assert len(calls) == 2
    assert BA.LLM_REMINDER.strip() in calls[1].read().decode("utf-8")


def test_llm_candidates_parses_fenced_json_and_reports_total_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        if body["temperature"] == 0.0:
            content = '```json\n{"candidates": [{"ar": "كيان", "en": "entity"}]}\n```'
        else:
            content = "not json at all"
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content}}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = BA.llm_candidates(client, "http://mock/v1", "datagen", "sys", "user",
                                   max_tokens=64, timeout=5.0)
    assert result == [{"ar": "كيان", "en": "entity"}]

    def always_failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "still not json"}}]})

    with httpx.Client(transport=httpx.MockTransport(always_failing)) as client:
        with pytest.raises(RuntimeError, match="all layouts failed"):
            BA.llm_candidates(client, "http://mock/v1", "datagen", "sys", "user",
                              max_tokens=64, timeout=5.0)


def test_llm_batches_respects_round_budgets():
    calls: list[str] = []
    pool = ("عنبر", "ياقوت", "زمرد", "فيروز", "عقيق", "لؤلؤ",
            "مرجان", "صدف", "نيلي", "كهرمان", "جمان", "بهرت")
    counters = {"sys-a": 0, "sys-b": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        system = body["messages"][0]["content"]
        calls.append(system)
        batch = 3 if system == "sys-a" else 2
        start = counters[system]
        counters[system] += batch
        candidates = [{"ar": pool[start + i], "en": f"gem {start + i}"}
                      for i in range(batch)]
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({"candidates": candidates},
                                                           ensure_ascii=False)}}]})

    def spec(name, target, batches):
        return {"name": name, "prompt": lambda avoid: (f"sys-{name}", "user"),
                "existing_ratios": [], "existing": [], "target": target,
                "validate": BA._bilingual, "key": lambda e: e["ar"],
                "ratio": lambda e: e["ar"], "batches": batches}

    specs = [spec("a", 2, (1, 2)), spec("b", 5, (2, 3))]
    args = type("Args", (), {"base_url": "http://mock/v1", "model": "datagen",
                             "max_tokens": 512, "timeout": 5.0, "concurrency": 2})()
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        batches = BA._llm_batches(specs, args, client)
    assert calls.count("sys-a") == 1, "a is satisfied after its minimum round"
    assert calls.count("sys-b") == 3, "b keeps fetching while short, up to its maximum"
    assert len(batches["a"]) == 3 and len(batches["b"]) == 6
    kept_a, _ = BA.select(batches["a"], target=2, validate=BA._bilingual,
                          key_of=lambda e: e["ar"], ratio_of=lambda e: e["ar"],
                          existing_keys=[], existing_ratios=[])
    assert len(kept_a) == 2
