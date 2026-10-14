#!/usr/bin/env python3
"""Draft and commit the prod-v1 scenario banks (plan phase 1).

draft
    Ask an OpenAI-compatible datagen server (``--source llm``) or the
    deterministic seed tables (``--source seed``, for rehearsal without the
    server) for bank candidates, keep the structurally valid and mutually
    distant ones - pairwise SequenceMatcher ratio < 0.85 against every kept
    entry and every pinned voice-v1 string - and write a review package:

      <output>/banks-draft.json        the accepted draft (no timestamps)
      <output>/review/<section>.review.jsonl     kept entries + provenance
      <output>/review/<section>.rejected.jsonl   rejects with reasons
      <output>/summary.txt             yield table per section

    Sections: 14 routing domains, 25 personas, entities (comparison 60,
    policy_logic 40, budget_check 40, safety_gate 60, triage_rubric 40),
    fact catalogs (policy_logic 24, safety_gate 24), 40 triage items, 40
    temporal opening-hour profiles, 5 weekend conventions, value pools (200
    names, 60 cities, 40 addresses, 200 phones/dates/amounts, 60
    emails/references) and - for the LLM source only - append-safe
    question-phrasing extensions that lift every kind's final phrase slot to
    12 choices.  Phones, dates, amounts, emails and references are drafted
    procedurally in both modes (an LLM adds nothing to digit pools) and pass
    the same validators.

apply
    Re-validate a reviewed banks-draft.json and commit it as code constants:
    the generator-side JSON at configs/data/banks_prod_v1.json and, when the
    draft carries phrasing extensions, the generated module
    packages/data/src/jev_data/rules/phrasings_llm.py, which rules/phrasings.py
    merges into the frozen VOCAB bank at import.  Only the append-safe final
    slot of each kind ever grows, so the byte-exact v1 index 0 and every
    pinned phrase index stay stable.  Requires ``--approved-by``; the
    approval is recorded in the committed provenance.

Examples:
  uv run python tools/bank_author.py draft --source llm \\
      --base-url http://localhost:8010/v1 --model datagen \\
      --output runs/reports/banks/prod-v1
  uv run python tools/bank_author.py draft --source seed \\
      --output runs/reports/banks/prod-v1-seed
  uv run python tools/bank_author.py apply \\
      --draft runs/reports/banks/prod-v1-seed/banks-draft.json \\
      --approved-by <reviewer-id>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

_TOOLS = str(Path(__file__).resolve().parent)
_ROOT = str(Path(__file__).resolve().parents[1])
for _path in (_TOOLS, f"{_ROOT}/packages/data/src", f"{_ROOT}/packages/contracts/src"):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import httpx  # noqa: E402

import bank_seed_v1 as seed_bank  # noqa: E402
import worlds_voice_v1 as voice_v1  # noqa: E402
from jev_data.io import fields, integer, require, text  # noqa: E402
from jev_data.rules.phrasings import NOISE_PROFILES, SLOTS, VOCAB  # noqa: E402
from jev_data.rules.routing import ARABIC_FALLBACKS, FALLBACK  # noqa: E402

BANK_VERSION = "prod-v1"
RATIO_CEILING = 0.85
OPS = ("<", "<=", ">", ">=", "==", "!=")
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
STYLES = ("msa", "egyptian", "levantine", "gulf", "code_switch", "english")
FINAL_SLOTS = 12
ARABIC_CHARS = re.compile(r"[\u0600-\u06FF]")
LATIN_CHARS = re.compile(r"[A-Za-z]")
SLUG = re.compile(r"^[a-z][a-z0-9_]{1,31}$")
TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
PHONE_RE = re.compile(r"^05\d{8}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
AMOUNT_RE = re.compile(r"^\d{1,7}\.\d{2}$")
EMAIL_RE = re.compile(r"^[a-z0-9][a-z0-9._%+-]*@[a-z0-9.-]+\.[a-z]{2,}$")
REF_RE = re.compile(r"^REF-\d{6}$")
PERSONA_ID = re.compile(r"^[a-z0-9-]{3,24}$")


# ------------------------------------------------------------------ validators

def _short(value, limit: int, label: str) -> None:
    require(type(value) is str and bool(value.strip()) and "\n" not in value,
            f"{label} must be one nonempty line")
    require(len(value) <= limit, f"{label} is longer than {limit} characters")


def _bilingual(entry: dict, limit: int = 60) -> None:
    fields(entry, {"ar", "en"})
    _short(entry["ar"], limit, "ar")
    _short(entry["en"], limit, "en")
    require(ARABIC_CHARS.search(entry["ar"]), "ar must contain Arabic letters")
    require(LATIN_CHARS.search(entry["en"]), "en must contain Latin letters")


def _word(entry: dict) -> None:
    fields(entry, {"ar", "en"})
    for key in ("ar", "en"):
        _short(entry[key], 24, f"keyword {key}")
        require(" " not in entry[key].strip(), f"keyword {key} must be a single word")
    require(ARABIC_CHARS.search(entry["ar"]), "keyword ar must be Arabic")
    require(LATIN_CHARS.search(entry["en"]), "keyword en must be Latin")


def validate_domain(entry: dict) -> None:
    fields(entry, {"id", "label", "intents", "fillers"})
    require(type(entry["id"]) is str and SLUG.match(entry["id"]) and len(entry["id"]) <= 20,
            "bad domain id")
    _bilingual(entry["label"])
    intents = entry["intents"]
    require(type(intents) is list and len(intents) == 4, "a domain needs exactly 4 intents")
    ar_labels: list[str] = []
    en_labels: list[str] = []
    ar_words: list[str] = []
    en_words: list[str] = []
    for intent in intents:
        fields(intent, {"id", "label", "keywords"})
        require(type(intent["id"]) is str and SLUG.match(intent["id"]), "bad intent id")
        require(intent["id"] != "other", "intent id 'other' is reserved for the fallback")
        _bilingual(intent["label"])
        ar_labels.append(intent["label"]["ar"])
        en_labels.append(intent["label"]["en"])
        keywords = intent["keywords"]
        require(type(keywords) is list and len(keywords) == 3,
                "an intent needs exactly 3 keywords")
        for word in keywords:
            _word(word)
            ar_words.append(word["ar"])
            en_words.append(word["en"])
    require(len(set(ar_labels)) == 4 and len(set(en_labels)) == 4,
            "intent labels must be distinguishable inside the domain")
    require(not any(label in ARABIC_FALLBACKS for label in ar_labels)
            and FALLBACK["english"] not in en_labels,
            "an intent label collides with the routing fallback label")
    require(len(set(ar_words)) == 12 and len(set(en_words)) == 12,
            "keyword words must be pairwise distinct inside the domain")
    fillers = entry["fillers"]
    require(type(fillers) is list and len(fillers) == 3, "a domain needs exactly 3 fillers")
    for filler in fillers:
        _word(filler)
        require(filler["ar"] not in ar_words and filler["en"] not in en_words,
                "a filler collides with a keyword")


def validate_persona(entry: dict) -> None:
    fields(entry, {"id", "region", "formality", "urgency", "note_ar", "note_en"})
    require(type(entry["id"]) is str and PERSONA_ID.match(entry["id"]), "bad persona id")
    require(entry["region"] in seed_bank.REGIONS, "unknown persona region")
    require(entry["formality"] in seed_bank.FORMALITIES, "unknown persona formality")
    require(entry["urgency"] in seed_bank.URGENCIES, "unknown persona urgency")
    _short(entry["note_ar"], 120, "note_ar")
    _short(entry["note_en"], 120, "note_en")
    require(ARABIC_CHARS.search(entry["note_ar"]), "note_ar must contain Arabic")
    require(LATIN_CHARS.search(entry["note_en"]), "note_en must contain Latin")


def validate_policy_fact(entry: dict) -> None:
    fields(entry, {"fact", "ar", "en", "type", "op", "value"})
    require(type(entry["fact"]) is str and SLUG.match(entry["fact"]), "bad fact id")
    _short(entry["ar"], 60, "ar")
    _short(entry["en"], 60, "en")
    require(ARABIC_CHARS.search(entry["ar"]) and LATIN_CHARS.search(entry["en"]),
            "fact labels must be bilingual")
    if entry["type"] == "bool":
        require(entry["op"] in ("==", "!=") and type(entry["value"]) is bool,
                "bool facts need ==/!= and a boolean value")
    elif entry["type"] == "number":
        require(entry["op"] in OPS, "unknown requirement operator")
        integer(entry["value"], -(10 ** 9), 10 ** 9)
    else:
        require(False, "fact type must be number or bool")


def validate_safety_fact(entry: dict) -> None:
    fields(entry, {"fact", "ar", "en", "type"})
    require(type(entry["fact"]) is str and SLUG.match(entry["fact"]), "bad fact id")
    _short(entry["ar"], 60, "ar")
    _short(entry["en"], 60, "en")
    require(ARABIC_CHARS.search(entry["ar"]) and LATIN_CHARS.search(entry["en"]),
            "fact labels must be bilingual")
    require(entry["type"] in ("number", "bool"), "fact type must be number or bool")


def validate_triage_item(entry: dict) -> None:
    fields(entry, {"id", "ar", "en"})
    require(type(entry["id"]) is str and SLUG.match(entry["id"]), "bad item id")
    _short(entry["ar"], 60, "ar")
    _short(entry["en"], 60, "en")
    require(ARABIC_CHARS.search(entry["ar"]), "ar must contain Arabic letters")
    require(LATIN_CHARS.search(entry["en"]), "en must contain Latin letters")


def validate_profile(entry: dict) -> None:
    fields(entry, {"opens", "closes"})
    require(type(entry["opens"]) is str and TIME_RE.match(entry["opens"]), "opens must be HH:MM")
    require(type(entry["closes"]) is str and TIME_RE.match(entry["closes"]), "closes must be HH:MM")
    require(entry["opens"] != entry["closes"], "opens and closes must differ")


def validate_weekend(entry: dict) -> None:
    fields(entry, {"weekend"})
    days = entry["weekend"]
    require(type(days) is list and 1 <= len(days) <= 2, "a weekend is 1 or 2 days")
    require(all(day in WEEKDAYS for day in days), "unknown weekday name")
    require(len(set(days)) == len(days), "duplicate weekday")


def validate_phone(entry: dict) -> None:
    fields(entry, {"phone"})
    require(type(entry["phone"]) is str and PHONE_RE.match(entry["phone"]), "bad phone")


def validate_date(entry: dict) -> None:
    fields(entry, {"date"})
    value = entry["date"]
    require(type(value) is str and DATE_RE.match(value), "bad ISO date")
    import datetime as _dt
    try:
        _dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("unreachable calendar date") from exc


def validate_amount(entry: dict) -> None:
    fields(entry, {"amount"})
    require(type(entry["amount"]) is str and AMOUNT_RE.match(entry["amount"]), "bad amount")


def validate_email(entry: dict) -> None:
    fields(entry, {"email"})
    require(type(entry["email"]) is str and EMAIL_RE.match(entry["email"]), "bad email")


def validate_reference(entry: dict) -> None:
    fields(entry, {"reference"})
    require(type(entry["reference"]) is str and REF_RE.match(entry["reference"]), "bad reference")


def question_validator(style: str, existing: tuple[str, ...]):
    """Validator for question-phrasing candidates of one (kind, style)."""
    mark = "?" if style == "english" else "؟"

    def validate(entry) -> None:
        require(type(entry) is str and bool(entry.strip()) and "\n" not in entry,
                "a phrasing must be one line of text")
        require(len(entry) <= 160, "a phrasing is at most 160 characters")
        require(entry.endswith(mark), f"a phrasing must end with {mark!r}")
        require(entry not in existing, "the phrasing duplicates a bank choice")

    return validate


# ------------------------------------------------- pinned voice-v1 exclusions

def _slot_values(field: str) -> list[str]:
    values = set()
    for form, claim, _ in voice_v1.SLOT_SCENARIOS:
        for name, value in form:
            if name == field and value:
                values.add(value)
        if claim[0] == field and claim[1]:
            values.add(claim[1])
    return sorted(values)


PINNED = {
    "domain_ids": sorted(voice_v1.DOMAINS),
    "domain_labels": [voice_v1.DOMAINS[name][0] for name in voice_v1.DOMAINS],
    "entities": {
        "comparison": [ar for ar, _ in voice_v1.COMPARISON_ENTITIES],
        "policy_logic": sorted({scenario[0] for scenario in voice_v1.POLICY_SCENARIOS}),
        "budget_check": [ar for ar, _ in voice_v1.BUDGET_ENTITIES],
        "safety_gate": sorted({scenario[0] for scenario in voice_v1.SAFETY_SCENARIOS}),
        "triage_rubric": [ar for ar, _ in voice_v1.TRIAGE_ENTITIES],
    },
    "fact_names": {
        "policy_logic": sorted({clause[0] for scenario in voice_v1.POLICY_SCENARIOS
                                for clause in scenario[3]}),
        "safety_gate": sorted(voice_v1.SAFETY_FACT_LABELS),
    },
    "fact_labels": {
        "policy_logic": sorted({clause[1] for scenario in voice_v1.POLICY_SCENARIOS
                                for clause in scenario[3]}),
        "safety_gate": sorted({label[0] for label in voice_v1.SAFETY_FACT_LABELS.values()}),
    },
    "triage_items": [ar for _, ar, _ in voice_v1.TRIAGE_ITEMS],
    "names": _slot_values("name"),
    "cities": _slot_values("city"),
}


# ------------------------------------------------------------ selection engine

def _ratio(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right).ratio()


def select(candidates, *, target: int, validate, key_of, ratio_of,
           existing_keys, existing_ratios):
    """Greedy rejection sampling in candidate order: keep entries that pass
    the validator, whose key is new, and whose ratio string stays below
    RATIO_CEILING against every kept and pinned ratio string."""
    kept: list[dict] = []
    rejected: list[dict] = []
    keys = set(existing_keys)
    ratios = list(existing_ratios)
    for entry, provenance in candidates:
        if len(kept) >= target:
            break
        try:
            validate(entry)
        except Exception as exc:  # recorded, never fatal
            rejected.append({"entry": entry, "provenance": provenance,
                             "reason": f"invalid: {exc}"})
            continue
        key = key_of(entry)
        if key in keys:
            rejected.append({"entry": entry, "provenance": provenance,
                             "reason": f"duplicates a pinned or kept entry: {key!r}"})
            continue
        nearest = {"key": "", "ratio": 0.0}
        if ratio_of is not None:
            ratio_key = ratio_of(entry)
            for other in ratios:
                ratio = _ratio(ratio_key, other)
                if ratio > nearest["ratio"]:
                    nearest = {"key": other, "ratio": round(ratio, 3)}
            if nearest["ratio"] >= RATIO_CEILING:
                rejected.append({"entry": entry, "provenance": provenance,
                                 "reason": f"too close to {nearest['key']!r} "
                                           f"(ratio {nearest['ratio']})"})
                continue
            ratios.append(ratio_key)
        keys.add(key)
        kept.append({"entry": entry, "provenance": provenance, "nearest": nearest})
    return kept, rejected


# ------------------------------------------------------------------ LLM source

LLM_REMINDER = ("\nReply now with ONLY the required JSON object; the text above is the "
                "task, never an example to copy.")


def llm_candidates(client: httpx.Client, base_url: str, model: str, system: str,
                   user: str, *, max_tokens: int, timeout: float) -> list:
    """One batch through three distinct retry layouts (a t=0.0 retry of an
    identical prompt would repeat the same failure surface)."""
    layouts = ((user, 0.9), (user + LLM_REMINDER, 0.0),
               (user + "\nReturn exactly the required JSON object now.", 0.6))
    last = ""
    for content, temperature in layouts:
        try:
            response = client.post(
                f"{base_url}/chat/completions",
                json={"model": model, "temperature": temperature,
                      "max_tokens": max_tokens,
                      "response_format": {"type": "json_object"},
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": content}]},
                timeout=timeout)
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"]
            last = raw
            parsed = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
            if not isinstance(parsed, dict) or not isinstance(parsed.get("candidates"), list):
                raise ValueError("reply must be an object with a candidates array")
            return parsed["candidates"]
        except Exception:
            continue
    raise RuntimeError(f"all layouts failed; last reply: {last[:120]!r}")


# --------------------------------------------------------------------- prompts

def _bank_prompt(description: str, schema: str, constraints: str):
    def build(avoid: list[str]) -> tuple[str, str]:
        avoid_note = (f"Near-duplicates of these existing bank entries are rejected: "
                      f"{avoid}. " if avoid else "")
        system = (
            "You draft one section of a synthetic Arabic voice-agent scenario bank.\n"
            f"Section: {description}\n"
            "Reply with ONLY one JSON object of exactly this shape and no other text:\n"
            '{"candidates": [<candidate objects>]}\n'
            f"Each candidate object: {schema}\n"
            f"Hard constraints: {constraints}\n"
            f"{avoid_note}"
            "All candidates must be mutually distinct and diverse. Write natural, "
            "idiomatic Arabic wherever Arabic is required. No commentary, no markdown "
            "fences, no keys beyond the schema.")
        return system, "Draft the candidates for this bank section now."

    return build


_ENTITY_CONTEXT = {
    "comparison": "quantities a service compares against a threshold (documents, balances, limits, rates)",
    "policy_logic": "requests a policy judges (refunds, registrations, coverage, guarantees)",
    "budget_check": "spend objects a budget check approves (orders, invoices, installment plans)",
    "safety_gate": "sensitive requests a safety gate screens",
    "triage_rubric": "tickets and reports a triage rubric scores",
}

DOMAIN_PROMPT = _bank_prompt(
    "one customer-service routing domain with four distinct intents.",
    '{"id": "<snake_case domain id>", "label": {"ar": "<Arabic domain name>", '
    '"en": "<English domain name>"}, "intents": [exactly 4 objects '
    '{"id": "<snake_case intent id>", "label": {"ar": "<Arabic>", "en": "<English>"}, '
    '"keywords": [exactly 3 objects {"ar": "<single Arabic word>", '
    '"en": "<single English word>"}]}], "fillers": [exactly 3 objects '
    '{"ar": "<single Arabic word>", "en": "<single English word>"}]}',
    "exactly 4 intents with exactly 3 single-word keywords each and exactly 3 filler words; "
    "keywords and fillers contain no spaces; the Arabic keyword words are pairwise distinct "
    "inside the domain and so are the English ones; fillers never repeat a keyword; intent "
    "labels differ from each other in both languages.")

PERSONA_PROMPT = _bank_prompt(
    "caller personas that condition how scenarios are composed.",
    '{"id": "<persona-XX>", "region": "egypt|levant|gulf|maghreb|neutral", '
    '"formality": "casual|neutral|formal", "urgency": "low|normal|high", '
    '"note_ar": "<one short Arabic sentence>", "note_en": "<English gloss>"}',
    "each (region, formality, urgency) combination at most once; the notes describe a "
    "plausible caller of a voice agent (home city, tone, urgency) without naming products.")


def _entity_prompt(kind: str):
    return _bank_prompt(
        f"bilingual noun phrases naming {kind} material: {_ENTITY_CONTEXT[kind]}."
        if kind in _ENTITY_CONTEXT else f"bilingual noun phrases for {kind}.",
        '{"ar": "<Arabic noun phrase>", "en": "<faithful English translation>"}',
        "1 to 4 words per language; the English phrase translates the Arabic one.")


POLICY_FACT_PROMPT = _bank_prompt(
    "typed facts with the single requirement a service policy can place on them.",
    '{"fact": "<snake_case fact id>", "ar": "<Arabic fact label>", '
    '"en": "<English fact label>", "type": "number|bool", '
    '"op": "<|<=|>|>=|==|!=", "value": <integer for number, true/false for bool>}',
    "number values between 1 and 100000; bool values only with == or !=; every requirement "
    "is plausible for a customer-service request.")

SAFETY_FACT_PROMPT = _bank_prompt(
    "typed facts a safety gate can check before acting on a customer request.",
    '{"fact": "<snake_case fact id>", "ar": "<Arabic fact label>", '
    '"en": "<English fact label>", "type": "number|bool"}',
    "fact ids are stable identifiers; labels read naturally inside a sentence of stated facts.")

TRIAGE_ITEM_PROMPT = _bank_prompt(
    "operational conditions that are simply met or not met when a support rubric scores a ticket.",
    '{"id": "<snake_case id>", "ar": "<Arabic condition phrase>", "en": "<English translation>"}',
    "each phrase names exactly one condition and contains no numbers.")

PROFILE_PROMPT = _bank_prompt(
    "opening hours of walk-in services.",
    '{"opens": "<HH:MM>", "closes": "<HH:MM>"}',
    "24-hour clock times; opens always differs from closes; opens may be later than closes "
    "(overnight service); the hours look like real services.")

WEEKEND_PROMPT = _bank_prompt(
    "weekend conventions of working calendars.",
    '{"weekend": ["<weekday>", "<weekday>"]}',
    "one or two distinct weekday names spelled monday..sunday in lowercase English.")

NAME_PROMPT = _bank_prompt(
    "personal names for identity form fields.",
    '{"ar": "<Arabic full name: given name + family name>", "en": "<transliteration>"}',
    "realistic Arabic personal names; vary given names and family names across candidates.")

CITY_PROMPT = _bank_prompt(
    "city names for address form fields.",
    '{"ar": "<Arabic city name>", "en": "<English city name>"}',
    "real cities of the Arab world.")

ADDRESS_PROMPT = _bank_prompt(
    "street addresses for form fields.",
    '{"ar": "<street and city in Arabic>", "en": "<street and city in English>"}',
    "exactly one street and one city per address.")

_STYLE_TEXT = {
    "msa": "Modern Standard Arabic (فصحى)",
    "egyptian": "Egyptian colloquial Arabic",
    "levantine": "Levantine (Shami) Arabic",
    "gulf": "Gulf (Khaleeji) Arabic",
    "code_switch": "Modern Standard Arabic mixed with common English tech terms",
    "english": "plain English",
}
_QUESTION_TASK = {
    "comparison": "decide whether a stated quantity satisfies a stated threshold policy",
    "temporal_window": "decide whether the first reply to a support ticket arrived inside "
                       "the agreed response window",
    "policy_logic": "decide whether a request satisfies an all/any policy over stated facts",
    "budget_check": "decide whether an installment plan covers the full stated price",
    "intent_routing": "pick which listed intent matches the customer request",
    "safety_gate": "pick allow, block or escalate for a sensitive request",
    "triage_rubric": "pick the severity level earned by the met conditions",
    "readback_match": "decide whether a read-back value matches the expected value after "
                      "unit conversion",
    "slot_assertion": "decide whether a claimed field value matches the form content",
}


def _phrasing_prompt(kind: str, style: str, existing: tuple[str, ...]):
    mark = "?" if style == "english" else "؟"

    def build(_avoid: list[str]) -> tuple[str, str]:
        system = (
            f"You write question instructions for a voice-agent decision task rendered in "
            f"{_STYLE_TEXT[style]}.\n"
            f"The task: {_QUESTION_TASK[kind]}. The reader has already seen the full scenario "
            "text in the same style, so the question stays generic: it never mentions option "
            "ids, concrete values, or the correct answer, and it always asks the reader to "
            "rely only on the stated information.\n"
            "Reply with ONLY one JSON object of exactly this shape and no other text:\n"
            '{"candidates": ["<question sentence>", ...]}\n'
            f"Hard constraints: every candidate is ONE sentence of 6 to 25 words ending with "
            f'"{mark}"; candidates are mutually distinct; none is a near-duplicate of these '
            f"existing phrasings: {list(existing)}.")
        return system, f"Draft the new question phrasings for the {kind} task now."

    return build


# --------------------------------------------------------------- section specs

def section_specs() -> list[dict]:
    """One spec per bank section; draft_key is the placement path inside
    banks-draft.json.  procedural=True sections come from the seed module in
    both modes; llm_only=True sections never run in seed mode."""
    specs: list[dict] = []

    def add(name, draft_key, target, *, validate, key, ratio=None, existing=(),
            existing_ratios=None, batch=30, batches=(3, 5), seed=None, prompt=None,
            procedural=False, llm_only=False):
        specs.append({
            "name": name, "draft_key": draft_key, "target": target,
            "validate": validate, "key": key, "ratio": ratio,
            "existing": list(existing),
            "existing_ratios": list(existing_ratios if existing_ratios is not None
                                   else existing),
            "batch": batch, "batches": batches, "seed": seed, "prompt": prompt,
            "procedural": procedural, "llm_only": llm_only,
        })

    add("domains", ("domains",), 14, validate=validate_domain,
        key=lambda e: e["id"], ratio=lambda e: e["label"]["ar"],
        existing=PINNED["domain_ids"], existing_ratios=PINNED["domain_labels"],
        batch=7, batches=(3, 6), seed=seed_bank.domains, prompt=DOMAIN_PROMPT)
    add("personas", ("personas",), 25, validate=validate_persona,
        key=lambda e: f"{e['region']}|{e['formality']}|{e['urgency']}",
        batch=12, batches=(3, 4), seed=seed_bank.personas, prompt=PERSONA_PROMPT)
    for kind, target, composer in (
            ("comparison", 60, seed_bank.comparison_entities),
            ("policy_logic", 40, seed_bank.policy_entities),
            ("budget_check", 40, seed_bank.budget_entities),
            ("safety_gate", 60, seed_bank.safety_entities),
            ("triage_rubric", 40, seed_bank.triage_entities)):
        add(f"entities.{kind}", ("entities", kind), target, validate=_bilingual,
            key=lambda e, _k=kind: e["ar"], ratio=lambda e, _k=kind: e["ar"],
            existing=PINNED["entities"][kind], batch=30, batches=(3, 5),
            seed=composer, prompt=_entity_prompt(kind))
    add("facts.policy_logic", ("fact_catalogs", "policy_logic"), 24,
        validate=validate_policy_fact, key=lambda e: e["fact"], ratio=lambda e: e["ar"],
        existing=PINNED["fact_names"]["policy_logic"],
        existing_ratios=PINNED["fact_labels"]["policy_logic"],
        batch=16, batches=(3, 4), seed=seed_bank.policy_facts, prompt=POLICY_FACT_PROMPT)
    add("facts.safety_gate", ("fact_catalogs", "safety_gate"), 24,
        validate=validate_safety_fact, key=lambda e: e["fact"], ratio=lambda e: e["ar"],
        existing=PINNED["fact_names"]["safety_gate"],
        existing_ratios=PINNED["fact_labels"]["safety_gate"],
        batch=16, batches=(3, 4), seed=seed_bank.safety_facts, prompt=SAFETY_FACT_PROMPT)
    add("triage_items", ("triage_items",), 40, validate=validate_triage_item,
        key=lambda e: e["ar"], ratio=lambda e: e["ar"], existing=PINNED["triage_items"],
        batch=24, batches=(3, 4), seed=seed_bank.triage_items, prompt=TRIAGE_ITEM_PROMPT)
    add("temporal.profiles", ("temporal", "profiles"), 40, validate=validate_profile,
        key=lambda e: f"{e['opens']}|{e['closes']}", ratio=None,
        batch=16, batches=(3, 4), seed=seed_bank.temporal_profiles, prompt=PROFILE_PROMPT)
    add("temporal.weekends", ("temporal", "weekends"), 5, validate=validate_weekend,
        key=lambda e: ",".join(sorted(e["weekend"])), ratio=None,
        batch=5, batches=(3, 3), seed=seed_bank.weekend_conventions, prompt=WEEKEND_PROMPT)
    add("values.names", ("values", "names"), 200,
        validate=lambda e: _bilingual(e, 40),
        key=lambda e: e["ar"], ratio=lambda e: e["ar"], existing=PINNED["names"],
        batch=70, batches=(3, 5), seed=seed_bank.names, prompt=NAME_PROMPT)
    add("values.cities", ("values", "cities"), 60,
        validate=lambda e: _bilingual(e, 40),
        key=lambda e: e["ar"], ratio=lambda e: e["ar"], existing=PINNED["cities"],
        batch=30, batches=(3, 4), seed=seed_bank.cities, prompt=CITY_PROMPT)
    add("values.addresses", ("values", "addresses"), 40,
        validate=lambda e: _bilingual(e, 80),
        key=lambda e: e["ar"], ratio=None,
        batch=20, batches=(3, 4), seed=seed_bank.addresses, prompt=ADDRESS_PROMPT)
    add("values.phones", ("values", "phones"), 200, validate=validate_phone,
        key=lambda e: e["phone"], ratio=None, seed=seed_bank.phones, procedural=True)
    add("values.dates", ("values", "dates"), 200, validate=validate_date,
        key=lambda e: e["date"], ratio=None, seed=seed_bank.dates, procedural=True)
    add("values.amounts", ("values", "amounts"), 200, validate=validate_amount,
        key=lambda e: e["amount"], ratio=None, seed=seed_bank.amounts, procedural=True)
    add("values.emails", ("values", "emails"), 60, validate=validate_email,
        key=lambda e: e["email"], ratio=None, seed=seed_bank.emails, procedural=True)
    add("values.references", ("values", "references"), 60, validate=validate_reference,
        key=lambda e: e["reference"], ratio=None, seed=seed_bank.references, procedural=True)
    for kind in SLOTS:
        final = SLOTS[kind][-1]
        for style in STYLES:
            existing = tuple(VOCAB[kind][style][final])
            target = FINAL_SLOTS - len(existing)
            if target <= 0:
                continue
            add(f"phrasings.{kind}.{style}", ("phrasings", kind, style), target,
                validate=question_validator(style, existing),
                key=lambda e: e, ratio=lambda e: e, existing=existing,
                batch=FINAL_SLOTS, batches=(3, 4), seed=None, llm_only=True,
                prompt=_phrasing_prompt(kind, style, existing))
    return specs


# ---------------------------------------------------------------------- draft

def _llm_batches(specs: list[dict], args, client: httpx.Client) -> dict[str, list]:
    """Fetch LLM candidate batches: every spec runs its minimum batch count;
    short specs keep fetching up to their maximum, round by round."""
    candidates: dict[str, list] = {spec["name"]: [] for spec in specs}
    failures: dict[str, int] = {spec["name"]: 0 for spec in specs}

    def fetch(spec: dict, batch: int):
        system, user = spec["prompt"](spec["existing_ratios"] or spec["existing"])
        entries = llm_candidates(client, args.base_url, args.model, system, user,
                                 max_tokens=args.max_tokens, timeout=args.timeout)
        return spec["name"], batch, entries

    for round_index in range(max(spec["batches"][1] for spec in specs)):
        pending = []
        for spec in specs:
            minimum, maximum = spec["batches"]
            if round_index >= maximum:
                continue
            if round_index >= minimum:
                kept, _ = select(candidates[spec["name"]], target=spec["target"],
                                 validate=spec["validate"], key_of=spec["key"],
                                 ratio_of=spec["ratio"], existing_keys=spec["existing"],
                                 existing_ratios=spec["existing_ratios"])
                if len(kept) >= spec["target"]:
                    continue
            pending.append(spec)
        if not pending:
            break
        with ThreadPoolExecutor(max_workers=max(args.concurrency, 1)) as pool:
            futures = [pool.submit(fetch, spec, round_index) for spec in pending]
            for future, spec in zip(futures, pending):
                try:
                    name, batch, entries = future.result()
                except RuntimeError as exc:
                    failures[spec["name"]] += 1
                    print(f"  {spec['name']}: batch {round_index} failed ({exc})",
                          file=sys.stderr)
                    continue
                candidates[name].extend(
                    (entry, {"source": "llm", "batch": batch}) for entry in entries)
    total_failures = sum(failures.values())
    if total_failures:
        print(f"note: {total_failures} LLM batches failed all retry layouts",
              file=sys.stderr)
    return candidates


def _place(node: dict, key_path: tuple, entries: list) -> None:
    for part in key_path[:-1]:
        node = node.setdefault(part, {})
    node[key_path[-1]] = entries


def run_draft(args) -> int:
    if args.concurrency < 1 or args.max_tokens < 256 or args.timeout <= 0:
        print("bank_author: --concurrency >= 1, --max-tokens >= 256, --timeout > 0",
              file=sys.stderr)
        return 2
    all_specs = section_specs()
    known = {spec["name"] for spec in all_specs}
    if args.sections:
        unknown = [name for name in args.sections if name not in known]
        if unknown:
            print(f"bank_author: unknown sections {unknown}", file=sys.stderr)
            return 2
    specs = [spec for spec in all_specs
             if not args.sections or spec["name"] in args.sections]
    output = Path(args.output)
    (output / "review").mkdir(parents=True, exist_ok=True)

    if args.source == "llm":
        live = [spec for spec in specs if not spec["procedural"]]
        with httpx.Client() as client:
            batches = _llm_batches(live, args, client)
    else:
        batches = {}

    draft: dict = {"schema_version": 1, "bank_version": BANK_VERSION, "provenance": {
        "source": args.source,
        "model": args.model if args.source == "llm" else None,
        "base_url": args.base_url if args.source == "llm" else None,
        "seed": seed_bank.SEED,
        "tool": "tools/bank_author.py",
        "noise_profiles": list(NOISE_PROFILES),
    }}
    summary: list[str] = [f"source={args.source} model={args.model} seed={seed_bank.SEED}",
                          f"{'section':28} {'target':>6} {'candidates':>10} "
                          f"{'kept':>5} {'rejected':>8}  note"]
    short: list[str] = []

    for spec in specs:
        if spec["llm_only"] and args.source == "seed":
            note = "skipped (LLM-only; the bank already offers >= 12 phrasings)"
            summary.append(f"{spec['name']:28} {spec['target']:>6} {0:>10} {0:>5} "
                           f"{0:>8}  {note}")
            continue
        if spec["procedural"] or args.source == "seed":
            candidates = [(entry, {"source": "seed", "batch": 0})
                          for entry in spec["seed"]()]
        else:
            candidates = batches.get(spec["name"], [])
        kept, rejected = select(
            candidates, target=spec["target"], validate=spec["validate"],
            key_of=spec["key"], ratio_of=spec["ratio"],
            existing_keys=spec["existing"], existing_ratios=spec["existing_ratios"])
        if len(kept) < spec["target"]:
            short.append(spec["name"])
        _place(draft, spec["draft_key"], [record["entry"] for record in kept])
        (output / f"review/{spec['name']}.review.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in kept),
            encoding="utf-8")
        (output / f"review/{spec['name']}.rejected.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in rejected),
            encoding="utf-8")
        summary.append(f"{spec['name']:28} {spec['target']:>6} {len(candidates):>10} "
                       f"{len(kept):>5} {len(rejected):>8}"
                       f"{'  SHORT' if len(kept) < spec['target'] else ''}")
        print(f"{spec['name']}: kept {len(kept)}/{spec['target']} "
              f"({len(rejected)} rejected, {len(candidates)} candidates)")

    (output / "banks-draft.json").write_text(
        json.dumps(draft, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (output / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"wrote {output / 'banks-draft.json'} and {output / 'summary.txt'}")
    if short:
        message = (f"short sections (below target): {short} - rerun with more batches or "
                   "review the rejected files")
        if not args.allow_short:
            print(f"bank_author: ABORT: {message}", file=sys.stderr)
            return 2
        print(f"bank_author: {message} (--allow-short set)", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------- apply

def _iter_draft(draft: dict):
    """Yield (section_name, entries) for every section present in a draft."""
    yield "domains", draft.get("domains", [])
    yield "personas", draft.get("personas", [])
    for kind in ("comparison", "policy_logic", "budget_check", "safety_gate",
                 "triage_rubric"):
        yield f"entities.{kind}", draft.get("entities", {}).get(kind, [])
    for kind in ("policy_logic", "safety_gate"):
        yield f"facts.{kind}", draft.get("fact_catalogs", {}).get(kind, [])
    yield "triage_items", draft.get("triage_items", [])
    yield "temporal.profiles", draft.get("temporal", {}).get("profiles", [])
    yield "temporal.weekends", draft.get("temporal", {}).get("weekends", [])
    for pool in ("names", "cities", "addresses", "phones", "dates", "amounts",
                 "emails", "references"):
        yield f"values.{pool}", draft.get("values", {}).get(pool, [])
    for kind, styles in draft.get("phrasings", {}).items():
        for style, choices in styles.items():
            yield f"phrasings.{kind}.{style}", choices


def _phrasings_module(extensions: dict, provenance: dict) -> str:
    lines = [
        '"""Append-safe question-phrasing extensions for rules/phrasings.py.',
        "",
        "Generated by tools/bank_author.py apply; do not edit by hand.",
        f"Provenance: {json.dumps(provenance, ensure_ascii=False)}",
        "",
        "Every tuple appends to the FINAL slot (the last question slot) of its",
        "kind - the one append-safe position in the frozen bank - so every",
        "existing phrase index, including the byte-exact v1 index 0, keeps its",
        "exact meaning. rules/phrasings.py merges and validates this module at",
        'import time (duplicate choices fail loudly).',
        '"""',
        "",
        "QUESTION_EXTENSIONS = {",
    ]
    for kind in sorted(extensions):
        lines.append(f"    {kind!r}: {{")
        for style in sorted(extensions[kind]):
            lines.append(f"        {style!r}: (")
            for choice in extensions[kind][style]:
                lines.append(f"            {choice!r},")
            lines.append("        ),")
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _verify_phrasings(root: Path, expected: dict) -> None:
    """Fresh-interpreter check that the merged bank is still valid and that
    extended final slots reached exactly 12 choices."""
    code = (
        "import json\n"
        "from jev_data.rules.phrasings import SLOTS, VOCAB, phrase_count\n"
        "expected = json.loads(sys_expected)\n"
        "styles = ('msa', 'egyptian', 'levantine', 'gulf', 'code_switch', 'english')\n"
        "for kind in SLOTS:\n"
        "    for style in styles:\n"
        "        assert phrase_count(kind, style) >= 12, (kind, style)\n"
        "for kind, styles in expected.items():\n"
        "    for style in styles:\n"
        "        final = SLOTS[kind][-1]\n"
        "        assert len(VOCAB[kind][style][final]) == 12, (kind, style)\n"
        "print('phrasings bank OK after extensions')\n"
    ).replace("sys_expected", repr(json.dumps(expected)))
    env = {**os.environ,
           "PYTHONPATH": f"{root / 'packages/contracts/src'}:"
                         f"{root / 'packages/data/src'}"}
    subprocess.run([sys.executable, "-c", code], env=env, check=True, cwd=str(root))


def run_apply(args) -> int:
    root = Path(args.root).resolve()
    draft_path = Path(args.draft)
    try:
        draft = json.loads(draft_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"bank_author: cannot read draft {draft_path}: {exc}", file=sys.stderr)
        return 2
    if draft.get("bank_version") != BANK_VERSION:
        print(f"bank_author: draft bank_version {draft.get('bank_version')!r} "
              f"!= {BANK_VERSION!r}", file=sys.stderr)
        return 2

    specs = {spec["name"]: spec for spec in section_specs()}
    issues: list[str] = []
    present = set()
    for name, entries in _iter_draft(draft):
        spec = specs.get(name)
        if spec is None:
            issues.append(f"unknown section {name}")
            continue
        present.add(name)
        if not entries:
            if spec["llm_only"]:
                continue
            issues.append(f"{name}: empty")
            continue
        if len(entries) != spec["target"]:
            if not args.allow_short or len(entries) < 1:
                issues.append(f"{name}: {len(entries)} entries, target {spec['target']}")
            else:
                print(f"bank_author: note: {name} short ({len(entries)} of "
                      f"{spec['target']}, --allow-short)", file=sys.stderr)
        kept, rejected = select(
            [(entry, {"source": "apply", "batch": 0}) for entry in entries],
            target=len(entries), validate=spec["validate"], key_of=spec["key"],
            ratio_of=spec["ratio"], existing_keys=spec["existing"],
            existing_ratios=spec["existing_ratios"])
        if rejected:
            issues.append(f"{name}: {len(rejected)} entries fail re-validation "
                          f"({rejected[0]['reason']})")
    for spec in specs.values():
        if spec["llm_only"]:
            continue
        if spec["name"] not in present:
            issues.append(f"{spec['name']}: missing from draft")
    if issues:
        for issue in issues:
            print(f"bank_author: {issue}", file=sys.stderr)
        return 2

    provenance = dict(draft.get("provenance", {}))
    provenance["approved_by"] = args.approved_by
    provenance["applied_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    banks = {key: value for key, value in draft.items() if key != "phrasings"}
    banks["provenance"] = provenance
    banks_path = root / "configs/data/banks_prod_v1.json"
    banks_path.parent.mkdir(parents=True, exist_ok=True)
    banks_path.write_text(json.dumps(banks, ensure_ascii=False, indent=1) + "\n",
                          encoding="utf-8")
    print(f"wrote {banks_path}")

    extensions = draft.get("phrasings", {})
    module_path = root / "packages/data/src/jev_data/rules/phrasings_llm.py"
    if extensions:
        expected = {kind: sorted(styles) for kind, styles in extensions.items()}
        module_path.parent.mkdir(parents=True, exist_ok=True)
        module_path.write_text(_phrasings_module(extensions, provenance),
                               encoding="utf-8")
        print(f"wrote {module_path}")
        if (root / "packages/data/src/jev_data/rules/phrasings.py").exists():
            _verify_phrasings(root, expected)
    elif module_path.exists():
        module_path.unlink()
        print(f"removed stale {module_path} (draft carries no extensions)")

    print("banks committed; review with git diff, then regenerate worlds")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    draft_p = sub.add_parser("draft", help="draft bank candidates for review")
    draft_p.add_argument("--source", choices=("llm", "seed"), default="llm",
                         help="candidate source: datagen server or seed tables")
    draft_p.add_argument("--base-url", default="http://localhost:8010/v1")
    draft_p.add_argument("--model", default="datagen")
    draft_p.add_argument("--output", default="runs/reports/banks/prod-v1")
    draft_p.add_argument("--concurrency", type=int, default=8,
                         help="parallel LLM calls (default 8)")
    draft_p.add_argument("--max-tokens", type=int, default=4096)
    draft_p.add_argument("--timeout", type=float, default=120.0)
    draft_p.add_argument("--sections", nargs="*", default=None,
                         help="restrict drafting to these section names")
    draft_p.add_argument("--allow-short", action="store_true",
                         help="finish the draft even when sections are short")
    draft_p.set_defaults(func=run_draft)

    apply_p = sub.add_parser("apply", help="commit a reviewed draft as code constants")
    apply_p.add_argument("--draft", required=True, help="banks-draft.json to apply")
    apply_p.add_argument("--approved-by", required=True,
                         help="reviewer id recorded in the committed provenance")
    apply_p.add_argument("--root", default=".", help="jev root (default .)")
    apply_p.add_argument("--allow-short", action="store_true")
    apply_p.set_defaults(func=run_apply)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
