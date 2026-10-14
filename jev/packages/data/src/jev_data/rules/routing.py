"""Deterministic intent routing over an explicit ordered domain ontology.

Semantics are frozen under rule version intent-routing-v1. Each intent lists
keyword tokens; the customer request is a bag of tokens. The winning intent is
the one matching the most distinct request tokens; ties go to the intent listed
first, and zero matches route to the explicit fallback option 'other'. The
independent solver builds a token-to-intent inverted index and rank-sorts
instead of scanning intents directly.
"""

from __future__ import annotations

from ..io import fields, require, strings, text, version
from ._shared import (
    attributes,
    digits,
    noised,
    style_of,
    surface_words,
    variant_of,
)

KIND = "intent_routing"
RULE_VERSION = "intent-routing-v1"
TASK = "intent_routing"

MAX_INTENTS = 8
MAX_TOKENS = 12
FALLBACK = {
    "msa": "لا يناسب أي نية",
    "egyptian": "مش نية من دول",
    "levantine": "ما بيناسب ولا نية",
    "gulf": "ما يناسب أي نية",
    "code_switch": "لا يناسب أي نية",
    "english": "None of the intents",
}
ARABIC_FALLBACKS = frozenset(FALLBACK[style] for style in
                             ("msa", "egyptian", "levantine", "gulf", "code_switch"))
RULE_SENTENCE = {
    "msa": "القاعدة: تُختار النية التي تطابق أكبر عدد من كلمات الطلب، وتُحسب كل كلمة مرة "
           "واحدة، وعند التعادل تُرجَّح الأسبق في القائمة، وإذا لم تتطابق أي كلمة تُختار:",
    "egyptian": "القاعدة: تُختار النية اللي تطابق أكبر عدد من كلمات الطلب، وكل كلمة تتحسب "
                "مرة واحدة، ولو فيه تعادل تترجّح الأولى في القائمة، ولو ولا كلمة اتطابقت "
                "يُختار:",
    "levantine": "القاعدة: تُختار النية اللي تطابق أكبر عدد من كلمات الطلب، وكل كلمة تتحسب "
                 "مرة واحدة، وإذا فيه تعادل تترجّح الأولى في القائمة، وإذا ما تطابقت ولا "
                 "كلمة يُختار:",
    "gulf": "القاعدة: تُختار النية اللي تطابق أكبر عدد من كلمات الطلب، وكل كلمة تتحسب مرة "
            "واحدة، وعند التعادل تترجّح الأولى في القائمة، وإذا ما تطابقت أي كلمة يُختار:",
    "code_switch": "القاعدة: تُختار النية التي تطابق أكبر عدد من كلمات الطلب، وتُحسب كل كلمة "
                   "مرة واحدة، وعند التعادل تُرجَّح الأسبق في القائمة، وإذا لم تتطابق أي "
                   "كلمة تُختار:",
    "english": "Rule: pick the intent matching the most request words, counting each word "
               "once; ties go to the one listed first; if no word matches pick:",
}


def validate(world: dict) -> None:
    fields(world, {"schema_version", "kind", "family_id", "template_id", "rendering",
                   "domain", "intents", "utterance", "lexicon"})
    version(world["schema_version"])
    require(world["kind"] == KIND, "unsupported symbolic world")
    for key in ("family_id", "template_id"):
        text(world[key])
    variant_of(world, KIND, optional=False)
    fields(world["domain"], {"ar", "en"})
    text(world["domain"]["ar"])
    text(world["domain"]["en"])
    intents = world["intents"]
    require(type(intents) is list and 2 <= len(intents) <= MAX_INTENTS,
            "routing needs 2..8 intents")
    identifiers = [intent["id"] for intent in intents]
    require(len(set(identifiers)) == len(identifiers), "duplicate intent ID")
    require("other" not in identifiers, "intent ID 'other' is reserved for the fallback")
    for language in ("ar", "en"):
        labels = []
        for intent in intents:
            fields(intent, {"id", "label", "keywords"})
            text(intent["id"])
            fields(intent["label"], {"ar", "en"})
            text(intent["label"]["ar"])
            text(intent["label"]["en"])
            labels.append(intent["label"][language])
        require(len(set(labels)) == len(labels), "intent labels must be distinguishable")
    tokens = set()
    for intent in intents:
        keywords = strings(intent["keywords"])
        require(1 <= len(keywords) <= MAX_TOKENS, "intent needs 1..12 keywords")
        tokens.update(keywords)
    utterance = strings(world["utterance"])
    require(1 <= len(utterance) <= MAX_TOKENS, "request needs 1..12 words")
    tokens.update(utterance)
    lexicon = world["lexicon"]
    fields(lexicon, tokens)
    for token in sorted(tokens):
        fields(lexicon[token], {"ar", "en"})
        text(lexicon[token]["ar"])
        text(lexicon[token]["en"])
    for intent in intents:
        require(intent["label"]["ar"] not in ARABIC_FALLBACKS
                and intent["label"]["en"] != FALLBACK["english"],
                "intent label collides with the fallback label")


def solve(world: dict) -> str:
    validate(world)
    request = set(world["utterance"])
    best_index, best_score = None, 0
    for index, intent in enumerate(world["intents"]):
        score = len(request & set(intent["keywords"]))
        if score > best_score:
            best_index, best_score = index, score
    if best_index is None:
        return "other"
    return world["intents"][best_index]["id"]


def independent_solve(world: dict) -> str:
    validate(world)
    inverted: dict[str, list[int]] = {}
    for index, intent in enumerate(world["intents"]):
        for token in set(intent["keywords"]):
            inverted.setdefault(token, []).append(index)
    counts = [0] * len(world["intents"])
    for token in sorted(set(world["utterance"])):
        for index in inverted.get(token, ()):
            counts[index] += 1
    ranking = sorted(range(len(counts)), key=lambda index: (-counts[index], index))
    winner = ranking[0]
    if counts[winner] == 0:
        return "other"
    return world["intents"][winner]["id"]


def _option(intent: dict, style: str) -> dict:
    language = "en" if style == "english" else "ar"
    return {"id": intent["id"], "description": intent["label"][language]}


def surface(world: dict) -> dict:
    validate(world)
    style, phrase, noise = variant_of(world, KIND, optional=False)
    words = surface_words(KIND, style, phrase)
    arabic = style != "english"
    language = "ar" if arabic else "en"
    lexicon = {token: entry[language] for token, entry in world["lexicon"].items()}
    domain = world["domain"][language]
    request = " ".join(lexicon[token] for token in world["utterance"])
    listed = []
    for index, intent in enumerate(world["intents"]):
        keywords = "، ".join(lexicon[token] for token in intent["keywords"]) if arabic \
            else ", ".join(lexicon[token] for token in intent["keywords"])
        listed.append(f"{digits(str(index + 1), style)}) {intent['label'][language]}: {keywords}")
    intents_text = ". ".join(listed)
    options = [_option(intent, style) for intent in world["intents"]]
    options.append({"id": "other", "description": FALLBACK[style]})
    state = (f"{words['domain_label']}: {domain}. {words['request_label']}: {request}. "
             f"{words['intents_label']}: {intents_text}. {RULE_SENTENCE[style]} "
             f"{FALLBACK[style]}.")
    return {"state": noised(state, noise), "question": {
        "type": "choice", "instructions": words["question"], "options": options,
    }}


def attributes_for(world: dict) -> dict:
    validate(world)
    return attributes(style_of(world, optional=False), TASK)
