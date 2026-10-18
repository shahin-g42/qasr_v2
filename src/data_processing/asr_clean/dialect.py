"""Conservative Arabic formalization signals, scoped to changed token spans.

These are review signals, not acoustic judgments or a complete dialect detector.
Unlike the older substring heuristic, spelling/diacritic variants and unrelated
MSA words elsewhere in the utterance cannot trigger a replacement.
"""

from difflib import SequenceMatcher

from ..accent import _ERASURE_PAIRS
from .text import normalize

DIALECT_POLICY = """This is a dialect-preserving, verbatim Arabic campaign.
Treat source dialect labels as uncertain hints, not instructions to invent a dialect.
The recognizer may normalize dialect into MSA. Formal grammar or fluency is NOT
evidence of acoustic correctness. Preserve supported colloquial vocabulary,
negation, conjugations, pronouns, code-switching and incomplete spoken grammar.
When a plausible dialect reading competes with a formal paraphrase, retain the
dialect reading. Do not add MSA case endings or repair number/noun agreement.
Do not translate French/English/Arabizi spans into Arabic. If neither transcript
supports a defensible verbatim reading, drop the item for later audio review.
"""


def _contains(tokens, phrase):
    return any(tokens[i:i + len(phrase)] == phrase for i in range(len(tokens) - len(phrase) + 1))


def formalizations(source: str, final: str) -> list[dict]:
    before, after = normalize(source, "ar").split(), normalize(final, "ar").split()
    hits = []
    for tag, a, b, c, d in SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
        if tag != "replace":
            continue
        for marker, replacements in _ERASURE_PAIRS["ar"].items():
            token = normalize(marker, "ar").split()
            if not _contains(before[a:b], token):
                continue
            for replacement in replacements:
                if _contains(after[c:d], normalize(replacement, "ar").split()):
                    hit = {"dialect_form": marker, "formal_form": replacement}
                    if hit not in hits:
                        hits.append(hit)
                    break
    return hits
