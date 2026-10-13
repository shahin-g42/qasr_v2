"""Versioned metrics. Normalization never drops Hindi/Malayalam vowel signs."""

from __future__ import annotations

import unicodedata

SCORER_VERSION = "qasr-lab-metrics-v1"
POLICY = {
    "strict": "NFC plus collapsed whitespace; CER excludes whitespace",
    "recognition": (
        "NFC, casefold, decimal digits to ASCII, punctuation to spaces except "
        "apostrophes removed; Arabic harakat U+064B..065F/U+0670 and tatweel removed "
        "for ar only. Other combining marks and Arabic letter distinctions preserved. "
        "No number verbalization, translation, or semantic equivalence."
    ),
    "primary": "CER for zh/ja; whitespace WER for other languages",
}


def normalize(text, language, mode):
    text = unicodedata.normalize("NFC", text)
    if mode == "recognition":
        result = []
        for char in text.casefold():
            if language == "ar" and (0x064B <= ord(char) <= 0x065F or char in "\u0670\u0640"):
                continue
            if char in "'’":
                continue
            if unicodedata.category(char).startswith("P"):
                result.append(" ")
            elif unicodedata.category(char) == "Nd":
                result.append(str(unicodedata.decimal(char)))
            else:
                result.append(char)
        text = "".join(result)
    elif mode != "strict":
        raise ValueError(f"Unknown scoring mode: {mode}")
    return " ".join(text.split())


def edit_counts(reference, hypothesis):
    """Minimum edit alignment; ties prefer substitution, then deletion, then insertion."""
    previous = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, ref in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, hyp in enumerate(hypothesis, 1):
            if ref == hyp:
                current.append(previous[j - 1])
                continue
            cost, sub, delete, insert = previous[j - 1]
            candidates = [(cost + 1, sub + 1, delete, insert)]
            cost, sub, delete, insert = previous[j]
            candidates.append((cost + 1, sub, delete + 1, insert))
            cost, sub, delete, insert = current[j - 1]
            candidates.append((cost + 1, sub, delete, insert + 1))
            current.append(min(candidates, key=lambda value: value[0]))
        previous = current
    errors, sub, delete, insert = previous[-1]
    return {
        "errors": errors,
        "substitutions": sub,
        "deletions": delete,
        "insertions": insert,
        "reference_units": len(reference),
    }


def score_pair(reference, hypothesis, language):
    output = {}
    for mode in ("strict", "recognition"):
        ref, hyp = (normalize(text, language, mode) for text in (reference, hypothesis))
        output[mode] = {
            "word": edit_counts(ref.split(), hyp.split()),
            "char": edit_counts(list(ref.replace(" ", "")), list(hyp.replace(" ", ""))),
            "exact_match": ref == hyp,
        }
    return output


def aggregate(rows, language=None):
    successful = [row for row in rows if row["status"] == "ok"]
    speech = [row for row in successful if row["is_speech"]]
    silence = [row for row in successful if not row["is_speech"]]
    result = {
        "samples": len(rows),
        "successful": len(successful),
        "failures": len(rows) - len(successful),
        "speech_samples": len(speech),
    }
    for mode in ("strict", "recognition"):
        result[mode] = {}
        for unit, metric in (("word", "wer"), ("char", "cer")):
            keys = ("errors", "substitutions", "deletions", "insertions", "reference_units")
            counts = {key: sum(row["scores"][mode][unit][key] for row in speech) for key in keys}
            result[mode][metric] = (
                counts["errors"] / counts["reference_units"] if counts["reference_units"] else None
            )
            result[mode][unit + "_counts"] = counts
        result[mode]["exact_match_rate"] = (
            sum(row["scores"][mode]["exact_match"] for row in speech) / len(speech)
            if speech
            else None
        )
    seconds = sum(row["audio_seconds"] for row in silence)
    result["non_speech"] = {
        "samples": len(silence),
        "audio_seconds": seconds,
        "false_transcription_rate": (
            sum(row["non_speech_output"] for row in silence) / len(silence) if silence else None
        ),
        "generated_words_per_minute": (
            60 * sum(row["output_words"] for row in silence) / seconds if seconds else None
        ),
        "generated_characters_per_minute": (
            60 * sum(row["output_characters"] for row in silence) / seconds if seconds else None
        ),
    }
    result["primary_metric"] = "cer" if language in ("zh", "ja") else ("wer" if language else None)
    return result
