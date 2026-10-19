"""Editorial guide using examples bound to the exact eligible release texts."""

from __future__ import annotations

import hashlib


# Deliberately curated illustrations, separate from the hash-sampled transcript browser.
# A changed text pair must be reviewed again before reusing its editorial explanation.
EXAMPLES = (
    (
        "sada:001168",
        "1f7c12cb4deb61558c5a399b2c72b31cd5c3947e08afeb5542eef3d5b0df3822",
        "Hamza, with dialect intact",
        "انا → أنا and اني → إني restore hamza in context. The Egyptian phrasing ما افتكرش and قبل كده stays intact, preserving the speaker’s register.",
    ),
    (
        "masc_clean:001957",
        "7322c566de124c2466006a59ae8dc81f66935057ce345f3742b0ac1cfa3f0a69",
        "Conventional word endings",
        "الحلقه القادمه → الحلقة القادمة restores tāʾ marbūṭa in these words. This is a contextual spelling decision; a blanket ه → ة replacement would damage other words.",
    ),
    (
        "sada:003324",
        "32f21ce2526919c32d20f34259867fc2c0fc354af2e267d34fd4fe56d5c1602b",
        "Final letters, without rewriting speech",
        "الي المطبخ → إلى المطبخ gives the preposition its conventional spelling. The repeated نطبخ نطبخ and conversational يلا remain, rather than being edited into formal prose.",
    ),
    (
        "mgb2:000418",
        "7907c1698f750c64dc391c577b89a444ce12e29f89ae873e074f67803f5ae5ca",
        "Selective diacritics",
        "تعدى → تعدّى adds shadda to mark the doubled consonant. The draft keeps دلوقتي and تمانية and leaves the spoken amount unchanged. Richness can mean one useful mark, rather than full vocalization.",
    ),
    (
        "commonvoice:001701",
        "bc9325c66a6a7e0e9957c1d1494fc52eaf52456ebb90e0896413438a3b1cea85",
        "Punctuation that clarifies reading",
        "The draft adds the Arabic question mark ؟ to an interrogative beginning with هل. The words stay the same; the sentence becomes easier to read as a question.",
    ),
    (
        "masc_clean:007744",
        "7606589856bb720bb03fd7e4f1c3f25ea2c17a90ba760c675b3069107de62651",
        "A consistent blessing convention",
        "The existing صلى الله عليه وسلم becomes ﷺ, alongside quotation marks and selective diacritics. This is a display convention: the scorer does not expand ﷺ, so writing out the same blessing can still count as errors, even after normalization.",
    ),
)


def reference_guide(drafts):
    from gqasr.arabic_benchmark.scoring import normalize_elm

    by_id = {r["id"]: r for r in drafts}
    examples = []
    for identifier, fingerprint, title, explanation in EXAMPLES:
        row = by_id.get(identifier)
        if row is None:
            continue
        original, draft = row["text"], row["ideal_text_draft"]
        if hashlib.sha256((original + "\0" + draft).encode()).hexdigest() != fingerprint:
            continue
        examples.append(
            {
                "id": identifier,
                "dataset": row["dataset"],
                "title": title,
                "explanation": explanation,
                "original": original,
                "ideal_draft": draft,
                "normalized_equal": normalize_elm(original) == normalize_elm(draft),
            }
        )
    return {
        "column": "ideal_text_draft",
        "selection": "Curated examples illustrate specific editorial features; they are not a representative quality sample or listening verification. Only exact text pairs present in this release’s eligible, non-speculative drafts are shown.",
        "examples": examples,
    }


def reference_guide_markdown(guide):
    sections = [
        "## What the enhanced reference adds",
        "`ideal_text_draft` is the populated, text-only enhanced reference column. The editing policy targets contextual Arabic spelling (أ / إ / آ / ؤ / ئ, ي / ى, ة / ه), readable punctuation, selective diacritics, conventional word boundaries and the existing blessing rendered as ﷺ. It aims to retain dialect, code-switching, names, numbers, negation and spoken repetition. These are editorial aims, not a guarantee that every draft is correct.",
        "Benchmarking the same saved predictions against both original and enhanced references helps reveal sensitivity to reference wording and writing conventions. Normalized scores suppress selected surface differences; strict scores retain them and measure agreement with a readable written rendering. Neither is a standalone measure of punctuation quality, and a lower enhanced score does not mean the model changed or improved.",
        "QASR/GQASR predictions helped form these drafts, which may favor those models over independent baselines. Use paired original scores alongside enhanced scores; listening review and an independent holdout are needed to establish general transcription accuracy.",
        guide["selection"],
    ]
    for example in guide["examples"]:
        sections.append(
            f"### {example['title']} · `{example['id']}`\n\n"
            f"**Original:** {example['original']}\n\n"
            f"**Enhanced draft:** {example['ideal_draft']}\n\n"
            f"{example['explanation']}"
        )
    return "\n\n".join(sections)
