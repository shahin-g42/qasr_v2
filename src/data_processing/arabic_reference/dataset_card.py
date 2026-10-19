"""A provenance-aware dataset card generated from the actual release statistics."""

from __future__ import annotations

import html
import json
from urllib.parse import quote


def dataset_card(datasets, provenance):
    title = provenance.get("dataset_title", "Arabic ASR Bench · Enhanced References")
    repo_id = provenance.get("dataset_repo", "shahink/asr_bench")
    audio = provenance.get("audio_included", False)
    qa = provenance.get("quality", {})
    counts = provenance["counts"]
    total = sum(c["rows"] for c in counts.values())
    proposed = sum(c["proposals"].get("proposed", 0) for c in counts.values())
    speculative = sum(c["proposals"].get("speculative_draft", 0) for c in counts.values())
    verified = sum(c["reviews"].get("verified", 0) for c in counts.values())
    hours = qa.get("hours", provenance.get("audio_encoding", {}).get("hours"))
    hours_label = (
        f"{hours:,.2f} hours" if hours is not None else "Duration reported with the audio release"
    )
    paths = provenance.get(
        "data_files",
        {
            ds: [f"data/{ds}/test-*.parquet" if audio else f"data/{ds}/test.parquet"]
            for ds in datasets
        },
    )
    header = (
        "---\npretty_name: "
        + json.dumps(title, ensure_ascii=False)
        + "\nlanguage:\n  - ar\ntask_categories:\n  - automatic-speech-recognition\n"
        + "tags:\n  - arabic\n  - speech-recognition\n  - evaluation-only\n  - enhanced-transcripts\n  - annotation-drafts\nconfigs:\n"
    )
    for name in ["all", *datasets]:
        selected_paths = [p for ds in datasets for p in paths[ds]] if name == "all" else paths[name]
        header += f"  - config_name: {name}\n"
        if name == "all":
            header += "    default: true\n"
        header += "    data_files:\n      - split: test\n        path:\n"
        header += "".join("          - " + json.dumps(p) + "\n" for p in selected_paths)
    header += "---\n\n"
    table = []
    for ds in datasets:
        c = counts[ds]
        duration = qa.get("datasets", {}).get(ds, {}).get("hours")
        duration_label = f"{duration:.2f}" if duration is not None else "—"
        table.append(
            f"| `{ds}` | {c['rows']:,} | {duration_label} | {c['proposals'].get('proposed', 0):,} | {c['proposals'].get('speculative_draft', 0):,} | {c['reviews'].get('verified', 0):,} |"
        )
    sources = []
    for source in qa.get("sources", []):
        repository, revision = source["repository"], source.get("revision")
        url = "https://huggingface.co/datasets/" + quote(repository, safe="/")
        if revision:
            url += "/tree/" + quote(revision, safe="")
        sources.append(
            f"| `{source['dataset']}` | [{repository}]({url}) | `{revision or 'not recorded'}` |"
        )
    sources_section = (
        "| Configuration | Source repository | Pinned revision |\n|---|---|---|\n"
        + "\n".join(sources)
        if sources
        else "Each row records its source repository, pinned revision and segment locator in `source_locator_json`."
    )
    checkpoints = sorted(
        {
            r["reported_checkpoint"]
            for r in provenance.get("corrector_runs", [])
            if r.get("reported_checkpoint")
        }
    )
    checkpoint = (
        ", ".join(f"`{name}`" for name in checkpoints)
        or "See `provenance.json` for the recorded corrector identity."
    )
    quality_status = (
        f"**Passed:** all {qa['decoded_audio_rows']:,} stored recordings were decoded and checked after writing. Every annotation column was compared with the frozen source bundle."
        if qa.get("status") == "passed"
        else "The cluster packaging stage must pass a full read-back audit before this release can be uploaded."
    )
    signal = qa.get("signal_observations", {})
    quality_details = (
        f"The audit records **{len(signal.get('all_zero_audio', [])):,} all-zero waveforms**, "
        f"**{len(signal.get('peak_above_one', [])):,} waveforms with peaks above 1.0**, and "
        f"**{qa.get('duplicate_audio_groups', 0):,} groups sharing the same prepared-audio hash** "
        f"({qa.get('duplicate_extra_rows', 0):,} additional rows). "
        "These observations are retained in `quality_report.json`; source rows are not silently dropped or gain-normalized."
        if qa.get("status") == "passed"
        else "Signal observations and duplicate-audio counts are recorded in the audio release's `quality_report.json`."
    )
    flags = qa.get("editorial_flags", {})
    flag_table = (
        "\n\n| Review cue | Rows flagged |\n|---|---:|\n"
        + "\n".join(f"| `{name}` | {n:,} |" for name, n in sorted(flags.items()))
        + "\n\nFlags can overlap. A mark difference or lexical-change flag is a review cue, not a confirmed transcription error."
        if flags
        else ""
    )
    audio_section = (
        "Audio is embedded directly in Parquet using the Hugging Face `Audio` feature: **16,000 Hz, mono, WAV FLOAT**. "
        "The first column is playable audio. This preserves the exact float32 benchmark samples, without lossy encoding, integer quantization, gain adjustment or clipping. "
        "It preserves the prepared benchmark waveform; it does not claim to preserve the original source container or pre-resampling signal."
        if audio
        else "This annotation-only bundle carries source locators and hashes. The cluster packaging stage adds the matching recordings without altering the transcript columns."
    )
    audio_example = (
        """
To inspect the embedded waveform without depending on a particular automatic audio decoder:

```python
import io
import soundfile as sf
from datasets import Audio

row = ds.cast_column("audio", Audio(decode=False))[0]
waveform, sample_rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
assert sample_rate == 16000
```
"""
        if audio
        else ""
    )
    return (
        header
        + f"""# {html.escape(title)}

**Arabic speech • Contextual orthography • Traceable editorial decisions**

{total:,} clips · {hours_label} · {len(datasets)} source configurations · Private research release

This edition brings the original benchmark references, saved QASR and GQASR predictions, and contextually edited Arabic transcript drafts into one reproducible dataset. The enhancement focuses on readable, linguistically informed Arabic while retaining the evidence needed to inspect every decision.

**The enhanced transcript is in `ideal_text_draft`.** Final `ideal_text` and `verified_text` are reserved for listening-reviewed references. This release has **{verified:,} audio-reviewed transcripts**; packaging validation does not certify the spoken wording.

## At a glance

| Property | This release |
|---|---|
| Release | `{provenance["release_id"]}` |
| Scope | `{provenance["selection_scope"]}` |
| Split | `test` — evaluation only |
| Transcript coverage | {proposed:,} LLM drafts; {speculative:,} explicitly speculative drafts |
| Audio | {"Embedded 16 kHz mono float32 WAV" if audio else "Source locators; audio packaging pending"} |
| Original evidence | Original reference + QASR prediction + GQASR prediction |
| Corrector checkpoint | {checkpoint} |
| Reproducibility | Source revisions, row bindings, audio hashes, run lineage and release checksums |

## What makes this edition different

**Contextual Arabic editing.** The corrector considers each candidate in context when proposing hamza and maddah (`أ / إ / آ / ؤ / ئ`), final `ي / ى`, `ة / ه`, conventional spelling and word boundaries. These are contextual decisions, not global character replacements.

**Readable Arabic with dialect retained.** The editing policy targets natural punctuation, useful vowel marks and shadda while preserving dialect, code-switching, names, negation, numbers and audible repetition. The aim is a faithful transcript, rather than a rewrite into formal Arabic. Automated review flags expose places where drafts may depart from that policy.

**A consistent blessing convention.** An existing `صلى الله عليه وسلم` is rendered as **ﷺ**, including supported diacritized forms. The convention does not authorize adding a blessing absent from the supplied text or replacing other blessings.

**Multiple candidates, preserved evidence.** The text-only corrector receives the original reference and saved QASR/GQASR candidates under anonymous labels in varying order. The release keeps each candidate, proposed edits, uncertainty notes and independently computed text differences. A fluent candidate is not automatically treated as correct.

**Explicit uncertainty.** The {speculative:,} assistant-reconstructed cases remain labelled `speculative=true`, with their rationale, qualitative confidence and uncertain spans. They are distinguishable from the {proposed:,} ordinary LLM drafts and cannot supply verified reference values.

The examples below illustrate the editorial policy; they are not claims that these particular edits were verified in the recordings.

| Area | Illustrative refinement | Constraint |
|---|---|---|
| Hamza | `انا` → `أنا` | Choose from context; preserve meaning |
| Hamza seat | `المسؤلية` → `المسؤولية` | Conventional spelling needs contextual support |
| Final letters | `الحلقه القادمه` → `الحلقة القادمة` | Do not apply blanket `ه` → `ة` substitution |
| Boundaries | `و ندمت` → `وندمت` | Preserve the spoken words |
| Diacritics | `كتابا` → `كتابًا` | Add useful marks conservatively |
| Blessing | `صلى الله عليه وسلم` → `ﷺ` | Render an existing phrase only |

## Composition

`all` combines the selected configurations below. Each named configuration exposes its own `test` split. Counts describe this frozen benchmark selection, not the full upstream corpora.

| Configuration | Clips | Hours | LLM drafts | Speculative | Audio-reviewed |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

## Load and inspect

Authenticate with an account that can read the private repository. Pin the commit returned by publication for repeatable experiments.

```python
from datasets import load_dataset

ds = load_dataset(
    {json.dumps(repo_id)},
    "all",                  # or a named configuration, e.g. {json.dumps(datasets[0])}
    split="test",
    revision="<release-commit-sha>",
    token=True,
)

row = ds[0]
print(row["text"])              # unchanged source reference
print(row["asr_text"])          # saved QASR prediction
print(row["ideal_text_draft"])  # enhanced, unverified transcript
print(row["proposal_status"], row["speculative"])
```
{audio_example}
## Schema guide

| Columns | Meaning |
|---|---|
| `audio`, `duration_seconds`, `audio_num_samples` | Embedded recording and measured extent; present in the audio release |
| `id`, `dataset`, `split`, `dialect` | Stable row identity, source configuration, test split and available dialect metadata |
| `text` | Immutable original reference |
| `asr_text`, `gqasr_text`, `candidate_texts_json` | Saved ASR candidates; no fresh ASR inference during packaging |
| `verified_text_draft` | Proposed spoken wording; despite its name, still unverified |
| `ideal_text_draft` | Contextually edited rendering of the proposed wording |
| `verified_text`, `ideal_text` | Final values only after an attested listening review; null otherwise |
| `proposal_status`, `correction_method`, `review_status`, `audio_reviewed` | How the draft was produced and whether a listening review exists |
| `speculative`, `speculation_confidence`, `speculative_reason`, `speculative_spans_json` | Explicit best-guess status and its supporting uncertainty record |
| `changes_json`, `checks_json`, `uncertainties_json`, `issue_notes_json` | Editorial evidence and computed review cues; JSON strings, decoded with `json.loads` |
| `source_locator_json`, `audio_sha256`, `audio_bytes_sha256`, `audio_samples_sha256` | Pinned source locator; hashes of prepared NumPy file, embedded WAV and little-endian float32 samples |
| `binding`, `policy_sha256`, `proposal_run_sha256`, `proposal_record_sha256`, `speculative_draft_sha256`, `reviewed_record_sha256`, `release_id` | Links to the exact source, policy, proposal, optional overlay and release |
| `cohorts`, `flags`, `purpose` | Sampling membership, diagnostic cues and evaluation-only purpose |

## Audio and release quality

{audio_section}

{quality_status}

The packaging checks cover unique row IDs, expected coverage, consistent schemas, source-cache hashes, WAV hashes, decoded sample hashes, sample rate, channel count, duration, finite samples and preserved transcript columns. The dataset card names the exact shards in the current release, so older files cannot accidentally enter a configuration after an update.

{quality_details}

## Reference quality and evaluation

These enhancements are **candidate-assisted editorial drafts**, not an independently transcribed gold standard. The corrector cannot hear audio. It can select an incorrect candidate, normalize a meaningful dialect form, omit a mark or repair grammar that the speaker actually used. The six-case speculative route, when included, makes additional contextual guesses.

The frozen benchmark retains source issues such as unclear-span annotations, competing script conventions and duplicate recordings with differing references. Duplicate groups are identifiable by audio hashes. A reference difference can reflect spelling, segmentation, annotation policy or model error; a lower WER alone does not resolve which is correct.{flag_table}

For evaluation:

- Use `text` for the original-reference track.
- Label any scores against `ideal_text_draft` as **candidate-assisted draft-reference scores**. QASR and GQASR contributed to reference development, so this track is not an independent comparison of those models.
- Report the scored subset and treatment of speculative rows. Never silently fill a null final reference from a draft or the original text.
- Use `verified_text` or `ideal_text` only where `audio_reviewed=true`; if coverage is zero, no verified-reference score exists.
- Report strict and normalized WER/CER separately, with the exact normalization recipe. Punctuation, diacritics, hamza, script choice and the ﷺ convention can change the score.
- Keep this test release out of training. Apply transferable editing rules to separate training data and evaluate on a fresh holdout.

## Sources and reproducibility

{sources_section}

The upstream datasets and their creators retain their attribution and source terms. This private enhancement release does not assign a blanket license to the underlying recordings.

`provenance.json` records the input prediction hashes, corrector identities, correction/recovery lineage and review coverage. `policy.json` captures the frozen annotation policy. `quality_report.json` contains the audio release audit. `release.json` binds the exact distributed files by SHA-256. Local cache paths, credentials, raw server replies and reviewer identities are excluded.

<details>
<summary>Frozen reference identifiers</summary>

- Plan digest: `{provenance["manifest_digest"]}`
- Policy digest: `{provenance["policy_sha256"]}`
- Annotation bundle: `{provenance.get("annotations_bundle_sha256", "This is the annotation bundle; see release.json.")}`

Use the Hugging Face commit SHA, rather than a moving branch name, as the published dataset version.

</details>
"""
    )
