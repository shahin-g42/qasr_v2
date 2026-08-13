"""Transform q3asr SFT filter manifests into per-language QASR training manifests.

Input format (one JSON object per line)::

    {"audio": "data/asr/.../file.wav", "text": "language Arabic<asr_text>..."}

Transform steps:
  1. Parse the ``language <NAME><asr_text><transcript>`` envelope into a
     language code and a clean transcript.
  2. Resolve the relative audio paths against ``--base-dir`` and probe the
     duration from the wav header (training manifests require it).
  3. Records tagged ``language None`` get their language identified:
     deterministic script detection first (Arabic / Devanagari / Malayalam /
     CJK map 1:1 to our language set), then the same vLLM service used by
     the cleaning pipeline for Latin-script/ambiguous text.
  4. Write per-language training manifests ``{audio_filepath, text, duration}``
     under ``<output-dir>/<lang>/<split>_<lang>_<corpus>.jsonl`` — the same
     language-namespaced layout and self-describing filenames as
     ``training_manifests/``.

Rejected piles (kept next to the outputs, never silently dropped):
    rejected_no_duration.jsonl       — audio header unreadable / file missing
    rejected_unknown_language.jsonl  — LLM + script heuristics both failed

Usage — check the planned routing first (free, no server needed)::

    PYTHONPATH=src python3 scripts/prepare_q3asr_filter.py --dry-run

Then, on a node with the vLLM server from the cleaning config running::

    PYTHONPATH=src python3 scripts/prepare_q3asr_filter.py \
        --inputs /lustrefs/shared/shahin.konadath/workspace/train/tts/vllm/qasr/data/asr/.dset/dpo/enriched/q3asr/json/sft/train_filter.jsonl \
                 /lustrefs/shared/shahin.konadath/workspace/train/tts/vllm/qasr/data/asr/.dset/dpo/enriched/q3asr/json/sft/eval_filter.jsonl

Each run rebuilds its outputs from scratch (existing manifests are
overwritten), matching the immutable-snapshot convention of
``assemble_training_manifests.py``. Re-running after an interruption is
therefore safe, but it re-pays the full LLM cost — point ``--output-dir``
somewhere else if you want to keep a previous attempt.

To spread the work over the cluster, pass ``--node-rank``/``--num-nodes`` (one
process per node, each against that node's own vLLM server, as
``scripts/run_clean_node.sh`` does) and then merge the per-node outputs with
``scripts/merge_q3asr_shards.py``. Sharding is by a stable hash of the audio
path, so the dedup and train/eval overlap checks stay exact within a shard.

Transcripts are length-checked with the real training tokenizer, so pass
``--tokenizer`` pointing at the model directory the training config loads (the
same ``model_name_or_path``). Over-length transcripts are rejected, never
truncated. Pass ``--no-length-check`` only if you accept that training will
substitute duplicates for whatever exceeds the budget.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_processing.arabic_utils import ARABIC_LETTER_RANGE
from data_processing.config import PipelineConfig
from data_processing.duration_probe import probe_duration
from data_processing.llm_client import VLLMClient
from data_processing.pipeline import setup_logging
from data_processing.text_utils import LANGUAGE_SCRIPTS

LOGGER = logging.getLogger("qasr.prepare_q3asr_filter")

__all__ = [
    "SUPPORTED_LANGUAGES",
    "SourceRecord",
    "apply_script_heuristics",
    "audit_language_labels",
    "count_target_tokens",
    "find_label_contradictions",
    "heuristic_language",
    "iter_source_records",
    "load_target_tokenizer",
    "normalize_language_value",
    "open_lid_client",
    "parse_text_envelope",
    "resolve_audio_path",
    "script_char_counts",
    "shard_records",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES = ("ar", "en", "hi", "ml", "zh")

# Full language names as they appear in the source envelope → ISO codes
LANGUAGE_NAME_TO_CODE: dict[str, str] = {
    "arabic": "ar",
    "english": "en",
    "hindi": "hi",
    "malayalam": "ml",
    "chinese": "zh",
}

# "language <NAME><asr_text><transcript>" — split on the FIRST marker only,
# the transcript itself may contain anything.
TEXT_ENVELOPE_RE = re.compile(
    r"^language\s+(?P<lang>.+?)<asr_text>(?P<text>.*)$", re.DOTALL
)

DEFAULT_INPUTS = [
    "/lustrefs/shared/shahin.konadath/workspace/train/tts/vllm/qasr/"
    "data/asr/.dset/dpo/enriched/q3asr/json/sft/train_filter.jsonl",
    "/lustrefs/shared/shahin.konadath/workspace/train/tts/vllm/qasr/"
    "data/asr/.dset/dpo/enriched/q3asr/json/sft/eval_filter.jsonl",
]
DEFAULT_BASE_DIR = "/lustrefs/shared/shahin.konadath/workspace/train/tts/vllm/qasr"
DEFAULT_OUTPUT_DIR = (
    "/lustrefs/shared/shahin.konadath/workspace/train/stt/qasr/q3asr_sft_manifests"
)
DEFAULT_CONFIG = "configs/data_processing/multilingual_cleaning.yaml"

# Records outside this band are silently dropped by the training data loader,
# so route them to a rejected pile instead of writing manifests whose counts
# overstate what will train. These are the values every configs/*.yaml sets,
# NOT the laxer QASRConfig dataclass defaults (max 30.0), which no training
# config actually uses. 35.0 is also the hard ceiling: train.py raises if
# max_duration_seconds exceeds the feature extractor's max_audio_clip_s (35.0).
TRAIN_MIN_DURATION = 0.1
TRAIN_MAX_DURATION = 35.0

# Transcripts longer than this are NOT truncated by training — with
# truncate_long_transcripts: false (every configs/*.yaml), ResilientAudioDataset
# raises TranscriptLengthError, marks the index unusable, and then serves a
# DIFFERENT record in its place (data.py __getitem__ walks to the next valid
# index). So an over-length record does not merely vanish: because __len__ is
# unchanged, its slot is filled by a duplicate of a neighbour, quietly
# oversampling that neighbour. Rejecting here keeps one epoch honest.
#
# Measured against the BARE manifest text with add_special_tokens=False,
# because that is exactly what data.py and collator.py count. The
# "language <NAME><asr_text>...<eos>" envelope that processing.py wraps around
# the target is NOT part of the budget; including it would reject records
# training would happily accept.
MAX_TARGET_TOKENS = 512

LID_SYSTEM_PROMPT = (
    "You are a language identification tool. Identify the dominant spoken "
    "language of the user's text. The answer must be exactly one of: "
    "Arabic (ar), English (en), Hindi (hi), Malayalam (ml), Chinese (zh). "
    "The text may contain code-switched words or Latin-script loanwords; "
    "pick the dominant language of the utterance. Respond with ONLY a JSON "
    'object: {"language": "<code>"} where <code> is one of: '
    "ar, en, hi, ml, zh."
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@dataclass
class SourceRecord:
    """One parsed record from the source JSONL."""

    split: str                  # "train" / "eval" (from the input file stem)
    audio: str                  # audio path as it appears in the source
    language: str | None        # ISO code, or None when tagged "language None"
    text: str                   # the transcript after <asr_text>
    source_file: str = ""
    audit: str = ""             # note when the source label was questioned


def parse_text_envelope(text: str) -> tuple[str, str] | None:
    """Split ``language <NAME><asr_text><transcript>`` into (name, transcript).

    Returns None when the envelope markers are missing or the transcript
    is empty.
    """
    match = TEXT_ENVELOPE_RE.match(text.strip())
    if not match:
        return None
    lang_name = match.group("lang").strip()
    transcript = match.group("text").strip()
    if not lang_name or not transcript:
        return None
    return lang_name, transcript


def normalize_language_value(value: str) -> str | None:
    """Map a language name or code to an ISO code; None for 'None'/unknown."""
    cleaned = value.strip().lower()
    if cleaned in ("", "none", "null", "unknown"):
        return None
    if cleaned in SUPPORTED_LANGUAGES:
        return cleaned
    return LANGUAGE_NAME_TO_CODE.get(cleaned)


def split_from_filename(path: Path) -> str:
    """Derive the split name from the input file stem (train_filter -> train)."""
    first_token = path.stem.split("_")[0].lower()
    if first_token in ("train", "eval", "dev", "test", "val"):
        return "eval" if first_token in ("eval", "dev", "test", "val") else "train"
    LOGGER.warning(
        "%s: could not derive split from filename, assuming 'train'", path.name
    )
    return "train"


def iter_source_records(path: str | Path) -> Iterator[SourceRecord]:
    """Stream parsed records from one source JSONL file.

    Corrupt lines and records without a usable envelope are counted and
    skipped rather than fatal (same philosophy as manifest_io.read_manifest).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input manifest not found: {path}")

    split = split_from_filename(path)
    bad_lines = 0
    bad_envelope = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            if not isinstance(raw, dict):
                bad_lines += 1
                continue

            audio = raw.get("audio") or raw.get("audio_filepath")
            text = raw.get("text")
            if not isinstance(audio, str) or not isinstance(text, str):
                bad_lines += 1
                continue

            parsed = parse_text_envelope(text)
            if parsed is None:
                bad_envelope += 1
                if bad_envelope <= 5:
                    LOGGER.warning(
                        "%s:%d: unparseable text envelope, skipping",
                        path.name,
                        line_number,
                    )
                continue

            lang_name, transcript = parsed
            yield SourceRecord(
                split=split,
                audio=audio.strip(),
                language=normalize_language_value(lang_name),
                text=transcript,
                source_file=str(path),
            )

    if bad_lines:
        LOGGER.warning("%s: skipped %d corrupt/unusable line(s)", path.name, bad_lines)
    if bad_envelope:
        LOGGER.warning(
            "%s: skipped %d line(s) without a valid language/<asr_text> envelope",
            path.name,
            bad_envelope,
        )


def resolve_audio_path(audio: str, base_dir: str | Path) -> str:
    """Join relative source audio paths onto the dataset base directory."""
    if audio.startswith("/"):
        return audio
    return str(Path(base_dir) / audio)


# ---------------------------------------------------------------------------
# Language identification for "language None" records
# ---------------------------------------------------------------------------

# A single stray character must not decide a record's language. Presence-only
# detection misrouted "The word for pain is ألم in Arabic" to ar and "The
# Hindi word नमस्ते means hello" to hi (both measured) — so require the
# winning script to *dominate*, as split_manifest_by_script.classify does.
MIN_SCRIPT_CHARS = 3
# Measured band where nothing misroutes: 0.28-0.45. Below it, English quoting
# a foreign word is stolen ("She said വണക്കം to greet us" is ml at share
# 0.27); above it, genuine Chinese carrying heavy Latin is lost ("我们的 meeting
# 在 3:30 开始", share 0.46). The two failures are NOT symmetric — undershooting
# bakes a wrong language into the training target, overshooting only costs one
# LLM call — so sit mid-band, slightly high. Coverage on real records is
# identical (30/43 resolved) at 0.30/0.40/0.50, so the margin is free.
MIN_SCRIPT_SHARE = 0.35

# Non-Latin scripts that map 1:1 onto our training languages.
_SCRIPT_PATTERNS: dict[str, re.Pattern[str]] = {
    "ar": ARABIC_LETTER_RANGE,
    "hi": LANGUAGE_SCRIPTS["hi"],
    "ml": LANGUAGE_SCRIPTS["ml"],
    "zh": LANGUAGE_SCRIPTS["zh"],
}


def script_char_counts(text: str) -> dict[str, int]:
    """Count characters per non-Latin script, with Latin counted under 'en'.

    Only non-zero entries are returned.
    """
    counts = {
        code: len(pattern.findall(text)) for code, pattern in _SCRIPT_PATTERNS.items()
    }
    counts["en"] = len(LANGUAGE_SCRIPTS["en"].findall(text))
    return {code: total for code, total in counts.items() if total}


def heuristic_language(text: str) -> str | None:
    """Deterministic script-based detection for our five training languages.

    Returns a language only when one non-Latin script dominates the record's
    alphabetic characters. Latin-dominant text returns None on purpose:
    English and romanized hi/ml are indistinguishable by script, so that
    judgement belongs to the LLM pass.
    """
    counts = script_char_counts(text)
    total = sum(counts.values())
    if not total:
        return None
    non_latin = {code: n for code, n in counts.items() if code != "en"}
    if not non_latin:
        return None
    winner = max(non_latin, key=lambda code: non_latin[code])
    hits = non_latin[winner]
    if hits >= MIN_SCRIPT_CHARS and hits / total >= MIN_SCRIPT_SHARE:
        return winner
    return None


def build_lid_messages(text: str) -> list[dict[str, str]]:
    """Chat messages for the vLLM language-identification call."""
    return [
        {"role": "system", "content": LID_SYSTEM_PROMPT},
        {"role": "user", "content": f"Identify the language of this text:\n{text}"},
    ]


async def llm_identify_language(
    client: VLLMClient,
    text: str,
    semaphore: asyncio.Semaphore,
) -> str | None:
    """Ask the cleaning pipeline's vLLM service for the dominant language."""
    async with semaphore:
        try:
            result = await client.chat_completion_json(
                build_lid_messages(text),
                temperature=0.2,
                max_tokens=64,
            )
        except Exception as exc:  # logged, heuristic fallback upstream
            LOGGER.warning("LLM language identification failed: %s", exc)
            return None
    if not isinstance(result, dict):
        return None
    return normalize_language_value(str(result.get("language", "")))


def apply_script_heuristics(records: list[SourceRecord]) -> list[SourceRecord]:
    """Resolve what script detection can, in place; return the LLM's worklist.

    Free and deterministic, so it always runs before any GPU time is spent.
    """
    unknown = [r for r in records if r.language is None]
    needs_llm: list[SourceRecord] = []
    for record in unknown:
        guess = heuristic_language(record.text)
        if guess is not None:
            record.language = guess
        else:
            needs_llm.append(record)
    if unknown:
        LOGGER.info(
            "%d record(s) tagged 'language None': %d resolved by script "
            "heuristics, %d need the LLM",
            len(unknown),
            len(unknown) - len(needs_llm),
            len(needs_llm),
        )
    return needs_llm


def shard_records(
    records: list[SourceRecord], node_rank: int, num_nodes: int
) -> list[SourceRecord]:
    """Take this node's slice, partitioned by a stable hash of the audio path.

    Deliberately NOT round-robin by line index. The dedup and train/eval
    overlap checks are whole-corpus properties, and hashing on the audio path
    lands every copy of a given file — including its train copy and its eval
    copy — on the same node. Round-robin would scatter those pairs across
    nodes and both checks would silently pass on data they never compared.

    blake2b rather than the builtin hash(): PYTHONHASHSEED randomises string
    hashing per process, so hash() would give each node a different partition
    function, simultaneously duplicating some records and dropping others.
    """
    if num_nodes <= 1:
        return records
    mine = [
        record
        for record in records
        if int.from_bytes(
            hashlib.blake2b(record.audio.encode("utf-8"), digest_size=8).digest(),
            "big",
        )
        % num_nodes
        == node_rank
    ]
    LOGGER.info(
        "Node %d/%d owns %d of %d parsed record(s)",
        node_rank,
        num_nodes,
        len(mine),
        len(records),
    )
    return mine


def open_lid_client(config: PipelineConfig, concurrency: int) -> VLLMClient:
    """A vLLM client whose rate ceiling matches our concurrency bound.

    The client's default 10 req/s token bucket suits the cleaning pipeline's
    one-client-per-worker design (workers_per_node=96, each building its own
    client in Worker.run, so ~960 req/s per node). This script shares ONE
    client across every request, so accepting the default would cap the whole
    process at 10 req/s — ~96x slower than the cleaning pipeline on identical
    hardware. Lift the ceiling so vLLM, not the bucket, sets the pace; the
    semaphore still bounds in-flight requests.

    Both LLM phases must go through here — they were briefly inconsistent,
    which silently throttled the label audit to 10 req/s.
    """
    rate = float(max(concurrency, 10))
    return VLLMClient(config, rate_limit=rate, rate_capacity=rate * 2)


async def resolve_unknown_languages(
    records: list[SourceRecord],
    config: PipelineConfig,
    concurrency: int,
) -> int:
    """Fill in ``language`` for records tagged None.

    Script heuristics run first (deterministic for our non-Latin languages);
    the remaining Latin-script records go to the vLLM service. Returns the
    number of records that got a language from the LLM.
    """
    needs_llm = apply_script_heuristics(records)
    if not needs_llm:
        return 0

    semaphore = asyncio.Semaphore(max(1, concurrency))
    async with open_lid_client(config, concurrency) as client:
        await client.wait_for_health(max_wait_seconds=300)
        verdicts = await asyncio.gather(
            *(llm_identify_language(client, r.text, semaphore) for r in needs_llm)
        )

    resolved = 0
    for record, verdict in zip(needs_llm, verdicts, strict=True):
        if verdict in SUPPORTED_LANGUAGES:
            record.language = verdict
            resolved += 1
        else:
            # Last resort: the heuristic may see a script the LLM missed
            record.language = heuristic_language(record.text)
    LOGGER.info("LLM resolved %d/%d ambiguous record(s)", resolved, len(needs_llm))
    return resolved


# ---------------------------------------------------------------------------
# Auditing the source labels
# ---------------------------------------------------------------------------


def find_label_contradictions(
    records: list[SourceRecord],
) -> list[tuple[SourceRecord, str]]:
    """Labelled records whose script evidence contradicts their own label.

    Source labels are NOT authoritative. Measured on a 20k random sample of
    training_manifests/v4.0/en/train_en_q3asr.jsonl: 0.3% of records labelled
    ``en`` are script-dominantly Chinese — Mandarin utterances carrying English
    loanwords (median 21 Latin chars, none Latin-free), i.e. roughly 28k
    mislabels in that one manifest. Those records teach the model the wrong
    language marker, so the contradiction is worth an LLM adjudication.
    """
    suspect: list[tuple[SourceRecord, str]] = []
    for record in records:
        if record.language is None:
            continue
        guess = heuristic_language(record.text)
        if guess is not None and guess != record.language:
            suspect.append((record, guess))
    return suspect


async def audit_language_labels(
    records: list[SourceRecord],
    config: PipelineConfig,
    concurrency: int,
) -> dict[str, int]:
    """Let the LLM adjudicate labels that script evidence contradicts.

    Three-way vote between the source label, the script, and the LLM:
      * LLM sides with the script  -> relabel (two independent signals agree)
      * LLM sides with the label   -> keep the label, script was fooled
      * LLM says a third thing / fails -> keep the label, mark disputed
    Deliberately conservative: we only overrule the dataset on consensus.
    """
    suspect = find_label_contradictions(records)
    stats = {"suspect": len(suspect), "relabelled": 0, "kept": 0, "disputed": 0}
    if not suspect:
        return stats

    LOGGER.info(
        "%d labelled record(s) contradict their own script evidence — asking "
        "the LLM to adjudicate",
        len(suspect),
    )
    semaphore = asyncio.Semaphore(max(1, concurrency))
    async with open_lid_client(config, concurrency) as client:
        await client.wait_for_health(max_wait_seconds=300)
        verdicts = await asyncio.gather(
            *(llm_identify_language(client, r.text, semaphore) for r, _ in suspect)
        )

    for (record, script_says), llm_says in zip(suspect, verdicts, strict=True):
        label = record.language
        if llm_says == script_says:
            record.audit = f"relabelled {label}->{script_says} (script+llm agree)"
            record.language = script_says
            stats["relabelled"] += 1
        elif llm_says == label:
            record.audit = f"kept {label} (llm backs the label, script said {script_says})"
            stats["kept"] += 1
        else:
            record.audit = (
                f"disputed: label={label} script={script_says} llm={llm_says}"
            )
            stats["disputed"] += 1

    LOGGER.info(
        "Label audit: %d relabelled, %d kept, %d disputed (of %d suspect)",
        stats["relabelled"],
        stats["kept"],
        stats["disputed"],
        stats["suspect"],
    )
    return stats


# ---------------------------------------------------------------------------
# Transcript length
# ---------------------------------------------------------------------------


def load_target_tokenizer(tokenizer_path: str) -> Any:
    """The tokenizer training will measure transcripts with.

    Must be the same ``model_name_or_path`` the training config loads: token
    counts are vocabulary-dependent, so measuring with a different tokenizer
    would give a confidently wrong answer. Imported lazily so the rest of the
    script (and --dry-run) works without transformers installed.
    """
    from transformers import AutoTokenizer

    # A mistyped Lustre path is the likely failure, and transformers reports it
    # as "Repo id must be in the form 'repo_name'..." because it falls back to
    # treating the value as a Hub id. Say what actually went wrong. Relative
    # values still fall through to the Hub, so repo ids keep working.
    if tokenizer_path.startswith("/") and not Path(tokenizer_path).is_dir():
        raise FileNotFoundError(f"no such tokenizer directory: {tokenizer_path}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if not getattr(tokenizer, "is_fast", False):
        LOGGER.warning(
            "%s loaded a slow tokenizer; counting millions of transcripts will "
            "be slow but correct",
            tokenizer_path,
        )
    return tokenizer


def count_target_tokens(
    texts: list[str], tokenizer: Any, batch_size: int = 2000
) -> list[int]:
    """Token counts under training's exact rule, one per input text.

    Mirrors ``ResilientAudioDataset._validate_transcript`` and
    ``QASRDataCollator.__call__``: the bare manifest text, add_special_tokens
    False, truncation False. Batched because tokenizing millions of strings one
    call at a time wastes the fast tokenizer's Rust-side parallelism.
    """
    counts: list[int] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        encoded = tokenizer(chunk, add_special_tokens=False, truncation=False)
        counts.extend(len(ids) for ids in encoded["input_ids"])
    return counts


# ---------------------------------------------------------------------------
# Duration probing
# ---------------------------------------------------------------------------


def fill_durations(records: list[SourceRecord], base_dir: str, max_workers: int) -> dict[str, float]:
    """Probe wav-header durations for every record, keyed by resolved path.

    Returns {resolved_audio_path: duration}; paths whose header is
    unreadable are simply absent from the dict.
    """
    resolved = {r.audio: resolve_audio_path(r.audio, base_dir) for r in records}
    unique_paths = sorted(set(resolved.values()))
    LOGGER.info("Probing durations for %d unique audio file(s)", len(unique_paths))

    with ThreadPoolExecutor(
        max_workers=max(1, max_workers), thread_name_prefix="duration-probe"
    ) as pool:
        durations = list(pool.map(probe_duration, unique_paths))

    return {
        path: duration
        for path, duration in zip(unique_paths, durations, strict=True)
        if duration is not None
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_manifest(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Write training-format records as JSONL (UTF-8, unescaped scripts)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def transform(
    inputs: list[str],
    base_dir: str,
    output_dir: str,
    config: PipelineConfig,
    corpus: str = "q3asr",
    concurrency: int = 256,
    probe_workers: int = 64,
    dry_run: bool = False,
    audit_labels: bool = True,
    min_duration: float = TRAIN_MIN_DURATION,
    max_duration: float = TRAIN_MAX_DURATION,
    node_rank: int = 0,
    num_nodes: int = 1,
    tokenizer: Any | None = None,
    max_target_tokens: int = MAX_TARGET_TOKENS,
) -> dict[str, int]:
    """Full transform: parse -> LID -> duration probe -> per-language outputs."""
    records: list[SourceRecord] = []
    for input_path in inputs:
        records.extend(iter_source_records(input_path))
    LOGGER.info("Parsed %d usable record(s) from %d input file(s)", len(records), len(inputs))
    if not records:
        return {}

    # Split before any dedup/overlap work: hash-partitioning by audio keeps
    # both of those checks exact within a shard (see shard_records).
    records = shard_records(records, node_rank, num_nodes)
    if not records:
        LOGGER.warning("Nothing assigned to node %d/%d", node_rank, num_nodes)
        return {}

    # Same audio twice inside one split is a duplicated training sample, not a
    # second observation. Keep the first occurrence.
    seen: set[tuple[str, str]] = set()
    deduped: list[SourceRecord] = []
    for record in records:
        key = (record.split, record.audio)
        if key not in seen:
            seen.add(key)
            deduped.append(record)
    if len(deduped) != len(records):
        LOGGER.warning(
            "Dropped %d intra-split duplicate record(s) (same audio listed "
            "twice in one split)",
            len(records) - len(deduped),
        )
        records = deduped

    # Contamination guard, before we spend any GPU time: the same audio in
    # both splits would silently inflate eval metrics.
    by_split: dict[str, set[str]] = {}
    for record in records:
        by_split.setdefault(record.split, set()).add(record.audio)
    overlap = by_split.get("train", set()) & by_split.get("eval", set())
    if overlap:
        LOGGER.warning(
            "%d audio file(s) appear in BOTH train and eval — eval metrics on "
            "these manifests would be contaminated. First: %s",
            len(overlap),
            sorted(overlap)[0],
        )

    # 1) Language identification for "language None" records. Script detection
    #    is free, so a dry run still reports the routing it would produce.
    if dry_run:
        needs_llm = apply_script_heuristics(records)
        planned: dict[str, int] = {}
        for record in records:
            key = record.language or "NEEDS-LLM"
            planned[key] = planned.get(key, 0) + 1
        LOGGER.info("DRY RUN — no LLM calls, no duration probing, nothing written")
        for lang in sorted(planned):
            LOGGER.info("  %-9s %d record(s)", lang, planned[lang])
        LOGGER.info(
            "  -> %d record(s) would need an LLM call (%.1f%% of the corpus)",
            len(needs_llm),
            100.0 * len(needs_llm) / len(records),
        )
        return {}

    asyncio.run(resolve_unknown_languages(records, config, concurrency))

    # 1b) The source labels are not authoritative — adjudicate the ones that
    #     contradict their own script evidence.
    if audit_labels:
        asyncio.run(audit_language_labels(records, config, concurrency))

    # 2) Durations (wav headers, no decode)
    duration_by_path = fill_durations(records, base_dir, probe_workers)

    # 3) Bucket into per-language outputs / rejected piles
    out_root = Path(output_dir)
    # write_manifest truncates, so every file this node emits needs a per-node
    # name or 8 nodes would leave only whichever finished last.
    shard_suffix = f".rank{node_rank}of{num_nodes}" if num_nodes > 1 else ""
    buckets: dict[Path, list[dict[str, Any]]] = {}
    rejected_no_duration: list[dict[str, Any]] = []
    rejected_out_of_band: list[dict[str, Any]] = []
    rejected_unknown_language: list[dict[str, Any]] = []
    rejected_too_long: list[dict[str, Any]] = []

    # Pass 1: the cheap gates. Survivors are held so the tokenizer only has to
    # measure records that would otherwise be written.
    survivors: list[tuple[SourceRecord, str, float]] = []
    for record in records:
        resolved_audio = resolve_audio_path(record.audio, base_dir)
        base = {
            "audio": record.audio,
            "audio_filepath": resolved_audio,
            "text": record.text,
            "source_file": record.source_file,
        }

        if record.language not in SUPPORTED_LANGUAGES:
            rejected_unknown_language.append({**base, "language": record.language})
            continue

        duration = duration_by_path.get(resolved_audio)
        if duration is None:
            rejected_no_duration.append({**base, "language": record.language})
            continue
        # Training silently drops records outside this band (the
        # min/max_duration_seconds in configs/*.yaml), so keep the counts honest.
        if not (min_duration <= duration <= max_duration):
            rejected_out_of_band.append(
                {**base, "language": record.language, "duration": duration}
            )
            continue

        survivors.append((record, resolved_audio, duration))

    # Pass 2: transcript length. Never truncated — a trimmed transcript is a
    # wrong transcript, and it would teach the model to stop mid-utterance.
    if tokenizer is not None and survivors:
        LOGGER.info("Counting transcript tokens for %d record(s)...", len(survivors))
        token_counts = count_target_tokens([r.text for r, _a, _d in survivors], tokenizer)
    else:
        token_counts = [0] * len(survivors)

    for (record, resolved_audio, duration), tokens in zip(
        survivors, token_counts, strict=True
    ):
        if tokenizer is not None and tokens > max_target_tokens:
            rejected_too_long.append(
                {
                    "audio": record.audio,
                    "audio_filepath": resolved_audio,
                    "text": record.text,
                    "source_file": record.source_file,
                    "language": record.language,
                    "duration": duration,
                    "token_count": tokens,
                }
            )
            continue

        # One file per language per split, plus a per-node suffix when sharded.
        # Merge with merge_q3asr_shards.py.
        out_path = (
            out_root
            / record.language
            / f"{record.split}_{record.language}_{corpus}{shard_suffix}.jsonl"
        )
        buckets.setdefault(out_path, []).append(
            {
                "audio_filepath": resolved_audio,
                "text": record.text,
                "duration": duration,
            }
        )

    # 4) Write outputs
    stats: dict[str, int] = {}
    for out_path, bucket_records in sorted(buckets.items()):
        written = write_manifest(out_path, bucket_records)
        stats[str(out_path)] = written
        LOGGER.info("Wrote %d record(s) -> %s", written, out_path)

    for name, pile in (
        (f"rejected_no_duration{shard_suffix}.jsonl", rejected_no_duration),
        (f"rejected_out_of_band_duration{shard_suffix}.jsonl", rejected_out_of_band),
        (f"rejected_unknown_language{shard_suffix}.jsonl", rejected_unknown_language),
        (f"rejected_too_long_transcript{shard_suffix}.jsonl", rejected_too_long),
    ):
        if pile:
            pile_path = out_root / name
            write_manifest(pile_path, pile)
            LOGGER.warning("Wrote %d rejected record(s) -> %s", len(pile), pile_path)
            stats[str(pile_path)] = len(pile)

    # Audit trail for every label the script questioned, so the relabels are
    # reviewable instead of invisible.
    audited = [
        {
            "audio_filepath": resolve_audio_path(r.audio, base_dir),
            "text": r.text,
            "language": r.language,
            "audit": r.audit,
        }
        for r in records
        if r.audit
    ]
    if audited:
        audit_path = out_root / f"language_audit{shard_suffix}.jsonl"
        write_manifest(audit_path, audited)
        LOGGER.warning("Wrote %d audited label(s) -> %s", len(audited), audit_path)
        stats[str(audit_path)] = len(audited)

    # Summary
    LOGGER.info("=" * 60)
    per_language: dict[str, int] = {}
    for record in records:
        lang = record.language or "unknown"
        per_language[lang] = per_language.get(lang, 0) + 1
    for lang in sorted(per_language):
        LOGGER.info("  %s: %d record(s)", lang, per_language[lang])
    LOGGER.info(
        "Rejected: %d without duration, %d outside the %.1f-%.1fs training "
        "band, %d with unknown language, %d over %d target tokens",
        len(rejected_no_duration),
        len(rejected_out_of_band),
        min_duration,
        max_duration,
        len(rejected_unknown_language),
        len(rejected_too_long),
        max_target_tokens,
    )
    if tokenizer is None:
        LOGGER.warning(
            "Transcript length was NOT checked (no tokenizer). Any transcript "
            "over %d tokens will be replaced at training time by a duplicate "
            "of another sample.",
            max_target_tokens,
        )
    # Completion marker, written LAST so its presence really does mean this
    # node finished. The merge step needs this because it cannot tell "node
    # still running" from "node legitimately had no ml records" by file
    # presence alone — empty per-language shards are normal for small
    # languages, a missing marker never is.
    if num_nodes > 1:
        marker = out_root / f"_shard_done.rank{node_rank}of{num_nodes}.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        tmp = marker.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "node_rank": node_rank,
                    "num_nodes": num_nodes,
                    "records": len(records),
                    "outputs": {str(k): v for k, v in stats.items()},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(tmp, marker)
        LOGGER.info("Node %d/%d done -> %s", node_rank, num_nodes, marker)

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform q3asr SFT filter manifests into per-language "
        "QASR training manifests."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=DEFAULT_INPUTS,
        help="Source JSONL files in the language/<asr_text> envelope format",
    )
    parser.add_argument(
        "--base-dir",
        default=DEFAULT_BASE_DIR,
        help="Directory that relative source audio paths resolve against",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Root output directory (language subdirs are created below it)",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Cleaning-pipeline YAML config supplying the vLLM endpoint",
    )
    parser.add_argument(
        "--corpus",
        default="q3asr",
        help="Corpus name used in output filenames (<split>_<lang>_<corpus>.jsonl)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=256,
        help="Max concurrent LLM language-identification requests",
    )
    parser.add_argument(
        "--probe-workers",
        type=int,
        default=64,
        help="Thread-pool size for wav-header duration probing",
    )
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and report the planned per-language routing, then stop: "
        "no LLM calls, no duration probing, no files written",
    )
    parser.add_argument(
        "--no-audit-labels",
        dest="audit_labels",
        action="store_false",
        help="Trust the source language labels instead of asking the LLM to "
        "adjudicate the ones their own script evidence contradicts",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=TRAIN_MIN_DURATION,
        help=f"Shortest audio kept, seconds (default {TRAIN_MIN_DURATION})",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=TRAIN_MAX_DURATION,
        help=f"Longest audio kept, seconds (default {TRAIN_MAX_DURATION})",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Model directory whose tokenizer training uses (the training "
        "config's model_name_or_path). Required unless --no-length-check or "
        "--dry-run is given, because transcript token counts cannot be "
        "estimated safely",
    )
    parser.add_argument(
        "--max-target-tokens",
        type=int,
        default=MAX_TARGET_TOKENS,
        help=f"Longest transcript kept, in tokens (default {MAX_TARGET_TOKENS}, "
        "the max_target_length every configs/*.yaml sets)",
    )
    parser.add_argument(
        "--no-length-check",
        dest="length_check",
        action="store_false",
        help="Skip the transcript token-length check. Over-length transcripts "
        "then reach training, where each is replaced by a duplicate of "
        "another sample",
    )
    parser.add_argument(
        "--node-rank",
        type=int,
        default=0,
        help="This node's index in [0, --num-nodes), matching the cleaning "
        "pipeline's convention (see scripts/run_clean_node.sh)",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=1,
        help="Total nodes sharing the work. Each node writes .rankNofM files; "
        "merge them with scripts/merge_q3asr_shards.py afterwards",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(verbose=args.verbose)

    if args.num_nodes < 1:
        LOGGER.error("--num-nodes must be at least 1")
        return 2
    if not 0 <= args.node_rank < args.num_nodes:
        LOGGER.error(
            "--node-rank must be in [0, %d), got %d", args.num_nodes, args.node_rank
        )
        return 2

    # Check every input before doing any work: iter_source_records raises
    # lazily, so a typo in the second path would only surface after the first
    # file had been parsed — and would do it as a traceback on all 8 nodes.
    missing = [p for p in args.inputs if not Path(p).is_file()]
    if missing:
        for path in missing:
            LOGGER.error("Input manifest not found: %s", path)
        return 2

    config = PipelineConfig.from_yaml(args.config)
    config.validate()

    # Loud by default: an unmeasured length budget is how over-length
    # transcripts got silently swapped for duplicates in the first place, so
    # skipping the check has to be a deliberate choice rather than an omission.
    tokenizer = None
    if args.dry_run:
        pass
    elif not args.length_check:
        LOGGER.warning(
            "--no-length-check: transcripts over %d tokens will be written to "
            "the manifests, and training will silently substitute a duplicate "
            "of another sample for each one",
            args.max_target_tokens,
        )
    elif not args.tokenizer:
        LOGGER.error(
            "--tokenizer is required: transcripts over max_target_length are "
            "not truncated by training, they are replaced by a duplicate of "
            "another record, and only the real tokenizer can tell which "
            "transcripts those are. Pass the training config's "
            "model_name_or_path, or --no-length-check to accept the loss."
        )
        return 2
    else:
        try:
            tokenizer = load_target_tokenizer(args.tokenizer)
        # Broad on purpose: whatever went wrong (missing transformers, bad path,
        # unreadable tokenizer config), the answer is to stop and say so. The
        # one outcome we must never reach is running with the check disabled.
        except Exception as exc:
            LOGGER.error("Could not load the tokenizer from %s: %s", args.tokenizer, exc)
            LOGGER.error(
                "Fix the path or pass --no-length-check; continuing without a "
                "tokenizer would quietly disable the check."
            )
            return 2

    transform(
        inputs=args.inputs,
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        config=config,
        corpus=args.corpus,
        concurrency=args.concurrency,
        probe_workers=args.probe_workers,
        dry_run=args.dry_run,
        audit_labels=args.audit_labels,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        node_rank=args.node_rank,
        num_nodes=args.num_nodes,
        tokenizer=tokenizer,
        max_target_tokens=args.max_target_tokens,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
