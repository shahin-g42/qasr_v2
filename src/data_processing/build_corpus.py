"""End-to-end corpus build: sources -> normalized, gated, tagged, batched data.

One command drives the whole chain for a set of languages and emits sequential
100k-sample batches. Every stage is the module that already owns that concern:

    datasets.stream    Phase A ingest, metadata only, sources interleaved
    normalize          deterministic ITN / punctuation / diacritic policy
    quality.gate       hard gates -- duration, script ratio, char runs, rate
    accent.detect      dialect/accent tagging from lexical evidence
    llm (optional)     batched transcript correction over an OpenAI endpoint
    accent.check_erasure  did the LLM normalize the speaker's dialect away?
    distribute         audio_filepath uniqueness, source caps, batch emission

The LLM stage is optional and off by default. A run without it still produces a
complete, gated, deduplicated, accent-tagged corpus -- which is what makes the
deterministic path testable end-to-end on a laptop with no GPU, and what makes
it possible to measure how much the LLM actually changes before paying for it.

Ordering matters
----------------
Erasure is checked against the *normalized* text, not the raw source. The LLM
sees normalized input, so diffing its output against the raw source would
attribute our own deterministic transforms to the model and inflate the damage
count.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
import urllib.request
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .accent import check_erasure, detect, preservation_block, reconcile
from .canonical import Meta, Sample
from .datasets import registry
from .datasets.base import IngestStats
from .datasets.stream import ingest, to_sample
from .distribute import Candidate, DistributeConfig, Distributor, SeenLedger
from .generic_prompts import build_generic_batch_cleaner_messages
from .llm_client import parse_json_response
from .normalize import DiacriticPolicy, normalize
from .prompts import build_batch_cleaner_messages
from .quality import QualityConfig, gate

LOGGER = logging.getLogger("data_processing.build_corpus")

#: Prompt families the corrector can send. ``rich`` (the default) swaps in the
#: language-specialized cleaner prompts the legacy pipeline battle-tested on
#: the 70M-record Arabic run; ``compact`` is the short fallback the staged
#: build grew up with. Deterministic enforcement is identical in both modes.
LLM_PROMPT_MODES = ("rich", "compact")

#: Stage subcommands the dispatcher routes to ``<module>.main``. Lazy imports
#: (in :func:`_run_stage`) keep the legacy build importable even when only a
#: subset of the stage modules is installed, and avoid the assemble ->
#: build_corpus import cycle at module load.
_STAGE_MAINS: dict[str, tuple[str, str]] = {
    "preflight": ("data_processing.datasets.preflight", "main"),
    "prepare": ("data_processing.prepare", "main"),
    "assemble": ("data_processing.assemble", "main"),
    "materialize": ("data_processing.materialize", "main"),
    "bundle": ("data_processing.bundle", "main"),
}


@dataclass(slots=True)
class BuildConfig:
    """Everything one build run is allowed to vary."""

    out_dir: str = "training_manifests/v8.0"
    ledger: str = "logs/corpus_ledger.sqlite3"
    langs: tuple[str, ...] = registry.LANGUAGES
    #: Batches to emit per language. 0 means "until the sources run out".
    batches_per_lang: int = 1
    #: Root for the registry's relative local patterns.
    root: str = registry.INTERNAL_ROOT
    #: Skip Hub specs whose terms must be accepted. Enables a tokenless run.
    include_gated: bool = True
    #: Restrict to these registry names; empty means the whole registry.
    only_sources: tuple[str, ...] = ()
    gzipped: bool = False
    dry_run: bool = False
    #: Load eval manifests into the ledger as exclusions before building.
    exclude_eval: bool = True

    # --- LLM stage ----------------------------------------------------------
    llm_url: str | None = None
    llm_model: str = "corrector"
    #: Transcripts per call. Sized for the 96-core nodes: 64 concurrent calls
    #: x 16 transcripts ~= 1k in-flight against a vLLM ``max_num_seqs=512``.
    llm_batch: int = 16
    llm_concurrency: int = 64
    llm_timeout: float = 600.0
    llm_max_tokens: int = 4096
    #: Send only samples the deterministic stage flagged as needing judgement.
    #: Measured on 60k real records, this is ~18% of traffic.
    llm_triage_only: bool = True
    #: Prompt family: ``rich`` (language-specialized cleaner prompts) or
    #: ``compact`` (short fallback). See :data:`LLM_PROMPT_MODES`.
    llm_prompt: str = "rich"

    quality: QualityConfig = field(default_factory=QualityConfig)
    distribute: DistributeConfig = field(default_factory=DistributeConfig)
    diacritic_policy: DiacriticPolicy = DiacriticPolicy.CRITICAL_ONLY


class BatchCorrector:
    """Batched, threaded corrector for an OpenAI-compatible chat endpoint.

    Batching is not a nicety here. At ~33 output tokens per transcript, one call
    per sample spends most of its budget re-reading the system prompt; twelve
    samples per call amortizes it. That is the difference between the 300M-token
    projection being ~5 hours and being ~30.

    Uses only ``urllib`` and a thread pool, so the build node needs no extra
    dependency and no event loop. Thinking is disabled per request: this is a
    normalization task and chain-of-thought multiplies output tokens several-fold
    for no measured quality gain.
    """

    def __init__(self, cfg: BuildConfig) -> None:
        if cfg.llm_prompt not in LLM_PROMPT_MODES:
            raise ValueError(
                f"unknown llm_prompt {cfg.llm_prompt!r}; expected one of {LLM_PROMPT_MODES}")
        self.cfg = cfg
        self.calls = 0
        self.samples = 0
        self.failures = 0
        self.seconds = 0.0
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.llm_url)

    def _system_prompt(self, lang: str, label: str | None) -> str:
        return (
            "You correct ASR transcripts for speech-to-text training data.\n"
            "Fix spelling errors and add punctuation where it is missing.\n"
            "Apply inverse text normalization: spoken numbers, dates, times and "
            "currencies become their written form.\n"
            "NEVER add diacritics or vocalization that the source does not have.\n"
            "NEVER convert colloquial or dialectal speech into the standard register.\n"
            "NEVER paraphrase, reorder, or drop words.\n"
            + preservation_block(lang, label)
            + "\nReturn JSON: {\"items\": [{\"i\": <int>, \"text\": <str>, "
            "\"dialect\": <str>, \"confidence\": <float 0-1>}]}"
        )

    def _post(self, payload: dict) -> dict:
        url = self.cfg.llm_url.rstrip("/") + "/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        token = os.environ.get("LLM_API_KEY") or os.environ.get("HF_TOKEN")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=self.cfg.llm_timeout) as resp:
            return json.load(resp)

    def _messages(self, lang: str, label: str | None, items: list[tuple[str, str | None]]) -> list[dict]:
        """The chat messages for one batch, per prompt mode.

        ``compact`` is the short fallback: one-line system prompt plus a JSON
        list. ``rich`` swaps in the language-specialized cleaner prompts
        (Arabic via :func:`prompts.build_batch_cleaner_messages`, others via
        :func:`generic_prompts.build_generic_batch_cleaner_messages`) with the
        dialect-specific preservation block for this (lang, accent) group --
        the grouping upstream in ``_correct_chunk``/``Builder.run`` already
        guarantees one label fits the whole batch. Result contracts the parser
        understands (bare array or ``{"items": [...]}``; 0-based ``i`` or
        1-based ``index``) are honoured by both modes, so parsing, order
        alignment, the failure fallback and the stats are mode-independent.
        """
        if self.cfg.llm_prompt == "rich":
            texts = [t for t, _ in items]
            if lang == "ar":
                return build_batch_cleaner_messages(texts, accent_label=label)
            return build_generic_batch_cleaner_messages(texts, lang, accent_label=label)
        return [
            {"role": "system", "content": self._system_prompt(lang, label)},
            {"role": "user", "content": json.dumps(
                [{"i": n, "text": t} for n, (t, _) in enumerate(items)],
                ensure_ascii=False
            )},
        ]

    def _call(self, lang: str, items: list[tuple[str, str | None]]) -> list[dict]:
        """Correct one batch of ``(text, accent_label)``. Output stays in order.

        The ``i`` sent to the model is always the position **within this chunk**,
        enumerated here. It cannot be an index supplied by the caller: ``correct``
        slices the work into chunks, so a group-wide index points past the end of
        the chunk-local result list and is discarded by the bounds check below --
        silently, and for every chunk after the first.
        """
        # A single accent block cannot describe a mixed batch, so grouping is by
        # (lang, accent) upstream; the prompt names the dominant label only.
        label = items[0][1] if items else None
        payload = {
            "model": self.cfg.llm_model,
            "temperature": 0,
            "max_tokens": self.cfg.llm_max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": self._messages(lang, label, items),
        }
        t0 = time.perf_counter()
        try:
            resp = self._post(payload)
            content = resp["choices"][0]["message"]["content"]
            parsed = parse_json_response(content)
        except Exception as exc:
            # Broad on purpose. ``parse_json_response`` raises
            # LLMResponseParseError, which derives from RuntimeError -- so the
            # narrow tuple this used to catch let one truncated JSON response
            # propagate out of the thread pool and kill a multi-hour run. That
            # is precisely the outcome the "keep the normalized text" fallback
            # exists to prevent. Every failure here is recoverable per sample;
            # losing the run is not. The exception type is logged so an
            # unexpected one stays diagnosable rather than silently absorbed.
            with self._lock:
                self.failures += len(items)
            LOGGER.warning("LLM batch failed (%s: %s); keeping normalized text",
                           type(exc).__name__, exc)
            return [{"text": t, "dialect": None, "confidence": 0.0} for t, _ in items]

        with self._lock:
            self.calls += 1
            self.samples += len(items)
            self.seconds += time.perf_counter() - t0

        out = [{"text": t, "dialect": None, "confidence": 0.0} for t, _ in items]
        rows = parsed.get("items") if isinstance(parsed, dict) else parsed
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                idx = _result_index(row)
                if idx is not None and 0 <= idx < len(out):
                    out[idx] = {
                        "text": str(row.get("text") or items[idx][0]),
                        "dialect": row.get("dialect"),
                        "confidence": float(row.get("confidence") or 0.0),
                    }
        return out

    def correct(self, lang: str, items: list[tuple[str, str | None]]) -> list[dict]:
        """Correct many batches concurrently. Input and output stay index-aligned."""
        cfg = self.cfg
        chunks = [items[i:i + cfg.llm_batch] for i in range(0, len(items), cfg.llm_batch)]
        results: list[list[dict] | None] = [None] * len(chunks)
        with ThreadPoolExecutor(max_workers=cfg.llm_concurrency) as pool:
            futures = {
                pool.submit(self._call, lang, chunk): n for n, chunk in enumerate(chunks)
            }
            for fut, n in futures.items():
                results[n] = fut.result()
        return [row for chunk in results if chunk for row in chunk]

    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "prompt": self.cfg.llm_prompt,
            "calls": self.calls,
            "samples_corrected": self.samples,
            "samples_failed": self.failures,
            "seconds": round(self.seconds, 1),
            "triage_only": self.cfg.llm_triage_only,
        }


def _result_index(row: dict) -> int | None:
    """The chunk-local position a result row refers to, or None.

    The compact contract sends and asks for 0-based ``i``. The rich batch
    templates mirror the legacy Arabic prompt, which numbers transcripts
    from 1 and asks for ``index``. Accepting both lets either prompt mode
    parse without the caller knowing which was sent; a row carrying neither
    (or junk) stays None and is ignored, per the omit-keep-fallback rule.
    """
    i = row.get("i")
    if isinstance(i, int) and not isinstance(i, bool) and i >= 0:
        return i
    alt = row.get("index")
    if isinstance(alt, int) and not isinstance(alt, bool) and alt >= 1:
        return alt - 1
    return None


@dataclass(slots=True)
class StageRecord:
    """One sample mid-pipeline, carrying everything the sidecar will need."""

    sample: Sample
    meta: Meta
    source: str
    accent_label: str | None = None
    needs_llm: bool = False


class Builder:
    """Runs the stage chain for one language."""

    def __init__(self, lang: str, cfg: BuildConfig, ledger: SeenLedger, corrector: BatchCorrector) -> None:
        self.lang = lang
        self.cfg = cfg
        self.ledger = ledger
        self.corrector = corrector
        self.dist = Distributor(lang, ledger, cfg.distribute)
        self.stage_counts: Counter = Counter()
        self.reject_reasons: Counter = Counter()
        self.accent_counts: Counter = Counter()

    # --- stages -------------------------------------------------------------
    def _prepare(self, spec, row: dict, stats: IngestStats) -> StageRecord | None:
        """Ingest -> normalize -> gate -> accent tag. No LLM, no distribution."""
        mapped = to_sample(spec, row)
        if mapped is None:
            if spec.audio_filepath(row) is None:
                stats.skipped_no_path += 1
            elif spec.duration_of(row) is None:
                stats.duration_missing += 1
                self.stage_counts["skip_no_duration"] += 1
            else:
                stats.skipped_no_text += 1
            return None
        sample, meta = mapped
        meta.dataset = spec.origin

        normalized, rules = normalize(sample.text, sample.lang, self.cfg.diacritic_policy)
        meta.normalized_text = normalized
        meta.itn_applied = "itn" in rules
        meta.stages["normalize"] = ",".join(rules) or "noop"
        if not normalized.strip():
            self.stage_counts["reject_empty_after_normalize"] += 1
            return None

        result = gate(normalized, sample.lang, sample.duration, self.cfg.quality)
        meta.quality = result.quality
        meta.metrics.update({k: float(v) for k, v in result.metrics.items()})
        if not result.ok:
            meta.reject = "|".join(result.reasons)
            self.reject_reasons[result.reasons[0].split(":")[0]] += 1
            self.stage_counts["reject_gate"] += 1
            return None

        tag = detect(normalized, sample.lang)
        meta.accent = tag.label
        meta.metrics["accent_confidence"] = round(tag.confidence, 3)
        meta.metrics["code_switch"] = round(tag.code_switch, 4)
        meta.stages["accent"] = tag.source
        self.accent_counts[tag.label] += 1

        # Triage: only spend tokens where the deterministic stage cannot settle
        # it. A clean, unpunctuated-free, in-script transcript with a confident
        # accent tag has nothing left for an LLM to decide.
        needs_llm = bool(rules) or tag.label == "unknown" or result.quality < 0.9
        return StageRecord(sample=sample, meta=meta, source=spec.name,
                           accent_label=tag.label, needs_llm=needs_llm)

    def _apply_llm(self, rec: StageRecord, out: dict) -> None:
        """Fold one LLM result into the record and check it for erasure."""
        final = (out.get("text") or "").strip()
        rec.meta.stages["llm"] = "ok"
        if not final:
            rec.meta.final_text = rec.meta.normalized_text
            rec.meta.stages["llm"] = "empty_response"
            return

        rec.meta.final_text = final
        rec.meta.accent = reconcile(
            self.lang, detect(rec.meta.normalized_text or "", self.lang),
            out.get("dialect"), float(out.get("confidence") or 0.0),
        ).label

        erasure = check_erasure(rec.meta.normalized_text or "", final, self.lang)
        rec.meta.metrics["erasure_severity"] = round(erasure.severity, 3)
        rec.meta.stages["erasure"] = "ok" if erasure.ok else "damaged"
        if not erasure.ok:
            # Refuse the rewrite, keep the deterministic text. Losing a
            # correction is recoverable; training on a transcript that
            # describes speech nobody produced is not.
            self.stage_counts["erasure_reverted"] += 1
            rec.meta.final_text = rec.meta.normalized_text
            rec.meta.reject = None

    # --- driver -------------------------------------------------------------
    def run(self, specs: Iterable, limit: int | None = None) -> dict:
        """Stream every spec for this language and emit batches."""
        cfg = self.cfg
        pending: list[StageRecord] = []
        processed = 0

        def flush() -> None:
            if not pending:
                return
            if self.corrector.enabled:
                todo = [r for r in pending if r.needs_llm or not cfg.llm_triage_only]
                if todo:
                    # Group by (lang, accent) because the preservation block in
                    # the system prompt is language- and dialect-specific; a
                    # mixed batch would get one block that fits nobody.
                    by_label: dict[tuple, list[StageRecord]] = {}
                    for r in todo:
                        by_label.setdefault((r.sample.lang, r.accent_label), []).append(r)
                    for _, group in by_label.items():
                        items = [(g.meta.normalized_text or "", g.accent_label) for g in group]
                        for rec, out in zip(group, self.corrector.correct(self.lang, items), strict=False):
                            self._apply_llm(rec, out)
                corrected = {id(r) for r in todo}
                for r in pending:
                    if id(r) not in corrected:
                        r.meta.stages.setdefault("llm", "skipped_triage")
                        r.meta.final_text = r.meta.normalized_text
            else:
                for r in pending:
                    r.meta.stages["llm"] = "disabled"
                    r.meta.final_text = r.meta.normalized_text

            for rec in pending:
                text = rec.meta.final_text or rec.meta.normalized_text or rec.sample.text
                sample = Sample(
                    audio_filepath=rec.sample.audio_filepath,
                    duration=rec.sample.duration,
                    text=text,
                    lang=rec.sample.lang,
                )
                rec.meta.metrics["final_chars"] = float(len(text))
                outcome = self.dist.offer(Candidate(sample, rec.meta, rec.source))
                self.stage_counts[f"dist_{outcome.value}"] += 1
            pending.clear()

        for spec, row, stats in ingest(specs, root=cfg.root, interleave_sources=True):
            rec = self._prepare(spec, row, stats)
            processed += 1
            if rec is not None:
                pending.append(rec)
                stats.emitted += 1
            if len(pending) >= max(cfg.llm_batch * cfg.llm_concurrency, 256):
                flush()
            if cfg.batches_per_lang and len(self.dist.batches) >= cfg.batches_per_lang:
                LOGGER.info("%s: reached %d batch(es), stopping", self.lang, cfg.batches_per_lang)
                break
            if limit and processed >= limit:
                LOGGER.info("%s: hit --limit %s rows", self.lang, f"{limit:,}")
                break
        flush()

        # Only close a short trailing batch when explicitly draining the
        # sources; otherwise leave it open and report its size, so the
        # 100k-per-language contract is never silently broken by a partial
        # shard landing on disk.
        carried = self.dist.pending().size
        if carried and cfg.batches_per_lang == 0:
            self.dist.close_batch(allow_partial=True)
            carried = 0

        written = []
        if not cfg.dry_run:
            for batch in self.dist.batches:
                written.append([str(q) for q in self.dist.write(batch, cfg.out_dir)])
        return self.report(processed, written, carried)

    def report(self, processed: int, written: list, carried: int) -> dict:
        d = self.dist.report()
        return {
            "lang": self.lang,
            "rows_read": processed,
            "batches_written": len(written),
            "files": written,
            "carried_over_partial": carried,
            "stage_counts": dict(self.stage_counts),
            "reject_reasons": dict(self.reject_reasons.most_common(15)),
            "accent_distribution": dict(self.accent_counts.most_common()),
            "distributor": d,
        }


def load_config(path: str | Path) -> BuildConfig:
    """Read a YAML build config. Absent file means defaults, not an error."""
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    flat = dict(raw.get("build", raw))
    nested = {}
    for key in ("quality", "distribute"):
        if isinstance(flat.get(key), dict):
            nested[key] = flat.pop(key)
    known = set(BuildConfig.__slots__)
    unknown = set(flat) - known
    if unknown:
        LOGGER.warning("ignoring unknown config keys: %s", sorted(unknown))
    cfg = BuildConfig(**{k: v for k, v in flat.items() if k in known})
    if "quality" in nested:
        cfg.quality = QualityConfig(**nested["quality"])
    if "distribute" in nested:
        cfg.distribute = DistributeConfig(**nested["distribute"])
    if isinstance(cfg.langs, str):
        cfg.langs = tuple(cfg.langs.split(","))
    cfg.langs = tuple(cfg.langs)
    cfg.only_sources = tuple(cfg.only_sources)
    cfg.diacritic_policy = DiacriticPolicy(cfg.diacritic_policy)
    return cfg


def select_specs(cfg: BuildConfig, lang: str) -> tuple:
    specs = registry.specs_for(lang, include_gated=cfg.include_gated)
    if cfg.only_sources:
        specs = tuple(s for s in specs if s.name in cfg.only_sources)
    return specs


def _run_stage(name: str, argv: list[str]) -> int:
    """Delegate to a stage module's ``main`` with literal argv passthrough."""
    import importlib

    module, attr = _STAGE_MAINS[name]
    return getattr(importlib.import_module(module), attr)(argv)


#: The staged config the dispatcher feeds stage sections from. Anchored to the
#: repo root (not the CWD) so running from anywhere resolves the same file; a
#: missing file simply means no staged defaults.
_DEFAULT_STAGE_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "corpus.yaml"


def _stage_defaults_argv(config_path: Path, stage: str) -> list[str]:
    """A stage's YAML section as leading argv flags (CLI flags still win).

    Keys are the stage's flag names. ``true`` emits a store_true flag,
    ``false``/``null``/empty emit nothing (the parser default stands), lists
    are comma-joined (the stages' list flags are comma-separated), everything
    else is stringified. *Prepending* rather than merging is what makes
    "CLI flags still win" fall out of argparse itself: the last occurrence
    of a repeated flag is the one that counts.

    Only the stage's own section (``prepare:``, ``assemble:``, ...) is read;
    ``build:`` belongs to the legacy path and ``paths:`` to the drivers, so
    neither is ever injected here.
    """
    try:
        import yaml
    except ImportError:  # pragma: no cover - yaml is a hard dep of load_config
        LOGGER.warning("PyYAML missing; staged config defaults disabled")
        return []
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return []
    except OSError as exc:
        LOGGER.warning("could not read staged config %s: %s", config_path, exc)
        return []
    section = raw.get(stage)
    if not section:
        return []
    if not isinstance(section, dict):
        LOGGER.warning("staged config section %r is not a mapping; ignoring", stage)
        return []

    argv: list[str] = []
    for key, value in section.items():
        flag = "--" + str(key).replace("_", "-")
        if value is None or value is False or value == "":
            continue
        if value is True:
            argv.append(flag)
        elif isinstance(value, (list, tuple)):
            argv += [flag, ",".join(str(v) for v in value)]
        else:
            argv += [flag, str(value)]
    if argv:
        LOGGER.info("stage %s defaults from %s: %s", stage, config_path,
                    " ".join(argv))
    return argv


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    config_path = _DEFAULT_STAGE_CONFIG
    # A leading --config before a stage subcommand re-targets the staged
    # defaults (build_corpus --config X prepare ...). Without a following
    # subcommand, --config belongs to the legacy build parser below.
    if args[:1] == ["--config"] and len(args) >= 3 and args[2] in _STAGE_MAINS:
        config_path = Path(args[1])
        args = args[2:]
    if args and args[0] in _STAGE_MAINS:
        # Staged path: each stage keeps its own argparse, --report and exit
        # codes; the dispatcher owns routing plus the staged YAML defaults,
        # which prepend so explicit flags still win.
        return _run_stage(args[0], _stage_defaults_argv(config_path, args[0]) + args[1:])
    if not args:
        LOGGER.info(
            "no subcommand: running the legacy single-process build; the staged "
            "9-node path is 'build_corpus <preflight|prepare|assemble|"
            "materialize|bundle> [flags]'"
        )
    ap = argparse.ArgumentParser(
        prog="build_corpus",
        description="Build sequential 100k-sample-per-language corpora.",
        epilog=("stage subcommands (the 9-node path): "
                + " | ".join(_STAGE_MAINS)
                + "  (e.g. 'build_corpus assemble --help'). Staged runs read "
                "their section of configs/corpus.yaml as defaults; explicit "
                "flags win. 'build_corpus --config X prepare ...' re-targets "
                "the staged defaults; without a subcommand --config is this "
                "legacy build's config."),
    )
    ap.add_argument("--config", help="YAML build config")
    ap.add_argument("--out-dir", help=f"output root (default {BuildConfig.out_dir})")
    ap.add_argument("--ledger", help="SQLite uniqueness ledger path")
    ap.add_argument("--langs", help="comma-separated, e.g. ar,zh,ml")
    ap.add_argument("--batches", type=int, help="batches per language; 0 = drain sources")
    ap.add_argument("--root", help="root for relative local dataset patterns")
    ap.add_argument("--only", action="append", help="restrict to a registry name (repeatable)")
    ap.add_argument("--no-gated", action="store_true", help="skip sources needing Hub terms")
    ap.add_argument("--gzip", action="store_true", help="write gzipped shards")
    ap.add_argument("--dry-run", action="store_true", help="do everything but write shards")
    ap.add_argument("--limit", type=int, help="max rows read per language (smoke testing)")
    ap.add_argument("--llm-url", help="OpenAI-compatible base URL, e.g. http://h:8000/v1")
    ap.add_argument("--llm-model", help="served model name (default 'corrector')")
    ap.add_argument("--llm-batch", type=int, help="transcripts per LLM call")
    ap.add_argument("--llm-prompt", choices=LLM_PROMPT_MODES,
                    help="corrector prompt family (default: rich; YAML wins unless given)")
    ap.add_argument("--llm-all", action="store_true", help="send every sample, not just triaged ones")
    ap.add_argument("--report", help="write the run report JSON here")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    cfg = load_config(args.config) if args.config else BuildConfig()
    if args.out_dir:
        cfg.out_dir = args.out_dir
    if args.ledger:
        cfg.ledger = args.ledger
    if args.langs:
        cfg.langs = tuple(x.strip() for x in args.langs.split(",") if x.strip())
    if args.batches is not None:
        cfg.batches_per_lang = args.batches
    if args.root:
        cfg.root = args.root
    if args.only:
        cfg.only_sources = tuple(args.only)
    if args.no_gated:
        cfg.include_gated = False
    if args.gzip:
        cfg.gzipped = True
    if args.dry_run:
        cfg.dry_run = True
    if args.llm_url:
        cfg.llm_url = args.llm_url
    if args.llm_model:
        cfg.llm_model = args.llm_model
    if args.llm_batch:
        cfg.llm_batch = args.llm_batch
    if args.llm_prompt:
        cfg.llm_prompt = args.llm_prompt
    if args.llm_all:
        cfg.llm_triage_only = False
    cfg.distribute.gzipped = cfg.gzipped

    LOGGER.info("build: langs=%s batches/lang=%s out=%s llm=%s dry_run=%s",
                ",".join(cfg.langs), cfg.batches_per_lang or "drain",
                cfg.out_dir, cfg.llm_url or "disabled", cfg.dry_run)

    if cfg.dry_run and cfg.ledger != ":memory:":
        # A dry run must leave no trace on disk. offer() claims every path it
        # processes and the ledger is opened unbuffered, so each claim is an
        # immediate INSERT. Pointed at the production ledger, a dry run would
        # mark those paths seen and a later REAL run would skip them as
        # duplicates and emit nothing -- a silent, data-loss-class surprise that
        # contradicts the "--dry-run: do everything but write shards" contract.
        # In-memory SQLite: mkdir on Path(":memory:").parent == "." is a no-op
        # and the WAL pragma falls back to memory without error.
        LOGGER.info("dry-run: ledger is ephemeral (in-memory); %s left untouched", cfg.ledger)
        cfg.ledger = ":memory:"

    corrector = BatchCorrector(cfg)
    report: dict = {"config": {k: v for k, v in asdict(cfg).items()
                                if k not in ("quality", "distribute")},
                    "languages": {}, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}

    with SeenLedger(cfg.ledger, buffer_claims=False) as ledger:
        if cfg.exclude_eval:
            report["eval_exclusions"] = ledger.load_eval_exclusions(cfg.root, cfg.langs)
            LOGGER.info("eval exclusions loaded: %s", report["eval_exclusions"])
        for lang in cfg.langs:
            specs = select_specs(cfg, lang)
            if not specs:
                LOGGER.error("%s: no sources selected -- nothing to build", lang)
                report["languages"][lang] = {"error": "no sources selected"}
                continue
            LOGGER.info("%s: %d source(s): %s", lang, len(specs), ", ".join(s.name for s in specs))
            builder = Builder(lang, cfg, ledger, corrector)
            report["languages"][lang] = builder.run(specs, limit=args.limit)
        report["llm"] = corrector.stats()
        report["ledger"] = ledger.stats()

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                                     encoding="utf-8")
        LOGGER.info("report written to %s", args.report)
    else:
        print(json.dumps({k: report[k] for k in ("languages", "llm", "ledger")},
                         indent=2, ensure_ascii=False, default=str))
    return 0


__all__ = ["LLM_PROMPT_MODES", "BatchCorrector", "BuildConfig", "Builder", "StageRecord", "load_config", "main", "select_specs"]


if __name__ == "__main__":
    sys.exit(main())
