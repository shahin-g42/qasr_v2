"""Stage 2: assemble + correct one language's pool into sequential 100k batches.

One process per language, co-located with that node's corrector at
``http://localhost:8010/v1``. It reads the language's Stage-1 candidate pools,
walks them **best-first** (``composite = quality x richness`` descending, via a
k-way merge), and fills the sequential batches ``b0000``, ``b0001``, ... The
per-source cap, the ``max_per_text`` budget and the diversity floor all live in
:class:`~data_processing.distribute.Distributor`, unchanged -- best-first ordering
changes *which* samples fill a batch, not the rules that admit them.

Why the LLM stays co-located and the ledger stays local
-------------------------------------------------------
Every node serves ``corrector`` on ``localhost:8010``, so the correction stage
runs where the model is; there is no load balancer and no multi-URL pool. And
:class:`~data_processing.distribute.SeenLedger` is SQLite-WAL, which is unsafe with
nine concurrent writers on Lustre/NFS -- so each language gets its own local
ledger (the languages have disjoint audio pools), and cross-language overlap is
audited to zero afterwards rather than prevented by a shared lock.

Inline materialization, bounded over-fetch
---------------------------------------
A batch manifest must carry a true ``duration``, and for an external clip that
number does not exist until the audio is decoded. So each external candidate is
materialized to its fixed 16 kHz mono path *at the moment it is selected* -- after
: meth:`Distributor.would_place` confirms it will actually land, so a capped or
duplicate candidate is never downloaded. The deferred duration/rate gate (the one
Stage 1 ran in text-only mode) is applied against the real duration before the
sample is offered.

Fetching serially inside the placement loop is correct but slow: ``en`` is
external-only (100k clips per batch) and zh/hi are partly external, so the
round-trip dominates wall-clock. ``--fetch-workers`` (default 16) therefore
opens a *windowed* prefetch: as records enter a chunk, external candidates that
``would_place`` submits to a thread pool, and placement later ``future.result()``s
the already-decoded clip. The window is bounded by ``fetch_workers`` (and by the
chunk size), so "zero over-fetch" degrades only to "bounded over-fetch of exactly
the clips the next batches need" -- a cap filling mid-flight leaves the fetched
file on disk, where it is idempotent and wanted by a later batch. ``offer``
re-checks admission on the true duration, so a prefetched clip is never placed
on stale information.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .accent import check_erasure, detect, reconcile
from .build_corpus import LLM_PROMPT_MODES, BatchCorrector, BuildConfig
from .candidate import PoolRecord, iter_pool_shards, merge_pools
from .canonical import Meta, Sample
from .datasets import registry
from .distribute import Candidate, DistributeConfig, Distributor, SeenLedger
from .materialize import DEFAULT_SAMPLE_RATE, Fetcher, MaterializeError, Materializer
from .quality import QualityConfig, gate

LOGGER = logging.getLogger("data_processing.assemble")


@dataclass
class _Item:
    """One pool record plus the sidecar being built for it.

    ``meta`` starts as the Stage-1 sidecar (``PoolRecord.to_meta``) and accrues
    the LLM/erasure stages here, so a placed sample carries the same provenance
    the legacy single-process builder produced.
    """

    rec: PoolRecord
    meta: Meta


# --- LLM correction (reuses BatchCorrector unchanged) -----------------------
def _apply_llm(meta: Meta, out: dict, lang: str, counters: Counter) -> None:
    """Fold one corrector result into the sidecar and guard against erasure.

    Mirrors :meth:`build_corpus.Builder._apply_llm`: erasure is measured against
    the *normalized* text (what the model saw), and a dialect-erasing rewrite is
    reverted to that normalized text rather than shipped.
    """
    final = (out.get("text") or "").strip()
    meta.stages["llm"] = "ok"
    if not final:
        meta.final_text = meta.normalized_text
        meta.stages["llm"] = "empty_response"
        return

    meta.final_text = final
    meta.accent = reconcile(
        lang, detect(meta.normalized_text or "", lang),
        out.get("dialect"), float(out.get("confidence") or 0.0),
    ).label

    erasure = check_erasure(meta.normalized_text or "", final, lang)
    meta.metrics["erasure_severity"] = round(erasure.severity, 3)
    meta.stages["erasure"] = "ok" if erasure.ok else "damaged"
    if not erasure.ok:
        counters["erasure_reverted"] += 1
        meta.final_text = meta.normalized_text


def _correct_chunk(items: list[_Item], corrector: BatchCorrector, lang: str, counters: Counter) -> None:
    """LLM-correct the triaged items in one chunk, batched and grouped by accent.

    Grouping by ``(lang, accent)`` is required because the preservation block in
    the system prompt is dialect-specific; a mixed batch would get one block that
    fits nobody. Items the triage does not select keep the normalized text.
    """
    if not corrector.enabled:
        for it in items:
            it.meta.stages["llm"] = "disabled"
            it.meta.final_text = it.meta.normalized_text
        return

    todo = [it for it in items if it.rec.needs_llm or not corrector.cfg.llm_triage_only]
    if todo:
        by_label: dict[tuple, list[_Item]] = {}
        for it in todo:
            by_label.setdefault((it.rec.lang, it.rec.accent), []).append(it)
        for _, group in by_label.items():
            payload = [(g.meta.normalized_text or "", g.rec.accent) for g in group]
            for it, out in zip(group, corrector.correct(lang, payload), strict=False):
                _apply_llm(it.meta, out, lang, counters)
    corrected = {id(it) for it in todo}
    for it in items:
        if id(it) not in corrected:
            it.meta.stages.setdefault("llm", "skipped_triage")
            it.meta.final_text = it.meta.normalized_text


# --- placement --------------------------------------------------------------
def _place_chunk(
    items: list[_Item],
    dist: Distributor,
    materializer: Materializer | None,
    specs_by_name: dict[str, object],
    lang: str,
    quality: QualityConfig,
    counters: Counter,
    batches: int,
    prefetch: dict[str, Future] | None = None,
) -> bool:
    """Offer one chunk best-first. Returns True when the batch target is met.

    For each candidate: skip unless the distributor would place it (so external
    audio is fetched only for samples that will land), materialize external clips
    inline to learn the true duration, re-run the deferred duration/rate gate on
    it, then offer. The distributor re-checks admission, so a placeable candidate
    is always accepted here (single process -- nothing changes in between).
    ``prefetch`` maps ``audio_filepath`` to an in-flight fetch submitted when the
    record entered the chunk; the awaited result is used instead of a fresh
    materialize. A record with no future (window miss, mode off) falls back to
    the serial inline path.
    """
    for it in items:
        if batches and len(dist.batches) >= batches:
            return True
        rec, meta = it.rec, it.meta
        if rec.lang != lang:  # defensive: a stray multi-lang row would raise in offer
            counters["skip_lang_mismatch"] += 1
            continue
        text = (meta.final_text or meta.normalized_text or "").strip()
        if not text:
            counters["skip_empty_final"] += 1
            continue
        if not dist.would_place(rec.audio_filepath, rec.source, text):
            counters["skipped_not_placeable"] += 1
            continue

        duration = rec.duration
        if rec.external:
            if materializer is None:
                counters["skip_external_unmaterialized"] += 1
                continue
            fut = prefetch.get(rec.audio_filepath) if prefetch else None
            try:
                if fut is not None:
                    res = fut.result()
                else:
                    res = materializer.materialize(
                        rec.audio_filepath, rec.native_id, spec=specs_by_name.get(rec.source),
                        expected_duration=None, source=rec.source,
                    )
            except MaterializeError as exc:
                counters["materialize_failed"] += 1
                LOGGER.debug("materialize failed for %s: %s", rec.audio_filepath, exc)
                continue
            duration = res.duration
            counters["materialized"] += 1
            # Deferred gate: this is the first moment the clip's true length is
            # known. When duration gating is armed this applies the band +
            # speech-rate checks Stage 1 had to skip in text-only mode; with the
            # default (gates off) it re-checks only the corrected text, so an
            # out-of-band clip is still materialized and placed.
            deferred = gate(text, lang, duration, quality, require_duration=True)
            if not deferred.ok:
                counters["reject_deferred_gate"] += 1
                continue

        if duration is None:
            counters["skip_no_duration"] += 1
            continue

        meta.final_text = text
        meta.metrics["final_chars"] = float(len(text))
        sample = Sample(audio_filepath=rec.audio_filepath, duration=float(duration),
                        text=text, lang=lang)
        outcome = dist.offer(Candidate(sample, meta, rec.source))
        counters[f"dist_{outcome.value}"] += 1
    return False


# --- orchestration ----------------------------------------------------------
def run_assemble(
    *,
    lang: str,
    pool_dir: str | Path,
    out_dir: str | Path,
    ledger: str | Path,
    audio_root: str | Path,
    quality: QualityConfig | None = None,
    distribute: DistributeConfig | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    materialize_external: bool = True,
    fetcher: Fetcher | None = None,
    llm_url: str | None = None,
    llm_model: str = "corrector",
    llm_prompt: str = "rich",
    llm_batch: int = 16,
    llm_concurrency: int = 64,
    llm_timeout: float = 600.0,
    llm_max_tokens: int = 4096,
    llm_triage_only: bool = True,
    batches: int = 1,
    chunk_size: int | None = None,
    fetch_workers: int = 16,
    exclude_eval: bool = False,
    root: str | None = None,
    include_gated: bool = True,
    gzipped: bool = False,
    dry_run: bool = False,
) -> dict:
    """Assemble one language's pool into up to ``batches`` sequential batches.

    ``batches=0`` drains the pool (and writes a short trailing batch). Batch
    numbering continues across runs via the ledger's ``next_index``.
    """
    quality = quality or QualityConfig()
    distribute = distribute or DistributeConfig()
    if gzipped:
        distribute.gzipped = True

    corrector = BatchCorrector(BuildConfig(
        llm_url=llm_url, llm_model=llm_model, llm_prompt=llm_prompt, llm_batch=llm_batch,
        llm_concurrency=llm_concurrency, llm_timeout=llm_timeout,
        llm_max_tokens=llm_max_tokens, llm_triage_only=llm_triage_only,
    ))
    materializer = Materializer(audio_root, sample_rate, fetcher=fetcher) \
        if materialize_external else None
    specs_by_name = {s.name: s for s in registry.specs_for(lang, include_gated=include_gated)}
    if chunk_size is None:
        chunk_size = max(llm_batch * llm_concurrency, 256)

    shards = iter_pool_shards(pool_dir, lang)
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    if not shards:
        LOGGER.error("%s: no pool shards under %s -- run prepare first", lang, pool_dir)
        return {"stage": "assemble", "lang": lang, "started": started,
                "error": "no pool shards", "pool_dir": str(pool_dir)}

    ledger_path = ":memory:" if dry_run else str(ledger)
    counters: Counter = Counter()
    read = 0
    with SeenLedger(ledger_path, buffer_claims=False) as led:
        eval_exclusions: dict = {}
        if exclude_eval and root:
            eval_exclusions = led.load_eval_exclusions(root, [lang])
            LOGGER.info("%s: eval exclusions loaded: %s", lang, eval_exclusions)
        dist = Distributor(lang, led, distribute)
        LOGGER.info("%s: %d pool shard(s), first batch %s, llm=%s",
                    lang, len(shards), dist.pending().label, llm_url or "disabled")

        # Windowed prefetch of external clips (see module docstring). Submit
        # happens at chunk-build time -- before LLM correction -- so the fetch
        # overlaps the corrector's latency; placement later re-checks admission
        # on the true duration, and a clip whose admission changed in flight is
        # simply left on disk for the next batch (idempotent, best-first pools).
        pool = ThreadPoolExecutor(
            max_workers=fetch_workers, thread_name_prefix="assemble-fetch",
        ) if materializer is not None and fetch_workers > 0 else None
        prefetch: dict[str, Future] = {}

        chunk: list[_Item] = []
        stopped = False
        for rec in merge_pools(shards):
            read += 1
            chunk.append(_Item(rec, rec.to_meta()))
            if pool is not None and rec.external:
                text = (rec.normalized_text or "").strip()
                if text and rec.lang == lang and dist.would_place(
                        rec.audio_filepath, rec.source, text):
                    prefetch[rec.audio_filepath] = pool.submit(
                        materializer.materialize, rec.audio_filepath, rec.native_id,
                        specs_by_name.get(rec.source), None, rec.source,
                    )
                    counters["prefetched"] += 1
            if len(chunk) >= chunk_size:
                _correct_chunk(chunk, corrector, lang, counters)
                stopped = _place_chunk(chunk, dist, materializer, specs_by_name,
                                       lang, quality, counters, batches, prefetch)
                chunk.clear()
                if stopped:
                    break
        if chunk and not stopped:
            _correct_chunk(chunk, corrector, lang, counters)
            _place_chunk(chunk, dist, materializer, specs_by_name, lang, quality,
                         counters, batches, prefetch)
            chunk.clear()
        if pool is not None:
            # Outstanding fetches (submitted, never placed) still run to
            # completion: their bytes are idempotent and wanted by a later batch.
            pool.shutdown(wait=True)
            prefetch.clear()

        carried = dist.pending().size
        if carried and batches == 0:
            # Draining: a short trailing batch is wanted, so close and write it.
            dist.close_batch(allow_partial=True)
            carried = 0
        elif carried and not dry_run:
            # Stopped on the batch target with an overflow tail open. Those paths
            # were claimed but will not be written, so release them -- otherwise a
            # future run sees them as duplicates and they are stranded forever.
            led.release_batch(lang, dist.pending().label)
            counters["released_pending"] = carried
            carried = 0

        written: list[list[str]] = []
        if not dry_run:
            for batch in dist.batches:
                written.append([str(p) for p in dist.write(batch, out_dir)])

        report = {
            "stage": "assemble", "lang": lang, "started": started,
            "rows_read": read, "batches_written": len(written), "files": written,
            "batch_labels": [b.label for b in dist.batches],
            "batch_hours": {b.label: round(b.hours(), 2) for b in dist.batches},
            "carried_over_partial": carried, "counters": dict(counters),
            "fetch_workers": fetch_workers,
            "eval_exclusions": eval_exclusions, "distributor": dist.report(),
            "llm": corrector.stats(), "ledger": led.stats(),
            "dry_run": dry_run, "out_dir": str(out_dir),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="assemble",
        description="Stage 2: assemble + correct one language's pool into batches.",
    )
    ap.add_argument("--lang", required=True, help="language to assemble (one process per lang)")
    ap.add_argument("--pool-dir", required=True, help="Stage-1 candidate pool root")
    ap.add_argument("--out-dir", required=True, help="batch manifest root (holds <lang>/)")
    ap.add_argument("--ledger", required=True, help="this language's local SQLite ledger")
    ap.add_argument("--audio-root", required=True, help="root external clips materialize under")
    ap.add_argument("--root", help="internal tree root (eval exclusions + registry)")
    ap.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    ap.add_argument("--batches", type=int, default=1, help="batches to emit; 0 = drain pool")
    ap.add_argument("--batch-size", type=int, default=100_000)
    ap.add_argument("--chunk-size", type=int, help="records corrected/placed per chunk")
    ap.add_argument("--llm-url", help="local corrector base URL, e.g. http://localhost:8010/v1")
    ap.add_argument("--llm-model", default="corrector")
    ap.add_argument("--llm-prompt", choices=LLM_PROMPT_MODES, default="rich",
                    help="corrector prompt family (default rich: language-specialized cleaner prompts)")
    ap.add_argument("--llm-batch", type=int, default=16)
    ap.add_argument("--llm-concurrency", type=int, default=64)
    ap.add_argument("--llm-all", action="store_true", help="send every sample, not just triaged")
    ap.add_argument("--fetch-workers", type=int, default=16,
                    help="parallel external-audio prefetch threads (0 = serial inline fetch)")
    ap.add_argument("--no-materialize", action="store_true", help="do not fetch external audio")
    ap.add_argument("--exclude-eval", action="store_true", help="load eval manifests as exclusions")
    ap.add_argument("--no-gated", action="store_true")
    ap.add_argument("--gzip", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="in-memory ledger, write nothing")
    ap.add_argument("--min-richness", type=float, default=0.0)
    ap.add_argument("--gate-duration", action="store_true",
                    help="re-arm the deferred duration band + speech-rate gate (off by default)")
    ap.add_argument("--max-per-source-fraction", type=float, default=0.40)
    ap.add_argument("--max-per-text", type=int, default=2)
    ap.add_argument("--min-distinct-text-fraction", type=float, default=0.50)
    ap.add_argument("--report", help="write the run report JSON here")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    distribute = DistributeConfig(
        batch_size=args.batch_size, max_per_source_fraction=args.max_per_source_fraction,
        max_per_text=args.max_per_text,
        min_distinct_text_fraction=args.min_distinct_text_fraction, gzipped=args.gzip,
    )
    report = run_assemble(
        lang=args.lang, pool_dir=args.pool_dir, out_dir=args.out_dir, ledger=args.ledger,
        audio_root=args.audio_root, root=args.root,
        quality=QualityConfig(min_richness=args.min_richness, gate_duration=args.gate_duration),
        distribute=distribute, sample_rate=args.sample_rate,
        materialize_external=not args.no_materialize,
        llm_url=args.llm_url, llm_model=args.llm_model, llm_prompt=args.llm_prompt,
        llm_batch=args.llm_batch, llm_concurrency=args.llm_concurrency,
        llm_triage_only=not args.llm_all,
        batches=args.batches, chunk_size=args.chunk_size, exclude_eval=args.exclude_eval,
        include_gated=not args.no_gated, gzipped=args.gzip, dry_run=args.dry_run,
        fetch_workers=args.fetch_workers,
        fetcher=None,
    )
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        LOGGER.info("report written to %s", args.report)
    else:
        print(json.dumps({k: report.get(k) for k in
                          ("lang", "rows_read", "batches_written", "batch_labels",
                           "counters", "distributor", "llm")},
                         indent=2, ensure_ascii=False, default=str))
    return 0


__all__ = ["main", "run_assemble"]


if __name__ == "__main__":
    sys.exit(main())
