"""Stage 1: metadata-only prepare -> sorted candidate pools, sharded by source.

This is the ``_prepare`` half of the old single-process :class:`~data_processing.build_corpus.Builder`,
lifted out so it can run on all nine nodes at once with no GPU, no LLM, no audio
bytes and no shared ledger. Each node is handed a disjoint slice of the registry
(``index % num_nodes == node_rank`` over the sorted ``(lang, source)`` list), so
the nodes write disjoint files and never coordinate.

Per source, one streaming pass
------------------------------
For every raw row: derive the identity, normalize, run the text gates, score
richness, tag the accent, and -- when ``probe_local`` is on (CLI ``--probe``;
OFF by default) -- for a local source whose metadata omits it, probe the
duration from the audio header. Survivors become :class:`~data_processing.candidate.PoolRecord`
entries, buffered and flushed to ``<pool_dir>/<lang>/<source>/part-#####.jsonl``
sorted by ``composite`` descending. That per-shard sort is what lets Stage 2 do a
best-first k-way merge across the whole language.

Duration is treated differently by locality, and the difference is the point
---------------------------------------------------------------------------
* **Local** (internal v7.6 tree, already-extracted Emilia): the duration comes
  from metadata when present; when absent the row is dropped unless
  ``probe_local`` is on, in which case a ``soundfile.info`` header probe runs
  first (slow on shared storage -- that is why it is opt-in now). A local row
  whose duration cannot be resolved is dropped: its audio is unreadable, so it
  could never be trained on anyway. The probe used to be the default; it
  closed the gap where the old builder silently discarded the ~58.6% of
  internal q3asr ``ar`` shards that carry no duration.
* **External** (Hub): metadata duration is used when present; when absent the row
  is gated in *text-only* mode and its duration stays ``None``. Stage 2 fetches
  the clip inline at selection time, learns the true duration, and applies the
  deferred duration/rate gate then. Probing here would mean fetching audio in the
  metadata stage, which defeats the whole two-phase split.

Identity is fixed once, here
----------------------------
``audio_filepath`` is written now and never changes downstream. A local row keeps
its own absolute path; an external row gets the deterministic
``<audio_root>/<lang>/<source>/<blake2b(native_id)[:16]>.flac`` from
:meth:`~data_processing.datasets.base.DatasetSpec.materialized_path`. Because the
ledger key, the batch manifest and the Stage-3 audio file all derive from this one
string, they cannot disagree.

Within a node: ``--jobs N``
---------------------------
A node's sources are independent (disjoint directories, no shared state), so
``--jobs N`` fans work out over ``N`` spawned worker processes. When the node
has at least ``N`` sources each worker takes one source; when it has fewer,
each source is split into N-way **byte-range chunks** of its files (deterministic
cuts at record boundaries, planned once in the parent) and the chunks run in
parallel -- sized for the 96-core nodes, where a single-source rank would
otherwise leave ~90 cores idle. Each chunk buffers and flushes its own shards,
shard names carry the chunk tag (``part-c03-00001.jsonl``) so they never
collide, and every shard is sorted by ``composite`` -- which is all the Stage-2
heap merge needs, so chunked output is byte-equivalent in semantics to the
serial run. Chunking applies to LOCAL_JSONL sources only (Hub streams are not
seekable) and is disabled for ``--limit`` smoke runs. Workers re-resolve their
spec from the registry by name, so loader callables never cross a process
boundary, and a crashed worker is reported like any other source error instead
of losing the node's run.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import sys
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .accent import detect
from .candidate import DEFAULT_POOL_SHARD_SIZE, PoolRecord, pool_shard_path, write_pool
from .datasets import registry
from .datasets.base import DatasetSpec, IngestStats, Kind
from .datasets.local import expand_paths
from .datasets.stream import stream_metadata
from .duration_probe import probe_duration
from .normalize import DiacriticPolicy, normalize
from .quality import QualityConfig, gate

LOGGER = logging.getLogger("data_processing.prepare")


# --- source -> node assignment ----------------------------------------------
def assign_sources(
    langs: Iterable[str],
    *,
    only_sources: Iterable[str] = (),
    include_gated: bool = True,
    node_rank: int = 0,
    num_nodes: int = 1,
) -> list[tuple[str, DatasetSpec]]:
    """The ``(lang, spec)`` pairs this node is responsible for.

    The full list is sorted by ``(lang, name)`` and sliced round-robin by rank, so
    the partition is deterministic, disjoint and complete across nodes: every
    source lands on exactly one node, and re-running the same rank reproduces the
    same slice. Sharding by *source* (not by row) keeps each node's output files
    disjoint, which is what makes Stage 1 embarrassingly parallel with no locking.
    """
    only = set(only_sources)
    pairs: list[tuple[str, DatasetSpec]] = []
    for lang in langs:
        for spec in registry.specs_for(lang, include_gated=include_gated):
            if only and spec.name not in only:
                continue
            pairs.append((lang, spec))
    pairs.sort(key=lambda pair: (pair[0], pair[1].name))
    if num_nodes <= 1:
        return pairs
    return [pair for i, pair in enumerate(pairs) if i % num_nodes == node_rank]


def _clear_source_pool(pool_dir: str | Path, lang: str, source: str) -> int:
    """Remove this source's stale shards so a re-run is authoritative.

    Streaming restarts from the beginning of a source, so a shorter second run
    (fewer rows, or a ``--limit``) would otherwise leave higher-index shards from
    the first run behind -- and Stage 2 would merge them and double-count. Only
    ``part-*.jsonl[.gz]`` directly under this source's own directory are touched;
    nothing else in the pool tree is.
    """
    src_dir = Path(pool_dir) / lang / source
    if not src_dir.is_dir():
        return 0
    removed = 0
    for path in src_dir.glob("part-*.jsonl*"):
        try:
            path.unlink()
            removed += 1
        except OSError as exc:  # pragma: no cover - permission/race, best effort
            LOGGER.warning("could not remove stale shard %s: %s", path, exc)
    return removed


# --- one row -> one PoolRecord ----------------------------------------------
def _row_to_record(
    spec: DatasetSpec,
    row: dict,
    *,
    audio_root: str | Path,
    quality: QualityConfig,
    diacritic_policy: DiacriticPolicy,
    probe_local: bool,
    counters: Counter,
) -> PoolRecord | None:
    """Gate and score one raw row, or return None with the reason counted.

    Mirrors :meth:`build_corpus.Builder._prepare` but keeps the two fields the old
    path could not: a duration that may be legitimately absent (external, deferred
    to Stage 2) and a final ``audio_filepath`` that is the materialized target for
    external clips rather than their remote reference.
    """
    native = spec.audio_filepath(row)
    if not native:
        counters["skip_no_path"] += 1
        return None
    text = spec.text_of(row)
    if not text or not text.strip():
        counters["skip_no_text"] += 1
        return None

    lang = spec.lang
    if spec.fields.lang:
        lang = str(row.get(spec.fields.lang) or lang)

    external = spec.needs_materialization
    duration = spec.duration_of(row)
    if duration is None and not external and probe_local:
        duration = probe_duration(native)
        counters["duration_probed"] += 1
    if duration is None:
        if external:
            # Not a defect: Stage 2 materializes the clip inline and gates the
            # real duration then. Counted so the report shows how much of the pool
            # is duration-deferred.
            counters["duration_deferred"] += 1
        else:
            # Local audio whose header cannot be read is unusable for training.
            counters["duration_missing"] += 1
            return None

    # Fix the identity now: external clips are named by their materialized path,
    # and ``native_id`` carries the fetch key Stage 3 needs to retrieve them.
    if external:
        audio_filepath = spec.materialized_path(audio_root, native)
        native_id = native
    else:
        audio_filepath = native
        native_id = ""

    normalized, rules = normalize(text, lang, diacritic_policy)
    if not normalized.strip():
        counters["reject_empty_after_normalize"] += 1
        return None

    result = gate(normalized, lang, duration, quality, require_duration=(duration is not None))
    if not result.ok:
        counters["reject_gate"] += 1
        counters["reason:" + result.reasons[0].split(":")[0]] += 1
        return None

    tag = detect(normalized, lang)
    # Triage (same rule the legacy builder used): only spend LLM tokens where the
    # deterministic stage cannot settle the transcript. ~18% of traffic.
    needs_llm = bool(rules) or tag.label == "unknown" or result.quality < 0.9

    metrics = {k: float(v) for k, v in result.metrics.items()}
    metrics["accent_confidence"] = round(tag.confidence, 3)
    metrics["code_switch"] = round(tag.code_switch, 4)
    return PoolRecord(
        audio_filepath=audio_filepath,
        source=spec.name,
        lang=lang,
        normalized_text=normalized,
        quality=result.quality,
        richness=result.richness,
        composite=result.composite,
        native_id=native_id,
        external=external,
        duration=duration,
        dataset=spec.origin,
        source_text=text,
        accent=tag.label,
        itn_applied="itn" in rules,
        diacritics_ratio=metrics.get("diacritics_ratio"),
        needs_llm=needs_llm,
        stages={"normalize": ",".join(rules) or "noop", "accent": tag.source},
        metrics=metrics,
    )


# --- one source -> its pool shards ------------------------------------------
def _plan_local_segments(
    spec: DatasetSpec, root: str | Path | None, n_chunks: int,
) -> dict[int, list[tuple[str, int, int]]]:
    """Cut a local source's files into ``n_chunks`` byte-range work lists.

    Runs once in the parent: every file is split into proportional byte spans
    and the spans are bucketed per chunk, so the union of all chunks covers
    every byte exactly once and the partition is deterministic. Boundaries are
    record-aligned at READ time (a worker seeks to its start and discards the
    partial line), so no pre-pass over the data is needed. Only plain
    ``.jsonl`` is chunkable; a gzipped file becomes one span for chunk 0.
    """
    files = expand_paths(spec, root)
    sizes = [(f, f.stat().st_size) for f in files]
    total = sum(size for _, size in sizes)
    if total == 0 or n_chunks <= 1:
        return {0: [(str(f), 0, size) for f, size in sizes]}
    bounds = [total * i // n_chunks for i in range(n_chunks + 1)]
    plans: dict[int, list[tuple[str, int, int]]] = {c: [] for c in range(n_chunks)}
    pos = 0
    for path, size in sizes:
        f_start, f_end = pos, pos + size
        if str(path).endswith(".gz"):
            plans[0].append((str(path), 0, size))
        else:
            for c in range(n_chunks):
                lo = max(bounds[c], f_start) - f_start
                hi = min(bounds[c + 1], f_end) - f_start
                if lo < hi:
                    plans[c].append((str(path), lo, hi))
        pos = f_end
    return {c: segs for c, segs in plans.items() if segs}


def _iter_segments(segments: list[tuple[str, int, int]]) -> Iterator[dict]:
    """Yield parsed rows from byte-range segments, record-aligned.

    Binary mode throughout: ``tell()``/``seek()`` must be exact byte offsets
    (text-mode cookies are opaque and would corrupt the boundaries). The start
    offset may land mid-line (then the partial record is discarded -- the
    previous chunk already consumed it in full) or exactly on a newline (then
    the record there belongs to THIS chunk and must not be skipped); the byte
    before the offset tells the two cases apart. The chunk ends after the last
    line that STARTS before its end offset, so adjacent chunks partition the
    rows exactly: no row twice, none skipped.
    """
    for path, start, end in segments:
        with open(path, "rb") as fh:
            if start > 0:
                fh.seek(start - 1)
                after_prev = fh.read(1)
                fh.seek(start)
                if after_prev != b"\n":
                    fh.readline()  # discard the partial record before the boundary
            while fh.tell() < end:
                line = fh.readline()
                if not line:
                    break
                row = json.loads(line.decode("utf-8"))
                if isinstance(row, dict):
                    yield row


def prepare_source(
    spec: DatasetSpec,
    *,
    pool_dir: str | Path,
    audio_root: str | Path,
    root: str | None,
    quality: QualityConfig,
    diacritic_policy: DiacriticPolicy = DiacriticPolicy.CRITICAL_ONLY,
    shard_size: int = DEFAULT_POOL_SHARD_SIZE,
    gzipped: bool = False,
    limit: int | None = None,
    token: str | None = None,
    probe_local: bool = False,
    overwrite: bool = True,
    segments: list[tuple[str, int, int]] | None = None,
    shard_tag: str = "",
) -> dict:
    """Stream one source, gate/score every row, write its sorted pool shards.

    ``probe_local`` defaults to OFF (user directive: the header probe on
    shared storage costs minutes per 100k rows and is not worth it during a
    big ingest); pass ``probe_local=True`` (CLI ``--probe``) to restore the
    header probe for local rows whose metadata omits duration -- such rows
    are DROPPED, not defaulted, when the probe is off.

    ``segments`` restricts the pass to explicit byte ranges of the source's
    files (parallel chunk workers, see ``run_prepare``); ``shard_tag`` must
    then namespace this worker's shard names so sibling chunks cannot
    collide. Both are parent-planned; a normal run passes neither.

    Returns a per-source report. Never raises for a source-level failure (a Hub
    401, a missing ``datasets`` install): the error is recorded and the caller
    moves on to the next source, so one bad spec cannot cost a node its whole
    assignment.
    """
    lang = spec.lang
    source = spec.name
    stats = IngestStats(spec_name=source)
    counters: Counter = Counter()
    if overwrite:
        counters["stale_shards_removed"] = _clear_source_pool(pool_dir, lang, source)

    buffer: list[PoolRecord] = []
    shard_index = 0
    shards: list[tuple[str, int]] = []
    read = 0

    def flush() -> None:
        nonlocal buffer, shard_index
        if not buffer:
            return
        path = pool_shard_path(pool_dir, lang, source, shard_index,
                               gzipped=gzipped, tag=shard_tag)
        shards.append((str(path), write_pool(path, buffer)))
        shard_index += 1
        buffer = []

    try:
        if segments is not None:
            rows: Iterator[dict] = _iter_segments(segments)
        else:
            rows = stream_metadata(spec, root, stats, token)
        for row in rows:
            read += 1
            rec = _row_to_record(
                spec, row, audio_root=audio_root, quality=quality,
                diacritic_policy=diacritic_policy, probe_local=probe_local,
                counters=counters,
            )
            if rec is not None:
                buffer.append(rec)
                stats.emitted += 1
                if len(buffer) >= shard_size:
                    flush()
            if limit is not None and read >= limit:
                counters["limit_reached"] += 1
                break
        flush()
    except Exception as exc:
        # A source that cannot be streamed is reported, not fatal: the remaining
        # sources on this node still prepare. Preflight (Stage 0) is where a
        # systematic failure like a bad repo_id or a missing token should surface.
        counters["stream_error"] += 1
        LOGGER.error("%s: prepare failed (%s): %s", source, type(exc).__name__, exc)
        return {
            "source": source, "lang": lang, "external": spec.needs_materialization,
            "read": read, "emitted": stats.emitted, "shards": shards,
            "error": f"{type(exc).__name__}: {exc}", "counters": dict(counters),
        }

    return {
        "source": source, "lang": lang, "external": spec.needs_materialization,
        "read": read, "emitted": stats.emitted, "shards": shards,
        "counters": dict(counters),
    }


# --- orchestration ----------------------------------------------------------
def _prepare_source_task(task: dict) -> dict:
    """ProcessPool worker: re-resolve the spec by name, prepare it, report.

    Runs in a spawned child, so it must be module-level and take only picklable
    arguments. Re-looking the spec up by name (instead of pickling the
    ``DatasetSpec``) keeps the loader callables -- and any auth state they
    closed over -- out of the pickle, and guarantees the child sees the same
    registry the parent assigned from.
    """
    logging.basicConfig(
        level=task["log_level"],
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    spec = registry.by_name(task["source"])
    return prepare_source(spec, **task["kwargs"])


def run_prepare(
    *,
    langs: Iterable[str],
    pool_dir: str | Path,
    audio_root: str | Path,
    root: str | None = None,
    quality: QualityConfig | None = None,
    diacritic_policy: DiacriticPolicy = DiacriticPolicy.CRITICAL_ONLY,
    only_sources: Iterable[str] = (),
    include_gated: bool = True,
    node_rank: int = 0,
    num_nodes: int = 1,
    shard_size: int = DEFAULT_POOL_SHARD_SIZE,
    gzipped: bool = False,
    limit: int | None = None,
    token: str | None = None,
    probe_local: bool = False,
    jobs: int = 1,
) -> dict:
    """Prepare every source assigned to this node. Returns the run report.

    ``jobs > 1`` fans work out over that many spawned worker processes: one
    source per worker while sources last, then byte-range chunks of the
    remaining sources (LOCAL_JSONL only, never under ``limit``). Chunk reports
    for the same source are merged into one entry, so the report shape is
    identical to the serial run's; a worker crash is recorded as that
    source's ``error`` rather than aborting the node.
    """
    quality = quality or QualityConfig()
    pairs = assign_sources(
        langs, only_sources=only_sources, include_gated=include_gated,
        node_rank=node_rank, num_nodes=num_nodes,
    )
    LOGGER.info(
        "prepare: node %d/%d -> %d source(s): %s",
        node_rank, num_nodes, len(pairs), ", ".join(s.name for _, s in pairs) or "(none)",
    )
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    shared = {
        "pool_dir": pool_dir, "audio_root": audio_root, "root": root,
        "quality": quality, "diacritic_policy": diacritic_policy,
        "shard_size": shard_size, "gzipped": gzipped, "limit": limit,
        "token": token, "probe_local": probe_local,
    }
    sources: list[dict] = []
    parallel = {"jobs": 1, "ok": None, "failed": 0}
    chunkable = (
        jobs > 1 and limit is None
        and any(spec.kind is Kind.LOCAL_JSONL for _, spec in pairs)
    )
    if (jobs > 1 and len(pairs) > 1) or chunkable:
        # Work items: (lang, spec, chunk_id, n_chunks, segments|None).
        # One source per worker while sources last; the leftover workers
        # become byte-range chunks of the chunkable sources.
        tasks: list[tuple[str, DatasetSpec, int, int, list | None]] = []
        if len(pairs) >= jobs or not chunkable:
            tasks = [(lang, spec, 0, 1, None) for lang, spec in pairs]
            n_workers = min(jobs, len(pairs))
        else:
            per_source = max(1, jobs // len(pairs))
            n_workers = 0
            for lang, spec in pairs:
                if spec.kind is not Kind.LOCAL_JSONL:
                    tasks.append((lang, spec, 0, 1, None))
                    n_workers += 1
                    continue
                plans = _plan_local_segments(spec, root, per_source)
                # The parent clears stale shards ONCE per source; chunk workers
                # run concurrently and must not unlink each other's output.
                _clear_source_pool(pool_dir, spec.lang, spec.name)
                for c, segs in plans.items():
                    tasks.append((lang, spec, c, len(plans), segs))
                n_workers += len(plans)
            parallel["chunks"] = per_source
        parallel["jobs"] = n_workers
        LOGGER.info("prepare: fanning %d task(s) over %d spawned worker(s)",
                    len(tasks), n_workers)
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers,
                                 mp_context=ctx) as pool:
            futures = [
                (lang, spec, n_chunks, pool.submit(_prepare_source_task, {
                    "lang": lang, "source": spec.name,
                    "log_level": LOGGER.getEffectiveLevel(),
                    "kwargs": {
                        **shared,
                        "overwrite": segments is None,
                        "segments": segments,
                        "shard_tag": f"c{chunk_id:02d}-" if n_chunks > 1 else "",
                    },
                }))
                for lang, spec, chunk_id, n_chunks, segments in tasks
            ]
            reports: dict[tuple[str, str], dict] = {}
            order: list[tuple[str, str]] = []
            for lang, spec, n_chunks, fut in futures:
                key = (lang, spec.name)
                if key not in reports:
                    reports[key] = {
                        "source": spec.name, "lang": lang,
                        "external": spec.needs_materialization,
                        "read": 0, "emitted": 0, "shards": [], "counters": {},
                    }
                    if n_chunks > 1:
                        reports[key]["chunks"] = n_chunks
                    order.append(key)
                try:
                    chunk_report = fut.result()
                except Exception as exc:  # one dead worker must not lose the node's run
                    LOGGER.error("%s: worker crashed (%s): %s", spec.name,
                                 type(exc).__name__, exc)
                    chunk_report = {
                        "source": spec.name, "lang": lang,
                        "read": 0, "emitted": 0, "shards": [],
                        "error": f"{type(exc).__name__}: {exc}",
                        "counters": {"worker_crashed": 1},
                    }
                merged = reports[key]
                merged["read"] += chunk_report["read"]
                merged["emitted"] += chunk_report["emitted"]
                merged["shards"].extend(chunk_report["shards"])
                for k, v in chunk_report["counters"].items():
                    merged["counters"][k] = merged["counters"].get(k, 0) + v
                if chunk_report.get("error"):
                    merged["error"] = chunk_report["error"]
        sources = [reports[key] for key in order]
        parallel["ok"] = sum(1 for s in sources if not s.get("error"))
        parallel["failed"] = len(sources) - parallel["ok"]
    else:
        for lang, spec in pairs:
            LOGGER.info("prepare %s (%s, %s)", spec.name, lang,
                        "external" if spec.needs_materialization else "local")
            sources.append(prepare_source(spec, **shared))

    totals = Counter()
    for src in sources:
        totals["read"] += src["read"]
        totals["emitted"] += src["emitted"]
        totals["shards"] += len(src["shards"])
        if src.get("error"):
            totals["errored_sources"] += 1
        for key, val in src["counters"].items():
            if isinstance(val, int):
                totals[key] += val
    return {
        "stage": "prepare", "started": started, "node_rank": node_rank,
        "num_nodes": num_nodes, "pool_dir": str(pool_dir), "audio_root": str(audio_root),
        "langs": list(langs), "sources": sources, "totals": dict(totals),
        "parallel": parallel,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="prepare",
        description="Stage 1: stream source metadata into sorted candidate pools.",
    )
    ap.add_argument("--pool-dir", required=True, help="root for candidate pool shards")
    ap.add_argument("--audio-root", required=True,
                    help="root external clips will be materialized under")
    ap.add_argument("--langs", help=f"comma-separated (default {','.join(registry.LANGUAGES)})")
    ap.add_argument("--root", default=registry.INTERNAL_ROOT,
                    help="root for relative local dataset patterns")
    ap.add_argument("--only", action="append", help="restrict to a registry name (repeatable)")
    ap.add_argument("--no-gated", action="store_true", help="skip sources needing Hub terms")
    ap.add_argument("--node-rank", type=int, default=0, help="this node's index (0-based)")
    ap.add_argument("--num-nodes", type=int, default=1, help="total nodes sharing the registry")
    ap.add_argument("--shard-size", type=int, default=DEFAULT_POOL_SHARD_SIZE,
                    help="records per sorted pool shard")
    ap.add_argument("--gzip", action="store_true", help="write gzipped pool shards")
    ap.add_argument("--limit", type=int, help="max rows read per source (smoke testing)")
    ap.add_argument("--probe", action="store_true",
                    help="probe local durations missing from metadata (OFF by "
                         "default: the header probe is slow on shared storage, "
                         "and unprobeable rows are dropped)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="worker processes over this node's work (spawn). More "
                         "workers than sources split each local source into "
                         "byte-range chunks; 1 = serial")
    ap.add_argument("--min-richness", type=float, default=0.0,
                    help="hard richness floor (0 disables; drops trivial transcripts)")
    ap.add_argument("--gate-duration", action="store_true",
                    help="re-arm the duration band + speech-rate gates (off by default)")
    ap.add_argument("--diacritic-policy", default=DiacriticPolicy.CRITICAL_ONLY.value,
                    choices=[p.value for p in DiacriticPolicy])
    ap.add_argument("--report", help="write the run report JSON here")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    langs = tuple(x.strip() for x in args.langs.split(",") if x.strip()) \
        if args.langs else registry.LANGUAGES
    quality = QualityConfig(min_richness=args.min_richness, gate_duration=args.gate_duration)
    report = run_prepare(
        langs=langs, pool_dir=args.pool_dir, audio_root=args.audio_root, root=args.root,
        quality=quality, diacritic_policy=DiacriticPolicy(args.diacritic_policy),
        only_sources=tuple(args.only or ()), include_gated=not args.no_gated,
        node_rank=args.node_rank, num_nodes=args.num_nodes, shard_size=args.shard_size,
        gzipped=args.gzip, limit=args.limit, token=os.environ.get("HF_TOKEN"),
        probe_local=args.probe, jobs=args.jobs,
    )
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        LOGGER.info("report written to %s", args.report)
    else:
        print(json.dumps(report["totals"], indent=2, ensure_ascii=False, default=str))
    return 0


__all__ = [
    "assign_sources",
    "main",
    "prepare_source",
    "run_prepare",
]


if __name__ == "__main__":
    sys.exit(main())
