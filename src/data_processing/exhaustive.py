"""Exhaustive build driver: workset -> partition -> strict slices -> audit.

This module wires the foundation pieces together for stage 2 and stage 3 of
the exhaustive internal build:

    workset.prepare_rank   -- rank-local exhaustive preparation of the exact
                              internal_ingest.yaml contract (80-way parallel)
    workset.build_language + finalize_worksets -- unique-clip worksets
    workset.partition_language -- one frozen hash partition per language
    run_exhaustive_slice   -- stage 2: one (lang, part) slice: continuous
                              80-request correction -> review -> publication
    run_vet_slice          -- stage 3: LLM judge over every stage-2 accepted
                              transcript; degenerate sources are rejected and
                              excluded from the vetted output
    audit_run              -- exact accounting: every input row is accepted,
                              quarantined, rejected, or structurally excluded

Every eligible unique clip is corrected and reviewed; unresolved items stay
in durable quarantine with their full history. Nothing enters the accepted
output without a passing review, and nothing enters the vetted output
without a passing stage-3 judgment. Exit codes: 0 clean, 2 unresolved
quarantine, 1 fatal.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

import httpx

from . import workset
from .correction_scheduler import CorrectionScheduler, SchedulerConfig
from .run_state import RunOwner, SliceState, preflight_storage

LOGGER = logging.getLogger("data_processing.exhaustive")

#: Input rows claimed per durable window (enqueue advances the cursor once
#: per window; 8192 rows keeps SQLite writes rare without unbounded memory).
WINDOW = 8192

#: Terminal exit codes (see module docstring).
EXIT_OK = 0
EXIT_FATAL = 1
EXIT_QUARANTINE = 2

__all__ = [
    "EXIT_FATAL",
    "EXIT_OK",
    "EXIT_QUARANTINE",
    "audit_run",
    "run_exhaustive_slice",
    "run_vet_slice",
]


def _slice_result(ok_result: dict, item: dict, generation: int) -> dict:
    """Map one scheduler outcome to a durable terminal result."""
    if ok_result.get("ok"):
        return {
            "id": item["id"], "status": "accepted",
            "text": ok_result.get("text") or item.get("normalized_text") or "",
            "dialect": ok_result.get("dialect"),
            "confidence": ok_result.get("confidence"),
            "item": item, "generation": generation,
        }
    return {
        "id": item["id"], "status": "quarantined",
        "reason": ok_result.get("reason", "unknown"),
        "issues": ok_result.get("issues"),
        "item": item, "generation": generation,
    }


def run_exhaustive_slice(root: str | Path, *, lang: str, part: int,
                         run_id: str = "default", url: str = "http://localhost:8010/v1",
                         model: str = "corrector", concurrency: int = 80,
                         batch_size: int = 16, max_tokens: int = 12288,
                         state_dir: str | Path | None = None,
                         output_dir: str | Path | None = None,
                         transport: httpx.BaseTransport | None = None) -> dict:
    """Process one (lang, part) partition slice end to end.

    Correct -> review -> publish for every unique clip in the slice, with the
    durable cursor and per-item history in SliceState. Returns the final
    slice accounting.
    """
    root = Path(root).resolve()
    preflight_storage(root / "preflight")
    state_dir = Path(state_dir) if state_dir else root / "state"
    output_dir = Path(output_dir) if output_dir else root / "output" / lang
    fingerprint = workset._fingerprint(
        workset._read_json(root / "partitions" / f"{lang}.json"))

    with RunOwner(root / "owners" / f"slice-{lang}-p{part:04d}", run_id), \
            SliceState(state_dir, output_dir, run_id=run_id, lang=lang,
                       part=part, fingerprint=fingerprint) as state:
        cfg = SchedulerConfig(url=url, model=model, concurrency=concurrency,
                              batch_size=batch_size, max_tokens=max_tokens)
        with CorrectionScheduler(cfg, event_callback=state.event,
                                 transport=transport) as sched:
            _drain_partition(root, lang, part, state, sched)
        return _finalize_slice(state)


def _drain_partition(root: Path, lang: str, part: int,
                     state: SliceState, sched: CorrectionScheduler) -> None:
    """Stream the partition once, processing durable windows as they fill.

    On resume the cursor skips already-owned rows in one O(start) scan, so
    the whole partition is read exactly once per attempt.
    """
    window: list[dict] = []
    for index, item in enumerate(workset.iter_partition(root, lang, part,
                                                        start=state.input_cursor)):
        item.setdefault("id", item.get("row_id") or item["audio_filepath"])
        item["_index"] = index
        window.append(item)
        if len(window) < WINDOW:
            continue
        _process_window(state, sched, lang, window)
        window = []
    if window:
        _process_window(state, sched, lang, window)


def _process_window(state: SliceState, sched: CorrectionScheduler,
                    lang: str, window: list[dict]) -> None:
    """Own one window durably, then correct -> review -> publish it."""
    start = window[0]["_index"]
    state.enqueue(window, cursor=start + len(window))
    # Metadata-blocked identities never reach the LLM; they quarantine
    # with their structural reason (a retry generation can resolve them
    # if the metadata is repaired upstream).
    eligible = []
    for item in window:
        if item.get("blocked_reason"):
            state.record_result({
                "id": item["id"], "status": "quarantined",
                "reason": f"metadata:{item['blocked_reason']}",
                "item": item, "generation": 0,
            })
        else:
            eligible.append(item)

    corrected = sched.process(eligible, lang=lang, stage="correct")
    review_items = []
    for item, res in zip(eligible, corrected, strict=False):
        if not res.get("ok"):
            state.record_result(_slice_result(res, item, 0))
            continue
        item["candidate"] = res["text"]
        review_items.append(item)
    if review_items:
        reviewed = sched.process(review_items, lang=lang, stage="review")
        for item, res in zip(review_items, reviewed, strict=False):
            if res.get("ok"):
                # Review returns the (possibly minimally repaired) text.
                res["text"] = res.get("text") or item["candidate"]
            state.record_result(_slice_result(res, item, 0))
    state.publish_ready()


def _finalize_slice(state: SliceState) -> dict:
    if state.stats()["results_unpublished"]:
        state.publish_ready()
    return state.stats()


# --- stage 3: LLM judgment of every accepted transcript -----------------------


def _iter_stage2_accepted(db_path: Path, *, start: int = 0):
    """Published accepted results of a finished stage-2 slice, by id.

    Ordered by id (the results primary key) so a resumed vet pass skips the
    same prefix it already owns.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        for pos, (blob,) in enumerate(conn.execute(
                "SELECT result FROM results WHERE published=1 AND "
                "json_extract(result,'$.status')='accepted' ORDER BY id")):
            if pos < start:
                continue
            yield json.loads(blob)
    finally:
        conn.close()


def _vet_result(ok_result: dict, item: dict) -> dict:
    """Map one judge outcome to a durable terminal result.

    A judge rejection is a DECISION (status "rejected", excluded from the
    vetted manifest); an exhausted/failed judgment is an unresolved item
    (status "quarantined", retryable) -- conflating them would hide rows
    that were never actually judged.
    """
    if ok_result.get("ok"):
        return {
            "id": item["id"], "status": "accepted",
            "text": ok_result.get("text") or item.get("candidate") or "",
            "item": item, "generation": 0,
        }
    reason = ok_result.get("reason", "unknown")
    status = "quarantined" if reason in ("exhausted", "internal") else "rejected"
    return {
        "id": item["id"], "status": status, "reason": reason,
        "issues": ok_result.get("issues"),
        "item": item, "generation": 0,
    }


def run_vet_slice(root: str | Path, *, lang: str, part: int,
                   run_id: str = "default", url: str = "http://localhost:8010/v1",
                   model: str = "corrector", concurrency: int = 80,
                   batch_size: int = 16, max_tokens: int = 12288,
                   state_dir: str | Path | None = None,
                   output_dir: str | Path | None = None,
                   stage2_state_dir: str | Path | None = None,
                   transport: httpx.BaseTransport | None = None) -> dict:
    """Stage 3: LLM-judge every stage-2 accepted transcript of one slice.

    Reads the slice's published accepted results (the stage-2 database is
    the input stream), asks the judge to keep or reject each final
    transcript, and publishes kept rows to a separate vetted output root.
    Rejections are terminal ``rejected`` results -- excluded from the
    vetted manifest with their issues recorded. Run after the stage-2
    slice has completed; the durable cursor makes a crashed pass resumable
    without re-judging owned rows.
    """
    root = Path(root).resolve()
    preflight_storage(root / "preflight")
    s2_dir = Path(stage2_state_dir) if stage2_state_dir else root / "state"
    stage2_db = s2_dir / f"slice_{lang}_p{part:04d}.sqlite3"
    if not stage2_db.exists():
        raise FileNotFoundError(
            f"stage-2 state {stage2_db} not found; vet runs after the slice")
    state_dir = Path(state_dir) if state_dir else root / "state_vet"
    output_dir = Path(output_dir) if output_dir else root / "vetted" / lang
    fingerprint = workset._fingerprint(
        workset._read_json(root / "partitions" / f"{lang}.json"))

    with RunOwner(root / "owners" / f"vet-{lang}-p{part:04d}", run_id), \
            SliceState(state_dir, output_dir, run_id=run_id, lang=lang,
                       part=part, fingerprint=fingerprint) as state:
        cfg = SchedulerConfig(url=url, model=model, concurrency=concurrency,
                              batch_size=batch_size, max_tokens=max_tokens)
        with CorrectionScheduler(cfg, event_callback=state.event,
                                 transport=transport) as sched:
            _drain_stage2_accepted(stage2_db, state, sched, lang)
        if state.stats()["results_unpublished"]:
            state.publish_ready()
        return state.stats()


def _drain_stage2_accepted(stage2_db: Path, state: SliceState,
                           sched: CorrectionScheduler, lang: str) -> None:
    """Stream the slice's accepted results once, judging durable windows."""
    window: list[dict] = []
    for index, result in enumerate(_iter_stage2_accepted(
            stage2_db, start=state.input_cursor)):
        item = dict(result.get("item") or {})
        item["id"] = result["id"]
        item["candidate"] = result.get("text") \
            or item.get("normalized_text") or ""
        item["_index"] = index
        window.append(item)
        if len(window) < WINDOW:
            continue
        _vet_window(state, sched, lang, window)
        window = []
    if window:
        _vet_window(state, sched, lang, window)


def _vet_window(state: SliceState, sched: CorrectionScheduler,
                lang: str, window: list[dict]) -> None:
    """Own one window durably, judge it, publish the verdicts."""
    start = window[0]["_index"]
    state.enqueue(window, cursor=start + len(window))
    judged = sched.process(window, lang=lang, stage="judge")
    for item, res in zip(window, judged, strict=False):
        state.record_result(_vet_result(res, item))
    state.publish_ready()


def _slice_counts(db: Path) -> tuple[dict | None, int]:
    """(Published status counts, unpublished-pending count) for one db.

    ``(None, 0)`` when the database does not exist.
    """
    if not db.exists():
        return None, 0
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT json_extract(result,'$.status'), COUNT(*) "
            "FROM results WHERE published=1 GROUP BY 1").fetchall()
        pending = conn.execute(
            "SELECT COUNT(*) FROM results WHERE published=0").fetchone()[0]
    finally:
        conn.close()
    return dict(rows), pending


def audit_run(root: str | Path, *, state_root: str | Path | None = None,
              vet_state_root: str | Path | None = None) -> tuple[int, dict]:
    """Exact-coverage audit across every slice state of a run.

    Reconciles per language: partitioned unique rows must equal accepted +
    quarantined + rejected with nothing pending. ``state_root`` must match
    where the slices were run (default ``<root>/state``). When any stage-3
    vet database exists under ``vet_state_root`` (default ``<root>/state_vet``),
    every slice must have a complete vet pass: kept + rejected + vet-
    quarantined must equal that slice's stage-2 accepted count, and the
    reported ``accepted`` becomes the vetted (kept) count. Judge rejections
    are decisions, not unresolved items -- they do not affect the exit code;
    quarantined rows do. Returns (exit_code, report).
    """
    root = Path(root).resolve()
    state_root = Path(state_root) if state_root else root / "state"
    vet_root = Path(vet_state_root) if vet_state_root else root / "state_vet"
    partitions = root / "partitions"
    report: dict[str, dict] = {}
    clean = True
    any_quarantined = False
    vet_mode = any(vet_root.glob("slice_*.sqlite3"))
    for marker in sorted(partitions.glob("*.json")):
        lang = marker.stem
        plan = workset._read_json(marker)
        total = s2_accepted = kept = quarantined = rejected = 0
        for part in range(plan["nparts"]):
            # Accounting is independent of whether the slice ran: the plan's
            # committed row count is the denominator.
            total += plan["parts"][part]["rows"]
            counts, pending = _slice_counts(
                state_root / f"slice_{lang}_p{part:04d}.sqlite3")
            if counts is None:
                clean = False
                report.setdefault(lang, {})[f"part{part}"] = "missing"
                continue
            s2_accepted += counts.get("accepted", 0)
            quarantined += counts.get("quarantined", 0)
            if pending:
                clean = False
            if not vet_mode:
                continue
            vcounts, vpending = _slice_counts(
                vet_root / f"slice_{lang}_p{part:04d}.sqlite3")
            if vcounts is None:
                clean = False
                report.setdefault(lang, {})[f"part{part}"] = "vet_missing"
                continue
            kept += vcounts.get("accepted", 0)
            rejected += vcounts.get("rejected", 0)
            quarantined += vcounts.get("quarantined", 0)
            vet_accounted = (vcounts.get("accepted", 0)
                             + vcounts.get("rejected", 0)
                             + vcounts.get("quarantined", 0))
            if vpending or vet_accounted != counts.get("accepted", 0):
                clean = False
        final_accepted = kept if vet_mode else s2_accepted
        accounted = final_accepted + quarantined + rejected
        any_quarantined = any_quarantined or quarantined > 0
        report[lang] = {
            "partitioned_rows": total, "accepted": final_accepted,
            "rejected": rejected, "quarantined": quarantined,
            "exact": accounted == total,
        }
        if accounted != total:
            clean = False
    code = (EXIT_FATAL if not clean
            else EXIT_QUARANTINE if any_quarantined else EXIT_OK)
    return code, {"status": "clean" if code == EXIT_OK else "incomplete",
                  "vetted": vet_mode, "languages": report}

def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m data_processing.exhaustive",
        description="Exhaustive internal corpus build (strict, no fallbacks)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_slice = sub.add_parser("slice", help="process one (lang, part) slice")
    p_slice.add_argument("--root", required=True)
    p_slice.add_argument("--lang", required=True)
    p_slice.add_argument("--part", type=int, required=True)
    p_slice.add_argument("--run-id", default="default")
    p_slice.add_argument("--url", default="http://localhost:8010/v1")
    p_slice.add_argument("--model", default="corrector")
    p_slice.add_argument("--concurrency", type=int, default=80)
    p_slice.add_argument("--batch-size", type=int, default=16)
    p_slice.add_argument("--max-tokens", type=int, default=12288)

    p_vet = sub.add_parser(
        "vet", help="stage 3: LLM-judge one slice's accepted transcripts")
    p_vet.add_argument("--root", required=True)
    p_vet.add_argument("--lang", required=True)
    p_vet.add_argument("--part", type=int, required=True)
    p_vet.add_argument("--run-id", default="default")
    p_vet.add_argument("--url", default="http://localhost:8010/v1")
    p_vet.add_argument("--model", default="corrector")
    p_vet.add_argument("--concurrency", type=int, default=80)
    p_vet.add_argument("--batch-size", type=int, default=16)
    p_vet.add_argument("--max-tokens", type=int, default=12288)
    p_vet.add_argument("--stage2-state-root", default=None,
                       help="stage-2 state root (default <root>/state)")

    p_audit = sub.add_parser("audit", help="exact-coverage audit of a run")
    p_audit.add_argument("--root", required=True)
    p_audit.add_argument("--state-root", default=None)
    p_audit.add_argument("--vet-state-root", default=None,
                         help="stage-3 vet state root (default <root>/state_vet)")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    if args.cmd == "slice":
        stats = run_exhaustive_slice(
            args.root, lang=args.lang, part=args.part, run_id=args.run_id,
            url=args.url, model=args.model, concurrency=args.concurrency,
            batch_size=args.batch_size, max_tokens=args.max_tokens)
        LOGGER.info("slice %s p%d done: %s", args.lang, args.part, stats)
        return EXIT_OK
    if args.cmd == "vet":
        stats = run_vet_slice(
            args.root, lang=args.lang, part=args.part, run_id=args.run_id,
            url=args.url, model=args.model, concurrency=args.concurrency,
            batch_size=args.batch_size, max_tokens=args.max_tokens,
            stage2_state_dir=args.stage2_state_root)
        LOGGER.info("vet %s p%d done: %s", args.lang, args.part, stats)
        return EXIT_OK
    code, report = audit_run(args.root, state_root=args.state_root,
                             vet_state_root=args.vet_state_root)
    print(json.dumps(report, indent=1, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(_main())
