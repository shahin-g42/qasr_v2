"""Exhaustive build driver: workset -> partition -> strict slices -> audit.

This module wires the three foundation pieces together for stage 2 of the
exhaustive internal build:

    workset.prepare_rank   -- rank-local exhaustive preparation of the exact
                              internal_ingest.yaml contract (80-way parallel)
    workset.build_language + finalize_worksets -- unique-clip worksets
    workset.partition_language -- one frozen hash partition per language
    run_exhaustive_slice   -- one (lang, part) slice: continuous 80-request
                              correction -> review -> durable publication
    audit_run              -- exact accounting: every input row is accepted,
                              quarantined, or structurally excluded

Every eligible unique clip is corrected and reviewed; unresolved items stay
in durable quarantine with their full history. Nothing enters the accepted
output without a passing review. Exit codes: 0 clean, 2 unresolved
quarantine, 1 fatal.
"""

from __future__ import annotations

import argparse
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
    state_dir = Path(state_dir) if state_dir else root / "state" / f"{lang}_p{part:04d}"
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


def audit_run(root: str | Path, *, state_root: str | Path | None = None) -> tuple[int, dict]:
    """Exact-coverage audit across every slice state of a run.

    Reconciles per language: partitioned unique rows must equal accepted +
    quarantined with nothing pending. ``state_root`` must match where the
    slices were run (default ``<root>/state``). Returns (exit_code, report).
    """
    root = Path(root).resolve()
    state_root = Path(state_root) if state_root else root / "state"
    partitions = root / "partitions"
    report: dict[str, dict] = {}
    clean = True
    for marker in sorted(partitions.glob("*.json")):
        lang = marker.stem
        plan = workset._read_json(marker)
        total = 0
        accepted = quarantined = 0
        for part in range(plan["nparts"]):
            # Accounting is independent of whether the slice ran: the plan's
            # committed row count is the denominator.
            total += plan["parts"][part]["rows"]
            db = state_root / f"slice_{lang}_p{part:04d}.sqlite3"
            if not db.exists():
                clean = False
                report.setdefault(lang, {})[f"part{part}"] = "missing"
                continue
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            try:
                rows = conn.execute(
                    "SELECT json_extract(result,'$.status'), COUNT(*) "
                    "FROM results WHERE published=1 GROUP BY 1").fetchall()
                pending = conn.execute(
                    "SELECT COUNT(*) FROM results WHERE published=0").fetchone()[0]
            finally:
                conn.close()
            counts = dict(rows)
            accepted += counts.get("accepted", 0)
            quarantined += counts.get("quarantined", 0)
            if pending:
                clean = False
        accounted = accepted + quarantined
        report[lang] = {
            "partitioned_rows": total, "accepted": accepted,
            "quarantined": quarantined, "exact": accounted == total,
        }
        if accounted != total:
            clean = False
    integrity_ok = clean
    code = (EXIT_FATAL if not integrity_ok
            else EXIT_QUARANTINE if quarantined else EXIT_OK)
    return code, {"status": "clean" if code == EXIT_OK else "incomplete",
                  "languages": report}

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

    p_audit = sub.add_parser("audit", help="exact-coverage audit of a run")
    p_audit.add_argument("--root", required=True)
    p_audit.add_argument("--state-root", default=None)

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
    code, report = audit_run(args.root, state_root=args.state_root)
    import json
    print(json.dumps(report, indent=1, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(_main())
