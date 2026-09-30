#!/usr/bin/env python3
"""Stage-1.5 planner: turn the measured pools into a full multi-node stage-2 layout.

Stage 2's correctness argument is "one local ledger, one writer per language" --
a ceiling of five on this corpus. ``assemble --pool-part K/N`` lifts that ceiling
without sharing a ledger: N processes slice one language's pool by
``blake2b(audio_filepath)``, each with its own ledger, and write part K of every
batch so the N parts compose one whole 100k batch. This planner decides, from
what stage 1 actually produced:

* how many slices each language gets -- N must divide the batch size and stay
  within ``--max-slices``; a language is only sliced when its measured pools
  hold at least half a batch per slice, so a thin language stays whole -- and
* which rank runs which slice (greedy least-loaded by estimated work).

Work is ``est_rows * needs_llm_fraction``, not bytes: stage-2 wall-clock is
dominated by corrector traffic, and byte-balancing puts a huge but mostly
already-clean pool (low needs_llm) on too many ranks while a smaller pool with
nearly every row flagged starves. When a language's sample yields no usable
row estimate, its byte size is the fallback weight.

It is a pure function of the pool directory: every node runs it after the
stage-1 barrier, computes the same plan, and keeps only its own rows
(``--assign-rank``). With ``--assign-rank`` the TSV for that rank goes to
stdout and the human table to stderr; without it the table goes to stdout.

Usage:
    plan_distribution.py --pool-dir DIR --langs "ar en zh hi ml" --nodes 9 \
        --json "$LOGS/plan_internal.json" --assign-rank 3
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
from pathlib import Path

LOGGER = logging.getLogger("corpus.plan_distribution")

#: Hard cap on slices per language, also the default. Divisors of the batch
#: size that are <= this are the candidate slice counts (100_000 gives
#: 1, 2, 4, 5, 8). Beyond a handful of slices a language buys little: every
#: slice re-reads the whole pool, and slice imbalance can strand a round.
DEFAULT_MAX_SLICES = 8
#: Pool rows sampled per language for the row-size / LLM-traffic estimates.
DEFAULT_SAMPLE_ROWS = 400


def divisors(n: int) -> list[int]:
    """All divisors of ``n``, ascending."""
    return [d for d in range(1, n + 1) if n % d == 0]


def work_units(info: dict) -> float:
    """Estimated stage-2 effort for one language's pool.

    Corrector calls dominate the run, so the honest currency is flagged rows
    (``est_rows * needs_llm_fraction``). Bytes are the fallback when the pool
    sample produced no row estimate.
    """
    rows = info.get("est_rows")
    frac = info.get("needs_llm_fraction")
    if rows and frac is not None:
        return float(rows) * float(frac)
    return float(info.get("bytes") or 0.0)


def _shard_files(pool_dir: Path, lang: str) -> list[Path]:
    base = pool_dir / lang
    if not base.is_dir():
        return []
    return sorted(p for p in base.rglob("part-*.jsonl*") if p.is_file())


def _sample_lines(files: list[Path], want: int) -> list[str]:
    """Up to ``want`` non-empty lines, spread across the first files.

    Spread, not from the first file only, so the needs_llm estimate is not
    dominated by a single source. A short read of one file degrades the sample
    size, never the caller.
    """
    picked = files[:64]
    per_file = max(4, want // max(1, len(picked)))
    lines: list[str] = []
    for path in picked:
        opener = gzip.open if path.suffix == ".gz" else open
        taken = 0
        try:
            with opener(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    lines.append(line)
                    taken += 1
                    if taken >= per_file or len(lines) >= want:
                        break
        except OSError as exc:
            LOGGER.warning("cannot sample %s: %s", path, exc)
        if len(lines) >= want:
            break
    return lines


def inventory(pool_dir: str | Path, langs: list[str], sample_rows: int) -> dict[str, dict]:
    """Per-language pool inventory: shards, bytes, and sampled row statistics."""
    pool_dir = Path(pool_dir)
    out: dict[str, dict] = {}
    for lang in langs:
        files = _shard_files(pool_dir, lang)
        sources: dict[str, dict[str, int]] = {}
        for path in files:
            source = sources.setdefault(path.parent.name, {"shards": 0, "bytes": 0})
            source["shards"] += 1
            source["bytes"] += path.stat().st_size
        total_bytes = sum(s["bytes"] for s in sources.values())
        lines = _sample_lines(files, sample_rows)
        mean_row = (sum(len(ln) + 1 for ln in lines) / len(lines)) if lines else None
        est_rows = int(total_bytes / mean_row) if mean_row else None
        parsed = flagged = 0
        for line in lines:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            parsed += 1
            if rec.get("needs_llm"):
                flagged += 1
        out[lang] = {
            "shards": len(files), "bytes": total_bytes,
            "sources": sources,
            "mean_row_bytes": round(mean_row, 1) if mean_row else None,
            "est_rows": est_rows,
            "needs_llm_fraction": round(flagged / parsed, 4) if parsed else None,
        }
    return out


def plan_slices(
    inv: dict[str, dict], *, nodes: int, batch_size: int, max_slices: int,
) -> dict[str, int]:
    """Greedy slice allocation: heavier languages get more slices, up to budget.

    Every language starts at one slice (that is today's whole-language build).
    Each step hands the next divisor step to the language with the largest
    ``work / next_count`` ratio, while ``nodes`` allows it. A language with no
    pool shards is planned at zero slices.
    """
    candidate_divs = [d for d in divisors(batch_size) if d <= max_slices]
    allowed: dict[str, list[int]] = {}
    for lang, info in inv.items():
        if not info["shards"]:
            allowed[lang] = []
            continue
        # A slice below half a batch of measured material would almost surely
        # end as a short part (and then a short, unshippable label).
        margin = batch_size // 2
        allowed[lang] = [d for d in candidate_divs
                         if d == 1 or (info["est_rows"] and info["est_rows"] >= d * margin)]

    chosen = {lang: (1 if allowed.get(lang) else 0) for lang in inv}
    budget = nodes - sum(chosen.values())
    while budget > 0:
        best: tuple[tuple, str, int] | None = None
        for lang, info in inv.items():
            current = chosen[lang]
            step = next((d for d in allowed.get(lang, []) if d > current), None)
            if step is None or step - current > budget:
                continue
            lang_work = work_units(info)
            score = (lang_work / step, lang_work, lang)
            if best is None or score > best[0]:
                best = (score, lang, step)
        if best is None:
            break
        _, lang, step = best
        budget -= step - chosen[lang]
        chosen[lang] = step
    return chosen


def assign_slices(
    inv: dict[str, dict], slices_by_lang: dict[str, int], *, nodes: int, batch_size: int,
) -> list[dict]:
    """Lay the slices out over the ranks: heaviest first, greedy least-loaded."""
    slices = []
    for lang, count in slices_by_lang.items():
        if count < 1:
            continue
        info = inv[lang]
        weight = work_units(info) / count
        for part in range(count):
            slices.append({
                "lang": lang, "part": part, "slices": count,
                "weight": weight,
                "est_rows": (info["est_rows"] // count) if info["est_rows"] else None,
                "batch_size": batch_size // count,
            })
    slices.sort(key=lambda s: (-s["weight"], s["lang"], s["part"]))
    loads = [0.0] * nodes
    for entry in slices:
        rank = min(range(nodes), key=lambda i: (loads[i], i))
        entry["rank"] = rank
        loads[rank] += entry["weight"]
    return slices


def build_payload(
    inv: dict[str, dict], slices: list[dict], *, nodes: int, batch_size: int,
) -> dict:
    per_lang = {}
    for lang, info in inv.items():
        parts = [s for s in slices if s["lang"] == lang]
        per_lang[lang] = {
            "shards": info["shards"], "bytes": info["bytes"], "sources": info["sources"],
            "est_rows": info["est_rows"], "needs_llm_fraction": info["needs_llm_fraction"],
            "slices": len(parts),
            "slice_batch_size": (batch_size // len(parts)) if parts else None,
        }
    loads = [0.0] * nodes
    for entry in slices:
        loads[entry["rank"]] += entry["weight"]
    return {
        "nodes": nodes, "batch_size": batch_size,
        "languages": per_lang,
        "totals": {
            "slices": len(slices),
            "idle_ranks": nodes - len({s["rank"] for s in slices}),
            "bytes": sum(info["bytes"] for info in inv.values()),
            "est_rows": sum(info["est_rows"] or 0 for info in inv.values()),
        },
        "assignment": [{"rank": s["rank"], "lang": s["lang"], "part": s["part"],
                        "slices": s["slices"], "batch_size": s["batch_size"]}
                       for s in slices],
        "rank_load_work": [round(x, 1) for x in loads],
    }


def _work(n: float) -> str:
    """Work units are flagged rows (or bytes when rows are unknown)."""
    return f"{n / 1e6:.1f}M"


def describe(inv: dict[str, dict], slices: list[dict], *, nodes: int, batch_size: int) -> str:
    lines = [f"nodes={nodes} batch_size={batch_size} slices={len(slices)}"]
    for lang, info in inv.items():
        parts = [s for s in slices if s["lang"] == lang]
        llm = (f"{info['needs_llm_fraction']:.0%}"
               if info["needs_llm_fraction"] is not None else "?")
        lines.append(
            f"  {lang}: {info['shards']} shard(s) from {len(info['sources'])} source(s), "
            f"{info['bytes'] / 1048576:.1f} MiB, est {info['est_rows'] or 0:,} rows, "
            f"needs_llm {llm} -> {len(parts)} slice(s)"
            + (f", batch_size {batch_size // len(parts)}" if parts else " (skipped)"))
    for rank in range(nodes):
        mine = [s for s in slices if s["rank"] == rank]
        load = _work(sum(s["weight"] for s in mine))
        what = ", ".join(f"{s['lang']} p{s['part']}/{s['slices']}" for s in mine)
        lines.append(f"  rank {rank}: {load} work  {what or '(idle)'}")
    return "\n".join(lines)


def _write_json(path: str | Path, payload: dict) -> None:
    """Atomic write; every rank races here with identical content, so the
    temp name carries the pid and a lost race is only a warning."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        os.replace(tmp, path)
    except OSError as exc:  # a concurrent rank moved its copy in first; content is identical
        LOGGER.warning("could not move %s into place: %s", tmp, exc)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="plan_distribution",
        description="Plan the multi-node stage-2 slice layout from the stage-1 pools.",
    )
    ap.add_argument("--pool-dir", required=True, help="Stage-1 candidate pool root")
    ap.add_argument("--langs", required=True,
                    help="space- or comma-separated languages to plan for")
    ap.add_argument("--nodes", type=int, required=True, help="ranks available for stage 2")
    ap.add_argument("--batch-size", type=int, default=100_000,
                    help="rows per whole batch (slice counts must divide it)")
    ap.add_argument("--max-slices", type=int, default=DEFAULT_MAX_SLICES,
                    help=f"cap on slices per language (default {DEFAULT_MAX_SLICES})")
    ap.add_argument("--sample-rows", type=int, default=DEFAULT_SAMPLE_ROWS,
                    help="pool rows sampled per language for the size estimates")
    ap.add_argument("--assign-rank", type=int,
                    help="print only this rank's TSV lines (lang, part, slices) to stdout")
    ap.add_argument("--json", dest="json_path", help="write the full plan JSON here")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    langs = [x for x in args.langs.replace(",", " ").split() if x]
    if not langs:
        ap.error("--langs is empty")
    if args.nodes < 1:
        ap.error(f"--nodes needs >= 1, got {args.nodes}")
    if args.max_slices < 1:
        ap.error(f"--max-slices needs >= 1, got {args.max_slices}")
    if args.assign_rank is not None and not 0 <= args.assign_rank < args.nodes:
        ap.error(f"--assign-rank needs 0 <= R < {args.nodes}, got {args.assign_rank}")

    inv = inventory(args.pool_dir, langs, args.sample_rows)
    for lang, info in inv.items():
        if not info["shards"]:
            LOGGER.warning("%s: no pool shards under %s -- skipped", lang, args.pool_dir)
    if not any(info["shards"] for info in inv.values()):
        LOGGER.error("no pool shards under %s for langs %s", args.pool_dir, ",".join(langs))
        return 2

    slices_by_lang = plan_slices(inv, nodes=args.nodes, batch_size=args.batch_size,
                                 max_slices=args.max_slices)
    slices = assign_slices(inv, slices_by_lang, nodes=args.nodes, batch_size=args.batch_size)
    payload = build_payload(inv, slices, nodes=args.nodes, batch_size=args.batch_size)
    if args.json_path:
        _write_json(args.json_path, payload)
        LOGGER.info("plan written to %s", args.json_path)

    table = describe(inv, slices, nodes=args.nodes, batch_size=args.batch_size)
    if args.assign_rank is None:
        print(table)
    else:
        print(table, file=sys.stderr)
        for entry in slices:
            if entry["rank"] == args.assign_rank:
                print(f"{entry['lang']}\t{entry['part']}\t{entry['slices']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
