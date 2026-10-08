"""CLI: python -m data_processing.asr_clean {plan,run,status,assemble}.

    # once, any node
    python -m data_processing.asr_clean plan --run-root $ROOT --data-dir data

    # on each LLM node (stable worker ids host:0..N-1, so a restart resumes its chunks)
    python -m data_processing.asr_clean run --run-root $ROOT --procs 16 \\
        --asr-url http://inception-H100-hpc-029:8020 --llm-url http://localhost:8010

    python -m data_processing.asr_clean status   --run-root $ROOT
    python -m data_processing.asr_clean assemble --run-root $ROOT
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import socket
import sys
from dataclasses import fields
from pathlib import Path

from .plan import build_plan, config_sources, data_dir_sources
from .worker import WorkerConfig, assemble, run_child, status


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m data_processing.asr_clean", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="chunk + dedup every manifest (run once)")
    p.add_argument("--run-root", required=True)
    p.add_argument("--data-dir", action="append", default=[],
                   help="flat folder of <split>_<lang>_<name>.json manifests (e.g. data/)")
    p.add_argument("--config", action="append", default=[], help="training YAML(s) to take manifests from")
    p.add_argument("--manifest", action="append", default=[], help="extra LANG:SPLIT:PATH")
    p.add_argument("--splits", default="eval,train")
    p.add_argument("--langs", help="comma list to restrict")
    p.add_argument("--chunk-mb", type=int, default=64)
    p.add_argument("--no-dedup", action="store_true")
    p.add_argument("--limit", type=int, help="only the first N records of every manifest (test runs)")
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 8)

    r = sub.add_parser("run", help="clean: claim chunks until none are left")
    r.add_argument("--run-root", required=True)
    r.add_argument("--asr-url", action="append", required=True, help="repeat for several ASR servers")
    r.add_argument("--llm-url", default="http://localhost:8010")
    r.add_argument("--procs", type=int, default=16, help="worker processes on this node")
    r.add_argument("--first-slot", type=int, default=0, help="worker ids are host:<first-slot + i>")
    for f in fields(WorkerConfig):
        if f.name in ("run_root", "asr_urls", "llm_url", "worker_id"):
            continue
        r.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)

    s = sub.add_parser("status")
    s.add_argument("--run-root", required=True)
    s.add_argument("--json", action="store_true")

    k = sub.add_parser("peek", help="print records side by side for spot checks")
    k.add_argument("--run-root", required=True)
    k.add_argument("-n", type=int, default=5, help="records per language")
    k.add_argument("--rejects", action="store_true", help="show rejected records instead")
    k.add_argument("--only-changed", action="store_true", help="skip records where text == org_text")

    a = sub.add_parser("assemble", help="one manifest per source from finished chunks")
    a.add_argument("--run-root", required=True)
    a.add_argument("--allow-partial", action="store_true")

    args = ap.parse_args(argv)

    if args.cmd == "plan":
        splits = tuple(args.splits.split(","))
        sources = [src for d in args.data_dir for src in data_dir_sources(d) if src["split"] in splits]
        sources += [src for cfg in args.config for src in config_sources(cfg, splits)]
        for spec in args.manifest:
            lang, split, path = spec.split(":", 2)
            sources.append({"lang": lang, "split": split, "path": path})
        if args.langs:
            keep = set(args.langs.split(","))
            sources = [s_ for s_ in sources if s_["lang"] in keep]
        seen: set[str] = set()
        sources = [s_ for s_ in sources if not (s_["path"] in seen or seen.add(s_["path"]))]
        if not sources:
            ap.error("no sources: pass --data-dir, --config and/or --manifest")
        build_plan(args.run_root, sources, chunk_mb=args.chunk_mb, dedup=not args.no_dedup, jobs=args.jobs,
                   limit=args.limit)
        return 0

    if args.cmd == "run":
        host = socket.gethostname().split(".")[0]
        base = {f.name: getattr(args, f.name) for f in fields(WorkerConfig)
                if f.name not in ("asr_urls", "worker_id")}
        base.update(run_root=args.run_root, asr_urls=args.asr_url, llm_url=args.llm_url)
        logs = Path(args.run_root) / "_logs"
        logs.mkdir(parents=True, exist_ok=True)
        ctx = mp.get_context("spawn")
        procs = []
        for i in range(args.first_slot, args.first_slot + args.procs):
            wid = f"{host}:{i}"
            proc = ctx.Process(target=run_child, args=({**base, "worker_id": wid}, str(logs / f"{host}-{i}.log")),
                               name=wid)
            proc.start()
            procs.append(proc)
        print(f"{host}: {len(procs)} workers started; logs in {logs}/{host}-*.log", flush=True)
        failed = 0
        for proc in procs:
            proc.join()
            failed += proc.exitcode != 0
        print(f"{host}: all workers finished ({failed} failed)", flush=True)
        return 1 if failed else 0

    if args.cmd == "status":
        st = status(args.run_root)
        if args.json:
            print(json.dumps(st, indent=1))
        else:
            pct = 100 * st["processed"] / max(st["to_clean"], 1)
            print(f"chunks  : {st['chunks_done']}/{st['chunks']} done, {st['chunks_started']} started")
            print(f"records : {st['processed']:,} / {st['to_clean']:,} processed ({pct:.2f}%), "
                  f"{st['written']:,} written; {st['duplicates_planned']:,} duplicates skipped by plan")
            print(f"rate    : {st['records_per_s']} records/s   ETA {st['eta_hours']} h")
            print(f"lanes   : {st['lanes']}")
            print(f"choices : {st['choices']}")
            print(f"rejects : {st['rejects']}")
            print(f"asr truncated (hit max_tokens): {st['asr_truncated']}")
            print(f"agree guard (LLM rewrote words both transcripts agree on; kept ORIGINAL): {st['agree_guard']:,}")
            d = st["arabic_diacritics"]
            print(f"arabic  : {d['marks_per_letter']} marks/letter, {d['records_with_marks_pct']}% of records "
                  f"marked; diacritized {d['diacritized']:,}; kept bare after failed check {d['failed']}")
            t = st["telemetry"]
            print(f"\nASR     : {t['asr_requests']:,} requests, mean {t['asr_mean_latency_s']} s, "
                  f"RTFx {t['asr_rtfx']} (demand from these workers)")
            print(f"LLM     : {t['llm_output_tok_per_s']:,} output tok/s; "
                  f"thinking fell back to non-thinking for {t['think_fallback_items']:,} items")
            print(f"{'lane':<18} {'reqs':>7} {'items/req':>9} {'answered':>8} {'latency':>8} "
                  f"{'prompt tok':>10} {'output tok':>10} {'hit max':>7}")
            for lane, v in t["llm"].items():
                print(f"{lane:<18} {v['requests']:>7,} {v['items_per_request']:>9} {v['items_answered_pct']:>7}% "
                      f"{v['mean_latency_s']:>7}s {v['mean_prompt_tokens']:>10,} {v['mean_completion_tokens']:>10,} "
                      f"{v['hit_max_tokens_pct']:>6}%")
        return 0

    if args.cmd == "peek":
        import random

        root = Path(args.run_root)
        base = root / "_rejects" if args.rejects else root
        for lang_dir in sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith("_")):
            rows = [json.loads(line) for f in sorted(lang_dir.rglob("part-*.jsonl"))
                    for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
            if args.only_changed:
                rows = [r for r in rows if r.get("text") != r.get("org_text")]
            print(f"===== {lang_dir.name}: {len(rows):,} {'rejected' if args.rejects else 'written'}")
            for r in random.Random(0).sample(rows, min(args.n, len(rows))):
                print(f"[{r.get('duration')}s] {r['audio_filepath']}")
                print(f"  ORG : {r.get('org_text')}")
                print(f"  ASR : {r.get('asr_text')}")
                if args.rejects:
                    print(f"  WHY : {r.get('reason')} {r.get('detail', '')}")
                    if r.get("llm_output"):
                        print(f"  LLM : {r['llm_output']}")
                else:
                    print(f"  TEXT: {r.get('text')}")
        return 0

    if args.cmd == "assemble":
        for line in assemble(args.run_root, args.allow_partial):
            print(line)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
