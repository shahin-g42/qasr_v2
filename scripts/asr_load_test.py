#!/usr/bin/env python3
"""Load-test the QASR vLLM deployment with clips sampled from our own manifests.

Sweeps client concurrency and reports, per level: throughput (requests/s and
RTFx = audio-seconds transcribed per wall-second), latency percentiles, errors,
and accuracy per language (CER/WER against the manifest reference, model
language-ID agreement, truncation and empty-output rates). The knee of the
throughput curve tells you the concurrency the relabel job should run at, and
``--project-hours`` turns the best level into a wall-clock estimate.

Stdlib only (runs from the base conda env on any node):

    # eval manifests of the v7.6 config, 200 clips/language, default sweep
    python3 scripts/asr_load_test.py --url http://inception-H100-hpc-029:8020 \\
        --config configs/v7.6/02_full_8node.yaml --split eval

    # specific manifests, several servers round-robin, custom sweep
    python3 scripts/asr_load_test.py --url http://hpc-029:8020 --url http://hpc-030:8020 \\
        --manifest ar=/lustrefs/.../train_ar_inworld_full.jsonl --manifest en=/lustrefs/.../x.jsonl \\
        --per-lang 500 --concurrency 32,128,512 --project-hours 250000

Sampling seeks to random byte offsets, so 100M-line manifests cost nothing to
sample. Results land in ``--out`` (report.json + requests.jsonl per level).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import struct
import sys
import threading
import time
import unicodedata
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr_vllm_client import LANGUAGES, local_path, transcribe

ENVELOPE_RE = re.compile(r"^\s*language\s+[^<]*<asr_text>", re.IGNORECASE)
ARABIC_DIACRITICS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭـ]")
ALEF_FORMS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})
DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
NO_SPACE_LANGS = {"zh"}


# ---------------------------------------------------------------- sampling --

def sample_lines(path: str, n: int, rng: random.Random, max_tries: int = 20,
                 small_bytes: int = 64 << 20) -> list[str]:
    """``n`` random JSONL lines. Small files are read whole; large ones are byte-seeked."""
    size = os.path.getsize(path)
    if size < small_bytes:
        with open(path, encoding="utf-8") as fh:
            lines = [line for line in fh if line.strip()]
        return rng.sample(lines, min(n, len(lines)))
    out: list[str] = []
    seen: set[int] = set()
    with open(path, "rb") as fh:
        for _ in range(n * max_tries):
            if len(out) >= n:
                break
            fh.seek(rng.randrange(size))
            fh.readline()  # skip the partial line we landed in
            start = fh.tell()
            line = fh.readline()
            if not line.strip() or start in seen:
                continue
            seen.add(start)
            out.append(line.decode("utf-8", errors="replace"))
    return out


def audio_duration(path: str) -> float | None:
    """Header-only duration for wav/flac (no decoding, no third-party deps)."""
    try:
        if path.endswith(".wav"):
            with wave.open(path) as w:
                return w.getnframes() / float(w.getframerate())
        if path.endswith(".flac"):
            with open(path, "rb") as fh:
                if fh.read(4) != b"fLaC":
                    return None
                fh.read(4)  # STREAMINFO block header (always the first block)
                info = fh.read(34)
            rate = (info[10] << 12) | (info[11] << 4) | (info[12] >> 4)
            total = ((info[13] & 0x0F) << 32) | struct.unpack(">I", info[14:18])[0]
            return total / rate if rate and total else None
    except (OSError, wave.Error, EOFError, IndexError, struct.error):
        return None
    return None


def parse_row(line: str, lang: str, source: str) -> dict | None:
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        return None
    path = row.get("audio_filepath") or row.get("wav_path")
    ref = row.get("text") or row.get("transcript") or ""
    if not isinstance(path, str) or not isinstance(ref, str):
        return None
    ref = ENVELOPE_RE.sub("", ref)  # q3asr SFT rows carry "language X<asr_text>"
    duration = row.get("duration")
    if not isinstance(duration, (int, float)) or duration <= 0:
        duration = audio_duration(local_path(path))
    return {"path": path, "ref": ref, "lang": lang, "duration": duration, "source": Path(source).name}


def config_manifests(config: str, key: str) -> dict[str, list[str]]:
    """``{lang: [paths]}`` under ``key`` of a training YAML.

    Uses PyYAML when present; otherwise reads the fixed shape every config
    uses (``key:`` / two-space ``lang:`` / ``- path`` items), so the base
    python on any node is enough.
    """
    try:
        import yaml
    except ImportError:
        yaml = None
    with open(config, encoding="utf-8") as fh:
        if yaml is not None:
            return {k: list(v) for k, v in ((yaml.safe_load(fh) or {}).get(key) or {}).items()}
        out: dict[str, list[str]] = {}
        inside, lang = False, None
        for raw in fh:
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            if not line.startswith(" "):
                inside, lang = line == f"{key}:", None
            elif inside and re.fullmatch(r"  [A-Za-z_]+:", line):
                lang = line.strip()[:-1]
            elif inside and lang and line.lstrip().startswith("- "):
                out.setdefault(lang, []).append(line.lstrip()[2:].strip().strip("'\""))
        return out


def load_sources(args: argparse.Namespace) -> dict[str, list[str]]:
    sources: dict[str, list[str]] = {}
    if args.config:
        key = "eval_manifest" if args.split == "eval" else "train_manifest"
        for lang, paths in config_manifests(args.config, key).items():
            sources.setdefault(lang, []).extend(paths)
    for spec in args.manifest or []:
        lang, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--manifest wants LANG=PATH, got {spec!r}")
        sources.setdefault(lang, []).append(path)
    if args.langs:
        keep = set(args.langs.split(","))
        sources = {k: v for k, v in sources.items() if k in keep}
    missing = [p for ps in sources.values() for p in ps if not os.path.exists(p)]
    for p in missing:
        print(f"  skip (missing): {p}", file=sys.stderr)
    sources = {k: [p for p in v if os.path.exists(p)] for k, v in sources.items()}
    return {k: v for k, v in sources.items() if v}


def build_pool(sources: dict[str, list[str]], per_lang: int, seed: int,
               min_s: float, max_s: float) -> list[dict]:
    rng = random.Random(seed)
    pool: list[dict] = []
    for lang, paths in sorted(sources.items()):
        quota = -(-per_lang // len(paths))
        got: list[dict] = []
        for path in paths:
            rows = [parse_row(line, lang, path) for line in sample_lines(path, quota * 2, rng)]
            rows = [r for r in rows if r and r["ref"].strip() and
                    (r["duration"] is None or min_s <= r["duration"] <= max_s)]
            got.extend(rows[:quota])
        rng.shuffle(got)
        pool.extend(got[:per_lang])
        print(f"  {lang}: {min(len(got), per_lang)} clips from {len(paths)} manifest(s)")
    rng.shuffle(pool)
    return pool


# ----------------------------------------------------------------- metrics --

def normalize(text: str, lang: str) -> str:
    """Scoring normalization: punctuation/symbols out, combining marks kept attached.

    (``qasr.evaluate.normalize_text`` turns combining marks into spaces, which
    splits Arabic/Hindi/Malayalam words; this does not.) Arabic additionally
    drops diacritics/tatweel and folds alef forms, so cleaned (diacritized)
    and verbatim references score alike.
    """
    text = unicodedata.normalize("NFKC", text).casefold().translate(DIGITS)
    if lang == "ar":
        text = ARABIC_DIACRITICS.sub("", text).translate(ALEF_FORMS)
    text = "".join(" " if unicodedata.category(c)[0] in "PSZC" else c for c in text)
    return " ".join(text.split())


try:  # ~100x faster when available; identical results
    from rapidfuzz.distance import Levenshtein as _lev

    def edit_distance(a: list | str, b: list | str) -> int:
        return _lev.distance(a, b)
except ImportError:
    def edit_distance(a: list | str, b: list | str) -> int:
        if len(a) < len(b):
            a, b = b, a
        prev = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            cur = [i]
            for j, y in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
            prev = cur
        return prev[-1]


def score(records: list[dict]) -> dict[str, dict]:
    """Corpus-level CER/WER, LID agreement, truncation/empty rates per language."""
    out: dict[str, dict] = {}
    for lang in sorted({r["lang"] for r in records}):
        rows = [r for r in records if r["lang"] == lang and r["ok"]]
        if not rows:
            continue
        ce = cn = we = wn = 0
        for r in rows:
            ref, hyp = normalize(r["ref"], lang), normalize(r["hyp"], lang)
            rc, hc = ref.replace(" ", ""), hyp.replace(" ", "")
            ce += edit_distance(rc, hc)
            cn += len(rc)
            rw, hw = (list(rc), list(hc)) if lang in NO_SPACE_LANGS else (ref.split(), hyp.split())
            we += edit_distance(rw, hw)
            wn += len(rw)
        name = LANGUAGES.get(lang, lang).lower()
        out[lang] = {
            "n": len(rows),
            "cer": round(ce / max(cn, 1), 4),
            "wer": round(we / max(wn, 1), 4),  # character-level for zh
            "lid_agree": round(sum((r.get("lid") or "").lower() == name for r in rows) / len(rows), 4),
            "truncated": round(sum(r.get("finish") == "length" for r in rows) / len(rows), 4),
            "empty": round(sum(not r["hyp"].strip() for r in rows) / len(rows), 4),
        }
    return out


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    return values[min(len(values) - 1, round(q * (len(values) - 1)))]


# -------------------------------------------------------------------- load --

def run_level(pool: list[dict], urls: list[str], concurrency: int, n_requests: int,
              timeout: float, max_tokens: int) -> tuple[list[dict], float]:
    lock = threading.Lock()
    counter = iter(range(n_requests))
    records: list[dict] = []

    def worker() -> None:
        while True:
            with lock:
                i = next(counter, None)
            if i is None:
                return
            item = pool[i % len(pool)]
            url = urls[i % len(urls)]
            t0 = time.perf_counter()
            rec = {**item, "url": url, "ok": False}
            try:
                res = transcribe(url, item["path"], item["lang"], max_tokens=max_tokens, timeout=timeout)
                rec.update(ok=True, hyp=res["text"], lid=res["language"], finish=res["finish_reason"],
                           out_tokens=res.get("completion_tokens") or 0)
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
            rec["latency"] = time.perf_counter() - t0
            with lock:
                records.append(rec)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for _ in range(concurrency):
            ex.submit(worker)
    return records, time.perf_counter() - t0


def summarize(records: list[dict], wall: float, concurrency: int) -> dict:
    ok = [r for r in records if r["ok"]]
    lat = [r["latency"] for r in ok]
    audio = sum(r["duration"] or 0.0 for r in ok)
    errors: dict[str, int] = {}
    for r in records:
        if not r["ok"]:
            key = r["error"].split(":")[0]
            errors[key] = errors.get(key, 0) + 1
    return {
        "concurrency": concurrency,
        "requests": len(records),
        "ok": len(ok),
        "error_rate": round(1 - len(ok) / max(len(records), 1), 4),
        "errors": errors,
        "wall_s": round(wall, 2),
        "req_per_s": round(len(ok) / wall, 2),
        "rtfx": round(audio / wall, 1),  # audio seconds per wall second
        "out_tok_per_s": round(sum(r.get("out_tokens", 0) for r in ok) / wall, 1),
        "lat_p50": round(pct(lat, 0.50), 3),
        "lat_p90": round(pct(lat, 0.90), 3),
        "lat_p99": round(pct(lat, 0.99), 3),
        "lat_mean": round(statistics.fmean(lat), 3) if lat else None,
        "mean_clip_s": round(audio / max(len(ok), 1), 2),
        "undated_clips": sum(r["duration"] is None for r in ok),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", action="append", required=True, help="server base URL (repeat for round-robin)")
    ap.add_argument("--config", help="training YAML whose manifests to sample (e.g. configs/v7.6/02_full_8node.yaml)")
    ap.add_argument("--split", choices=("eval", "train"), default="eval")
    ap.add_argument("--manifest", action="append", help="LANG=PATH (repeatable)")
    ap.add_argument("--langs", help="comma list to restrict, e.g. ar,en")
    ap.add_argument("--per-lang", type=int, default=200, help="clips sampled per language")
    ap.add_argument("--concurrency", default="8,32,64,128,256,512", help="comma-separated sweep")
    ap.add_argument("--requests", type=int, default=0, help="requests per level (default: max(pool, 4x concurrency))")
    ap.add_argument("--warmup", type=int, default=32)
    ap.add_argument("--min-duration", type=float, default=0.3)
    ap.add_argument("--max-duration", type=float, default=35.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project-hours", type=float, default=0, help="corpus hours to project wall time for")
    ap.add_argument("--out", default=f"logs/asr_load_test_{time.strftime('%Y%m%d_%H%M%S')}")
    args = ap.parse_args()

    sources = load_sources(args)
    if not sources:
        raise SystemExit("no manifests: pass --config and/or --manifest")
    print("=== sampling clips ===")
    pool = build_pool(sources, args.per_lang, args.seed, args.min_duration, args.max_duration)
    if not pool:
        raise SystemExit("sampled nothing usable")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"=== warmup: {args.warmup} requests ===")
    warm, _ = run_level(pool, args.url, min(args.warmup, 32), args.warmup, args.timeout, args.max_tokens)
    bad = [r for r in warm if not r["ok"]]
    if len(bad) == len(warm):
        raise SystemExit(f"every warmup request failed, e.g. {bad[0]['error']}")

    levels = [int(c) for c in args.concurrency.split(",")]
    report = {"urls": args.url, "pool": len(pool), "langs": sorted(sources), "levels": []}
    hdr = f"{'conc':>5} {'req/s':>7} {'RTFx':>7} {'tok/s':>8} {'p50':>6} {'p90':>6} {'p99':>6} {'err%':>5}"
    print(f"=== sweep over {levels} ({len(args.url)} server(s), {len(pool)} clips) ===\n{hdr}")
    for c in levels:
        n = args.requests or max(len(pool), 4 * c)
        t_start = time.time()
        records, wall = run_level(pool, args.url, c, n, args.timeout, args.max_tokens)
        s = summarize(records, wall, c)
        s["t_start"], s["t_end"] = round(t_start, 1), round(time.time(), 1)  # for GPU/CPU monitors
        s["accuracy"] = score(records)
        report["levels"].append(s)
        with open(out / f"requests_c{c}.jsonl", "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{c:>5} {s['req_per_s']:>7} {s['rtfx']:>7} {s['out_tok_per_s']:>8} "
              f"{s['lat_p50']:>6} {s['lat_p90']:>6} {s['lat_p99']:>6} {100 * s['error_rate']:>5.1f}"
              + (f"  errors={s['errors']}" if s["errors"] else ""))

    best = max(report["levels"], key=lambda s: s["rtfx"])
    # Knee: the smallest concurrency within 5% of the best throughput.
    knee = min((s for s in report["levels"] if s["rtfx"] >= 0.95 * best["rtfx"]), key=lambda s: s["concurrency"])
    report["best"], report["knee"] = best["concurrency"], knee["concurrency"]
    print(f"\npeak RTFx {best['rtfx']} at concurrency {best['concurrency']}; "
          f"knee (>=95% of peak) at concurrency {knee['concurrency']} "
          f"(p90 {knee['lat_p90']} s)")

    print("\n=== accuracy at the knee (normalized; zh WER is character-level) ===")
    print(f"{'lang':>4} {'n':>5} {'CER':>7} {'WER':>7} {'LID':>6} {'trunc':>6} {'empty':>6}")
    for lang, a in knee["accuracy"].items():
        print(f"{lang:>4} {a['n']:>5} {a['cer']:>7.2%} {a['wer']:>7.2%} {a['lid_agree']:>6.1%} "
              f"{a['truncated']:>6.1%} {a['empty']:>6.1%}")

    if args.project_hours and knee["rtfx"] > 0:
        hours = args.project_hours / knee["rtfx"]
        report["projection"] = {"corpus_hours": args.project_hours, "wall_hours": round(hours, 1)}
        print(f"\nprojection: {args.project_hours:,.0f} h of audio at RTFx {knee['rtfx']} "
              f"-> {hours:,.1f} h wall ({hours / 24:,.1f} days) on {len(args.url)} server(s)")

    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"\nreport: {out / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
