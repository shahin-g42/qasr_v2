"""Freeze the work: sources -> byte-range chunks -> exact audio_filepath dedup.

Run once (any node) before the workers. Everything a worker needs lives under
``<run_root>/_state/``:

- ``plan.json``          every chunk: id, lang, split, source path, stem
                         (``<tree>/<file stem>``, e.g. ``v7.6/train_ar_q3asr``),
                         byte range, line count, duplicate count;
- ``dups/<chunk>.npy``   packed bitmask over the chunk's lines, 1 = an earlier
                         line anywhere in the plan has the same audio path.

Dedup order is eval files first, then train in config order, so a clip that
appears in both splits stays in eval and its train copies are dropped (no
train/eval leakage), and overlapping slices (e.g. ``train_ar_q3asr`` range
files) are cleaned once.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .text import local_path

PLAN_VERSION = 1
_PATH_RE = re.compile(rb'"(?:audio_filepath|wav_path)"\s*:\s*"((?:[^"\\]|\\.)*)"')


def config_sources(config: str, splits: tuple[str, ...] = ("eval", "train")) -> list[dict]:
    """``[{lang, split, path}]`` from a training YAML, eval first (the dedup priority)."""
    try:
        import yaml
    except ImportError:
        yaml = None
    out: list[dict] = []
    for split in splits:
        key = f"{split}_manifest"
        with open(config, encoding="utf-8") as fh:
            if yaml is not None:
                block = (yaml.safe_load(fh) or {}).get(key) or {}
            else:  # the fixed `key:` / `  lang:` / `    - path` shape every config uses
                block, inside, lang = {}, False, None
                for raw in fh:
                    line = raw.split("#", 1)[0].rstrip()
                    if not line.strip():
                        continue
                    if not line.startswith(" "):
                        inside, lang = line == f"{key}:", None
                    elif inside and re.fullmatch(r"  [A-Za-z_]+:", line):
                        lang = line.strip()[:-1]
                    elif inside and lang and line.lstrip().startswith("- "):
                        block.setdefault(lang, []).append(line.lstrip()[2:].strip().strip("'\""))
        for lang, paths in block.items():
            for path in paths:
                out.append({"lang": lang, "split": split, "path": path})
    return out


_DATA_NAME_RE = re.compile(r"^(eval|train)_([a-z]{2})_(.+)\.jsonl?$")


def data_dir_sources(data_dir: str) -> list[dict]:
    """Sources from a flat manifest folder named ``<split>_<lang>_<name>.json[l]``.

    The repo's ``data/`` folder (the validated manifests) is the source of
    truth; split and language come from the file name. Eval files sort first,
    which is the dedup priority (a clip in both splits stays in eval).
    """
    from .text import LANGUAGE_NAMES

    out, bad = [], []
    for f in sorted(Path(data_dir).iterdir()):
        if not f.is_file() or f.suffix not in (".json", ".jsonl"):
            continue
        m = _DATA_NAME_RE.match(f.name)
        if not m or m.group(2) not in LANGUAGE_NAMES:
            bad.append(f.name)
            continue
        out.append({"lang": m.group(2), "split": m.group(1), "path": str(f.resolve())})
    if bad:
        raise SystemExit(f"cannot tell split/language of: {', '.join(bad)} "
                         "(expected <eval|train>_<ar|en|zh|hi|ml>_<name>.json)")
    out.sort(key=lambda s_: (s_["split"] != "eval", s_["path"]))
    return out


def check_jsonl(path: str) -> None:
    """Refuse a file that is not one JSON object per line (e.g. a single JSON array)."""
    with open(path, "rb") as fh:
        first = fh.readline(1 << 20).strip()
    if not first:
        return
    try:
        obj = json.loads(first)
    except json.JSONDecodeError:
        obj = None
    if not isinstance(obj, dict):
        raise SystemExit(f"{path}: first line is not a JSON object -- expected JSON Lines "
                         f"(one record per line), got: {first[:80]!r}")


def _stem(path: str) -> str:
    return Path(path).name.removesuffix(".jsonl").removesuffix(".json")


def _tree(path: str, lang: str) -> str:
    """Which manifest tree a file belongs to: the directory above its language dir.

    ``training_manifests/v7.6/ar/x.jsonl`` -> ``v7.6``;
    ``q3asr_sft_manifests/ar/x.jsonl``    -> ``q3asr_sft_manifests``.
    Both trees hold a ``train_ar_q3asr.jsonl`` with different content, so the
    tree is part of every output path.
    """
    parent = Path(path).parent
    return (parent.parent.name if parent.name == lang else parent.name) or "root"


def _limit_offset(path: str, limit: int) -> int:
    """Byte offset just past the first ``limit`` lines."""
    with open(path, "rb") as fh:
        for _ in range(limit):
            if not fh.readline():
                break
        return fh.tell()


def _byte_ranges(path: str, target: int, size: int | None = None) -> list[tuple[int, int]]:
    """Split ``[0, size)`` of a file into ~``target``-byte ranges on line boundaries."""
    size = os.path.getsize(path) if size is None else size
    ranges, start = [], 0
    with open(path, "rb") as fh:
        while start < size:
            end = start + target
            if end >= size:
                ranges.append((start, size))
                break
            fh.seek(end)
            fh.readline()
            end = fh.tell()
            ranges.append((start, end))
            start = end
    return ranges


def iter_chunk_lines(path: str, start: int, end: int):
    """Yield ``(line_index, end_offset, raw_line)`` for every line in ``[start, end)``.

    The planner and the workers both enumerate lines with this, so a line's
    index (the dedup-mask position) is the same on both sides.
    """
    with open(path, "rb") as fh:
        fh.seek(start)
        pos, i = start, 0
        while pos < end:
            line = fh.readline()
            if not line:
                break
            pos += len(line)
            yield i, pos, line
            i += 1


def path_key(raw: bytes) -> int | None:
    """64-bit identity of a line's audio path (after the /vast remap), or None."""
    m = _PATH_RE.search(raw)
    if not m:
        return None
    try:
        path = json.loads(b'"' + m.group(1) + b'"')
    except json.JSONDecodeError:
        return None
    return int.from_bytes(hashlib.blake2b(local_path(path).encode(), digest_size=8).digest(), "little")


def _hash_chunk(args: tuple[str, int, int]):
    import numpy as np

    path, start, end = args
    keys = []
    for i, _, raw in iter_chunk_lines(path, start, end):
        k = path_key(raw)
        # Unparseable lines get a unique key so they are never "duplicates";
        # the worker rejects them on its own.
        keys.append(k if k is not None else (hash((path, start, i)) & 0xFFFFFFFFFFFFFFFF) | 1 << 63)
    return np.asarray(keys, dtype=np.uint64)


def build_plan(run_root: str, sources: list[dict], *, chunk_mb: int = 64, dedup: bool = True,
               jobs: int = os.cpu_count() or 8, limit: int | None = None) -> dict:
    """Freeze the chunk list. ``limit`` keeps only the first N lines of every
    manifest (a quick test run over the real data)."""
    root = Path(run_root)
    state = root / "_state"
    plan_path = state / "plan.json"
    if plan_path.exists():
        raise SystemExit(f"plan already exists: {plan_path} (a run's plan is frozen; use a new RUN_ID)")
    (state / "dups").mkdir(parents=True, exist_ok=True)

    missing = [s["path"] for s in sources if not os.path.exists(s["path"])]
    if missing:
        raise SystemExit("missing manifests:\n  " + "\n  ".join(missing))
    for s in sources:
        check_jsonl(s["path"])
    stems: dict[tuple[str, str], str] = {}
    chunks: list[dict] = []
    for s in sources:
        out = f"{_tree(s['path'], s['lang'])}/{_stem(s['path'])}"
        key = (s["lang"], out)
        if key in stems and stems[key] != s["path"]:
            raise SystemExit(f"two sources map to the same output {key}: {stems[key]} and {s['path']}")
        stems[key] = s["path"]
        size = _limit_offset(s["path"], limit) if limit else None
        for k, (a, b) in enumerate(_byte_ranges(s["path"], chunk_mb << 20, size)):
            chunks.append({"id": f"{s['lang']}__{out.replace('/', '__')}__{k:05d}", "lang": s["lang"],
                           "split": s["split"], "source": s["path"], "stem": out, "part": k,
                           "start": a, "end": b})

    print(f"plan: {len(sources)} manifests -> {len(chunks)} chunks of ~{chunk_mb} MB; hashing paths "
          f"with {jobs} processes", flush=True)
    import numpy as np

    with ProcessPoolExecutor(max_workers=jobs) as pool:
        arrays = list(pool.map(_hash_chunk, [(c["source"], c["start"], c["end"]) for c in chunks], chunksize=1))
    sizes = [len(a) for a in arrays]
    for c, n in zip(chunks, sizes, strict=True):
        c["lines"] = n

    dup_total = 0
    if dedup and chunks:
        keys = np.concatenate(arrays)
        _, first = np.unique(keys, return_index=True)
        is_dup = np.ones(len(keys), dtype=bool)
        is_dup[first] = False
        offset = 0
        for c, n in zip(chunks, sizes, strict=True):
            mask = is_dup[offset:offset + n]
            offset += n
            c["dups"] = int(mask.sum())
            dup_total += c["dups"]
            np.save(state / "dups" / f"{c['id']}.npy", np.packbits(mask))
    else:
        for c in chunks:
            c["dups"] = 0

    plan = {"version": PLAN_VERSION, "chunks": chunks, "dedup": dedup, "limit": limit,
            "lines": sum(sizes), "duplicates": dup_total,
            "sources": [{**s, "stem": f"{_tree(s['path'], s['lang'])}/{_stem(s['path'])}"} for s in sources]}
    tmp = plan_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(plan, indent=1))
    os.replace(tmp, plan_path)
    print(f"plan: {plan['lines']:,} lines, {dup_total:,} duplicate audio paths skipped, "
          f"{plan['lines'] - dup_total:,} to clean -> {plan_path}", flush=True)
    if limit:
        # Scale each file's sampled bytes-per-record to its full size: turns a
        # test run's records/s into a wall-clock estimate for the real run.
        full = 0
        for s_ in sources:
            cs = [c for c in chunks if c["source"] == s_["path"]]
            n, nbytes = sum(c["lines"] for c in cs), sum(c["end"] - c["start"] for c in cs)
            full += round(os.path.getsize(s_["path"]) * n / nbytes) if nbytes else 0
        plan["estimated_full_lines"] = full
        tmp.write_text(json.dumps(plan, indent=1))
        os.replace(tmp, plan_path)
        print(f"plan: the full run (no --limit) would have ~{full:,} records before dedup", flush=True)
    return plan


def load_plan(run_root: str) -> dict:
    return json.loads((Path(run_root) / "_state" / "plan.json").read_text())


def load_dup_mask(run_root: str, chunk: dict):
    """Boolean per line of the chunk, or None when the plan has no dedup."""
    path = Path(run_root) / "_state" / "dups" / f"{chunk['id']}.npy"
    if not path.exists():
        return None
    import numpy as np

    return np.unpackbits(np.load(path))[: chunk["lines"]].astype(bool)
