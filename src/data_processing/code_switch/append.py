"""Add Arabic partners without decoding or indexing the existing pools again."""

from __future__ import annotations

import copy
import json
import os
import shutil
import sqlite3
import time
from pathlib import Path

from .common import LANGS, atomic_json, chunk_paths, digest, load_plan, lock
from .index import _byte_ranges, _scan_source, check_frozen, identity


def signature(path):
    st = Path(path).stat()
    return [st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino]


def check_base(plan):
    base = plan["append_base"]
    for path, expected in base["signatures"].items():
        if signature(path) != expected:
            raise ValueError(f"Retained code-switch dependency changed: {path}")
    if digest(base["index"]) != base["sha256"]:
        raise ValueError("Retained code-switch index changed")


def plan_append(root, source_index, manifests):
    root, source_index = Path(root).resolve(), Path(source_index).resolve()
    old = json.loads(source_index.read_text())
    old_root = Path(old["root"])
    if root == old_root:
        raise ValueError("Append requires a new run root")
    paths = [str(Path(p).resolve()) for p in manifests]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("Specify unique new Arabic manifests")
    if any(not Path(p).name.startswith("train_") for p in paths):
        raise ValueError("Additions must be named train_*.jsonl")
    with lock(old_root / "_state/lifecycle.lock"), lock(root / "_state/lifecycle.lock"):
        if (root / "_state/plan.json").exists():
            raise ValueError("Frozen plan exists; use run to resume")
        frozen = load_plan(old_root)
        check_frozen(old_root, frozen)
        if digest(old_root / "_state/plan.json") != old["plan_sha256"]:
            raise ValueError("Base index/plan mismatch")
        if old["sources"] != frozen["sources"] or old["config"] != frozen["config"]:
            raise ValueError("Base index sources/config do not match its plan")
        if set(paths).intersection(s["path"] for s in old["sources"]):
            raise ValueError("A proposed manifest is already indexed")
        signatures = {str(source_index): signature(source_index),
                      str(old_root / "_state/plan.json"): signature(old_root / "_state/plan.json")}
        for lang in LANGS:
            pool = (source_index.parent / f"{lang}.sqlite").resolve()
            if pool.stat().st_size != old["pools"][lang]["size"]:
                raise ValueError(f"Base pool size changed: {pool}")
            signatures[str(pool)] = signature(pool)
        additions = [_scan_source(path) for path in paths]
        plan = copy.deepcopy(frozen)
        # Exclusions already incorporate any earlier quarantine revision.
        plan.pop("derived_index", None)
        plan.update(root=str(root), created=time.time(), chunks=[],
                    sources=old["sources"] + additions,
                    baseline_records=old["baseline_records"] + sum(s["rows"] for s in additions),
                    planned_source_rows=sum(s["rows"] for s in additions),
                    append_base={"index": str(source_index), "sha256": digest(source_index),
                                 "signatures": signatures, "new_sources": paths})
        for path in paths:
            for start, end in _byte_ranges(path, plan["config"]["chunk_mb"] << 20):
                plan["chunks"].append({"id": f"{len(plan['chunks']):07d}", "lang": "ar",
                                           "source": path, "start": start, "end": end})
        state = root / "_state"
        tokenizer = state / "tokenizer"
        if tokenizer.exists():
            shutil.rmtree(tokenizer)  # unpublished planner retry only
        shutil.copytree(old_root / "_state/tokenizer", tokenizer)
        shutil.copyfile(old_root / "_state/excluded.json", state / "excluded.json")
        check_base(plan)
        atomic_json(state / "plan.json", plan)
        print(f"Append plan: {len(paths)} new manifests, {len(plan['chunks'])} Arabic chunks; old pools retained", flush=True)
        return plan


def assemble_pools(root, output, plan):
    """Keep old Arabic IDs, append new IDs, and reference the other four pools."""
    check_base(plan)
    source = Path(plan["append_base"]["index"])
    old = json.loads(source.read_text())
    pools = copy.deepcopy(old["pools"])
    path = output / "ar.sqlite"
    path.unlink(missing_ok=True)
    print("append: copying existing Arabic pool; no old audio is opened", flush=True)
    shutil.copyfile(source.parent / "ar.sqlite", path)
    count = pools["ar"]["count"]
    seconds = pools["ar"]["seconds"]
    duplicates = pools["ar"].get("duplicates", 0)
    connections = {}
    try:
        for lang in LANGS:
            if lang != "ar":
                parent = (source.parent / f"{lang}.sqlite").resolve()
                connections[lang] = sqlite3.connect(parent.as_uri() + "?mode=ro&immutable=1", uri=True)
                link = output / f"{lang}.sqlite"
                link.unlink(missing_ok=True)
                link.symlink_to(parent)
        with sqlite3.connect(path) as db:
            if db.execute("SELECT COUNT(*),MAX(id) FROM pool").fetchone() != (count, count - 1):
                raise ValueError("Base Arabic pool IDs/count are inconsistent")
            for chunk in plan["chunks"]:
                with chunk_paths(root, chunk)[0].open() as stream:
                    for raw in stream:
                        row = json.loads(raw)
                        key = identity(row["audio_filepath"])
                        if any(other.execute("SELECT 1 FROM pool WHERE audio_key=?", (key,)).fetchone()
                               for other in connections.values()):
                            raise ValueError(f"New Arabic partner overlaps a retained language: {key}")
                        cursor = db.execute("INSERT OR IGNORE INTO pool VALUES (?,?,?)", (count, key, raw.strip()))
                        if cursor.rowcount:
                            count += 1
                            seconds += row["duration"]
                        else:
                            duplicates += 1
                db.commit()
                print(f"append: chunk {chunk['id']}, {count - pools['ar']['count']:,} new Arabic partners", flush=True)
    finally:
        for connection in connections.values():
            connection.close()
    with path.open("rb") as complete:
        os.fsync(complete.fileno())
    pools["ar"] = {"count": count, "seconds": seconds, "duplicates": duplicates,
                       "size": path.stat().st_size, "sha256": digest(path)}
    check_base(plan)
    return pools
