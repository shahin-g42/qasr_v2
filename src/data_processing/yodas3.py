"""Additive Yodas3 campaign: freeze inputs, publish new manifests, write updates.

Existing training manifests and indexes are never edited. Long clips remain in
the cleaner's rejection files for a later segmentation pass; audio is not cropped.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

import yaml

from .asr_clean.plan import build_plan, config_sources, load_plan
from .asr_clean.text import strip_envelope
from .asr_clean.worker import OUTPUT_KEYS, assemble, status, verify
from .code_switch.common import atomic_json, digest, lock

POLICY = {"dialect_guard": "quarantine", "ar_diacritics": "critical", "max_duration": 35.0}
CODES = {"dz", "iq", "ly", "msa", "sa", "sy", "ye", "bh", "eg", "lb", "ma", "ps", "sd", "tn"}
META_KEYS = ("source_id", "id", "start", "end", "dialect", "dialect_score", "dialect_probs",
             "code_switch", "audio_sha256", "source_repo", "source_revision", "speech_ratio", "clean_cut")


def stat_signature(path):
    s = Path(path).stat()
    return [s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_ino]


def plan_campaign(root, sources, jobs=8, limit=None):
    root = Path(root).resolve()
    selected = config_sources(sources, ("train",))
    codes = [Path(s["path"]).stem.removeprefix("manifest_") for s in selected]
    if set(codes) != CODES or len(codes) != 14 or any(s["lang"] != "ar" for s in selected):
        raise ValueError("Campaign requires exactly the 14 Arabic Yodas3 source manifests")
    if limit is not None and limit < 1:
        raise ValueError("Pilot limit must be positive")
    with lock(root / "_state/campaign.lock"):
        if (root / "_state/plan.json").exists():
            raise ValueError("Plan exists; resume run or choose a new RUN_ID")
        snapshots = {s["path"]: stat_signature(s["path"]) for s in selected}
        for source in selected:
            with open(source["path"]) as stream:
                sample = json.loads(next(stream))
            if not isinstance(sample.get("text"), str) or not isinstance(sample.get("audio_filepath"), str):
                raise ValueError(f"Incompatible Yodas schema: {source['path']}")
        plan = build_plan(str(root), selected, jobs=jobs, limit=limit)
        if any(stat_signature(path) != sig for path, sig in snapshots.items()):
            raise ValueError("Source changed while planning; discard this unstarted run and plan again")
        atomic_json(root / "_state/campaign.json", {"policy": POLICY, "sources": snapshots,
                    "plan_sha256": digest(root / "_state/plan.json"), "pilot": limit is not None})
        return plan


def check_campaign(root):
    root = Path(root).resolve()
    campaign = json.loads((root / "_state/campaign.json").read_text())
    if campaign["policy"] != POLICY or campaign["plan_sha256"] != digest(root / "_state/plan.json"):
        raise ValueError("Campaign policy/plan changed; use a new run")
    for path, expected in campaign["sources"].items():
        if stat_signature(path) != expected:
            raise ValueError(f"Frozen source changed: {path}")
    return campaign


def install_links(release, destination):
    """Idempotent no-clobber installation, including recovery after a partial link pass."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for spec in release["manifests"]:
        target = destination / Path(spec["file"]).name
        source = Path(spec["file"])
        if digest(source) != spec["sha256"]:
            raise ValueError(f"Published manifest changed: {source}")
        if os.path.lexists(target) and (not target.is_symlink() or target.resolve() != source.resolve()):
            raise FileExistsError(f"Refusing to replace existing manifest: {target}")
    for spec in release["manifests"]:
        target = destination / Path(spec["file"]).name
        if not os.path.lexists(target):
            target.symlink_to(spec["file"])


def publish(root, destination, base_index, jobs=8):
    root, base_index = Path(root).resolve(), Path(base_index).resolve()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", root.name):
        raise ValueError("RUN_ID may contain only letters, digits, underscore and hyphen")
    with lock(root / "_state/campaign.lock"):
        campaign = check_campaign(root)
        if campaign["pilot"]:
            raise ValueError("Pilot output cannot be published for production")
        release_dir = root / "published"
        if release_dir.exists():
            release = json.loads((release_dir / "release.json").read_text())
            if release["base_index"] != str(base_index):
                raise ValueError("Publication belongs to a different base index")
            install_links(release, destination)
            return release
        if not base_index.is_file() or base_index.with_name(base_index.name + ".scan.lock").exists():
            raise ValueError("Need a completed GQASR base index for overlap exclusion")
        signature = stat_signature(base_index)
        st = status(str(root))
        if st["chunks_done"] != st["chunks"]:
            raise ValueError("Cleaning is incomplete; wait for all chunks")
        plan = load_plan(str(root))
        source_counts = {}
        for chunk in plan["chunks"]:
            progress = json.loads((root / "_state/progress" / (chunk["id"] + ".json")).read_text())
            if any(progress.get("policy", {}).get(k) != v for k, v in POLICY.items()):
                raise ValueError("Completed chunk did not use the campaign's dialect/duration policy")
            source_counts.setdefault(chunk["source"], Counter()).update(progress["counts"])
        if verify(str(root), jobs=jobs)["bad"]:
            raise ValueError("Cleaning verification failed")
        assemble(str(root))
        staging = Path(tempfile.mkdtemp(prefix=".publishing-", dir=root))
        try:
            folder = staging / "ar/data"
            folder.mkdir(parents=True)
            manifests = []
            with sqlite3.connect(base_index.as_uri() + "?mode=ro", uri=True) as previous, \
                    sqlite3.connect(staging / "seen.sqlite") as seen, \
                    (staging / "excluded.jsonl").open("w") as rejected:
                base_meta = json.loads(previous.execute("SELECT value FROM metadata WHERE key='index'").fetchone()[0])
                seen.execute("CREATE TABLE seen (audio TEXT PRIMARY KEY, sha TEXT UNIQUE, segment TEXT UNIQUE)")
                for source in plan["sources"]:
                    code = Path(source["path"]).stem.removeprefix("manifest_")
                    name = f"train_ar_{root.name}_{code}.jsonl"
                    cleaned = root / source["lang"] / (source["stem"] + ".jsonl")
                    counters = Counter()
                    seconds = 0.0
                    with open(source["path"]) as originals, cleaned.open() as accepted, (folder / name).open("w") as out:
                        for line in accepted:
                            row = json.loads(line)
                            if tuple(row) != OUTPUT_KEYS:
                                raise ValueError(f"Unexpected cleaner schema: {cleaned}")
                            original = None
                            for raw in originals:
                                candidate = json.loads(raw)
                                if candidate["audio_filepath"] == row["audio_filepath"]:
                                    original = candidate
                                    break
                            if original is None or strip_envelope(original["text"]) != row["org_text"]:
                                raise ValueError(f"Cannot reconcile source provenance for {row['audio_filepath']}")
                            audio = row["audio_filepath"]
                            recording = str(original.get("source_id") or "")
                            reason = None
                            if previous.execute("SELECT 1 FROM samples WHERE audio=? LIMIT 1", (audio,)).fetchone():
                                reason = "existing_audio"
                            elif recording and previous.execute("SELECT 1 FROM samples WHERE split='eval' AND recording_id=? LIMIT 1", (recording,)).fetchone():
                                reason = "eval_recording"
                            segment = json.dumps([recording, original.get("start"), original.get("end")]) if recording else None
                            if not reason:
                                inserted = seen.execute("INSERT OR IGNORE INTO seen VALUES (?,?,?)", (audio, original.get("audio_sha256") or None, segment))
                                if not inserted.rowcount:
                                    reason = "duplicate_new_audio"
                            if reason:
                                counters[reason] += 1
                                rejected.write(json.dumps({**row, "reason": reason, "source_manifest": source["path"]}, ensure_ascii=False) + "\n")
                                continue
                            enriched = {**row, **{key: original[key] for key in META_KEYS if key in original},
                                        "language": "ar", "recording_id": recording, "source_manifest": source["path"],
                                        "dataset_name": "yodas3", "cleaning_run": root.name}
                            out.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                            seconds += row["duration"]
                            counters["accepted"] += 1
                        out.flush()
                        os.fsync(out.fileno())
                    seen.commit()
                    manifests.append({"code": code, "file": str(release_dir / "ar/data" / name),
                                          "sha256": digest(folder / name), "counts": dict(counters), "seconds": seconds,
                                          "cleaner_counts": dict(source_counts.get(source["path"], {}))})
                    print(json.dumps(dict(source=code, **counters)), flush=True)
            (staging / "seen.sqlite").unlink()
            check_campaign(root)
            if stat_signature(base_index) != signature:
                raise ValueError("Base index changed during publication")
            release = {"run_root": str(root), "base_index": str(base_index), "base_data_contract": base_meta["data_contract"],
                           "manifests": manifests, "cleaner_status": st, "policy": POLICY,
                           "overlap_scope": "Existing indexed audio paths and eval recording IDs; new audio hashes/segments. Different-path duplicates absent from those identities are not detected."}
            atomic_json(staging / "release.json", release)
            os.rename(staging, release_dir)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        install_links(release, destination)
        return release


def write_updates(root, qasr_config, gqasr_config, cs_root, repo):
    root, repo = Path(root).resolve(), Path(repo).resolve()
    release = json.loads((root / "published/release.json").read_text())
    added = [m["file"] for m in release["manifests"] if m["counts"].get("accepted", 0)]
    if not added:
        raise ValueError("No accepted additions to train on")
    for spec in release["manifests"]:
        if digest(spec["file"]) != spec["sha256"]:
            raise ValueError("Published manifest changed")
    qpath, gpath = Path(qasr_config).resolve(), Path(gqasr_config).resolve()
    q, g = yaml.safe_load(qpath.read_text()), yaml.safe_load(gpath.read_text())
    for cfg in (q, g):
        for key in ("train_manifest", "eval_manifest"):
            if cfg[key] != release["base_data_contract"][key]:
                raise ValueError(f"Base {key} differs from the publication's existing index; select matching base configs")
    old_g_index = str((gpath.parent / g["index_path"]).resolve())
    if old_g_index != release["base_index"]:
        raise ValueError("GQASR config index_path differs from publication base")
    update_dir = root / "updates"
    if update_dir.exists():
        raise FileExistsError("Updates already exist; do not overwrite a training configuration")
    q, g = copy.deepcopy(q), copy.deepcopy(g)
    for cfg in (q, g):
        cfg["train_manifest"]["ar"].extend(added)
        cfg["code_switch_index"] = str(Path(cs_root).resolve() / "assembled/index.json")
        cfg["run_name"] = str(cfg.get("run_name") or "train") + "-" + root.name
    q["output_dir"] = str(repo / "output" / root.name / "qasr")
    q["resume_from_checkpoint"] = None
    for key in ("base_model", "code_switch_exclude_manifest"):
        if g.get(key):
            g[key] = str((gpath.parent / g[key]).resolve())
    g["output_dir"] = str(repo / "gemma_speech/gqasr/outputs" / root.name)
    g["index_path"] = str(repo / "gemma_speech/gqasr/indexes" / (root.name + ".sqlite"))
    # Replace the existing raw Yodas stream; retain unrelated streaming sources.
    streams = g.get("streaming_datasets", {})
    if "yodas" in str(streams.get("ar", {}).get("dataset", "")).lower():
        streams.pop("ar")
    if not streams:
        g.pop("streaming_datasets", None)
    staging = Path(tempfile.mkdtemp(prefix=".updates-", dir=root))
    try:
        for name, cfg in (("qasr.yaml", q), ("gqasr.yaml", g)):
            (staging / name).write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        atomic_json(staging / "additions.json", {"train_manifest": {"ar": added},
                    "base_gqasr_index": old_g_index, "new_gqasr_index": g["index_path"],
                    "code_switch_index": q["code_switch_index"], "includes_msa": True})
        os.rename(staging, update_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"qasr": str(update_dir / "qasr.yaml"), "gqasr": str(update_dir / "gqasr.yaml")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "check-run", "publish", "configs"):
        p = sub.add_parser(name)
        p.add_argument("--run-root", required=True)
        if name == "plan":
            p.add_argument("--sources", default="configs/data_processing/yodas3_asr_sources.yaml")
            p.add_argument("--jobs", type=int, default=8)
            p.add_argument("--limit", type=int)
        if name == "publish":
            p.add_argument("--destination", required=True)
            p.add_argument("--base-index", required=True)
            p.add_argument("--jobs", type=int, default=8)
        if name == "configs":
            for arg in ("qasr-config", "gqasr-config", "cs-root", "repo"):
                p.add_argument("--" + arg, required=True)
    args = vars(parser.parse_args(argv))
    command, root = args.pop("command"), args.pop("run_root")
    if command == "plan":
        plan_campaign(root, **args)
    elif command == "check-run":
        check_campaign(root)
    elif command == "publish":
        publish(root, **args)
    else:
        print(json.dumps(write_updates(root, **args), indent=2))


if __name__ == "__main__":
    main()
