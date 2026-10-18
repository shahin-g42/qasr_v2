import json
import sqlite3
from pathlib import Path

import pytest
import yaml

from data_processing.asr_clean.worker import RunPaths
from data_processing.yodas3 import (
    CODES,
    POLICY,
    check_campaign,
    plan_campaign,
    publish,
    write_updates,
)


def campaign(tmp_path, pilot=False):
    sources = tmp_path / "Yodas3"
    sources.mkdir()
    paths = []
    for code in sorted(CODES):
        path = sources / f"manifest_{code}.jsonl"
        row = {"audio_filepath": f"/audio/{code}.opus", "text": "شو بدك تعمل هلق", "duration": 3,
                   "source_id": f"video-{code}", "start": 0, "end": 3, "dialect": code.upper(),
                   "dialect_score": .7, "audio_sha256": f"hash-{code}", "code_switch": False}
        path.write_text(json.dumps(row) + "\n")
        paths.append(str(path))
    config = tmp_path / "sources.yaml"
    config.write_text(yaml.safe_dump({"train_manifest": {"ar": paths}}))
    root = tmp_path / "yodas3_v1"
    plan = plan_campaign(root, config, jobs=1, limit=1 if pilot else None)
    runs = RunPaths(str(root))
    for c in plan["chunks"]:
        original = json.loads(Path(c["source"]).read_text())
        row = {"audio_filepath": original["audio_filepath"], "duration": 3.0, "text": original["text"],
                   "org_text": original["text"], "asr_text": original["text"]}
        part = runs.part(c)
        part.parent.mkdir(parents=True)
        part.write_text(json.dumps(row) + "\n")
        (runs.progress / (c["id"] + ".json")).write_text(json.dumps({"status": "done", "out_bytes": part.stat().st_size,
             "counts": {"lines": 1, "written": 1}, "policy": POLICY, "started": 1, "updated": 2}))
    base = tmp_path / "base.sqlite"
    contract = {"train_manifest": {"ar": ["/old/train_ar.jsonl"]}, "eval_manifest": {"ar": ["/old/eval_ar.jsonl"]}}
    with sqlite3.connect(base) as db:
        db.execute("CREATE TABLE samples (audio TEXT, split TEXT, recording_id TEXT)")
        db.execute("INSERT INTO samples VALUES ('/audio/bh.opus','train','old-bh')")
        db.execute("INSERT INTO samples VALUES ('/other/heldout.opus','eval','video-dz')")
        db.execute("CREATE TABLE metadata (key TEXT, value TEXT)")
        db.execute("INSERT INTO metadata VALUES ('index',?)", (json.dumps({"data_contract": contract}),))
    return root, base, tmp_path / "full_v1/ar/data", contract


def test_publication_enriches_new_files_excludes_old_audio_and_preserves_base(tmp_path):
    root, base, destination, _contract = campaign(tmp_path)
    destination.mkdir(parents=True)
    existing = destination / "train_ar_existing.jsonl"
    existing.write_text("untouched\n")
    before = base.read_bytes()
    release = publish(root, destination, base, jobs=1)
    assert len(release["manifests"]) == 14
    assert sum(m["counts"].get("accepted", 0) for m in release["manifests"]) == 12
    assert base.read_bytes() == before and existing.read_text() == "untouched\n"
    path = destination / "train_ar_yodas3_v1_iq.jsonl"
    assert path.is_symlink()
    row = json.loads(path.read_text())
    assert row["dialect"] == "IQ" and row["recording_id"] == "video-iq"
    assert row["source_manifest"].endswith("manifest_iq.jsonl")
    assert row["org_text"] == row["asr_text"] == row["text"]
    assert len((root / "published/excluded.jsonl").read_text().splitlines()) == 2
    path.unlink()  # retry a partial installation without rewriting any data
    assert publish(root, destination, base, jobs=1) == release
    assert path.is_symlink()
    path.unlink()
    path.write_text("unrelated data")
    with pytest.raises(FileExistsError, match="Refusing"):
        publish(root, destination, base, jobs=1)
    assert path.read_text() == "unrelated data"


def test_updates_append_all_sources_remove_duplicate_stream_and_keep_bases(tmp_path):
    root, base, destination, contract = campaign(tmp_path)
    publish(root, destination, base, jobs=1)
    qpath, gpath = tmp_path / "qasr.yaml", tmp_path / "gqasr.yaml"
    qpath.write_text(yaml.safe_dump({**contract, "code_switch_index": "/old/index.json", "output_dir": "/old/q"}))
    gpath.write_text(yaml.safe_dump({**contract, "index_path": "base.sqlite", "base_model": "base-model",
        "output_dir": "/old/g", "streaming_datasets": {"ar": {"dataset": "oddadmix/yodas-v3-ar-labeled"},
                                                       "en": {"dataset": "unrelated"}}}))
    originals = [qpath.read_bytes(), gpath.read_bytes()]
    updates = write_updates(root, qpath, gpath, tmp_path / "cs_new", tmp_path)
    q, g = [yaml.safe_load(Path(updates[k]).read_text()) for k in ("qasr", "gqasr")]
    assert q["train_manifest"]["ar"][0] == "/old/train_ar.jsonl"
    assert len(q["train_manifest"]["ar"]) == 13  # 12 accepted + one retained
    assert any(p.endswith("_msa.jsonl") for p in q["train_manifest"]["ar"])
    assert g["train_manifest"] == q["train_manifest"]
    assert g["streaming_datasets"] == {"en": {"dataset": "unrelated"}}
    assert g["base_model"] == str(tmp_path / "base-model")
    assert g["index_path"] != str(base)
    assert q["code_switch_index"] == g["code_switch_index"]
    assert q["eval_manifest"] == contract["eval_manifest"]
    assert [qpath.read_bytes(), gpath.read_bytes()] == originals
    with pytest.raises(FileExistsError):
        write_updates(root, qpath, gpath, tmp_path / "cs_new", tmp_path)


def test_pilot_and_changed_inputs_cannot_be_published(tmp_path):
    root, base, destination, _contract = campaign(tmp_path, pilot=True)
    with pytest.raises(ValueError, match="Pilot"):
        publish(root, destination, base, jobs=1)
    path = tmp_path / "Yodas3/manifest_iq.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="Frozen source"):
        check_campaign(root)


def test_completed_chunks_without_dialect_policy_are_refused(tmp_path):
    root, base, destination, _contract = campaign(tmp_path)
    progress = next((root / "_state/progress").glob("*.json"))
    p = json.loads(progress.read_text())
    p.pop("policy")
    progress.write_text(json.dumps(p))
    with pytest.raises(ValueError, match="campaign's dialect"):
        publish(root, destination, base, jobs=1)
