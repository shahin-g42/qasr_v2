"""Replace one language's train/eval records without rescanning other languages."""

import copy
import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path

from .data import data_contract, scan


def rescan_language(
    cfg,
    processor,
    source_index,
    language="ml",
    output=None,
    workers=0,
    chunk_size=128,
    progress_seconds=5.0,
    temp_dir=None,
):
    source = Path(source_index).resolve()
    destination = Path(output or cfg["index_path"]).resolve()
    if source == destination or destination.exists():
        raise FileExistsError("Rescan requires a NEW output index; the source is never modified")
    if not source.is_file():
        raise FileNotFoundError(f"Completed source index required: {source}")
    if source.with_name(source.name + ".scan.lock").exists():
        raise ValueError("Source scan is still running; wait for completion")
    if language not in cfg["language_batch_counts"]:
        raise ValueError(f"Unknown language: {language}")
    with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as db:
        old = json.loads(db.execute("SELECT value FROM metadata WHERE key='index'").fetchone()[0])
    contract = data_contract(cfg)

    def other_languages(c):
        c = copy.deepcopy(c)
        for key in ("train_manifest", "eval_manifest"):
            c[key].pop(language, None)
        return c

    if other_languages(old["data_contract"]) != other_languages(contract):
        raise ValueError(
            "Only the selected language manifests may change; other settings require a full scan"
        )
    token_hash = hashlib.sha256(processor.tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
    if old["tokenizer_sha256"] != token_hash:
        raise ValueError("Tokenizer changed; a full scan is required")
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.with_name(destination.name + ".scan.lock")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    published = None

    def progress(event, **fields):
        print(json.dumps(dict(event=event, language=language, **fields)), flush=True)

    try:
        with tempfile.TemporaryDirectory(prefix="gqasr-rescan-", dir=temp_dir) as scratch:
            scratch = Path(scratch)
            subset = copy.deepcopy(cfg)
            subset["index_path"] = str(scratch / "language.sqlite")
            subset["language_batch_counts"] = {language: cfg["language_batch_counts"][language]}
            for field in ("train_manifest", "eval_manifest"):
                subset[field] = {language: cfg[field][language]}
            progress("language_rescan_start", source_index=str(source), output=str(destination))
            fresh = scan(
                subset,
                processor,
                verify_audio=True,
                workers=workers,
                chunk_size=chunk_size,
                progress_seconds=progress_seconds,
                temp_dir=str(scratch),
            )
            merged = scratch / "merged.sqlite"
            with (
                sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as src,
                sqlite3.connect(merged) as db,
            ):
                import time

                last = [0.0]

                def backup_progress(status, remaining, total):
                    if time.monotonic() - last[0] >= progress_seconds or not remaining:
                        progress(
                            "copy_existing_index", pages_remaining=remaining, pages_total=total
                        )
                        last[0] = time.monotonic()

                src.backup(db, pages=4096, progress=backup_progress)
                db.execute("ATTACH DATABASE ? AS fresh", (subset["index_path"],))
                # Different languages cannot silently reuse/conflict on an audio path;
                # train/eval recording leakage is also checked across languages.
                conflict = db.execute(
                    """SELECT f.audio FROM fresh.samples f JOIN main.samples s
                    ON f.audio=s.audio WHERE s.language!=? LIMIT 1""",
                    (language,),
                ).fetchone()
                if conflict:
                    raise ValueError(f"Audio overlap with retained language: {conflict[0]}")
                conflict = db.execute(
                    """SELECT f.recording_id FROM fresh.samples f JOIN main.samples s
                    ON f.recording_id=s.recording_id AND f.split!=s.split
                    WHERE f.recording_id!='' AND s.language!=? LIMIT 1""",
                    (language,),
                ).fetchone()
                if conflict:
                    raise ValueError(f"Train/eval recording leakage: {conflict[0]}")
                db.execute("DELETE FROM main.samples WHERE language=?", (language,))
                db.execute("INSERT INTO main.samples SELECT * FROM fresh.samples")
                metadata = copy.deepcopy(old)
                metadata["data_contract"] = contract
                for split in ("train", "eval"):
                    metadata["counts"][split][language] = fresh["counts"][split][language]
                replaced = {
                    p
                    for field in ("train_manifest", "eval_manifest")
                    for p in old["data_contract"][field][language]
                }
                metadata["sources"] = [
                    s for s in old["sources"] if s["path"] not in replaced
                ] + fresh["sources"]
                metadata["audio_headers_checked"] = bool(old["audio_headers_checked"])
                # Original skip totals cannot be separated by language retrospectively.
                metadata.pop("skipped", None)
                metadata["language_rescan"] = {
                    "language": language,
                    "source_index": str(source),
                    "previous_counts": old["counts"],
                    "rescanned_skipped": fresh["skipped"],
                    "rescanned_audio_headers_checked": True,
                    "retained_records": "Copied from completed source index without rereading audio or manifests; unchanged sources assumed.",
                }
                db.execute("UPDATE metadata SET value=? WHERE key='index'", (json.dumps(metadata),))
                db.commit()
            fd, published = tempfile.mkstemp(prefix=".gqasr-rescan-", dir=destination.parent)
            os.close(fd)
            progress("publishing_rescanned_index")
            size = merged.stat().st_size
            with merged.open("rb") as src, open(published, "wb") as dst:
                copied = 0
                last_progress = time.monotonic()
                while block := src.read(16 * 1024 * 1024):
                    dst.write(block)
                    copied += len(block)
                    if time.monotonic() - last_progress >= progress_seconds:
                        progress("publish_progress", copied_bytes=copied, total_bytes=size)
                        last_progress = time.monotonic()
                dst.flush()
                os.fsync(dst.fileno())
            os.link(published, destination)
            progress("language_rescan_complete", counts=metadata["counts"])
            return metadata
    finally:
        if published:
            Path(published).unlink(missing_ok=True)
        lock.unlink(missing_ok=True)
