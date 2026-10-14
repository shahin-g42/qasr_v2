"""Disk-backed manifest indexing and exact per-update language sampling."""

import hashlib
import json
import math
import multiprocessing
import os
import shutil
import sqlite3
import tempfile
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .scan_workers import check_scan_batch, initialize_scan_worker, normalize_row  # noqa: F401


def data_contract(cfg):
    return {
        k: cfg[k]
        for k in (
            "train_manifest",
            "eval_manifest",
            "audio_root",
            "audio_path_rewrites",
            "min_duration_seconds",
            "max_duration_seconds",
            "max_target_length",
        )
    }


def available_scan_workers():
    """Respect the CPU allocation rather than the machine-wide CPU count."""
    if hasattr(os, "sched_getaffinity"):
        count = len(os.sched_getaffinity(0))
    else:
        count = os.cpu_count() or 1
    allocation = os.environ.get("SLURM_CPUS_PER_TASK")
    return max(1, min(count, int(allocation))) if allocation else max(1, count)


def scan(
    cfg,
    processor,
    verify_audio=False,
    workers=1,
    chunk_size=128,
    progress_seconds=5.0,
    temp_dir=None,
):
    workers = available_scan_workers() if workers == 0 else workers
    if workers < 1 or chunk_size < 1 or progress_seconds <= 0:
        raise ValueError("scan workers/chunk size/progress interval must be positive")
    destination = Path(cfg["index_path"])
    if destination.exists():
        raise FileExistsError(f"index exists; use a new index_path to rebuild: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.with_name(destination.name + ".scan.lock")
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(
            f"scan lock exists: {lock}; stop the owning scan before removing a stale lock"
        ) from exc
    os.write(lock_fd, f"pid={os.getpid()} host={__import__('socket').gethostname()}\n".encode())
    os.close(lock_fd)
    pool, db, temporary, published_temp = None, None, None, None
    started = last_progress = time.monotonic()
    counts, skipped, sources = Counter(), Counter(), []
    processed_bytes = processed_rows = 0
    entries = [
        (split, lang, str(path))
        for split, field in (("train", "train_manifest"), ("eval", "eval_manifest"))
        for lang, paths in cfg[field].items()
        for path in paths
    ]
    total_bytes = 0

    def progress(event, path=None, force=False):
        nonlocal last_progress
        now = time.monotonic()
        if not force and now - last_progress < progress_seconds:
            return
        elapsed = max(now - started, 0.001)
        remaining = max(total_bytes - processed_bytes, 0)
        print(
            json.dumps(
                dict(
                    event=event,
                    workers=workers,
                    manifest=path,
                    manifests_done=len(sources),
                    manifests_total=len(entries),
                    rows_checked=processed_rows,
                    rows_kept=sum(counts.values()),
                    rows_skipped=sum(skipped.values()),
                    skipped=dict(skipped),
                    rows_per_second=round(processed_rows / elapsed, 1),
                    elapsed_seconds=round(elapsed, 1),
                    manifest_bytes_percent=round(100 * processed_bytes / max(1, total_bytes), 2),
                    estimated_remaining_seconds=round(remaining * elapsed / processed_bytes)
                    if processed_bytes
                    else None,
                )
            ),
            flush=True,
        )
        last_progress = now

    try:
        total_bytes = sum(Path(path).stat().st_size for _, _, path in entries)
        # SQLite random writes stay on node-local scratch; only the finished index is copied.
        fd, temporary = tempfile.mkstemp(prefix="gqasr-index-", suffix=".sqlite", dir=temp_dir)
        os.close(fd)
        db = sqlite3.connect(temporary)
        db.execute("PRAGMA journal_mode=OFF")
        db.execute("PRAGMA synchronous=OFF")
        db.execute("PRAGMA cache_size=-65536")
        db.execute(
            "CREATE TABLE samples (split TEXT, language TEXT, ordinal INTEGER, audio TEXT, text TEXT, duration REAL, recording_id TEXT, dialect TEXT, source TEXT, line INTEGER, PRIMARY KEY(split,language,ordinal), UNIQUE(split,audio))"
        )
        db.execute("CREATE INDEX audio_index ON samples(audio)")
        db.execute("CREATE INDEX recording_index ON samples(recording_id)")
        db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        tokenizer_json = processor.tokenizer.backend_tokenizer.to_str()
        worker_args = (tokenizer_json, data_contract(cfg), verify_audio)
        progress("scan_start", force=True)
        print(
            json.dumps(
                dict(
                    event="scan_scratch",
                    directory=str(Path(temporary).parent),
                    free_gib=round(shutil.disk_usage(Path(temporary).parent).free / 2**30, 1),
                    chunk_size=chunk_size,
                    max_pending_batches=2 * workers,
                )
            ),
            flush=True,
        )
        if workers > 1:
            pool = multiprocessing.get_context("spawn").Pool(
                workers, initializer=initialize_scan_worker, initargs=worker_args
            )
        else:
            initialize_scan_worker(*worker_args)
        for split, language, path in entries:
            progress("manifest_start", path, force=True)
            hasher, source_kept = hashlib.sha256(), 0
            with Path(path).open("rb") as stream:
                lines = enumerate(stream, 1)
                pending = deque()
                exhausted = False
                while pending or not exhausted:
                    while not exhausted and len(pending) < 2 * workers:
                        batch = []
                        for _ in range(chunk_size):
                            try:
                                item = next(lines)
                            except StopIteration:
                                exhausted = True
                                break
                            hasher.update(item[1])
                            batch.append(item)
                        if batch:
                            size = sum(len(line) for _, line in batch)
                            task = (path, language, batch)
                            pending.append(
                                (
                                    pool.apply_async(check_scan_batch, task)
                                    if pool
                                    else check_scan_batch(*task),
                                    size,
                                )
                            )
                    if not pending:
                        break
                    result, size = pending.popleft()
                    if pool:
                        while True:
                            try:
                                results = result.get(timeout=progress_seconds)
                                break
                            except multiprocessing.TimeoutError:
                                progress("scan_progress", path, force=True)
                    else:
                        results = result
                    for line_number, row, reason in results:
                        processed_rows += 1
                        if reason:
                            skipped[reason] += 1
                            continue
                        audio, text, duration, recording, dialect = row
                        try:
                            previous = db.execute(
                                "SELECT language,text FROM samples WHERE split=? AND audio=?",
                                (split, audio),
                            ).fetchone()
                            if previous:
                                if previous != (language, text):
                                    raise ValueError("conflicting duplicate audio/transcript")
                                skipped["identical_duplicate"] += 1
                                continue
                            if db.execute(
                                "SELECT 1 FROM samples WHERE split!=? AND audio=? LIMIT 1",
                                (split, audio),
                            ).fetchone():
                                raise ValueError("train/eval audio leakage")
                            if (
                                recording
                                and db.execute(
                                    "SELECT 1 FROM samples WHERE split!=? AND recording_id=? LIMIT 1",
                                    (split, recording),
                                ).fetchone()
                            ):
                                raise ValueError("train/eval recording leakage")
                            counts[(split, language)] += 1
                            db.execute(
                                "INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?)",
                                (
                                    split,
                                    language,
                                    counts[(split, language)],
                                    audio,
                                    text,
                                    duration,
                                    recording,
                                    dialect,
                                    path,
                                    line_number,
                                ),
                            )
                            source_kept += 1
                        except Exception as exc:
                            raise ValueError(f"{path}:{line_number}: {exc}") from exc
                    processed_bytes += size
                    progress("scan_progress", path)
            db.commit()
            sources.append(dict(path=path, sha256=hasher.hexdigest(), rows_kept=source_kept))
            progress("manifest_complete", path, force=True)
        for split in ("train", "eval"):
            for language in cfg["language_batch_counts"]:
                if not counts[(split, language)]:
                    raise ValueError(f"no usable {split} records for {language}")
        metadata = dict(
            format_version=1,
            data_contract=data_contract(cfg),
            sources=sources,
            counts={
                split: {lang: counts[(split, lang)] for lang in cfg["language_batch_counts"]}
                for split in ("train", "eval")
            },
            skipped=dict(skipped),
            audio_headers_checked=verify_audio,
            tokenizer_sha256=hashlib.sha256(tokenizer_json.encode()).hexdigest(),
        )
        db.execute("INSERT INTO metadata VALUES ('index',?)", (json.dumps(metadata),))
        db.commit()
        db.close()
        db = None
        if pool:
            pool.close()
            pool.join()
            pool = None
        progress("publishing_index", force=True)
        fd, published_temp = tempfile.mkstemp(prefix=".gqasr-index-", dir=destination.parent)
        os.close(fd)
        index_size = Path(temporary).stat().st_size
        copied = 0
        with open(temporary, "rb") as source, open(published_temp, "wb") as target:
            while block := source.read(16 * 1024 * 1024):
                target.write(block)
                copied += len(block)
                if time.monotonic() - last_progress >= progress_seconds:
                    print(
                        json.dumps(
                            dict(
                                event="publishing_index_progress",
                                copied_bytes=copied,
                                total_bytes=index_size,
                                percent=round(100 * copied / max(1, index_size), 2),
                            )
                        ),
                        flush=True,
                    )
                    last_progress = time.monotonic()
        with open(published_temp, "rb") as complete:
            os.fsync(complete.fileno())
        # Atomic no-clobber publication on the destination filesystem.
        os.link(published_temp, destination)
        progress("scan_complete", force=True)
        return metadata
    finally:
        if pool:
            pool.terminate()
            pool.join()
        if db:
            db.close()
        for path in (temporary, published_temp):
            if path:
                Path(path).unlink(missing_ok=True)
        lock.unlink(missing_ok=True)


class ManifestIndex(Dataset):
    def __init__(self, path, split):
        self.path, self.split, self.db = str(Path(path).resolve()), split, None
        with sqlite3.connect(f"file:{self.path}?mode=ro", uri=True) as db:
            self.metadata = json.loads(
                db.execute("SELECT value FROM metadata WHERE key='index'").fetchone()[0]
            )
        self.counts = self.metadata["counts"][split]

    def __len__(self):
        return sum(self.counts.values())

    def __getstate__(self):
        return dict(self.__dict__, db=None)

    def __getitem__(self, key):
        language, ordinal = key
        if self.db is None:
            self.db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        row = self.db.execute(
            "SELECT audio,text,duration,dialect FROM samples WHERE split=? AND language=? AND ordinal=?",
            (self.split, language, int(ordinal)),
        ).fetchone()
        if row is None:
            raise IndexError(key)
        return {
            "id": f"{self.split}:{language}:{ordinal}",
            "audio": row[0],
            "text": row[1],
            "duration": row[2],
            "dialect": row[3],
            "language": language,
        }


class StratifiedUpdateSampler(Sampler):
    """Exact source-language quotas over one accumulated update on EVERY rank.

    Stateless per-rank/per-update RNG permits exact restart without storing a
    corpus-sized shuffled index. Repeated examples are intentional sampling.
    """

    def __init__(self, counts, quotas, batch_size, steps, rank=0, world=1, seed=42, start=0):
        self.counts, self.quotas, self.batch_size, self.steps = counts, quotas, batch_size, steps
        self.rank, self.world, self.seed, self.start = rank, world, seed, start
        self.accumulation = sum(quotas.values()) // batch_size
        if (
            set(counts) != set(quotas)
            or any(v < 1 for v in counts.values())
            or any(v < 1 for v in quotas.values())
            or sum(quotas.values()) % batch_size
        ):
            raise ValueError("invalid language quotas or counts")

    def __len__(self):
        return (self.steps - self.start) * self.accumulation

    def __iter__(self):
        for update in range(self.start, self.steps):
            generator = torch.Generator().manual_seed(self.seed + update * self.world + self.rank)
            keys = [
                (language, int(i))
                for language, count in sorted(self.quotas.items())
                for i in torch.randint(1, self.counts[language] + 1, (count,), generator=generator)
            ]
            order = torch.randperm(len(keys), generator=generator).tolist()
            for start in range(0, len(keys), self.batch_size):
                yield [keys[i] for i in order[start : start + self.batch_size]]


def load_wave(path, minimum, maximum):
    import soundfile as sf
    from scipy.signal import resample_poly

    info = sf.info(path)
    if not minimum <= info.frames / info.samplerate <= maximum:
        raise ValueError(f"duration outside training bounds: {path}")
    wave, rate = sf.read(path, dtype="float32", always_2d=True)
    wave = wave.mean(1)
    if not np.isfinite(wave).all():
        raise ValueError(f"nonfinite waveform: {path}")
    if rate != 16000:
        gcd = math.gcd(rate, 16000)
        wave = resample_poly(wave, 16000 // gcd, rate // gcd).astype(np.float32)
    return np.ascontiguousarray(wave)


class Collator:
    def __init__(self, processor, cfg):
        self.processor, self.cfg = processor, cfg

    def __call__(self, rows):
        waves = [
            load_wave(
                r["audio"], self.cfg["min_duration_seconds"], self.cfg["max_duration_seconds"]
            )
            for r in rows
        ]
        batch = self.processor.batch(
            waves,
            [r["text"] for r in rows],
            [r["language"] for r in rows],
            self.cfg["max_target_length"],
        )
        return batch, rows
