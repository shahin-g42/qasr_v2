"""Disk-backed manifest indexing and exact per-update language sampling."""

import hashlib
import json
import math
import os
import sqlite3
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


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


def normalize_row(row, manifest, language, cfg):
    audio = row.get("audio_filepath") or row.get("wav_path") or row.get("audio")
    text = next(
        (row[k] for k in ("text", "transcript") if isinstance(row.get(k), str) and row[k].strip()),
        None,
    )
    if not isinstance(audio, str) or not audio.strip():
        raise ValueError("missing audio path")
    if text is None or not any(c.isalnum() for c in text):
        return None
    row_language = row.get("language")
    if isinstance(row_language, str):
        from .processing import LANGUAGES

        aliases = {name.lower(): code for code, name in LANGUAGES.items()}
        row_language = aliases.get(row_language.lower(), row_language.lower())
    if row_language not in (None, "", language):
        raise ValueError(f"row language {row['language']} conflicts with manifest group {language}")
    for before, after in cfg["audio_path_rewrites"].items():
        if audio == before or audio.startswith(before + "/"):
            audio = after + audio[len(before) :]
            break
    audio = Path(audio).expanduser()
    if not audio.is_absolute():
        audio = Path(cfg["audio_root"] or Path(manifest).parent) / audio
    text = " ".join(unicodedata.normalize("NFC", text).split())
    duration = row.get("duration")
    if duration is not None:
        duration = float(duration)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("invalid duration")
    return (
        str(audio.resolve()),
        text,
        duration,
        str(row.get("recording_id") or ""),
        str(row.get("dialect", "unspecified")),
    )


def scan(cfg, processor, verify_audio=False):
    import soundfile as sf

    destination = Path(cfg["index_path"])
    if destination.exists():
        raise FileExistsError(f"index exists; use a new index_path to rebuild: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".gqasr-index-", dir=destination.parent)
    os.close(fd)
    db = sqlite3.connect(temporary)
    counts, skipped, sources = Counter(), Counter(), []
    try:
        db.execute("PRAGMA journal_mode=OFF")
        db.execute(
            "CREATE TABLE samples (split TEXT, language TEXT, ordinal INTEGER, audio TEXT, text TEXT, duration REAL, recording_id TEXT, dialect TEXT, source TEXT, line INTEGER, PRIMARY KEY(split,language,ordinal), UNIQUE(split,audio))"
        )
        db.execute("CREATE INDEX audio_index ON samples(audio)")
        db.execute("CREATE INDEX recording_index ON samples(recording_id)")
        db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        for split, field in (("train", "train_manifest"), ("eval", "eval_manifest")):
            for language, paths in cfg[field].items():
                for path in paths:
                    hasher, source_kept = hashlib.sha256(), 0
                    with Path(path).open("rb") as f:
                        for line_number, line in enumerate(f, 1):
                            hasher.update(line)
                            if not line.strip():
                                continue
                            try:
                                row = normalize_row(json.loads(line), path, language, cfg)
                                if row is None:
                                    skipped["unusable_transcript"] += 1
                                    continue
                                audio, text, duration, recording, dialect = row
                                if verify_audio or duration is None:
                                    info = sf.info(audio)
                                    duration = info.frames / info.samplerate
                                if (
                                    not cfg["min_duration_seconds"]
                                    <= duration
                                    <= cfg["max_duration_seconds"]
                                ):
                                    skipped["duration"] += 1
                                    continue
                                if len(processor.target(text, language)) > cfg["max_target_length"]:
                                    skipped["overlong_target"] += 1
                                    continue
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
                                        str(path),
                                        line_number,
                                    ),
                                )
                                source_kept += 1
                            except Exception as exc:
                                raise ValueError(f"{path}:{line_number}: {exc}") from exc
                    sources.append(
                        {"path": str(path), "sha256": hasher.hexdigest(), "rows_kept": source_kept}
                    )
                    print(json.dumps({"indexed": str(path), "rows_kept": source_kept}), flush=True)
        for split in ("train", "eval"):
            for language in cfg["language_batch_counts"]:
                if not counts[(split, language)]:
                    raise ValueError(f"no usable {split} records for {language}")
        metadata = {
            "format_version": 1,
            "data_contract": data_contract(cfg),
            "sources": sources,
            "counts": {
                split: {lang: counts[(split, lang)] for lang in cfg["language_batch_counts"]}
                for split in ("train", "eval")
            },
            "skipped": dict(skipped),
            "audio_headers_checked": verify_audio,
            "tokenizer_sha256": hashlib.sha256(
                processor.tokenizer.backend_tokenizer.to_str().encode()
            ).hexdigest(),
        }
        db.execute("INSERT INTO metadata VALUES ('index',?)", (json.dumps(metadata),))
        db.commit()
        db.close()
        os.rename(temporary, destination)
        return metadata
    except BaseException:
        db.close()
        Path(temporary).unlink(missing_ok=True)
        raise


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
