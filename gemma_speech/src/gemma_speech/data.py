"""Paired manifests, leakage checks, audio loading and deterministic sampling."""

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler

from .text import normalize_text


@dataclass(frozen=True)
class Record:
    id: str
    audio: str
    text: str
    language: str
    speaker: str
    tts_text: str
    recording_id: str = ""
    dialect: str = "unspecified"


def read_manifest(path, task="asr"):
    path = Path(path).resolve()
    records, ids, paths = [], set(), set()
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            where = f"{path}:{line_no}"
            audio = row.get("audio_filepath") or row.get("audio") or row.get("wav_path")
            text = row.get("text", row.get("transcript"))
            language = row.get("language")
            if not isinstance(audio, str) or not isinstance(text, str) or not language:
                raise ValueError(f"{where}: audio path, text/transcript and language required")
            audio = str((path.parent / Path(audio).expanduser()).resolve())
            uid = str(row.get("id", audio))
            if uid in ids or audio in paths:
                raise ValueError(f"{where}: duplicate ID or audio path")
            ids.add(uid)
            paths.add(audio)
            tts_text = normalize_text(row.get("tts_text", text))
            speaker = str(row.get("speaker_id", ""))
            if task == "tts" and (not tts_text or not speaker):
                raise ValueError(f"{where}: TTS requires nonempty tts_text/text and speaker_id")
            records.append(
                Record(
                    uid,
                    audio,
                    normalize_text(text),
                    str(language),
                    speaker,
                    tts_text,
                    str(row.get("recording_id", "")),
                    str(row.get("dialect", "unspecified")),
                )
            )
    if not records:
        raise ValueError(f"empty manifest: {path}")
    return records


def check_disjoint(train, validation):
    for field in ("id", "audio", "recording_id"):
        left = {getattr(r, field) for r in train if getattr(r, field)}
        right = {getattr(r, field) for r in validation if getattr(r, field)}
        if overlap := left & right:
            raise ValueError(f"train/validation leakage in {field}: {next(iter(overlap))}")


def load_audio(path, sample_rate, max_seconds):
    import soundfile as sf
    from scipy.signal import resample_poly

    info = sf.info(path)
    if not 0.05 <= info.frames / info.samplerate <= max_seconds:
        raise ValueError(f"{path}: duration outside [0.05, {max_seconds}] seconds; segment first")
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if not np.isfinite(audio).all():
        raise ValueError(f"{path}: nonfinite waveform")
    if sr != sample_rate:
        divisor = math.gcd(sr, sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, sr // divisor).astype(np.float32)
    return np.ascontiguousarray(audio)


def sampling_weights(records, language_weights=None):
    counts = Counter(r.language for r in records)
    masses = language_weights or {lang: count**0.5 for lang, count in counts.items()}
    if set(masses) != set(counts):
        raise ValueError(
            f"language_weights must cover exactly the training languages: {sorted(counts)}"
        )
    if any(v <= 0 or not math.isfinite(v) for v in masses.values()):
        raise ValueError("language sampling weights must be finite and positive")
    return torch.tensor(
        [masses[r.language] / counts[r.language] for r in records], dtype=torch.double
    )


class WeightedDistributedSampler(Sampler):
    """All ranks draw the same global sequence, then take disjoint stride positions.

    Sampling is with replacement: repeated examples are intentional, including across ranks.
    Epoch length is rounded to complete optimizer updates on every rank.
    """

    def __init__(self, weights, samples, batch_size, accumulation, rank=0, world=1, seed=42):
        self.weights, self.rank, self.world, self.seed = weights, rank, world, seed
        quantum = world * batch_size * accumulation
        self.total = math.ceil(samples / quantum) * quantum
        self.epoch = 0
        self.offset = 0

    def __len__(self):
        return self.total // self.world - self.offset

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.weights, self.total, True, generator=generator)
        return iter(indices[self.rank :: self.world][self.offset :].tolist())


def pad_sequences(sequences, value=0):
    lengths = torch.tensor([len(x) for x in sequences], dtype=torch.long)
    if not len(lengths) or lengths.max() == 0:
        # Empty ASR transcripts are valid; allocate one masked padding position.
        width = 1
    else:
        width = int(lengths.max())
    ids = torch.full((len(sequences), width), value, dtype=torch.long)
    for i, seq in enumerate(sequences):
        ids[i, : len(seq)] = torch.as_tensor(seq, dtype=torch.long)
    mask = torch.arange(width)[None] < lengths[:, None]
    return ids, mask
