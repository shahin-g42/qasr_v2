"""Lightweight scan workers: no torch, Transformers or model weights imported."""

import json
import math
import os
import unicodedata
from pathlib import Path


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
        aliases = {
            "arabic": "ar",
            "english": "en",
            "hindi": "hi",
            "malayalam": "ml",
            "chinese": "zh",
        }
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


_TOKENIZER = None
_CONFIG = None
_VERIFY_AUDIO = False


def initialize_scan_worker(tokenizer_json, cfg, verify_audio):
    global _TOKENIZER, _CONFIG, _VERIFY_AUDIO
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from tokenizers import Tokenizer

    _TOKENIZER = Tokenizer.from_str(tokenizer_json)
    _TOKENIZER.no_padding()
    _TOKENIZER.no_truncation()
    _CONFIG, _VERIFY_AUDIO = cfg, verify_audio


def check_scan_batch(path, language, batch):
    import soundfile as sf

    names = {"ar": "Arabic", "en": "English", "hi": "Hindi", "ml": "Malayalam", "zh": "Chinese"}
    results = []
    for line_number, line in batch:
        if not line.strip():
            continue
        try:
            row = normalize_row(json.loads(line), path, language, _CONFIG)
            reason = None
            if row is None:
                reason = "unusable_transcript"
            else:
                audio, text, duration, recording, dialect = row
                if _VERIFY_AUDIO or duration is None:
                    info = sf.info(audio)
                    duration = info.frames / info.samplerate
                if (
                    not _CONFIG["min_duration_seconds"]
                    <= duration
                    <= _CONFIG["max_duration_seconds"]
                ):
                    reason = "duration"
                elif (
                    len(
                        _TOKENIZER.encode(
                            f"language {names[language]}<asr_text>{text}", add_special_tokens=False
                        ).ids
                    )
                    + 1
                    > _CONFIG["max_target_length"]
                ):
                    reason = "overlong_target"
                row = (audio, text, duration, recording, dialect)
            results.append((line_number, row, reason))
        except Exception as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return results
