from __future__ import annotations

import json
import logging
import math
import os
import re
from array import array
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from tqdm.auto import tqdm

from .audio import AudioDurationError, AudioLoadingError, load_mono_audio, validate_audio_duration

_PROGRESS_MIN_BYTES = 1024 * 1024
LOGGER = logging.getLogger("qasr")


class ManifestError(ValueError):
    """Raised when a JSONL manifest contains a malformed record."""


class _UnusableTranscript(ManifestError):
    """Internal signal for records that should be skipped during indexing."""


class TranscriptLengthError(ValueError):
    """Raised when a transcript cannot fit in the target window."""


@dataclass(frozen=True)
class ManifestStats:
    total_records: int
    kept_records: int
    skipped_by_duration: int
    skipped_by_text: int
    missing_duration: int
    total_kept_hours: float


def _first_present(record: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    return None


def _first_usable_text(record: dict[str, Any], names: tuple[str, ...]) -> str | None:
    """Return the first transcript containing at least one letter or number."""
    for name in names:
        value = record.get(name)
        if isinstance(value, str):
            value = value.strip()
            if value and any(character.isalnum() for character in value):
                return value
    return None


def parse_record(
    record: Any,
    *,
    manifest_path: Path,
    line_number: int,
    audio_root: Path | None,
    language: str | None,
    empty_target_ok: bool = False,
) -> dict[str, Any]:
    location = f"{manifest_path}:{line_number}"
    if not isinstance(record, dict):
        raise ManifestError(f"{location}: each line must contain a JSON object")

    text_value = _first_usable_text(record, ("text", "transcript"))
    if text_value is None:
        if not empty_target_ok:
            raise _UnusableTranscript(
                f"{location}: missing, blank, or punctuation-only text or transcript"
            )
        # Non-speech / silence data: the model must learn to emit nothing for
        # this audio, so an empty target string is the correct label.
        text_value = ""

    audio_value = _first_present(record, ("audio_filepath", "wav_path"))
    duration_value = record.get("duration")
    if not isinstance(audio_value, str) or not audio_value.strip():
        raise ManifestError(f"{location}: missing non-empty audio_filepath or wav_path")

    duration: float | None = None
    if duration_value is not None:
        try:
            duration = float(duration_value)
        except (TypeError, ValueError) as exc:
            raise ManifestError(f"{location}: duration must be a number") from exc
        if not math.isfinite(duration) or duration <= 0:
            raise ManifestError(f"{location}: duration must be finite and greater than zero")

    audio_value = re.sub(r"^/vast", "/lustrefs/taiga/vast40", audio_value)
    audio_path = Path(audio_value).expanduser()
    if not audio_path.is_absolute():
        base = audio_root if audio_root is not None else manifest_path.parent
        audio_path = base / audio_path

    return {
        "audio_path": str(audio_path),
        "text": text_value,
        "duration": duration,
        "line_number": line_number,
        "language": language,
    }


class JsonlSpeechDataset:
    """Map-style JSONL dataset backed by compact byte offsets.

    The manifest is scanned once to validate records and apply duration filters when
    duration metadata is available. Audio is decoded only when a worker requests it.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        min_duration_seconds: float,
        max_duration_seconds: float,
        audio_root: str | Path | None = None,
        language: str | None = None,
        validate_audio_paths: bool = False,
        empty_target_ok: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest does not exist: {self.manifest_path}")
        self.audio_root = Path(audio_root).expanduser().resolve() if audio_root else None
        self.language = language
        self.min_duration_seconds = min_duration_seconds
        self.max_duration_seconds = max_duration_seconds
        self.validate_audio_paths = validate_audio_paths
        self.empty_target_ok = empty_target_ok

        self._offsets: array[int] = array("Q")
        self._line_numbers: array[int] = array("Q")
        self._file: BinaryIO | None = None
        self._file_pid: int | None = None
        self.stats = self._build_index()
        if not self._offsets:
            raise ValueError(
                f"No usable records in {self.manifest_path} after applying duration range "
                f"[{self.min_duration_seconds}, {self.max_duration_seconds}] seconds"
            )

    def _build_index(self) -> ManifestStats:
        total = 0
        skipped_by_duration = 0
        skipped_by_text = 0
        missing_duration = 0
        total_kept_seconds = 0.0

        manifest_size = self.manifest_path.stat().st_size
        try:
            show_progress = int(os.environ.get("RANK", "0")) == 0
        except ValueError:
            show_progress = True
        show_progress = show_progress and manifest_size >= _PROGRESS_MIN_BYTES
        description = f"Indexing {self.manifest_path.parent.name}/{self.manifest_path.name}"

        with self.manifest_path.open("rb") as handle, tqdm(
            total=manifest_size,
            desc=description,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            mininterval=1.0,
            disable=not show_progress,
        ) as progress:
            line_number = 0
            while True:
                offset = handle.tell()
                raw_line = handle.readline()
                if not raw_line:
                    break
                progress.update(len(raw_line))
                line_number += 1
                if not raw_line.strip():
                    continue
                total += 1
                try:
                    record = json.loads(raw_line)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise ManifestError(
                        f"{self.manifest_path}:{line_number}: invalid JSON: {exc}"
                    ) from exc
                try:
                    parsed = parse_record(
                        record,
                        manifest_path=self.manifest_path,
                        line_number=line_number,
                        audio_root=self.audio_root,
                        language=self.language,
                        empty_target_ok=self.empty_target_ok,
                    )
                except _UnusableTranscript:
                    skipped_by_text += 1
                    continue

                duration = parsed["duration"]
                if duration is None:
                    missing_duration += 1
                elif duration < self.min_duration_seconds or duration > self.max_duration_seconds:
                    skipped_by_duration += 1
                    continue
                else:
                    total_kept_seconds += duration

                if self.validate_audio_paths and not Path(parsed["audio_path"]).is_file():
                    raise FileNotFoundError(
                        f"{self.manifest_path}:{line_number}: audio file does not exist: "
                        f"{parsed['audio_path']}"
                    )
                self._offsets.append(offset)
                self._line_numbers.append(line_number)

            progress.set_postfix(
                kept=len(self._offsets),
                skipped_duration=skipped_by_duration,
                skipped_text=skipped_by_text,
                deferred_duration=missing_duration,
                refresh=False,
            )

        return ManifestStats(
            total_records=total,
            kept_records=len(self._offsets),
            skipped_by_duration=skipped_by_duration,
            skipped_by_text=skipped_by_text,
            missing_duration=missing_duration,
            total_kept_hours=total_kept_seconds / 3600.0,
        )

    def __len__(self) -> int:
        return len(self._offsets)

    def _get_file(self) -> BinaryIO:
        current_pid = os.getpid()
        if self._file is None or self._file.closed or self._file_pid != current_pid:
            if self._file is not None and not self._file.closed:
                self._file.close()
            self._file = self.manifest_path.open("rb")
            self._file_pid = current_pid
        return self._file

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        handle = self._get_file()
        handle.seek(self._offsets[index])
        raw_line = handle.readline()
        line_number = int(self._line_numbers[index])
        try:
            record = json.loads(raw_line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ManifestError(
                f"{self.manifest_path}:{line_number}: invalid JSON: {exc}"
            ) from exc
        return parse_record(
            record,
            manifest_path=self.manifest_path,
            line_number=line_number,
            audio_root=self.audio_root,
            language=self.language,
            empty_target_ok=self.empty_target_ok,
        )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_file"] = None
        state["_file_pid"] = None
        return state

    def __del__(self) -> None:
        file_handle = getattr(self, "_file", None)
        if file_handle is not None:
            file_handle.close()


class CombinedSpeechDataset:
    """Concatenate multiple manifest datasets into one map-style dataset."""

    def __init__(self, datasets: list[JsonlSpeechDataset]) -> None:
        if not datasets:
            raise ValueError("At least one training dataset is required")
        self.datasets = tuple(datasets)
        self.cumulative_sizes: list[int] = []
        running_size = 0
        for dataset in self.datasets:
            running_size += len(dataset)
            self.cumulative_sizes.append(running_size)

        self.stats = ManifestStats(
            total_records=sum(dataset.stats.total_records for dataset in self.datasets),
            kept_records=sum(dataset.stats.kept_records for dataset in self.datasets),
            skipped_by_duration=sum(
                dataset.stats.skipped_by_duration for dataset in self.datasets
            ),
            skipped_by_text=sum(dataset.stats.skipped_by_text for dataset in self.datasets),
            missing_duration=sum(dataset.stats.missing_duration for dataset in self.datasets),
            total_kept_hours=sum(dataset.stats.total_kept_hours for dataset in self.datasets),
        )

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        dataset_index = bisect_right(self.cumulative_sizes, index)
        previous_size = 0 if dataset_index == 0 else self.cumulative_sizes[dataset_index - 1]
        return self.datasets[dataset_index][index - previous_size]


class ResilientAudioDataset:
    """Decode lazily and replace unreadable audio or oversized transcripts.

    Skip counters (see ``skip_stats``) are per-process: every DataLoader worker
    holds its own copy of this dataset, so aggregate across workers and ranks
    when reporting a global drop rate. With ``persistent_workers=True`` the
    counters accumulate across epochs instead of resetting.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        sampling_rate: int,
        min_audio_seconds: float,
        max_audio_seconds: float,
        tokenizer: Any | None = None,
        max_target_length: int | None = None,
        truncate_long_transcripts: bool = False,
    ) -> None:
        if len(dataset) == 0:
            raise ValueError("Cannot wrap an empty audio dataset")
        self.dataset = dataset
        self.sampling_rate = sampling_rate
        self.min_audio_seconds = min_audio_seconds
        self.max_audio_seconds = max_audio_seconds
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length
        self.truncate_long_transcripts = truncate_long_transcripts
        self._invalid_indices: set[int] = set()
        self._audio_load_errors = 0
        self._duration_errors = 0
        self._transcript_errors = 0
        self._substitutions = 0
        self._next_failure_summary = 1

    @property
    def skip_stats(self) -> dict[str, int]:
        """Per-process drop and substitution counts (see class docstring)."""
        return {
            "audio_load_errors": self._audio_load_errors,
            "duration_errors": self._duration_errors,
            "transcript_errors": self._transcript_errors,
            "substitutions": self._substitutions,
            "invalid_indices": len(self._invalid_indices),
        }

    def _validate_transcript(self, feature: dict[str, Any]) -> tuple[list[int] | None, str]:
        if self.tokenizer is None or self.max_target_length is None:
            return None, feature["text"]
        transcript_ids = list(
            self.tokenizer(
                feature["text"],
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
        )
        if len(transcript_ids) <= self.max_target_length:
            return transcript_ids, feature["text"]
        if not self.truncate_long_transcripts:
            raise TranscriptLengthError(
                f"{feature['audio_path']} (manifest line {feature.get('line_number', '?')}): "
                f"transcript has {len(transcript_ids)} tokens, exceeding "
                f"max_target_length={self.max_target_length}"
            )
        transcript_ids = transcript_ids[: self.max_target_length]
        return transcript_ids, self.tokenizer.decode(
            transcript_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        dataset_size = len(self)
        if index < 0:
            index += dataset_size
        if index < 0 or index >= dataset_size:
            raise IndexError(index)

        for offset in range(dataset_size):
            candidate_index = (index + offset) % dataset_size
            if candidate_index in self._invalid_indices:
                continue
            feature = self.dataset[candidate_index]
            try:
                transcript_ids, transcript = self._validate_transcript(feature)
                waveform = load_mono_audio(feature["audio_path"], self.sampling_rate)
                validate_audio_duration(
                    waveform,
                    path=feature["audio_path"],
                    sampling_rate=self.sampling_rate,
                    min_audio_seconds=self.min_audio_seconds,
                    max_audio_seconds=self.max_audio_seconds,
                )
            except (AudioLoadingError, AudioDurationError, TranscriptLengthError) as exc:
                if isinstance(exc, AudioLoadingError):
                    self._audio_load_errors += 1
                elif isinstance(exc, AudioDurationError):
                    self._duration_errors += 1
                else:
                    self._transcript_errors += 1
                self._invalid_indices.add(candidate_index)
                LOGGER.debug(
                    "Skipping unusable sample at dataset index %d: %s", candidate_index, exc
                )
                failures = (
                    self._audio_load_errors + self._duration_errors + self._transcript_errors
                )
                if failures == self._next_failure_summary:
                    self._next_failure_summary *= 10
                    LOGGER.warning(
                        "Dropped %d unusable samples so far in this process (dataset size %d): "
                        "audio_load_errors=%d duration_errors=%d transcript_errors=%d "
                        "substitutions=%d",
                        failures,
                        dataset_size,
                        self._audio_load_errors,
                        self._duration_errors,
                        self._transcript_errors,
                        self._substitutions,
                    )
                continue

            result = dict(feature)
            result["waveform"] = waveform
            result["text"] = transcript
            if transcript_ids is not None:
                result["transcript_ids"] = transcript_ids
            if candidate_index != index:
                self._substitutions += 1
                LOGGER.warning(
                    "Replaced unusable dataset index %d with index %d (%s)",
                    index,
                    candidate_index,
                    feature["audio_path"],
                )
            return result

        raise RuntimeError(
            "Could not find a usable sample after checking every sample in the effective "
            f"dataset ({dataset_size} records)"
        )


__all__ = [
    "CombinedSpeechDataset",
    "JsonlSpeechDataset",
    "ManifestError",
    "ManifestStats",
    "ResilientAudioDataset",
    "TranscriptLengthError",
    "parse_record",
]
