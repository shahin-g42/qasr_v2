"""Shared, atomic, memory-mapped indexes for immutable JSONL manifests."""

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from array import array
from dataclasses import asdict
from pathlib import Path

import numpy as np
from qasr.data import (
    JsonlSpeechDataset,
    ManifestError,
    ManifestStats,
    _UnusableTranscript,
    parse_record,
)
from tqdm.auto import tqdm


class IndexedSpeechDataset(JsonlSpeechDataset):
    def __init__(self, *args, cache_directory, **kwargs):
        self.cache_directory = Path(cache_directory)
        super().__init__(*args, **kwargs)

    def _identity(self):
        stat = self.manifest_path.stat()
        return dict(
            version=1,
            path=str(self.manifest_path),
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            ctime_ns=stat.st_ctime_ns,
            minimum=self.min_duration_seconds,
            maximum=self.max_duration_seconds,
            audio_root=str(self.audio_root),
            language=self.language,
            empty_target_ok=self.empty_target_ok,
        )

    def _build_index(self):
        # Path validation must still examine the actual files when requested.
        if self.validate_audio_paths:
            return super()._build_index()
        identity = self._identity()
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        self.index_directory = self.cache_directory / key
        with (self.cache_directory / f"{key}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not self.index_directory.exists():
                self._publish_index(identity)
            metadata = json.loads((self.index_directory / "metadata.json").read_text())
            if metadata["identity"] != identity:
                raise ValueError("Manifest cache identity mismatch")
            self._open_arrays()
            expected = metadata["stats"]["kept_records"]
            if any(len(a) != expected for a in (self._offsets, self._line_numbers, self.durations)):
                raise ValueError(
                    "Incomplete manifest index; remove its cache directory and rebuild"
                )
            return ManifestStats(**metadata["stats"])

    def _open_arrays(self):
        self._offsets = np.load(self.index_directory / "offsets.npy", mmap_mode="r")
        self._line_numbers = np.load(self.index_directory / "lines.npy", mmap_mode="r")
        self.durations = np.load(self.index_directory / "durations.npy", mmap_mode="r")

    def _publish_index(self, identity):
        offsets, lines, durations = array("Q"), array("Q"), array("f")
        total = skipped_duration = skipped_text = missing = 0
        seconds = 0.0
        staging = Path(
            tempfile.mkdtemp(prefix=f".{self.index_directory.name}-", dir=self.cache_directory)
        )
        try:
            with (
                self.manifest_path.open("rb") as stream,
                tqdm(
                    total=identity["size"],
                    unit="B",
                    unit_scale=True,
                    desc=f"Caching {self.manifest_path.name}",
                    disable=int(os.environ.get("RANK", "0")) != 0,
                ) as progress,
            ):
                line_number = 0
                while True:
                    offset, raw = stream.tell(), stream.readline()
                    if not raw:
                        break
                    line_number += 1
                    progress.update(len(raw))
                    if not raw.strip():
                        continue
                    total += 1
                    try:
                        row = parse_record(
                            json.loads(raw),
                            manifest_path=self.manifest_path,
                            line_number=line_number,
                            audio_root=self.audio_root,
                            language=self.language,
                            empty_target_ok=self.empty_target_ok,
                        )
                    except _UnusableTranscript:
                        skipped_text += 1
                        continue
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        raise ManifestError(f"{self.manifest_path}:{line_number}: {exc}") from exc
                    duration = row["duration"]
                    if duration is None:
                        missing += 1
                    elif not self.min_duration_seconds <= duration <= self.max_duration_seconds:
                        skipped_duration += 1
                        continue
                    else:
                        seconds += duration
                    offsets.append(offset)
                    lines.append(line_number)
                    durations.append(float("nan") if duration is None else duration)
            if self._identity() != identity:
                raise ValueError("Manifest changed while building its index")
            stats = ManifestStats(
                total, len(offsets), skipped_duration, skipped_text, missing, seconds / 3600
            )
            for name, values in (("offsets", offsets), ("lines", lines), ("durations", durations)):
                np.save(staging / f"{name}.npy", np.asarray(values))
            (staging / "metadata.json").write_text(
                json.dumps(dict(identity=identity, stats=asdict(stats)))
            )
            staging.rename(self.index_directory)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def __getstate__(self):
        state = super().__getstate__()
        if hasattr(self, "index_directory"):
            for key in ("_offsets", "_line_numbers", "durations"):
                state.pop(key, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if hasattr(self, "index_directory"):
            self._open_arrays()
