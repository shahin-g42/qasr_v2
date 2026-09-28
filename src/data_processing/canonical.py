"""Canonical record schema and minimal manifest I/O.

Two files per corpus shard, and only two:

``train_<lang>_<dataset>_p####.jsonl``
    The training manifest. Exactly four keys per record, in this order::

        {"audio_filepath": ..., "duration": ..., "text": ..., "lang": ...}

    Nothing else. The dataloader never has to skip a key and the format cannot
    drift as the pipeline grows.

``train_<lang>_<dataset>_p####.meta.jsonl``
    The sidecar. Same order, one record per manifest line, keyed by
    ``audio_filepath``. Carries provenance, accent, quality and the per-stage
    verdicts -- everything needed to audit a corpus after the fact without
    touching the training path.

The split exists because those two goals conflict: training wants a tiny hot
record, auditing wants rich history. Keeping them in one file forces the
dataloader to parse fields it discards.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

LOGGER = logging.getLogger("data_processing.canonical")

#: The complete set of keys permitted in a training manifest record.
MANIFEST_KEYS: tuple[str, ...] = ("audio_filepath", "duration", "text", "lang")

#: Millisecond precision. Audio durations are never known better than this,
#: and rounding keeps the manifest measurably smaller at 10M records.
DURATION_PRECISION = 3


@dataclass(frozen=True, slots=True)
class Sample:
    """One training sample: the minimal, complete unit of the corpus."""

    audio_filepath: str
    duration: float
    text: str
    lang: str

    def to_manifest_dict(self) -> dict[str, object]:
        """Render as a manifest record, keys in canonical order."""
        return {
            "audio_filepath": self.audio_filepath,
            "duration": round(float(self.duration), DURATION_PRECISION),
            "text": self.text,
            "lang": self.lang,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> Sample:
        """Build from a parsed JSON object, tolerating extra source keys."""
        missing = [k for k in MANIFEST_KEYS if k not in raw]
        if missing:
            raise ValueError(f"record is missing required key(s): {missing}")
        return cls(
            audio_filepath=str(raw["audio_filepath"]),
            duration=float(raw["duration"]),
            text=str(raw["text"]),
            lang=str(raw["lang"]),
        )


@dataclass(slots=True)
class Meta:
    """Sidecar record: provenance and quality history for one sample.

    Every field is optional except ``audio_filepath``. A sample that was never
    LLM-processed still gets a sidecar entry so the audit can tell "clean" from
    "not attempted".
    """

    audio_filepath: str
    dataset: str = ""
    accent: str | None = None
    #: 0.0-1.0 composite quality score from the hard gates.
    quality: float | None = None
    #: Transcript exactly as it arrived from the source dataset.
    source_text: str | None = None
    #: Transcript after deterministic normalization, before the LLM.
    normalized_text: str | None = None
    #: Transcript after the full cleaner -> validator -> corrector chain.
    final_text: str | None = None
    itn_applied: bool = False
    #: Diacritics/letters ratio for scripts that use them; None otherwise.
    diacritics_ratio: float | None = None
    #: Per-stage outcome: {"cleaner": "ok", "validator": "pass", ...}.
    stages: dict[str, str] = field(default_factory=dict)
    #: Why the sample was dropped, or None if it was kept.
    reject: str | None = None
    #: Identifier of the duplicate group this sample belongs to, if any.
    dup_group: str | None = None
    #: Free-form measurements (chars/sec, script ratio, ...) for audit.
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Render as a sidecar record, dropping fields that were never set."""
        out = {k: v for k, v in asdict(self).items() if v not in (None, "", {}, False)}
        out["audio_filepath"] = self.audio_filepath
        return out


def _is_gzipped(path: Path) -> bool:
    """Whether a filename asks for gzip.

    Uses ``suffixes`` rather than ``suffix`` so the answer survives a temp name
    that carries the full original stem -- ``.jsonl.gz`` and ``.jsonl.gz.tmp``
    both compress, and disagreeing about that is how a corrupt shard happens.
    """
    return ".gz" in path.suffixes


def _open_write(path: Path):
    """Open a text writer, gzipping when the filename asks for it."""
    if _is_gzipped(path):
        return gzip.open(path, "wt", encoding="utf-8", compresslevel=6)
    return path.open("w", encoding="utf-8")


def _open_read(path: Path):
    """Open a text reader, transparently handling ``.gz``."""
    if _is_gzipped(path):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _atomic_write(path: Path, write_fn) -> None:
    """Write via a sibling temp file then rename, so a crash can't truncate.

    A half-written manifest is worse than no manifest: the next run sees a
    complete-looking file and silently trains on a prefix.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    # Compression follows the DESTINATION, decided here rather than inferred from
    # ``tmp`` by name. Inferring it is what produced plain-text bytes renamed to
    # .jsonl.gz -- a shard that reads fine under open() and dies with
    # BadGzipFile under gzip.open(), i.e. corrupt only for whichever consumer
    # happens to check. ``_open_write`` stays for callers that pass a real path.
    if _is_gzipped(path):
        handle = gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6)  # noqa: SIM115 - closed by `with handle:` below
    else:
        handle = tmp.open("w", encoding="utf-8")
    try:
        with handle:
            write_fn(handle)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_manifest(path: str | Path, samples: Iterable[Sample]) -> int:
    """Write the minimal 4-key training manifest. Returns records written."""
    path = Path(path)
    count = 0

    def emit(handle) -> None:
        nonlocal count
        for sample in samples:
            handle.write(json.dumps(sample.to_manifest_dict(), ensure_ascii=False))
            handle.write("\n")
            count += 1

    _atomic_write(path, emit)
    LOGGER.info("wrote %s (%s records)", path, f"{count:,}")
    return count


def write_sidecar(path: str | Path, metas: Iterable[Meta]) -> int:
    """Write the sidecar metadata file. Returns records written."""
    path = Path(path)
    count = 0

    def emit(handle) -> None:
        nonlocal count
        for meta in metas:
            handle.write(json.dumps(meta.to_dict(), ensure_ascii=False))
            handle.write("\n")
            count += 1

    _atomic_write(path, emit)
    LOGGER.info("wrote %s (%s records)", path, f"{count:,}")
    return count


def read_manifest(path: str | Path) -> Iterator[Sample]:
    """Stream a training manifest. Blank lines are skipped, not fatal."""
    with _open_read(Path(path)) as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield Sample.from_dict(json.loads(line))
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc


def read_sidecar(path: str | Path) -> dict[str, Meta]:
    """Load a sidecar into a dict keyed by ``audio_filepath``."""
    out: dict[str, Meta] = {}
    with _open_read(Path(path)) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            known = set(Meta.__slots__)
            out[raw["audio_filepath"]] = Meta(**{k: v for k, v in raw.items() if k in known})
    return out


def shard_paths(out_dir: str | Path, lang: str, dataset: str, index: int, gzipped: bool = False) -> tuple[Path, Path]:
    """Return ``(manifest_path, sidecar_path)`` for one shard."""
    stem = f"train_{lang}_{dataset}_p{index:04d}"
    suffix = ".jsonl.gz" if gzipped else ".jsonl"
    base = Path(out_dir)
    return base / f"{stem}{suffix}", base / f"{stem}.meta{suffix}"


def iter_shards(out_dir: str | Path, lang: str | None = None, dataset: str | None = None) -> Iterator[tuple[Path, Path]]:
    """Yield ``(manifest, sidecar)`` pairs already present in ``out_dir``."""
    base = Path(out_dir)
    if not base.is_dir():
        return
    for manifest in sorted(base.glob("train_*_p*.jsonl*")):
        if manifest.name.endswith(".meta.jsonl") or manifest.name.endswith(".meta.jsonl.gz"):
            continue
        parts = manifest.name.split("_")
        if len(parts) < 4:
            continue
        if lang and parts[1] != lang:
            continue
        if dataset and dataset not in manifest.name:
            continue
        sidecar = manifest.with_name(manifest.name.replace(".jsonl", ".meta.jsonl"))
        yield manifest, sidecar


__all__ = [
    "DURATION_PRECISION",
    "MANIFEST_KEYS",
    "Meta",
    "Sample",
    "iter_shards",
    "read_manifest",
    "read_sidecar",
    "shard_paths",
    "write_manifest",
    "write_sidecar",
]
