"""Dataset specification and the source protocol.

Everything a dataset needs to be ingested is *data*, not code. Adding a source
should mean adding a :class:`DatasetSpec` entry to the registry -- writing a new
loader class is reserved for the handful of corpora whose layout genuinely
refuses to be described declaratively (Emilia's tar shards, WenetSpeech's
application gate).

Two-phase ingest
----------------
Phase A streams **metadata only**: text, source identifier, duration. No audio
bytes are fetched. That is enough to run normalization, the hard gates, dedup on
``audio_filepath``, accent tagging and LLM correction -- which is where nearly
all the compute lives.

Phase B materializes audio **only for samples that survived into a batch**.

The arithmetic is why this is not an optional optimization. A candidate pool of
~50M unique clips at ~100k hours is on the order of a petabyte at 32 kbps; a
single 100k x 5-language batch is ~1,111 hours, or about 16 GB. Materializing
the pool is impossible. Materializing the batches is trivial.

Verification
------------
Repo identifiers in the registry carry ``verified``. Several were recalled
rather than confirmed, and a wrong ``repo_id`` fails slowly -- after the download
has started, or after it has silently fetched the wrong corpus. Run
``preflight.py`` before any ingest: it resolves every entry against the Hub and
reports existence, gating, auth requirement and license, so a bad identifier is
a one-line report instead of a wasted day.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("data_processing.datasets.base")

#: Per-dataset ceiling from the corpus spec: at most 10M samples per source.
DEFAULT_MAX_SAMPLES = 10_000_000

#: Samples per emitted shard. 100k matches one batch-language quota exactly, so
#: a shard is the natural unit of work for Phase B and for resumption.
DEFAULT_SHARD_SIZE = 100_000


class Kind(str, Enum):
    """How a source is reached."""

    #: Hugging Face Hub, ungated. ``load_dataset(..., streaming=True)``.
    HF_STREAM = "hf_stream"
    #: Hugging Face Hub, but terms must be accepted and a token supplied.
    HF_GATED = "hf_gated"
    #: JSONL manifests already on disk -- the internal v7.6 corpora.
    LOCAL_JSONL = "local_jsonl"
    #: Audio already extracted locally; only the manifest is remote or absent.
    LOCAL_AUDIO = "local_audio"
    #: Needs a bespoke loader. ``DatasetSpec.loader`` names the callable.
    CUSTOM = "custom"

    @property
    def is_local(self) -> bool:
        """Whether the audio already sits at a stable path on shared storage.

        Local sources -- the internal v7.6 JSONL tree and already-extracted
        Emilia -- keep their ``audio_filepath`` verbatim. There is nothing to
        fetch, so materializing them would only duplicate a petabyte-scale tree.
        """
        return self in (Kind.LOCAL_JSONL, Kind.LOCAL_AUDIO)

    @property
    def needs_materialization(self) -> bool:
        """Whether Stage 3 must fetch, decode and rewrite the audio locally.

        Everything reached over the Hub (or via a bespoke loader) arrives as a
        remote reference, so its clips are downloaded to a deterministic path
        under ``audio_root`` at 16 kHz mono. CUSTOM defaults to external: a
        loader that already yields local paths should use LOCAL_AUDIO instead.
        """
        return not self.is_local


def _unwrap_audio(value: object) -> str | None:
    """Reduce an HF audio feature to its path. Never to its samples.

    ``datasets`` decodes an Audio column into ``{"path": ..., "array": ...}``.
    The array must not reach ``audio_filepath``, because that field is both the
    dedup key and the ledger's primary key. Stringifying a decoded feature
    produces an ~80 KB repr that differs for every clip, so uniqueness would
    report zero duplicates while storing terabytes of keys at 50M rows.

    Applied to the mapped column as well as to the ``audio`` fallback: for the
    ``_F_HF_AUDIO`` and ``_F_FLEURS`` field maps the mapped column *is*
    ``audio``, which is exactly the case the fallback-only version missed.
    """
    if isinstance(value, dict):
        inner = value.get("path") or value.get("audio_file_name")
        return str(inner) if inner else None
    if isinstance(value, str) and value:
        return value
    return None


def derive_materialized_path(
    audio_root: str | Path,
    lang: str,
    source: str,
    native_id: str,
    ext: str = ".flac",
) -> str:
    """Deterministic on-disk path for one external clip.

    ``<audio_root>/<lang>/<source>/<blake2b(native_id)[:16]>.flac``

    The digest, not the native id, names the file: Hub identifiers are URLs, tar
    keys or bare filenames that collide across configs and carry characters no
    filesystem wants. blake2b over the id is stable across runs and nodes, so
    Stage 1 (which fixes the identity), the ledger key, the batch manifest and
    the Stage 3 write all agree on one path without coordinating. Eight bytes is
    16 hex chars; at 50M clips the birthday-bound collision odds are ~1e-4, and
    the ledger's ``audio_filepath`` uniqueness turns any collision into a
    rejected duplicate rather than a silent overwrite.
    """
    digest = hashlib.blake2b(native_id.encode("utf-8"), digest_size=8).hexdigest()
    root = str(audio_root).rstrip("/")
    return f"{root}/{lang}/{source}/{digest}{ext}"


@dataclass(frozen=True, slots=True)
class FieldMap:
    """Maps a source's own column names onto the canonical four.

    Sources are wildly inconsistent here: Common Voice uses ``sentence`` and
    ``path``, AISHELL uses ``text`` and ``audio``, Emilia uses ``text`` and a
    ``__url__``/``__key__`` pair from its tar shards. Describing that as data
    keeps the loaders to one implementation.
    """

    #: Transcript column.
    text: str = "text"
    #: Duration in seconds. None means the source omits it and it must be
    #: probed from the audio header -- a cost worth knowing about up front,
    #: because probing forces a partial fetch and defeats metadata-only Phase A.
    duration: str | None = "duration"
    #: Audio identifier. May be a filename, a full path, or a tar key.
    path: str = "audio_filepath"
    #: Language column, for multi-language sources that carry one.
    lang: str | None = None

    @property
    def needs_duration_probe(self) -> bool:
        return self.duration is None


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """One ingestible corpus."""

    #: Unique registry key. Appears in output filenames and in the ledger.
    name: str
    #: Canonical language code: ar | en | zh | hi | ml.
    lang: str
    kind: Kind
    #: Human-readable origin, recorded in every emitted sidecar.
    license: str
    #: Approximate size, for planning. Never trusted for accounting -- the
    #: ledger counts what was actually read.
    est_hours: float = 0.0

    # --- Hub sources ---------------------------------------------------------
    repo_id: str | None = None
    config: str | None = None
    split: str = "train"
    revision: str | None = None

    # --- Local sources -------------------------------------------------------
    #: Glob or explicit paths, for LOCAL_JSONL / LOCAL_AUDIO.
    paths: tuple[str, ...] = ()
    #: Substrings that exclude a matched file. The internal v7.6 tree needs this:
    #: ``_still_rejected_`` shards are recovery inputs folded into their parent,
    #: and counting them is how ml came to be overstated by 41%.
    exclude: tuple[str, ...] = ()
    #: Root for a LOCAL_* spec in a SECOND on-disk tree. The q3asr SFT
    #: manifests sit beside -- not under -- the v7.6 manifests, and the stages
    #: pass one ``--root`` for the v7.6 tree; a spec with ``local_root`` set
    #: resolves its relative ``paths`` against it and ignores the caller's
    #: root (see ``local.effective_root``, the single choke point).
    local_root: str | None = None

    # --- Shape ---------------------------------------------------------------
    fields: FieldMap = field(default_factory=FieldMap)
    #: Prepended to the source's path column to form ``audio_filepath``. Must be
    #: absolute and stable, because ``audio_filepath`` is the dedup key.
    path_prefix: str = ""
    #: For tar-style sources where the row has no path at all.
    path_from: Callable[[dict], str] | None = None

    # --- Policy --------------------------------------------------------------
    max_samples: int = DEFAULT_MAX_SAMPLES
    #: Terms must be accepted on the Hub before download will succeed.
    gated: bool = False
    #: Whether the repo_id / config pair has been confirmed to resolve.
    #: Unverified entries are reported by preflight, not silently attempted.
    verified: bool = False
    #: Best-effort notes: known pitfalls, subset structure, why hours are rough.
    notes: str = ""
    #: Custom loader, when ``kind`` is CUSTOM. Takes the spec and yields raw
    #: dicts; everything downstream is shared.
    loader: Callable[[DatasetSpec], Iterator[dict]] | None = None

    @property
    def origin(self) -> str:
        """Stable provenance string written into every sidecar record."""
        return self.repo_id or self.name

    def audio_filepath(self, raw: dict) -> str | None:
        """Derive the canonical dedup key from one raw source row."""
        if self.path_from is not None:
            return self.path_from(raw)
        value = _unwrap_audio(raw.get(self.fields.path))
        if not value:
            # Take the path out of a decoded feature so Phase A never has to
            # touch the waveform, whichever column happened to carry it.
            value = _unwrap_audio(raw.get("audio"))
        if not value:
            return None
        if not self.path_prefix or value.startswith("/"):
            return value
        return f"{self.path_prefix.rstrip('/')}/{value.lstrip('/')}"

    def duration_of(self, raw: dict) -> float | None:
        """Duration in seconds, or None when the source does not carry it."""
        if self.fields.duration is None:
            return None
        value = raw.get(self.fields.duration)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def text_of(self, raw: dict) -> str | None:
        value = raw.get(self.fields.text)
        if value is None:
            # Common Voice and several AI4Bharat sets use `sentence`.
            value = raw.get("sentence")
        return None if value is None else str(value)

    @property
    def is_local(self) -> bool:
        """Audio already at a stable local path; ``audio_filepath`` is kept as-is."""
        return self.kind.is_local

    @property
    def needs_materialization(self) -> bool:
        """Audio must be fetched and rewritten under ``audio_root`` at 16 kHz mono."""
        return self.kind.needs_materialization

    def materialized_path(self, audio_root: str | Path, native_id: str, ext: str = ".flac") -> str:
        """The deterministic target path for one of this source's external clips.

        ``native_id`` is the source's own identifier for the clip -- what
        :meth:`audio_filepath` returns before any rewriting -- so the derivation
        is reproducible from the raw row alone and needs no audio bytes.
        """
        return derive_materialized_path(audio_root, self.lang, self.name, native_id, ext)


@dataclass(slots=True)
class IngestStats:
    """What one ingest run actually did. Written to the run report."""

    spec_name: str
    read: int = 0
    emitted: int = 0
    skipped_no_path: int = 0
    skipped_no_text: int = 0
    skipped_seen: int = 0
    cap_reached: bool = False
    #: Rows where duration was absent and had to be probed or defaulted.
    duration_missing: int = 0
    errors: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec_name,
            "read": self.read,
            "emitted": self.emitted,
            "skipped_no_path": self.skipped_no_path,
            "skipped_no_text": self.skipped_no_text,
            "skipped_seen": self.skipped_seen,
            "duration_missing": self.duration_missing,
            "cap_reached": self.cap_reached,
            "errors": dict(self.errors),
        }


def require_datasets() -> Any:
    """Import the HF ``datasets`` library, or fail with the fix.

    ``datasets`` is not currently a declared dependency of this project -- only
    ``huggingface-hub`` is. Failing here with the exact install command beats an
    ImportError surfacing from inside a loader three stack frames down, after a
    long download has already begun.
    """
    try:
        import datasets
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "the HF `datasets` library is required for streaming ingest but is "
            "not installed. Add it with:\n"
            "    pip install 'datasets>=3.0' 'huggingface-hub[hf_transfer]'\n"
            "and set HF_HUB_ENABLE_HF_TRANSFER=1 for parallel downloads."
        ) from exc
    return datasets


__all__ = [
    "DEFAULT_MAX_SAMPLES",
    "DEFAULT_SHARD_SIZE",
    "DatasetSpec",
    "FieldMap",
    "IngestStats",
    "Kind",
    "derive_materialized_path",
    "require_datasets",
]
