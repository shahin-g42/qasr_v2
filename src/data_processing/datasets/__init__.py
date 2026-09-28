"""External and internal corpus ingest for the sequential batch builds.

Layout
------
``base``
    :class:`DatasetSpec` / :class:`FieldMap` / :class:`Kind` -- the declarative
    contract. A source is data, not a loader class.
``registry``
    Every known source. Repo identifiers here were resolved against the Hub API
    before being recorded; ``verified`` says so.
``preflight``
    Re-resolves the registry and probes column names. Run it first: 11 of 27
    candidate identifiers did not survive contact with the API.
``local``
    On-disk sources: the internal v7.6 manifests and already-extracted audio.
``stream``
    Phase A. Metadata-only streaming, field mapping, and the round-robin
    interleaver that makes a batch genuinely mixed across sources.

Two phases, and why
-------------------
Phase A moves text and durations only. Phase B materializes audio for the
samples that survived into a batch. A ~50M-clip candidate pool is on the order
of a petabyte; one 100k x 5-language batch is ~16 GB. The design is not an
optimization, it is what makes the build possible at all.
"""

from __future__ import annotations

from .base import (
    DEFAULT_MAX_SAMPLES,
    DEFAULT_SHARD_SIZE,
    DatasetSpec,
    FieldMap,
    IngestStats,
    Kind,
    require_datasets,
)
from .local import duration_coverage, effective_root, expand_paths, iter_local, iter_local_audio
from .preflight import CheckResult, check_hub, check_local, probe_fields
from .registry import (
    EMILIA_ROOT,
    INTERNAL_ROOT,
    LANGUAGES,
    SFT_ROOT,
    all_specs,
    by_name,
    specs_for,
    summary,
)
from .stream import ingest, interleave, stats_report, stream_metadata, to_sample

__all__ = [
    "DEFAULT_MAX_SAMPLES",
    "DEFAULT_SHARD_SIZE",
    "EMILIA_ROOT",
    "INTERNAL_ROOT",
    "LANGUAGES",
    "SFT_ROOT",
    "CheckResult",
    "DatasetSpec",
    "FieldMap",
    "IngestStats",
    "Kind",
    "all_specs",
    "by_name",
    "check_hub",
    "check_local",
    "duration_coverage",
    "effective_root",
    "expand_paths",
    "ingest",
    "interleave",
    "iter_local",
    "iter_local_audio",
    "probe_fields",
    "require_datasets",
    "specs_for",
    "stats_report",
    "stream_metadata",
    "summary",
    "to_sample",
]
