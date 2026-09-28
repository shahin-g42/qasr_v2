"""Phase A: metadata-only ingest.

One entry point per source kind, one shared row-to-``Sample`` mapping, and a
round-robin interleaver. No audio bytes are fetched anywhere in this module.

Why interleaving lives here
---------------------------
The distributor caps any single source at ``max_per_source_fraction`` of a
batch, but a cap only *rejects* -- it cannot make a batch diverse. If sources
are drained one at a time, the first source fills 40% of the batch and
everything after it is capped or starved, and the result is still lopsided.
Pulling one row from each source in turn means the cap almost never binds and
the batch is mixed by construction.

Duration is the awkward field
-----------------------------
``quality.gate`` rejects a sample with no duration, so a source that omits it
would be silently discarded rather than loudly reported. This module therefore
returns ``None`` for such rows and increments ``IngestStats.duration_missing``.
Probing a header means fetching audio, which defeats Phase A -- so the decision
to pay that cost belongs to the caller, made with the count in hand.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from collections.abc import Iterable, Iterator
from typing import Any

from ..canonical import Meta, Sample
from .base import DatasetSpec, IngestStats, Kind, require_datasets
from .local import iter_local, iter_local_audio

LOGGER = logging.getLogger("data_processing.datasets.stream")


def _hf_stream(spec: DatasetSpec, token: str | None) -> Iterator[dict]:
    """Stream a Hub dataset without downloading its audio.

    ``streaming=True`` is what keeps Phase A cheap: rows arrive lazily over HTTP
    and the audio feature is a path/URL reference rather than a decoded array,
    provided nothing downstream touches it.
    """
    datasets = require_datasets()
    kwargs: dict[str, Any] = {
        "path": spec.repo_id,
        "name": spec.config,
        "split": spec.split,
        "streaming": True,
        "trust_remote_code": False,
    }
    if spec.revision:
        kwargs["revision"] = spec.revision
    if token:
        kwargs["token"] = token
    if spec.kind is Kind.HF_GATED and not token:
        LOGGER.warning(
            "%s is gated but no HF_TOKEN is set; the request will 401. "
            "Accept the terms at https://huggingface.co/datasets/%s and export HF_TOKEN.",
            spec.name, spec.repo_id,
        )
    try:
        yield from datasets.load_dataset(**kwargs)
    except Exception as exc:
        LOGGER.error("%s: Hub stream failed (%s): %s", spec.name, type(exc).__name__, exc)
        raise


def stream_metadata(
    spec: DatasetSpec,
    root: str | None = None,
    stats: IngestStats | None = None,
    token: str | None = None,
) -> Iterator[dict]:
    """Yield raw source rows for any spec kind, capped at ``spec.max_samples``."""
    stats = stats if stats is not None else IngestStats(spec_name=spec.name)
    token = token or os.environ.get("HF_TOKEN")

    if spec.loader is not None:
        source: Iterator[dict] = spec.loader(spec)
    elif spec.kind in (Kind.HF_STREAM, Kind.HF_GATED):
        if not spec.repo_id:
            raise ValueError(f"{spec.name}: kind={spec.kind} requires repo_id")
        source = _hf_stream(spec, token)
    elif spec.kind is Kind.LOCAL_JSONL:
        source = iter_local(spec, root, stats)
    elif spec.kind is Kind.LOCAL_AUDIO:
        source = iter_local_audio(spec, root, stats)
    elif spec.kind is Kind.CUSTOM:
        raise ValueError(f"{spec.name}: kind=CUSTOM requires a loader callable")
    else:  # pragma: no cover - Enum is exhaustive
        raise ValueError(f"{spec.name}: unhandled kind {spec.kind}")

    if spec.loader is not None or spec.kind in (Kind.HF_STREAM, Kind.HF_GATED):
        # Local loaders already count and cap; keep the two paths consistent.
        for row in source:
            stats.read += 1
            yield row
            if stats.read >= spec.max_samples:
                stats.cap_reached = True
                return
    else:
        yield from source


def to_sample(spec: DatasetSpec, row: dict) -> tuple[Sample, Meta] | None:
    """Map one raw row onto ``(Sample, Meta)``, or None if it is unusable.

    Returns None -- never raises -- for the three benign cases: no path, no
    text, no duration. Each is counted by the caller so a source that is
    systematically missing a field shows up in the run report instead of
    quietly contributing nothing.
    """
    fp = spec.audio_filepath(row)
    if not fp:
        return None
    text = spec.text_of(row)
    if not text or not text.strip():
        return None
    duration = spec.duration_of(row)
    if duration is None:
        return None

    lang = spec.lang
    if spec.fields.lang:
        lang = str(row.get(spec.fields.lang) or lang)

    sample = Sample(audio_filepath=fp, duration=float(duration), text=text.strip(), lang=lang)
    meta = Meta(
        audio_filepath=fp,
        dataset=spec.origin,
        source_text=text,
    )
    meta.metrics["duration"] = round(float(duration), 3)
    return sample, meta


def interleave(
    streams: Iterable[tuple[str, Iterator[Any]]],
    stop_on_exhausted: bool = False,
) -> Iterator[tuple[str, Any]]:
    """Round-robin over named streams, yielding ``(source_name, item)``.

    Exhausted streams drop out and the rest continue, so one small corpus cannot
    truncate a build. ``stop_on_exhausted=True`` instead halts as soon as any
    stream ends, which is what a balanced-composition build wants.
    """
    pending = deque((name, it) for name, it in streams)
    while pending:
        name, it = pending.popleft()
        try:
            item = next(it)
        except StopIteration:
            LOGGER.info("source %s exhausted after round-robin", name)
            if stop_on_exhausted:
                return
            continue                      # dropped, not requeued
        yield name, item
        pending.append((name, it))        # requeued at the back == round robin


def ingest(
    specs: Iterable[DatasetSpec],
    root: str | None = None,
    token: str | None = None,
    interleave_sources: bool = True,
) -> Iterator[tuple[DatasetSpec, dict, IngestStats]]:
    """Drive several specs as one stream of ``(spec, raw_row, stats)``.

    This is the seam the rest of the pipeline hangs off: normalize -> gate ->
    accent-tag -> LLM-correct -> distribute all consume this and never need to
    know whether a row came from Lustre or the Hub.
    """
    specs = list(specs)
    stats_by_name = {s.name: IngestStats(spec_name=s.name) for s in specs}
    by_name = {s.name: s for s in specs}

    if not interleave_sources or len(specs) == 1:
        for spec in specs:
            st = stats_by_name[spec.name]
            for row in stream_metadata(spec, root, st, token):
                yield spec, row, st
        return

    streams = [
        (spec.name, stream_metadata(spec, root, stats_by_name[spec.name], token))
        for spec in specs
    ]
    for name, row in interleave(streams):
        yield by_name[name], row, stats_by_name[name]


def stats_report(all_stats: Iterable[IngestStats]) -> dict[str, Any]:
    """Aggregate per-source stats into one run-level report."""
    rows = [s.as_dict() for s in all_stats]
    return {
        "sources": rows,
        "totals": {
            "read": sum(r["read"] for r in rows),
            "emitted": sum(r["emitted"] for r in rows),
            "skipped_no_path": sum(r["skipped_no_path"] for r in rows),
            "skipped_no_text": sum(r["skipped_no_text"] for r in rows),
            "skipped_seen": sum(r["skipped_seen"] for r in rows),
            "duration_missing": sum(r["duration_missing"] for r in rows),
            "cap_reached": sum(1 for r in rows if r["cap_reached"]),
        },
    }


__all__ = ["ingest", "interleave", "stats_report", "stream_metadata", "to_sample"]
