"""Audio duration probing for manifest records that carry no duration.

Durations are read from the audio header (``soundfile.info`` — no decode),
so this stays cheap at manifest scale, and the reads run in a shared thread
pool so the event loop keeps issuing LLM requests while the filesystem is
being hit.

Records that already have a duration cost nothing: they are filtered out
before any I/O happens. Without this pass a source manifest missing the
``duration`` key produces cleaned records with no duration, which training
needs — and recovering them later means a separate full pass over every
audio file (scripts/backfill_durations.py).
"""

from __future__ import annotations

import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor

from .manifest_io import ManifestRecord

try:
    import soundfile as sf
except ImportError:  # probing is optional — the pipeline runs without it
    sf = None  # type: ignore[assignment]

LOGGER = logging.getLogger("data_processing.duration")

# Same rewrite the training loader applies (src/qasr/data.py)
_VAST = re.compile(r"^/vast")

# One probe pool per process: every worker is an asyncio task on the same
# event loop, so a pool per worker would multiply into thousands of threads.
_EXECUTOR: ThreadPoolExecutor | None = None
_MISSING_SOUNDFILE_WARNED = False


def probe_duration(audio_filepath: str) -> float | None:
    """Duration in seconds from the audio header, or None if unreadable."""
    if sf is None:
        return None
    resolved = _VAST.sub("/lustrefs/taiga/vast40", audio_filepath)
    try:
        info = sf.info(resolved)
    except Exception:  # missing file, unsupported codec, truncated header
        return None
    if info.frames > 0 and info.samplerate > 0:
        return round(info.frames / info.samplerate, 3)
    return None


def _get_executor(max_workers: int) -> ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ThreadPoolExecutor(
            max_workers=max(1, max_workers), thread_name_prefix="duration-probe"
        )
    return _EXECUTOR


async def fill_missing_durations(
    records: list[ManifestRecord], max_workers: int = 32
) -> tuple[int, int]:
    """Set ``duration`` on records that have none.

    Returns ``(filled, failed)`` — how many durations were read and how many
    probes found no usable header — so callers can surface the shortfall.

    Mutates the records in place, so it must run BEFORE the cleaner copies
    them into CleanedRecords — that way accepted and rejected records alike
    carry the duration.
    """
    targets = [r for r in records if r.duration is None and r.audio_filepath]
    if not targets:
        return 0, 0

    if sf is None:
        global _MISSING_SOUNDFILE_WARNED
        if not _MISSING_SOUNDFILE_WARNED:
            LOGGER.warning(
                "soundfile is not installed — cannot probe missing durations; "
                "run scripts/backfill_durations.py on the cleaned output instead"
            )
            _MISSING_SOUNDFILE_WARNED = True
        return 0, len(targets)

    loop = asyncio.get_running_loop()
    executor = _get_executor(max_workers)
    durations = await asyncio.gather(
        *(
            loop.run_in_executor(executor, probe_duration, record.audio_filepath)
            for record in targets
        )
    )

    filled = 0
    for record, duration in zip(targets, durations, strict=True):
        if duration is not None:
            record.duration = duration
            filled += 1

    failed = len(targets) - filled
    if failed:
        # WARNING, not DEBUG: probe_duration swallows every per-file
        # exception, so this summary is the only sign records are shipping
        # without durations — 92% of zh and 100% of hi once did, unseen.
        LOGGER.warning(
            "Probed %d/%d missing durations (%d audio headers unreadable)",
            filled,
            len(targets),
            failed,
        )
    return filled, failed


__all__ = ["fill_missing_durations", "probe_duration"]
