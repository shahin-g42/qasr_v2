"""Loader for corpora already on disk: internal manifests and extracted audio.

Two of the five languages need no network at all. The internal v7.6 tree is
already a set of JSONL manifests, and Emilia-ZH is already extracted under
``/vast``. Both are handled here, which also makes them the only sources that
can be tested end-to-end without a Hub token or a multi-hour download.

Exclusion is the important part, not the reading. The v7.6 tree interleaves
three kinds of file that look alike:

``train_*.jsonl``
    Real training data.
``*_still_rejected_p*.jsonl``
    Recovery inputs already folded into their parent corpus. Counting them is
    how ml was overstated by 41%.
``eval_*.jsonl``
    Held-out. 962 ar eval paths were measured as also present in train, so
    these must be excluded from ingest *and* loaded into the ledger as
    exclusions before any batch is built.

A glob of ``*.jsonl`` matches all three. Every spec therefore carries an
``exclude`` tuple, and this loader applies it before reading a single line.
"""

from __future__ import annotations

import gzip
import json
import logging
from collections.abc import Callable, Iterator
from pathlib import Path

from .base import DatasetSpec, IngestStats, Kind

LOGGER = logging.getLogger("data_processing.datasets.local")

#: What a LOCAL_JSONL manifest looks like. Covers ``.jsonl`` and ``.jsonl.gz``.
JSONL_TOKEN = ".jsonl"

#: What a LOCAL_AUDIO clip looks like. Emilia-style trees carry one audio file
#: per clip with a sibling JSON sidecar and no manifest at all, so there is
#: nothing with a ``.jsonl`` in its name to find.
AUDIO_SUFFIXES = frozenset({".wav", ".flac", ".mp3", ".opus", ".m4a", ".ogg", ".webm"})


def _wanted(spec: DatasetSpec) -> Callable[[Path], bool]:
    """Which files in a directory belong to this spec.

    One predicate rather than a glob per extension, because ``rglob`` walks the
    tree once per pattern and the trees here hold millions of files -- five
    audio extensions would mean five full walks of Emilia.
    """
    if spec.kind is Kind.LOCAL_AUDIO:
        return lambda x: x.suffix.lower() in AUDIO_SUFFIXES
    return lambda x: JSONL_TOKEN in x.name.lower()


def _open_text(path: Path):
    """Read ``.jsonl`` and ``.jsonl.gz`` identically."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def is_excluded(path: Path, exclude: tuple[str, ...]) -> bool:
    """True when any exclusion substring appears in the filename."""
    name = path.name
    return any(token in name for token in exclude)


def effective_root(spec: DatasetSpec, root: str | Path | None) -> str | Path | None:
    """The root a spec's relative ``paths`` resolve against.

    Specs in a second on-disk tree (the q3asr SFT manifests, a sibling of the
    v7.6 manifests rather than a subdirectory of them) carry their own
    ``local_root``; every other spec resolves against the caller's ``root``.
    Resolving here -- the one function every local loader call goes through --
    is what keeps prepare, preflight and the stream engine from disagreeing
    about where a tree lives.
    """
    return spec.local_root if spec.local_root else root


def expand_paths(spec: DatasetSpec, root: str | Path | None = None) -> list[Path]:
    """Resolve a spec's globs into a sorted, de-duplicated file list.

    ``root`` is prepended to relative patterns so a registry can stay portable
    across machines with different mount points. A pattern must therefore be
    relative to ``root`` and must not also contain it: joining the two turns
    ``training_manifests/v7.6`` + ``ar/*.jsonl`` into the right path, while
    joining it with an already-rooted pattern silently doubles the prefix and
    matches nothing at all. A spec with ``local_root`` set (the SFT tree)
    ignores the caller's ``root`` entirely.

    Directories are filtered by kind, so a LOCAL_AUDIO spec enumerates audio and
    a LOCAL_JSONL spec enumerates manifests.
    """
    root = effective_root(spec, root)
    out: set[Path] = set()
    wanted = _wanted(spec)
    for pattern in spec.paths:
        p = Path(pattern)
        if not p.is_absolute() and root is not None:
            p = Path(root) / p
        if p.is_dir():
            matched = sorted(x for x in p.rglob("*") if x.is_file() and wanted(x))
        elif any(ch in pattern for ch in "*?["):
            matched = sorted(x for x in p.parent.glob(p.name) if x.is_file())
        elif p.exists():
            matched = [p]
        else:
            LOGGER.warning("%s: pattern matched nothing: %s", spec.name, pattern)
            continue
        for m in matched:
            if not is_excluded(m, spec.exclude):
                out.add(m)
    return sorted(out)


def iter_local(spec: DatasetSpec, root: str | Path | None = None, stats: IngestStats | None = None) -> Iterator[dict]:
    """Yield raw source rows from a LOCAL_JSONL spec.

    Rows are yielded as parsed, with no field mapping applied -- that is the
    stream engine's job, so every loader returns the same shape of thing.
    Stops at ``spec.max_samples`` and records that it did.
    """
    stats = stats if stats is not None else IngestStats(spec_name=spec.name)
    if spec.kind not in (Kind.LOCAL_JSONL, Kind.LOCAL_AUDIO):
        raise ValueError(f"{spec.name}: local loader cannot handle kind={spec.kind}")

    files = expand_paths(spec, root)
    if not files:
        LOGGER.error("%s: no files matched after exclusions", spec.name)
        return
    LOGGER.info("%s: %s file(s) after exclusions", spec.name, f"{len(files):,}")

    for path in files:
        try:
            with _open_text(path) as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        stats.errors[f"bad_json:{path.name}"] = stats.errors.get(f"bad_json:{path.name}", 0) + 1
                        continue
                    if not isinstance(row, dict):
                        stats.errors["not_an_object"] = stats.errors.get("not_an_object", 0) + 1
                        continue
                    # Records that name their origin file make audits possible
                    # without re-running the loader.
                    row.setdefault("__source_file__", str(path))
                    stats.read += 1
                    yield row
                    if stats.read >= spec.max_samples:
                        stats.cap_reached = True
                        LOGGER.info("%s: hit max_samples=%s", spec.name, f"{spec.max_samples:,}")
                        return
        except OSError as exc:
            stats.errors[f"io:{path.name}"] = stats.errors.get(f"io:{path.name}", 0) + 1
            LOGGER.warning("%s: cannot read %s: %s", spec.name, path, exc)


def iter_local_audio(spec: DatasetSpec, root: str | Path | None = None, stats: IngestStats | None = None) -> Iterator[dict]:
    """Walk an already-extracted audio tree, for sources with no manifest.

    Emilia-ZH is the motivating case: ~49,900 hours of tar-extracted wav with a
    per-file JSON sidecar rather than one big manifest. Audio is *listed*, never
    opened -- Phase A stays metadata-only, and duration comes from the sidecar
    or is reported missing.
    """
    stats = stats if stats is not None else IngestStats(spec_name=spec.name)
    for path in expand_paths(spec, root):
        # clip0.wav -> clip0.json. The previous form built clip0.wav.json by
        # string-replacing ".jsonl" in a suffix that never contained it.
        sidecar = path.with_suffix(".json")
        row: dict = {"audio_filepath": str(path), "__source_file__": str(path)}
        if sidecar.exists():
            try:
                row.update(json.loads(sidecar.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError) as exc:
                stats.errors["bad_sidecar"] = stats.errors.get("bad_sidecar", 0) + 1
                LOGGER.debug("%s: sidecar unreadable %s: %s", spec.name, sidecar, exc)
        stats.read += 1
        yield row
        if stats.read >= spec.max_samples:
            stats.cap_reached = True
            return


def duration_coverage(spec: DatasetSpec, root: str | Path | None = None, sample: int = 20_000) -> dict:
    """Measure how often a source actually carries a usable duration.

    Worth running before ingest rather than after. A source that omits duration
    forces a header probe, and probing means fetching audio -- which silently
    defeats the metadata-only Phase A that the whole two-phase design rests on.
    The internal q3asr shards were measured at 58.6% missing, which is the
    difference between a cheap pass and a petabyte-scale one.
    """
    present = absent = 0
    for row in iter_local(spec, root):
        if spec.duration_of(row) is None:
            absent += 1
        else:
            present += 1
        if present + absent >= sample:
            break
    total = present + absent
    return {
        "spec": spec.name,
        "sampled": total,
        "duration_present": present,
        "duration_missing": absent,
        "missing_fraction": round(absent / total, 4) if total else 0.0,
        "needs_probe": spec.fields.needs_duration_probe,
    }


__all__ = [
    "AUDIO_SUFFIXES",
    "JSONL_TOKEN",
    "duration_coverage",
    "effective_root",
    "expand_paths",
    "is_excluded",
    "iter_local",
    "iter_local_audio",
]
