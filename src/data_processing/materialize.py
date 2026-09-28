"""Stage 3: materialize external audio to 16 kHz mono FLAC at a fixed path.

Internal sources (the v7.6 JSONL tree, already-extracted Emilia) keep their
``audio_filepath`` verbatim -- the bytes are already on shared storage and
copying them would only duplicate a petabyte-scale tree. External sources arrive
as remote references, so their clips are downloaded, decoded, resampled to a
uniform 16 kHz mono and written to the deterministic path Stage 1 already fixed
(:func:`~data_processing.datasets.base.derive_materialized_path`). One sample
rate across the whole corpus is what lets the training loader skip per-file
resampling and the duration in the manifest equal the duration on disk.

Where this runs
---------------
The primitives here (decode -> resample -> atomic FLAC write -> verify) are used
in two places:

* **Stage 2, inline.** A batch manifest must carry a true ``duration``, and for
  an external clip that number only exists once the audio is decoded. So
  ``assemble`` materializes each *selected* clip as it places it -- zero
  over-fetch, because only samples that survived ranking and the caps are ever
  downloaded -- and reads the duration back off the file it just wrote.
* **Stage 3, verify/backfill (the CLI here).** After the batches exist, walk
  every external path they reference, confirm it is present at 16 kHz mono with
  the expected duration, and re-materialize the ones a crashed or refused stage
  left missing. Idempotent: a correct file is skipped, never rewritten.
* **Stage 3 prewarm (the CLI's ``--prewarm``).** While the five language nodes
  run assemble, the four idle nodes can walk a language's pool *ahead* of
  selection and materialize its top external candidates (composite order, so
  it is exactly the head assemble will ask for). Placement re-verifies and
  learns the duration from the same file, so prewarming changes no decision --
  it only moves the fetch latency off the critical path. Same idempotency: a
  clip prewarmed then selected is a header check, not a re-download.

Everything is written through a sibling temp file + ``os.replace``, so a crash
mid-write leaves no truncated FLAC that a later run mistakes for complete.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .canonical import read_manifest
from .datasets.base import DatasetSpec

LOGGER = logging.getLogger("data_processing.materialize")

#: Uniform target sample rate for every materialized clip.
DEFAULT_SAMPLE_RATE = 16_000

#: FLAC subtype. 16-bit PCM is the ASR standard and halves the size of a 24-bit
#: write for no measurable loss at 16 kHz speech.
FLAC_SUBTYPE = "PCM_16"

#: Same ``/vast`` -> cluster mount rewrite the training loader and the duration
#: prober apply, so a native id recorded on one machine resolves on another.
_VAST = re.compile(r"^/vast")
_VAST_REPLACEMENT = "/lustrefs/taiga/vast40"

#: A fetcher turns ``(spec, native_id)`` into a raw ``(array, sample_rate)`` pair.
#: Injection is what makes this module testable offline: the default reaches the
#: Hub, but a test hands in a callable that reads a fixture.
Fetcher = Callable[[DatasetSpec, str], "tuple[np.ndarray, int]"]


class MaterializeError(RuntimeError):
    """A clip could not be fetched, decoded, resampled or written."""


@dataclass(slots=True)
class MaterializeResult:
    """What one materialization produced."""

    target_path: str
    duration: float
    sample_rate: int
    #: True when the file already existed at the right rate/duration and was left
    #: untouched -- the idempotency signal a resumed run depends on.
    skipped: bool = False
    source: str = ""


@dataclass(slots=True)
class MaterializeReport:
    """Aggregate outcome of a materialize/verify pass."""

    verified: int = 0
    written: int = 0
    skipped: int = 0
    errors: Counter = field(default_factory=Counter)
    examples: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "verified": self.verified,
            "written": self.written,
            "skipped": self.skipped,
            "errors": dict(self.errors),
            "examples": self.examples[:20],
        }


# --- signal primitives ------------------------------------------------------
def _to_mono(array: np.ndarray) -> np.ndarray:
    """Collapse ``(n, ch)`` to ``(n,)`` float32; leave 1-D untouched."""
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1, dtype=np.float32)
    return np.ascontiguousarray(arr, dtype=np.float32)


def _resample(array: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Polyphase resample to ``target_sr``, mirroring ``qasr.audio``."""
    sr = int(sr)
    if sr <= 0:
        raise MaterializeError(f"invalid source sample rate {sr}")
    if sr == target_sr:
        return np.ascontiguousarray(array, dtype=np.float32)
    divisor = math.gcd(sr, int(target_sr))
    out = resample_poly(array, up=target_sr // divisor, down=sr // divisor)
    return np.ascontiguousarray(out, dtype=np.float32)


def write_flac(array: np.ndarray, target_path: str | Path, sample_rate: int) -> float:
    """Atomically write float32 mono ``array`` as 16-bit FLAC. Returns duration."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if array.shape[0] == 0:
        raise MaterializeError(f"empty audio for {target_path}")
    if not np.isfinite(array).all():
        raise MaterializeError(f"audio contains NaN/inf for {target_path}")
    fd, tmp_name = tempfile.mkstemp(dir=str(target_path.parent), prefix=f".{target_path.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        sf.write(str(tmp), array, sample_rate, format="FLAC", subtype=FLAC_SUBTYPE)
        os.replace(tmp, target_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return round(array.shape[0] / sample_rate, 3)


def materialize_array(
    array: np.ndarray, sr: int, target_path: str | Path, target_sr: int = DEFAULT_SAMPLE_RATE
) -> MaterializeResult:
    """Decode-ready array -> mono -> ``target_sr`` -> atomic FLAC at ``target_path``."""
    mono = _to_mono(array)
    rs = _resample(mono, sr, target_sr)
    dur = write_flac(rs, target_path, target_sr)
    return MaterializeResult(str(target_path), dur, target_sr, skipped=False)


def materialize_file(
    src_path: str | Path, target_path: str | Path, target_sr: int = DEFAULT_SAMPLE_RATE
) -> MaterializeResult:
    """Resample an on-disk audio file to ``target_sr`` mono FLAC.

    Decodes via :func:`local_file_fetcher` (float32, ``always_2d``) and resamples
    via :func:`_resample` -- the same mean-over-channels + polyphase
    ``resample_poly`` algorithm ``qasr.audio.load_mono_audio`` uses, kept local so
    this GPU-free stage never imports the heavy model package (and its
    transformers/torch dependencies) just to move samples.
    """
    array, sr = local_file_fetcher(None, str(src_path))
    return materialize_array(array, sr, target_path, target_sr)


# --- verification -----------------------------------------------------------
def probe(path: str | Path) -> tuple[float, int, int] | None:
    """``(duration, sample_rate, channels)`` from the header, or None if unreadable."""
    try:
        info = sf.info(str(path))
    except Exception:
        return None
    if info.frames <= 0 or info.samplerate <= 0:
        return None
    return round(info.frames / info.samplerate, 3), int(info.samplerate), int(info.channels)


def verify(
    path: str | Path,
    target_sr: int = DEFAULT_SAMPLE_RATE,
    expected_duration: float | None = None,
    tol: float = 0.05,
) -> bool:
    """True when ``path`` is present, mono, at ``target_sr`` and the right length.

    Duration is checked only when the caller knows it: a resample can shift the
    length by a sample or two, so the tolerance absorbs rounding rather than
    flagging a good file as corrupt.
    """
    p = probe(path)
    if p is None:
        return False
    dur, sr, ch = p
    if sr != target_sr or ch != 1:
        return False
    return expected_duration is None or abs(dur - expected_duration) <= tol


# --- fetching ---------------------------------------------------------------
def _resolve_local(native_id: str) -> str | None:
    """A readable local path for ``native_id``, or None if it is a remote ref."""
    if not native_id:
        return None
    for candidate in (native_id, _VAST.sub(_VAST_REPLACEMENT, native_id)):
        if candidate.startswith("/") and Path(candidate).exists():
            return candidate
    return None


def local_file_fetcher(spec: DatasetSpec | None, native_id: str) -> tuple[np.ndarray, int]:
    """Read ``native_id`` as an on-disk audio file. Also the test seam."""
    path = _resolve_local(native_id)
    if path is None:
        raise MaterializeError(f"native_id is not a readable local file: {native_id!r}")
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception as exc:
        raise MaterializeError(f"could not decode {path}: {exc}") from exc
    return data, int(sr)


def hf_hub_file_fetcher(spec: DatasetSpec, native_id: str) -> tuple[np.ndarray, int]:
    """Download one addressable file from the Hub, then decode it.

    Works for datasets whose audio is stored as individually addressable files.
    Parquet-backed datasets have no per-clip file to download; for those the
    bulk :func:`materialize_source_stream` pass (one streaming decode per source)
    is the route, and this raises rather than silently fetching the wrong thing.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise MaterializeError(
            "huggingface_hub is required to fetch external audio; "
            "pip install 'huggingface-hub[hf_transfer]'"
        ) from exc
    if spec is None or not spec.repo_id:
        raise MaterializeError(f"cannot fetch {native_id!r} without a repo_id")
    try:
        local = hf_hub_download(
            repo_id=spec.repo_id, filename=native_id, repo_type="dataset",
            revision=spec.revision, token=os.environ.get("HF_TOKEN"),
        )
    except Exception as exc:
        raise MaterializeError(f"hf_hub_download failed for {spec.repo_id}/{native_id}: {exc}") from exc
    return local_file_fetcher(spec, local)


class Materializer:
    """Fetches, resamples and writes external clips to their fixed target path."""

    def __init__(
        self,
        audio_root: str | Path,
        target_sr: int = DEFAULT_SAMPLE_RATE,
        fetcher: Fetcher | None = None,
        verify_existing: bool = True,
    ) -> None:
        self.audio_root = str(audio_root)
        self.target_sr = target_sr
        self.fetcher = fetcher
        self.verify_existing = verify_existing

    def is_external(self, audio_filepath: str) -> bool:
        """Whether a path lives under ``audio_root`` and so is ours to materialize."""
        root = self.audio_root.rstrip("/") + "/"
        return audio_filepath.startswith(root)

    def _fetch(self, spec: DatasetSpec | None, native_id: str) -> tuple[np.ndarray, int]:
        # A native id that already resolves to a local file (a warm cache, a
        # pre-downloaded tree) is read directly -- no Hub round-trip.
        if _resolve_local(native_id) is not None:
            return local_file_fetcher(spec, native_id)
        fetcher = self.fetcher or hf_hub_file_fetcher
        return fetcher(spec, native_id)

    def materialize(
        self,
        target_path: str,
        native_id: str,
        spec: DatasetSpec | None = None,
        expected_duration: float | None = None,
        source: str = "",
    ) -> MaterializeResult:
        """Ensure ``target_path`` holds ``native_id``'s audio at 16 kHz mono.

        Idempotent: an already-correct file is verified and skipped, so a resumed
        run neither re-downloads nor rewrites it.
        """
        if self.verify_existing and verify(target_path, self.target_sr, expected_duration):
            dur = probe(target_path)[0]
            return MaterializeResult(target_path, dur, self.target_sr, skipped=True, source=source)
        try:
            array, sr = self._fetch(spec, native_id)
        except MaterializeError:
            raise
        except Exception as exc:
            raise MaterializeError(f"fetch failed for {native_id!r}: {exc}") from exc
        result = materialize_array(array, sr, target_path, self.target_sr)
        result.source = source
        return result


def materialize_many(
    items: Iterable[tuple], materializer: Materializer, max_workers: int = 8
) -> tuple[list[MaterializeResult], list[tuple[str, str, str]]]:
    """Materialize many clips concurrently.

    ``items`` are ``(target_path, native_id, spec, expected_duration, source)``
    tuples. Returns ``(results, errors)`` where each error is
    ``(target_path, exception_type, message)`` -- one unreadable clip is reported,
    never allowed to abort a multi-hour pass.
    """
    results: list[MaterializeResult] = []
    errors: list[tuple[str, str, str]] = []
    items = list(items)
    if not items:
        return results, errors
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = {pool.submit(materializer.materialize, *it): it[0] for it in items}
        for fut, target in futures.items():
            try:
                results.append(fut.result())
            except Exception as exc:
                errors.append((target, type(exc).__name__, str(exc)))
    return results, errors


def materialize_source_stream(
    rows: Iterable[dict],
    decode: Callable[[dict], tuple[np.ndarray, int]],
    selected: dict[str, str],
    target_sr: int = DEFAULT_SAMPLE_RATE,
) -> Iterator[MaterializeResult]:
    """Materialize only the selected clips from one already-streamed source.

    The efficient bulk path for Hub datasets with no per-clip addressable file:
    stream the source once (the caller supplies ``rows``) and decode audio only
    for rows whose target path is in ``selected`` (target_path -> native_id).
    ``decode`` extracts ``(array, sr)`` from a raw row, so this stays testable
    without a network -- a test passes a fixture row list and a trivial decoder.
    """
    remaining = dict(selected)
    for row in rows:
        if not remaining:
            return
        target = next((t for t in remaining if _row_matches(row, t, remaining[t])), None)
        if target is None:
            continue
        array, sr = decode(row)
        yield materialize_array(array, sr, target, target_sr)
        remaining.pop(target, None)


def _row_matches(row: dict, target: str, native_id: str) -> bool:
    """Whether a raw streamed row is the clip that maps to ``target``."""
    for key in ("audio_filepath", "path", "audio", "file"):
        val = row.get(key)
        if isinstance(val, dict):
            val = val.get("path")
        if isinstance(val, str) and val and (val == native_id or val.endswith(native_id)):
            return True
    return False


def external_paths_in_batches(out_dir: str | Path, lang: str, audio_root: str) -> list[str]:
    """Every materialized (under ``audio_root``) path a language's batches reference."""
    root = audio_root.rstrip("/") + "/"
    out: list[str] = []
    lang_dir = Path(out_dir) / lang
    if not lang_dir.is_dir():
        return out
    for manifest, _sidecar in _iter_batch_shards(lang_dir, lang):
        for sample in read_manifest(manifest):
            if sample.audio_filepath.startswith(root):
                out.append(sample.audio_filepath)
    return out


def _iter_batch_shards(lang_dir: Path, lang: str) -> Iterator[tuple[Path, Path]]:
    from .canonical import iter_shards

    yield from iter_shards(lang_dir, lang=lang)


def ensure_batch_audio(
    out_dir: str | Path,
    lang: str,
    pool_dir: str | Path,
    audio_root: str | Path,
    target_sr: int = DEFAULT_SAMPLE_RATE,
    fetcher: Fetcher | None = None,
    max_workers: int = 8,
) -> MaterializeReport:
    """Stage-3 verify/backfill for one language's written batches.

    Confirms every external clip the batches reference exists at ``target_sr``
    mono; re-materializes the missing ones from the pool's ``native_id`` (the pool
    is the source of truth for the remote ref, so ``Meta`` needs no schema change).
    """
    from .candidate import iter_pool_shards, read_pool

    report = MaterializeReport()
    audio_root = str(audio_root)
    paths = external_paths_in_batches(out_dir, lang, audio_root)
    materializer = Materializer(audio_root, target_sr, fetcher=fetcher)

    missing = [p for p in paths if not verify(p, target_sr)]
    report.verified = len(paths) - len(missing)
    if not missing:
        return report

    # Map the missing targets back to their native ids from the pool.
    wanted = set(missing)
    native: dict[str, tuple[str, str]] = {}
    for shard in iter_pool_shards(pool_dir, lang):
        for rec in read_pool(shard):
            if rec.audio_filepath in wanted:
                native[rec.audio_filepath] = (rec.native_id, rec.source)

    items = []
    for path in missing:
        nid, src = native.get(path, ("", ""))
        if not nid:
            report.errors["no_native_id"] += 1
            if len(report.examples) < 20:
                report.examples.append(path)
            continue
        items.append((path, nid, None, None, src))

    results, errors = materialize_many(items, materializer, max_workers=max_workers)
    for res in results:
        report.written += 0 if res.skipped else 1
        report.skipped += 1 if res.skipped else 0
    for target, exc_type, msg in errors:
        report.errors[exc_type] += 1
        if len(report.examples) < 20:
            report.examples.append(f"{target}: {exc_type}: {msg}")
    return report


def prewarm_pool(
    lang: str,
    pool_dir: str | Path,
    audio_root: str | Path,
    budget: int = 130_000,
    *,
    target_sr: int = DEFAULT_SAMPLE_RATE,
    fetcher: Fetcher | None = None,
    max_workers: int = 32,
    include_gated: bool = True,
) -> MaterializeReport:
    """Materialize a language's top external pool candidates before selection.

    Walks the pool in composite order (:func:`candidate.merge_pools`) and
    materializes the first ``budget`` external candidates -- the same head
    Stage-2 assemble will ask for, fetched ahead of time from an idle node so
    the selecting node's placement becomes a header check. ``budget <= 0``
    drains every external candidate in the pool.

    Deliberately does *not* apply the per-source/text caps or the diversity
    floor: those are admission decisions that belong to assemble. A prewarmed
    clip assemble never selects simply stays on disk, idempotent and wanted by
    a later batch -- the cost of guessing ahead is bounded by ``budget``.
    """
    from .candidate import iter_pool_shards, merge_pools
    from .datasets import registry

    specs_by_name = {s.name: s for s in registry.specs_for(lang, include_gated=include_gated)}
    materializer = Materializer(audio_root, target_sr, fetcher=fetcher)
    report = MaterializeReport()

    items: list[tuple] = []
    shards = iter_pool_shards(pool_dir, lang)
    if not shards:
        LOGGER.error("%s: no pool shards under %s -- run prepare first", lang, pool_dir)
        return report
    for rec in merge_pools(shards):
        if budget > 0 and len(items) >= budget:
            break
        if not rec.external or not rec.native_id:
            continue
        items.append((rec.audio_filepath, rec.native_id,
                      specs_by_name.get(rec.source), None, rec.source))
    if not items:
        LOGGER.warning("%s: no external candidates to prewarm under %s", lang, pool_dir)
        return report
    LOGGER.info("prewarm %s: materializing %d top external candidate(s) "
                "(budget %s, %d worker(s))", lang, len(items),
                budget if budget > 0 else "drain", max_workers)

    # Blocks, not one giant submission: a 130k-clip prewarm runs for hours, and
    # a progress line every block is the difference between monitorable and
    # silent. Block size trades log noise against reporting latency.
    block = 10_000
    for start in range(0, len(items), block):
        results, errors = materialize_many(items[start:start + block], materializer,
                                           max_workers=max_workers)
        for res in results:
            report.written += 0 if res.skipped else 1
            report.skipped += 1 if res.skipped else 0
        for target, exc_type, msg in errors:
            report.errors[exc_type] += 1
            if len(report.examples) < 20:
                report.examples.append(f"{target}: {exc_type}: {msg}")
        done = min(start + block, len(items))
        LOGGER.info("prewarm %s: %d/%d candidate(s) done "
                    "(%d written, %d skipped, %d error(s))",
                    lang, done, len(items), report.written, report.skipped,
                    sum(report.errors.values()))
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="materialize",
        description="Stage 3: verify/backfill external clips, or prewarm a pool's top candidates.",
    )
    ap.add_argument("--out-dir", help="batch manifest root (holds <lang>/); backfill mode")
    ap.add_argument("--lang", required=True, help="language to verify or prewarm")
    ap.add_argument("--pool-dir", required=True, help="Stage-1 pool root (native_id source)")
    ap.add_argument("--audio-root", required=True, help="root external clips were written under")
    ap.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    ap.add_argument("--workers", type=int, default=32,
                    help="parallel fetch/decode threads (96-core nodes)")
    ap.add_argument("--prewarm", action="store_true",
                    help="materialize the pool's top external candidates ahead of "
                         "assemble (idle-node fetch farm) instead of backfilling")
    ap.add_argument("--budget", type=int, default=130_000,
                    help="prewarm: top-N candidates (~1.3x a 100k batch; <=0 = drain)")
    ap.add_argument("--no-gated", action="store_true",
                    help="prewarm: skip Hub sources needing terms")
    ap.add_argument("--report", help="write the JSON report here")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    if args.prewarm:
        report = prewarm_pool(
            args.lang, args.pool_dir, args.audio_root, args.budget,
            target_sr=args.sample_rate, max_workers=args.workers,
            include_gated=not args.no_gated,
        )
        payload = {"mode": "prewarm", "lang": args.lang,
                   "budget": args.budget, **report.as_dict()}
    else:
        if not args.out_dir:
            ap.error("--out-dir is required in backfill mode (or pass --prewarm)")
        report = ensure_batch_audio(
            args.out_dir, args.lang, args.pool_dir, args.audio_root,
            target_sr=args.sample_rate, max_workers=args.workers,
        )
        payload = report.as_dict()
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))
    LOGGER.info("materialize %s: %s", args.lang, payload)
    return 0


__all__ = [
    "DEFAULT_SAMPLE_RATE",
    "Fetcher",
    "MaterializeError",
    "MaterializeReport",
    "MaterializeResult",
    "Materializer",
    "ensure_batch_audio",
    "external_paths_in_batches",
    "hf_hub_file_fetcher",
    "local_file_fetcher",
    "main",
    "materialize_array",
    "materialize_file",
    "materialize_many",
    "materialize_source_stream",
    "prewarm_pool",
    "probe",
    "verify",
    "write_flac",
]


if __name__ == "__main__":
    sys.exit(main())
