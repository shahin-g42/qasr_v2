"""Stage-1 candidate pools: the on-disk handoff between prepare and assemble.

The staged build splits the old single-process ``Builder`` at its natural seam.
Stage 1 (``prepare``) streams metadata for the sources assigned to one node,
runs everything that needs *no* audio bytes -- normalize, the text gates,
richness, accent tagging -- and writes the survivors here as :class:`PoolRecord`
shards. Stage 2 (``assemble``) reads a whole language's pools back, ranks them,
and fills the sequential 100k batches.

Why an on-disk pool rather than a stream
----------------------------------------
Two constraints force the batch-assembly decision to see a whole language at
once, and neither survives a single streaming pass:

* **Best-first selection.** "Quality *and* richness" means filling a batch with
  the highest ``composite`` candidates, not the first to arrive. That needs every
  candidate ranked before any is placed.
* **Bounded memory.** A language pool is millions of records. Holding them all is
  out; holding one record per sorted shard is not.

So each shard is written sorted by ``composite`` descending, and Stage 2 does a
k-way merge (``heapq.merge``) across shards. The merge yields globally best-first
order while keeping only one record per shard live -- the same trick an external
merge sort uses. The per-source cap in :class:`~data_processing.distribute.Distributor`
then guarantees ``max_per_source_fraction`` even though the stream is score-ordered
rather than round-robin: once a source fills its share its remaining candidates are
refused and the next-best from another source takes the slot, so the batch stays
mixed *and* best-first.

Identity is fixed here, once
----------------------------
``audio_filepath`` is written at prepare time and never changes downstream. For a
local source it is the source's own path; for an external source it is the
deterministic materialized path from :func:`~data_processing.datasets.base.derive_materialized_path`.
Because the ledger key, the batch manifest and the Stage-3 audio file all derive
from this one string, they cannot disagree.
"""

from __future__ import annotations

import heapq
import json
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .canonical import Meta, _atomic_write, _open_read

LOGGER = logging.getLogger("data_processing.candidate")

#: Records per pool shard. Matches one batch-language quota so a shard is a
#: natural unit of work for resumption, and bounds the in-memory sort Stage 1
#: does before writing. Larger shards mean fewer open files during the Stage-2
#: merge; smaller ones mean a cheaper sort and finer resumption.
DEFAULT_POOL_SHARD_SIZE = 100_000


@dataclass(slots=True)
class PoolRecord:
    """One gated, scored candidate awaiting batch placement.

    Carries everything Stage 2 needs to rank the sample, reconstruct its
    ``Sample``/``Meta``, and -- for external sources -- fetch its audio. It does
    *not* carry audio bytes or a final transcript: the LLM correction and audio
    materialization both happen in Stage 2, so ``normalized_text`` is the input to
    both and ``duration`` is None until an external clip is decoded.
    """

    #: Final identity key. Local: the source path. External: the materialized path.
    audio_filepath: str
    #: Registry name of the source. Drives the per-source cap and the merge grouping.
    source: str
    lang: str
    #: Transcript after deterministic normalization; the LLM's input and the
    #: fallback final text.
    normalized_text: str

    #: Correctness composite from ``quality.gate`` (script purity, rate, length).
    quality: float = 0.0
    #: Content richness (lexical variety + distinct-unit count + length band).
    richness: float = 0.0
    #: ``quality x richness``: the best-first ranking key.
    composite: float = 0.0

    #: The source's own clip identifier, needed to fetch external audio. Empty
    #: for local sources, whose ``audio_filepath`` already is the fetch key.
    native_id: str = ""
    #: True when Stage 3 must download/decode/rewrite this clip at 16 kHz mono.
    external: bool = False
    #: Known duration in seconds, or None until an external clip is materialized.
    duration: float | None = None

    # --- sidecar provenance, carried so Stage 2 can rebuild Meta verbatim ----
    dataset: str = ""
    source_text: str | None = None
    accent: str | None = None
    itn_applied: bool = False
    diacritics_ratio: float | None = None
    #: Triaged at prepare time: only these are sent to the LLM (~18% of traffic).
    needs_llm: bool = False
    stages: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

    # --- serialization ------------------------------------------------------
    def to_dict(self) -> dict[str, object]:
        """Compact JSON form: always-present keys inline, the rest only when set.

        Floats and booleans are written explicitly rather than filtered by
        falsiness -- ``0.0 == False`` in Python, so a "drop falsy" rule would
        silently discard a legitimate zero-valued score.
        """
        out: dict[str, object] = {
            "audio_filepath": self.audio_filepath,
            "source": self.source,
            "lang": self.lang,
            "normalized_text": self.normalized_text,
            "quality": self.quality,
            "richness": self.richness,
            "composite": self.composite,
        }
        if self.native_id:
            out["native_id"] = self.native_id
        if self.external:
            out["external"] = True
        if self.duration is not None:
            out["duration"] = self.duration
        if self.dataset:
            out["dataset"] = self.dataset
        # source_text usually equals normalized_text after a no-op normalize; only
        # store it when it actually differs, which is the audit-relevant case.
        if self.source_text is not None and self.source_text != self.normalized_text:
            out["source_text"] = self.source_text
        if self.accent:
            out["accent"] = self.accent
        if self.itn_applied:
            out["itn_applied"] = True
        if self.diacritics_ratio is not None:
            out["diacritics_ratio"] = self.diacritics_ratio
        if self.needs_llm:
            out["needs_llm"] = True
        if self.stages:
            out["stages"] = self.stages
        if self.metrics:
            out["metrics"] = self.metrics
        return out

    @classmethod
    def from_dict(cls, raw: dict) -> PoolRecord:
        """Rebuild from a parsed pool line, ignoring any unknown key."""
        known = set(cls.__slots__)
        return cls(**{k: v for k, v in raw.items() if k in known})

    # --- reconstruction helpers for Stage 2 ---------------------------------
    def to_meta(self) -> Meta:
        """The sidecar record as Stage 1 knows it. Stage 2 adds llm/erasure."""
        meta = Meta(audio_filepath=self.audio_filepath, dataset=self.dataset or self.source)
        meta.accent = self.accent
        meta.quality = self.quality
        meta.source_text = self.source_text
        meta.normalized_text = self.normalized_text
        meta.itn_applied = self.itn_applied
        meta.diacritics_ratio = self.diacritics_ratio
        meta.stages = dict(self.stages)
        meta.metrics = dict(self.metrics)
        meta.metrics.setdefault("richness", self.richness)
        return meta


# --- shard naming -----------------------------------------------------------
def pool_shard_path(
    pool_dir: str | Path,
    lang: str,
    source: str,
    index: int,
    gzipped: bool = False,
) -> Path:
    """``<pool_dir>/<lang>/<source>/part-<index>.jsonl[.gz]``.

    One directory per source keeps a source's shards together, so Stage 2 can
    glob a single language and still tell which source each shard came from.
    """
    suffix = ".jsonl.gz" if gzipped else ".jsonl"
    return Path(pool_dir) / lang / source / f"part-{index:05d}{suffix}"


def iter_pool_shards(pool_dir: str | Path, lang: str, source: str | None = None) -> list[Path]:
    """Sorted pool shard paths for one language (optionally one source).

    Sorted so a resumed Stage 2 reads shards in a stable order and so the merge
    is reproducible run to run.
    """
    base = Path(pool_dir) / lang
    if not base.is_dir():
        return []
    if source is not None:
        base = base / source
    if not base.is_dir():
        return []
    shards = [p for p in base.rglob("part-*.jsonl*") if p.is_file()]
    return sorted(shards)


# --- I/O --------------------------------------------------------------------
def write_pool(path: str | Path, records: Iterable[PoolRecord], *, sort: bool = True) -> int:
    """Write one pool shard atomically, sorted by ``composite`` descending.

    Sorting at the write boundary is what makes the Stage-2 k-way merge a global
    best-first read: ``heapq.merge`` only produces a sorted stream if every input
    is already sorted the same way. ``sort=False`` is for callers that hand in a
    pre-sorted buffer and want to skip the re-sort.
    """
    rows = list(records)
    if sort:
        rows.sort(key=lambda r: r.composite, reverse=True)

    def emit(handle) -> None:
        for rec in rows:
            handle.write(json.dumps(rec.to_dict(), ensure_ascii=False))
            handle.write("\n")

    _atomic_write(Path(path), emit)
    return len(rows)


def read_pool(path: str | Path) -> Iterator[PoolRecord]:
    """Stream a pool shard back, transparently handling ``.gz``."""
    with _open_read(Path(path)) as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield PoolRecord.from_dict(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: malformed pool record: {exc}") from exc


def merge_pools(paths: Iterable[str | Path]) -> Iterator[PoolRecord]:
    """K-way merge of sorted pool shards, best-first (``composite`` descending).

    Each shard must already be sorted descending -- :func:`write_pool` guarantees
    it. The merge holds one record per shard, so memory is O(shards), not
    O(records): the external-sort property that lets a language pool of millions
    be ranked on one node. Open-file count equals the shard count, so prefer
    fewer/larger shards when ``ulimit -n`` is tight.
    """
    iterators = [read_pool(p) for p in paths]
    if not iterators:
        return
    yield from heapq.merge(*iterators, key=lambda r: r.composite, reverse=True)


def interleave_pools(paths: Iterable[str | Path]) -> Iterator[PoolRecord]:
    """Round-robin across pool shards, first-arrival within each.

    The source-balanced alternative to :func:`merge_pools`: it makes no ranking
    guarantee, so it is what a caller uses when it wants even source coverage and
    will do its own selection downstream (or when testing placement independent of
    score). Exhausted shards drop out and the rest continue.
    """
    from collections import deque

    pending = deque(read_pool(p) for p in paths)
    while pending:
        it = pending.popleft()
        try:
            rec = next(it)
        except StopIteration:
            continue
        yield rec
        pending.append(it)


__all__ = [
    "DEFAULT_POOL_SHARD_SIZE",
    "PoolRecord",
    "interleave_pools",
    "iter_pool_shards",
    "merge_pools",
    "pool_shard_path",
    "read_pool",
    "write_pool",
]
