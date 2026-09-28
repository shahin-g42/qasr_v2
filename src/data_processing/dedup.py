"""Text-level deduplication and content-diversity metrics.

The pipeline's only identity key was ``(audio_filepath, original_text)``, which
detects the same record arriving twice across cleaning passes. It cannot detect
the same *sentence* arriving twice under different audio -- which is exactly
what a TTS-rendered dataset does. That single gap let 7,851 distinct Chinese
sentences enter training as 156,540 records reported as 198 hours, when the
real unique content was about 10 hours. The model then scored 40% CER against
an eval set drawn from an entirely different corpus.

Keys come from ``normalize.dedup_key``, which folds case, punctuation,
whitespace and vocalization, so two renditions of one sentence collapse no
matter how differently they were punctuated or vocalized.

``max_per_key`` defaults to 2 rather than 1 on purpose. Multiple recordings of
one sentence are *legitimate* ASR data -- different speakers, rooms and accents
teach acoustic invariance, and collapsing to one would throw that away. The
defect being targeted is 228 renditions of "Hi, I'm fine. And you?", not two.

Memory: the per-key heaps scale with the number of DISTINCT keys, not the
number of records, so a duplicated corpus costs almost nothing and a diverse
one costs roughly 100 bytes per unique transcript. At 10M unique keys that is
about 1 GB -- run large corpora in shards if the build node is tight.
"""

from __future__ import annotations

import hashlib
import heapq
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

from .normalize import dedup_key

LOGGER = logging.getLogger("data_processing.dedup")


def key_hash(text: str, lang: str) -> int:
    """64-bit digest of the dedup key.

    Storing an int instead of the transcript keeps the working set small at
    scale. At 10M distinct keys the birthday-bound collision probability is
    about 3e-6, which is far below every other error source in the pipeline.
    """
    digest = hashlib.blake2b(dedup_key(text, lang).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


@dataclass(slots=True)
class DedupConfig:
    """Deduplication policy."""

    #: Renditions of one transcript to keep. The best-scoring ones win.
    max_per_key: int = 2
    #: A dataset whose distinct-text fraction falls below this is rejected
    #: wholesale. This is the gate that would have stopped the zh build before
    #: any GPU time was spent: that corpus sits at 0.05.
    min_distinct_fraction: float = 0.50
    #: Remember an example transcript once a key repeats this often, so the
    #: report can name the problem instead of only quantifying it.
    example_threshold: int = 5
    #: Cap on retained examples, to bound memory on pathological corpora.
    max_examples: int = 50
    top_repeats: int = 20


@dataclass(frozen=True, slots=True)
class DiversityReport:
    """Content-diversity measurement for one dataset or language."""

    total: int
    distinct: int
    kept: int
    dropped: int
    max_repeat: int
    top: tuple[tuple[int, str], ...] = ()
    #: Datasets below ``min_distinct_fraction`` are refused, not merely warned
    #: about. A warning is what the previous pipeline effectively had.
    acceptable: bool = True

    @property
    def distinct_fraction(self) -> float:
        return self.distinct / self.total if self.total else 1.0

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "distinct": self.distinct,
            "distinct_fraction": round(self.distinct_fraction, 5),
            "kept": self.kept,
            "dropped": self.dropped,
            "max_repeat": self.max_repeat,
            "acceptable": self.acceptable,
            "top_repeats": [{"count": c, "text": t} for c, t in self.top],
        }


@dataclass(slots=True)
class _Slot:
    """Bounded best-N tracker for one dedup key."""

    heap: list = field(default_factory=list)   # min-heap of (quality, seq)
    count: int = 0
    example: str | None = None


class Deduper:
    """Streaming deduplicator that keeps the highest-quality renditions."""

    def __init__(self, lang: str, config: DedupConfig | None = None) -> None:
        self.lang = lang
        self.config = config or DedupConfig()
        self._slots: dict[int, _Slot] = {}
        self._dropped: set[int] = set()
        #: hash -> slot, for the handful of keys we retain an example of.
        self._examples: dict[int, _Slot] = {}
        self._total = 0

    @property
    def dropped_seqs(self) -> set[int]:
        """Sequence numbers evicted after having been accepted.

        A caller writing records as it goes must delete or skip these at the
        end. Keeping them explicit avoids requiring the whole corpus in memory.
        """
        return self._dropped

    def offer(self, seq: int, text: str, quality: float = 1.0) -> bool:
        """Decide whether the sample at ``seq`` should be kept.

        Returns True when the sample is currently in the keep set. A True can
        later be revoked by a better sample for the same key, in which case
        ``seq`` appears in :attr:`dropped_seqs`.
        """
        self._total += 1
        cfg = self.config
        h = key_hash(text, self.lang)
        slot = self._slots.get(h)

        if slot is None:
            slot = _Slot()
            self._slots[h] = slot

        slot.count += 1
        if slot.count >= cfg.example_threshold and slot.example is None and len(self._examples) < cfg.max_examples:
            slot.example = text[:160]
            self._examples[h] = slot

        if len(slot.heap) < cfg.max_per_key:
            heapq.heappush(slot.heap, (quality, seq))
            return True

        worst_quality, worst_seq = slot.heap[0]
        if quality > worst_quality:
            heapq.heapreplace(slot.heap, (quality, seq))
            self._dropped.add(worst_seq)
            return True

        self._dropped.add(seq)
        return False

    def report(self) -> DiversityReport:
        """Summarize diversity and decide whether the dataset is acceptable."""
        cfg = self.config
        counts = [s.count for s in self._slots.values()]
        distinct = len(self._slots)
        max_repeat = max(counts) if counts else 0
        kept = sum(min(c, cfg.max_per_key) for c in counts)

        hottest = sorted(self._slots.items(), key=lambda kv: -kv[1].count)[: cfg.top_repeats]
        top = tuple(
            (s.count, s.example or "<example not retained>")
            for _, s in hottest
            if s.count > 1
        )

        fraction = distinct / self._total if self._total else 1.0
        acceptable = fraction >= cfg.min_distinct_fraction
        return DiversityReport(
            total=self._total,
            distinct=distinct,
            kept=kept,
            dropped=self._total - kept,
            max_repeat=max_repeat,
            top=top,
            acceptable=acceptable,
        )


def dedup_stream(
    records: Iterable[tuple[int, str, float]],
    lang: str,
    config: DedupConfig | None = None,
) -> tuple[list[int], DiversityReport]:
    """Convenience wrapper: returns ``(kept_seqs, report)`` in one pass.

    Buffers the sequence numbers, so drive :class:`Deduper` directly for
    corpora at the 10M scale and let the caller hold the ordering instead.
    """
    deduper = Deduper(lang, config)
    seqs: list[int] = []
    for seq, text, quality in records:
        seqs.append(seq)
        deduper.offer(seq, text, quality)
    dropped = deduper.dropped_seqs
    return [s for s in seqs if s not in dropped], deduper.report()


__all__ = ["DedupConfig", "Deduper", "DiversityReport", "dedup_stream", "key_hash"]
