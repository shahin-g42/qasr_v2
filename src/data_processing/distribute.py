"""Batch distributor: turns a scored candidate stream into sequential datasets.

Emits batches of exactly ``batch_size`` unique samples per language, labelled
``b0000``, ``b0001``, ... so each batch is a self-contained small dataset.

Identity
--------
``audio_filepath`` is the **only** identity key, per the corpus spec. Two
records with different paths are different samples even when the transcripts are
byte-identical. That is the right call for ASR: the same sentence recorded in
two rooms by two speakers is genuinely two training samples.

It is also a call that has to be defended, because it is exactly what let the
zh corpus fail -- 7,851 distinct transcripts entered as 156,540 records, every
one of them "unique" by path. So text repetition is tracked here as a separate,
reported and capped concern rather than as identity. ``max_per_text`` bounds how
many renditions of one transcript a batch may hold, and
``min_distinct_text_fraction`` refuses a batch that is mostly copies. Neither
overrides the path key; both are measured against it.

Ledger
------
Uniqueness must hold *across batches and across runs*, so membership lives in a
SQLite table rather than a Python set. At 50M paths a set of strings costs
several GB of RSS and is lost on exit; SQLite costs bounded RAM, survives a
crash, and stays queryable for the overlap audits that found 962 eval paths
already present in the ar train split.

Claiming is one statement -- ``INSERT OR IGNORE`` -- and its ``rowcount`` is the
answer. That makes the check-and-reserve atomic, so two workers pointed at the
same ledger cannot both accept one path.

Cost: roughly 50k-100k claims/second. At 50M candidates that is 10-15 minutes,
which is noise next to the LLM stage and is deliberately not optimized further.
"""

from __future__ import annotations

import gzip
import json
import logging
import sqlite3
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .canonical import (
    Meta,
    Sample,
    iter_shards,
    read_manifest,
    shard_paths,
    write_manifest,
    write_sidecar,
)
from .dedup import DedupConfig, Deduper, DiversityReport, key_hash

LOGGER = logging.getLogger("data_processing.distribute")

#: Batch label reserved for paths that are unavailable by policy rather than by
#: having been emitted -- eval splits, or anything explicitly poisoned.
EXCLUDED_BATCH = "__excluded__"


class Outcome(str, Enum):
    """Why a candidate was or was not accepted. Counted in the run report."""

    ACCEPTED = "accepted"
    #: ``audio_filepath`` already claimed by an earlier batch or earlier run.
    DUP_PATH = "dup_path"
    #: Path is in the exclusion table -- typically an eval-split member.
    EXCLUDED = "excluded"
    #: The transcript already has ``max_per_text`` renditions in this language.
    TEXT_CAPPED = "text_capped"
    #: This source already holds its share of the open batch.
    SOURCE_CAPPED = "source_capped"
    #: The open batch was already full when the candidate arrived. Unreachable
    #: by construction today -- ``offer`` rotates the moment a batch fills, so
    #: the open batch is never over capacity. Kept as the invariant guard that
    #: makes it stay true if auto-rotation is ever removed.
    BATCH_FULL = "batch_full"
    #: A whole batch discarded at close for falling below the diversity floor.
    #: Counted per sample so the report reconciles against ACCEPTED.
    REFUSED_LOW_DIVERSITY = "refused_low_diversity"


@dataclass(slots=True)
class DistributeConfig:
    """Distribution policy. Every bound here is inclusive."""

    #: Samples per language per batch. The spec fixes this at 100k.
    batch_size: int = 100_000
    #: No single source may exceed this share of one batch. Without it a batch
    #: drains whichever source streams first, and "distributed" becomes
    #: "one dataset, sequentially". 0.40 keeps at least three sources present.
    max_per_source_fraction: float = 0.40
    #: Renditions of one transcript allowed per language, across all batches.
    max_per_text: int = 2
    #: A batch below this distinct-transcript fraction is refused, not warned
    #: about. This is the gate the zh build needed.
    min_distinct_text_fraction: float = 0.50
    #: Emit gzip shards. Phase B audio is what dominates storage, not text.
    gzipped: bool = False


@dataclass(frozen=True, slots=True)
class Candidate:
    """One scored sample awaiting placement."""

    sample: Sample
    meta: Meta
    #: Registry name of the source it came from. Drives the per-source cap.
    source: str

    #: Weight given to accent-erasure severity in :attr:`rank`. Fixed rather
    #: than configurable on purpose: rank feeds only the diversity *measurement*
    #: in ``Deduper`` -- its best-N revocation is deliberately unused, because
    #: revoking a sample from an already-written shard would mean rewriting it.
    #: A tunable that changes no decision is a trap, so there is no knob.
    ERASURE_WEIGHT = 0.5

    @property
    def rank(self) -> float:
        """Tie-break score: quality, discounted for dialect damage."""
        q = self.meta.quality if self.meta.quality is not None else 1.0
        sev = float(self.meta.metrics.get("erasure_severity", 0.0))
        return q - self.ERASURE_WEIGHT * sev


@dataclass(slots=True)
class Batch:
    """A completed batch, ready to write."""

    lang: str
    label: str
    index: int
    samples: list[Sample] = field(default_factory=list)
    metas: list[Meta] = field(default_factory=list)
    sources: Counter = field(default_factory=Counter)
    accents: Counter = field(default_factory=Counter)
    diversity: DiversityReport | None = None
    total_duration: float = 0.0

    @property
    def size(self) -> int:
        return len(self.samples)

    def hours(self) -> float:
        return self.total_duration / 3600.0


class SeenLedger:
    """Durable record of every ``audio_filepath`` already consumed or excluded."""

    def __init__(self, path: str | Path, buffer_claims: bool = True) -> None:
        """Open (and create) the ledger.

        ``buffer_claims=False`` makes every claim an immediate
        ``INSERT OR IGNORE`` whose ``rowcount`` is the answer, so the reserve is
        atomic across *processes* -- use that when several workers share one
        ledger. The default buffers the *writes* only; every claim still reads
        the table, so a resumed run pointed at an existing ledger sees what the
        previous run claimed. Two buffered processes can still race between the
        read and the flush -- that window is exactly what ``buffer_claims=False``
        closes.
        """
        self.path = Path(path)
        self.buffer_claims = buffer_claims
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        # NORMAL is the right trade here: a crash can lose the last transaction,
        # which release_batch() exists to reconcile. It cannot corrupt the table.
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS seen ("
            "  fp     TEXT PRIMARY KEY,"
            "  lang   TEXT NOT NULL,"
            "  batch  TEXT NOT NULL,"
            "  source TEXT NOT NULL DEFAULT '',"
            "  ts     REAL NOT NULL"
            ")"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS seen_batch ON seen(batch)")
        self._pending: list[tuple] = []
        #: Mirror of _pending for O(1) membership; cleared on flush.
        self._pending_fps: set[str] = set()
        self.claims = 0
        self.duplicate_hits = 0

    _INSERT = "INSERT OR IGNORE INTO seen(fp, lang, batch, source, ts) VALUES (?,?,?,?,?)"

    # --- membership ---------------------------------------------------------
    def claim(self, fp: str, lang: str, batch: str, source: str = "") -> bool:
        """Reserve ``fp``. True only for the caller that actually got it."""
        if not self.buffer_claims:
            cur = self._conn.execute(self._INSERT, (fp, lang, batch, source, time.time()))
            won = cur.rowcount == 1
            self.claims += 1 if won else 0
            self.duplicate_hits += 0 if won else 1
            return won
        if fp in self._pending_fps or self._committed(fp):
            self.duplicate_hits += 1
            return False
        self._pending.append((fp, lang, batch, source, time.time()))
        self._pending_fps.add(fp)
        if len(self._pending) >= 1024:
            self.flush()
        return True

    def _committed(self, fp: str) -> bool:
        """Whether ``fp`` is already in the durable table.

        An indexed point SELECT, so no fsync: buffering still saves the write
        transaction per claim, which is the expensive half. At ~1-2us a lookup
        this costs well under a minute across 50M candidates.
        """
        return self._conn.execute("SELECT 1 FROM seen WHERE fp = ?", (fp,)).fetchone() is not None

    def is_claimed(self, fp: str) -> bool:
        """Membership test that sees both committed and buffered claims."""
        return self.batch_of(fp) is not None

    def batch_of(self, fp: str) -> str | None:
        """Which batch holds ``fp``, or None if it is still available.

        One lookup answers both "already used?" and "or excluded by policy?" --
        the batch column distinguishes them, so callers never need two queries.
        """
        if fp in self._pending_fps:
            for p in reversed(self._pending):
                if p[0] == fp:
                    return p[2]
            return None
        row = self._conn.execute("SELECT batch FROM seen WHERE fp = ?", (fp,)).fetchone()
        return row[0] if row else None

    def flush(self) -> int:
        """Commit buffered claims. Returns how many were genuinely new."""
        if not self._pending:
            return 0
        before = self._conn.total_changes
        self._conn.executemany(self._INSERT, self._pending)
        inserted = self._conn.total_changes - before
        self.claims += inserted
        self.duplicate_hits += len(self._pending) - inserted
        self._pending.clear()
        self._pending_fps.clear()
        return inserted

    # --- exclusions ---------------------------------------------------------
    @staticmethod
    def _iter_paths(path: Path) -> Iterator[str]:
        """Pull ``audio_filepath`` out of a manifest, tolerantly.

        Deliberately NOT :func:`canonical.read_manifest`. That function enforces
        the four-key *output* contract, and real v7.6 inputs do not meet it:
        every file carries only ``{audio_filepath, text, duration}``, with no
        ``lang`` and ``duration`` as a string. Using the strict reader here made
        every eval file raise, get caught, log a warning and contribute zero
        paths -- so the exclusion set was silently empty and the measured
        ar eval/train leak stayed wide open. Exclusion needs exactly one field,
        so it parses for exactly one field.
        """
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("audio_filepath"):
                    yield str(row["audio_filepath"])

    def exclude(self, fps: Iterable[str], lang: str, reason: str = "eval") -> int:
        """Mark paths unavailable. Used to load eval splits before any build."""
        rows = [(fp, lang, EXCLUDED_BATCH, reason, time.time()) for fp in fps]
        before = self._conn.total_changes
        self._conn.executemany(
            "INSERT OR IGNORE INTO seen(fp, lang, batch, source, ts) VALUES (?,?,?,?,?)",
            rows,
        )
        added = self._conn.total_changes - before
        LOGGER.info("excluded %s path(s) as %r (%s already present)", f"{added:,}", reason, f"{len(rows) - added:,}")
        return added

    def load_eval_exclusions(self, root: str | Path, langs: Iterable[str] | None = None) -> dict[str, int]:
        """Pull every ``eval_*`` manifest under ``root`` into the exclusion set.

        This is not a theoretical concern: the ar split was measured with 962 of
        its 21,867 eval paths also present in train. Loading eval first makes
        that class of leak structurally impossible for every later batch.
        """
        root = Path(root)
        out: dict[str, int] = {}
        for lang_dir in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
            lang = lang_dir.name
            if langs and lang not in langs:
                continue
            paths: list[str] = []
            for manifest in sorted(lang_dir.glob("eval_*.jsonl*")):
                try:
                    paths.extend(self._iter_paths(manifest))
                except OSError as exc:
                    LOGGER.warning("could not read eval manifest %s: %s", manifest, exc)
            if paths:
                out[lang] = self.exclude(paths, lang, reason="eval")
        return out

    # --- batch bookkeeping --------------------------------------------------
    def release_batch(self, lang: str, label: str) -> int:
        """Drop claims belonging to one batch, so an interrupted build can retry.

        Needed because a crash after claiming but before writing the shard would
        otherwise strand those paths: absent from every manifest, unavailable to
        every future run.
        """
        self.flush()
        cur = self._conn.execute("DELETE FROM seen WHERE lang = ? AND batch = ?", (lang, label))
        n = cur.rowcount
        LOGGER.info("released %s claim(s) for %s/%s", f"{n:,}", lang, label)
        return n

    def released_incomplete(self, root: str | Path, lang: str, labels: Iterable[str]) -> int:
        """Release claims for batches whose shard is missing or short on disk."""
        total = 0
        for label in labels:
            manifest, _ = shard_paths(root, lang, label, 0)
            if manifest.exists():
                continue
            total += self.release_batch(lang, label)
        return total

    def batch_labels(self, lang: str) -> list[str]:
        self.flush()
        rows = self._conn.execute(
            "SELECT DISTINCT batch FROM seen WHERE lang = ? AND batch != ? ORDER BY batch",
            (lang, EXCLUDED_BATCH),
        ).fetchall()
        return [r[0] for r in rows]

    def next_index(self, lang: str) -> int:
        """First unused batch number, so runs continue the sequence."""
        labels = [b for b in self.batch_labels(lang) if b.startswith("b")]
        if not labels:
            return 0
        return max(int(b[1:]) for b in labels) + 1

    def stats(self) -> dict[str, int]:
        self.flush()
        total = self._conn.execute("SELECT COUNT(*) FROM seen").fetchone()[0]
        excluded = self._conn.execute(
            "SELECT COUNT(*) FROM seen WHERE batch = ?", (EXCLUDED_BATCH,)
        ).fetchone()[0]
        return {
            "claimed": total - excluded,
            "excluded": excluded,
            "pending_buffer": len(self._pending),
        }

    def overlap(self, lang_a: str, lang_b: str) -> int:
        """Paths claimed in both languages -- should always be zero."""
        self.flush()
        return self._conn.execute(
            "SELECT COUNT(*) FROM seen a JOIN seen b ON a.fp = b.fp "
            "WHERE a.lang = ? AND b.lang = ? AND a.batch != ? AND b.batch != ?",
            (lang_a, lang_b, EXCLUDED_BATCH, EXCLUDED_BATCH),
        ).fetchone()[0]

    def close(self) -> None:
        self.flush()
        self._conn.close()

    def __enter__(self) -> SeenLedger:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class Distributor:
    """Places candidates into sequential per-language batches."""

    def __init__(
        self,
        lang: str,
        ledger: SeenLedger,
        config: DistributeConfig | None = None,
        start_index: int | None = None,
    ) -> None:
        self.lang = lang
        self.ledger = ledger
        self.config = config or DistributeConfig()
        self._index = self.ledger.next_index(lang) if start_index is None else start_index
        self.outcomes: Counter = Counter()
        self.batches: list[Batch] = []
        #: Batches discarded at close for failing the diversity floor.
        self.refusals = 0
        #: transcript hash -> renditions accepted in already-flushed batches.
        self._committed_text: Counter = Counter()
        self._open_text: Counter = Counter()
        self._batch_deduper = Deduper(lang, self._dedup_config())
        self._open = Batch(lang=lang, label=self._label(self._index), index=self._index)

    def _dedup_config(self) -> DedupConfig:
        # min_distinct_fraction MUST be forwarded: without it the batch refuses
        # at dedup.py's default rather than at the configured threshold.
        return DedupConfig(
            max_per_key=self.config.max_per_text,
            min_distinct_fraction=self.config.min_distinct_text_fraction,
        )

    @staticmethod
    def _label(index: int) -> str:
        return f"b{index:04d}"

    # --- policy checks ------------------------------------------------------
    def _source_cap(self) -> int:
        return max(1, int(self.config.batch_size * self.config.max_per_source_fraction))

    def _source_full(self, source: str) -> bool:
        cap = self._source_cap()
        # The cap binds only once the batch has enough samples for it to mean
        # something; otherwise the first source is starved by its own cap.
        if self._open.size < cap:
            return False
        return self._open.sources[source] >= cap

    def _text_full(self, text: str) -> bool:
        h = key_hash(text, self.lang)
        return (self._committed_text[h] + self._open_text[h]) >= self.config.max_per_text

    # --- placement ----------------------------------------------------------
    def _admission(self, fp: str, source: str, text: str) -> Outcome:
        """The placement decision for an identity, with no side effects.

        Shared by :meth:`offer` and :meth:`would_place` so a pre-flight check can
        never disagree with the real offer. Order matters: exclusion/dup first
        (cheapest and most absolute), then the batch/source/text caps.
        """
        # Exclusion and prior-claim live in the same table, so one lookup
        # answers both and the batch column says which.
        held_by = self.ledger.batch_of(fp)
        if held_by is not None:
            return Outcome.EXCLUDED if held_by == EXCLUDED_BATCH else Outcome.DUP_PATH
        if self._open.size >= self.config.batch_size:
            return Outcome.BATCH_FULL
        if self._source_full(source):
            return Outcome.SOURCE_CAPPED
        if self._text_full(text):
            return Outcome.TEXT_CAPPED
        return Outcome.ACCEPTED

    def would_place(self, fp: str, source: str, text: str) -> bool:
        """Whether a candidate with this identity would be placed right now.

        Lets a caller do expensive, irreversible work -- fetching and decoding an
        external clip -- only for candidates that will actually land in a batch,
        without duplicating (and drifting from) the cap logic in :meth:`offer`.
        Rejection is monotonic within an open batch, so a False here stays False
        until the batch rotates.
        """
        return self._admission(fp, source, text) is Outcome.ACCEPTED

    def offer(self, cand: Candidate) -> Outcome:
        """Try to place one candidate. Returns why it was or was not taken."""
        if cand.sample.lang != self.lang:
            raise ValueError(f"candidate lang {cand.sample.lang!r} != distributor {self.lang!r}")

        text = cand.sample.text
        outcome = self._admission(cand.sample.audio_filepath, cand.source, text)
        self.outcomes[outcome] += 1
        if outcome is not Outcome.ACCEPTED:
            return outcome

        # Measuring instrument, not decision-maker: the batch-local Deduper
        # counts every placed offer so the diversity report is accurate. Its
        # best-N revocation is deliberately unused -- revoking a sample from an
        # already-written shard would mean rewriting it, and candidates arrive
        # pre-scored and best-first so placement order is already near-optimal.
        self._batch_deduper.offer(self._open.size, text, cand.rank)

        h = key_hash(text, self.lang)
        self._open_text[h] += 1
        self.ledger.claim(cand.sample.audio_filepath, self.lang, self._open.label, cand.source)

        self._open.samples.append(cand.sample)
        self._open.metas.append(cand.meta)
        self._open.sources[cand.source] += 1
        self._open.accents[cand.meta.accent or "unknown"] += 1
        self._open.total_duration += cand.sample.duration

        if self._open.size >= self.config.batch_size:
            self.close_batch()
        return Outcome.ACCEPTED

    # --- batch lifecycle ----------------------------------------------------
    def close_batch(self, allow_partial: bool = False) -> Batch | None:
        """Finalize the open batch. Returns None if it was empty."""
        batch = self._open
        if batch.size == 0:
            return None
        if batch.size < self.config.batch_size and not allow_partial:
            LOGGER.warning(
                "%s batch %s is short: %s of %s. Held open -- it is NOT written, "
                "because a short batch silently breaks the 100k-per-language contract.",
                self.lang, batch.label, f"{batch.size:,}", f"{self.config.batch_size:,}",
            )
            return None

        batch.diversity = self._batch_deduper.report()
        if not batch.diversity.acceptable:
            LOGGER.error(
                "%s batch %s REFUSED: distinct-transcript fraction %.3f < %.3f. "
                "%s distinct of %s samples. This is the zh failure mode.",
                self.lang, batch.label, batch.diversity.distinct_fraction,
                self.config.min_distinct_text_fraction,
                f"{batch.diversity.distinct:,}", f"{batch.diversity.total:,}",
            )
            # "Refused" has to mean neither written nor consumed. Releasing the
            # claims is what keeps those paths available to a later, better-mixed
            # batch -- without it a refusal would strand a whole batch of samples
            # that no manifest ever carried. The text budget is left unspent for
            # the same reason. Logging an error and emitting anyway is what the
            # zh build effectively had, and it shipped 7,851 transcripts as
            # 156,540 records.
            self.ledger.release_batch(self.lang, batch.label)
            self.refusals += 1
            self.outcomes[Outcome.REFUSED_LOW_DIVERSITY] += batch.size
            self._open_text.clear()
            self._rotate()
            return None

        for h, n in self._open_text.items():
            self._committed_text[h] += n
        self._open_text.clear()

        self.ledger.flush()
        self.batches.append(batch)
        LOGGER.info(
            "%s batch %s closed: %s samples, %.1f h, %s source(s), distinct=%.3f",
            self.lang, batch.label, f"{batch.size:,}", batch.hours(),
            len(batch.sources), batch.diversity.distinct_fraction,
        )
        self._rotate()
        return batch

    def _rotate(self) -> None:
        """Advance to the next batch label with a fresh deduper and counter."""
        self._index += 1
        self._batch_deduper = Deduper(self.lang, self._dedup_config())
        self._open = Batch(lang=self.lang, label=self._label(self._index), index=self._index)

    def pending(self) -> Batch:
        """The open batch, for progress reporting and for a forced flush."""
        return self._open

    # --- writing ------------------------------------------------------------
    def write(self, batch: Batch, out_dir: str | Path, part: int = 0) -> tuple[Path, Path]:
        """Write one batch as a manifest + sidecar pair.

        ``part`` is the shard index within the batch label: a whole-language
        assemble writes part 0, while a hash-sliced multi-node assemble
        (``assemble.run_assemble(pool_part=...)``) writes its slice's part, so
        the parts of one label compose one whole batch.
        """
        out_dir = Path(out_dir) / self.lang
        manifest, sidecar = shard_paths(
            out_dir, self.lang, batch.label, part, gzipped=self.config.gzipped)
        write_manifest(manifest, batch.samples)
        write_sidecar(sidecar, batch.metas)
        return manifest, sidecar

    def report(self) -> dict:
        self.ledger.flush()
        return {
            "lang": self.lang,
            "batches_closed": len(self.batches),
            "batches_refused": self.refusals,
            "batch_labels": [b.label for b in self.batches],
            "open_batch_size": self._open.size,
            "outcomes": dict(self.outcomes),
            "ledger": self.ledger.stats(),
        }


def iter_candidates(
    samples: Iterable[Sample],
    metas: Iterable[Meta] | None = None,
    source: str = "unknown",
) -> Iterator[Candidate]:
    """Zip samples with sidecars into candidates.

    When no sidecar is supplied, a minimal one is synthesized so the distributor
    never has to special-case unprocessed input -- but ``quality`` stays None so
    a missing score is distinguishable from a measured 0.0.
    """
    if metas is None:
        for s in samples:
            yield Candidate(s, Meta(audio_filepath=s.audio_filepath, dataset=source), source)
        return
    for s, m in zip(samples, metas, strict=False):
        yield Candidate(s, m, source)


def existing_paths(root: str | Path, lang: str) -> set[str]:
    """Every ``audio_filepath`` in already-written shards, for offline audits."""
    out: set[str] = set()
    for manifest, _ in iter_shards(Path(root) / lang, lang=lang):
        out.update(s.audio_filepath for s in read_manifest(manifest))
    return out


__all__ = [
    "EXCLUDED_BATCH",
    "Batch",
    "Candidate",
    "DistributeConfig",
    "Distributor",
    "Outcome",
    "SeenLedger",
    "existing_paths",
    "iter_candidates",
]
