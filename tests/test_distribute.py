"""Tests for the batch distributor and its uniqueness ledger.

The two invariants worth the most are pinned here explicitly, because both were
broken once already:

``audio_filepath`` is the only identity key
    Identical transcripts under different paths are different samples. Text
    repetition is a separate, capped and *reported* concern -- never identity.

Uniqueness holds across batches and across runs
    Membership lives in SQLite, not a Python set, and claiming is a single
    ``INSERT OR IGNORE`` whose ``rowcount`` is the answer. That is what makes
    the check-and-reserve atomic across processes.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from data_processing.canonical import MANIFEST_KEYS, Meta, Sample, read_manifest, read_sidecar
from data_processing.distribute import (
    EXCLUDED_BATCH,
    Batch,
    Candidate,
    DistributeConfig,
    Distributor,
    Outcome,
    SeenLedger,
    existing_paths,
    iter_candidates,
)


def _sample(fp: str, text: str = "نص تجريبي", lang: str = "ar", duration: float = 4.0) -> Sample:
    return Sample(audio_filepath=fp, duration=duration, text=text, lang=lang)


def _candidate(
    fp: str,
    text: str = "نص تجريبي",
    source: str = "src_a",
    quality: float | None = None,
    erasure: float = 0.0,
    lang: str = "ar",
    duration: float = 4.0,
) -> Candidate:
    meta = Meta(audio_filepath=fp, dataset=source, quality=quality)
    if erasure:
        meta.metrics["erasure_severity"] = erasure
    return Candidate(_sample(fp, text, lang, duration), meta, source)


def _fill(dist: Distributor, n: int, prefix: str = "a", source: str = "src_a") -> list[Outcome]:
    """Offer ``n`` distinct-path, distinct-text candidates."""
    return [
        dist.offer(_candidate(f"/audio/{prefix}{i}.wav", f"{prefix} نص {i}", source))
        for i in range(n)
    ]


class TestSeenLedgerClaims(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_unbuffered_claim_is_atomic_second_caller_loses(self):
        """The documented cross-process guarantee: exactly one winner per path."""
        with SeenLedger(self.tmp / "l.db", buffer_claims=False) as led:
            self.assertTrue(led.claim("/a.wav", "ar", "b0000"))
            self.assertFalse(led.claim("/a.wav", "ar", "b0000"))
            self.assertTrue(led.claim("/b.wav", "ar", "b0000"))
            self.assertEqual(led.claims, 2)
            self.assertEqual(led.duplicate_hits, 1)

    def test_unbuffered_claim_is_visible_to_a_second_connection(self):
        """Atomicity is worthless unless the row is committed before the flush.

        A second connection stands in for a second worker process.
        """
        with SeenLedger(self.tmp / "l.db", buffer_claims=False) as led:
            led.claim("/shared.wav", "ar", "b0000")
            other = sqlite3.connect(str(self.tmp / "l.db"))
            try:
                row = other.execute("SELECT batch FROM seen WHERE fp = '/shared.wav'").fetchone()
            finally:
                other.close()
        self.assertEqual(row, ("b0000",))

    def test_buffered_claim_dedupes_before_flush(self):
        with SeenLedger(self.tmp / "l.db") as led:
            self.assertTrue(led.claim("/a.wav", "ar", "b0000"))
            self.assertFalse(led.claim("/a.wav", "ar", "b0000"))
            self.assertEqual(led.flush(), 1)
            self.assertTrue(led.is_claimed("/a.wav"))

    def test_buffered_claim_rejects_a_path_committed_by_another_ledger(self):
        """The buffered fast path must not shadow the durable table."""
        with SeenLedger(self.tmp / "l.db", buffer_claims=False) as first:
            first.claim("/a.wav", "ar", "b0000")
        with SeenLedger(self.tmp / "l.db") as second:
            self.assertFalse(second.claim("/a.wav", "ar", "b0001"))

    def test_buffered_claim_refuses_a_committed_path_before_flush(self):
        """Regression: buffering used to skip the read and re-claim freely.

        A resumed run points a fresh buffered ledger at a populated database. If
        ``claim`` only consults its own pending set, every path from the previous
        run looks available and the cross-run uniqueness guarantee -- the reason
        the ledger is in SQLite at all -- is silently void.
        """
        with SeenLedger(self.tmp / "l.db", buffer_claims=False) as first:
            first.claim("/carried.wav", "ar", "b0000")
        with SeenLedger(self.tmp / "l.db") as resumed:
            self.assertFalse(resumed.claim("/carried.wav", "ar", "b0001"))
            self.assertTrue(resumed.claim("/fresh.wav", "ar", "b0001"))
            self.assertEqual(resumed.flush(), 1)

    def test_flush_counts_only_rows_inserted(self):
        with SeenLedger(self.tmp / "l.db") as led:
            led.claim("/a.wav", "ar", "b0000")
            led.claim("/b.wav", "ar", "b0000")
            self.assertEqual(led.flush(), 2)
            self.assertEqual(led.flush(), 0)  # nothing pending

    def test_batch_of_distinguishes_excluded_from_emitted(self):
        """One lookup answers both 'used?' and 'excluded by policy?'."""
        with SeenLedger(self.tmp / "l.db", buffer_claims=False) as led:
            led.exclude(["/eval.wav"], "ar")
            led.claim("/train.wav", "ar", "b0003")
            self.assertEqual(led.batch_of("/eval.wav"), EXCLUDED_BATCH)
            self.assertEqual(led.batch_of("/train.wav"), "b0003")
            self.assertIsNone(led.batch_of("/unseen.wav"))
            self.assertTrue(led.is_claimed("/eval.wav"))
            self.assertFalse(led.is_claimed("/unseen.wav"))

    def test_stats_separate_claimed_from_excluded(self):
        with SeenLedger(self.tmp / "l.db", buffer_claims=False) as led:
            led.exclude(["/e1.wav", "/e2.wav"], "ar")
            led.claim("/t1.wav", "ar", "b0000")
            stats = led.stats()
        self.assertEqual(stats["excluded"], 2)
        self.assertEqual(stats["claimed"], 1)

    def test_release_batch_frees_claims_but_never_exclusions(self):
        """A crash between claiming and writing must not strand the paths."""
        with SeenLedger(self.tmp / "l.db", buffer_claims=False) as led:
            led.exclude(["/eval.wav"], "ar")
            _fill(Distributor("ar", led, DistributeConfig(batch_size=100)), 3)
            self.assertEqual(led.release_batch("ar", "b0000"), 3)
            self.assertFalse(led.is_claimed("/audio/a0.wav"))
            self.assertEqual(led.batch_of("/eval.wav"), EXCLUDED_BATCH)

    def test_next_index_continues_an_existing_sequence(self):
        with SeenLedger(self.tmp / "l.db", buffer_claims=False) as led:
            self.assertEqual(led.next_index("ar"), 0)
            led.claim("/a.wav", "ar", "b0000")
            led.claim("/b.wav", "ar", "b0001")
            led.exclude(["/e.wav"], "ar")
            self.assertEqual(led.next_index("ar"), 2)
            self.assertEqual(led.batch_labels("ar"), ["b0000", "b0001"])
            self.assertEqual(led.next_index("zh"), 0)

    def test_overlap_between_languages_is_zero(self):
        with SeenLedger(self.tmp / "l.db", buffer_claims=False) as led:
            led.claim("/ar.wav", "ar", "b0000")
            led.claim("/zh.wav", "zh", "b0000")
            self.assertEqual(led.overlap("ar", "zh"), 0)
            led.claim("/shared.wav", "ar", "b0001")
        # fp is the primary key, so one path cannot legitimately hold two langs;
        # simulate the audit finding by writing the second row directly.
        conn = sqlite3.connect(str(self.tmp / "l.db"))
        with conn:
            conn.execute("UPDATE seen SET lang='zh' WHERE fp='/ar.wav'")
        conn.close()
        with SeenLedger(self.tmp / "l.db", buffer_claims=False) as led:
            self.assertEqual(led.overlap("ar", "zh"), 0)

    def test_in_memory_ledger_is_viable_for_dry_runs(self):
        """``:memory:`` must work unbuffered -- build_corpus.main() relies on it
        to make ``--dry-run`` ephemeral. The non-obvious parts: Path(":memory:")
        .parent == "." so the mkdir is a harmless no-op, and PRAGMA
        journal_mode=WAL silently falls back to memory rather than raising."""
        with SeenLedger(":memory:", buffer_claims=False) as led:
            self.assertTrue(led.claim("/a.wav", "ar", "b0000"))
            self.assertFalse(led.claim("/a.wav", "ar", "b0000"))
            self.assertEqual(led.batch_of("/a.wav"), "b0000")
            self.assertEqual(led.stats()["claimed"], 1)
        self.assertFalse(Path(":memory:").exists())  # nothing landed on disk


class TestEvalExclusions(unittest.TestCase):
    """Regression cover for the most serious bug this module has had.

    ``load_eval_exclusions`` originally routed through ``read_manifest``, which
    enforces the four-key *output* contract. Real v7.6 inputs do not meet it --
    they carry only ``{audio_filepath, text, duration}``, with no ``lang`` and
    ``duration`` as a string -- so every eval file raised, was caught, logged a
    warning and contributed zero paths. The exclusion set was silently empty and
    the measured ar eval/train leak stayed wide open while looking plugged.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _write(self, rel: str, rows: list[dict], gzipped: bool = False) -> Path:
        path = self.tmp / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if gzipped else open
        with opener(path, "wt", encoding="utf-8") as fh:  # type: ignore[operator]
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    def test_real_v76_shape_is_accepted(self):
        """No ``lang`` key, string ``duration`` -- exactly what ships on disk."""
        self._write("ar/eval_ar_ae.jsonl", [
            {"audio_filepath": "/data/e1.wav", "text": "نص", "duration": "2.396"},
            {"audio_filepath": "/data/e2.wav", "text": "نص", "duration": "1.000"},
        ])
        with SeenLedger(self.tmp / "l.db", buffer_claims=False) as led:
            counts = led.load_eval_exclusions(self.tmp)
            self.assertEqual(counts, {"ar": 2})
            self.assertEqual(led.batch_of("/data/e1.wav"), EXCLUDED_BATCH)

    def test_canonical_four_key_shape_is_also_accepted(self):
        self._write("ar/eval_x.jsonl", [
            {"audio_filepath": "/data/e1.wav", "duration": 2.4, "text": "نص", "lang": "ar"},
        ])
        with SeenLedger(self.tmp / "l.db", buffer_claims=False) as led:
            self.assertEqual(led.load_eval_exclusions(self.tmp), {"ar": 1})

    def test_gzipped_eval_manifest(self):
        self._write("ml/eval_ml.jsonl.gz", [{"audio_filepath": "/m/1.wav", "duration": "1"}],
                    gzipped=True)
        with SeenLedger(self.tmp / "l.db", buffer_claims=False) as led:
            self.assertEqual(led.load_eval_exclusions(self.tmp), {"ml": 1})

    def test_train_files_are_not_treated_as_exclusions(self):
        self._write("ar/train_ar_q3asr.jsonl", [{"audio_filepath": "/t.wav", "duration": "1"}])
        self._write("ar/eval_ar.jsonl", [{"audio_filepath": "/e.wav", "duration": "1"}])
        with SeenLedger(self.tmp / "l.db", buffer_claims=False) as led:
            led.load_eval_exclusions(self.tmp)
            self.assertEqual(led.batch_of("/e.wav"), EXCLUDED_BATCH)
            self.assertIsNone(led.batch_of("/t.wav"))

    def test_malformed_lines_are_skipped_not_fatal(self):
        path = self._write("ar/eval_bad.jsonl", [{"audio_filepath": "/ok.wav", "duration": "1"}])
        with path.open("a", encoding="utf-8") as fh:
            fh.write("not json at all\n")
            fh.write("\n")
            fh.write(json.dumps(["a", "list"]) + "\n")
            fh.write(json.dumps({"text": "no path"}) + "\n")
        with SeenLedger(self.tmp / "l.db", buffer_claims=False) as led:
            self.assertEqual(led.load_eval_exclusions(self.tmp), {"ar": 1})

    def test_lang_filter_and_idempotence(self):
        self._write("ar/eval_a.jsonl", [{"audio_filepath": "/a.wav", "duration": "1"}])
        self._write("zh/eval_z.jsonl", [{"audio_filepath": "/z.wav", "duration": "1"}])
        with SeenLedger(self.tmp / "l.db", buffer_claims=False) as led:
            self.assertEqual(led.load_eval_exclusions(self.tmp, ["ar"]), {"ar": 1})
            self.assertIsNone(led.batch_of("/z.wav"))
            self.assertEqual(led.load_eval_exclusions(self.tmp), {"ar": 0, "zh": 1})

    def test_missing_root_is_not_an_error(self):
        with SeenLedger(self.tmp / "l.db", buffer_claims=False) as led:
            self.assertEqual(led.load_eval_exclusions(self.tmp / "nope"), {})

    def test_excluded_path_is_refused_by_the_distributor(self):
        self._write("ar/eval_a.jsonl", [{"audio_filepath": "/held.wav", "duration": "1"}])
        with SeenLedger(self.tmp / "l.db", buffer_claims=False) as led:
            led.load_eval_exclusions(self.tmp)
            dist = Distributor("ar", led, DistributeConfig(batch_size=10))
            self.assertIs(dist.offer(_candidate("/held.wav")), Outcome.EXCLUDED)
            self.assertEqual(dist.pending().size, 0)


class TestCandidateRanking(unittest.TestCase):
    def test_missing_quality_is_not_a_measured_zero(self):
        self.assertEqual(_candidate("/a.wav", quality=None).rank, 1.0)
        self.assertEqual(_candidate("/a.wav", quality=0.0).rank, 0.0)

    def test_erasure_damage_lowers_rank(self):
        intact = _candidate("/a.wav", quality=0.9, erasure=0.0)
        damaged = _candidate("/a.wav", quality=0.9, erasure=1.0)
        self.assertAlmostEqual(intact.rank, 0.9)
        self.assertAlmostEqual(damaged.rank, 0.4)
        self.assertGreater(intact.rank, damaged.rank)


class TestDistributorPlacement(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.ledger = SeenLedger(self.tmp / "l.db", buffer_claims=False)
        self.addCleanup(self.ledger.close)

    def _dist(self, **kw) -> Distributor:
        kw.setdefault("batch_size", 10)
        kw.setdefault("max_per_source_fraction", 1.0)
        return Distributor("ar", self.ledger, DistributeConfig(**kw))

    def test_duplicate_path_is_the_only_identity_key(self):
        """Same transcript, different path -> two samples. Same path -> one."""
        dist = self._dist()
        self.assertIs(dist.offer(_candidate("/a.wav", "نص واحد")), Outcome.ACCEPTED)
        self.assertIs(dist.offer(_candidate("/b.wav", "نص واحد")), Outcome.ACCEPTED)
        self.assertIs(dist.offer(_candidate("/a.wav", "نص مختلف تماما")), Outcome.DUP_PATH)
        self.assertEqual(dist.pending().size, 2)

    def test_duplicate_across_runs_is_still_a_duplicate(self):
        first = self._dist()
        first.offer(_candidate("/a.wav"))
        second = Distributor("ar", self.ledger, DistributeConfig(batch_size=10))
        self.assertIs(second.offer(_candidate("/a.wav")), Outcome.DUP_PATH)

    def test_max_per_text_caps_renditions_without_making_them_identity(self):
        dist = self._dist(max_per_text=2)
        outcomes = [dist.offer(_candidate(f"/r{i}.wav", "نص مكرر")) for i in range(4)]
        self.assertEqual(outcomes[:2], [Outcome.ACCEPTED, Outcome.ACCEPTED])
        self.assertEqual(outcomes[2:], [Outcome.TEXT_CAPPED, Outcome.TEXT_CAPPED])
        self.assertEqual(dist.pending().size, 2)

    def test_text_cap_spans_closed_batches(self):
        """The budget is per language across all batches, not per batch."""
        dist = self._dist(batch_size=2, max_per_text=2)
        dist.offer(_candidate("/r0.wav", "نص مكرر"))
        dist.offer(_candidate("/r1.wav", "نص مكرر"))
        self.assertEqual(len(dist.batches), 1)
        self.assertIs(dist.offer(_candidate("/r2.wav", "نص مكرر")), Outcome.TEXT_CAPPED)
        self.assertIs(dist.offer(_candidate("/r3.wav", "نص آخر")), Outcome.ACCEPTED)

    def test_per_source_cap_keeps_a_batch_mixed(self):
        dist = self._dist(batch_size=10, max_per_source_fraction=0.30)
        got = _fill(dist, 6, prefix="a", source="src_a")
        self.assertEqual(got, [Outcome.ACCEPTED] * 3 + [Outcome.SOURCE_CAPPED] * 3)
        self.assertEqual(dist.pending().sources["src_a"], 3)
        self.assertIs(dist.offer(_candidate("/audio/b0.wav", "b نص 0", "src_b")), Outcome.ACCEPTED)

    def test_batch_closes_exactly_at_batch_size(self):
        dist = self._dist(batch_size=5)
        self.assertEqual(_fill(dist, 4), [Outcome.ACCEPTED] * 4)
        self.assertEqual(len(dist.batches), 0)
        self.assertIs(dist.offer(_candidate("/audio/a4.wav", "a نص 4")), Outcome.ACCEPTED)
        self.assertEqual(len(dist.batches), 1)
        self.assertEqual(dist.batches[0].size, 5)
        self.assertEqual(dist.batches[0].label, "b0000")

    def test_filling_a_batch_rotates_straight_into_the_next_one(self):
        """Closing is automatic, so the stream never has to pause at a boundary.

        ``BATCH_FULL`` is therefore unreachable by construction -- the open batch
        is rotated the instant it fills. This pins that invariant so the enum
        member stays honest documentation rather than a state callers wait for.
        """
        dist = self._dist(batch_size=2)
        self.assertEqual(_fill(dist, 5), [Outcome.ACCEPTED] * 5)
        self.assertEqual([b.label for b in dist.batches], ["b0000", "b0001"])
        self.assertEqual(dist.pending().label, "b0002")
        self.assertEqual(dist.pending().size, 1)
        self.assertNotIn(Outcome.BATCH_FULL, dist.outcomes)

    def test_labels_are_sequential_and_index_continues_across_instances(self):
        dist = self._dist(batch_size=2)
        _fill(dist, 6)
        self.assertEqual([b.label for b in dist.batches], ["b0000", "b0001", "b0002"])
        again = Distributor("ar", self.ledger, DistributeConfig(batch_size=2))
        self.assertEqual(again.pending().label, "b0003")

    def test_short_batch_is_held_open_not_written(self):
        """A partial shard silently breaks the 100k-per-language contract."""
        dist = self._dist(batch_size=100)
        _fill(dist, 7)
        self.assertIsNone(dist.close_batch())
        self.assertEqual(len(dist.batches), 0)
        self.assertEqual(dist.pending().size, 7)

    def test_allow_partial_closes_a_short_batch_when_draining(self):
        dist = self._dist(batch_size=100)
        _fill(dist, 7)
        batch = dist.close_batch(allow_partial=True)
        self.assertIsNotNone(batch)
        self.assertEqual(batch.size, 7)
        self.assertEqual(len(dist.batches), 1)

    def test_empty_batch_returns_none_even_when_partial(self):
        dist = self._dist(batch_size=100)
        self.assertIsNone(dist.close_batch(allow_partial=True))

    def test_low_diversity_batch_is_refused_and_its_paths_released(self):
        """The zh gate. Refused must mean neither written nor consumed."""
        dist = self._dist(batch_size=4, max_per_text=4, min_distinct_text_fraction=0.90)
        for i in range(4):
            self.assertIs(dist.offer(_candidate(f"/z{i}.wav", "同一句话")), Outcome.ACCEPTED)
        self.assertEqual(len(dist.batches), 0)
        self.assertEqual(dist.refusals, 1)
        self.assertEqual(dist.outcomes[Outcome.REFUSED_LOW_DIVERSITY], 4)
        for i in range(4):
            self.assertIsNone(self.ledger.batch_of(f"/z{i}.wav"))
        self.assertEqual(dist.pending().label, "b0001")
        self.assertEqual(dist.report()["batches_refused"], 1)

    def test_refused_batch_does_not_spend_its_text_budget(self):
        dist = self._dist(batch_size=2, max_per_text=2, min_distinct_text_fraction=0.99)
        dist.offer(_candidate("/z0.wav", "نص مكرر"))
        dist.offer(_candidate("/z1.wav", "نص مكرر"))
        self.assertEqual(dist.refusals, 1)
        self.assertIs(dist.offer(_candidate("/z2.wav", "نص مكرر")), Outcome.ACCEPTED)

    def test_diverse_batch_passes_the_same_floor(self):
        dist = self._dist(batch_size=4, min_distinct_text_fraction=0.90)
        self.assertEqual(_fill(dist, 4), [Outcome.ACCEPTED] * 4)
        self.assertEqual(len(dist.batches), 1)
        self.assertTrue(dist.batches[0].diversity.acceptable)
        self.assertEqual(dist.batches[0].diversity.distinct_fraction, 1.0)

    def test_min_distinct_fraction_is_forwarded_to_the_deduper(self):
        """Regression: the config used to have no effect at all.

        ``DedupConfig`` was built without ``min_distinct_fraction``, so the
        refusal threshold silently came from dedup.py's default of 0.50 and the
        configured value was ignored.
        """
        dist = self._dist(min_distinct_text_fraction=0.77)
        self.assertEqual(dist._dedup_config().min_distinct_fraction, 0.77)
        self.assertEqual(dist._dedup_config().max_per_key, dist.config.max_per_text)

    def test_wrong_language_raises_rather_than_misfiling(self):
        dist = self._dist()
        with self.assertRaises(ValueError):
            dist.offer(_candidate("/zh.wav", "中文", lang="zh"))

    def test_batch_hours_and_source_counts(self):
        dist = self._dist(batch_size=3)
        for i in range(2):
            dist.offer(_candidate(f"/h{i}.wav", f"نص {i}", duration=3600.0))
        batch = dist.close_batch(allow_partial=True)
        self.assertAlmostEqual(batch.hours(), 2.0)
        self.assertEqual(batch.sources["src_a"], 2)


class TestDistributorWrite(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.ledger = SeenLedger(self.tmp / "l.db", buffer_claims=False)
        self.addCleanup(self.ledger.close)

    def _closed_batch(self, gzipped: bool = False) -> tuple[Distributor, Batch]:
        cfg = DistributeConfig(batch_size=3, max_per_source_fraction=1.0, gzipped=gzipped)
        dist = Distributor("ar", self.ledger, cfg)
        _fill(dist, 3)
        return dist, dist.batches[0]

    def test_write_emits_manifest_and_sidecar_with_four_keys(self):
        dist, batch = self._closed_batch()
        manifest, sidecar = dist.write(batch, self.tmp / "out")
        self.assertTrue(manifest.exists())
        self.assertTrue(sidecar.exists())
        self.assertEqual(manifest.parent.name, "ar")
        self.assertEqual(manifest.name, "train_ar_b0000_p0000.jsonl")
        self.assertEqual(sidecar.name, "train_ar_b0000_p0000.meta.jsonl")

        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertEqual(tuple(row), MANIFEST_KEYS)
            self.assertIsInstance(row["duration"], float)

    def test_sidecar_is_aligned_with_the_manifest(self):
        dist, batch = self._closed_batch()
        manifest, sidecar = dist.write(batch, self.tmp / "out")
        written = list(read_manifest(manifest))
        metas = read_sidecar(sidecar)
        self.assertEqual(len(written), 3)
        self.assertEqual({s.audio_filepath for s in written}, set(metas))
        for s in written:
            self.assertEqual(metas[s.audio_filepath].audio_filepath, s.audio_filepath)

    def test_gzipped_shards_roundtrip(self):
        dist, batch = self._closed_batch(gzipped=True)
        manifest, sidecar = dist.write(batch, self.tmp / "out")
        self.assertTrue(manifest.name.endswith(".jsonl.gz"))
        self.assertTrue(sidecar.name.endswith(".meta.jsonl.gz"))
        # gzip.open raises BadGzipFile on a plain file, so reading BOTH through
        # it is what proves the suffix was honored. The earlier bug named shards
        # .jsonl.gz yet wrote them uncompressed via a .tmp temp file.
        with gzip.open(manifest, "rt", encoding="utf-8") as fh:
            self.assertEqual(len(fh.readlines()), 3)
        with gzip.open(sidecar, "rt", encoding="utf-8") as fh:
            self.assertEqual(len(fh.readlines()), 3)
        self.assertEqual(len(list(read_manifest(manifest))), 3)

    def test_existing_paths_reads_back_what_was_written(self):
        dist, batch = self._closed_batch()
        dist.write(batch, self.tmp / "out")
        paths = existing_paths(self.tmp / "out", "ar")
        self.assertEqual(paths, {f"/audio/a{i}.wav" for i in range(3)})

    def test_written_shard_ledger_and_batch_all_agree(self):
        """Three views of one batch: the shard, the ledger, the in-memory batch."""
        dist, batch = self._closed_batch()
        dist.write(batch, self.tmp / "out")
        on_disk = existing_paths(self.tmp / "out", "ar")
        in_batch = {s.audio_filepath for s in batch.samples}
        in_ledger = {fp for fp in in_batch if self.ledger.batch_of(fp) == "b0000"}
        self.assertEqual(on_disk, in_batch)
        self.assertEqual(in_ledger, in_batch)

    def test_report_shape(self):
        dist, _ = self._closed_batch()
        report = dist.report()
        self.assertEqual(report["lang"], "ar")
        self.assertEqual(report["batches_closed"], 1)
        self.assertEqual(report["batch_labels"], ["b0000"])
        self.assertEqual(report["outcomes"][Outcome.ACCEPTED.value], 3)
        self.assertIn("ledger", report)


class TestIterCandidates(unittest.TestCase):
    def test_synthesized_meta_leaves_quality_unmeasured(self):
        samples = [_sample("/a.wav"), _sample("/b.wav")]
        cands = list(iter_candidates(samples, source="internal_v76_ar"))
        self.assertEqual(len(cands), 2)
        self.assertIsNone(cands[0].meta.quality)
        self.assertEqual(cands[0].meta.dataset, "internal_v76_ar")
        self.assertEqual(cands[0].source, "internal_v76_ar")

    def test_supplied_metas_zip_in_order(self):
        samples = [_sample("/a.wav"), _sample("/b.wav")]
        metas = [Meta(audio_filepath="/a.wav", quality=0.2), Meta(audio_filepath="/b.wav", quality=0.9)]
        cands = list(iter_candidates(samples, metas, "s"))
        self.assertEqual([c.meta.quality for c in cands], [0.2, 0.9])


if __name__ == "__main__":
    unittest.main()
