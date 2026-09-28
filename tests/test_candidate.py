"""Tests for the Stage-1 candidate pool: serialization, sorted I/O, best-first merge."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from data_processing.candidate import (
    PoolRecord,
    interleave_pools,
    iter_pool_shards,
    merge_pools,
    pool_shard_path,
    read_pool,
    write_pool,
)


def _rec(fp: str, source: str, composite: float, **kw) -> PoolRecord:
    kw.setdefault("lang", "ar")
    kw.setdefault("normalized_text", f"text for {fp}")
    return PoolRecord(audio_filepath=fp, source=source, composite=composite, **kw)


class _PoolDir(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)


class TestPoolRecordSerialization(_PoolDir):
    def test_dict_round_trip_is_stable(self):
        rec = _rec("/a/1.wav", "srcA", 0.42, quality=0.7, richness=0.6,
                   accent="egyptian", needs_llm=True, itn_applied=True,
                   stages={"normalize": "western_digits"}, metrics={"chars": 30.0})
        self.assertEqual(PoolRecord.from_dict(rec.to_dict()).to_dict(), rec.to_dict())

    def test_zero_valued_scores_survive_serialization(self):
        """``0.0 == False`` in Python; a falsy-drop rule would lose a real zero."""
        rec = _rec("/a/0.wav", "srcA", 0.0, quality=0.0, richness=0.0)
        back = PoolRecord.from_dict(rec.to_dict())
        self.assertEqual(back.quality, 0.0)
        self.assertEqual(back.composite, 0.0)

    def test_external_fields_round_trip(self):
        rec = _rec("/audio/ar/srcB/deadbeef.flac", "srcB", 0.5,
                   external=True, native_id="http://hub/clip_1.flac", duration=None)
        back = PoolRecord.from_dict(rec.to_dict())
        self.assertTrue(back.external)
        self.assertEqual(back.native_id, "http://hub/clip_1.flac")
        self.assertIsNone(back.duration)

    def test_unknown_keys_are_ignored_on_read(self):
        raw = _rec("/a/1.wav", "srcA", 0.5).to_dict()
        raw["a_field_from_the_future"] = 1
        self.assertEqual(PoolRecord.from_dict(raw).audio_filepath, "/a/1.wav")

    def test_source_text_omitted_when_it_equals_normalized(self):
        rec = _rec("/a/1.wav", "srcA", 0.5)
        rec.source_text = rec.normalized_text
        self.assertNotIn("source_text", rec.to_dict())
        rec.source_text = "something else"
        self.assertIn("source_text", rec.to_dict())

    def test_to_meta_carries_provenance_and_richness(self):
        rec = _rec("/a/1.wav", "srcA", 0.5, quality=0.8, richness=0.6,
                   dataset="repo/x", accent="msa")
        meta = rec.to_meta()
        self.assertEqual(meta.dataset, "repo/x")
        self.assertEqual(meta.accent, "msa")
        self.assertEqual(meta.quality, 0.8)
        self.assertEqual(meta.metrics["richness"], 0.6)


class TestWriteReadPool(_PoolDir):
    def test_write_sorts_by_composite_descending(self):
        path = pool_shard_path(self.dir, "ar", "srcA", 0)
        recs = [_rec(f"/a/{i}.wav", "srcA", c) for i, c in enumerate([0.2, 0.9, 0.5])]
        self.assertEqual(write_pool(path, recs), 3)
        got = [round(r.composite, 3) for r in read_pool(path)]
        self.assertEqual(got, [0.9, 0.5, 0.2])

    def test_sort_false_preserves_input_order(self):
        path = pool_shard_path(self.dir, "ar", "srcA", 0)
        recs = [_rec(f"/a/{i}.wav", "srcA", c) for i, c in enumerate([0.2, 0.9, 0.5])]
        write_pool(path, recs, sort=False)
        self.assertEqual([r.composite for r in read_pool(path)], [0.2, 0.9, 0.5])

    def test_gzipped_shard_round_trips(self):
        path = pool_shard_path(self.dir, "ar", "srcA", 0, gzipped=True)
        self.assertTrue(str(path).endswith(".jsonl.gz"))
        write_pool(path, [_rec("/a/1.wav", "srcA", 0.5)])
        self.assertEqual(len(list(read_pool(path))), 1)

    def test_empty_shard_reads_back_empty(self):
        path = pool_shard_path(self.dir, "ar", "srcA", 0)
        write_pool(path, [])
        self.assertEqual(list(read_pool(path)), [])


class TestShardDiscovery(_PoolDir):
    def test_shard_path_layout(self):
        p = pool_shard_path("/pools", "en", "librispeech", 7)
        self.assertEqual(p, Path("/pools/en/librispeech/part-00007.jsonl"))

    def test_iter_finds_all_sources_for_a_language_sorted(self):
        for src in ("srcB", "srcA"):
            write_pool(pool_shard_path(self.dir, "ar", src, 0), [_rec(f"/{src}/1", src, 0.5)])
        shards = iter_pool_shards(self.dir, "ar")
        self.assertEqual(len(shards), 2)
        self.assertEqual(shards, sorted(shards))

    def test_iter_can_filter_to_one_source(self):
        for src in ("srcA", "srcB"):
            write_pool(pool_shard_path(self.dir, "ar", src, 0), [_rec(f"/{src}/1", src, 0.5)])
        self.assertEqual(len(iter_pool_shards(self.dir, "ar", "srcA")), 1)

    def test_missing_language_dir_is_empty_not_an_error(self):
        self.assertEqual(iter_pool_shards(self.dir, "zz"), [])


class TestMergeAndInterleave(_PoolDir):
    def _two_sources(self):
        a = pool_shard_path(self.dir, "ar", "srcA", 0)
        b = pool_shard_path(self.dir, "ar", "srcB", 0)
        write_pool(a, [_rec("/a/1", "srcA", 0.45), _rec("/a/2", "srcA", 0.10)])
        write_pool(b, [_rec("/b/1", "srcB", 0.48), _rec("/b/2", "srcB", 0.30)])
        return iter_pool_shards(self.dir, "ar")

    def test_merge_is_globally_best_first_across_sources(self):
        shards = self._two_sources()
        self.assertEqual([round(r.composite, 3) for r in merge_pools(shards)],
                         [0.48, 0.45, 0.30, 0.10])

    def test_merge_of_nothing_is_empty(self):
        self.assertEqual(list(merge_pools([])), [])

    def test_interleave_round_robins_sources(self):
        shards = self._two_sources()
        sources = [r.source for r in interleave_pools(shards)]
        self.assertEqual(sources, ["srcA", "srcB", "srcA", "srcB"])

    def test_merge_preserves_every_record(self):
        shards = self._two_sources()
        self.assertEqual(len(list(merge_pools(shards))), 4)


if __name__ == "__main__":
    unittest.main()
