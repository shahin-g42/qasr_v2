"""Tests for Stage-2 assemble: best-first selection, inline materialize, batching.

External audio is exercised through an injected fetcher backed by real fixture
files, so the whole materialize -> deferred-gate -> place path runs offline. The
LLM path is driven by a scripted ``BatchCorrector._post``, never a network.
"""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from data_processing.assemble import _apply_llm, in_pool_part, main, run_assemble
from data_processing.build_corpus import BatchCorrector
from data_processing.candidate import PoolRecord, pool_shard_path, write_pool
from data_processing.canonical import MANIFEST_KEYS, iter_shards, read_manifest, read_sidecar
from data_processing.datasets.base import derive_materialized_path
from data_processing.distribute import DistributeConfig
from data_processing.quality import QualityConfig

try:
    import numpy as np
    import soundfile as sf

    from data_processing import materialize as M
except ImportError:  # pragma: no cover - environment dependent
    np = None

_AR = "سبحان الله كل شي تطور دفعة واحدة في كل المجالات الحديثة"


def _parse_user_batch(content: str) -> list[tuple[int, str]]:
    """The items a corrector request asked about: ``(chunk position, text)``.

    Handles both prompt families so the scripted corrector works regardless of
    ``--llm-prompt``: the compact JSON list (0-based ``i``), or the rich
    numbered ``N. <<<text>>>`` lines (1-based for the Arabic template, 0-based
    for the generic one -- told apart by a ``0.`` line). Replies always use the
    0-based ``i`` contract, which :func:`build_corpus._result_index` accepts
    under either mode.
    """
    try:
        rows = json.loads(content)
    except json.JSONDecodeError:
        pairs = [(int(m.group(1)), m.group(2)) for m in
                 re.finditer(r"^\s*(\d+)\. <<<(.*)>>>$", content, re.MULTILINE)]
        if any(n == 0 for n, _ in pairs):
            return pairs
        return [(n - 1, t) for n, t in pairs]
    return [(int(r["i"]), r["text"]) for r in rows]


class _Tmp(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.pool = self.dir / "pool"
        self.out = self.dir / "out"
        self.audio_root = self.dir / "audio"
        self.ledger = self.dir / "ledger.sqlite3"
        self.addCleanup(self._tmp.cleanup)

    # --- pool construction --------------------------------------------------
    def put_pool(self, lang, source, recs):
        write_pool(pool_shard_path(self.pool, lang, source, 0), recs)

    def local_rec(self, i, composite, *, lang="ar", source="internal_v76_ar", dur=4.0):
        return PoolRecord(
            audio_filepath=f"/data/{source}/a{i}.wav", source=source, lang=lang,
            normalized_text=f"{_AR} رقم {i}", quality=composite, richness=1.0,
            composite=composite, external=False, duration=dur, dataset=source,
        )

    def assemble(self, lang="ar", **kw):
        kw.setdefault("pool_dir", self.pool)
        kw.setdefault("out_dir", self.out)
        kw.setdefault("ledger", self.ledger)
        kw.setdefault("audio_root", self.audio_root)
        kw.setdefault("distribute", DistributeConfig(batch_size=5, max_per_source_fraction=1.0))
        kw.setdefault("batches", 1)
        return run_assemble(lang=lang, **kw)

    def written(self, lang="ar"):
        out = []
        for manifest, _ in iter_shards(self.out / lang, lang=lang):
            out.extend(read_manifest(manifest))
        return out


class TestNoPool(_Tmp):
    def test_missing_pool_reports_error_not_crash(self):
        report = self.assemble()
        self.assertEqual(report["error"], "no pool shards")
        self.assertEqual(self.written(), [])


class TestLocalAssemble(_Tmp):
    def test_emits_one_full_batch_with_canonical_manifest(self):
        self.put_pool("ar", "internal_v76_ar", [self.local_rec(i, 0.5) for i in range(8)])
        report = self.assemble(materialize_external=False)
        self.assertEqual(report["batches_written"], 1)
        self.assertEqual(report["batch_labels"], ["b0000"])
        samples = self.written()
        self.assertEqual(len(samples), 5)  # batch_size
        self.assertEqual(report["batch_hours"]["b0000"], round(5 * 4.0 / 3600, 2))
        # canonical 4-key contract
        manifest = next(iter_shards(self.out / "ar", lang="ar"))[0]
        first = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(list(first.keys()), list(MANIFEST_KEYS))

    def test_best_first_selects_the_highest_composite(self):
        recs = [self.local_rec(i, c) for i, c in enumerate([0.1, 0.9, 0.5, 0.7, 0.3, 0.2])]
        self.put_pool("ar", "internal_v76_ar", recs)
        self.assemble(materialize_external=False)
        # Top-5 by composite: 0.9, 0.7, 0.5, 0.3, 0.2 -> a1, a3, a2, a4, a5
        got = {s.audio_filepath for s in self.written()}
        self.assertEqual(got, {f"/data/internal_v76_ar/a{i}.wav" for i in (1, 3, 2, 4, 5)})
        self.assertNotIn("/data/internal_v76_ar/a0.wav", got)  # 0.1 dropped

    def test_short_tail_is_held_open_not_written(self):
        self.put_pool("ar", "internal_v76_ar", [self.local_rec(i, 0.5) for i in range(3)])
        report = self.assemble(materialize_external=False)
        self.assertEqual(report["batches_written"], 0)
        self.assertEqual(report["carried_over_partial"], 0)  # released, not stranded
        self.assertEqual(report["counters"].get("released_pending"), 3)
        self.assertEqual(self.written(), [])

    def test_a_second_eval_root_excludes_its_paths_too(self):
        """The q3asr SFT tree's eval files reach the exclusion set via
        ``--eval-root``: a path leaked from THERE cannot ship, exactly like
        one leaked from a v7.6 eval file."""
        self.put_pool("ar", "internal_v76_ar", [self.local_rec(i, 0.5) for i in range(8)])
        sft = self.dir / "q3asr_sft_manifests"
        (sft / "ar").mkdir(parents=True)
        (sft / "ar" / "eval_ar_q3asr.jsonl").write_text(
            json.dumps({"audio_filepath": "/data/internal_v76_ar/a1.wav"}) + "\n",
            encoding="utf-8")
        report = self.assemble(materialize_external=False, exclude_eval=True,
                               root=None, eval_roots=[sft])
        self.assertEqual(report["batches_written"], 1)
        got = {s.audio_filepath for s in self.written()}
        self.assertNotIn("/data/internal_v76_ar/a1.wav", got)
        self.assertEqual(len(got), 5)  # the next-best record took its slot

    def test_drain_writes_partial_tail(self):
        self.put_pool("ar", "internal_v76_ar", [self.local_rec(i, 0.5) for i in range(3)])
        report = self.assemble(materialize_external=False, batches=0)
        self.assertEqual(report["batches_written"], 1)
        self.assertEqual(len(self.written()), 3)

    def test_source_cap_keeps_a_batch_mixed(self):
        dist = DistributeConfig(batch_size=10, max_per_source_fraction=0.4)
        for name, comp in (("srcA", 0.9), ("srcB", 0.8), ("srcC", 0.7)):
            self.put_pool("ar", name,
                          [self.local_rec(i, comp, source=name) for i in range(10)])
        self.assemble(materialize_external=False, distribute=dist)
        samples = self.written()
        self.assertEqual(len(samples), 10)
        per = Counter(s.audio_filepath.split("/")[2] for s in samples)
        self.assertLessEqual(per["srcA"], 4)  # int(10 * 0.4)
        self.assertLessEqual(per["srcB"], 4)
        self.assertLessEqual(per["srcC"], 4)
        self.assertEqual(len(per), 3)  # all three sources present

    def test_batch_index_continues_across_runs(self):
        self.put_pool("ar", "internal_v76_ar", [self.local_rec(i, 0.5) for i in range(12)])
        r1 = self.assemble(materialize_external=False, batches=1)
        self.assertEqual(r1["batch_labels"], ["b0000"])
        r2 = self.assemble(materialize_external=False, batches=1)
        self.assertEqual(r2["batch_labels"], ["b0001"])

    def test_dry_run_writes_nothing(self):
        self.put_pool("ar", "internal_v76_ar", [self.local_rec(i, 0.5) for i in range(6)])
        report = self.assemble(materialize_external=False, dry_run=True)
        self.assertTrue(report["dry_run"])
        self.assertEqual(report["files"], [])
        self.assertFalse(self.out.exists() and any(self.out.rglob("*.jsonl")))

    def test_sidecar_carries_richness_and_provenance(self):
        self.put_pool("ar", "internal_v76_ar", [self.local_rec(i, 0.5) for i in range(5)])
        self.assemble(materialize_external=False)
        _manifest, sidecar = next(iter_shards(self.out / "ar", lang="ar"))
        metas = read_sidecar(sidecar)
        self.assertEqual(len(metas), 5)
        for meta in metas.values():
            self.assertIn("richness", meta.metrics)
            self.assertEqual(meta.stages["llm"], "disabled")
            self.assertIsNotNone(meta.final_text)


class TestInPoolPart(unittest.TestCase):
    def test_single_slice_takes_everything(self):
        self.assertTrue(in_pool_part("/data/a.wav", 0, 1))

    def test_partition_is_total_disjoint_and_stable(self):
        paths = [f"/data/clips/a{i}.wav" for i in range(500)]
        for count in (2, 4, 5, 8):
            for path in paths:
                owners = [k for k in range(count) if in_pool_part(path, k, count)]
                self.assertEqual(len(owners), 1)  # exactly one slice owns a path
            for k in range(count):  # every slice is non-trivial at this size
                self.assertGreater(sum(in_pool_part(p, k, count) for p in paths), 0)
        self.assertEqual(in_pool_part("/x.wav", 3, 4), in_pool_part("/x.wav", 3, 4))


class TestPoolSlicing(_Tmp):
    """Multi-node stage 2: hash slices compose one batch, part by part.

    The pool mirrors the real layout: every clip appears twice, once from the
    v7.6 pool and once from the q3asr SFT pool, under the SAME
    ``audio_filepath``. Keeping that duplicate pair inside one part is exactly
    what the keyed-by-path hash buys.
    """

    N = 32  # unique clips; each of the two sources carries all of them

    def source_records(self, source, quality):
        return [
            PoolRecord(
                audio_filepath=f"/data/clips/a{i}.wav", source=source, lang="ar",
                normalized_text=f"{_AR} رقم {i}", quality=quality, richness=1.0,
                composite=quality, external=False, duration=4.0, dataset=source,
            )
            for i in range(self.N)
        ]

    def build_pool(self):
        self.put_pool("ar", "internal_v76_ar", self.source_records("internal_v76_ar", 0.6))
        self.put_pool("ar", "internal_sft_ar", self.source_records("internal_sft_ar", 0.5))

    def test_two_slices_compose_one_batch_without_crossing_duplicates(self):
        self.build_pool()
        dist = DistributeConfig(batch_size=64, max_per_source_fraction=1.0)
        self.assemble(materialize_external=False, batches=0, distribute=dist,
                      ledger=self.dir / "base.sqlite3", out_dir=self.dir / "base_out")
        base_rows = [s for m, _ in iter_shards(self.dir / "base_out" / "ar", lang="ar")
                     for s in read_manifest(m)]
        want = {s.audio_filepath for s in base_rows}
        self.assertEqual(len(want), self.N)

        for part in (0, 1):
            report = self.assemble(
                materialize_external=False, batches=0, distribute=dist,
                ledger=self.dir / f"slice{part}.sqlite3",
                pool_part=(part, 2), batch_part=part)
            self.assertEqual(report["pool_part"], [part, 2])
            self.assertEqual(report["batch_part"], part)
            self.assertEqual(report["batch_labels"], ["b0000"])
            self.assertTrue(report["files"][0][0].endswith(f"train_ar_b0000_p{part:04d}.jsonl"))

        rows = self.written()
        self.assertEqual(len(rows), self.N)
        self.assertEqual({s.audio_filepath for s in rows}, want)
        self.assertEqual(len({s.audio_filepath for s in rows}), len(rows))  # no cross-part dup
        for manifest, _ in iter_shards(self.out / "ar", lang="ar"):
            self.assertGreater(len(list(read_manifest(manifest))), 0)

    def test_slice_params_are_validated(self):
        with self.assertRaises(ValueError):
            self.assemble(materialize_external=False, pool_part=(2, 2))
        with self.assertRaises(ValueError):
            # batch_size 5 is not divisible by 3: parts would not compose a batch
            self.assemble(materialize_external=False, pool_part=(0, 3))
        with self.assertRaises(ValueError):
            dist = DistributeConfig(batch_size=64, max_per_source_fraction=1.0)
            self.assemble(materialize_external=False, distribute=dist,
                          pool_part=(0, 2), batch_part=2)

    def test_cli_rejects_malformed_pool_part(self):
        base = ["--lang", "ar", "--pool-dir", str(self.pool), "--out-dir", str(self.out),
                "--ledger", str(self.ledger), "--audio-root", str(self.audio_root)]
        for bad in ("abc", "1/0", "2/2", "-1/4"):
            with self.assertRaises(SystemExit):
                main([*base, "--pool-part", bad])


@unittest.skipIf(np is None, "audio dependencies (numpy/scipy/soundfile) are not installed")
class TestExternalMaterialize(_Tmp):
    def _fixture(self, name, seconds=2.0, sr=8_000):
        path = self.dir / "fixtures" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
        sf.write(str(path), (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr)
        return str(path)

    def ext_rec(self, i, composite, native, *, lang="ar", source="fake_ext"):
        fp = derive_materialized_path(self.audio_root, lang, source, native)
        return PoolRecord(
            audio_filepath=fp, source=source, lang=lang, normalized_text=f"{_AR} رقم {i}",
            quality=composite, richness=1.0, composite=composite, external=True,
            native_id=native, duration=None, dataset=source,
        )

    def test_external_clip_is_materialized_to_16k_mono_and_placed(self):
        natives = [self._fixture(f"c{i}.wav") for i in range(5)]
        recs = [self.ext_rec(i, 0.5, natives[i]) for i in range(5)]
        self.put_pool("ar", "fake_ext", recs)
        report = self.assemble(fetcher=M.local_file_fetcher)
        self.assertEqual(report["batches_written"], 1)
        self.assertEqual(report["counters"]["materialized"], 5)
        samples = self.written()
        self.assertEqual(len(samples), 5)
        for s in samples:
            self.assertTrue(s.audio_filepath.startswith(str(self.audio_root)))
            self.assertEqual(M.probe(s.audio_filepath), (round(s.duration, 3), 16_000, 1))
            self.assertAlmostEqual(s.duration, 2.0, places=2)

    def test_out_of_band_clip_is_placed_when_duration_gates_off(self):
        # gate_duration defaults to False, so the deferred gate no longer drops a
        # 0.1s clip -- it is materialized and placed like any other.
        short = self._fixture("short.wav", seconds=0.1)
        good = [self._fixture(f"g{i}.wav") for i in range(4)]
        recs = [self.ext_rec(i, 0.5, good[i]) for i in range(4)]
        recs.append(self.ext_rec(99, 0.9, short))
        self.put_pool("ar", "fake_ext", recs)
        report = self.assemble(fetcher=M.local_file_fetcher)
        self.assertEqual(report["counters"]["materialized"], 5)
        self.assertNotIn("reject_deferred_gate", report["counters"])
        paths = {s.audio_filepath for s in self.written()}
        self.assertIn(derive_materialized_path(self.audio_root, "ar", "fake_ext", short), paths)
        self.assertEqual(len(paths), 5)

    def test_deferred_gate_drops_out_of_band_clip_when_armed(self):
        short = self._fixture("short.wav", seconds=0.1)  # 0.1s < min_duration 0.5
        good = [self._fixture(f"g{i}.wav") for i in range(5)]
        recs = [self.ext_rec(i, 0.9, good[i]) for i in range(5)]
        recs.append(self.ext_rec(99, 0.99, short))  # highest composite, but too short
        self.put_pool("ar", "fake_ext", recs)
        report = self.assemble(fetcher=M.local_file_fetcher,
                               quality=QualityConfig(gate_duration=True))
        self.assertEqual(report["counters"]["reject_deferred_gate"], 1)
        paths = {s.audio_filepath for s in self.written()}
        self.assertNotIn(derive_materialized_path(self.audio_root, "ar", "fake_ext", short), paths)
        self.assertEqual(len(paths), 5)

    def test_materialize_failure_is_isolated(self):
        good = [self._fixture(f"g{i}.wav") for i in range(5)]
        recs = [self.ext_rec(i, 0.5, good[i]) for i in range(5)]
        recs.append(self.ext_rec(50, 0.99, "/does/not/exist.wav"))
        self.put_pool("ar", "fake_ext", recs)
        report = self.assemble(fetcher=M.local_file_fetcher)
        self.assertEqual(report["counters"]["materialize_failed"], 1)
        self.assertEqual(report["batches_written"], 1)
        self.assertEqual(len(self.written()), 5)

    def test_prefetch_workers_submit_and_place_external_clips(self):
        natives = [self._fixture(f"p{i}.wav") for i in range(5)]
        recs = [self.ext_rec(i, 0.5, natives[i]) for i in range(5)]
        self.put_pool("ar", "fake_ext", recs)
        report = self.assemble(fetcher=M.local_file_fetcher, fetch_workers=4)
        # Every external candidate was submitted while the chunk built, then
        # awaited at placement: the batch is written with true durations.
        self.assertEqual(report["counters"]["prefetched"], 5)
        self.assertEqual(report["counters"]["materialized"], 5)
        self.assertEqual(report["fetch_workers"], 4)
        self.assertEqual(report["batches_written"], 1)
        samples = self.written()
        self.assertEqual(len(samples), 5)
        for s in samples:
            self.assertEqual(M.probe(s.audio_filepath), (round(s.duration, 3), 16_000, 1))

    def test_fetch_workers_zero_falls_back_to_serial_inline_fetch(self):
        natives = [self._fixture(f"z{i}.wav") for i in range(5)]
        recs = [self.ext_rec(i, 0.5, natives[i]) for i in range(5)]
        self.put_pool("ar", "fake_ext", recs)
        report = self.assemble(fetcher=M.local_file_fetcher, fetch_workers=0)
        self.assertNotIn("prefetched", report["counters"])
        self.assertEqual(report["counters"]["materialized"], 5)
        self.assertEqual(len(self.written()), 5)

    def test_unmaterialized_external_is_skipped_when_disabled(self):
        natives = [self._fixture(f"c{i}.wav") for i in range(5)]
        self.put_pool("ar", "fake_ext", [self.ext_rec(i, 0.5, natives[i]) for i in range(5)])
        report = self.assemble(materialize_external=False)
        self.assertEqual(report["counters"]["skip_external_unmaterialized"], 5)
        self.assertEqual(self.written(), [])


class TestLLMCorrection(_Tmp):
    def _fake_post(self, payload):
        got = _parse_user_batch(payload["messages"][1]["content"])
        out = {"items": [{"i": n, "text": t + ".", "dialect": None, "confidence": 0.9}
                         for n, t in got]}
        return {"choices": [{"message": {"content": json.dumps(out, ensure_ascii=False)}}]}

    def test_corrector_runs_and_flows_into_the_manifest(self):
        recs = [self.local_rec(i, 0.5) for i in range(5)]
        for r in recs:
            r.needs_llm = True
        self.put_pool("ar", "internal_v76_ar", recs)
        with mock.patch.object(BatchCorrector, "_post", self._fake_post):
            report = self.assemble(materialize_external=False, llm_url="http://localhost:8010/v1")
        self.assertGreaterEqual(report["llm"]["calls"], 1)
        self.assertGreaterEqual(report["llm"]["samples_corrected"], 1)
        self.assertEqual(report["counters"].get("erasure_reverted", 0), 0)
        for s in self.written():
            self.assertTrue(s.text.endswith("."))


class TestApplyLLMUnit(unittest.TestCase):
    def test_empty_response_keeps_normalized(self):
        from data_processing.canonical import Meta

        meta = Meta(audio_filepath="/a.wav", normalized_text="original")
        counters = Counter()
        _apply_llm(meta, {"text": "   "}, "ar", counters)
        self.assertEqual(meta.final_text, "original")
        self.assertEqual(meta.stages["llm"], "empty_response")


if __name__ == "__main__":
    unittest.main()
