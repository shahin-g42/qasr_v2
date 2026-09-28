"""Tests for Stage-1 prepare: source->node assignment, gating, pool emission.

External sources are exercised through ``DatasetSpec.loader`` injection, which
``stream_metadata`` honors before it ever reaches the Hub -- so these tests cover
the external path (deterministic materialized identity, deferred duration) with no
network and no ``datasets`` install. Local duration probing is patched rather than
backed by real audio, keeping the suite fast and dependency-light.
"""

from __future__ import annotations

import json
import multiprocessing
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from data_processing.candidate import iter_pool_shards, read_pool
from data_processing.datasets import registry
from data_processing.datasets.base import (
    DatasetSpec,
    FieldMap,
    Kind,
    derive_materialized_path,
)
from data_processing.normalize import DiacriticPolicy
from data_processing.prepare import (
    _clear_source_pool,
    _row_to_record,
    assign_sources,
    prepare_source,
    run_prepare,
)
from data_processing.quality import QualityConfig

_AR_RICH = "سبحان الله كل شي تطور دفعة واحدة في كل المجالات الحديثة"
_AR_TRIVIAL = "نعم"


class _Tmp(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.pool = self.dir / "pool"
        self.audio_root = self.dir / "audio"
        self.addCleanup(self._tmp.cleanup)


def _local_spec(name: str = "internal_v76_ar", lang: str = "ar", **kw) -> DatasetSpec:
    kw.setdefault("paths", (f"{lang}/*.jsonl*",))
    kw.setdefault("exclude", ("_still_rejected_", "eval_"))
    return DatasetSpec(name=name, lang=lang, kind=Kind.LOCAL_JSONL, license="internal",
                       fields=FieldMap(), **kw)


def _write_jsonl(root: Path, rel: str, rows: list[dict]) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _ext_spec(rows: list[dict], *, name="fake_ext", lang="ar", duration_field=None) -> DatasetSpec:
    """An external (HF_STREAM) spec whose stream is a loader, not the network."""
    return DatasetSpec(
        name=name, lang=lang, kind=Kind.HF_STREAM, license="test", repo_id="me/fake",
        fields=FieldMap(path="path", text="text", duration=duration_field),
        loader=lambda spec: iter(rows),
    )


def _collect(pool_dir, lang, source):
    recs = []
    for shard in iter_pool_shards(pool_dir, lang, source):
        recs.extend(read_pool(shard))
    return recs


class TestAssignSources(unittest.TestCase):
    def test_single_node_gets_everything(self):
        full = assign_sources(("ar",), num_nodes=1)
        self.assertTrue(full)
        self.assertEqual(len(assign_sources(("ar",), node_rank=0, num_nodes=1)), len(full))

    def test_ranks_partition_disjointly_and_completely(self):
        full = {(lang, s.name) for lang, s in assign_sources(("ar", "zh"), num_nodes=1)}
        parts = [
            {(lang, s.name) for lang, s in assign_sources(("ar", "zh"), node_rank=r, num_nodes=3)}
            for r in range(3)
        ]
        union = set().union(*parts)
        self.assertEqual(union, full)
        self.assertEqual(sum(len(p) for p in parts), len(full))  # disjoint
        for a in range(3):
            for b in range(a + 1, 3):
                self.assertFalse(parts[a] & parts[b])

    def test_assignment_is_deterministic(self):
        a = [s.name for _, s in assign_sources(("ar",), node_rank=1, num_nodes=4)]
        b = [s.name for _, s in assign_sources(("ar",), node_rank=1, num_nodes=4)]
        self.assertEqual(a, b)

    def test_only_sources_filters(self):
        specs = assign_sources(("ar",), num_nodes=1)
        if not specs:
            self.skipTest("no ar specs in registry")
        target = specs[0][1].name
        got = assign_sources(("ar",), only_sources=(target,), num_nodes=1)
        self.assertEqual([s.name for _, s in got], [target])


class TestPrepareLocalSource(_Tmp):
    def _run(self, rows, *, quality=None, shard_size=100_000, limit=None, spec=None):
        root = self.dir / "v7.6"
        _write_jsonl(root, "ar/train_0.jsonl", rows)
        spec = spec or _local_spec()
        return prepare_source(
            spec, pool_dir=self.pool, audio_root=self.audio_root, root=str(root),
            quality=quality or QualityConfig(), shard_size=shard_size, limit=limit,
        )

    def test_keeps_local_path_verbatim_and_records_duration(self):
        rows = [{"audio_filepath": f"/data/a{i}.wav", "text": f"{_AR_RICH} {i}",
                 "duration": "4.000"} for i in range(3)]
        report = self._run(rows)
        self.assertEqual(report["emitted"], 3)
        recs = _collect(self.pool, "ar", "internal_v76_ar")
        self.assertEqual(len(recs), 3)
        for r in recs:
            self.assertTrue(r.audio_filepath.startswith("/data/a"))
            self.assertFalse(r.external)
            self.assertEqual(r.native_id, "")
            self.assertEqual(r.duration, 4.0)

    def test_shards_are_sorted_by_composite_descending(self):
        rows = [{"audio_filepath": f"/data/a{i}.wav", "text": f"{_AR_RICH} {i}",
                 "duration": "4.000"} for i in range(5)]
        self._run(rows, shard_size=2)
        shards = iter_pool_shards(self.pool, "ar", "internal_v76_ar")
        self.assertEqual(len(shards), 3)  # 2 + 2 + 1
        for shard in shards:
            comps = [r.composite for r in read_pool(shard)]
            self.assertEqual(comps, sorted(comps, reverse=True))
        self.assertEqual(len(_collect(self.pool, "ar", "internal_v76_ar")), 5)

    def test_empty_text_rows_are_skipped(self):
        rows = [{"audio_filepath": "/data/a0.wav", "text": "   ", "duration": "4.0"},
                {"audio_filepath": "/data/a1.wav", "text": _AR_RICH, "duration": "4.0"}]
        report = self._run(rows)
        self.assertEqual(report["emitted"], 1)
        self.assertEqual(report["counters"]["skip_no_text"], 1)

    def test_out_of_band_duration_is_kept_when_gates_off(self):
        # gate_duration defaults to False, so a 999s clip is measured, not rejected.
        rows = [{"audio_filepath": "/data/a0.wav", "text": _AR_RICH, "duration": "999.0"}]
        report = self._run(rows)
        self.assertEqual(report["emitted"], 1)
        self.assertNotIn("reject_gate", report["counters"])
        recs = _collect(self.pool, "ar", "internal_v76_ar")
        self.assertEqual(recs[0].duration, 999.0)

    def test_duration_band_rejects_when_gating_armed(self):
        rows = [{"audio_filepath": "/data/a0.wav", "text": _AR_RICH, "duration": "999.0"}]
        report = self._run(rows, quality=QualityConfig(gate_duration=True))
        self.assertEqual(report["emitted"], 0)
        self.assertEqual(report["counters"]["reject_gate"], 1)
        self.assertIn("reason:duration_out_of_band", report["counters"])

    def test_min_richness_drops_trivial_transcripts(self):
        rows = [{"audio_filepath": "/data/triv.wav", "text": _AR_TRIVIAL, "duration": "4.0"},
                {"audio_filepath": "/data/rich.wav", "text": _AR_RICH, "duration": "4.0"}]
        report = self._run(rows, quality=QualityConfig(min_richness=0.35))
        recs = _collect(self.pool, "ar", "internal_v76_ar")
        self.assertEqual(len(recs), 1)
        self.assertTrue(recs[0].audio_filepath.endswith("rich.wav"))
        self.assertIn("reason:low_richness", report["counters"])

    def test_limit_caps_rows_read(self):
        rows = [{"audio_filepath": f"/data/a{i}.wav", "text": f"{_AR_RICH} {i}",
                 "duration": "4.0"} for i in range(10)]
        report = self._run(rows, limit=3)
        self.assertEqual(report["read"], 3)
        self.assertEqual(report["emitted"], 3)
        self.assertEqual(report["counters"]["limit_reached"], 1)

    def test_missing_local_duration_is_probed(self):
        rows = [{"audio_filepath": "/data/a0.wav", "text": _AR_RICH}]  # no duration
        with mock.patch("data_processing.prepare.probe_duration", return_value=3.5):
            report = self._run(rows)
        recs = _collect(self.pool, "ar", "internal_v76_ar")
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].duration, 3.5)
        self.assertEqual(report["counters"]["duration_probed"], 1)

    def test_unprobeable_local_duration_is_dropped(self):
        rows = [{"audio_filepath": "/data/a0.wav", "text": _AR_RICH}]
        with mock.patch("data_processing.prepare.probe_duration", return_value=None):
            report = self._run(rows)
        self.assertEqual(report["emitted"], 0)
        self.assertEqual(report["counters"]["duration_missing"], 1)

    def test_no_probe_flag_skips_probing_and_drops(self):
        rows = [{"audio_filepath": "/data/a0.wav", "text": _AR_RICH}]
        root = self.dir / "v7.6"
        _write_jsonl(root, "ar/train_0.jsonl", rows)
        with mock.patch("data_processing.prepare.probe_duration") as probe:
            report = prepare_source(_local_spec(), pool_dir=self.pool,
                                    audio_root=self.audio_root, root=str(root),
                                    quality=QualityConfig(), probe_local=False)
        probe.assert_not_called()
        self.assertEqual(report["emitted"], 0)
        self.assertEqual(report["counters"]["duration_missing"], 1)


class TestPrepareExternalSource(_Tmp):
    def _run(self, rows, *, duration_field=None, quality=None, name="fake_ext"):
        spec = _ext_spec(rows, name=name, duration_field=duration_field)
        return prepare_source(spec, pool_dir=self.pool, audio_root=self.audio_root,
                              root=None, quality=quality or QualityConfig())

    def test_identity_is_the_deterministic_materialized_path(self):
        rows = [{"path": f"audio/clip_{i}.flac", "text": f"{_AR_RICH} {i}"} for i in range(3)]
        report = self._run(rows)
        self.assertEqual(report["emitted"], 3)
        recs = _collect(self.pool, "ar", "fake_ext")
        expected = {derive_materialized_path(self.audio_root, "ar", "fake_ext",
                                             f"audio/clip_{i}.flac") for i in range(3)}
        self.assertEqual({r.audio_filepath for r in recs}, expected)
        for r in recs:
            self.assertTrue(r.external)
            self.assertTrue(r.audio_filepath.endswith(".flac"))
            self.assertTrue(r.native_id.startswith("audio/clip_"))

    def test_deferred_duration_keeps_row_via_text_only_gate(self):
        rows = [{"path": "audio/clip_0.flac", "text": _AR_RICH}]
        report = self._run(rows)  # duration_field None -> no metadata duration
        recs = _collect(self.pool, "ar", "fake_ext")
        self.assertEqual(len(recs), 1)
        self.assertIsNone(recs[0].duration)
        self.assertEqual(report["counters"]["duration_deferred"], 1)

    def test_metadata_duration_is_used_and_fully_gated(self):
        rows = [{"path": "audio/clip_0.flac", "text": _AR_RICH, "duration": 4.0}]
        report = self._run(rows, duration_field="duration")
        recs = _collect(self.pool, "ar", "fake_ext")
        self.assertEqual(recs[0].duration, 4.0)
        self.assertNotIn("duration_deferred", report["counters"])

    def test_needs_llm_triage_flag_is_carried(self):
        rows = [{"path": "audio/clip_0.flac", "text": _AR_RICH}]
        self._run(rows)
        rec = _collect(self.pool, "ar", "fake_ext")[0]
        self.assertIsInstance(rec.needs_llm, bool)
        self.assertIn("normalize", rec.stages)
        self.assertIn("accent", rec.stages)


class TestOverwriteAndClear(_Tmp):
    def test_rerun_removes_stale_higher_index_shards(self):
        root = self.dir / "v7.6"
        rows = [{"audio_filepath": f"/data/a{i}.wav", "text": f"{_AR_RICH} {i}",
                 "duration": "4.0"} for i in range(3)]
        _write_jsonl(root, "ar/train_0.jsonl", rows)
        spec = _local_spec()
        prepare_source(spec, pool_dir=self.pool, audio_root=self.audio_root,
                       root=str(root), quality=QualityConfig(), shard_size=1)
        self.assertEqual(len(iter_pool_shards(self.pool, "ar", "internal_v76_ar")), 3)
        # A shorter second run must not leave the old part-00001/2 behind.
        prepare_source(spec, pool_dir=self.pool, audio_root=self.audio_root,
                       root=str(root), quality=QualityConfig(), shard_size=1, limit=1)
        shards = iter_pool_shards(self.pool, "ar", "internal_v76_ar")
        self.assertEqual(len(shards), 1)
        self.assertEqual(len(_collect(self.pool, "ar", "internal_v76_ar")), 1)

    def test_clear_is_a_noop_for_a_missing_source_dir(self):
        self.assertEqual(_clear_source_pool(self.pool, "ar", "never_written"), 0)


class TestRowToRecordUnit(_Tmp):
    def test_no_path_is_counted_and_returns_none(self):
        counters = Counter()
        spec = _ext_spec([])
        rec = _row_to_record(spec, {"text": _AR_RICH}, audio_root=self.audio_root,
                             quality=QualityConfig(), diacritic_policy=DiacriticPolicy.CRITICAL_ONLY,
                             probe_local=True, counters=counters)
        self.assertIsNone(rec)
        self.assertEqual(counters["skip_no_path"], 1)

    def test_composite_equals_quality_times_richness(self):
        counters = Counter()
        spec = _ext_spec([{"path": "audio/c.flac", "text": _AR_RICH}])
        rec = _row_to_record(spec, {"path": "audio/c.flac", "text": _AR_RICH},
                             audio_root=self.audio_root, quality=QualityConfig(),
                             diacritic_policy=DiacriticPolicy.CRITICAL_ONLY,
                             probe_local=True, counters=counters)
        self.assertAlmostEqual(rec.composite, round(rec.quality * rec.richness, 4), places=4)


class TestRunPrepare(_Tmp):
    def test_aggregates_and_isolates_a_failing_source(self):
        good = _ext_spec([{"path": f"audio/c{i}.flac", "text": f"{_AR_RICH} {i}"}
                          for i in range(2)], name="good_ext")

        def bad_loader(spec):
            raise RuntimeError("hub 401")

        bad = DatasetSpec(name="bad_ext", lang="ar", kind=Kind.HF_STREAM, license="test",
                          repo_id="me/bad", fields=FieldMap(path="path", text="text",
                          duration=None), loader=bad_loader)

        def fake_specs_for(lang, *, include_gated=True, external_only=False):
            if lang == "ar":
                return (good, bad)
            return ()

        with mock.patch.object(registry, "specs_for", side_effect=fake_specs_for):
            report = run_prepare(langs=("ar",), pool_dir=self.pool, audio_root=self.audio_root,
                                 root=None, quality=QualityConfig(), num_nodes=1)
        self.assertEqual(report["stage"], "prepare")
        self.assertEqual(report["totals"]["emitted"], 2)
        self.assertEqual(report["totals"]["errored_sources"], 1)
        self.assertEqual(report["totals"]["stream_error"], 1)
        by_name = {s["source"]: s for s in report["sources"]}
        self.assertIn("error", by_name["bad_ext"])
        self.assertEqual(len(_collect(self.pool, "ar", "good_ext")), 2)

    def test_node_rank_selects_a_disjoint_slice(self):
        specs = [_ext_spec([{"path": f"audio/{n}.flac", "text": _AR_RICH}], name=n)
                 for n in ("s0", "s1", "s2", "s3")]

        def fake_specs_for(lang, *, include_gated=True, external_only=False):
            return tuple(specs) if lang == "ar" else ()

        with mock.patch.object(registry, "specs_for", side_effect=fake_specs_for):
            r0 = run_prepare(langs=("ar",), pool_dir=self.pool, audio_root=self.audio_root,
                             root=None, quality=QualityConfig(), node_rank=0, num_nodes=2)
            r1 = run_prepare(langs=("ar",), pool_dir=self.pool, audio_root=self.audio_root,
                             root=None, quality=QualityConfig(), node_rank=1, num_nodes=2)
        n0 = {s["source"] for s in r0["sources"]}
        n1 = {s["source"] for s in r1["sources"]}
        self.assertFalse(n0 & n1)
        self.assertEqual(n0 | n1, {"s0", "s1", "s2", "s3"})


def _semaphores_available() -> bool:
    """Whether spawned worker processes can be created here at all.

    Sandboxes that deny POSIX named semaphores cannot start a spawn context's
    pool, so the real-process test below guards on this instead of erroring.
    """
    try:
        multiprocessing.get_context("spawn").Lock()
    except (ImportError, OSError, PermissionError):
        return False
    return True


class _FakeFuture:
    """A resolved Future: ``result()`` raises when the worker 'crashed'."""

    def __init__(self, value):
        self._value = value

    def result(self):
        if isinstance(self._value, BaseException):
            raise self._value
        return self._value


class _FakePool:
    """A ProcessPoolExecutor stand-in that records tasks and runs them inline.

    ``run_prepare`` only needs ``submit`` and the context manager, so this is
    enough to prove the fan-out's orchestration -- which sources were
    submitted, with what payload, and how a crashed worker folds into the
    report -- without spawning processes (the real-spawn smoke below covers
    that half).
    """

    def __init__(self):
        self.tasks = []

    def submit(self, fn, task):
        self.tasks.append(task)
        try:
            return _FakeFuture(fn(task))
        except Exception as exc:  # mirrors a dead worker
            return _FakeFuture(exc)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestRunPrepareJobs(_Tmp):
    """``--jobs`` fan-out: partition, submission-order aggregation, crash isolation."""

    def _run_fanned(self, names, jobs, *, boom=(), **kw):
        pool = _FakePool()

        def fake_specs_for(lang, *, include_gated=True, external_only=False):
            if lang != "ar":
                return ()
            return tuple(_ext_spec([{"path": f"audio/{n}.flac", "text": _AR_RICH}],
                                    name=n) for n in names)

        def fake_task(task):
            if task["source"] in boom:
                raise RuntimeError("worker died")
            return {"source": task["source"], "lang": task["lang"], "external": True,
                    "read": 2, "emitted": 2, "shards": ["part-00000.jsonl"],
                    "counters": {}}

        with mock.patch.object(registry, "specs_for", side_effect=fake_specs_for), \
             mock.patch("data_processing.prepare._prepare_source_task", fake_task), \
             mock.patch("data_processing.prepare.ProcessPoolExecutor",
                        side_effect=lambda **kwargs: pool):
            report = run_prepare(langs=("ar",), pool_dir=self.pool,
                                 audio_root=self.audio_root, root=None,
                                 quality=QualityConfig(), jobs=jobs, **kw)
        return report, pool

    def test_fanout_aggregates_in_submission_order(self):
        report, pool = self._run_fanned(("s0", "s1", "s2", "s3"), 3,
                                        shard_size=7, limit=5)
        self.assertEqual(report["parallel"], {"jobs": 3, "ok": 4, "failed": 0})
        self.assertEqual([s["source"] for s in report["sources"]],
                         ["s0", "s1", "s2", "s3"])
        self.assertEqual(report["totals"]["read"], 8)
        self.assertEqual(report["totals"]["emitted"], 8)
        self.assertEqual(report["totals"]["shards"], 4)
        # Each worker got one picklable task, in assignment order.
        self.assertEqual([t["source"] for t in pool.tasks],
                         ["s0", "s1", "s2", "s3"])
        for task in pool.tasks:
            self.assertEqual(task["lang"], "ar")
            self.assertIsInstance(task["log_level"], int)
            self.assertEqual(task["kwargs"]["pool_dir"], self.pool)
            self.assertEqual(task["kwargs"]["audio_root"], self.audio_root)
            self.assertIsNone(task["kwargs"]["root"])
            self.assertEqual(task["kwargs"]["shard_size"], 7)
            self.assertEqual(task["kwargs"]["limit"], 5)
            self.assertIsInstance(task["kwargs"]["quality"], QualityConfig)

    def test_worker_crash_is_reported_not_fatal(self):
        report, _pool = self._run_fanned(("s0", "s1", "boom", "s2"), 3,
                                         boom={"boom"})
        self.assertEqual(report["parallel"], {"jobs": 3, "ok": 3, "failed": 1})
        by_name = {s["source"]: s for s in report["sources"]}
        self.assertEqual(by_name["boom"]["error"], "RuntimeError: worker died")
        self.assertEqual(by_name["boom"]["counters"], {"worker_crashed": 1})
        self.assertEqual(report["totals"]["read"], 6)  # three survivors x 2
        self.assertEqual(report["totals"]["emitted"], 6)
        self.assertEqual(report["totals"]["errored_sources"], 1)
        self.assertEqual(report["totals"]["worker_crashed"], 1)

    def test_jobs_is_capped_at_the_source_count(self):
        report, pool = self._run_fanned(("s0", "s1"), 99)
        self.assertEqual(report["parallel"]["jobs"], 2)
        self.assertEqual(len(pool.tasks), 2)

    def test_single_source_stays_serial_even_with_jobs(self):
        report, pool = self._run_fanned(("s0",), 4)
        self.assertEqual(report["parallel"], {"jobs": 1, "ok": None, "failed": 0})
        self.assertEqual(pool.tasks, [])  # serial: prepare_source ran inline
        self.assertEqual(report["totals"]["emitted"], 1)  # the fake spec's one row


@unittest.skipUnless(_semaphores_available(),
                     "spawned workers unavailable (no POSIX semaphores)")
class TestRunPrepareJobsRealSpawn(_Tmp):
    """One real fan-out over two real registry specs, on a tiny local tree."""

    def test_children_prepare_disjoint_sources_into_the_shared_pool(self):
        root = self.dir / "v7.6"
        _write_jsonl(root, "ar/train_a.jsonl",
                     [{"audio_filepath": f"/data/ar/a{i}.wav", "text": f"{_AR_RICH} {i}",
                       "duration": "4.0"} for i in range(3)])
        _write_jsonl(root, "ml/train_b.jsonl",
                     [{"audio_filepath": f"/data/ml/b{i}.wav",
                       "text": f"ഇതൊരു മലയാളം സാമ്പിൾ വാക്യം ആണ് ഇത് {i}",
                       "duration": "4.0"} for i in range(3)])
        report = run_prepare(
            langs=("ar", "ml"), only_sources=("internal_v76_ar", "internal_v76_ml"),
            pool_dir=self.pool, audio_root=self.audio_root, root=str(root),
            quality=QualityConfig(), probe_local=False, jobs=2,
        )
        self.assertEqual(report["parallel"], {"jobs": 2, "ok": 2, "failed": 0})
        self.assertEqual({s["source"] for s in report["sources"]},
                         {"internal_v76_ar", "internal_v76_ml"})
        self.assertEqual(report["totals"]["read"], 6)
        self.assertEqual(report["totals"]["emitted"], 6)
        # The spawned children really wrote the shared pool directory.
        self.assertEqual(len(_collect(self.pool, "ar", "internal_v76_ar")), 3)
        self.assertEqual(len(_collect(self.pool, "ml", "internal_v76_ml")), 3)


if __name__ == "__main__":
    unittest.main()
