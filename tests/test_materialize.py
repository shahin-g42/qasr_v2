"""Tests for Stage-3 audio materialization.

Covers the signal path (mono collapse, polyphase resample, atomic 16-bit FLAC
write), verification/idempotency, the injectable fetcher seam, concurrent
error isolation, streamed-source selection, and the ``ensure_batch_audio``
verify/backfill pass that reconciles written batches against the pool.

``qasr.audio`` is deliberately *not* imported here: :mod:`data_processing.materialize`
carries its own decode/resample primitives so this GPU-free stage never pulls in
the model package. That also means these tests run wherever numpy/scipy/soundfile
are present, instead of being skipped like ``test_audio.py``.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from data_processing.candidate import PoolRecord, pool_shard_path, write_pool
from data_processing.canonical import Sample, shard_paths, write_manifest

try:
    import numpy as np
    import soundfile as sf

    from data_processing import materialize as M
except ImportError:  # pragma: no cover - environment dependent
    np = None


def _sine(seconds: float, sr: int, freq: float = 440.0):
    t = np.linspace(0.0, seconds, int(seconds * sr), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _stereo(seconds: float, sr: int):
    left = _sine(seconds, sr, 440.0)
    right = _sine(seconds, sr, 880.0) * 0.5
    return np.stack([left, right], axis=1).astype(np.float32)


@unittest.skipIf(np is None, "audio dependencies (numpy/scipy/soundfile) are not installed")
class _Tmp(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)


class TestSignalPrimitives(_Tmp):
    def test_to_mono_collapses_channels_by_averaging(self):
        stereo = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
        mono = M._to_mono(stereo)
        self.assertEqual(mono.ndim, 1)
        np.testing.assert_allclose(mono, [2.0, 3.0])

    def test_to_mono_leaves_1d_untouched(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_allclose(M._to_mono(arr), arr)

    def test_resample_scales_length_by_rate_ratio(self):
        out = M._resample(_sine(1.0, 48_000), 48_000, 16_000)
        self.assertEqual(out.dtype, np.float32)
        self.assertLessEqual(abs(len(out) - 16_000), 2)

    def test_resample_same_rate_is_a_passthrough(self):
        arr = _sine(0.5, 16_000)
        out = M._resample(arr, 16_000, 16_000)
        np.testing.assert_allclose(out, arr)

    def test_resample_rejects_nonpositive_rate(self):
        with self.assertRaises(M.MaterializeError):
            M._resample(_sine(0.1, 16_000), 0, 16_000)


class TestWriteFlac(_Tmp):
    def test_writes_16k_mono_and_reports_duration(self):
        target = self.dir / "out.flac"
        dur = M.write_flac(_sine(1.0, 16_000), target, 16_000)
        self.assertAlmostEqual(dur, 1.0, places=2)
        self.assertEqual(M.probe(target), (round(dur, 3), 16_000, 1))

    def test_creates_missing_parent_directories(self):
        target = self.dir / "a" / "b" / "out.flac"
        M.write_flac(_sine(0.2, 16_000), target, 16_000)
        self.assertTrue(target.is_file())

    def test_leaves_no_temp_file_behind(self):
        target = self.dir / "out.flac"
        M.write_flac(_sine(0.2, 16_000), target, 16_000)
        self.assertEqual(list(self.dir.glob("*.tmp")), [])

    def test_empty_audio_raises(self):
        with self.assertRaises(M.MaterializeError):
            M.write_flac(np.zeros(0, dtype=np.float32), self.dir / "e.flac", 16_000)

    def test_nonfinite_audio_raises_and_writes_nothing(self):
        bad = np.array([1.0, np.nan, 3.0], dtype=np.float32)
        target = self.dir / "bad.flac"
        with self.assertRaises(M.MaterializeError):
            M.write_flac(bad, target, 16_000)
        self.assertFalse(target.exists())


class TestMaterializeArrayAndFile(_Tmp):
    def test_stereo_48k_becomes_mono_16k(self):
        target = self.dir / "clip.flac"
        res = M.materialize_array(_stereo(1.0, 48_000), 48_000, target, 16_000)
        self.assertFalse(res.skipped)
        self.assertEqual(res.sample_rate, 16_000)
        self.assertAlmostEqual(res.duration, 1.0, places=2)
        self.assertEqual(M.probe(target)[1:], (16_000, 1))

    def test_materialize_file_resamples_an_on_disk_source(self):
        src = self.dir / "src.wav"
        sf.write(str(src), _sine(1.0, 8_000), 8_000)
        target = self.dir / "dst.flac"
        res = M.materialize_file(src, target, 16_000)
        self.assertEqual(M.probe(target), (round(res.duration, 3), 16_000, 1))


class TestProbeVerify(_Tmp):
    def test_probe_missing_file_is_none(self):
        self.assertIsNone(M.probe(self.dir / "nope.flac"))

    def test_verify_rejects_wrong_sample_rate(self):
        target = self.dir / "c.flac"
        M.write_flac(_sine(0.5, 16_000), target, 16_000)
        self.assertTrue(M.verify(target, 16_000))
        self.assertFalse(M.verify(target, 22_050))

    def test_verify_rejects_stereo(self):
        target = self.dir / "stereo.flac"
        sf.write(str(target), _stereo(0.5, 16_000), 16_000)
        self.assertFalse(M.verify(target, 16_000))

    def test_verify_duration_tolerance(self):
        target = self.dir / "d.flac"
        M.write_flac(_sine(1.0, 16_000), target, 16_000)
        self.assertTrue(M.verify(target, 16_000, expected_duration=1.0))
        self.assertFalse(M.verify(target, 16_000, expected_duration=2.0, tol=0.05))

    def test_verify_missing_is_false(self):
        self.assertFalse(M.verify(self.dir / "gone.flac", 16_000))


class TestLocalFetcher(_Tmp):
    def test_reads_a_local_file(self):
        src = self.dir / "clip.wav"
        sf.write(str(src), _sine(0.4, 16_000), 16_000)
        array, sr = M.local_file_fetcher(None, str(src))
        self.assertEqual(sr, 16_000)
        self.assertEqual(array.ndim, 2)

    def test_missing_path_raises_materialize_error(self):
        with self.assertRaises(M.MaterializeError):
            M.local_file_fetcher(None, str(self.dir / "absent.wav"))


class TestMaterializer(_Tmp):
    def _fake_fetcher(self, sr: int = 8_000):
        def fetch(spec, native_id):
            return _sine(0.5, sr), sr

        return fetch

    def test_is_external_boundary(self):
        mat = M.Materializer(str(self.dir / "audio"), 16_000)
        self.assertTrue(mat.is_external(os.path.join(str(self.dir / "audio"), "ar/x.flac")))
        self.assertFalse(mat.is_external("/internal/tree/x.wav"))

    def test_uses_injected_fetcher_for_remote_native_id(self):
        audio_root = self.dir / "audio"
        mat = M.Materializer(str(audio_root), 16_000, fetcher=self._fake_fetcher(8_000))
        target = str(audio_root / "ar" / "src" / "deadbeef.flac")
        res = mat.materialize(target, "remote/clip_1", None, None, "src")
        self.assertFalse(res.skipped)
        self.assertEqual(res.source, "src")
        self.assertEqual(M.probe(target), (round(res.duration, 3), 16_000, 1))

    def test_idempotent_skip_when_already_correct(self):
        audio_root = self.dir / "audio"
        mat = M.Materializer(str(audio_root), 16_000, fetcher=self._fake_fetcher())
        target = str(audio_root / "ar" / "src" / "aa.flac")
        first = mat.materialize(target, "remote/clip", None, None, "src")
        second = mat.materialize(target, "remote/clip", None, None, "src")
        self.assertFalse(first.skipped)
        self.assertTrue(second.skipped)
        self.assertAlmostEqual(second.duration, first.duration, places=2)

    def test_local_native_id_bypasses_fetcher(self):
        # A native_id that resolves on disk is read directly, no Hub round-trip.
        src = self.dir / "warm.wav"
        sf.write(str(src), _sine(0.5, 16_000), 16_000)

        def boom(spec, native_id):  # would fail the test if called
            raise AssertionError("fetcher must not be used for a resolvable local path")

        audio_root = self.dir / "audio"
        mat = M.Materializer(str(audio_root), 16_000, fetcher=boom)
        target = str(audio_root / "ar" / "src" / "bb.flac")
        res = mat.materialize(target, str(src), None, None, "src")
        self.assertFalse(res.skipped)
        self.assertTrue(Path(target).is_file())

    def test_generic_fetch_error_is_wrapped(self):
        def bad(spec, native_id):
            raise ValueError("network exploded")

        mat = M.Materializer(str(self.dir / "audio"), 16_000, fetcher=bad)
        with self.assertRaises(M.MaterializeError):
            mat.materialize(str(self.dir / "audio/x.flac"), "remote/clip", None, None, "src")


class TestMaterializeMany(_Tmp):
    def test_empty_items_short_circuits(self):
        mat = M.Materializer(str(self.dir / "audio"), 16_000, fetcher=lambda s, n: (_sine(0.1, 8_000), 8_000))
        results, errors = M.materialize_many([], mat)
        self.assertEqual((results, errors), ([], []))

    def test_one_bad_clip_does_not_abort_the_rest(self):
        audio_root = self.dir / "audio"

        def fetcher(spec, native_id):
            if native_id == "bad":
                raise ValueError("nope")
            return _sine(0.3, 8_000), 8_000

        mat = M.Materializer(str(audio_root), 16_000, fetcher=fetcher)
        items = [
            (str(audio_root / "ar/s/good1.flac"), "good1", None, None, "s"),
            (str(audio_root / "ar/s/bad.flac"), "bad", None, None, "s"),
            (str(audio_root / "ar/s/good2.flac"), "good2", None, None, "s"),
        ]
        results, errors = M.materialize_many(items, mat, max_workers=3)
        self.assertEqual(len(results), 2)
        self.assertEqual(len(errors), 1)
        target, exc_type, _msg = errors[0]
        self.assertTrue(target.endswith("bad.flac"))
        self.assertEqual(exc_type, "MaterializeError")


class TestSourceStream(_Tmp):
    def test_only_selected_rows_are_decoded_and_written(self):
        rows = [
            {"audio": {"path": "clipA"}},
            {"audio": {"path": "clipB"}},
            {"audio": {"path": "clipC"}},
        ]
        decoded = []

        def decode(row):
            decoded.append(row["audio"]["path"])
            return _sine(0.2, 8_000), 8_000

        tgt_a = str(self.dir / "audio/ar/s/a.flac")
        tgt_c = str(self.dir / "audio/ar/s/c.flac")
        selected = {tgt_a: "clipA", tgt_c: "clipC"}
        out = list(M.materialize_source_stream(rows, decode, selected, 16_000))
        self.assertEqual(len(out), 2)
        self.assertEqual(sorted(decoded), ["clipA", "clipC"])
        self.assertTrue(Path(tgt_a).is_file())
        self.assertTrue(Path(tgt_c).is_file())
        self.assertFalse((self.dir / "audio/ar/s/b.flac").exists())

    def test_row_matches_plain_and_nested_and_suffix(self):
        self.assertTrue(M._row_matches({"path": "clipA"}, "t", "clipA"))
        self.assertTrue(M._row_matches({"audio": {"path": "x/clipA"}}, "t", "clipA"))
        self.assertTrue(M._row_matches({"file": "clipA"}, "t", "clipA"))
        self.assertFalse(M._row_matches({"audio": {"path": "clipZ"}}, "t", "clipA"))


@unittest.skipIf(np is None, "audio dependencies (numpy/scipy/soundfile) are not installed")
class TestEnsureBatchAudio(_Tmp):
    def _write_batch(self, out_dir, lang, dataset, audio_filepath, duration=1.0, text="hello"):
        manifest, _sidecar = shard_paths(Path(out_dir) / lang, lang, dataset, 0)
        write_manifest(manifest, [Sample(audio_filepath, duration, text, lang)])
        return manifest

    def test_backfills_a_missing_external_clip_from_the_pool(self):
        out_dir = self.dir / "out"
        pool_dir = self.dir / "pool"
        audio_root = self.dir / "audio"
        src = self.dir / "warm.wav"
        sf.write(str(src), _sine(1.0, 8_000), 8_000)

        external = str(audio_root / "ar" / "srcX" / "abcd1234.flac")
        self._write_batch(out_dir, "ar", "srcX", external)
        write_pool(
            pool_shard_path(pool_dir, "ar", "srcX", 0),
            [PoolRecord(audio_filepath=external, source="srcX", lang="ar",
                        normalized_text="hello", external=True, native_id=str(src))],
        )

        report = M.ensure_batch_audio(out_dir, "ar", pool_dir, audio_root,
                                      fetcher=M.local_file_fetcher)
        self.assertEqual(report.written, 1)
        self.assertEqual(report.verified, 0)
        self.assertTrue(M.verify(external, 16_000))

        # Second pass finds it present -> verified, never rewritten.
        again = M.ensure_batch_audio(out_dir, "ar", pool_dir, audio_root,
                                     fetcher=M.local_file_fetcher)
        self.assertEqual(again.written, 0)
        self.assertEqual(again.verified, 1)

    def test_missing_native_id_is_reported_not_fatal(self):
        out_dir = self.dir / "out"
        pool_dir = self.dir / "pool"
        audio_root = self.dir / "audio"
        external = str(audio_root / "ar" / "srcX" / "ffff.flac")
        self._write_batch(out_dir, "ar", "srcX", external)
        # Pool exists but has no record for this path.
        write_pool(pool_shard_path(pool_dir, "ar", "srcX", 0), [])

        report = M.ensure_batch_audio(out_dir, "ar", pool_dir, audio_root,
                                      fetcher=M.local_file_fetcher)
        self.assertEqual(report.errors["no_native_id"], 1)
        self.assertEqual(report.written, 0)
        self.assertIn(external, report.examples)

    def test_internal_paths_are_ignored(self):
        out_dir = self.dir / "out"
        pool_dir = self.dir / "pool"
        audio_root = self.dir / "audio"
        self._write_batch(out_dir, "ar", "internal_v76_ar", "/internal/tree/ar/x.wav")

        report = M.ensure_batch_audio(out_dir, "ar", pool_dir, audio_root,
                                      fetcher=M.local_file_fetcher)
        self.assertEqual(report.verified, 0)
        self.assertEqual(report.written, 0)
        self.assertEqual(dict(report.errors), {})

    def test_report_as_dict_shape(self):
        rep = M.MaterializeReport(verified=2, written=1, skipped=3)
        rep.errors["MaterializeError"] = 4
        rep.examples.append("x")
        d = rep.as_dict()
        self.assertEqual(d["verified"], 2)
        self.assertEqual(d["errors"], {"MaterializeError": 4})
        self.assertEqual(d["examples"], ["x"])


class TestPrewarmPool(_Tmp):
    def _ext(self, audio_root, native, composite, *, lang="ar", source="srcX"):
        external = str(audio_root / lang / source / f"{Path(native).stem}.flac")
        rec = PoolRecord(audio_filepath=external, source=source, lang=lang,
                         normalized_text="hello", composite=composite,
                         external=True, native_id=str(native))
        return rec, external

    def test_prewarms_only_the_top_external_candidates(self):
        pool_dir, audio_root = self.dir / "pool", self.dir / "audio"
        natives = [self.dir / f"n{i}.wav" for i in range(4)]
        for n in natives:
            sf.write(str(n), _sine(1.0, 8_000), 8_000)
        recs, externals = [], []
        for i, n in enumerate(natives):
            rec, ext = self._ext(audio_root, n, composite=0.9 - 0.1 * i)
            recs.append(rec)
            externals.append(ext)
        # Highest composite of all, but internal: never fetched, never counted.
        recs.append(PoolRecord(audio_filepath="/internal/tree/ar/x.wav", source="srcI",
                               lang="ar", normalized_text="hi", composite=1.0,
                               external=False, duration=2.0))
        write_pool(pool_shard_path(pool_dir, "ar", "srcX", 0), recs)

        report = M.prewarm_pool("ar", pool_dir, audio_root, 2,
                                fetcher=M.local_file_fetcher, max_workers=2)
        self.assertEqual(report.written, 2)
        for ext in externals[:2]:  # top-2 by composite, now 16 kHz mono
            self.assertTrue(M.verify(ext, 16_000))
        for ext in externals[2:]:  # beyond the budget
            self.assertFalse(Path(ext).exists())

    def test_prewarm_is_idempotent_on_a_rerun(self):
        pool_dir, audio_root = self.dir / "pool", self.dir / "audio"
        native = self.dir / "warm.wav"
        sf.write(str(native), _sine(1.0, 8_000), 8_000)
        rec, ext = self._ext(audio_root, native, composite=0.5)
        write_pool(pool_shard_path(pool_dir, "ar", "srcX", 0), [rec])

        first = M.prewarm_pool("ar", pool_dir, audio_root, 1,
                               fetcher=M.local_file_fetcher)
        self.assertEqual(first.written, 1)
        second = M.prewarm_pool("ar", pool_dir, audio_root, 1,
                                fetcher=M.local_file_fetcher)
        self.assertEqual(second.written, 0)
        self.assertEqual(second.skipped, 1)
        self.assertTrue(M.verify(ext, 16_000))

    def test_nonpositive_budget_drains_every_external_candidate(self):
        pool_dir, audio_root = self.dir / "pool", self.dir / "audio"
        natives = [self.dir / f"d{i}.wav" for i in range(3)]
        for n in natives:
            sf.write(str(n), _sine(1.0, 8_000), 8_000)
        recs = [self._ext(audio_root, n, composite=0.5)[0] for n in natives]
        write_pool(pool_shard_path(pool_dir, "ar", "srcX", 0), recs)

        report = M.prewarm_pool("ar", pool_dir, audio_root, 0,
                                fetcher=M.local_file_fetcher)
        self.assertEqual(report.written, 3)

    def test_missing_pool_returns_an_empty_report(self):
        report = M.prewarm_pool("ar", self.dir / "pool", self.dir / "audio", 10,
                                fetcher=M.local_file_fetcher)
        self.assertEqual(report.written, 0)
        self.assertEqual(dict(report.errors), {})

    def test_cli_prewarm_mode_needs_no_out_dir(self):
        pool_dir, audio_root = self.dir / "pool", self.dir / "audio"
        native = self.dir / "cli.wav"
        sf.write(str(native), _sine(1.0, 8_000), 8_000)
        rec, ext = self._ext(audio_root, native, composite=0.5)
        write_pool(pool_shard_path(pool_dir, "ar", "srcX", 0), [rec])

        report_path = self.dir / "report.json"
        code = M.main(["--lang", "ar", "--pool-dir", str(pool_dir),
                       "--audio-root", str(audio_root), "--prewarm", "--budget", "1",
                       "--workers", "2", "--report", str(report_path)])
        self.assertEqual(code, 0)
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["mode"], "prewarm")
        self.assertEqual(payload["budget"], 1)
        self.assertEqual(payload["written"], 1)
        self.assertTrue(M.verify(ext, 16_000))

    def test_cli_backfill_mode_still_requires_out_dir(self):
        with self.assertRaises(SystemExit) as ctx:
            M.main(["--lang", "ar", "--pool-dir", str(self.dir / "pool"),
                    "--audio-root", str(self.dir / "audio")])
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
