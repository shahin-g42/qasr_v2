"""Tests for Stage 4: the K rule, MANIFEST.json, and the ship audit.

Batches are synthesized directly (manifest + sidecar via canonical I/O), so
every audit gate is provable in isolation: the K = min rule and surplus
reporting, the count contract, canonical record shape, intra/cross-language
duplicate paths, eval leak, external-audio presence (real 16 kHz mono FLACs
where audio is involved), the recomputed diversity floor, positive durations,
--dry-run, and the never-overwrite rule for a good MANIFEST.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from data_processing import bundle as B
from data_processing.canonical import Meta, Sample, write_manifest, write_sidecar

try:
    import numpy as np
    import soundfile as sf
except ImportError:  # pragma: no cover - environment dependent
    np = None


class _Tmp(unittest.TestCase):
    BATCH = 3

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.out = self.dir / "out"
        self.audio_root = self.dir / "audio"
        self.eval_root = self.dir / "v7.6"
        self.eval_root.mkdir(parents=True)
        self.addCleanup(self._tmp.cleanup)

    # --- batch construction --------------------------------------------------
    def write_batch(self, lang, label, rows, *, metas=None):
        """``rows`` are ``(audio_filepath, duration, text)``. Returns manifest path."""
        lang_dir = self.out / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        manifest = lang_dir / f"train_{lang}_{label}_p0000.jsonl"
        write_manifest(manifest, [Sample(p, d, t, lang) for p, d, t in rows])
        if metas is None:
            metas = [Meta(audio_filepath=p, dataset="src", quality=0.9)
                     for p, _d, _t in rows]
        write_sidecar(self._sidecar(manifest), metas)
        return manifest

    @staticmethod
    def _sidecar(manifest: Path) -> Path:
        return manifest.with_name(manifest.name.replace(".jsonl", ".meta.jsonl"))

    def rows(self, lang, n, *, label="x", duration=2.0, prefix=None, text=None):
        prefix = prefix if prefix is not None else f"/data/{lang}"
        return [(f"{prefix}/{label}{i}.wav", duration,
                 text if text is not None else f"unique transcript {lang} {label} {i}")
                for i in range(n)]

    def green_lang(self, lang, labels=("b0000",)):
        for label in labels:
            self.write_batch(lang, label, self.rows(lang, self.BATCH, label=label))

    def run_bundle(self, **kw):
        kw.setdefault("out_dir", self.out)
        kw.setdefault("audio_root", self.audio_root)
        kw.setdefault("root", self.eval_root)
        kw.setdefault("batch_size", self.BATCH)
        return B.run_bundle(**kw)


class TestKRule(_Tmp):
    def test_k_is_the_minimum_and_surplus_is_reported(self):
        self.green_lang("ar", ("b0000", "b0001"))
        self.green_lang("en", ("b0000", "b0001"))
        self.green_lang("zh", ("b0000",))
        self.write_batch("zh", "b0001", self.rows("zh", 2, label="b0001"))  # short
        self.green_lang("hi", ("b0000",))
        self.green_lang("ml", ("b0000",))
        report = self.run_bundle()
        self.assertTrue(report["audit"]["ok"], report["audit"]["failures"])
        self.assertEqual(report["k"], 1)
        self.assertEqual(report["full_batches"],
                         {"ar": 2, "en": 2, "zh": 1, "hi": 1, "ml": 1})
        self.assertEqual(report["surplus_full_batches"],
                         {"ar": ["b0001"], "en": ["b0001"]})
        self.assertEqual(report["short_batches"], {"zh": {"b0001": 2}})
        for lang in ("ar", "en", "zh", "hi", "ml"):
            self.assertEqual(report["shippable"][lang], ["b0000"])
        # A short batch is never shipped, so it never enters the manifest.
        self.assertNotIn("b0001", report["languages"]["zh"])

    def test_laggard_language_bounds_k(self):
        self.green_lang("ar", ("b0000", "b0001", "b0002"))
        for lang in ("en", "zh", "hi", "ml"):
            self.green_lang(lang, ("b0000",))
        report = self.run_bundle()
        self.assertEqual(report["k"], 1)
        self.assertEqual(report["surplus_full_batches"]["ar"], ["b0001", "b0002"])

    def test_no_full_batches_anywhere_fails_the_audit(self):
        for lang in ("ar", "en", "zh", "hi", "ml"):
            self.write_batch(lang, "b0000", self.rows(lang, 2, label="b0000"))
        report = self.run_bundle()
        self.assertEqual(report["k"], 0)
        self.assertFalse(report["audit"]["ok"])
        self.assertTrue(any("no shippable bundle" in f
                            for f in report["audit"]["failures"]))
        self.assertFalse(report["manifest_written"])
        self.assertFalse((self.out / "MANIFEST.json").exists())


class TestManifestContent(_Tmp):
    def test_manifest_carries_counts_stats_and_paths(self):
        rows = [(f"/data/ar/a{i}.wav", d, f"transcript number {i}")
                for i, d in enumerate((1200.0, 1800.0, 2400.0))]
        metas = [
            Meta(audio_filepath=p, dataset="srcA" if i < 2 else "srcB",
                 accent="msa" if i < 2 else None, quality=0.9,
                 metrics={"richness": 0.5 + 0.1 * i})
            for i, (p, _d, _t) in enumerate(rows)
        ]
        self.write_batch("ar", "b0000", rows, metas=metas)
        # An eval file with an unrelated path: loaded, but nothing leaks.
        (self.eval_root / "ar").mkdir(exist_ok=True)
        (self.eval_root / "ar" / "eval_ar.jsonl").write_text(
            json.dumps({"audio_filepath": "/not/shipped.wav"}) + "\n", encoding="utf-8")

        report = self.run_bundle(langs=("ar",))
        self.assertTrue(report["audit"]["ok"], report["audit"]["failures"])
        self.assertEqual(report["audit"]["eval_paths_loaded"], 1)
        stats = report["languages"]["ar"]["b0000"]
        self.assertEqual(stats["rows"], 3)
        self.assertEqual(stats["hours"], 1.5)  # 5400 s of audio, in hours
        self.assertEqual(stats["sources"], {"srcA": 2, "srcB": 1})
        self.assertEqual(stats["accents"], {"msa": 2, "unknown": 1})
        self.assertEqual(stats["external_rows"], 0)
        self.assertEqual(stats["internal_rows"], 3)
        self.assertEqual(stats["distinct_text_fraction"], 1.0)
        self.assertEqual(stats["mean_quality"], 0.9)
        self.assertEqual(stats["mean_richness"], 0.6)
        self.assertIn("p50", stats["durations"])
        # Manifests are referenced by path, never copied.
        self.assertTrue(Path(stats["manifest"]).exists())
        self.assertEqual(Path(stats["manifest"]).name, "train_ar_b0000_p0000.jsonl")

        manifest = json.loads((self.out / "MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["version"], "v8.0")
        self.assertEqual(manifest["generator"], "data_processing.bundle/1")
        self.assertEqual(manifest["bundles"], 1)
        self.assertEqual(manifest["bundle_size"],
                         {"per_language": 3, "languages": ["ar"]})
        self.assertEqual(manifest["batch_labels"], {"ar": ["b0000"]})
        self.assertEqual(manifest["languages"]["ar"]["b0000"]["rows"], 3)
        self.assertEqual(manifest["audio_root"], str(self.audio_root.resolve()))


class TestAuditFailures(_Tmp):
    def _one_lang_report(self, **kw):
        return self.run_bundle(langs=("ar",), **kw)

    def test_duplicate_path_across_languages(self):
        shared = [("/shared/clip.wav", 2.0, "a shared transcript")]
        self.write_batch("ar", "b0000", self.rows("ar", 2, label="a") + shared)
        self.write_batch("en", "b0000", self.rows("en", 2, label="e") + shared)
        report = self.run_bundle(langs=("ar", "en"))
        self.assertFalse(report["audit"]["ok"])
        self.assertTrue(any("duplicate audio_filepath" in f
                            for f in report["audit"]["failures"]))

    def test_duplicate_path_within_a_language(self):
        rows = self.rows("ar", 2, label="d")
        dup = [*rows, (rows[0][0], 2.0, "a different transcript for the same clip")]
        self.write_batch("ar", "b0000", dup)
        report = self._one_lang_report()
        self.assertFalse(report["audit"]["ok"])
        self.assertTrue(any("duplicate audio_filepath" in f
                            for f in report["audit"]["failures"]))

    def test_eval_leak_fails_the_audit(self):
        rows = self.rows("ar", 3, label="a")
        self.write_batch("ar", "b0000", rows)
        (self.eval_root / "ar").mkdir(parents=True, exist_ok=True)
        (self.eval_root / "ar" / "eval_ar.jsonl").write_text(
            json.dumps({"audio_filepath": rows[0][0]}) + "\n", encoding="utf-8")
        report = self._one_lang_report()
        self.assertFalse(report["audit"]["ok"])
        self.assertTrue(any("eval-leaked" in f for f in report["audit"]["failures"]))

    def test_eval_leak_via_a_second_eval_root_fails_the_audit(self):
        """The q3asr SFT tree ships its own eval sets; a path leaked from
        THERE must fail the audit even though it is in no v7.6 eval file."""
        rows = self.rows("ar", 3, label="a")
        self.write_batch("ar", "b0000", rows)
        sft = self.dir / "q3asr_sft_manifests"
        (sft / "ar").mkdir(parents=True)
        (sft / "ar" / "eval_ar_q3asr.jsonl").write_text(
            json.dumps({"audio_filepath": rows[0][0]}) + "\n", encoding="utf-8")
        report = self._one_lang_report(eval_roots=[sft])
        self.assertFalse(report["audit"]["ok"])
        self.assertTrue(any("eval-leaked" in f for f in report["audit"]["failures"]))
        # The same second root is inert once nothing leaks from it.
        (sft / "ar" / "eval_ar_q3asr.jsonl").write_text(
            json.dumps({"audio_filepath": "/not/shipped.wav"}) + "\n", encoding="utf-8")
        report = self._one_lang_report(eval_roots=[sft])
        self.assertTrue(report["audit"]["ok"], report["audit"]["failures"])
        self.assertEqual(report["audit"]["eval_paths_loaded"], 1)

    def test_missing_external_audio_fails_the_audit(self):
        rows = [(f"{self.audio_root}/ar/src/a{i}.flac", 2.0, f"external text {i}")
                for i in range(3)]
        self.write_batch("ar", "b0000", rows)
        report = self._one_lang_report()
        self.assertFalse(report["audit"]["ok"])
        self.assertTrue(any("missing or not 16000 Hz mono" in f
                            for f in report["audit"]["failures"]))

    def test_non_canonical_record_breaks_the_count_contract(self):
        manifest = self.write_batch("ar", "b0000", self.rows("ar", 3, label="a"))
        lines = manifest.read_text(encoding="utf-8").splitlines()
        lines[1] = json.dumps({"audio_filepath": "/data/ar/broken.wav"})  # missing keys
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        report = self._one_lang_report()
        self.assertFalse(report["audit"]["ok"])
        joined = "\n".join(report["audit"]["failures"])
        self.assertIn("non-canonical record", joined)
        # The line count still says BATCH, so the parsed shortfall breaks it too.
        self.assertIn("count contract broken", joined)

    def test_sidecar_missing_fails_the_audit(self):
        manifest = self.write_batch("ar", "b0000", self.rows("ar", 3, label="a"))
        self._sidecar(manifest).unlink()
        report = self._one_lang_report()
        self.assertFalse(report["audit"]["ok"])
        self.assertTrue(any("sidecar missing" in f
                            for f in report["audit"]["failures"]))

    def test_sidecar_out_of_sync_fails_the_audit(self):
        rows = self.rows("ar", 3, label="a")
        metas = [Meta(audio_filepath=rows[0][0], dataset="src", quality=0.9)] * 3
        metas[1] = Meta(audio_filepath="/data/ar/somewhere_else.wav",
                        dataset="src", quality=0.9)
        self.write_batch("ar", "b0000", rows, metas=metas)
        report = self._one_lang_report()
        self.assertFalse(report["audit"]["ok"])
        self.assertTrue(any("sidecar out of sync" in f
                            for f in report["audit"]["failures"]))

    def test_sidecar_short_of_the_manifest_fails_the_audit(self):
        # A sidecar with the right line count but blank rows: the manifest and
        # sidecar stay in lockstep, yet only 1 of 3 rows carry provenance.
        rows = self.rows("ar", 3, label="a")
        metas = [Meta(audio_filepath=rows[0][0], dataset="src", quality=0.9)]
        manifest = self.write_batch("ar", "b0000", rows, metas=metas)
        with self._sidecar(manifest).open("a", encoding="utf-8") as fh:
            fh.write("\n\n")  # rows 2 and 3: blank sidecar lines
        report = self._one_lang_report()
        self.assertFalse(report["audit"]["ok"])
        self.assertTrue(any("sidecar covers 1 of 3" in f
                            for f in report["audit"]["failures"]))

    def test_diversity_floor_is_recomputed_from_the_shipped_texts(self):
        rows = self.rows("ar", 3, label="a", text="the same transcript over and over")
        self.write_batch("ar", "b0000", rows)
        report = self._one_lang_report()
        self.assertFalse(report["audit"]["ok"])
        self.assertTrue(any("distinct-transcript fraction" in f
                            for f in report["audit"]["failures"]))
        self.assertLess(report["languages"]["ar"]["b0000"]["distinct_text_fraction"], 0.5)

    def test_non_positive_duration_fails_the_audit(self):
        rows = self.rows("ar", 3, label="a")
        rows[1] = (rows[1][0], 0.0, rows[1][2])
        self.write_batch("ar", "b0000", rows)
        report = self._one_lang_report()
        self.assertFalse(report["audit"]["ok"])
        self.assertTrue(any("non-positive duration" in f
                            for f in report["audit"]["failures"]))

    def test_out_of_band_durations_and_source_caps_only_warn(self):
        rows = [(f"/data/ar/a{i}.wav", 0.1, f"tiny clip transcript {i}")
                for i in range(3)]
        self.write_batch("ar", "b0000", rows)
        report = self._one_lang_report()
        self.assertTrue(report["audit"]["ok"], report["audit"]["failures"])
        warnings = "\n".join(report["audit"]["warnings"])
        self.assertIn("duration band", warnings)
        self.assertIn("source 'src'", warnings)  # one source holds 100% > 40% cap

    def test_dry_run_audits_but_writes_nothing(self):
        self.green_lang("ar")
        report = self._one_lang_report(dry_run=True)
        self.assertTrue(report["audit"]["ok"])
        self.assertTrue(report["dry_run"])
        self.assertFalse(report["manifest_written"])
        self.assertFalse((self.out / "MANIFEST.json").exists())

    def test_failed_audit_never_overwrites_a_good_manifest(self):
        self.green_lang("ar")
        first = self._one_lang_report()
        self.assertTrue(first["audit"]["ok"])
        good = (self.out / "MANIFEST.json").read_text(encoding="utf-8")
        # Corrupt: one row now points at a missing external clip.
        rows = self.rows("ar", 3, label="a")
        rows[2] = (f"{self.audio_root}/ar/src/missing.flac", 2.0, rows[2][2])
        self.write_batch("ar", "b0000", rows)
        second = self._one_lang_report()
        self.assertFalse(second["audit"]["ok"])
        self.assertFalse(second["manifest_written"])
        self.assertEqual((self.out / "MANIFEST.json").read_text(encoding="utf-8"), good)


@unittest.skipIf(np is None, "audio dependencies (numpy/scipy/soundfile) are not installed")
class TestExternalAudioAudit(_Tmp):
    def _flac(self, rel, *, sr=16_000, channels=1, seconds=1.0):
        path = self.audio_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
        data = 0.3 * np.sin(2 * np.pi * 440 * t)
        if channels == 2:
            data = np.stack([data, data], axis=1)
        sf.write(str(path), data.astype(np.float32), sr, format="FLAC")
        return str(path)

    def _rows(self, paths):
        return [(p, 1.0, f"external transcript {i}") for i, p in enumerate(paths)]

    def test_good_external_audio_passes_full_verification(self):
        paths = [self._flac(f"ar/src/a{i}.flac") for i in range(3)]
        self.write_batch("ar", "b0000", self._rows(paths))
        report = self.run_bundle(langs=("ar",))
        self.assertTrue(report["audit"]["ok"], report["audit"]["failures"])
        stats = report["languages"]["ar"]["b0000"]
        self.assertEqual(stats["external_rows"], 3)
        self.assertEqual(stats["internal_rows"], 0)

    def test_wrong_sample_rate_fails_the_audit(self):
        paths = [self._flac(f"ar/src/a{i}.flac", sr=8_000) for i in range(3)]
        self.write_batch("ar", "b0000", self._rows(paths))
        report = self.run_bundle(langs=("ar",))
        self.assertFalse(report["audit"]["ok"])
        self.assertTrue(any("not 16000 Hz mono" in f
                            for f in report["audit"]["failures"]))

    def test_stereo_fails_the_audit(self):
        paths = [self._flac(f"ar/src/a{i}.flac", channels=2) for i in range(3)]
        self.write_batch("ar", "b0000", self._rows(paths))
        report = self.run_bundle(langs=("ar",))
        self.assertFalse(report["audit"]["ok"])
        self.assertTrue(any("not 16000 Hz mono" in f
                            for f in report["audit"]["failures"]))

    def test_spot_check_passes_when_the_sampled_clips_are_good(self):
        paths = [self._flac(f"ar/src/a{i}.flac") for i in range(3)]
        self.write_batch("ar", "b0000", self._rows(paths))
        report = self.run_bundle(langs=("ar",), spot_check=2)
        self.assertTrue(report["audit"]["ok"], report["audit"]["failures"])
        self.assertEqual(report["audit"]["spot_check"], 2)


class TestDiscovery(_Tmp):
    def test_legacy_dataset_shards_are_not_batch_candidates(self):
        self.green_lang("ar")
        legacy = self.out / "ar" / "train_ar_ar_ae_p0000.jsonl"
        write_manifest(legacy, [Sample("/old.wav", 2.0, "legacy row", "ar")])
        write_sidecar(self._sidecar(legacy),
                      [Meta(audio_filepath="/old.wav", dataset="old")])
        labels = B.discover_batches(self.out, "ar")
        self.assertEqual(list(labels), ["b0000"])

    def test_gzipped_batches_are_discovered_and_audited(self):
        lang_dir = self.out / "ar"
        lang_dir.mkdir(parents=True, exist_ok=True)
        manifest = lang_dir / "train_ar_b0000_p0000.jsonl.gz"
        rows = self.rows("ar", 3, label="a")
        write_manifest(manifest, [Sample(p, d, t, "ar") for p, d, t in rows])
        write_sidecar(self._sidecar(manifest),
                      [Meta(audio_filepath=p, dataset="src", quality=0.9)
                       for p, _d, _t in rows])
        report = self.run_bundle(langs=("ar",))
        self.assertTrue(report["audit"]["ok"], report["audit"]["failures"])
        self.assertEqual(report["languages"]["ar"]["b0000"]["rows"], 3)


class TestMain(_Tmp):
    def test_exit_code_is_the_verdict(self):
        self.green_lang("ar")
        report_path = self.dir / "bundle.json"
        self.assertEqual(B.main([
            "--out-dir", str(self.out), "--audio-root", str(self.audio_root),
            "--root", str(self.eval_root), "--langs", "ar",
            "--batch-size", "3", "--report", str(report_path),
        ]), 0)
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertTrue(payload["audit"]["ok"])
        self.assertTrue((self.out / "MANIFEST.json").exists())

        # A broken batch flips the exit code to 1: the pipeline's ship gate.
        manifest = self.out / "ar" / "train_ar_b0000_p0000.jsonl"
        self._sidecar(manifest).unlink()
        self.assertEqual(B.main([
            "--out-dir", str(self.out), "--audio-root", str(self.audio_root),
            "--root", str(self.eval_root), "--langs", "ar", "--batch-size", "3",
        ]), 1)


if __name__ == "__main__":
    unittest.main()
