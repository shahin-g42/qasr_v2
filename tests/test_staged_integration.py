"""End-to-end integration test for the staged nine-node build.

One synthetic external source is driven through all four stages --
prepare -> assemble -> materialize -> bundle -- with the corrector scripted
(``BatchCorrector._post``) and the fetcher backed by real 8 kHz wav fixtures
that the stages themselves resample to 16 kHz mono FLAC. The first pass must
end green: a full corrected batch, a backfill that verifies everything, a
passing audit and a written ``MANIFEST.json``. Then one shipped clip is
corrupted and the same audit must fail with exit code 1 -- the ship gate the
SLURM chain depends on -- without touching the good MANIFEST.
"""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from data_processing import bundle as B
from data_processing.assemble import run_assemble
from data_processing.build_corpus import BatchCorrector
from data_processing.canonical import iter_shards
from data_processing.datasets import registry
from data_processing.datasets.base import DatasetSpec, FieldMap, Kind
from data_processing.distribute import DistributeConfig
from data_processing.prepare import run_prepare
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

    Same dual-format helper as test_assemble, so the scripted corrector works
    under either prompt family (compact JSON list or rich ``N. <<<t>>>`` lines).
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


@unittest.skipIf(np is None, "audio dependencies (numpy/scipy/soundfile) are not installed")
class TestStagedPipeline(unittest.TestCase):
    BATCH = 5

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.pool = self.dir / "pool"
        self.out = self.dir / "out"
        self.audio_root = self.dir / "audio"
        self.ledger = self.dir / "ledger.sqlite3"
        # An eval tree with one unrelated path: loaded by the Stage-4 leak
        # audit, leaking nothing.
        self.eval_root = self.dir / "v7.6"
        (self.eval_root / "ar").mkdir(parents=True)
        (self.eval_root / "ar" / "eval_ar.jsonl").write_text(
            json.dumps({"audio_filepath": "/not/shipped.wav"}) + "\n", encoding="utf-8")
        self.addCleanup(self._tmp.cleanup)

    # --- fixtures ------------------------------------------------------------
    def _fixture(self, name, seconds=2.0, sr=8_000):
        path = self.dir / "fixtures" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
        sf.write(str(path), (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr)
        return str(path)

    def _fake_post(self, payload):
        got = _parse_user_batch(payload["messages"][1]["content"])
        out = {"items": [{"i": n, "text": t + ".", "dialect": None, "confidence": 0.9}
                         for n, t in got]}
        return {"choices": [{"message": {"content": json.dumps(out, ensure_ascii=False)}}]}

    def _fake_spec(self, natives):
        """One external source whose stream is a loader over local fixtures.

        ``repo_id=None`` so ``spec.origin`` falls back to the name -- the
        provenance string the Stage-4 audit will report under ``sources``.
        """
        rows = [{"path": p, "text": f"{_AR} رقم {i}"} for i, p in enumerate(natives)]
        return DatasetSpec(
            name="fake_ext", lang="ar", kind=Kind.HF_STREAM, license="test",
            repo_id=None, fields=FieldMap(path="path", text="text", duration=None),
            loader=lambda spec: iter(rows),
        )

    # --- the pipeline --------------------------------------------------------
    def test_green_pipeline_ships_and_corruption_fails_the_audit(self):
        natives = [self._fixture(f"c{i}.wav") for i in range(self.BATCH)]

        # Stage 1: prepare -- metadata only, durations deferred for external rows.
        def fake_specs_for(lang, *, include_gated=True, external_only=False):
            return (self._fake_spec(natives),) if lang == "ar" else ()

        with mock.patch.object(registry, "specs_for", side_effect=fake_specs_for):
            prep = run_prepare(langs=("ar",), pool_dir=self.pool,
                               audio_root=self.audio_root, root=None,
                               quality=QualityConfig())
        self.assertEqual(prep["totals"]["emitted"], self.BATCH)
        self.assertNotIn("errored_sources", prep["totals"])

        # Stage 2: assemble -- best-first, LLM-corrected, clips materialized inline.
        with mock.patch.object(BatchCorrector, "_post", self._fake_post):
            asm = run_assemble(
                lang="ar", pool_dir=self.pool, out_dir=self.out, ledger=self.ledger,
                audio_root=self.audio_root, quality=QualityConfig(),
                distribute=DistributeConfig(batch_size=self.BATCH,
                                            max_per_source_fraction=1.0),
                fetcher=M.local_file_fetcher, fetch_workers=2,
                llm_url="http://localhost:8010/v1", llm_triage_only=False, batches=1,
            )
        self.assertEqual(asm["batches_written"], 1)
        self.assertEqual(asm["batch_labels"], ["b0000"])
        self.assertEqual(asm["llm"]["samples_corrected"], self.BATCH)
        self.assertEqual(asm["llm"]["samples_failed"], 0)
        self.assertEqual(asm["counters"]["prefetched"], self.BATCH)
        self.assertEqual(asm["counters"]["materialized"], self.BATCH)
        self.assertNotIn("materialize_failed", asm["counters"])
        self.assertEqual(asm["counters"].get("erasure_reverted", 0), 0)

        manifest = next(iter_shards(self.out / "ar", lang="ar"))[0]
        samples = [json.loads(line)
                   for line in manifest.read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual(len(samples), self.BATCH)
        for s in samples:
            self.assertTrue(s["text"].endswith("."))  # the corrector's output shipped
            self.assertTrue(M.verify(s["audio_filepath"], 16_000))  # 16 kHz mono FLAC
            self.assertAlmostEqual(s["duration"], 2.0, places=2)

        # Stage 3: materialize backfill -- every clip already verified, none written.
        stage3 = self.dir / "stage3.json"
        self.assertEqual(M.main([
            "--lang", "ar", "--out-dir", str(self.out), "--pool-dir", str(self.pool),
            "--audio-root", str(self.audio_root), "--workers", "4",
            "--report", str(stage3),
        ]), 0)
        payload3 = json.loads(stage3.read_text(encoding="utf-8"))
        self.assertEqual(payload3["verified"], self.BATCH)
        self.assertEqual(payload3["written"], 0)
        self.assertEqual(payload3["errors"], {})

        # Stage 4: bundle -- green audit, MANIFEST written.
        rep = B.run_bundle(out_dir=self.out, audio_root=self.audio_root,
                           root=self.eval_root, langs=("ar",), batch_size=self.BATCH)
        self.assertTrue(rep["audit"]["ok"], rep["audit"]["failures"])
        self.assertEqual(rep["k"], 1)
        self.assertTrue(rep["manifest_written"])
        stats = rep["languages"]["ar"]["b0000"]
        self.assertEqual(stats["rows"], self.BATCH)
        self.assertEqual(stats["external_rows"], self.BATCH)
        self.assertEqual(stats["internal_rows"], 0)
        self.assertEqual(stats["sources"], {"fake_ext": self.BATCH})
        self.assertEqual(stats["hours"], round(self.BATCH * 2.0 / 3600, 2))
        self.assertEqual(rep["audit"]["eval_paths_loaded"], 1)
        shipping = json.loads((self.out / "MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual(shipping["bundles"], 1)
        self.assertEqual(shipping["batch_labels"], {"ar": ["b0000"]})
        self.assertEqual(shipping["bundle_size"],
                         {"per_language": self.BATCH, "languages": ["ar"]})

        # Corrupt one shipped clip: the audit must fail, exit 1, and leave the
        # good MANIFEST untouched.
        Path(samples[0]["audio_filepath"]).write_bytes(b"this is not flac audio")
        good_manifest = (self.out / "MANIFEST.json").read_text(encoding="utf-8")
        report_path = self.dir / "bundle.json"
        self.assertEqual(B.main([
            "--out-dir", str(self.out), "--audio-root", str(self.audio_root),
            "--root", str(self.eval_root), "--langs", "ar",
            "--batch-size", str(self.BATCH), "--report", str(report_path),
        ]), 1)
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["k"], 1)  # discovery still sees a full batch
        self.assertFalse(payload["audit"]["ok"])
        self.assertTrue(any("not 16000 Hz mono" in f
                            for f in payload["audit"]["failures"]))
        self.assertFalse(payload["manifest_written"])
        self.assertEqual((self.out / "MANIFEST.json").read_text(encoding="utf-8"),
                         good_manifest)


if __name__ == "__main__":
    unittest.main()
