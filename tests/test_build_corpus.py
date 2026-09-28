"""Tests for the end-to-end build: config, the batched corrector, the stage chain.

The LLM stage is exercised against a scripted stand-in for ``_post``, never a
network. That keeps the tests deterministic and also pins the behaviour that
matters most here: whatever the endpoint does -- clean answer, partial answer,
junk, timeout -- the build must still emit a valid corpus, falling back to the
deterministic text rather than losing the sample.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from data_processing.assemble import main as assemble_main
from data_processing.build_corpus import (
    _STAGE_MAINS,
    BatchCorrector,
    BuildConfig,
    Builder,
    StageRecord,
    load_config,
    main,
    select_specs,
)
from data_processing.bundle import main as bundle_main
from data_processing.canonical import MANIFEST_KEYS, Meta, Sample, read_manifest, read_sidecar
from data_processing.datasets import registry
from data_processing.datasets.base import DatasetSpec, FieldMap, IngestStats, Kind
from data_processing.distribute import EXCLUDED_BATCH, DistributeConfig, SeenLedger
from data_processing.materialize import main as materialize_main
from data_processing.normalize import DiacriticPolicy
from data_processing.prepare import main as prepare_main
from data_processing.quality import QualityConfig

_AR = "سبحان الله كل شي تطور دفعة واحد في كل المجالات"
_AR_DIALECT = "عشان كده دلوقتي لازم نشتغل على المشروع"
_REPO = Path(__file__).resolve().parent.parent


def _rows(n: int, text: str = _AR, dur: str = "4.000") -> list[dict]:
    """The real v7.6 record shape: three keys, duration as a string."""
    return [{"audio_filepath": f"/data/a{i}.wav", "text": f"{text} {i}", "duration": dur}
            for i in range(n)]


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


def _stage_flags(module_main) -> set[str]:
    """Every flag a stage's argparse declares, via the parser its main builds."""
    parsers: list[argparse.ArgumentParser] = []
    real_init = argparse.ArgumentParser.__init__

    def spy(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        parsers.append(self)

    with mock.patch.object(argparse.ArgumentParser, "__init__", spy), \
            contextlib.redirect_stdout(io.StringIO()), \
            contextlib.suppress(SystemExit):  # argparse's --help exit
        module_main(["--help"])
    flags: set[str] = set()
    for parser in parsers:
        for action in parser._actions:  # introspection in a test
            flags.update(action.option_strings)
    return flags


class _Fixture(unittest.TestCase):
    """A synthetic internal-v7.6-shaped tree plus a spec that points at it."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "v7.6"
        self.out = Path(self._tmp.name) / "out"
        self.addCleanup(self._tmp.cleanup)

    def write(self, rel: str, rows: list[dict]) -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    def spec(self, lang: str = "ar", **kw) -> DatasetSpec:
        kw.setdefault("name", f"internal_v76_{lang}")
        kw.setdefault("paths", (f"{lang}/*.jsonl*",))
        kw.setdefault("exclude", ("_still_rejected_", "eval_"))
        return DatasetSpec(lang=lang, kind=Kind.LOCAL_JSONL, license="internal",
                           fields=FieldMap(), **kw)

    def cfg(self, **kw) -> BuildConfig:
        kw.setdefault("out_dir", str(self.out))
        kw.setdefault("ledger", str(self.root.parent / "ledger.sqlite3"))
        kw.setdefault("root", str(self.root))
        kw.setdefault("batches_per_lang", 0)
        kw.setdefault("distribute", DistributeConfig(batch_size=100, max_per_source_fraction=1.0))
        return BuildConfig(**kw)


class TestLoadConfig(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _write(self, text: str, name: str = "c.yaml") -> Path:
        p = self.tmp / name
        p.write_text(text, encoding="utf-8")
        return p

    def test_missing_file_raises_rather_than_defaulting(self):
        with self.assertRaises(FileNotFoundError):
            load_config(self.tmp / "absent.yaml")

    def test_build_block_is_unwrapped(self):
        cfg = load_config(self._write("build:\n  out_dir: /x\n  batches_per_lang: 3\n"))
        self.assertEqual(cfg.out_dir, "/x")
        self.assertEqual(cfg.batches_per_lang, 3)

    def test_flat_mapping_also_works(self):
        cfg = load_config(self._write("out_dir: /y\n"))
        self.assertEqual(cfg.out_dir, "/y")

    def test_nested_quality_and_distribute_become_dataclasses(self):
        cfg = load_config(self._write(
            "build:\n"
            "  quality:\n    min_duration: 1.5\n    max_duration: 20.0\n"
            "  distribute:\n    batch_size: 500\n    max_per_text: 3\n"
        ))
        self.assertIsInstance(cfg.quality, QualityConfig)
        self.assertIsInstance(cfg.distribute, DistributeConfig)
        self.assertEqual(cfg.quality.min_duration, 1.5)
        self.assertEqual(cfg.quality.max_duration, 20.0)
        self.assertEqual(cfg.distribute.batch_size, 500)
        self.assertEqual(cfg.distribute.max_per_text, 3)

    def test_unknown_nested_key_is_an_error_not_a_silent_ignore(self):
        with self.assertRaises(TypeError):
            load_config(self._write("build:\n  distribute:\n    batch_siz: 10\n"))

    def test_unknown_top_level_key_warns_and_is_ignored(self):
        with self.assertLogs("data_processing.build_corpus", level="WARNING") as logs:
            cfg = load_config(self._write("build:\n  out_dir: /x\n  nonsense: 1\n"))
        self.assertEqual(cfg.out_dir, "/x")
        self.assertIn("nonsense", "\n".join(logs.output))

    def test_langs_accepts_a_comma_string_or_a_list(self):
        self.assertEqual(load_config(self._write("build:\n  langs: ar,zh\n")).langs, ("ar", "zh"))
        self.assertEqual(load_config(self._write("build:\n  langs: [ar, ml]\n")).langs, ("ar", "ml"))

    def test_diacritic_policy_becomes_the_enum(self):
        cfg = load_config(self._write("build:\n  diacritic_policy: strip_all\n"))
        self.assertIs(cfg.diacritic_policy, DiacriticPolicy.STRIP_ALL)

    def test_bad_diacritic_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            load_config(self._write("build:\n  diacritic_policy: sometimes\n"))

    def test_empty_file_gives_defaults(self):
        cfg = load_config(self._write(""))
        self.assertEqual(cfg.langs, registry.LANGUAGES)
        self.assertEqual(cfg.diacritic_policy, DiacriticPolicy.CRITICAL_ONLY)

    def test_shipped_corpus_yaml_loads(self):
        cfg = load_config(_REPO / "configs" / "corpus.yaml")
        self.assertEqual(cfg.distribute.batch_size, 100_000)
        self.assertEqual(cfg.distribute.max_per_text, 2)
        self.assertTrue(cfg.exclude_eval)
        self.assertIsNone(cfg.llm_url)  # tokenless by default
        self.assertEqual(cfg.llm_batch, 16)
        self.assertEqual(cfg.llm_concurrency, 64)
        self.assertEqual(cfg.llm_prompt, "rich")

    def test_shipped_stage_sections_only_declare_flags_the_stage_reads(self):
        """Wiring honesty for the staged sections: the dispatcher feeds each
        section as prepended flags, so a key the stage's argparse does not
        declare would be a silent no-op -- a knob nothing reads is worse than
        no knob (the same contract the ``build:`` check above enforces)."""
        raw = yaml.safe_load((_REPO / "configs" / "corpus.yaml").read_text(encoding="utf-8"))
        stage_mains = {
            "prepare": prepare_main,
            "assemble": assemble_main,
            "materialize": materialize_main,
            "bundle": bundle_main,
        }
        for section, module_main in stage_mains.items():
            flags = _stage_flags(module_main)
            for key in raw.get(section, {}):
                self.assertIn("--" + key.replace("_", "-"), flags,
                              f"corpus.yaml {section}: {key!r} is not a flag of that stage")
        # paths: is driver/env context (several need per-language joins), so
        # it is never injected as flags -- but it stays closed-world: adding a
        # path key must be a conscious act mirrored in scripts/corpus/env.sh.
        self.assertEqual(set(raw.get("paths", {})),
                         {"internal_root", "pool_dir", "audio_root", "out_dir",
                          "ledger_dir", "logs"})

    def test_shipped_corpus_yaml_has_no_knob_that_nothing_reads(self):
        """A config option that silently does nothing is worse than no option.

        ``erasure_penalty`` and ``commit_every`` were both declared, settable
        from YAML, and read by no code path at all.
        """
        raw = json.loads(json.dumps(
            __import__("yaml").safe_load((_REPO / "configs" / "corpus.yaml").read_text())["build"],
            default=str))
        top = set(BuildConfig.__slots__)
        for key in raw:
            if key in ("quality", "distribute"):
                continue
            self.assertIn(key, top, f"corpus.yaml sets unknown build key {key!r}")
        nested = {"quality": set(QualityConfig.__slots__), "distribute": set(DistributeConfig.__slots__)}
        for block, fields in nested.items():
            for key in raw.get(block, {}):
                self.assertIn(key, fields, f"corpus.yaml sets unknown {block} key {key!r}")


class TestSelectSpecs(unittest.TestCase):
    def test_only_sources_filters(self):
        cfg = BuildConfig(only_sources=("internal_v76_ar",))
        specs = select_specs(cfg, "ar")
        self.assertEqual([s.name for s in specs], ["internal_v76_ar"])

    def test_no_gated_means_no_token_needed(self):
        cfg = BuildConfig(include_gated=False)
        for spec in select_specs(cfg, "ml"):
            self.assertFalse(spec.gated)

    def test_unknown_only_source_selects_nothing(self):
        self.assertEqual(select_specs(BuildConfig(only_sources=("nope",)), "ar"), ())


class TestBatchCorrector(unittest.TestCase):
    def _cfg(self, **kw) -> BuildConfig:
        kw.setdefault("llm_batch", 2)
        kw.setdefault("llm_concurrency", 2)
        return BuildConfig(**kw)

    def test_disabled_without_a_url(self):
        self.assertFalse(BatchCorrector(self._cfg()).enabled)
        self.assertTrue(BatchCorrector(self._cfg(llm_url="http://h:8000/v1")).enabled)

    def test_system_prompt_carries_the_preservation_rules(self):
        c = BatchCorrector(self._cfg(llm_url="x", llm_prompt="compact"))
        prompt = c._system_prompt("ar", "egyptian")
        self.assertIn("NEVER add diacritics", prompt)
        self.assertIn("NEVER convert colloquial", prompt)
        self.assertIn("NEVER paraphrase", prompt)
        self.assertIn('"items"', prompt)

    def test_system_prompt_names_the_dialect_it_is_protecting(self):
        c = BatchCorrector(self._cfg(llm_url="x", llm_prompt="compact"))
        self.assertNotEqual(c._system_prompt("ar", "egyptian"), c._system_prompt("ar", "levantine"))
        self.assertIn("egyptian", c._system_prompt("ar", "egyptian").lower())

    @staticmethod
    def _resp(items: list[dict]) -> dict:
        return {"choices": [{"message": {"content": json.dumps({"items": items}, ensure_ascii=False)}}]}

    def test_results_stay_index_aligned_with_the_input(self):
        c = BatchCorrector(self._cfg(llm_url="http://h/v1"))
        with mock.patch.object(c, "_post", return_value=self._resp([
            {"i": 1, "text": "الثاني", "dialect": "egyptian", "confidence": 0.9},
            {"i": 0, "text": "الأول", "dialect": "msa", "confidence": 0.8},
        ])):
            out = c._call("ar", [("واحد", None), ("اثنين", None)])
        self.assertEqual([o["text"] for o in out], ["الأول", "الثاني"])
        self.assertEqual(out[1]["dialect"], "egyptian")
        self.assertEqual(c.calls, 1)
        self.assertEqual(c.samples, 2)

    def test_items_the_model_omitted_keep_their_input_text(self):
        c = BatchCorrector(self._cfg(llm_url="http://h/v1"))
        with mock.patch.object(c, "_post", return_value=self._resp([
            {"i": 0, "text": "مصحح", "dialect": None, "confidence": 0.5}])):
            out = c._call("ar", [("واحد", None), ("اثنين", None)])
        self.assertEqual(out[0]["text"], "مصحح")
        self.assertEqual(out[1]["text"], "اثنين")  # untouched, not lost

    def test_out_of_range_indices_and_junk_rows_are_ignored(self):
        c = BatchCorrector(self._cfg(llm_url="http://h/v1"))
        with mock.patch.object(c, "_post", return_value=self._resp([
            {"i": 99, "text": "خارج"}, "not a dict", {"i": -1, "text": "سالب"},
            {"text": "بلا فهرس"},
        ])):
            out = c._call("ar", [("واحد", None)])
        self.assertEqual(out[0]["text"], "واحد")

    def test_a_failed_call_falls_back_and_counts_the_failure(self):
        c = BatchCorrector(self._cfg(llm_url="http://h/v1"))
        with mock.patch.object(c, "_post", side_effect=TimeoutError("boom")):
            out = c._call("ar", [("واحد", None), ("اثنين", None)])
        self.assertEqual([o["text"] for o in out], ["واحد", "اثنين"])
        self.assertEqual(c.failures, 2)
        self.assertEqual(c.calls, 0)

    def test_malformed_json_response_falls_back(self):
        c = BatchCorrector(self._cfg(llm_url="http://h/v1"))
        with mock.patch.object(c, "_post", return_value={"choices": [{"message": {"content": "{"}}]}):
            out = c._call("ar", [("واحد", None)])
        self.assertEqual(out[0]["text"], "واحد")

    def test_correct_chunks_but_preserves_global_order(self):
        c = BatchCorrector(self._cfg(llm_url="http://h/v1", llm_batch=2, llm_concurrency=4))
        items = [(f"نص {i}", None) for i in range(7)]

        def fake_post(payload):
            got = _parse_user_batch(payload["messages"][1]["content"])
            return self._resp([{"i": n, "text": f"fix:{t}", "confidence": 1.0}
                               for n, t in got])

        with mock.patch.object(c, "_post", side_effect=fake_post):
            out = c.correct("ar", items)
        self.assertEqual(len(out), 7)
        self.assertEqual([o["text"] for o in out], [f"fix:نص {i}" for i in range(7)])

    def test_request_disables_thinking_and_pins_temperature(self):
        c = BatchCorrector(self._cfg(llm_url="http://h/v1"))
        seen = {}

        def fake_post(payload):
            seen.update(payload)
            return self._resp([])

        with mock.patch.object(c, "_post", side_effect=fake_post):
            c._call("ar", [("نص", None)])
        self.assertEqual(seen["temperature"], 0)
        self.assertEqual(seen["chat_template_kwargs"], {"enable_thinking": False})

    def test_stats_shape(self):
        stats = BatchCorrector(self._cfg()).stats()
        self.assertFalse(stats["enabled"])
        self.assertEqual(stats["calls"], 0)
        self.assertTrue(stats["triage_only"])


class TestApplyLLM(_Fixture):
    def _rec(self, text: str = _AR_DIALECT, lang: str = "ar") -> StageRecord:
        sample = Sample(audio_filepath="/a.wav", duration=4.0, text=text, lang=lang)
        meta = Meta(audio_filepath="/a.wav", normalized_text=text, source_text=text)
        return StageRecord(sample=sample, meta=meta, source="s", accent_label="egyptian")

    def _builder(self, cfg: BuildConfig | None = None) -> Builder:
        cfg = cfg or self.cfg()
        ledger = SeenLedger(cfg.ledger, buffer_claims=False)
        self.addCleanup(ledger.close)
        return Builder("ar", cfg, ledger, BatchCorrector(cfg))

    def test_a_clean_correction_is_kept(self):
        b = self._builder()
        rec = self._rec()
        b._apply_llm(rec, {"text": "عشان كده دلوقتي لازم نشتغل على المشروع.", "confidence": 0.9})
        self.assertTrue(rec.meta.final_text.endswith("."))
        self.assertIn("عشان", rec.meta.final_text)  # dialect preserved
        self.assertEqual(rec.meta.stages["llm"], "ok")

    def test_an_empty_response_keeps_the_normalized_text(self):
        b = self._builder()
        rec = self._rec()
        b._apply_llm(rec, {"text": "   "})
        self.assertEqual(rec.meta.final_text, _AR_DIALECT)
        self.assertEqual(rec.meta.stages["llm"], "empty_response")

    def test_dialect_erasure_is_reverted(self):
        """The LLM rewrote Egyptian into MSA. Refuse the rewrite, keep ours."""
        b = self._builder()
        rec = self._rec()
        b._apply_llm(rec, {"text": "لأن لذلك الآن لازم نشتغل على المشروع"})
        self.assertEqual(rec.meta.final_text, _AR_DIALECT)
        self.assertIn("عشان", rec.meta.final_text)
        self.assertEqual(rec.meta.stages["erasure"], "damaged")
        self.assertEqual(b.stage_counts["erasure_reverted"], 1)
        self.assertGreater(rec.meta.metrics["erasure_severity"], 0.0)

    def test_erasure_is_measured_against_the_normalized_text(self):
        """Not against the raw source: that would bill the model for our own
        deterministic transforms and inflate the damage count."""
        b = self._builder()
        rec = self._rec()
        rec.meta.normalized_text = _AR_DIALECT
        rec.sample = Sample(audio_filepath="/a.wav", duration=4.0, text="RAW   SOURCE", lang="ar")
        b._apply_llm(rec, {"text": _AR_DIALECT + "."})  # identical apart from a period
        self.assertEqual(rec.meta.stages["erasure"], "ok")
        self.assertEqual(b.stage_counts["erasure_reverted"], 0)

    def test_accent_is_reconciled_not_overwritten(self):
        b = self._builder()
        rec = self._rec()
        b._apply_llm(rec, {"text": _AR_DIALECT, "dialect": "msa", "confidence": 0.95})
        self.assertNotEqual(rec.meta.accent, "msa")


class TestPrepare(_Fixture):
    def _builder(self, **cfg_kw) -> Builder:
        cfg = self.cfg(**cfg_kw)
        ledger = SeenLedger(cfg.ledger, buffer_claims=False)
        self.addCleanup(ledger.close)
        return Builder("ar", cfg, ledger, BatchCorrector(cfg))

    def test_a_good_row_survives_with_its_audit_trail(self):
        b = self._builder()
        stats = IngestStats(spec_name="s")
        rec = b._prepare(self.spec(), _rows(1)[0], stats)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.meta.normalized_text, f"{_AR} 0")
        self.assertIsNotNone(rec.meta.quality)
        self.assertIn("normalize", rec.meta.stages)
        self.assertIn("accent", rec.meta.stages)

    def test_missing_duration_is_counted_separately_from_missing_text(self):
        b = self._builder()
        stats = IngestStats(spec_name="s")
        self.assertIsNone(b._prepare(self.spec(), {"audio_filepath": "/x.wav", "text": _AR}, stats))
        self.assertEqual(stats.duration_missing, 1)
        self.assertEqual(b.stage_counts["skip_no_duration"], 1)
        self.assertIsNone(b._prepare(self.spec(), {"audio_filepath": "/y.wav", "duration": 1.0}, stats))
        self.assertEqual(stats.skipped_no_text, 1)
        self.assertIsNone(b._prepare(self.spec(), {"text": _AR, "duration": 1.0}, stats))
        self.assertEqual(stats.skipped_no_path, 1)

    def test_a_gate_rejection_is_counted_with_its_reason(self):
        # duration gates are off by default; arm them to exercise the reject path
        b = self._builder(quality=QualityConfig(gate_duration=True))
        stats = IngestStats(spec_name="s")
        rec = b._prepare(self.spec(), {"audio_filepath": "/z.wav", "text": _AR, "duration": 999.0},
                         stats)
        self.assertIsNone(rec)
        self.assertEqual(b.stage_counts["reject_gate"], 1)
        self.assertTrue(b.reject_reasons)

    def test_triage_matches_the_documented_rule(self):
        """needs_llm == rules fired OR accent unknown OR quality below 0.9.

        ~18% of real traffic meets it, which is the difference between the token
        budget being hours and being days.
        """
        b = self._builder()
        stats = IngestStats(spec_name="s")
        rows = [
            ("clean", {"audio_filepath": "/c.wav", "text": _AR, "duration": 4.0}),
            ("digits", {"audio_filepath": "/d.wav", "text": "٣ تفاحات و ٤ برتقالات", "duration": 4.0}),
            ("lowq", {"audio_filepath": "/l.wav", "text": _AR, "duration": 8.0}),
        ]
        checked = 0
        for label, row in rows:
            rec = b._prepare(self.spec(), row, stats)
            if rec is None:
                continue
            checked += 1
            expected = (rec.meta.stages["normalize"] != "noop"
                        or rec.meta.accent == "unknown"
                        or rec.meta.quality < 0.9)
            self.assertEqual(rec.needs_llm, expected, f"{label}: {rec.meta.stages}")
        self.assertGreater(checked, 0)


class TestBuilderRun(_Fixture):
    def test_deterministic_run_emits_a_canonical_corpus(self):
        self.write("ar/train_a.jsonl", _rows(12))
        cfg = self.cfg(langs=("ar",))
        ledger = SeenLedger(cfg.ledger, buffer_claims=False)
        self.addCleanup(ledger.close)
        builder = Builder("ar", cfg, ledger, BatchCorrector(cfg))
        report = builder.run([self.spec()])

        self.assertGreater(report["rows_read"], 0)
        self.assertEqual(report["batches_written"], 1)
        manifest, sidecar = (Path(p) for p in report["files"][0])
        self.assertTrue(manifest.exists())

        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 12)
        for row in rows:
            self.assertEqual(tuple(row), MANIFEST_KEYS)
            self.assertEqual(row["lang"], "ar")
            self.assertIsInstance(row["duration"], float)
        self.assertEqual(len({r["audio_filepath"] for r in rows}), 12)

        metas = list(read_sidecar(sidecar).values())
        self.assertEqual(len(metas), 12)
        self.assertEqual([s.audio_filepath for s in read_manifest(manifest)],
                         [m.audio_filepath for m in metas])
        self.assertTrue(all(m.normalized_text for m in metas))
        self.assertTrue(all(m.stages.get("llm") == "disabled" for m in metas))

    def test_eval_paths_never_reach_a_batch(self):
        self.write("ar/train_a.jsonl", _rows(5))
        self.write("ar/eval_ar.jsonl",
                   [{"audio_filepath": "/data/a0.wav", "text": _AR, "duration": "4.0"}])
        cfg = self.cfg(langs=("ar",))
        with SeenLedger(cfg.ledger, buffer_claims=False) as ledger:
            ledger.load_eval_exclusions(cfg.root, cfg.langs)
            self.assertEqual(ledger.batch_of("/data/a0.wav"), EXCLUDED_BATCH)
            report = Builder("ar", cfg, ledger, BatchCorrector(cfg)).run([self.spec()])
        paths = {json.loads(line)["audio_filepath"]
                 for line in Path(report["files"][0][0]).read_text(encoding="utf-8").splitlines()}
        self.assertNotIn("/data/a0.wav", paths)
        self.assertEqual(len(paths), 4)
        self.assertEqual(report["distributor"]["outcomes"]["excluded"], 1)

    def test_recovery_shards_are_not_ingested(self):
        self.write("ar/train_a.jsonl", _rows(3))
        self.write("ar/train_a_still_rejected_p0000.jsonl",
                   [{"audio_filepath": "/data/dup.wav", "text": _AR, "duration": "4.0"}])
        cfg = self.cfg(langs=("ar",))
        with SeenLedger(cfg.ledger, buffer_claims=False) as ledger:
            report = Builder("ar", cfg, ledger, BatchCorrector(cfg)).run([self.spec()])
        self.assertEqual(report["rows_read"], 3)

    def test_dry_run_writes_nothing_but_still_reports(self):
        self.write("ar/train_a.jsonl", _rows(4))
        cfg = self.cfg(langs=("ar",), dry_run=True)
        with SeenLedger(cfg.ledger, buffer_claims=False) as ledger:
            report = Builder("ar", cfg, ledger, BatchCorrector(cfg)).run([self.spec()])
        self.assertEqual(report["files"], [])
        self.assertFalse(self.out.exists())

    def test_dry_run_leaves_the_configured_ledger_untouched(self):
        """Regression: --dry-run must not persist claims.

        offer() claims every path it processes and main() opens the ledger
        unbuffered, so a dry run aimed at the production ledger would mark those
        paths seen -- and a later REAL run would skip them as duplicates and
        emit nothing. main() swaps in an ephemeral in-memory ledger to stop that.
        """
        self.write("ar/train_a.jsonl", _rows(4))
        prod = self.root.parent / "production.sqlite3"
        argv = ["--root", str(self.root), "--langs", "ar", "--batches", "0",
                "--out-dir", str(self.out), "--ledger", str(prod)]
        with mock.patch("data_processing.build_corpus.select_specs",
                        return_value=(self.spec(),)):
            self.assertEqual(main([*argv, "--dry-run"]), 0)
        self.assertFalse(prod.exists(), "dry-run must not create the ledger file")
        self.assertFalse(self.out.exists(), "dry-run must not write shards")

        # The same command without --dry-run persists ledger AND shards.
        with mock.patch("data_processing.build_corpus.select_specs",
                        return_value=(self.spec(),)):
            self.assertEqual(main(argv), 0)
        self.assertTrue(prod.exists(), "real run must persist the ledger")
        with SeenLedger(prod, buffer_claims=False) as led:
            self.assertEqual(led.batch_of("/data/a0.wav"), "b0000")

    def test_limit_stops_early(self):
        self.write("ar/train_a.jsonl", _rows(50))
        cfg = self.cfg(langs=("ar",))
        with SeenLedger(cfg.ledger, buffer_claims=False) as ledger:
            report = Builder("ar", cfg, ledger, BatchCorrector(cfg)).run([self.spec()], limit=10)
        self.assertEqual(report["rows_read"], 10)

    def test_batch_size_is_honoured_end_to_end(self):
        self.write("ar/train_a.jsonl", _rows(12))
        cfg = self.cfg(langs=("ar",),
                       distribute=DistributeConfig(batch_size=5, max_per_source_fraction=1.0))
        with SeenLedger(cfg.ledger, buffer_claims=False) as ledger:
            report = Builder("ar", cfg, ledger, BatchCorrector(cfg)).run([self.spec()])
        sizes = [len(Path(p[0]).read_text(encoding="utf-8").splitlines()) for p in report["files"]]
        self.assertEqual(sizes, [5, 5, 2])
        self.assertEqual(report["batches_written"], 3)
        self.assertEqual(report["carried_over_partial"], 0)

    def test_draining_closes_the_partial_while_a_bounded_run_does_not(self):
        """``batches_per_lang == 0`` means "until the sources run out", so the
        tail is written. A bounded run leaves it open rather than landing a short
        shard that silently breaks the 100k-per-language contract."""
        self.write("ar/train_a.jsonl", _rows(7))
        dist = DistributeConfig(batch_size=5, max_per_source_fraction=1.0)
        drained = self.cfg(langs=("ar",), batches_per_lang=0, distribute=dist)
        with SeenLedger(drained.ledger, buffer_claims=False) as ledger:
            report = Builder("ar", drained, ledger, BatchCorrector(drained)).run([self.spec()])
        self.assertEqual(report["carried_over_partial"], 0)
        self.assertEqual(report["batches_written"], 2)

        bounded_root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(bounded_root, ignore_errors=True))
        bounded = self.cfg(langs=("ar",), batches_per_lang=1, distribute=dist,
                           out_dir=str(bounded_root / "out"),
                           ledger=str(bounded_root / "l.sqlite3"))
        with SeenLedger(bounded.ledger, buffer_claims=False) as ledger:
            report = Builder("ar", bounded, ledger, BatchCorrector(bounded)).run([self.spec()])
        self.assertEqual(report["batches_written"], 1)
        self.assertEqual(report["carried_over_partial"], 2)

    def test_report_is_json_serialisable(self):
        self.write("ar/train_a.jsonl", _rows(3))
        cfg = self.cfg(langs=("ar",))
        with SeenLedger(cfg.ledger, buffer_claims=False) as ledger:
            report = Builder("ar", cfg, ledger, BatchCorrector(cfg)).run([self.spec()])
        json.dumps(report, default=str)


class TestMainCLI(_Fixture):
    def test_end_to_end_run_writes_shards_and_a_report(self):
        self.write("ar/train_a.jsonl", _rows(8))
        self.write("ar/eval_ar.jsonl",
                   [{"audio_filepath": "/data/a0.wav", "text": _AR, "duration": "4.0"}])
        report_path = self.root.parent / "report.json"
        rc = main(["--langs", "ar", "--root", str(self.root), "--out-dir", str(self.out),
                   "--ledger", str(self.root.parent / "l.sqlite3"), "--batches", "0",
                   "--only", "internal_v76_ar", "--report", str(report_path)])
        self.assertEqual(rc, 0)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertIn("ar", report["languages"])
        self.assertEqual(report["languages"]["ar"]["batches_written"], 1)
        self.assertEqual(report["eval_exclusions"], {"ar": 1})
        self.assertFalse(report["llm"]["enabled"])
        self.assertEqual(report["ledger"]["excluded"], 1)

    def test_cli_flags_override_the_config_file(self):
        cfg_path = self.root.parent / "c.yaml"
        cfg_path.write_text("build:\n  out_dir: /from/yaml\n  batches_per_lang: 9\n"
                            "  langs: [zh]\n", encoding="utf-8")
        self.write("ar/train_a.jsonl", _rows(2))
        report_path = self.root.parent / "r.json"
        main(["--config", str(cfg_path), "--langs", "ar", "--out-dir", str(self.out),
              "--ledger", str(self.root.parent / "l.sqlite3"), "--batches", "0",
              "--root", str(self.root), "--only", "internal_v76_ar", "--report", str(report_path)])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["config"]["out_dir"], str(self.out))
        self.assertEqual(report["config"]["batches_per_lang"], 0)
        self.assertEqual(list(report["languages"]), ["ar"])

    def test_a_language_with_no_selected_source_is_reported_not_silent(self):
        rc = main(["--langs", "ar", "--root", str(self.root), "--out-dir", str(self.out),
                   "--ledger", str(self.root.parent / "l.sqlite3"), "--only", "nope",
                   "--report", str(self.root.parent / "r.json")])
        self.assertEqual(rc, 0)
        report = json.loads((self.root.parent / "r.json").read_text(encoding="utf-8"))
        self.assertEqual(report["languages"]["ar"]["error"], "no sources selected")

    def test_gzip_flag_produces_a_readable_gzipped_shard(self):
        self.write("ar/train_a.jsonl", _rows(4))
        main(["--langs", "ar", "--root", str(self.root), "--out-dir", str(self.out),
              "--ledger", str(self.root.parent / "l.sqlite3"), "--batches", "0", "--gzip",
              "--only", "internal_v76_ar", "--report", str(self.root.parent / "r.json")])
        shards = [s for s in self.out.rglob("*.jsonl.gz") if ".meta." not in s.name]
        self.assertEqual(len(shards), 1)
        self.assertEqual(len(list(read_manifest(shards[0]))), 4)


class TestDispatcher(unittest.TestCase):
    """The staged entry point: ``build_corpus <stage>`` routes to the stage
    module's own ``main`` with literal argv passthrough, and feeds that
    stage's ``configs/corpus.yaml`` section as prepended defaults (explicit
    CLI flags win via argparse's last-occurrence rule)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def _empty_config(self) -> Path:
        path = self.dir / "empty.yaml"
        path.write_text("{}\n", encoding="utf-8")
        return path

    def test_each_subcommand_routes_with_literal_passthrough(self):
        empty = self._empty_config()
        for stage, (module_name, _attr) in _STAGE_MAINS.items():
            mod = importlib.import_module(module_name)
            with mock.patch.object(mod, "main", return_value=7) as fake:
                code = main(["--config", str(empty), stage, "--whatever", "x"])
            self.assertEqual(code, 7, stage)  # exit codes are the audit gates
            fake.assert_called_once_with(["--whatever", "x"])

    def test_unknown_subcommand_falls_to_the_legacy_parser(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["definitely-not-a-stage", "--whatever"])
        self.assertEqual(ctx.exception.code, 2)

    def test_no_subcommand_runs_the_legacy_build(self):
        # --config with no stage after it must reach the legacy parser (an
        # empty mapping parses to BuildConfig defaults); Builder is mocked so
        # no source is actually read, and the eval root points nowhere so the
        # exclusion load is a fast no-op.
        with mock.patch("data_processing.build_corpus.Builder") as builder:
            code = main(["--config", str(self._empty_config()), "--ledger", ":memory:",
                         "--root", "/nonexistent-eval-root"])
        self.assertEqual(code, 0)
        self.assertEqual(builder.call_count, len(registry.LANGUAGES))

    def test_stage_defaults_prepend_and_cli_flags_win(self):
        cfg = self.dir / "staged.yaml"
        cfg.write_text("prepare:\n  jobs: 3\n  min_richness: 0.25\n", encoding="utf-8")
        with mock.patch("data_processing.prepare.main", return_value=0) as fake:
            main(["--config", str(cfg), "prepare", "--jobs", "5"])
        # YAML defaults first, the user's flag last: argparse's last-occurrence
        # rule is what makes "CLI flags still win" hold.
        fake.assert_called_once_with(["--jobs", "3", "--min-richness", "0.25",
                                      "--jobs", "5"])

    def test_default_staged_config_is_the_shipped_corpus_yaml(self):
        with mock.patch("data_processing.prepare.main", return_value=0) as fake:
            main(["prepare", "--pool-dir", "/p"])
        fake.assert_called_once_with(["--jobs", "4", "--min-richness", "0.0",
                                      "--pool-dir", "/p"])


if __name__ == "__main__":
    unittest.main()
