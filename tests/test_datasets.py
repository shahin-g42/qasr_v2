"""Tests for the ingest layer: specs, registry, local loaders, Phase A stream.

No network. Every Hub-touching function is exercised through a local stand-in,
because a test that needs a token and a download is a test that does not run.
"""

from __future__ import annotations

import gzip
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import yaml

from data_processing.canonical import Meta, Sample
from data_processing.datasets import registry
from data_processing.datasets.base import (
    DEFAULT_MAX_SAMPLES,
    DatasetSpec,
    FieldMap,
    IngestStats,
    Kind,
    require_datasets,
)
from data_processing.datasets.local import (
    AUDIO_SUFFIXES,
    duration_coverage,
    effective_root,
    expand_paths,
    is_excluded,
    iter_local,
    iter_local_audio,
)
from data_processing.datasets.stream import (
    ingest,
    interleave,
    stats_report,
    stream_metadata,
    to_sample,
)

_F = FieldMap()
_AR_TEXT = "سبحان الله كل شي تطور"


def _spec(**kw) -> DatasetSpec:
    kw.setdefault("name", "t_spec")
    kw.setdefault("lang", "ar")
    kw.setdefault("kind", Kind.LOCAL_JSONL)
    kw.setdefault("license", "internal")
    kw.setdefault("fields", _F)
    return DatasetSpec(**kw)


class _TmpTree(unittest.TestCase):
    """A synthetic stand-in for the internal v7.6 manifest tree."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def write(self, rel: str, rows: list[dict], gzipped: bool = False) -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if gzipped else open
        with opener(path, "wt", encoding="utf-8") as fh:  # type: ignore[operator]
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    def rec(self, i: int, text: str | None = None) -> dict:
        """The real v7.6 record shape: three keys, duration as a STRING."""
        return {
            "audio_filepath": f"/data/{self.root.name}_{i}.wav",
            "text": text if text is not None else f"{_AR_TEXT} {i}",
            "duration": f"{2.0 + i * 0.5:.3f}",
        }


class TestFieldMapAndSpec(unittest.TestCase):
    def test_duration_column_absent_means_a_probe_is_needed(self):
        self.assertTrue(FieldMap(duration=None).needs_duration_probe)
        self.assertFalse(FieldMap().needs_duration_probe)

    def test_canonical_field_names(self):
        self.assertEqual((_F.text, _F.duration, _F.path), ("text", "duration", "audio_filepath"))

    def test_audio_filepath_uses_the_mapped_column(self):
        spec = _spec(fields=FieldMap(path="path"))
        self.assertEqual(spec.audio_filepath({"path": "/a.wav"}), "/a.wav")

    def test_audio_filepath_prefixes_relative_but_not_absolute(self):
        spec = _spec(path_prefix="/mnt/data")
        self.assertEqual(spec.audio_filepath({"audio_filepath": "a.wav"}), "/mnt/data/a.wav")
        self.assertEqual(spec.audio_filepath({"audio_filepath": "/abs/a.wav"}), "/abs/a.wav")

    def test_audio_filepath_unwraps_a_decoded_hf_audio_feature(self):
        """Phase A must never touch the waveform array."""
        spec = _spec(fields=FieldMap(path="audio"))
        row = {"audio": {"path": "/hf/a.wav", "array": [0.0] * 16000}}
        self.assertEqual(spec.audio_filepath(row), "/hf/a.wav")
        self.assertEqual(spec.audio_filepath({"audio": "/hf/b.wav"}), "/hf/b.wav")

    def test_audio_filepath_none_when_absent(self):
        self.assertIsNone(_spec().audio_filepath({"text": "x"}))

    def test_path_from_overrides_everything(self):
        spec = _spec(path_from=lambda r: f"{r['__url__']}#{r['__key__']}")
        self.assertEqual(
            spec.audio_filepath({"__url__": "shard.tar", "__key__": "0001"}), "shard.tar#0001"
        )

    def test_duration_coerces_strings_and_rejects_junk(self):
        spec = _spec()
        self.assertEqual(spec.duration_of({"duration": "2.396"}), 2.396)
        self.assertEqual(spec.duration_of({"duration": 3}), 3.0)
        self.assertIsNone(spec.duration_of({"duration": None}))
        self.assertIsNone(spec.duration_of({"duration": "abc"}))
        self.assertIsNone(spec.duration_of({}))
        self.assertIsNone(_spec(fields=FieldMap(duration=None)).duration_of({"duration": 1.0}))

    def test_text_falls_back_to_sentence(self):
        self.assertEqual(_spec().text_of({"sentence": "hello"}), "hello")
        self.assertIsNone(_spec().text_of({"nope": "x"}))
        self.assertEqual(_spec().text_of({"text": 42}), "42")

    def test_origin_prefers_repo_id(self):
        self.assertEqual(_spec(repo_id="google/fleurs").origin, "google/fleurs")
        self.assertEqual(_spec().origin, "t_spec")

    def test_defaults(self):
        spec = _spec()
        self.assertEqual(spec.max_samples, DEFAULT_MAX_SAMPLES)
        self.assertEqual(spec.split, "train")
        self.assertFalse(spec.gated)
        self.assertFalse(spec.verified)


class TestRegistry(unittest.TestCase):
    def test_names_are_unique(self):
        names = [s.name for s in registry.all_specs()]
        self.assertEqual(len(names), len(set(names)))

    def test_languages_are_the_five_from_the_spec(self):
        self.assertEqual(registry.LANGUAGES, ("ar", "en", "zh", "hi", "ml"))
        for spec in registry.all_specs():
            self.assertIn(spec.lang, registry.LANGUAGES)

    def test_by_name_roundtrips_and_rejects_unknown(self):
        self.assertEqual(registry.by_name("internal_v76_ar").name, "internal_v76_ar")
        with self.assertRaises(KeyError) as ctx:
            registry.by_name("not_a_dataset")
        self.assertIn("known", str(ctx.exception))

    def test_specs_for_rejects_an_unsupported_language(self):
        with self.assertRaises(ValueError):
            registry.specs_for("fr")

    def test_include_gated_false_needs_no_token(self):
        for lang in registry.LANGUAGES:
            for spec in registry.specs_for(lang, include_gated=False):
                self.assertFalse(spec.gated, spec.name)
                self.assertIsNot(spec.kind, Kind.HF_GATED, spec.name)

    def test_external_only_drops_local_sources(self):
        for lang in registry.LANGUAGES:
            for spec in registry.specs_for(lang, external_only=True):
                self.assertNotIn(spec.kind, (Kind.LOCAL_JSONL, Kind.LOCAL_AUDIO))

    def test_registry_order_is_preserved(self):
        specs = registry.specs_for("ar")
        self.assertEqual(specs[0].name, "internal_v76_ar")
        self.assertEqual([s.name for s in registry.iter_specs(("ar",))], [s.name for s in specs])

    def test_internal_specs_cover_all_five_languages_in_both_trees(self):
        """configs/v7.6/internal_ds_sources.yaml names internal sources for
        every language: the v7.6 cleaned manifests AND the raw q3asr SFT
        envelopes. The registry mirrors that (the file-level pin is
        TestInternalSourcesMatchTrainingConfig below)."""
        internal = {s.name for s in registry.all_specs() if s.name.startswith("internal_")}
        self.assertEqual(
            internal,
            {f"internal_v76_{lang}" for lang in registry.LANGUAGES}
            | {f"internal_sft_{lang}" for lang in registry.LANGUAGES},
        )

    def test_sft_specs_resolve_against_their_own_root(self):
        """The SFT tree is a sibling of -- not under -- the v7.6 root the
        stages pass, so only a spec-level root can reach it."""
        for lang in registry.LANGUAGES:
            spec = registry.by_name(f"internal_sft_{lang}")
            self.assertEqual(spec.local_root, registry.SFT_ROOT)
            self.assertEqual(spec.paths, (f"{lang}/*.jsonl*",))
        for lang in registry.LANGUAGES:
            self.assertIsNone(registry.by_name(f"internal_v76_{lang}").local_root)

    def test_internal_patterns_are_relative_to_root_not_already_rooted(self):
        """Regression for the path-doubling bug preflight caught.

        ``expand_paths`` joins ``root`` onto a relative pattern. A pattern that
        already contains the root becomes
        ``training_manifests/v7.6/training_manifests/v7.6/ar/*.jsonl`` and
        matches nothing -- silently, because zero files is also what an empty
        mount looks like.
        """
        for spec in registry.all_specs():
            if spec.kind is not Kind.LOCAL_JSONL:
                continue
            for pattern in spec.paths:
                self.assertFalse(Path(pattern).is_absolute(), f"{spec.name}: {pattern}")
                self.assertNotIn(registry.INTERNAL_ROOT, pattern, f"{spec.name}: {pattern}")
                self.assertNotIn(registry.SFT_ROOT, pattern, f"{spec.name}: {pattern}")

    def test_internal_specs_exclude_recovery_and_eval_shards(self):
        for spec in registry.all_specs():
            if spec.kind is Kind.LOCAL_JSONL:
                self.assertIn("_still_rejected_", spec.exclude)
                self.assertIn("eval_", spec.exclude)

    def test_fleurs_mandarin_config_is_cmn_hans_cn(self):
        """Not ``zh_cn``. Checked against the Hub API; a wrong config 404s late."""
        self.assertEqual(registry.by_name("fleurs_cmn_hans").config, "cmn_hans_cn")
        self.assertEqual(registry.by_name("fleurs_ar_eg").config, "ar_eg")

    def test_summary_reports_a_tokenless_plan(self):
        summary = registry.summary()
        self.assertEqual(set(summary), set(registry.LANGUAGES))
        for lang, row in summary.items():
            self.assertGreater(row["sources"], 0, lang)
            self.assertLessEqual(row["est_hours_no_gated"], row["est_hours"], lang)
            self.assertIsInstance(row["unverified"], list)

    def test_unverified_specs_are_flagged_not_hidden(self):
        unverified = [s.name for s in registry.all_specs() if not s.verified]
        self.assertTrue(unverified)  # some genuinely are unconfirmed
        for name in unverified:
            self.assertTrue(registry.by_name(name).notes, f"{name} lacks an explanatory note")

    def test_require_datasets_returns_the_module_when_present(self):
        """Positive path, exercised without the heavy real dependency.

        ``datasets`` pulls pyarrow/pandas and ships as an optional extra, so
        the suite must stay green where it is not installed. A stub in
        ``sys.modules`` is exactly what ``import datasets`` resolves against.
        """
        stub = types.ModuleType("datasets")
        stub.load_dataset = lambda *a, **k: None
        with mock.patch.dict(sys.modules, {"datasets": stub}):
            self.assertIs(require_datasets(), stub)

    def test_require_datasets_raises_with_install_command_when_absent(self):
        """The branch a fresh node actually hits: a clear fix, not a bare
        ImportError surfacing three frames down inside a loader."""
        with (
            mock.patch.dict(sys.modules, {"datasets": None}),
            self.assertRaises(RuntimeError) as ctx,
        ):
            require_datasets()
        msg = str(ctx.exception)
        self.assertIn("pip install", msg)
        self.assertIn("datasets>=3.0", msg)


class TestInternalSourcesMatchTrainingConfig(unittest.TestCase):
    """The registry's internal specs must cover exactly the datasets the
    v7.6 training config names in ``configs/v7.6/internal_ds_sources.yaml``.

    That file is the source of truth for what "internal" means: its
    ``train_manifest`` lists the v7.6 cleaned corpora and the raw q3asr SFT
    envelopes per language. Every listed file must be ingestible by its
    language's spec -- routed to the right tree, matched by the glob, and not
    caught by the exclusion tokens. A dataset added to the training config
    but not the registry (or vice versa) fails here, not three hours into
    Stage 1."""

    def setUp(self) -> None:
        self.cfg = yaml.safe_load(
            (Path(__file__).resolve().parent.parent
             / "configs" / "v7.6" / "internal_ds_sources.yaml").read_text(encoding="utf-8"))

    def test_every_training_config_source_is_ingestible_by_its_spec(self):
        listed = self.cfg["train_manifest"]
        self.assertEqual(set(listed), set(registry.LANGUAGES), list(listed))
        sft_langs = set()
        for lang, files in listed.items():
            self.assertTrue(files, f"{lang}: empty train_manifest")
            for f in files:
                path = Path(f)
                self.assertEqual(path.parent.name, lang, f)
                self.assertTrue(path.name.endswith(".jsonl"), f)
                if "q3asr_sft_manifests" in path.parts:
                    spec = registry.by_name(f"internal_sft_{lang}")
                    sft_langs.add(lang)
                else:
                    self.assertIn("training_manifests", path.parts, f)
                    self.assertIn("v7.6", path.parts, f)
                    spec = registry.by_name(f"internal_v76_{lang}")
                self.assertFalse(is_excluded(path, spec.exclude), f)
        self.assertEqual(sft_langs, set(registry.LANGUAGES))

    def test_v76_and_sft_specs_stay_in_the_registry_plan(self):
        """Internal specs are ingestible by prepare (LOCAL_JSONL, canonical
        fields) so a YAML/registry drift cannot hide behind a spec that is
        declared but unusable."""
        for lang in registry.LANGUAGES:
            for name in (f"internal_v76_{lang}", f"internal_sft_{lang}"):
                spec = registry.by_name(name)
                self.assertIs(spec.kind, Kind.LOCAL_JSONL, name)
                self.assertEqual(spec.fields, FieldMap(), name)
                self.assertEqual(spec.lang, lang, name)


class TestExpandPaths(_TmpTree):
    def test_glob_under_root_matches(self):
        self.write("ar/train_a.jsonl", [self.rec(0)])
        self.write("ar/train_b.jsonl", [self.rec(1)])
        spec = _spec(paths=("ar/*.jsonl",))
        self.assertEqual([p.name for p in expand_paths(spec, self.root)],
                         ["train_a.jsonl", "train_b.jsonl"])

    def test_already_rooted_pattern_matches_nothing(self):
        """The doubling bug, demonstrated rather than merely prevented.

        Needs a RELATIVE root: an absolute already-rooted pattern skips the join
        and still works, which is why the bug survived an absolute-path review.
        """
        self.write("v7.6/ar/train_a.jsonl", [self.rec(0)])
        import os
        cwd = os.getcwd()
        os.chdir(self.root)
        try:
            doubled = _spec(paths=("v7.6/ar/*.jsonl*",))
            self.assertEqual(expand_paths(doubled, "v7.6"), [])
            self.assertEqual(str(Path("v7.6") / "v7.6/ar/train_a.jsonl"),
                             "v7.6/v7.6/ar/train_a.jsonl")
            correct = _spec(paths=("ar/*.jsonl*",))
            self.assertEqual(len(expand_paths(correct, "v7.6")), 1)
        finally:
            os.chdir(cwd)

    def test_excludes_recovery_and_eval_shards(self):
        self.write("ar/train_a.jsonl", [self.rec(0)])
        self.write("ar/train_a_still_rejected_p0000.jsonl", [self.rec(1)])
        self.write("ar/eval_ar.jsonl", [self.rec(2)])
        spec = _spec(paths=("ar/*.jsonl",), exclude=("_still_rejected_", "eval_"))
        self.assertEqual([p.name for p in expand_paths(spec, self.root)], ["train_a.jsonl"])

    def test_is_excluded_matches_on_filename_only(self):
        self.assertTrue(is_excluded(Path("/x/eval_ar.jsonl"), ("eval_",)))
        self.assertFalse(is_excluded(Path("/x/train_ar.jsonl"), ("eval_",)))
        self.assertFalse(is_excluded(Path("/x/train_ar.jsonl"), ()))

    def test_directory_pattern_recurses_and_filters_by_kind(self):
        self.write("ar/sub/deep.jsonl", [self.rec(0)])
        (self.root / "ar" / "clip.wav").write_bytes(b"x")
        self.assertEqual([p.name for p in expand_paths(_spec(paths=("ar",)), self.root)],
                         ["deep.jsonl"])
        audio = _spec(kind=Kind.LOCAL_AUDIO, paths=("ar",))
        self.assertEqual([p.name for p in expand_paths(audio, self.root)], ["clip.wav"])

    def test_absolute_pattern_ignores_root(self):
        self.write("ar/train_a.jsonl", [self.rec(0)])
        spec = _spec(paths=(str(self.root / "ar" / "train_a.jsonl"),))
        self.assertEqual(len(expand_paths(spec, "/nowhere")), 1)

    def test_explicit_existing_file(self):
        path = self.write("ar/train_a.jsonl", [self.rec(0)])
        self.assertEqual(expand_paths(_spec(paths=(str(path),))), [path])

    def test_unmatched_pattern_yields_nothing_without_raising(self):
        self.assertEqual(expand_paths(_spec(paths=("nope/*.jsonl",)), self.root), [])

    def test_gzipped_manifests_need_the_trailing_wildcard(self):
        """Why the registry pattern is ``*.jsonl*`` and not ``*.jsonl``."""
        self.write("ar/train_a.jsonl.gz", [self.rec(0)], gzipped=True)
        self.assertEqual(expand_paths(_spec(paths=("ar/*.jsonl",)), self.root), [])
        self.assertEqual([p.name for p in expand_paths(_spec(paths=("ar/*.jsonl*",)), self.root)],
                         ["train_a.jsonl.gz"])

    def test_directory_branch_agrees_with_the_glob_branch_on_gzip(self):
        self.write("ar/train_a.jsonl.gz", [self.rec(0)], gzipped=True)
        by_dir = expand_paths(_spec(paths=("ar",)), self.root)
        by_glob = expand_paths(_spec(paths=("ar/*.jsonl*",)), self.root)
        self.assertEqual(by_dir, by_glob)

    def test_duplicates_across_patterns_are_collapsed(self):
        self.write("ar/train_a.jsonl", [self.rec(0)])
        spec = _spec(paths=("ar/*.jsonl", "ar/train_a.jsonl"))
        self.assertEqual(len(expand_paths(spec, self.root)), 1)

    def test_local_root_resolves_a_second_tree_ignoring_the_passed_root(self):
        """The SFT specs live in a sibling tree; their patterns must resolve
        against ``spec.local_root``, not the v7.6 root the stages pass --
        joining the v7.6 root would match nothing and the source would
        silently contribute zero rows."""
        self.write("v7.6/ar/train_ar_inworld.jsonl", [self.rec(0)])
        self.write("sft/ar/train_ar_q3asr.jsonl", [self.rec(1)])
        sft = _spec(paths=("ar/*.jsonl*",), local_root=str(self.root / "sft"))
        self.assertEqual([p.name for p in expand_paths(sft, self.root / "v7.6")],
                         ["train_ar_q3asr.jsonl"])

    def test_local_root_none_falls_back_to_the_callers_root(self):
        self.write("v7.6/ar/train_a.jsonl", [self.rec(0)])
        self.assertEqual(effective_root(_spec(), "v7.6"), "v7.6")
        self.assertEqual(effective_root(_spec(local_root="sft"), "v7.6"), "sft")
        self.assertEqual(effective_root(_spec(local_root="sft"), None), "sft")
        plain = _spec(paths=("ar/*.jsonl*",))
        self.assertEqual([p.name for p in expand_paths(plain, self.root / "v7.6")],
                         ["train_a.jsonl"])


class TestIterLocal(_TmpTree):
    def test_rows_are_parsed_and_tagged_with_their_origin_file(self):
        self.write("ar/train_a.jsonl", [self.rec(0), self.rec(1)])
        stats = IngestStats(spec_name="t")
        rows = list(iter_local(_spec(paths=("ar/*.jsonl",)), self.root, stats))
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0]["__source_file__"].endswith("train_a.jsonl"))
        self.assertEqual(stats.read, 2)

    def test_gzipped_input_reads_identically(self):
        self.write("ar/train_a.jsonl.gz", [self.rec(0)], gzipped=True)
        rows = list(iter_local(_spec(paths=("ar/*.jsonl*",)), self.root))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["audio_filepath"], self.rec(0)["audio_filepath"])

    def test_max_samples_caps_and_reports(self):
        self.write("ar/train_a.jsonl", [self.rec(i) for i in range(10)])
        stats = IngestStats(spec_name="t")
        rows = list(iter_local(_spec(paths=("ar/*.jsonl",), max_samples=3), self.root, stats))
        self.assertEqual(len(rows), 3)
        self.assertTrue(stats.cap_reached)

    def test_bad_json_is_counted_not_fatal(self):
        path = self.write("ar/train_a.jsonl", [self.rec(0)])
        with path.open("a", encoding="utf-8") as fh:
            fh.write("{not json}\n\n")
            fh.write(json.dumps([1, 2]) + "\n")
        stats = IngestStats(spec_name="t")
        rows = list(iter_local(_spec(paths=("ar/*.jsonl",)), self.root, stats))
        self.assertEqual(len(rows), 1)
        self.assertEqual(stats.errors.get("bad_json:train_a.jsonl"), 1)
        self.assertEqual(stats.errors.get("not_an_object"), 1)

    def test_no_files_after_exclusion_logs_and_yields_nothing(self):
        self.write("ar/eval_ar.jsonl", [self.rec(0)])
        spec = _spec(paths=("ar/*.jsonl",), exclude=("eval_",))
        self.assertEqual(list(iter_local(spec, self.root)), [])

    def test_wrong_kind_raises(self):
        with self.assertRaises(ValueError):
            list(iter_local(_spec(kind=Kind.HF_STREAM), self.root))


class TestIterLocalAudio(_TmpTree):
    def _tree(self) -> DatasetSpec:
        for i in range(2):
            (self.root / "ZH").mkdir(parents=True, exist_ok=True)
            (self.root / "ZH" / f"clip{i}.wav").write_bytes(b"RIFF")
            (self.root / "ZH" / f"clip{i}.json").write_text(
                json.dumps({"text": "你好世界", "duration": 3.5}), encoding="utf-8")
        return _spec(name="emilia", lang="zh", kind=Kind.LOCAL_AUDIO, paths=("ZH",),
                     fields=FieldMap(text="text", duration="duration", path="audio_filepath"))

    def test_audio_is_enumerated_with_its_sidecar_merged(self):
        """Regression: this used to return zero rows.

        ``expand_paths`` globbed only ``*.jsonl*``, so an extracted audio tree --
        which has no manifest at all -- matched nothing. zh depends almost
        entirely on this source, so a zh build would have produced an empty
        corpus that looked like an unmounted disk.
        """
        rows = list(iter_local_audio(self._tree(), self.root))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["text"], "你好世界")
        self.assertEqual(rows[0]["duration"], 3.5)
        self.assertTrue(rows[0]["audio_filepath"].endswith(".wav"))

    def test_missing_sidecar_still_yields_the_path(self):
        (self.root / "ZH").mkdir(parents=True, exist_ok=True)
        (self.root / "ZH" / "lonely.wav").write_bytes(b"RIFF")
        spec = self._tree()
        rows = [r for r in iter_local_audio(spec, self.root) if r["audio_filepath"].endswith("lonely.wav")]
        self.assertEqual(len(rows), 1)
        self.assertNotIn("text", rows[0])

    def test_json_sidecars_are_not_themselves_enumerated_as_audio(self):
        rows = list(iter_local_audio(self._tree(), self.root))
        self.assertTrue(all(r["audio_filepath"].endswith(tuple(AUDIO_SUFFIXES)) for r in rows))

    def test_max_samples_caps(self):
        stats = IngestStats(spec_name="emilia")
        spec = _spec(name="emilia", lang="zh", kind=Kind.LOCAL_AUDIO, paths=("ZH",), max_samples=1)
        self._tree()
        rows = list(iter_local_audio(spec, self.root, stats))
        self.assertEqual(len(rows), 1)
        self.assertTrue(stats.cap_reached)


class TestDurationCoverage(_TmpTree):
    def test_measures_a_source_that_omits_duration(self):
        self.write("ar/train_a.jsonl", [
            {"audio_filepath": "/1.wav", "text": "أ", "duration": 1.0},
            {"audio_filepath": "/2.wav", "text": "ب"},
            {"audio_filepath": "/3.wav", "text": "ج"},
        ])
        report = duration_coverage(_spec(paths=("ar/*.jsonl",)), self.root)
        self.assertEqual(report["duration_present"], 1)
        self.assertEqual(report["duration_missing"], 2)
        self.assertAlmostEqual(report["missing_fraction"], 0.6667, places=4)
        self.assertFalse(report["needs_probe"])

    def test_empty_source_is_not_a_division_by_zero(self):
        self.write("ar/train_a.jsonl", [])
        self.assertEqual(duration_coverage(_spec(paths=("ar/*.jsonl",)), self.root)["sampled"], 0)


class TestToSample(unittest.TestCase):
    def _row(self, **over) -> dict:
        row = {"audio_filepath": "/a.wav", "text": f" {_AR_TEXT} ", "duration": "4.250"}
        row.update(over)
        return row

    def test_maps_the_real_v76_shape(self):
        sample, meta = to_sample(_spec(lang="ar"), self._row())
        self.assertIsInstance(sample, Sample)
        self.assertEqual(sample.audio_filepath, "/a.wav")
        self.assertEqual(sample.duration, 4.25)
        self.assertEqual(sample.text, _AR_TEXT)  # stripped
        self.assertEqual(sample.lang, "ar")
        self.assertEqual(meta.dataset, "t_spec")
        self.assertEqual(meta.source_text, f" {_AR_TEXT} ")  # raw preserved for audit
        self.assertEqual(meta.metrics["duration"], 4.25)

    def test_lang_column_overrides_the_spec_when_mapped(self):
        spec = _spec(fields=FieldMap(lang="language"))
        self.assertEqual(to_sample(spec, self._row(language="ml"))[0].lang, "ml")
        self.assertEqual(to_sample(spec, self._row())[0].lang, "ar")

    def test_returns_none_instead_of_raising_for_the_benign_cases(self):
        spec = _spec()
        for row in (
            {"text": "x", "duration": 1.0},                      # no path
            {"audio_filepath": "/a.wav", "duration": 1.0},        # no text
            {"audio_filepath": "/a.wav", "text": "   ", "duration": 1.0},  # blank
            {"audio_filepath": "/a.wav", "text": "x"},            # no duration
            {"audio_filepath": "/a.wav", "text": "x", "duration": None},
        ):
            self.assertIsNone(to_sample(spec, row), row)

    def test_a_source_without_duration_is_reportable_not_silent(self):
        """The gate rejects a missing duration, so it must be counted upstream."""
        spec = _spec(fields=FieldMap(duration=None))
        self.assertIsNone(to_sample(spec, self._row()))

    def test_empty_row(self):
        self.assertIsNone(to_sample(_spec(), {}))


class TestInterleave(unittest.TestCase):
    def test_round_robin_is_fair(self):
        """Regression: ``items.pop(idx % len(items))`` shifted the index after
        each removal and could skip a stream entirely."""
        got = list(interleave([("A", iter(range(3))), ("B", iter(range(3))), ("C", iter(range(3)))]))
        self.assertEqual([n for n, _ in got], ["A", "B", "C"] * 3)
        self.assertEqual([v for _, v in got], [0, 0, 0, 1, 1, 1, 2, 2, 2])

    def test_uneven_streams_drop_out_and_the_rest_continue(self):
        got = list(interleave([("A", iter(range(10))), ("B", iter(range(3))), ("C", iter(range(5)))]))
        counts = {}
        for name, _ in got:
            counts[name] = counts.get(name, 0) + 1
        self.assertEqual(counts, {"A": 10, "B": 3, "C": 5})
        self.assertEqual(len(got), 18)

    def test_order_within_a_stream_is_preserved(self):
        got = list(interleave([("A", iter("abc")), ("B", iter("xy"))]))
        self.assertEqual([v for n, v in got if n == "A"], ["a", "b", "c"])
        self.assertEqual([v for n, v in got if n == "B"], ["x", "y"])

    def test_stop_on_exhausted_halts_the_whole_stream(self):
        """Five, not four: B's exhaustion is only discovered on B's next turn,
        and A gets that turn first. That is correct round-robin, not an off-by-one.
        """
        got = list(interleave([("A", iter(range(10))), ("B", iter(range(2)))],
                              stop_on_exhausted=True))
        self.assertEqual([n for n, _ in got], ["A", "B", "A", "B", "A"])
        self.assertEqual([v for _, v in got], [0, 0, 1, 1, 2])

    def test_single_and_empty(self):
        self.assertEqual([v for _, v in interleave([("A", iter([1, 2]))])], [1, 2])
        self.assertEqual(list(interleave([])), [])
        self.assertEqual(list(interleave([("A", iter([]))])), [])


class TestStreamMetadata(_TmpTree):
    def test_local_jsonl_is_counted_once_not_twice(self):
        """``iter_local`` already counts and caps; the wrapper must not double."""
        self.write("ar/train_a.jsonl", [self.rec(i) for i in range(4)])
        stats = IngestStats(spec_name="t")
        rows = list(stream_metadata(_spec(paths=("ar/*.jsonl",)), str(self.root), stats))
        self.assertEqual(len(rows), 4)
        self.assertEqual(stats.read, 4)

    def test_custom_loader_is_capped_by_the_wrapper(self):
        spec = _spec(kind=Kind.CUSTOM, loader=lambda s: iter({"audio_filepath": f"/{i}.wav",
                                                              "text": "x", "duration": 1.0}
                                                             for i in range(50)),
                     max_samples=5)
        stats = IngestStats(spec_name="t")
        self.assertEqual(len(list(stream_metadata(spec, None, stats))), 5)
        self.assertTrue(stats.cap_reached)

    def test_hf_kind_without_repo_id_raises_early(self):
        with self.assertRaises(ValueError):
            list(stream_metadata(_spec(kind=Kind.HF_STREAM, repo_id=None)))

    def test_custom_kind_without_loader_raises(self):
        with self.assertRaises(ValueError):
            list(stream_metadata(_spec(kind=Kind.CUSTOM, loader=None)))


class TestIngest(_TmpTree):
    def _two_specs(self) -> list[DatasetSpec]:
        self.write("ar/train_a.jsonl", [self.rec(i) for i in range(3)])
        self.write("ar/train_b.jsonl", [self.rec(i + 100) for i in range(3)])
        return [
            _spec(name="src_a", paths=("ar/train_a.jsonl",)),
            _spec(name="src_b", paths=("ar/train_b.jsonl",)),
        ]

    def test_interleaved_ingest_alternates_sources(self):
        specs = self._two_specs()
        got = [(s.name, r["audio_filepath"]) for s, r, _ in ingest(specs, str(self.root))]
        self.assertEqual([n for n, _ in got], ["src_a", "src_b"] * 3)

    def test_sequential_ingest_drains_one_source_at_a_time(self):
        specs = self._two_specs()
        got = [s.name for s, _, _ in ingest(specs, str(self.root), interleave_sources=False)]
        self.assertEqual(got, ["src_a"] * 3 + ["src_b"] * 3)

    def test_single_spec_never_interleaves(self):
        specs = self._two_specs()[:1]
        self.assertEqual(len(list(ingest(specs, str(self.root)))), 3)

    def test_stats_are_per_source_and_reachable_by_the_caller(self):
        specs = self._two_specs()
        seen = {}
        for spec, _, stats in ingest(specs, str(self.root)):
            seen[spec.name] = stats
        self.assertEqual(seen["src_a"].spec_name, "src_a")
        self.assertEqual(seen["src_b"].spec_name, "src_b")
        self.assertEqual(seen["src_a"].read, 3)


class TestStatsReport(unittest.TestCase):
    def test_aggregates_and_counts_caps(self):
        a = IngestStats(spec_name="a", read=10, emitted=8, skipped_no_text=2, cap_reached=True)
        b = IngestStats(spec_name="b", read=5, emitted=5, duration_missing=1)
        report = stats_report([a, b])
        self.assertEqual(report["totals"]["read"], 15)
        self.assertEqual(report["totals"]["emitted"], 13)
        self.assertEqual(report["totals"]["skipped_no_text"], 2)
        self.assertEqual(report["totals"]["duration_missing"], 1)
        self.assertEqual(report["totals"]["cap_reached"], 1)
        self.assertEqual(len(report["sources"]), 2)

    def test_empty(self):
        self.assertEqual(stats_report([])["totals"]["read"], 0)

    def test_stats_as_dict_is_json_safe(self):
        json.dumps(IngestStats(spec_name="a", errors={"bad_json:x": 2}).as_dict())

    def test_meta_is_returned_for_provenance(self):
        _, meta = to_sample(_spec(name="masc_ar", repo_id="MohamedRashad/MASC-Arabic"),
                            {"audio_filepath": "/a.wav", "text": _AR_TEXT, "duration": 1.0})
        self.assertIsInstance(meta, Meta)
        self.assertEqual(meta.dataset, "MohamedRashad/MASC-Arabic")


if __name__ == "__main__":
    unittest.main()
