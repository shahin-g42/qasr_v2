"""Tests for scripts/corpus/check_internal_sources.py -- the tree-vs-config gate.

The checker is the only component that can see the REAL trees, so its logic is
exercised against a synthetic clone of both trees, built from the basenames the
config lists. That keeps the test honest: if any name in
``configs/corpus/internal_ingest.yaml`` stopped surviving a round trip
through the registry specs, a clone planting exactly those names would report
a finding and the clean-clone test below would fail.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "corpus" / "check_internal_sources.py"
CONFIG_PATH = REPO_ROOT / "configs" / "corpus" / "internal_ingest.yaml"

spec = importlib.util.spec_from_file_location("check_internal_sources", SCRIPT_PATH)
check = importlib.util.module_from_spec(spec)
sys.modules["check_internal_sources"] = check
spec.loader.exec_module(check)


class _Clone(unittest.TestCase):
    """A temp clone of BOTH internal trees, holding the config's basenames."""

    def setUp(self) -> None:
        self.cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.roots = {
            "v76": self.base / "training_manifests" / "v7.6",
            "sft": self.base / "q3asr_sft_manifests",
        }
        for root in self.roots.values():
            root.mkdir(parents=True, exist_ok=True)

    def plant(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"audio_filepath": "/x/y.wav"}\n', encoding="utf-8")
        return path

    def plant_config(self, cfg: dict | None = None) -> None:
        """Plant every file the config names, at ``<root>/<lang>/<basename>``."""
        cfg = self.cfg if cfg is None else cfg
        for entry in ("train_manifest", "eval_manifest"):
            for lang, files in cfg[entry].items():
                for f in files:
                    path = Path(f)
                    tree = "sft" if "q3asr_sft_manifests" in path.parts else "v76"
                    if check._tree_of(path) == tree:
                        self.plant(self.roots[tree] / lang / path.name)

    def mutate_cfg(self, mutate: Callable[[dict], None]) -> dict:
        cfg = copy.deepcopy(self.cfg)
        mutate(cfg)
        return cfg

    def write_cfg(self, cfg: dict) -> Path:
        path = self.base / "internal_ds_sources.yaml"
        path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
        return path

    def run_audit(self, config_path: Path | None = None) -> dict:
        return check.audit(
            config_path if config_path is not None else CONFIG_PATH,
            internal_root=self.roots["v76"],
            sft_root=self.roots["sft"],
        )

    def source(self, report: dict, lang: str, tree: str) -> dict:
        for src in report["sources"]:
            if src["lang"] == lang and src["tree"] == tree:
                return src
        raise AssertionError(f"no source {lang}/{tree} in the report")


class FaithfulCloneTest(_Clone):
    def test_clone_of_the_config_is_clean(self):
        self.plant_config()
        report = self.run_audit()
        self.assertEqual(report["problems"], [])
        self.assertTrue(report["ok"])
        totals = report["totals"]
        # Every config entry was routed to a tree and checked: the strongest
        # form of "use all the datasets mentioned in the config".
        self.assertEqual(totals["listed_train"], totals["config_train_entries"])
        self.assertEqual(totals["eval_listed"], totals["config_eval_entries"])
        self.assertEqual(totals["matched_train"], totals["listed_train"])
        self.assertEqual(totals["missing_train"], 0)
        self.assertEqual(totals["extra_train"], 0)
        self.assertEqual(totals["eval_ingested"], 0)
        self.assertEqual(totals["eval_gate_invisible"], 0)
        self.assertEqual(totals["eval_missing"], 0)
        self.assertEqual(len(report["sources"]), 10)
        for src in report["sources"]:
            self.assertEqual(src["listed"], src["matched"], (src["lang"], src["tree"]))

    def test_unlisted_eval_file_is_harmless(self):
        """Extra eval files only join the exclusion set; they cannot be ingested."""
        self.plant_config()
        self.plant(self.roots["v76"] / "ar" / "eval_ar_bonus.jsonl")
        report = self.run_audit()
        self.assertEqual(report["problems"], [])

    def test_still_rejected_leftovers_are_ignored(self):
        self.plant_config()
        self.plant(self.roots["v76"] / "ar" / "train_ar_inworld_full_still_rejected_p0000.jsonl")
        report = self.run_audit()
        self.assertEqual(report["problems"], [])

    def test_api_returns_paths_resolvable_from_flags_alone(self):
        self.plant_config()
        report = self.run_audit()
        self.assertEqual(report["roots"]["v76"], str(self.roots["v76"]))
        self.assertEqual(report["roots"]["sft"], str(self.roots["sft"]))


class FindingsTest(_Clone):
    def test_missing_train_file_is_reported_not_on_disk(self):
        self.plant_config()
        (self.roots["v76"] / "ar" / "train_ar_dialect_gulf.jsonl").unlink()
        report = self.run_audit()
        self.assertFalse(report["ok"])
        src = self.source(report, "ar", "v76")
        self.assertIn("train_ar_dialect_gulf.jsonl", src["missing"])
        self.assertEqual(
            src["missing_reasons"]["train_ar_dialect_gulf.jsonl"], "not on disk in this tree"
        )
        self.assertTrue(any("not on disk" in p for p in report["problems"]), report["problems"])

    def test_stray_on_disk_train_file_is_ignored_by_explicit_paths(self):
        """The specs pin exact file names, so a file the config never named is
        structurally un-ingestible -- unlike the glob era, a stray cannot leak
        in, and the checker must not flag it."""
        self.plant_config()
        self.plant(self.roots["sft"] / "en" / "train_en_q3asr_r0_100.jsonl")
        report = self.run_audit()
        self.assertTrue(report["ok"])
        src = self.source(report, "en", "sft")
        self.assertEqual(src["extra"], [])

    def test_excluded_name_reports_the_spec_tokens(self):
        """A train entry a spec's exclude tokens catch can never be ingested."""

        def mutate(cfg: dict) -> None:
            files = cfg["train_manifest"]["ar"]
            files[files.index(next(f for f in files if f.endswith("/train_ar_dialect_gulf.jsonl")))] = (
                "/lustrefs/x/training_manifests/v7.6/ar/eval_ar_dialect_gulf.jsonl"
            )

        cfg = self.mutate_cfg(mutate)
        self.plant_config(cfg)
        report = self.run_audit(self.write_cfg(cfg))
        self.assertFalse(report["ok"])
        src = self.source(report, "ar", "v76")
        reason = src["missing_reasons"]["eval_ar_dialect_gulf.jsonl"]
        self.assertIn("excluded by the spec's tokens", reason)
        self.assertIn("eval_", reason)

    def test_config_eval_file_hidden_from_the_gate_is_flagged(self):
        """An eval name that skips both the spec tokens and the eval_*.jsonl*
        glob is the leak the gate cannot see."""

        def mutate(cfg: dict) -> None:
            files = cfg["eval_manifest"]["ar"]
            files[files.index(next(f for f in files if f.endswith("/eval_ar_ar_ae.jsonl")))] = (
                "/lustrefs/x/training_manifests/v7.6/ar/holdout_ar_ar_ae.jsonl"
            )

        cfg = self.mutate_cfg(mutate)
        self.plant_config(cfg)
        report = self.run_audit(self.write_cfg(cfg))
        self.assertFalse(report["ok"])
        src = self.source(report, "ar", "v76")
        self.assertIn("holdout_ar_ar_ae.jsonl", src["eval_ingested"])
        self.assertIn("holdout_ar_ar_ae.jsonl", src["eval_gate_invisible"])
        # Not flagged as a train 'extra': the explicit spec paths cannot
        # ingest a stray, so the two eval findings above are the whole story.

    def test_unroutable_config_entry_is_flagged(self):
        def mutate(cfg: dict) -> None:
            cfg["train_manifest"]["ar"].append("/data/elsewhere/train_ar_oops.jsonl")

        cfg = self.mutate_cfg(mutate)
        self.plant_config()
        report = self.run_audit(self.write_cfg(cfg))
        self.assertFalse(report["ok"])
        self.assertTrue(
            any("not under either internal tree" in p for p in report["problems"]),
            report["problems"],
        )
        # The orphan entry was not routed, so it is outside the checked set.
        self.assertEqual(
            report["totals"]["listed_train"], report["totals"]["config_train_entries"] - 1
        )

    def test_entry_filed_under_the_wrong_language_dir_is_flagged(self):
        def mutate(cfg: dict) -> None:
            cfg["train_manifest"]["en"].append(cfg["train_manifest"]["ar"][0])

        cfg = self.mutate_cfg(mutate)
        self.plant_config()
        report = self.run_audit(self.write_cfg(cfg))
        self.assertFalse(report["ok"])
        self.assertTrue(
            any("filed under 'ar', not 'en'" in p for p in report["problems"]),
            report["problems"],
        )


class RootsTest(_Clone):
    def test_config_derived_roots_point_at_both_trees(self):
        v76 = check._config_tree(self.cfg, "v76")
        sft = check._config_tree(self.cfg, "sft")
        self.assertTrue(str(v76).endswith("training_manifests/v7.6"), v76)
        self.assertTrue(str(sft).endswith("q3asr_sft_manifests"), sft)

    def test_unreachable_tree_raises_with_the_flag_hint(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            check.audit(CONFIG_PATH, internal_root=self.base / "nope", sft_root=self.roots["sft"])
        self.assertIn("--internal-root", str(ctx.exception))

    def test_missing_tree_message_names_the_sft_flag(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            check.audit(CONFIG_PATH, internal_root=self.roots["v76"], sft_root=self.base / "nope")
        self.assertIn("--sft-root", str(ctx.exception))


class CliTest(_Clone):
    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--config", str(CONFIG_PATH), *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_clean_clone_exits_zero_with_json(self):
        self.plant_config()
        proc = self._run(
            "--json", "--internal-root", str(self.roots["v76"]), "--sft-root", str(self.roots["sft"])
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = json.loads(proc.stdout)
        self.assertTrue(report["ok"])

    def test_findings_exit_one(self):
        self.plant_config()
        (self.roots["v76"] / "ar" / "train_ar_dialect_gulf.jsonl").unlink()
        proc = self._run("--internal-root", str(self.roots["v76"]), "--sft-root", str(self.roots["sft"]))
        self.assertEqual(proc.returncode, 1, proc.stdout)
        self.assertIn("PROBLEM", proc.stderr)

    def test_unreachable_trees_exit_two(self):
        proc = self._run("--internal-root", str(self.base / "missing"), "--sft-root", str(self.base / "missing"))
        self.assertEqual(proc.returncode, 2)
        self.assertIn("cannot verify", proc.stderr)


if __name__ == "__main__":
    unittest.main()
