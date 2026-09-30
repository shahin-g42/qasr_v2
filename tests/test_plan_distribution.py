"""Tests for scripts/corpus/plan_distribution.py -- the 9-node stage-2 planner.

The planner is a pure function of the pool directory: allocation math is tested
on synthetic inventories, the inventory/CLI contract on a planted pool.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "corpus" / "plan_distribution.py"

spec = importlib.util.spec_from_file_location("plan_distribution", SCRIPT_PATH)
planner = importlib.util.module_from_spec(spec)
sys.modules["plan_distribution"] = planner
spec.loader.exec_module(planner)


def _inv(entries: dict[str, tuple[int, int]], *, needs_llm: float | None = None) -> dict[str, dict]:
    """Synthetic inventory: ``lang -> (est_rows, bytes)``, same needs_llm for all."""
    return {
        lang: {
            "shards": 2, "bytes": nbytes,
            "sources": {"s1": {"shards": 1, "bytes": nbytes // 2}},
            "mean_row_bytes": 200.0, "est_rows": rows, "needs_llm_fraction": needs_llm,
        }
        for lang, (rows, nbytes) in entries.items()
    }


class TestDivisors(unittest.TestCase):
    def test_batch_size_divisors(self):
        divs = planner.divisors(12)
        self.assertEqual(divs, [1, 2, 3, 4, 6, 12])
        self.assertEqual([d for d in planner.divisors(100_000) if d <= 8], [1, 2, 4, 5, 8])


class TestAllocation(unittest.TestCase):
    def test_single_language_gets_the_whole_budget_up_to_cap(self):
        inv = _inv({"ar": (40_000_000, 8_000_000_000)})
        plan = planner.plan_slices(inv, nodes=9, batch_size=100_000, max_slices=8)
        self.assertEqual(plan, {"ar": 8})

    def test_max_slices_flag_is_honored(self):
        inv = _inv({"ar": (40_000_000, 8_000_000_000)})
        plan = planner.plan_slices(inv, nodes=9, batch_size=100_000, max_slices=4)
        self.assertEqual(plan, {"ar": 4})

    def test_small_batch_size_uses_its_own_divisors(self):
        inv = _inv({"ar": (40_000_000, 8_000_000_000)})
        plan = planner.plan_slices(inv, nodes=99, batch_size=12, max_slices=8)
        self.assertEqual(plan, {"ar": 6})  # largest divisor of 12 within the cap

    def test_thin_language_is_never_sliced(self):
        # est_rows below half a batch per slice cannot slice past 1.
        inv = _inv({"zh": (60_000, 12_000_000)})
        plan = planner.plan_slices(inv, nodes=9, batch_size=100_000, max_slices=8)
        self.assertEqual(plan, {"zh": 1})

    def test_margin_rule_allows_slicing_at_half_a_batch_per_slice(self):
        # est_rows = d * batch/2 is allowed; one row fewer excludes d = 8 but
        # 5 (the next divisor down) still fits.
        inv = _inv({"ar": (400_000, 80_000_000)})
        self.assertEqual(
            planner.plan_slices(inv, nodes=9, batch_size=100_000, max_slices=8)["ar"], 8)
        inv = _inv({"ar": (399_999, 79_999_800)})
        self.assertEqual(
            planner.plan_slices(inv, nodes=9, batch_size=100_000, max_slices=8)["ar"], 5)

    def test_heavier_language_upgrades_first(self):
        inv = _inv({
            "ar": (10_000_000, 2_000_000_000),
            "en": (500_000, 100_000_000),
            "hi": (500_000, 100_000_000),
        })
        # budget 1 above the three base slices: only the heaviest upgrades.
        plan = planner.plan_slices(inv, nodes=4, batch_size=100_000, max_slices=8)
        self.assertEqual((plan["ar"], plan["en"], plan["hi"]), (2, 1, 1))
        # budget 4: ar keeps beating everyone at every affordable step.
        plan = planner.plan_slices(inv, nodes=7, batch_size=100_000, max_slices=8)
        self.assertEqual((plan["ar"], plan["en"], plan["hi"]), (5, 1, 1))

    def test_no_material_plans_zero_slices(self):
        inv = _inv({})
        inv["zh"] = {"shards": 0, "bytes": 0, "sources": {}, "mean_row_bytes": None,
                     "est_rows": None, "needs_llm_fraction": None}
        plan = planner.plan_slices(inv, nodes=9, batch_size=100_000, max_slices=8)
        self.assertEqual(plan, {"zh": 0})

    def test_work_units_prefer_flagged_rows_over_raw_bytes(self):
        # Regression for the live-run imbalance: ar has the biggest pool but
        # only 46% needs_llm; en is half the size but 96% flagged. Corrector
        # traffic must win the rank budget, so en out-slices ar.
        inv = _inv({
            "ar": (54_000_000, 46_868_000_000),
            "en": (31_600_000, 24_350_000_000),
            "hi": (7_200_000, 5_317_000_000),
            "ml": (1_300_000, 1_091_000_000),
            "zh": (600_000, 369_000_000),
        }, needs_llm=None)
        inv["ar"]["needs_llm_fraction"] = 0.46
        inv["en"]["needs_llm_fraction"] = 0.96
        inv["hi"]["needs_llm_fraction"] = 0.84
        inv["ml"]["needs_llm_fraction"] = 0.99
        inv["zh"]["needs_llm_fraction"] = 1.0
        plan = planner.plan_slices(inv, nodes=9, batch_size=100_000, max_slices=8)
        self.assertGreaterEqual(plan["en"], plan["ar"])
        slices = planner.assign_slices(inv, plan, nodes=9, batch_size=100_000)
        per_rank = {}
        for s in slices:
            per_rank.setdefault(s["rank"], 0.0)
            per_rank[s["rank"]] += s["weight"]
        # No rank carries more than 2x the average slice load (divisor
        # granularity + the 1-slice floor for thin languages make perfect
        # balance impossible; bytes-balancing put ~2x the average on en's two
        # ranks alone).
        avg = sum(per_rank.values()) / len(per_rank)
        self.assertLessEqual(max(per_rank.values()), avg * 2.0)

    def test_bytes_fallback_when_the_row_estimate_is_missing(self):
        inv = _inv({"ar": (1_000_000, 500_000_000), "zh": (1_000_000, 100_000_000)},
                   needs_llm=None)
        # No needs_llm anywhere: bytes decide, ar is heavier.
        plan = planner.plan_slices(inv, nodes=3, batch_size=100_000, max_slices=8)
        self.assertEqual((plan["ar"], plan["zh"]), (2, 1))


class TestAssignment(unittest.TestCase):
    def test_every_slice_assigned_once_and_loads_balanced(self):
        inv = _inv({"ar": (40_000_000, 8_000_000_000), "en": (1_000_000, 200_000_000),
                    "zh": (1_000_000, 200_000_000)})
        plan = planner.plan_slices(inv, nodes=9, batch_size=100_000, max_slices=8)
        slices = planner.assign_slices(inv, plan, nodes=9, batch_size=100_000)
        self.assertEqual(len(slices), sum(plan.values()))
        self.assertEqual(len({(s["lang"], s["part"]) for s in slices}), len(slices))
        self.assertEqual({s["part"] for s in slices if s["lang"] == "ar"},
                         set(range(plan["ar"])))
        for entry in slices:
            self.assertIn(entry["rank"], range(9))
            self.assertEqual(entry["slices"], plan[entry["lang"]])
            self.assertEqual(entry["batch_size"], 100_000 // plan[entry["lang"]])
        loads = [sum(s["weight"] for s in slices if s["rank"] == r) for r in range(9)]
        biggest = max(s["weight"] for s in slices)
        self.assertLessEqual(max(loads) - min(loads), biggest)

    def test_more_languages_than_nodes_still_covers_everything(self):
        inv = _inv({f"l{i}": (1_000_000, 100_000_000) for i in range(5)})
        plan = planner.plan_slices(inv, nodes=2, batch_size=100_000, max_slices=8)
        self.assertEqual(set(plan.values()), {1})
        slices = planner.assign_slices(inv, plan, nodes=2, batch_size=100_000)
        self.assertEqual(len(slices), 5)
        self.assertEqual({s["rank"] for s in slices}, {0, 1})


class _Pool(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.pool = Path(self._tmp.name) / "pool"

    def plant(self, lang, source, shards, rows, *, needs_llm=False):
        for index in range(shards):
            path = self.pool / lang / source / f"part-{index:05d}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            row = json.dumps({"audio_filepath": f"{path.parent}/a.bin",
                              "needs_llm": needs_llm, "pad": "x" * 280})
            path.write_text((row + "\n") * rows, encoding="utf-8")
        return path


class TestInventory(_Pool):
    def test_reads_sources_and_samples_rows(self):
        self.plant("ar", "internal_v76_ar", 2, 50)
        self.plant("ar", "internal_sft_ar", 1, 50, needs_llm=True)
        inv = planner.inventory(self.pool, ["ar", "zh"], sample_rows=100)
        self.assertEqual(inv["ar"]["shards"], 3)
        self.assertEqual(set(inv["ar"]["sources"]), {"internal_sft_ar", "internal_v76_ar"})
        self.assertGreater(inv["ar"]["est_rows"], 0)
        self.assertIsNotNone(inv["ar"]["mean_row_bytes"])
        self.assertEqual(inv["ar"]["needs_llm_fraction"], round(33 / 99, 4))
        self.assertEqual(inv["zh"]["shards"], 0)
        self.assertIsNone(inv["zh"]["est_rows"])


class TestMain(_Pool):
    def test_json_is_deterministic_and_assign_rank_matches_it(self):
        self.plant("ar", "internal_v76_ar", 2, 50)
        self.plant("en", "internal_v76_en", 1, 50)
        first = Path(self._tmp.name) / "p1.json"
        second = Path(self._tmp.name) / "p2.json"
        base = ["--pool-dir", str(self.pool), "--langs", "ar,en", "--nodes", "3"]
        self.assertEqual(planner.main([*base, "--json", str(first)]), 0)
        self.assertEqual(planner.main([*base, "--json", str(second)]), 0)
        self.assertEqual(first.read_text(encoding="utf-8"), second.read_text(encoding="utf-8"))
        plan = json.loads(first.read_text(encoding="utf-8"))
        self.assertEqual(plan["nodes"], 3)
        for entry in plan["assignment"]:
            self.assertEqual(entry["slices"], plan["languages"][entry["lang"]]["slices"])
            self.assertLess(entry["rank"], 3)

        out = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), *base, "--assign-rank", "1"],
            capture_output=True, text=True, check=True)
        lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
        want = sorted(f"{a['lang']}\t{a['part']}\t{a['slices']}"
                      for a in plan["assignment"] if a["rank"] == 1)
        self.assertEqual(sorted(lines), want)
        for line in lines:
            lang, part, slices = line.split("\t")
            self.assertIn(lang, {"ar", "en"})
            self.assertEqual(int(slices), plan["languages"][lang]["slices"])
            self.assertGreaterEqual(int(part), 0)

    def test_no_material_exits_2(self):
        rc = planner.main(["--pool-dir", str(self.pool), "--langs", "zh", "--nodes", "2"])
        self.assertEqual(rc, 2)

    def test_bad_arguments_are_rejected(self):
        with self.assertRaises(SystemExit):
            planner.main(["--pool-dir", str(self.pool), "--langs", ",", "--nodes", "2"])
        with self.assertRaises(SystemExit):
            planner.main(["--pool-dir", str(self.pool), "--langs", "ar", "--nodes", "0"])
        with self.assertRaises(SystemExit):
            planner.main(["--pool-dir", str(self.pool), "--langs", "ar", "--nodes", "2",
                          "--assign-rank", "2"])


if __name__ == "__main__":
    unittest.main()
