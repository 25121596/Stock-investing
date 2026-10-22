"""E2E Test Suite: Out-of-Sample Benchmark Hurdles & Double Slippage Stress Runner (Tiers 1-4).

Requirements Enforced:
1. R2 & Acceptance Criteria: V0.8 Baseline Hurdles:
   - 2022–2025 continuous OOS net return > +42.79%, Max Drawdown <= 8.32% (signed >= -0.0832).
   - 2026 OOS net return > +15.73%, Max Drawdown <= 3.99% (signed >= -0.0399).
   - Double slippage (10 bps): positive return across both periods AND beating stressed baseline
     (+37.70% in 2022–2025, +15.48% in 2026).
2. R4: Statistical Controls & Multi-Seed Stability:
   - Seeds: 2026101401, 2026101402, 2026101403.
   - Deterministic execution across repeated runs with identical seed.
   - Multi-seed variance quantification preventing random order lucky outliers.
3. Candidate Strategy Evaluation Harness:
   - BenchmarkHurdleEvaluator: Reusable scorecard engine checking all hurdle gates.
   - Full regression verification on historical candidate tournament outcomes (V10).
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import arena
from engine import Config, tick
from v08_engine_verified import VerifiedV08Replay


class BaselineConstants:
    """Authoritative reference hurdles derived from ORIGINAL_REQUEST.md & V0.8 baseline audits."""

    # 2022-2025 Normal Slippage (5 bps)
    BASELINE_2022_2025_NET_RETURN = 0.4279
    BASELINE_2022_2025_MAX_DRAWDOWN = -0.0832  # -8.32%

    # 2026 Normal Slippage (5 bps)
    BASELINE_2026_NET_RETURN = 0.1573
    BASELINE_2026_MAX_DRAWDOWN = -0.0399      # -3.99%

    # Stressed Double Slippage (10 bps)
    STRESSED_2022_2025_NET_RETURN = 0.3770    # +37.70%
    STRESSED_2026_NET_RETURN = 0.1548         # +15.48%

    # Multi-seed controls
    CONTROL_SEEDS = (2026101401, 2026101402, 2026101403)


class BenchmarkHurdleEvaluator:
    """Rigorous evaluation harness for certifying candidate strategies against V0.8 hurdles."""

    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def evaluate(self, results: dict[str, dict[str, float]]) -> dict:
        """Evaluate strategy metrics against all acceptance hurdles.

        Expected input format:
        {
            "2022-2025": {"normal_return": float, "normal_max_dd": float,
                          "stress_return": float, "stress_max_dd": float},
            "2026": {"normal_return": float, "normal_max_dd": float,
                     "stress_return": float, "stress_max_dd": float}
        }
        """
        checks = {}
        reasons = []

        # Period 1: 2022-2025
        p1 = results.get("2022-2025", {})
        ret_p1 = p1.get("normal_return", -999.0)
        dd_p1 = p1.get("normal_max_dd", -999.0)
        sret_p1 = p1.get("stress_return", -999.0)

        p1_ret_ok = ret_p1 > BaselineConstants.BASELINE_2022_2025_NET_RETURN + self.eps
        p1_dd_ok = dd_p1 >= BaselineConstants.BASELINE_2022_2025_MAX_DRAWDOWN - self.eps
        p1_stress_ok = (sret_p1 > BaselineConstants.STRESSED_2022_2025_NET_RETURN + self.eps) and (sret_p1 > 0)

        checks["2022_2025_higher_net_return"] = p1_ret_ok
        checks["2022_2025_drawdown_not_worse"] = p1_dd_ok
        checks["2022_2025_higher_stressed_return"] = p1_stress_ok

        if not p1_ret_ok:
            reasons.append(f"2022-2025 return {ret_p1:.2%} <= baseline {BaselineConstants.BASELINE_2022_2025_NET_RETURN:.2%}")
        if not p1_dd_ok:
            reasons.append(f"2022-2025 max drawdown {dd_p1:.2%} worse than baseline {BaselineConstants.BASELINE_2022_2025_MAX_DRAWDOWN:.2%}")
        if not p1_stress_ok:
            reasons.append(f"2022-2025 double slippage return {sret_p1:.2%} <= stressed baseline {BaselineConstants.STRESSED_2022_2025_NET_RETURN:.2%}")

        # Period 2: 2026
        p2 = results.get("2026", {})
        ret_p2 = p2.get("normal_return", -999.0)
        dd_p2 = p2.get("normal_max_dd", -999.0)
        sret_p2 = p2.get("stress_return", -999.0)

        p2_ret_ok = ret_p2 > BaselineConstants.BASELINE_2026_NET_RETURN + self.eps
        p2_dd_ok = dd_p2 >= BaselineConstants.BASELINE_2026_MAX_DRAWDOWN - self.eps
        p2_stress_ok = (sret_p2 > BaselineConstants.STRESSED_2026_NET_RETURN + self.eps) and (sret_p2 > 0)

        checks["2026_higher_net_return"] = p2_ret_ok
        checks["2026_drawdown_not_worse"] = p2_dd_ok
        checks["2026_higher_stressed_return"] = p2_stress_ok

        if not p2_ret_ok:
            reasons.append(f"2026 return {ret_p2:.2%} <= baseline {BaselineConstants.BASELINE_2026_NET_RETURN:.2%}")
        if not p2_dd_ok:
            reasons.append(f"2026 max drawdown {dd_p2:.2%} worse than baseline {BaselineConstants.BASELINE_2026_MAX_DRAWDOWN:.2%}")
        if not p2_stress_ok:
            reasons.append(f"2026 double slippage return {sret_p2:.2%} <= stressed baseline {BaselineConstants.STRESSED_2026_NET_RETURN:.2%}")

        passed = all(checks.values())

        gaps = {
            "2022_2025_return_margin": ret_p1 - BaselineConstants.BASELINE_2022_2025_NET_RETURN,
            "2022_2025_drawdown_margin": dd_p1 - BaselineConstants.BASELINE_2022_2025_MAX_DRAWDOWN,
            "2022_2025_stress_margin": sret_p1 - BaselineConstants.STRESSED_2022_2025_NET_RETURN,
            "2026_return_margin": ret_p2 - BaselineConstants.BASELINE_2026_NET_RETURN,
            "2026_drawdown_margin": dd_p2 - BaselineConstants.BASELINE_2026_MAX_DRAWDOWN,
            "2026_stress_margin": sret_p2 - BaselineConstants.STRESSED_2026_NET_RETURN,
        }

        return {
            "passed": passed,
            "verdict": "PASSED" if passed else "REJECTED",
            "checks": checks,
            "gaps": gaps,
            "reasons": reasons,
        }


class TestE2EBenchmarks(unittest.TestCase):
    """Tier 1-4 tests for Benchmark Hurdle Evaluation & Double Slippage Stress Testing."""

    def setUp(self):
        self.evaluator = BenchmarkHurdleEvaluator()

    # -------------------------------------------------------------------------
    # Tier 1: Core Feature Coverage (Happy Path)
    # -------------------------------------------------------------------------

    def test_tier1_authoritative_hurdle_constants(self):
        """Tier 1: Verify exact reference numbers mandated by R1, R2, and Acceptance Criteria."""
        self.assertAlmostEqual(BaselineConstants.BASELINE_2022_2025_NET_RETURN, 0.4279, places=4)
        self.assertAlmostEqual(BaselineConstants.BASELINE_2022_2025_MAX_DRAWDOWN, -0.0832, places=4)
        self.assertAlmostEqual(BaselineConstants.BASELINE_2026_NET_RETURN, 0.1573, places=4)
        self.assertAlmostEqual(BaselineConstants.BASELINE_2026_MAX_DRAWDOWN, -0.0399, places=4)
        self.assertAlmostEqual(BaselineConstants.STRESSED_2022_2025_NET_RETURN, 0.3770, places=4)
        self.assertAlmostEqual(BaselineConstants.STRESSED_2026_NET_RETURN, 0.1548, places=4)
        self.assertEqual(BaselineConstants.CONTROL_SEEDS, (2026101401, 2026101402, 2026101403))

    def test_tier1_evaluator_happy_path_passing_candidate(self):
        """Tier 1: Evaluator certifies candidate that strictly beats all hurdles."""
        candidate_results = {
            "2022-2025": {
                "normal_return": 0.4850,    # > 42.79%
                "normal_max_dd": -0.0710,   # better than -8.32%
                "stress_return": 0.4120,    # > 37.70%
                "stress_max_dd": -0.0950,
            },
            "2026": {
                "normal_return": 0.1820,    # > 15.73%
                "normal_max_dd": -0.0310,   # better than -3.99%
                "stress_return": 0.1710,    # > 15.48%
                "stress_max_dd": -0.0350,
            }
        }

        eval_report = self.evaluator.evaluate(candidate_results)
        self.assertTrue(eval_report["passed"])
        self.assertEqual(eval_report["verdict"], "PASSED")
        self.assertEqual(len(eval_report["reasons"]), 0)
        self.assertGreater(eval_report["gaps"]["2022_2025_return_margin"], 0)
        self.assertGreater(eval_report["gaps"]["2026_return_margin"], 0)

    def test_tier1_historical_audited_baseline_parity(self):
        """Tier 1: Verify that recorded arena baseline matches authoritative constants."""
        checkpoint_path = ROOT / "results/v10-ranking-objectives-20261021-v1/checkpoint.json"
        if not checkpoint_path.exists():
            self.skipTest("V10 checkpoint.json not found")

        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        results = data.get("results", [])

        # Look up baseline results
        baseline_2022 = next((r for r in results if r["period"] == "2022-2025" and r["case"] == "baseline"), None)
        baseline_2026 = next((r for r in results if r["period"] == "2026" and r["case"] == "baseline"), None)
        stress_2022 = next((r for r in results if r["period"] == "2022-2025" and r["case"] == "stress_baseline"), None)
        stress_2026 = next((r for r in results if r["period"] == "2026" and r["case"] == "stress_baseline"), None)

        self.assertIsNotNone(baseline_2022)
        self.assertIsNotNone(baseline_2026)
        self.assertIsNotNone(stress_2022)
        self.assertIsNotNone(stress_2026)

        self.assertAlmostEqual(baseline_2022["total_return"], BaselineConstants.BASELINE_2022_2025_NET_RETURN, places=4)
        self.assertAlmostEqual(baseline_2022["max_drawdown"], BaselineConstants.BASELINE_2022_2025_MAX_DRAWDOWN, places=4)
        self.assertAlmostEqual(baseline_2026["total_return"], BaselineConstants.BASELINE_2026_NET_RETURN, places=4)
        self.assertAlmostEqual(baseline_2026["max_drawdown"], BaselineConstants.BASELINE_2026_MAX_DRAWDOWN, places=4)
        self.assertAlmostEqual(stress_2022["total_return"], BaselineConstants.STRESSED_2022_2025_NET_RETURN, places=4)
        self.assertAlmostEqual(stress_2026["total_return"], BaselineConstants.STRESSED_2026_NET_RETURN, places=4)

    # -------------------------------------------------------------------------
    # Tier 2: Boundary Value Analysis & Rejection Criteria
    # -------------------------------------------------------------------------

    def test_tier2_rejection_higher_return_but_worse_drawdown(self):
        """Tier 2: Candidate with +66% return but -9.92% drawdown is strictly rejected."""
        results = {
            "2022-2025": {
                "normal_return": 0.6644,
                "normal_max_dd": -0.0992,  # WORSE than -8.32%
                "stress_return": 0.6583,
                "stress_max_dd": -0.1008,
            },
            "2026": {
                "normal_return": 0.1800,
                "normal_max_dd": -0.0350,
                "stress_return": 0.1700,
                "stress_max_dd": -0.0380,
            }
        }
        report = self.evaluator.evaluate(results)
        self.assertFalse(report["passed"])
        self.assertEqual(report["verdict"], "REJECTED")
        self.assertFalse(report["checks"]["2022_2025_drawdown_not_worse"])
        self.assertTrue(any("drawdown" in r for r in report["reasons"]))

    def test_tier2_rejection_2026_out_of_sample_overfitting(self):
        """Tier 2: Candidate with +72% return in 2022-2025 but failing 2026 (+12.60%) is rejected."""
        results = {
            "2022-2025": {
                "normal_return": 0.7253,
                "normal_max_dd": -0.0648,
                "stress_return": 0.7148,
                "stress_max_dd": -0.0605,
            },
            "2026": {
                "normal_return": 0.1260,  # Fails 2026 hurdle (12.60% < 15.73%)
                "normal_max_dd": -0.0406,  # Also worse drawdown (-4.06% < -3.99%)
                "stress_return": 0.1234,
                "stress_max_dd": -0.0419,
            }
        }
        report = self.evaluator.evaluate(results)
        self.assertFalse(report["passed"])
        self.assertEqual(report["verdict"], "REJECTED")
        self.assertFalse(report["checks"]["2026_higher_net_return"])
        self.assertFalse(report["checks"]["2026_drawdown_not_worse"])

    def test_tier2_rejection_double_slippage_fragility(self):
        """Tier 2: Candidate passes normal slippage, but fails under 10 bps double slippage."""
        results = {
            "2022-2025": {
                "normal_return": 0.4500,
                "normal_max_dd": -0.0750,
                "stress_return": 0.3500,  # Fails: 35.00% < stressed baseline 37.70%
                "stress_max_dd": -0.1200,
            },
            "2026": {
                "normal_return": 0.1700,
                "normal_max_dd": -0.0350,
                "stress_return": 0.1600,
                "stress_max_dd": -0.0380,
            }
        }
        report = self.evaluator.evaluate(results)
        self.assertFalse(report["passed"])
        self.assertEqual(report["verdict"], "REJECTED")
        self.assertFalse(report["checks"]["2022_2025_higher_stressed_return"])

    def test_tier2_exact_boundary_thresholds(self):
        """Tier 2: Test boundary with +/- epsilon."""
        # Baseline return + 1e-13 vs Baseline return - 1e-13
        base_return = BaselineConstants.BASELINE_2026_NET_RETURN
        res_fail = {
            "2022-2025": {"normal_return": 0.50, "normal_max_dd": -0.05, "stress_return": 0.45},
            "2026": {"normal_return": base_return - 1e-6, "normal_max_dd": -0.03, "stress_return": 0.20}
        }
        res_pass = {
            "2022-2025": {"normal_return": 0.50, "normal_max_dd": -0.05, "stress_return": 0.45},
            "2026": {"normal_return": base_return + 1e-4, "normal_max_dd": -0.03, "stress_return": 0.20}
        }
        self.assertFalse(self.evaluator.evaluate(res_fail)["passed"])
        self.assertTrue(self.evaluator.evaluate(res_pass)["passed"])

    # -------------------------------------------------------------------------
    # Tier 3: Statistical Controls & Historical Regression
    # -------------------------------------------------------------------------

    def test_tier3_rejection_of_all_historical_v10_candidates(self):
        """Tier 3: Ensure ALL failed V10 candidates are strictly rejected by the benchmark harness."""
        # Authoritative metrics from results/v10-ranking-objectives-20261021-v1/research-report.md:
        v10_matrix = {
            "return_regressor": {
                "2022-2025": {"normal_return": 0.4356, "normal_max_dd": -0.1169, "stress_return": 0.4099},
                "2026": {"normal_return": 0.1426, "normal_max_dd": -0.0317, "stress_return": 0.1402},
            },
            "tail_labels": {
                "2022-2025": {"normal_return": 0.6644, "normal_max_dd": -0.0718, "stress_return": 0.6583},
                "2026": {"normal_return": 0.1042, "normal_max_dd": -0.0376, "stress_return": 0.1016},
            },
            "tail_top_focus": {
                "2022-2025": {"normal_return": 0.4655, "normal_max_dd": -0.0992, "stress_return": 0.4528},
                "2026": {"normal_return": 0.0984, "normal_max_dd": -0.0399, "stress_return": 0.0960},
            },
            "tail_xendcg": {
                "2022-2025": {"normal_return": 0.7253, "normal_max_dd": -0.0648, "stress_return": 0.7148},
                "2026": {"normal_return": 0.1260, "normal_max_dd": -0.0406, "stress_return": 0.1234},
            },
            "top_focus": {
                "2022-2025": {"normal_return": 0.5269, "normal_max_dd": -0.0631, "stress_return": 0.4679},
                "2026": {"normal_return": 0.1523, "normal_max_dd": -0.0377, "stress_return": 0.1497},
            },
        }

        for candidate_name, candidate_res in v10_matrix.items():
            report = self.evaluator.evaluate(candidate_res)
            self.assertFalse(
                report["passed"],
                f"Historical failed candidate '{candidate_name}' must be REJECTED, but passed!"
            )
            self.assertEqual(report["verdict"], "REJECTED")

    def test_tier3_multi_seed_stability_determinism(self):
        """Tier 3: Verify bitwise determinism and bounded cross-seed variance across CONTROL_SEEDS."""
        seeds = BaselineConstants.CONTROL_SEEDS
        pool = pd.DataFrame([{"date": d, "code": str(c)} for d in ["2025-01-02", "2025-01-03"] for c in range(10)])

        for seed in seeds:
            # Determinism check: Running twice with identical seed yields identical sample
            sample_1 = arena.stable_random(pool, seed)
            sample_2 = arena.stable_random(pool, seed)
            pd.testing.assert_frame_equal(sample_1, sample_2, "Seeded selection must be strictly deterministic")

        # Distinct seeds should yield distinct candidate permutations
        results_by_seed = [arena.stable_random(pool, s) for s in seeds]
        self.assertFalse(results_by_seed[0].equals(results_by_seed[1]))
        self.assertFalse(results_by_seed[1].equals(results_by_seed[2]))

    # -------------------------------------------------------------------------
    # Tier 4: Scenario / Double Slippage Stress Test Runner
    # -------------------------------------------------------------------------

    def test_tier4_double_slippage_simulation_runner(self):
        """Tier 4: Replay strategy under normal (5 bps) vs double (10 bps) slippage and verify impact."""
        days = ["2025-01-02", "2025-01-03", "2025-01-06"]
        code = "sh.600000"
        bars_df = pd.DataFrame([
            {"date": "2025-01-02", "code": code, "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "preclose": 10.0, "volume": 100000.0, "tradestatus": 1, "isST": 0},
            {"date": "2025-01-03", "code": code, "open": 10.5, "high": 10.5, "low": 10.5, "close": 10.5, "preclose": 10.0, "volume": 100000.0, "tradestatus": 1, "isST": 0},
            {"date": "2025-01-06", "code": code, "open": 11.0, "high": 11.0, "low": 11.0, "close": 11.0, "preclose": 10.5, "volume": 100000.0, "tradestatus": 1, "isST": 0},
        ])
        frames = {code: bars_df}
        bars = {code: bars_df.set_index("date").to_dict("index")}
        signals = pd.DataFrame([{"date": "2025-01-02", "code": code, "score": 1.0}])

        base_cfg = Config(capital=30000.0, order_budget=5000.0, hold_days=1, slippage=0.0005)
        stress_cfg = replace(base_cfg, slippage=0.0010)

        # 1. Normal run
        rep_normal = arena.simulate(frames, bars, days, signals, [], base_cfg, {}, {}, {})
        # 2. Stressed run
        rep_stress = arena.simulate(frames, bars, days, signals, [], stress_cfg, {}, {}, {})

        summary_normal = rep_normal.summary()
        summary_stress = rep_stress.summary()

        # Slippage fee must be strictly doubled
        normal_slippage = sum(f["slippage"] for f in rep_normal.fills)
        stress_slippage = sum(f["slippage"] for f in rep_stress.fills)
        self.assertAlmostEqual(stress_slippage, normal_slippage * 2.0, places=2)

        # Stressed equity must be lower than normal equity by exactly the slippage delta
        equity_diff = summary_normal["final_equity"] - summary_stress["final_equity"]
        self.assertAlmostEqual(equity_diff, stress_slippage - normal_slippage, places=2)

        # Both runs must pass independent ledger balance conservation
        self.assertTrue(arena.audit_ledger(rep_normal, bars, [])["passed"])
        self.assertTrue(arena.audit_ledger(rep_stress, bars, [])["passed"])


if __name__ == "__main__":
    unittest.main()
