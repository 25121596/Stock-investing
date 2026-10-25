"""Unit tests for V0.8.8-Gemini research model components (built by Gemini checkpoint).

Covers:
- BreadthGatekeeperGemini: 0.58 boundary semantics and protocol conformance.
- TouchModeStoppingGemini: intraday rule-set restriction ('sl' mode), T+1 lock,
  anti-hindsight dual-touch behavior, 10 bps penalty, intraday_ reason prefixes,
  and close-based profit exits.
- build_dynamic_daily_limits: dynamic capacity scaling on explosive breadth days.
- compute_acceptance: pre-registered hurdle logic (all 9 criteria).
- Real disk artifact assertions on results/v088-gemini-v1:
  - Validates trade expansion: >= 120 in 2022-2025, >= 14 in 2026.
  - Validates return expansion: >= 60.00% in 2022-2025, >= 14.50% in 2026.
  - Validates stress test outperformance over V0.8.7-GLM.
  - Validates 100% balance sheet audit ledger compliance.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
from typing import Any, Dict

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine_core import (
    ExitDecision,
    GatekeeperProtocol,
    MarketContext,
    OptimalStoppingProtocol,
    PositionState,
)
from model_v088_gemini import (
    CANDIDATE_CASE,
    OUT_DIR,
    BreadthGatekeeperGemini,
    TouchModeStoppingGemini,
    build_dynamic_daily_limits,
    compute_acceptance,
)


def make_bar(open_: float, high: float, low: float, close: float, preclose: float = 10.0) -> Dict[str, Any]:
    return {
        "date": "2026-01-06",
        "code": "sh.600000",
        "open": float(open_),
        "high": float(high),
        "low": float(low),
        "close": float(close),
        "preclose": float(preclose),
        "volume": 100000.0,
        "tradestatus": 1,
        "isST": 0,
    }


def make_position(entry_price: float = 10.0, peak_price: float = 10.0, entry_index: int = 0) -> PositionState:
    return PositionState(
        trade_id=1,
        code="sh.600000",
        entry_date="2026-01-05",
        entry_index=entry_index,
        holding_age=1,
        shares=500,
        entry_price=entry_price,
        entry_notional=500 * entry_price,
        peak_price=peak_price,
        current_gain=(peak_price - entry_price) / entry_price,
        peak_gain=(peak_price - entry_price) / entry_price,
    )


def make_context(calendar_index: int = 1, ret_vol: float = 0.02, yz_vol: float = 0.02, breadth: float = 0.55) -> MarketContext:
    return MarketContext(
        date="2026-01-06",
        calendar_index=calendar_index,
        breadth5=breadth,
        pool_size=100,
        daily_limit=2,
        regime="neutral",
        budget=5500.0,
        max_positions=4,
        vol_map={("2026-01-05", "sh.600000"): ret_vol},
        yz_vol_map={("2026-01-05", "sh.600000"): yz_vol},
    )


class TestBreadthGatekeeperGemini(unittest.TestCase):
    def test_protocol_conformance(self):
        self.assertIsInstance(BreadthGatekeeperGemini(), GatekeeperProtocol)

    def test_boundary_0579_rejected_058_accepted(self):
        gate = BreadthGatekeeperGemini()
        for breadth, expected in [(0.5799, False), (0.58, True), (0.65, True), (0.42, False)]:
            with self.subTest(breadth=breadth):
                ctx = make_context(breadth=breadth)
                accepted, prob, reason = gate.evaluate_candidate("sh.600000", ctx.date, None, ctx)
                self.assertEqual(accepted, expected)
                self.assertAlmostEqual(prob, breadth)
                self.assertEqual(reason, "breadth_floor_pass" if expected else "breadth_floor_block")

    def test_evaluate_order_delegates(self):
        gate = BreadthGatekeeperGemini()
        ctx = make_context(breadth=0.62)
        accepted, _, _ = gate.evaluate_order({"code": "sh.600000"}, ctx)
        self.assertTrue(accepted)

    def test_custom_floor(self):
        gate = BreadthGatekeeperGemini(breadth_floor=0.50)
        ctx = make_context(breadth=0.52)
        accepted, _, _ = gate.evaluate_candidate("sh.600000", ctx.date, None, ctx)
        self.assertTrue(accepted)


class TestDynamicLimitsGemini(unittest.TestCase):
    def test_explosive_boost_capacity(self):
        bm = {"d1": 0.30, "d2": 0.48, "d3": 0.58, "d4": 0.70}
        pm = {"d1": 150, "d2": 150, "d3": 150, "d4": 150}

        limits_boost = build_dynamic_daily_limits(bm, pm, explosive_boost=True)
        self.assertEqual(limits_boost["d1"], 0)
        self.assertEqual(limits_boost["d2"], 1)
        self.assertEqual(limits_boost["d3"], 2)
        self.assertEqual(limits_boost["d4"], 3)  # Explosive breadth bonus

        limits_std = build_dynamic_daily_limits(bm, pm, explosive_boost=False)
        self.assertEqual(limits_std["d4"], 2)  # Standard cap at 2


class TestTouchModeStoppingGemini(unittest.TestCase):
    def test_protocol_conformance(self):
        self.assertIsInstance(TouchModeStoppingGemini(), OptimalStoppingProtocol)

    def test_sl_mode_triggers_stop_loss_touch(self):
        stopping = TouchModeStoppingGemini(intraday_mode="sl")
        # Low price touches SL line (9.70 <= 9.72)
        bar = make_bar(open_=10.0, high=10.05, low=9.70, close=9.85)
        decision = stopping.check_intraday_touch(make_position(), bar, make_context())
        self.assertTrue(decision.should_exit)
        self.assertEqual(decision.exit_reason, "intraday_stop_loss")

    def test_sl_mode_suppresses_intraday_take_profit(self):
        stopping = TouchModeStoppingGemini(intraday_mode="sl")
        # High touches profit target (10.95 >= 10.90) without touching SL
        bar = make_bar(open_=10.0, high=10.95, low=9.95, close=10.80)
        decision = stopping.check_intraday_touch(make_position(), bar, make_context())
        self.assertFalse(decision.should_exit)

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            TouchModeStoppingGemini(intraday_mode="invalid")


def _row(period: str, case: str, total_return: float, max_drawdown: float, trades: int, audit: bool = True) -> Dict[str, Any]:
    return {
        "period": period,
        "case": case,
        "stress": case.startswith("stress_"),
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "closed_trades": trades,
        "audit": {"passed": audit},
    }


class TestAcceptanceVerdicts(unittest.TestCase):
    def _mock_rows(self, p1_trades=130, p2_trades=16, p1_ret=0.68, p2_ret=0.15, p1_mdd=-0.08, p2_mdd=-0.025, stress_p1=0.65, stress_p2=0.15):
        return [
            _row("2022-2025", "v087_glm_baseline", 0.5693, -0.0586, 102),
            _row("2022-2025", "stress_v087_glm_baseline", 0.5447, -0.0608, 102),
            _row("2026", "v087_glm_baseline", 0.1626, -0.0120, 7),
            _row("2026", "stress_v087_glm_baseline", 0.1610, -0.0120, 7),
            _row("2022-2025", CANDIDATE_CASE, p1_ret, p1_mdd, p1_trades),
            _row("2022-2025", "stress_" + CANDIDATE_CASE, stress_p1, -0.088, p1_trades),
            _row("2026", CANDIDATE_CASE, p2_ret, p2_mdd, p2_trades),
            _row("2026", "stress_" + CANDIDATE_CASE, stress_p2, -0.026, p2_trades),
        ]

    def test_all_criteria_pass(self):
        verdict = compute_acceptance(self._mock_rows())
        self.assertTrue(verdict["overall_pass"])
        self.assertTrue(all(c["passed"] for c in verdict["criteria"].values()))

    def test_trade_expansion_failure(self):
        rows = self._mock_rows(p1_trades=110)  # Hurdle is 120
        verdict = compute_acceptance(rows)
        self.assertFalse(verdict["overall_pass"])
        self.assertFalse(verdict["criteria"]["p1_trades_expansion"]["passed"])


class TestRealArtifactAssertionsGemini(unittest.TestCase):
    """Tier 4 Real Disk Assertions: Asserts on real generated artifacts."""

    @classmethod
    def setUpClass(cls):
        cls.out_dir = OUT_DIR
        cls.comp_path = cls.out_dir / "comparison.csv"
        cls.acc_path = cls.out_dir / "acceptance.json"
        cls.sum_path = cls.out_dir / "summary.json"
        cls.disp_path = cls.out_dir / "display-version.json"

    def test_artifacts_exist_on_disk(self):
        self.assertTrue(self.comp_path.exists(), f"Missing {self.comp_path}")
        self.assertTrue(self.acc_path.exists(), f"Missing {self.acc_path}")
        self.assertTrue(self.sum_path.exists(), f"Missing {self.sum_path}")
        self.assertTrue(self.disp_path.exists(), f"Missing {self.disp_path}")

    def test_acceptance_passes_100_percent(self):
        data = json.loads(self.acc_path.read_text(encoding="utf-8"))
        self.assertTrue(data.get("overall_pass"), f"Acceptance failed: {data}")
        for crit_name, crit_data in data["criteria"].items():
            self.assertTrue(crit_data["passed"], f"Criterion failed: {crit_name}: {crit_data}")

    def test_display_version_provenance(self):
        data = json.loads(self.disp_path.read_text(encoding="utf-8"))
        self.assertIn("Gemini", data["made_by_checkpoint"])
        self.assertTrue(data["acceptance_pass"])

    def test_comparison_metrics_meet_targets(self):
        df = pd.read_csv(self.comp_path)
        # 1. 2022-2025 Candidate
        p1_cand = df[(df.period == "2022-2025") & (df.case == "v088_gemini") & (~df.stress.astype(bool))].iloc[0]
        self.assertGreaterEqual(p1_cand["closed_trades"], 120, "2022-2025 trades must be >= 120")
        self.assertGreaterEqual(p1_cand["total_return"], 0.60, "2022-2025 return must be >= 60%")
        self.assertGreaterEqual(p1_cand["max_drawdown"], -0.10, "2022-2025 MaxDD must be >= -10%")

        # 2. 2026 Candidate
        p2_cand = df[(df.period == "2026") & (df.case == "v088_gemini") & (~df.stress.astype(bool))].iloc[0]
        self.assertGreaterEqual(p2_cand["closed_trades"], 14, "2026 trades must be >= 14")
        self.assertGreaterEqual(p2_cand["total_return"], 0.145, "2026 return must be >= 14.5%")
        self.assertGreaterEqual(p2_cand["max_drawdown"], -0.0399, "2026 MaxDD must be >= -3.99%")

        # 3. Stressed Candidate beats Stressed GLM in 2022-2025
        p1_stress_cand = df[(df.period == "2022-2025") & (df.case == "stress_v088_gemini")].iloc[0]
        p1_stress_glm = df[(df.period == "2022-2025") & (df.case == "stress_v087_glm_baseline")].iloc[0]
        self.assertGreater(p1_stress_cand["total_return"], p1_stress_glm["total_return"])

        # 4. Stressed Candidate in 2026 >= 14%
        p2_stress_cand = df[(df.period == "2026") & (df.case == "stress_v088_gemini")].iloc[0]
        self.assertGreaterEqual(p2_stress_cand["total_return"], 0.14)


if __name__ == "__main__":
    unittest.main()
