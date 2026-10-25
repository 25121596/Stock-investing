"""Unit tests for the V0.8.7-GLM research candidate components (built by GLM checkpoint).

Covers:
- BreadthGatekeeperGLM: 0.58 boundary semantics (V0.9 fixed rule) and protocol conformance.
- TouchModeStoppingGLM: intraday rule-set restriction ('sl' / 'tp' / 'all'),
  T+1 lock, limit-down lock, anti-hindsight dual-touch behaviour, 10 bps penalty,
  intraday_ reason prefixes, and that profit exits stay close-based in 'sl' mode.
- compute_acceptance / compute_damping_verdict: pre-registered verdict logic.
"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine_core import ExitDecision, GatekeeperProtocol, MarketContext, OptimalStoppingProtocol, PositionState
from model_v087_glm import (
    CANDIDATE_CASE,
    BreadthGatekeeperGLM,
    TouchModeStoppingGLM,
    compute_acceptance,
    compute_damping_verdict,
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


class TestBreadthGatekeeperGLM(unittest.TestCase):
    def test_protocol_conformance(self):
        self.assertIsInstance(BreadthGatekeeperGLM(), GatekeeperProtocol)

    def test_boundary_0579_rejected_058_accepted(self):
        gate = BreadthGatekeeperGLM()
        for breadth, expected in [(0.5799, False), (0.58, True), (0.62, True), (0.44, False)]:
            with self.subTest(breadth=breadth):
                ctx = make_context(breadth=breadth)
                accepted, prob, reason = gate.evaluate_candidate("sh.600000", ctx.date, None, ctx)
                self.assertEqual(accepted, expected)
                self.assertAlmostEqual(prob, breadth)
                self.assertEqual(reason, "breadth_floor_pass" if expected else "breadth_floor_block")

    def test_evaluate_order_delegates(self):
        gate = BreadthGatekeeperGLM()
        ctx = make_context(breadth=0.60)
        accepted, _, _ = gate.evaluate_order({"code": "sh.600000"}, ctx)
        self.assertTrue(accepted)

    def test_custom_floor(self):
        gate = BreadthGatekeeperGLM(breadth_floor=0.45)
        ctx = make_context(breadth=0.46)
        accepted, _, _ = gate.evaluate_candidate("sh.600000", ctx.date, None, ctx)
        self.assertTrue(accepted)


class TestTouchModeStoppingGLMSLMode(unittest.TestCase):
    """mode='sl': only the stop-loss leg executes intraday."""

    def setUp(self):
        self.stopping = TouchModeStoppingGLM(intraday_mode="sl")

    def test_protocol_conformance(self):
        self.assertIsInstance(self.stopping, OptimalStoppingProtocol)

    def test_stop_loss_touch_fires_intraday_with_penalty_and_prefix(self):
        # eff_sl = clamp(1.4*0.02, 2.5%, 4.5%) = 2.8% -> p_sl = 9.72
        decision = self.stopping.check_intraday_touch(make_position(), make_bar(10.0, 10.10, 9.70, 9.95), make_context())
        self.assertTrue(decision.should_exit)
        self.assertEqual(decision.exit_reason, "intraday_stop_loss")
        self.assertAlmostEqual(decision.slippage_bps_penalty, 0.0010)
        self.assertAlmostEqual(decision.target_price, 9.72, places=6)

    def test_take_profit_touch_suppressed_intraday(self):
        # eff_tp = 9.0% -> p_tp = 10.90; high 10.95 touches, low 10.10 never near p_sl
        decision = self.stopping.check_intraday_touch(make_position(), make_bar(10.0, 10.95, 10.10, 10.80), make_context())
        self.assertFalse(decision.should_exit)

    def test_plan_a_floor_suppressed_intraday(self):
        # peak 11.00 -> peak_gain 10% > 2.5*yz_vol(5%); floor = 0.5*(10+11) = 10.50
        pos = make_position(peak_price=11.0)
        decision = self.stopping.check_intraday_touch(pos, make_bar(10.60, 10.60, 10.45, 10.50), make_context())
        self.assertFalse(decision.should_exit)

    def test_close_based_take_profit_still_fires(self):
        # Profit exits stay close-based: close gain 10% >= eff_tp 9% -> close_signal
        pos = make_position()
        decision = self.stopping.check_close_stopping(pos, make_bar(10.0, 11.05, 10.10, 11.00), make_context())
        self.assertTrue(decision.should_exit)
        self.assertEqual(decision.exit_type, "close_signal")
        self.assertEqual(decision.exit_reason, "take_profit")
        self.assertTrue(pos.stop_pending)

    def test_t_plus_one_lock(self):
        decision = self.stopping.check_intraday_touch(
            make_position(entry_index=1), make_bar(10.0, 10.10, 9.50, 9.60), make_context(calendar_index=1)
        )
        self.assertFalse(decision.should_exit)

    def test_limit_down_locked_bar_rejected(self):
        # One-word limit-down bar: high == low == preclose*(1-10%) = 9.00
        decision = self.stopping.check_intraday_touch(
            make_position(), make_bar(9.00, 9.00, 9.00, 9.00, preclose=10.0), make_context()
        )
        self.assertFalse(decision.should_exit)

    def test_gap_down_below_stop_fills_at_open(self):
        decision = self.stopping.check_intraday_touch(
            make_position(), make_bar(9.50, 9.60, 9.45, 9.55), make_context()
        )
        self.assertTrue(decision.should_exit)
        self.assertAlmostEqual(decision.target_price, 9.50, places=6)


class TestTouchModeStoppingGLMTPAndAllModes(unittest.TestCase):
    def test_tp_mode_fires_take_profit_with_prefix(self):
        stopping = TouchModeStoppingGLM(intraday_mode="tp")
        decision = stopping.check_intraday_touch(make_position(), make_bar(10.0, 10.95, 10.10, 10.80), make_context())
        self.assertTrue(decision.should_exit)
        self.assertEqual(decision.exit_reason, "intraday_take_profit")

    def test_tp_mode_suppresses_stop_loss(self):
        stopping = TouchModeStoppingGLM(intraday_mode="tp")
        decision = stopping.check_intraday_touch(make_position(), make_bar(10.0, 10.10, 9.70, 9.95), make_context())
        self.assertFalse(decision.should_exit)

    def test_tp_mode_suppresses_dual_touch_day_anti_hindsight(self):
        # Both lines touched (high 11.00 >= 10.90, low 9.60 <= 9.72): the underlying
        # state machine returns stop-loss priority; 'tp' mode must defer the day.
        stopping = TouchModeStoppingGLM(intraday_mode="tp")
        decision = stopping.check_intraday_touch(make_position(), make_bar(10.0, 11.00, 9.60, 10.20), make_context())
        self.assertFalse(decision.should_exit)

    def test_all_mode_replicates_v086_priority_and_labels(self):
        stopping = TouchModeStoppingGLM(intraday_mode="all")
        decision = stopping.check_intraday_touch(make_position(), make_bar(10.0, 11.00, 9.60, 10.20), make_context())
        self.assertTrue(decision.should_exit)
        self.assertEqual(decision.exit_reason, "intraday_stop_loss_intraday_priority")

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            TouchModeStoppingGLM(intraday_mode="nope")


def _row(period: str, case: str, total_return: float, max_drawdown: float) -> Dict[str, Any]:
    return {"period": period, "case": case, "stress": period.startswith("stress_") or case.startswith("stress_"), "total_return": total_return, "max_drawdown": max_drawdown}


class TestAcceptanceAndDampingVerdicts(unittest.TestCase):
    def _full_rows(self, cand_p1=0.50, cand_p2=0.16, stress_c1=0.45, stress_c2=0.16):
        return [
            _row("2022-2025", "v08_baseline", 0.4279, -0.0832),
            _row("2022-2025", CANDIDATE_CASE, cand_p1, -0.06),
            _row("2026", "v08_baseline", 0.1573, -0.0399),
            _row("2026", CANDIDATE_CASE, cand_p2, -0.02),
            _row("2022-2025", "stress_v08_baseline", 0.3770, -0.1139),
            _row("2022-2025", "stress_" + CANDIDATE_CASE, stress_c1, -0.07),
            _row("2026", "stress_v08_baseline", 0.1548, -0.0408),
            _row("2026", "stress_" + CANDIDATE_CASE, stress_c2, -0.02),
        ]

    def test_all_criteria_pass(self):
        verdict = compute_acceptance(self._full_rows())
        self.assertTrue(verdict["overall_pass"])
        self.assertTrue(all(c["passed"] for c in verdict["criteria"].values()))

    def test_p1_return_below_hurdle_fails(self):
        verdict = compute_acceptance(self._full_rows(cand_p1=0.44))
        self.assertFalse(verdict["overall_pass"])
        self.assertFalse(verdict["criteria"]["p1_return_gt_45pct"]["passed"])
        self.assertTrue(verdict["criteria"]["p2_return_gt_15pct"]["passed"])

    def test_p2_drawdown_worse_than_hurdle_fails(self):
        verdict = compute_acceptance(self._full_rows(cand_p2=0.16))
        rows = self._full_rows()
        rows[3]["max_drawdown"] = -0.045
        verdict = compute_acceptance(rows)
        self.assertFalse(verdict["criteria"]["p2_maxdd_not_worse_than_-3.99pct"]["passed"])

    def test_missing_stress_row_fails(self):
        rows = self._full_rows()
        rows = [r for r in rows if r["case"] != "stress_" + CANDIDATE_CASE]
        verdict = compute_acceptance(rows)
        self.assertFalse(verdict["overall_pass"])
        self.assertTrue(verdict["criteria"]["stress_p1_beats_stressed_baseline"]["missing_row"])

    def test_stress_below_stressed_baseline_fails(self):
        verdict = compute_acceptance(self._full_rows(stress_c1=0.36))
        self.assertFalse(verdict["criteria"]["stress_p1_beats_stressed_baseline"]["passed"])

    def test_damping_keep(self):
        rows = self._full_rows(cand_p1=0.50) + [
            _row("2022-2025", "v087_glm_damping", 0.46, -0.06),
            _row("2026", "v087_glm_damping", 0.16, -0.02),
        ]
        verdict = compute_damping_verdict(rows)
        self.assertEqual(verdict["verdict"], "KEEP_DAMPING")

    def test_damping_reject_when_p1_gives_up_too_much(self):
        rows = self._full_rows(cand_p1=0.50) + [
            _row("2022-2025", "v087_glm_damping", 0.33, -0.06),  # < 90% of 0.50
            _row("2026", "v087_glm_damping", 0.16, -0.02),
        ]
        verdict = compute_damping_verdict(rows)
        self.assertEqual(verdict["verdict"], "REJECT_DAMPING")

    def test_damping_reject_when_p2_below_baseline(self):
        rows = self._full_rows(cand_p1=0.50) + [
            _row("2022-2025", "v087_glm_damping", 0.48, -0.06),
            _row("2026", "v087_glm_damping", 0.14, -0.02),  # < baseline 0.1573
        ]
        verdict = compute_damping_verdict(rows)
        self.assertEqual(verdict["verdict"], "REJECT_DAMPING")


if __name__ == "__main__":
    unittest.main()
