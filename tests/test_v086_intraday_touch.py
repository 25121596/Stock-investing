"""Unit Test Suite for V0.8.6 Intraday Touch Execution & Plan A/B Stopping.

Validates:
1. Conformance to OptimalStoppingProtocol.
2. Clean Take-Profit and Stop-Loss touch triggers with 10 bps penalty slippage surcharge.
3. Strict Anti-Hindsight Stop-Loss Priority on Sky-and-Earth dual-touch.
4. Fills at Open on gaps (gap down below SL and gap up above TP).
5. Strict T+1 settlement enforcement.
6. Microstructure limit-down locked rejection.
7. Plan A dynamic 50% floating profit protection floor math.
8. Plan B tight 3.5% Moonbag floor lock and secondary trailing bound math.
9. ModularReplayEngine integration with balance sheet conservation and independent ledger audit.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import Config, Position, tick
from engine_core import (
    ExitDecision,
    MarketContext,
    ModularReplayEngine,
    OptimalStoppingProtocol,
    PositionState,
    audit_ledger,
)
from engine_intraday import (
    IntradayTouchStateMachine,
    IntradayTouchStopping,
    compute_intraday_stop_levels,
)


def make_test_bar(
    date: str = "2026-01-06",
    code: str = "sh.600000",
    open_: float = 10.0,
    high: float = 10.2,
    low: float = 9.8,
    close: float = 10.0,
    preclose: float = 10.0,
    volume: float = 100000.0,
    tradestatus: int = 1,
    isST: int = 0,
) -> dict:
    return {
        "date": date,
        "code": code,
        "open": float(open_),
        "high": float(high),
        "low": float(low),
        "close": float(close),
        "preclose": float(preclose),
        "volume": float(volume),
        "tradestatus": int(tradestatus),
        "isST": int(isST),
    }


def make_test_context(
    date: str = "2026-01-06",
    calendar_index: int = 1,
    code: str = "sh.600000",
    entry_date: str = "2026-01-05",
    ret_vol: float = 0.02,
    yz_vol: float = 0.02,
) -> MarketContext:
    return MarketContext(
        date=date,
        calendar_index=calendar_index,
        breadth5=0.55,
        pool_size=100,
        daily_limit=2,
        regime="neutral",
        budget=5500.0,
        max_positions=4,
        vol_map={(entry_date, code): ret_vol},
        yz_vol_map={(entry_date, code): yz_vol, (date, code): yz_vol},
    )


class TestIntradayTouchProtocolAndInit(unittest.TestCase):
    """Test protocol conformance and state machine initialization."""

    def test_protocol_runtime_checkable(self):
        stopping = IntradayTouchStopping()
        self.assertTrue(isinstance(stopping, OptimalStoppingProtocol))

    def test_invalid_plan_mode_raises(self):
        with self.assertRaises(ValueError):
            IntradayTouchStateMachine(plan_mode="plan_c")

    def test_compute_intraday_stop_levels_clamping(self):
        # ret_vol = 0.01: eff_sl clamps to min 0.025, eff_tp clamps to min 0.090
        sl1, tp1, p_sl1, p_tp1 = compute_intraday_stop_levels(10.0, 0.01)
        self.assertAlmostEqual(sl1, 0.025)
        self.assertAlmostEqual(tp1, 0.090)
        self.assertEqual(p_sl1, 9.75)
        self.assertEqual(p_tp1, 10.90)

        # ret_vol = 0.05: eff_sl clamps to max 0.045, eff_tp clamps to max 0.160
        sl2, tp2, p_sl2, p_tp2 = compute_intraday_stop_levels(10.0, 0.05)
        self.assertAlmostEqual(sl2, 0.045)
        self.assertAlmostEqual(tp2, 0.160)
        self.assertEqual(p_sl2, 9.55)
        self.assertEqual(p_tp2, 11.60)


class TestIntradayTouchTriggersAndPriority(unittest.TestCase):
    """Test take-profit, stop-loss, anti-hindsight priority, and gap pricing."""

    def setUp(self):
        self.sm = IntradayTouchStateMachine()
        self.ctx = make_test_context(calendar_index=1, ret_vol=0.02)
        # Entry price 10.00, entry index 0 (T+1 satisfied)
        # ret_vol = 0.02 -> eff_sl = 0.028 -> P_SL = 9.72; eff_tp = 0.090 -> P_TP = 10.90
        self.pos = PositionState(
            trade_id=1,
            code="sh.600000",
            entry_date="2026-01-05",
            entry_index=0,
            holding_age=1,
            shares=500,
            entry_price=10.00,
            entry_notional=5000.0,
            peak_price=10.00,
            current_gain=0.0,
            peak_gain=0.0,
        )

    def test_clean_take_profit_touch(self):
        bar = make_test_bar(open_=10.20, high=10.95, low=10.10, close=10.80)
        decision = self.sm.evaluate_bar(self.pos, bar, self.ctx)
        self.assertIsNotNone(decision)
        self.assertTrue(decision.should_exit)
        self.assertEqual(decision.exit_type, "intraday_touch")
        self.assertEqual(decision.exit_reason, "take_profit")
        self.assertEqual(decision.target_price, 10.90)
        self.assertEqual(decision.slippage_bps_penalty, 0.0010)

    def test_clean_stop_loss_touch(self):
        bar = make_test_bar(open_=9.90, high=10.05, low=9.70, close=9.75)
        decision = self.sm.evaluate_bar(self.pos, bar, self.ctx)
        self.assertIsNotNone(decision)
        self.assertTrue(decision.should_exit)
        self.assertEqual(decision.exit_type, "intraday_touch")
        self.assertEqual(decision.exit_reason, "stop_loss")
        self.assertEqual(decision.target_price, 9.72)
        self.assertEqual(decision.slippage_bps_penalty, 0.0010)

    def test_sky_and_earth_dual_touch_anti_hindsight_priority(self):
        # High touches 11.20 >= 10.90 (TP), Low touches 9.60 <= 9.72 (SL)
        bar = make_test_bar(open_=10.10, high=11.20, low=9.60, close=9.80)
        decision = self.sm.evaluate_bar(self.pos, bar, self.ctx)
        self.assertIsNotNone(decision)
        self.assertTrue(decision.should_exit)
        # Strict priority: stop loss must execute, take profit must be completely discarded
        self.assertEqual(decision.exit_reason, "stop_loss_intraday_priority")
        self.assertEqual(decision.target_price, 9.72)
        self.assertEqual(decision.slippage_bps_penalty, 0.0010)

    def test_gap_down_below_stop_loss_fills_at_open(self):
        # P_SL = 9.72. Stock gaps down and opens at 9.40 <= 9.72.
        bar = make_test_bar(open_=9.40, high=9.55, low=9.30, close=9.45)
        decision = self.sm.evaluate_bar(self.pos, bar, self.ctx)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.exit_reason, "stop_loss")
        self.assertEqual(decision.target_price, 9.40)
        self.assertEqual(decision.slippage_bps_penalty, 0.0010)

    def test_gap_up_above_take_profit_fills_at_open(self):
        # P_TP = 10.90. Stock gaps up and opens at 11.25 >= 10.90.
        bar = make_test_bar(open_=11.25, high=11.60, low=11.10, close=11.50)
        decision = self.sm.evaluate_bar(self.pos, bar, self.ctx)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.exit_reason, "take_profit")
        self.assertEqual(decision.target_price, 11.25)
        self.assertEqual(decision.slippage_bps_penalty, 0.0010)

    def test_sky_and_earth_gap_down_fills_at_open(self):
        # Gap down open at 9.40 (< 9.72), intra-day surge to High 11.30 (>= 10.90)
        bar = make_test_bar(open_=9.40, high=11.30, low=9.35, close=10.60)
        decision = self.sm.evaluate_bar(self.pos, bar, self.ctx)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.exit_reason, "stop_loss_intraday_priority")
        self.assertEqual(decision.target_price, 9.40)
        self.assertEqual(decision.slippage_bps_penalty, 0.0010)


class TestIntradayMicrostructureAndTPlusOne(unittest.TestCase):
    """Test T+1 settlement enforcement and limit-down locking."""

    def setUp(self):
        self.sm = IntradayTouchStateMachine()
        self.ctx_t0 = make_test_context(calendar_index=0, ret_vol=0.02)
        self.ctx_t1 = make_test_context(calendar_index=1, ret_vol=0.02)
        self.pos = PositionState(
            trade_id=1,
            code="sh.600000",
            entry_date="2026-01-05",
            entry_index=0,
            holding_age=0,
            shares=500,
            entry_price=10.00,
            entry_notional=5000.0,
            peak_price=10.00,
            current_gain=0.0,
            peak_gain=0.0,
        )

    def test_t_plus_one_same_day_exit_blocked(self):
        # On day of entry (calendar_index 0 == entry_index 0), price plunges to 9.00
        bar = make_test_bar(open_=10.0, high=10.0, low=9.00, close=9.10)
        decision = self.sm.evaluate_bar(self.pos, bar, self.ctx_t0)
        self.assertIsNone(decision, "T+1 rule must block any intraday exit on entry day")

    def test_limit_down_locked_rejects_sell(self):
        # Normal stock preclose 10.0 -> Limit down = 9.00. High == Low == 9.00
        bar = make_test_bar(open_=9.00, high=9.00, low=9.00, close=9.00, preclose=10.0)
        decision = self.sm.evaluate_bar(self.pos, bar, self.ctx_t1)
        self.assertIsNone(decision, "Must reject sell order when stock is locked at limit down")

    def test_st_limit_down_locked_rejects_sell(self):
        # ST stock preclose 10.0 -> 5% Limit down = 9.50. High == Low == 9.50
        bar = make_test_bar(open_=9.50, high=9.50, low=9.50, close=9.50, preclose=10.0, isST=1)
        decision = self.sm.evaluate_bar(self.pos, bar, self.ctx_t1)
        self.assertIsNone(decision, "Must reject sell order when ST stock is locked at 5% limit down")

    def test_limit_down_unlocked_can_sell(self):
        # Stock touches limit down 9.00, but high is 9.50 (not locked)
        bar = make_test_bar(open_=9.30, high=9.50, low=9.00, close=9.10, preclose=10.0)
        decision = self.sm.evaluate_bar(self.pos, bar, self.ctx_t1)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.exit_reason, "stop_loss")


class TestPlanAvsPlanBMath(unittest.TestCase):
    """Test Plan A dynamic 50% profit floor and Plan B tight 3.5% floor & trailing bound."""

    def test_plan_a_activation_and_floor_math(self):
        sm_a = IntradayTouchStateMachine(plan_mode="plan_a")
        ctx = make_test_context(calendar_index=1, ret_vol=0.02, yz_vol=0.02)
        # Threshold: 2.5 * yz_vol = 2.5 * 0.02 = 0.050 (5.0%)

        # Peak gain = 4.0% <= 5.0% -> Inactive
        pos_inactive = PositionState(
            trade_id=1, code="sh.600000", entry_date="2026-01-05", entry_index=0, holding_age=1,
            shares=500, entry_price=10.00, entry_notional=5000.0, peak_price=10.40,
            current_gain=0.02, peak_gain=0.04,
        )
        bar1 = make_test_bar(open_=10.20, high=10.30, low=10.15, close=10.20)
        self.assertIsNone(sm_a.evaluate_bar(pos_inactive, bar1, ctx))

        # Peak gain = 10.0% > 5.0% -> Active!
        # P_floor_A = 0.5 * (10.00 + 11.00) = 10.50
        pos_active = PositionState(
            trade_id=2, code="sh.600000", entry_date="2026-01-05", entry_index=0, holding_age=1,
            shares=500, entry_price=10.00, entry_notional=5000.0, peak_price=11.00,
            current_gain=0.08, peak_gain=0.10,
        )
        # Drop touches 10.45 <= 10.50
        bar2 = make_test_bar(open_=10.70, high=10.80, low=10.45, close=10.48)
        decision = sm_a.evaluate_bar(pos_active, bar2, ctx)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.exit_reason, "plan_a_floor")
        self.assertEqual(decision.target_price, 10.50)
        self.assertEqual(decision.slippage_bps_penalty, 0.0010)

    def test_plan_b_moonbag_floor_and_trailing_bound(self):
        sm_b = IntradayTouchStateMachine(plan_mode="plan_b")
        ctx = make_test_context(calendar_index=1, ret_vol=0.02, yz_vol=0.02)
        # Moonbag threshold: max(0.075, 2.2 * 0.02) = 0.075 (7.5%)

        # Moonbag active position. Entry = 10.00. P_floor_B = 1.035 * 10.0 = 10.35
        pos_moonbag = PositionState(
            trade_id=1, code="sh.600000", entry_date="2026-01-05", entry_index=0, holding_age=1,
            shares=500, entry_price=10.00, entry_notional=5000.0, peak_price=10.85,
            current_gain=0.05, peak_gain=0.085, moonbag_active=True,
        )
        # Drops to 10.30 <= 10.35 -> Triggers plan_b_floor
        bar1 = make_test_bar(open_=10.50, high=10.60, low=10.30, close=10.32)
        decision1 = sm_b.evaluate_bar(pos_moonbag, bar1, ctx)
        self.assertIsNotNone(decision1)
        self.assertEqual(decision1.exit_reason, "plan_b_floor")
        self.assertEqual(decision1.target_price, 10.35)
        self.assertEqual(decision1.slippage_bps_penalty, 0.0010)

        # Secondary Trailing Hurdle: peak_gain >= 0.080
        # Use ret_vol = 0.04 so eff_tp = 0.160 -> P_TP = 11.60. Then High 11.50 does not clip TP.
        ctx_trail = make_test_context(calendar_index=1, ret_vol=0.04, yz_vol=0.02)
        # Peak = 11.50 (15% gain). P_trail = 11.50 * (1 - max(0.02, 0.02)) = 11.50 * 0.98 = 11.27
        pos_trail = PositionState(
            trade_id=2, code="sh.600000", entry_date="2026-01-05", entry_index=0, holding_age=1,
            shares=500, entry_price=10.00, entry_notional=5000.0, peak_price=11.50,
            current_gain=0.14, peak_gain=0.15, moonbag_active=True,
        )
        # Drops to 11.20 <= 11.27, but well above 10.35
        bar2 = make_test_bar(open_=11.40, high=11.50, low=11.20, close=11.25)
        decision2 = sm_b.evaluate_bar(pos_trail, bar2, ctx_trail)
        self.assertIsNotNone(decision2)
        self.assertEqual(decision2.exit_reason, "vol_trailing_stop")
        self.assertEqual(decision2.target_price, 11.27)
        self.assertEqual(decision2.slippage_bps_penalty, 0.0010)


class TestModularEngineIntegrationWithIntradayTouch(unittest.TestCase):
    """Test full integration between ModularReplayEngine and IntradayTouchStopping."""

    def test_engine_intraday_touch_execution_and_conservation(self):
        days = ["2026-02-02", "2026-02-03", "2026-02-04", "2026-02-05"]
        code = "sh.600000"
        # Day 1: open 10.0, buy queued
        # Day 2: open 10.0, buy fills at open. High 10.2, Low 9.8.
        # Day 3: High reaches 10.95 (touches TP 10.90), Low is 10.10. Intraday touch liquidation!
        # Day 4: Cash only
        bars_df = pd.DataFrame([
            make_test_bar(days[0], code, open_=10.0, high=10.2, low=9.8, close=10.0, preclose=10.0),
            make_test_bar(days[1], code, open_=10.0, high=10.2, low=9.8, close=10.1, preclose=10.0),
            make_test_bar(days[2], code, open_=10.2, high=10.95, low=10.1, close=10.8, preclose=10.1),
            make_test_bar(days[3], code, open_=10.8, high=11.0, low=10.7, close=10.9, preclose=10.8),
        ])
        frames = {code: bars_df}
        signals = pd.DataFrame([{"date": days[0], "code": code, "score": 1.0}])

        stopping = IntradayTouchStopping(plan_mode="plan_a")
        engine = ModularReplayEngine(
            frames=frames,
            calendar=days,
            signals=signals,
            config=Config(capital=30000.0, order_budget=6000.0, max_positions=1),
            stopping=stopping,
            enable_intraday_touch=True,
            vol_map={("2026-02-03", code): 0.02},
            yz_vol_map={("2026-02-03", code): 0.02, ("2026-02-04", code): 0.02},
        )
        engine.run()

        # Check fills
        self.assertEqual(len(engine.fills), 2)
        buy_fill = engine.fills[0]
        sell_fill = engine.fills[1]

        self.assertEqual(buy_fill["side"], "buy")
        self.assertEqual(buy_fill["price"], 10.0)

        self.assertEqual(sell_fill["side"], "sell")
        self.assertEqual(sell_fill["date"], days[2])
        self.assertEqual(sell_fill["price"], 10.90)

        # Slippage on sell must include base 0.0005 + 0.0010 penalty = 0.0015
        expected_slippage = tick(sell_fill["notional"] * (0.0005 + 0.0010))
        self.assertEqual(sell_fill["slippage"], expected_slippage)

        # Audit ledger balance conservation
        audit = audit_ledger(engine, {code: bars_df.set_index("date").to_dict("index")}, [])
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["daily_nav_checked"], len(days))


if __name__ == "__main__":
    unittest.main()
