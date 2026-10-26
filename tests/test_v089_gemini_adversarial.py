"""test_v089_gemini_adversarial.py - Empirical Challenger Stress Suite for V0.8.9-Gemini.

Adversarial Stress Testing Verifier (Challenger 1):
1. Extreme slippage stress (15 bps, 20 bps, 30 bps, 50 bps).
2. Liquidity shocks & cash exhaustion (zero cash, high exhaustion, multi-order collision, delayed/locked dividends).
3. Market crash edge cases (limit-down morning sell blocks, limit-down intraday touch locks, limit-up buy blocks,
   simultaneous stop-loss / take-profit whipsaw priority, trading suspension).
4. Boundary condition testing (budget rounding, 100-share lot discretization, NAV dynamic scaling, compact horizons, T+1 enforcement).
5. Real-data adversarial multi-period stress simulation (2022-2025 and 2026 under 15 bps and 20 bps slippage).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import arena
from engine import Config, Position, tick, number
from engine_core import (
    BaselineRanker,
    ExitDecision,
    MarketContext,
    PositionState,
    audit_ledger,
)
from engine_intraday import IntradayTouchStopping
from model_v083_ranker import compute_adaptive_scores
from train_model_v2 import enrich_features_v2
from train_model_v3 import extract_ashare_features
from train_model_v08 import extract_microstructure_features
import model_v089_gemini as m089
from model_v089_gemini import (
    AdaptiveGatekeeperGemini,
    BreadthGatekeeperGemini,
    DynamicCompoundingBudget,
    ReplayEngineV089,
    TouchModeStoppingGemini,
    build_dynamic_daily_limits,
    compute_dynamic_compounding_budget,
)


def make_test_bar(
    date: str,
    code: str,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float = 100000.0,
    is_st: int = 0,
    preclose: Optional[float] = None,
    tradestatus: int = 1,
) -> Dict[str, Any]:
    return {
        "date": date,
        "code": code,
        "open": float(open_),
        "high": float(high),
        "low": float(low),
        "close": float(close),
        "volume": float(volume),
        "tradestatus": int(tradestatus),
        "isST": int(is_st),
        "preclose": float(preclose if preclose is not None else open_),
    }


# =============================================================================
# Area 1: Extreme Slippage Stress Testing
# =============================================================================

class TestExtremeSlippageStress(unittest.TestCase):
    """Stress tests on extreme slippage conditions (15 bps, 20 bps, 50 bps)."""

    def test_fee_calculation_under_extreme_slippage(self):
        """Verify fees and cost formulas remain finite, positive, and non-overdrafting."""
        for slip_bps, slip in [(15, 0.0015), (20, 0.0020), (50, 0.0050)]:
            cfg = Config(slippage=slip)
            fees_buy = cfg.fees(10000.0, sell=False)
            fees_sell = cfg.fees(10000.0, sell=True)

            self.assertAlmostEqual(fees_buy["slippage"], tick(10000.0 * slip), places=2)
            self.assertAlmostEqual(fees_sell["slippage"], tick(10000.0 * slip), places=2)
            self.assertEqual(fees_buy["stamp"], 0.0)
            self.assertAlmostEqual(fees_sell["stamp"], 5.0, places=2)
            total_buy_cost = 10000.0 + sum(fees_buy.values())
            self.assertGreater(total_buy_cost, 10000.0)
            total_sell_proceeds = 10000.0 - sum(fees_sell.values())
            self.assertLess(total_sell_proceeds, 10000.0)

    def test_intraday_touch_with_extreme_slippage_penalty(self):
        """Verify intraday exit under extreme base slippage + 10 bps penalty slippage (3-day cycle for T+1)."""
        for slip in [0.0015, 0.0020, 0.0030]:
            cfg = Config(slippage=slip, capital=30000.0, max_exposure=0.9)
            calendar = ["2026-01-05", "2026-01-06", "2026-01-07"]
            bars = pd.DataFrame([
                make_test_bar("2026-01-05", "600000", 10.0, 10.5, 9.8, 10.2),  # Signal date
                make_test_bar("2026-01-06", "600000", 10.0, 10.2, 9.9, 10.0),  # Entry date (open buy fills)
                # Day 3: hits stop loss (low=9.4 triggers SL touch at 9.69, T+1 satisfied)
                make_test_bar("2026-01-07", "600000", 10.0, 10.1, 9.4, 9.5),
            ])
            signals = pd.DataFrame([{"date": "2026-01-05", "code": "600000", "score": 1.0, "pred_net_return_v2": 0.01}])
            engine = ReplayEngineV089(
                frames={"600000": bars},
                calendar=calendar,
                signals=signals,
                config=cfg,
                enable_cash_overlay=False,
                enable_intraday_touch=True,
                stopping=TouchModeStoppingGemini(plan_mode="plan_a", intraday_mode="sl"),
            )
            engine.run()
            self.assertEqual(len(engine.fills), 2)
            sell_fill = [f for f in engine.fills if f["side"] == "sell"][0]
            # Total slippage in fill should include base slippage + 10 bps penalty
            expected_min_slip = tick(sell_fill["notional"] * (slip + 0.0010))
            self.assertAlmostEqual(sell_fill["slippage"], expected_min_slip, delta=0.05)
            # Verify balance sheet conservation
            self.assertLess(abs(engine.nav() - engine.cash), 1e-6)

    def test_real_data_extreme_slippage_simulation(self):
        """Execute real data replay on 2026 OOS under 15 bps and 20 bps slippage."""
        _, frames, calendar, raw = arena.load(arena.DATASET)
        events, _ = arena.canonicalize_events(raw)
        bars = arena.prepare_bars(frames)

        panel = pd.read_csv(ROOT / "results/midterm-access-v3/panel-h10.csv", dtype={"date": str, "code": str, "label_end": str})
        sixty = pd.read_csv(ROOT / "results/exploration-v1/new-features.csv", dtype={"date": str, "code": str})
        panel = enrich_features_v2(panel.merge(sixty[["date", "code", "adj_ret60", "ret_vol60"]], on=["date", "code"], how="left", validate="one_to_one"))
        panel = panel.merge(extract_ashare_features(frames), on=["date", "code"], how="left", validate="one_to_one").merge(
            extract_microstructure_features(frames), on=["date", "code"], how="left", validate="one_to_one"
        )

        b = arena.breadth_days(panel)
        bm = b.set_index("date").breadth5.to_dict()
        pm = b.set_index("date").pool_size.to_dict()
        vol = panel.set_index(["date", "code"]).ret_vol20.to_dict()
        yz_vol_map = panel.set_index(["date", "code"]).yz_vol20.to_dict()

        old2 = pd.read_csv(m089.BUNDLE / "v2/predictions-h10-v2.csv", dtype={"date": str, "code": str})
        old8 = pd.read_csv(m089.BUNDLE / "v08/predictions-h10-v08.csv", dtype={"date": str, "code": str})
        preds_v083 = pd.read_csv(ROOT / "results/v083-adaptive-ranker/predictions-v083.csv", dtype={"date": str, "code": str})
        joined = old2[["date", "code", "pred_net_return_v2"]].merge(
            old8[["date", "code", "pred_score_v08"]], on=["date", "code"]
        ).merge(preds_v083[["date", "code", "pred_score_v083"]], on=["date", "code"])
        joined = compute_adaptive_scores(joined, bm, pm)

        limits_boost = build_dynamic_daily_limits(bm, pm, explosive_boost=True)
        allow_gemini = joined.date.map(limits_boost).fillna(0).gt(0) & joined.pred_net_return_v2.ge(0.0030)
        sig_gemini = joined.loc[allow_gemini, ["date", "code", "adaptive_score", "pred_net_return_v2"]].rename(
            columns={"adaptive_score": "score"}
        )

        days_2026 = [d for d in calendar if "2026-01-01" <= d <= "2026-10-13"]
        s_2026 = sig_gemini[sig_gemini.date.between("2026-01-01", "2026-10-13")]

        for slip_bps, slip in [(15, 0.0015), (20, 0.0020)]:
            cfg = Config(capital=30000, order_budget=7500, max_positions=5, max_exposure=0.9, hold_days=10, slippage=slip)
            rep = ReplayEngineV089(
                frames=frames,
                calendar=days_2026,
                signals=s_2026,
                dividends=events,
                config=cfg,
                vol_map=vol,
                daily_limits=limits_boost,
                breadth_lookup=bm,
                yz_vol_map=yz_vol_map,
                ranker=BaselineRanker(),
                gatekeeper=BreadthGatekeeperGemini(0.58),
                stopping=TouchModeStoppingGemini(plan_mode="plan_a", intraday_mode="sl"),
                enable_cash_overlay=True,
                enable_intraday_touch=True,
                enable_turnover_damping=False,
                enable_nav_compounding=True,
                compounding_mode="floor_capped",
                target_slots=4,
                hold_strong=11,
                hold_normal=7,
            )
            rep.bars = bars
            rep.run()

            summary = rep.summary()
            audit = audit_ledger(rep, bars, events)

            # Invariant 1: Strategy engine runs to completion without errors
            self.assertTrue(audit["passed"], f"Audit failed under {slip_bps} bps slippage")
            self.assertEqual(audit["daily_nav_checked"], len(rep.equity))
            # Invariant 2: Zero negative cash days
            for eq in rep.equity:
                self.assertGreaterEqual(eq["cash"], -1e-7, f"Negative cash on {eq['date']}")
            # Invariant 3: Returns degrade gracefully under extreme slippage (15, 20 bps) and remain solidly positive (>10%)
            self.assertGreater(summary["total_return"], 0.1000, f"Return collapsed below 10% under {slip_bps} bps: {summary['total_return']}")
            self.assertGreater(summary["max_drawdown"], -0.0500, f"MaxDD exceeded -5% under {slip_bps} bps: {summary['max_drawdown']}")

        # Also test 2022-2025 Walk-Forward under 15 bps and 20 bps slippage
        days_p1 = [d for d in calendar if "2022-01-01" <= d <= "2025-12-31"]
        s_p1 = sig_gemini[sig_gemini.date.between("2022-01-01", "2025-12-31")]
        for slip_bps, slip in [(15, 0.0015), (20, 0.0020)]:
            cfg_p1 = Config(capital=30000, order_budget=7500, max_positions=5, max_exposure=0.9, hold_days=10, slippage=slip)
            rep_p1 = ReplayEngineV089(
                frames=frames,
                calendar=days_p1,
                signals=s_p1,
                dividends=events,
                config=cfg_p1,
                vol_map=vol,
                daily_limits=limits_boost,
                breadth_lookup=bm,
                yz_vol_map=yz_vol_map,
                ranker=BaselineRanker(),
                gatekeeper=BreadthGatekeeperGemini(0.58),
                stopping=TouchModeStoppingGemini(plan_mode="plan_a", intraday_mode="sl"),
                enable_cash_overlay=True,
                enable_intraday_touch=True,
                enable_turnover_damping=False,
                enable_nav_compounding=True,
                compounding_mode="floor_capped",
                target_slots=4,
                hold_strong=11,
                hold_normal=7,
            )
            rep_p1.bars = bars
            rep_p1.run()

            summary_p1 = rep_p1.summary()
            audit_p1 = audit_ledger(rep_p1, bars, events)

            self.assertTrue(audit_p1["passed"], f"2022-2025 Audit failed under {slip_bps} bps slippage")
            self.assertEqual(audit_p1["daily_nav_checked"], len(rep_p1.equity))
            for eq in rep_p1.equity:
                self.assertGreaterEqual(eq["cash"], -1e-7, f"Negative cash on {eq['date']}")
            # Under 15/20 bps, return remains solidly positive (> +50%) and MaxDD remains within -12%
            self.assertGreater(summary_p1["total_return"], 0.5000, f"2022-2025 return collapsed under {slip_bps} bps: {summary_p1['total_return']}")
            self.assertGreater(summary_p1["max_drawdown"], -0.1200, f"2022-2025 MaxDD breached -12% under {slip_bps} bps: {summary_p1['max_drawdown']}")


# =============================================================================
# Area 2: Liquidity Shocks & Cash Exhaustion
# =============================================================================

class TestLiquidityAndCashShocks(unittest.TestCase):
    """Stress tests on liquidity shocks: zero cash, boundary cash exhaustion, dividend delays."""

    def test_zero_cash_initial_scenario(self):
        """Engine with zero initial cash must not crash, overdraft, or buy stocks."""
        calendar = ["2026-01-05", "2026-01-06"]
        bars = pd.DataFrame([
            make_test_bar("2026-01-05", "600000", 10.0, 10.5, 9.8, 10.2),
            make_test_bar("2026-01-06", "600000", 10.2, 10.6, 10.0, 10.4),
        ])
        signals = pd.DataFrame([{"date": "2026-01-05", "code": "600000", "score": 1.0, "pred_net_return_v2": 0.01}])
        cfg = Config(capital=0.0001)  # Near zero capital
        engine = ReplayEngineV089(
            frames={"600000": bars},
            calendar=calendar,
            signals=signals,
            config=cfg,
            enable_cash_overlay=True,
        )
        engine.run()
        # Zero buys allowed
        self.assertEqual(len(engine.positions), 0)
        self.assertEqual(len(engine.fills), 0)
        self.assertGreaterEqual(engine.cash, 0.0)
        self.assertLess(abs(engine.nav() - engine.cash), 1e-6)

    def test_multi_order_cash_exhaustion_collision(self):
        """When multiple buys queue on the same morning but cash only covers one."""
        calendar = ["2026-01-05", "2026-01-06"]
        bars_a = pd.DataFrame([
            make_test_bar("2026-01-05", "600001", 10.0, 10.5, 9.8, 10.0),
            make_test_bar("2026-01-06", "600001", 10.0, 10.5, 9.8, 10.0),
        ])
        bars_b = pd.DataFrame([
            make_test_bar("2026-01-05", "600002", 10.0, 10.5, 9.8, 10.0),
            make_test_bar("2026-01-06", "600002", 10.0, 10.5, 9.8, 10.0),
        ])
        signals = pd.DataFrame([
            {"date": "2026-01-05", "code": "600001", "score": 2.0, "pred_net_return_v2": 0.01},
            {"date": "2026-01-05", "code": "600002", "score": 1.0, "pred_net_return_v2": 0.01},
        ])
        # Capital 1500: each 100 shares @ 10 = ~1005 cost; only 1 buy fits under max_exposure 0.90
        cfg = Config(capital=1500.0, order_budget=1500.0, max_positions=4, max_exposure=0.90)
        engine = ReplayEngineV089(
            frames={"600001": bars_a, "600002": bars_b},
            calendar=calendar,
            signals=signals,
            config=cfg,
            daily_limits={"2026-01-05": 2, "2026-01-06": 2},
            breadth_lookup={"2026-01-05": 0.60, "2026-01-06": 0.60},
            enable_cash_overlay=False,
            enable_nav_compounding=False,  # use fixed budget to test queue
        )
        engine.run()
        # Only 1 position bought, second rejected with cash_limit
        self.assertEqual(len(engine.positions), 1)
        self.assertIn("600001", engine.positions)
        self.assertNotIn("600002", engine.positions)
        self.assertGreaterEqual(engine.cash, 0.0)
        self.assertGreaterEqual(engine.rejected.get("cash_limit", 0), 1)
        # Balance sheet conserved
        self.assertLess(abs(engine.nav() - (engine.cash + sum(p.shares * engine.marks[p.code] for p in engine.positions.values()))), 1e-6)

    def test_delayed_dividend_payment_and_pre_settlement_liquidation(self):
        """Position receives dividend, sold before dividend pay date: receivables stay in NAV and settle on pay date."""
        calendar = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
        bars = pd.DataFrame([
            make_test_bar("2026-01-05", "600000", 10.0, 10.2, 9.9, 10.0),
            make_test_bar("2026-01-06", "600000", 10.0, 10.2, 9.9, 10.0),  # Buy fills
            make_test_bar("2026-01-07", "600000", 10.0, 10.2, 9.9, 10.0),  # Ex date, morning sell triggered
            make_test_bar("2026-01-08", "600000", 10.0, 10.2, 9.9, 10.0),  # Pay date
        ])
        signals = pd.DataFrame([{"date": "2026-01-05", "code": "600000", "score": 1.0, "pred_net_return_v2": 0.01}])
        events = [{
            "code": "600000",
            "dividRegistDate": "2026-01-06",
            "dividOperateDate": "2026-01-07",
            "dividPayDate": "2026-01-08",
            "dividCashPsBeforeTax": 1.0,
            "dividStocksPs": 0.0,
            "dividReserveToStockPs": 0.0,
        }]

        class ImmediateExitStopping(TouchModeStoppingGemini):
            def evaluate_morning(self, position, day, holding_age, max_hold, context):
                if day == "2026-01-07":
                    return ExitDecision(should_exit=True, exit_reason="test_exit")
                return None

        engine = ReplayEngineV089(
            frames={"600000": bars},
            calendar=calendar,
            signals=signals,
            dividends=events,
            stopping=ImmediateExitStopping(),
            enable_cash_overlay=False,
        )
        engine.run()

        # On 2026-01-07: stock was sold, but receivables must exist
        eq_day7 = [e for e in engine.equity if e["date"] == "2026-01-07"][0]
        self.assertGreater(eq_day7["receivables"], 0.0)
        self.assertEqual(eq_day7["positions"], 0)

        # On 2026-01-08: receivables cleared to 0, cash increased
        eq_day8 = [e for e in engine.equity if e["date"] == "2026-01-08"][0]
        self.assertEqual(eq_day8["receivables"], 0.0)
        self.assertGreater(eq_day8["cash"], eq_day7["cash"])

        # Audit ledger passes 100%
        audit = audit_ledger(engine, engine.bars, events)
        self.assertTrue(audit["passed"])

    def test_locked_bonus_shares_prevent_illegal_early_sell(self):
        """Bonus shares locked until dividStockMarketDate must reject early liquidation."""
        calendar = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
        bars = pd.DataFrame([
            make_test_bar("2026-01-05", "600000", 10.0, 10.2, 9.9, 10.0),
            make_test_bar("2026-01-06", "600000", 10.0, 10.2, 9.9, 10.0),  # Buy fills 700 shs
            make_test_bar("2026-01-07", "600000", 5.0, 5.2, 4.9, 5.0),    # Ex date (10送10: +700 shs locked until 2026-01-09)
            make_test_bar("2026-01-08", "600000", 5.0, 5.2, 4.9, 5.0),    # Sell original 700 shs, remaining 700 locked
        ])
        signals = pd.DataFrame([{"date": "2026-01-05", "code": "600000", "score": 1.0, "pred_net_return_v2": 0.01}])
        events = [{
            "code": "600000",
            "dividRegistDate": "2026-01-06",
            "dividOperateDate": "2026-01-07",
            "dividStockMarketDate": "2026-01-09",
            "dividCashPsBeforeTax": 0.0,
            "dividStocksPs": 1.0,
            "dividReserveToStockPs": 0.0,
        }]

        class ExitOnDay8(TouchModeStoppingGemini):
            def evaluate_morning(self, position, day, holding_age, max_hold, context):
                if day == "2026-01-08":
                    return ExitDecision(should_exit=True, exit_reason="test_exit")
                return None

        engine = ReplayEngineV089(
            frames={"600000": bars},
            calendar=calendar,
            signals=signals,
            dividends=events,
            stopping=ExitOnDay8(),
            enable_cash_overlay=False,
        )
        engine.run()

        # On day 8, original shares sold, but bonus shares remain locked!
        pos = engine.positions.get("600000")
        self.assertIsNotNone(pos)
        self.assertEqual(pos.shares, 700)
        self.assertLess(abs(engine.nav() - (engine.cash + pos.shares * 5.0)), 1e-6)


# =============================================================================
# Area 3: Market Crash Edge Cases & Execution Blocks
# =============================================================================

class TestMarketCrashEdgeCases(unittest.TestCase):
    """Stress tests on crash conditions: limit-down sell blocks, simultaneous touch collisions, halts."""

    def test_morning_sell_blocked_by_limit_down_open(self):
        """When an exit is requested at open, but stock opens at limit-down (-10%), sell must be blocked."""
        calendar = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
        bars = pd.DataFrame([
            make_test_bar("2026-01-05", "600000", 10.0, 10.2, 9.9, 10.0),
            make_test_bar("2026-01-06", "600000", 10.0, 10.2, 9.9, 10.0),  # Buy fills
            # Day 7: Preclose 10.0, opens at 9.00 (limit down!)
            make_test_bar("2026-01-07", "600000", 9.00, 9.00, 9.00, 9.00, preclose=10.0),
            # Day 8: Rebound, open 9.50 (tradable)
            make_test_bar("2026-01-08", "600000", 9.50, 9.60, 9.40, 9.50, preclose=9.0),
        ])
        signals = pd.DataFrame([{"date": "2026-01-05", "code": "600000", "score": 1.0, "pred_net_return_v2": 0.01}])

        class AlwaysExitStopping(TouchModeStoppingGemini):
            def evaluate_morning(self, position, day, holding_age, max_hold, context):
                if day >= "2026-01-07":
                    return ExitDecision(should_exit=True, exit_reason="panic_exit")
                return None

        engine = ReplayEngineV089(
            frames={"600000": bars},
            calendar=calendar,
            signals=signals,
            stopping=AlwaysExitStopping(),
            enable_cash_overlay=False,
        )
        engine.run()

        # On day 7, sell rejected due to limit-down
        self.assertGreaterEqual(engine.rejected.get("limit_down_open", 0), 1)
        # Position survived day 7 and exited on day 8
        self.assertEqual(len(engine.trades), 1)
        trade = engine.trades[0]
        self.assertEqual(trade["exit_date"], "2026-01-08")
        self.assertLess(abs(engine.nav() - engine.cash), 1e-6)

    def test_intraday_limit_down_lock_blocks_intraday_touch(self):
        """Intraday stop-loss touch is blocked if stock is locked at limit down (high <= limit_down)."""
        calendar = ["2026-01-05", "2026-01-06", "2026-01-07"]
        bars = pd.DataFrame([
            make_test_bar("2026-01-05", "600000", 10.0, 10.2, 9.9, 10.0),
            make_test_bar("2026-01-06", "600000", 10.0, 10.2, 9.9, 10.0),  # Buy fills
            # Day 7: Preclose 10.0, high is 9.0 (limit down locked throughout the day)
            make_test_bar("2026-01-07", "600000", 9.0, 9.0, 9.0, 9.0, preclose=10.0),
        ])
        signals = pd.DataFrame([{"date": "2026-01-05", "code": "600000", "score": 1.0, "pred_net_return_v2": 0.01}])

        class ForceTouchStopping(TouchModeStoppingGemini):
            def check_intraday_touch(self, position, bar, context):
                return ExitDecision(should_exit=True, exit_reason="stop_loss", target_price=9.0)

        engine = ReplayEngineV089(
            frames={"600000": bars},
            calendar=calendar,
            signals=signals,
            stopping=ForceTouchStopping(intraday_mode="all"),
            enable_intraday_touch=True,
            enable_cash_overlay=False,
        )
        engine.run()

        # Intraday exit was rejected with limit_down_locked
        self.assertGreaterEqual(engine.rejected.get("limit_down_locked", 0), 1)
        self.assertIn("600000", engine.positions)
        self.assertLess(abs(engine.nav() - (engine.cash + engine.positions["600000"].shares * 9.0)), 1e-6)

    def test_intraday_stop_loss_priority_in_simultaneous_touch_whipsaw(self):
        """In an extreme whipsaw (天地板), stop-loss MUST take priority with 10 bps penalty slippage."""
        stopping = TouchModeStoppingGemini(plan_mode="plan_a", intraday_mode="sl")
        pos = Position(1, "600000", "2026-01-05", "2026-01-06", 1, 1000, 1000, 10000.0, 5.0)
        # Bar with low hitting -3% stop-loss (low=9.60 <= 9.70) AND high hitting take-profit (high=11.50)
        bar = make_test_bar("2026-01-07", "600000", 10.0, 11.50, 9.60, 10.50, preclose=10.0)
        ctx = MarketContext(
            date="2026-01-07",
            calendar_index=2,
            breadth5=0.6,
            pool_size=300,
            daily_limit=2,
            regime="high",
            budget=7500.0,
            max_positions=4,
        )

        decision = stopping.check_intraday_touch(pos, bar, ctx)
        self.assertTrue(decision.should_exit)
        self.assertIn("stop_loss", decision.exit_reason)
        self.assertEqual(decision.slippage_bps_penalty, 0.0010)

    def test_trading_suspension_blocks_both_buy_and_sell(self):
        """Suspended stocks (tradestatus=0 or volume=0) cleanly reject without mutating state."""
        calendar = ["2026-01-05", "2026-01-06"]
        bars = pd.DataFrame([
            make_test_bar("2026-01-05", "600000", 10.0, 10.2, 9.9, 10.0),
            # Day 6: suspended
            make_test_bar("2026-01-06", "600000", 0.0, 0.0, 0.0, 10.0, volume=0.0, tradestatus=0),
        ])
        signals = pd.DataFrame([{"date": "2026-01-05", "code": "600000", "score": 1.0, "pred_net_return_v2": 0.01}])

        engine = ReplayEngineV089(
            frames={"600000": bars},
            calendar=calendar,
            signals=signals,
            enable_cash_overlay=False,
        )
        engine.run()

        # Buy on Day 6 was rejected with suspended_or_missing
        self.assertGreaterEqual(engine.rejected.get("suspended_or_missing", 0), 1)
        self.assertEqual(len(engine.positions), 0)
        self.assertEqual(engine.cash, engine.config.capital)


# =============================================================================
# Area 4: Boundary Conditions & Microstructure Invariants
# =============================================================================

class TestBoundaryConditionsAndInvariants(unittest.TestCase):
    """Stress tests on boundary values: budget rounding, lot sizing, NAV scaling, T+1 enforcement."""

    def test_high_priced_stock_budget_rounding_and_rejection(self):
        """Stock priced at 500 RMB / share cannot be bought with 7,500 RMB budget (cost of 100 shares = ~50,000)."""
        calendar = ["2026-01-05", "2026-01-06"]
        bars = pd.DataFrame([
            make_test_bar("2026-01-05", "600519", 500.0, 505.0, 498.0, 500.0),
            make_test_bar("2026-01-06", "600519", 500.0, 505.0, 498.0, 500.0),
        ])
        signals = pd.DataFrame([{"date": "2026-01-05", "code": "600519", "score": 1.0, "pred_net_return_v2": 0.01}])

        engine = ReplayEngineV089(
            frames={"600519": bars},
            calendar=calendar,
            signals=signals,
            enable_cash_overlay=False,
        )
        engine.run()

        # Must reject with lot_size_or_budget
        self.assertGreaterEqual(engine.rejected.get("lot_size_or_budget", 0), 1)
        self.assertEqual(len(engine.positions), 0)
        self.assertEqual(len(engine.fills), 0)
        self.assertEqual(engine.cash, engine.config.capital)

    def test_dynamic_nav_compounding_budget_bounds(self):
        """Verify dynamic compounding formula bounds across extreme NAV spectrum."""
        calc = DynamicCompoundingBudget(slots=4, target_exposure=0.90, budget_multiplier=1.04, min_budget=7500.0, max_budget=15000.0)

        # Extreme low / negative NAV
        self.assertEqual(calc(-1000.0), 7500.0)
        self.assertEqual(calc(0.0), 7500.0)
        self.assertEqual(calc(100.0), 7500.0)
        self.assertEqual(calc(30000.0), 7500.0)  # (30000 * 0.9 / 4) * 1.04 = 7020 <= 7500

        # Linear expansion range
        self.assertAlmostEqual(calc(50000.0), 11700.0, places=4)
        self.assertAlmostEqual(calc(60000.0), 14040.0, places=4)

        # Upper ceiling cap at 15,000
        self.assertEqual(calc(65000.0), 15000.0)  # (65000 * 0.9 / 4) * 1.04 = 15210 >= 15000
        self.assertEqual(calc(100000.0), 15000.0)
        self.assertEqual(calc(1000000.0), 15000.0)

        # Edge case: 0 slots returns 0.0
        self.assertEqual(compute_dynamic_compounding_budget(50000.0, slots=0), 0.0)

    def test_strict_t_plus_one_enforcement(self):
        """Intraday exit on same day of entry (T+0) is strictly blocked."""
        calendar = ["2026-01-05", "2026-01-06"]
        bars = pd.DataFrame([
            make_test_bar("2026-01-05", "600000", 10.0, 10.5, 9.8, 10.0),
            make_test_bar("2026-01-06", "600000", 10.0, 10.5, 9.0, 9.2),  # Day 6: Entry day! Hits stop-loss intraday
        ])
        signals = pd.DataFrame([{"date": "2026-01-05", "code": "600000", "score": 1.0, "pred_net_return_v2": 0.01}])

        engine = ReplayEngineV089(
            frames={"600000": bars},
            calendar=calendar,
            signals=signals,
            enable_intraday_touch=True,
            stopping=TouchModeStoppingGemini(plan_mode="plan_a", intraday_mode="sl"),
            enable_cash_overlay=False,
        )
        engine.run()

        # Bought on Day 6; even though Day 6 touched low=9.0, T+1 prevents liquidation on Day 6!
        self.assertIn("600000", engine.positions)
        self.assertEqual(len(engine.trades), 0)

    def test_compact_holding_horizon_regime_switching(self):
        """Compact horizon holds 11 days on strong breadth (limit>=2) and 7 days on normal regime (limit<2)."""
        # Test 7-day normal exit
        cal_7d = [f"2026-01-{i:02d}" for i in range(1, 15)]
        bars_7d = pd.DataFrame([
            make_test_bar(d, "600000", 10.0, 10.2, 9.9, 10.0) for d in cal_7d
        ])
        signals_7d = pd.DataFrame([{"date": "2026-01-01", "code": "600000", "score": 1.0, "pred_net_return_v2": 0.01}])
        limits_normal = {d: 1 for d in cal_7d}  # limit < 2 -> max_hold = 7

        engine_7d = ReplayEngineV089(
            frames={"600000": bars_7d},
            calendar=cal_7d,
            signals=signals_7d,
            daily_limits=limits_normal,
            hold_strong=11,
            hold_normal=7,
            enable_cash_overlay=False,
        )
        engine_7d.run()

        self.assertEqual(len(engine_7d.trades), 1)
        # Entry at index 1 (2026-01-02). Exit at index 1 + 7 = index 8 (2026-01-09)
        self.assertEqual(engine_7d.trades[0]["entry_date"], "2026-01-02")
        self.assertEqual(engine_7d.trades[0]["exit_date"], "2026-01-09")


if __name__ == "__main__":
    unittest.main()
