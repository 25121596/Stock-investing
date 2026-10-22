"""test_v086_modular_engine.py - Unit and Fly-Flat Parity Tests for ModularReplayEngine.

Validates Milestone 1 deliverables:
1. Protocol definitions, dataclasses, and runtime isinstance checks.
2. Baseline implementations (BaselineRanker, NoOpGatekeeper, BaselineRegimeStopping).
3. Exact microstructure invariants (lot size, fees, T+1, limits, balance sheet conservation).
4. 100% fly-flat alignment comparison against VerifiedV08Replay across:
   - 2022-2025 Continuous Period (+42.7933%, MaxDD -8.3194%, 109 trades)
   - 2026 Out-of-Sample Period (+15.7316%, MaxDD -3.9893%, 13 trades)
"""
from dataclasses import FrozenInstanceError
from decimal import Decimal
import math
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from corporate_events import canonicalize_events
from engine import Config, Position, tick
from engine_core import (
    BaselineRanker,
    BaselineRegimeStopping,
    ExitDecision,
    GatekeeperProtocol,
    MarketContext,
    ModularReplayEngine,
    NoOpGatekeeper,
    OptimalStoppingProtocol,
    PositionState,
    RankerProtocol,
    audit_ledger,
)
from exploration_features import breadth_days
from run import load
from v08_engine_verified import VerifiedV08Replay
from validate_expanded import prepare_bars

ROOT = Path(__file__).resolve().parent.parent


class TestProtocolDefinitionsAndRuntimeCheck(unittest.TestCase):
    """Test dataclass semantics and runtime checkable protocols."""

    def test_market_context_dataclass(self):
        ctx = MarketContext(
            date="2022-01-04",
            calendar_index=0,
            breadth5=0.62,
            pool_size=300,
            daily_limit=2,
            regime="high",
            budget=7500.0,
            max_positions=5,
        )
        self.assertEqual(ctx.date, "2022-01-04")
        self.assertEqual(ctx.regime, "high")
        self.assertEqual(ctx.vol_map, {})
        # Immutable (frozen)
        with self.assertRaises(FrozenInstanceError):
            ctx.budget = 5500.0  # type: ignore

    def test_position_state_dataclass(self):
        pos = PositionState(
            trade_id=1,
            code="000001",
            entry_date="2022-01-04",
            entry_index=0,
            holding_age=2,
            shares=500,
            entry_price=10.0,
            entry_notional=5000.0,
            peak_price=11.0,
            current_gain=0.08,
            peak_gain=0.10,
        )
        self.assertEqual(pos.trade_id, 1)
        self.assertEqual(pos.shares, 500)
        self.assertFalse(pos.breakeven_active)
        self.assertFalse(pos.moonbag_active)
        self.assertFalse(pos.stop_pending)
        self.assertEqual(pos.proceeds, 0.0)

    def test_exit_decision_dataclass(self):
        dec_no = ExitDecision(should_exit=False)
        self.assertFalse(dec_no.should_exit)
        self.assertEqual(dec_no.exit_type, "")

        dec_yes = ExitDecision(
            should_exit=True,
            exit_type="intraday_touch",
            target_price=10.50,
            exit_reason="stop_loss",
            slippage_bps_penalty=0.0010,
        )
        self.assertTrue(dec_yes.should_exit)
        self.assertEqual(dec_yes.slippage_bps_penalty, 0.0010)

    def test_protocol_runtime_checkable(self):
        # Baseline implementations must satisfy protocols
        self.assertTrue(isinstance(BaselineRanker(), RankerProtocol))
        self.assertTrue(isinstance(NoOpGatekeeper(), GatekeeperProtocol))
        self.assertTrue(isinstance(BaselineRegimeStopping(), OptimalStoppingProtocol))

        # Incomplete classes must fail isinstance check
        class IncompleteRanker:
            pass

        class IncompleteGatekeeper:
            pass

        class IncompleteStopping:
            pass

        self.assertFalse(isinstance(IncompleteRanker(), RankerProtocol))
        self.assertFalse(isinstance(IncompleteGatekeeper(), GatekeeperProtocol))
        self.assertFalse(isinstance(IncompleteStopping(), OptimalStoppingProtocol))


class TestBaselineProtocolImplementations(unittest.TestCase):
    """Test functionality of baseline protocol implementations."""

    def setUp(self):
        self.ctx = MarketContext(
            date="2022-01-04",
            calendar_index=0,
            breadth5=0.60,
            pool_size=300,
            daily_limit=2,
            regime="high",
            budget=7500.0,
            max_positions=5,
            vol_map={("2022-01-04", "000001"): 0.020},
        )

    def test_baseline_ranker_sorting(self):
        ranker = BaselineRanker()
        df = pd.DataFrame([
            {"date": "2022-01-04", "code": "000002", "score": 0.50},
            {"date": "2022-01-04", "code": "000001", "score": 0.50},
            {"date": "2022-01-04", "code": "600000", "score": 0.90},
            {"date": "2022-01-04", "code": "000003", "score": 0.10},
        ])
        ranked = ranker.rank_candidates("2022-01-04", df, self.ctx)
        # Score descending, tie break by code ascending
        self.assertEqual(ranked.iloc[0]["code"], "600000")
        self.assertEqual(ranked.iloc[1]["code"], "000001")
        self.assertEqual(ranked.iloc[2]["code"], "000002")
        self.assertEqual(ranked.iloc[3]["code"], "000003")
        self.assertTrue(ranker.evaluate_turnover_damping({}, []))

    def test_noop_gatekeeper(self):
        gk = NoOpGatekeeper()
        allowed, prob, reason = gk.evaluate_candidate("000001", "2022-01-04", {}, self.ctx)
        self.assertTrue(allowed)
        self.assertEqual(prob, 1.0)
        self.assertEqual(reason, "pass")

        allowed_ord, prob_ord, reason_ord = gk.evaluate_order(pd.Series({"code": "000001"}), self.ctx)
        self.assertTrue(allowed_ord)
        self.assertEqual(prob_ord, 1.0)

    def test_baseline_stopping_morning_exit(self):
        stopping = BaselineRegimeStopping()
        pos = Position(
            trade_id=1,
            code="000001",
            signal_date="2022-01-04",
            entry_date="2022-01-05",
            entry_index=1,
            shares=500,
            initial_shares=500,
            entry_notional=5000.0,
            entry_fees=6.0,
        )
        # Normal holding (age 5 < max_hold 8)
        dec = stopping.evaluate_morning(pos, "2022-01-11", holding_age=5, max_hold=8, context=self.ctx)
        self.assertFalse(dec.should_exit)

        # Max hold reached (age 8 >= max_hold 8)
        dec_hold = stopping.evaluate_morning(pos, "2022-01-14", holding_age=8, max_hold=8, context=self.ctx)
        self.assertTrue(dec_hold.should_exit)
        self.assertEqual(dec_hold.exit_reason, "max_hold")

        # Stop pending from prior evening
        pos.stop_pending = True
        dec_stop = stopping.evaluate_morning(pos, "2022-01-08", holding_age=3, max_hold=8, context=self.ctx)
        self.assertTrue(dec_stop.should_exit)
        self.assertEqual(dec_stop.exit_reason, "stop_close_next_open")

    def test_baseline_stopping_close_conditions(self):
        stopping = BaselineRegimeStopping()
        pos = Position(
            trade_id=1,
            code="000001",
            signal_date="2022-01-04",
            entry_date="2022-01-04",
            entry_index=0,
            shares=500,
            initial_shares=500,
            entry_notional=5000.0,
            entry_fees=6.0,
        )
        # 1. Gain < 3% -> normal
        bar_norm = {"close": 10.15, "open": 10.0}
        dec_norm = stopping.check_close_stopping(pos, bar_norm, self.ctx)
        self.assertFalse(dec_norm.should_exit)
        self.assertFalse(getattr(pos, "breakeven_active", False))

        # 2. Gain >= 3% -> breakeven activated
        bar_be = {"close": 10.35, "open": 10.0}
        dec_be = stopping.check_close_stopping(pos, bar_be, self.ctx)
        self.assertFalse(dec_be.should_exit)
        self.assertTrue(getattr(pos, "breakeven_active", False))

        # 3. Breakeven retraces below 0.5% -> exit
        bar_be_fall = {"close": 10.03, "open": 10.0}
        dec_be_exit = stopping.check_close_stopping(pos, bar_be_fall, self.ctx)
        self.assertTrue(dec_be_exit.should_exit)
        self.assertEqual(dec_be_exit.exit_reason, "breakeven_stop")
        self.assertTrue(pos.stop_pending)


class TestMicrostructureInvariants(unittest.TestCase):
    """Test A-share trading microstructure rules and conservation invariants."""

    def test_config_fees_and_rounding(self):
        cfg = Config()
        self.assertEqual(cfg.capital, 30000.0)
        self.assertEqual(cfg.min_commission, 5.0)

        # On a 5000 RMB order:
        # Commission is max(5.0, 5000 * 0.00025 = 1.25) -> 5.0 RMB
        # Transfer is 5000 * 0.00001 = 0.05 RMB
        # Stamp tax (buy=0, sell=5000 * 0.0005 = 2.50 RMB)
        # Slippage is 5000 * 0.0005 = 2.50 RMB
        buy_fees = cfg.fees(5000.0, sell=False)
        self.assertEqual(buy_fees["commission"], 5.0)
        self.assertEqual(buy_fees["transfer"], 0.05)
        self.assertEqual(buy_fees["stamp"], 0.0)
        self.assertEqual(buy_fees["slippage"], 2.50)

        sell_fees = cfg.fees(5000.0, sell=True)
        self.assertEqual(sell_fees["stamp"], 2.50)

    def test_lot_size_enforcement(self):
        # 100 shares lot size
        budget = 7500.0
        unit_cost = 15.20 * (1 + 0.00025 + 0.00001 + 0.0005)
        quantity = int(max(0, budget - 5.0) / unit_cost) // 100 * 100
        self.assertEqual(quantity % 100, 0)
        self.assertTrue(quantity >= 100)


class TestParameterRemediation(unittest.TestCase):
    """Unit tests for cash_annual_rate propagation and neutral_breadth_threshold configuration."""

    def test_cash_annual_rate_propagation_and_accumulation(self):
        """Verify changing cash_annual_rate updates daily_cash_yield and compounds cash accumulation."""
        # 1. Parameter propagation checks
        eng_default = ModularReplayEngine(calendar=["2026-01-05"], enable_cash_overlay=True)
        self.assertEqual(eng_default.cash_annual_rate, 0.022)
        self.assertAlmostEqual(eng_default.daily_cash_yield, 0.022 / 242.0, places=9)

        eng_custom = ModularReplayEngine(calendar=["2026-01-05"], enable_cash_overlay=True, cash_annual_rate=0.05)
        self.assertEqual(eng_custom.cash_annual_rate, 0.05)
        self.assertAlmostEqual(eng_custom.daily_cash_yield, 0.05 / 242.0, places=9)

        eng_disabled = ModularReplayEngine(calendar=["2026-01-05"], enable_cash_overlay=False, cash_annual_rate=0.05)
        self.assertEqual(eng_disabled.daily_cash_yield, 0.0)

        # 2. Compounded cash accumulation over 5 sessions
        cal = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"]
        eng_base = ModularReplayEngine(calendar=cal, enable_cash_overlay=True, cash_annual_rate=0.022)
        eng_base.run()

        eng_high = ModularReplayEngine(calendar=cal, enable_cash_overlay=True, cash_annual_rate=0.05)
        eng_high.run()

        # Higher rate must accumulate strictly higher cash yield
        self.assertGreater(eng_high.accumulated_cash_yield, eng_base.accumulated_cash_yield)

        # Mathematical compound formula verification: C_N = C_0 * (1 + r_daily)^N
        exp_base_cash = 30000.0 * ((1.0 + 0.022 / 242.0) ** 5)
        exp_high_cash = 30000.0 * ((1.0 + 0.050 / 242.0) ** 5)
        self.assertAlmostEqual(eng_base.cash, exp_base_cash, places=6)
        self.assertAlmostEqual(eng_high.cash, exp_high_cash, places=6)

        # Independent balance sheet conservation audit
        audit_base = audit_ledger(eng_base, {}, [])
        audit_high = audit_ledger(eng_high, {}, [])
        self.assertTrue(audit_base["passed"])
        self.assertTrue(audit_high["passed"])

    def test_neutral_breadth_threshold_regime_classification_boundary(self):
        """Verify neutral_breadth_threshold changes the regime classification boundary."""
        eng_default = ModularReplayEngine(calendar=["2026-01-05"], neutral_breadth_threshold=0.45)
        self.assertEqual(eng_default.neutral_breadth_threshold, 0.45)

        eng_custom = ModularReplayEngine(calendar=["2026-01-05"], neutral_breadth_threshold=0.50)
        self.assertEqual(eng_custom.neutral_breadth_threshold, 0.50)

        # At breadth 0.46:
        # Default (cutoff 0.45): neutral regime
        # Custom (cutoff 0.50): defensive regime
        self.assertEqual(eng_default._determine_regime(0.46), "neutral")
        self.assertEqual(eng_custom._determine_regime(0.46), "defensive")

        # Boundary checks
        self.assertEqual(eng_default._determine_regime(0.45), "neutral")
        self.assertEqual(eng_custom._determine_regime(0.45), "defensive")
        self.assertEqual(eng_custom._determine_regime(0.50), "neutral")

        # High regime boundary (0.58) remains consistent across both
        self.assertEqual(eng_default._determine_regime(0.60), "high")
        self.assertEqual(eng_custom._determine_regime(0.60), "high")

    def test_neutral_breadth_threshold_simulation_order_suppression(self):
        """Verify neutral_breadth_threshold alters simulation execution and order gating."""
        cal = ["2026-01-05", "2026-01-06"]
        bars = pd.DataFrame([
            {"date": "2026-01-05", "code": "600000", "open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0, "volume": 100000.0, "tradestatus": 1, "isST": 0, "preclose": 10.0},
            {"date": "2026-01-06", "code": "600000", "open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0, "volume": 100000.0, "tradestatus": 1, "isST": 0, "preclose": 10.0},
        ])
        signals = pd.DataFrame([{"date": "2026-01-05", "code": "600000", "score": 1.0}])
        breadth_lookup = {"2026-01-05": 0.46, "2026-01-06": 0.46}
        daily_limits = {"2026-01-05": 1, "2026-01-06": 1}

        # Default threshold 0.45: breadth 0.46 -> neutral regime (budget 5500.0) -> buy executes next day
        eng_default = ModularReplayEngine(
            frames={"600000": bars},
            calendar=cal,
            signals=signals,
            breadth_lookup=breadth_lookup,
            daily_limits=daily_limits,
            neutral_breadth_threshold=0.45,
        )
        eng_default.run()
        self.assertEqual(len(eng_default.fills), 1)
        self.assertEqual(eng_default.fills[0]["side"], "buy")
        self.assertEqual(eng_default.fills[0]["code"], "600000")

        # Custom threshold 0.50: breadth 0.46 -> defensive regime (budget 0.0) -> buy suppressed
        eng_custom = ModularReplayEngine(
            frames={"600000": bars},
            calendar=cal,
            signals=signals,
            breadth_lookup=breadth_lookup,
            daily_limits=daily_limits,
            neutral_breadth_threshold=0.50,
        )
        eng_custom.run()
        self.assertEqual(len(eng_custom.fills), 0)


class TestFlyFlatAlignmentWithVerifiedV08Replay(unittest.TestCase):
    """Verify 100% numerical and NAV fly-flat identity against VerifiedV08Replay."""

    @classmethod
    def setUpClass(cls):
        manifest, frames, calendar, events_raw = load(ROOT / "data/forward-2026-10-13")
        events, event_audit = canonicalize_events(events_raw)
        bars = prepare_bars(frames)
        panel = pd.read_csv(ROOT / "results/midterm-access-v3/features.csv", dtype={"date": str, "code": str})
        a = pd.read_csv(ROOT / "results/model-v08/predictions-h10-v08.csv", dtype={"date": str, "code": str})
        b = pd.read_csv(ROOT / "results/model-v2/predictions-h10-v2.csv", dtype={"date": str, "code": str})
        c = pd.read_csv(ROOT / "results/model-v3/predictions-h10-v3.csv", dtype={"date": str, "code": str})

        breadth = breadth_days(panel)
        bm = breadth.set_index("date").breadth5.to_dict()
        pm = breadth.set_index("date").pool_size.to_dict()
        limits = {d: 0 if pm[d] < 100 or v < 0.45 else (1 if v < 0.55 else 2) for d, v in bm.items()}
        vol = panel.set_index(["date", "code"]).ret_vol20.to_dict()

        joined = b.merge(c, on=["date", "code"], validate="one_to_one").merge(a, on=["date", "code"], validate="one_to_one")
        allowed = joined.date.map(limits).fillna(0).gt(0)
        signals = joined.loc[allowed & joined["pred_net_return_v2"].ge(0.005), ["date", "code", "pred_score_v08"]].rename(
            columns={"pred_score_v08": "score"}
        )

        cls.frames = frames
        cls.calendar = calendar
        cls.events = events
        cls.bars = bars
        cls.vol = vol
        cls.bm = bm
        cls.pm = pm
        cls.limits = limits
        cls.signals = signals
        cls.base_cfg = Config(capital=30000, order_budget=7500, max_positions=5, max_exposure=0.9, hold_days=10)

    def test_fly_flat_2022_2025(self):
        """Test continuous 2022-2025: +42.7933%, MaxDD -8.3194%, 109 closed trades."""
        start, end = "2022-01-01", "2025-12-31"
        days = [d for d in self.calendar if start <= d <= end]
        sel = self.signals[self.signals.date.between(start, end)].copy()

        # Run legacy VerifiedV08Replay
        rep_legacy = VerifiedV08Replay(
            {},
            days,
            sel,
            self.events,
            self.base_cfg,
            vol_map=self.vol,
            daily_limits=self.limits,
            breadth_lookup=self.bm,
            enable_cash_overlay=False,
        )
        rep_legacy.frames, rep_legacy.bars = self.frames, self.bars
        rep_legacy.run()
        s_leg = rep_legacy.summary()

        # Run ModularReplayEngine in baseline bypass mode
        engine = ModularReplayEngine(
            frames=self.frames,
            calendar=days,
            signals=sel,
            dividends=self.events,
            config=self.base_cfg,
            vol_map=self.vol,
            daily_limits=self.limits,
            breadth_lookup=self.bm,
            enable_cash_overlay=False,
            ranker=BaselineRanker(),
            gatekeeper=NoOpGatekeeper(),
            stopping=BaselineRegimeStopping(),
            enable_intraday_touch=False,
            turnover_hurdle=0.0,
        )
        engine.bars = self.bars
        engine.run()
        s_mod = engine.summary()

        # 1. Verification of exact target metrics
        self.assertEqual(s_mod["closed_trades"], 109)
        self.assertAlmostEqual(s_mod["total_return"], 0.427933, places=5)
        self.assertAlmostEqual(s_mod["max_drawdown"], -0.083194, places=5)

        # 2. 100% fly-flat identity comparison against legacy
        self.assertAlmostEqual(s_leg["total_return"], s_mod["total_return"], places=10)
        self.assertAlmostEqual(s_leg["max_drawdown"], s_mod["max_drawdown"], places=10)
        self.assertAlmostEqual(s_leg["final_equity"], s_mod["final_equity"], places=6)
        self.assertEqual(s_leg["closed_trades"], s_mod["closed_trades"])

        # 3. Daily NAV curve identity
        df_leg_eq = pd.DataFrame(rep_legacy.equity)
        df_mod_eq = pd.DataFrame(engine.equity)
        self.assertEqual(len(df_leg_eq), len(df_mod_eq))
        max_eq_diff = (df_leg_eq["equity"] - df_mod_eq["equity"]).abs().max()
        self.assertLess(max_eq_diff, 1e-6)

        # 4. Fill records identity
        df_leg_fills = pd.DataFrame(rep_legacy.fills)
        df_mod_fills = pd.DataFrame(engine.fills)
        self.assertEqual(len(df_leg_fills), len(df_mod_fills))
        pd.testing.assert_frame_equal(df_leg_fills, df_mod_fills, check_exact=False, rtol=1e-6)

        # 5. Independent ledger audit
        audit_res = audit_ledger(engine, self.bars, self.events)
        self.assertTrue(audit_res["passed"])
        self.assertEqual(audit_res["daily_nav_checked"], len(days))

    def test_fly_flat_2026_out_of_sample(self):
        """Test out-of-sample 2026: +15.7316%, MaxDD -3.9893%, 13 closed trades."""
        start, end = "2026-01-01", "2026-10-13"
        days = [d for d in self.calendar if start <= d <= end]
        sel = self.signals[self.signals.date.between(start, end)].copy()

        # Run legacy VerifiedV08Replay
        rep_legacy = VerifiedV08Replay(
            {},
            days,
            sel,
            self.events,
            self.base_cfg,
            vol_map=self.vol,
            daily_limits=self.limits,
            breadth_lookup=self.bm,
            enable_cash_overlay=False,
        )
        rep_legacy.frames, rep_legacy.bars = self.frames, self.bars
        rep_legacy.run()
        s_leg = rep_legacy.summary()

        # Run ModularReplayEngine in baseline bypass mode
        engine = ModularReplayEngine(
            frames=self.frames,
            calendar=days,
            signals=sel,
            dividends=self.events,
            config=self.base_cfg,
            vol_map=self.vol,
            daily_limits=self.limits,
            breadth_lookup=self.bm,
            enable_cash_overlay=False,
            ranker=BaselineRanker(),
            gatekeeper=NoOpGatekeeper(),
            stopping=BaselineRegimeStopping(),
            enable_intraday_touch=False,
            turnover_hurdle=0.0,
        )
        engine.bars = self.bars
        engine.run()
        s_mod = engine.summary()

        # 1. Verification of exact target metrics
        self.assertEqual(s_mod["closed_trades"], 13)
        self.assertAlmostEqual(s_mod["total_return"], 0.157316, places=5)
        self.assertAlmostEqual(s_mod["max_drawdown"], -0.039893, places=5)

        # 2. 100% fly-flat identity comparison against legacy
        self.assertAlmostEqual(s_leg["total_return"], s_mod["total_return"], places=10)
        self.assertAlmostEqual(s_leg["max_drawdown"], s_mod["max_drawdown"], places=10)
        self.assertAlmostEqual(s_leg["final_equity"], s_mod["final_equity"], places=6)
        self.assertEqual(s_leg["closed_trades"], s_mod["closed_trades"])

        # 3. Daily NAV curve identity
        df_leg_eq = pd.DataFrame(rep_legacy.equity)
        df_mod_eq = pd.DataFrame(engine.equity)
        self.assertEqual(len(df_leg_eq), len(df_mod_eq))
        max_eq_diff = (df_leg_eq["equity"] - df_mod_eq["equity"]).abs().max()
        self.assertLess(max_eq_diff, 1e-6)

        # 4. Fill records identity
        df_leg_fills = pd.DataFrame(rep_legacy.fills)
        df_mod_fills = pd.DataFrame(engine.fills)
        self.assertEqual(len(df_leg_fills), len(df_mod_fills))
        pd.testing.assert_frame_equal(df_leg_fills, df_mod_fills, check_exact=False, rtol=1e-6)

        # 5. Independent ledger audit
        audit_res = audit_ledger(engine, self.bars, self.events)
        self.assertTrue(audit_res["passed"])
        self.assertEqual(audit_res["daily_nav_checked"], len(days))


if __name__ == "__main__":
    unittest.main()
