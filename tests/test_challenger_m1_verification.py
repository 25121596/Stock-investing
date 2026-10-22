"""test_challenger_m1_verification.py - Empirical Challenger Stress Suite for Milestone 1.

Adversarially stress-tests:
1. Protocol boundary conditions, direct instantiation, and runtime isinstance checks.
2. Malformed/incomplete mock implementations to verify TypeErrors/AttributeErrors.
3. Balance sheet conservation under floating dividends, fractional tax withholding,
   and odd lot cash yields.
4. Independent audit_ledger reconstruction under floating corporate actions.
5. Uniform propagation of config options (commission, min_commission, stamp_tax, slippage,
   dividend_tax, capital, cash_annual_rate).
"""
from collections import defaultdict
from decimal import Decimal
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import unittest

import numpy as np
import pandas as pd

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

ROOT = Path(__file__).resolve().parent.parent


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
) -> Dict[str, Any]:
    return {
        "date": date,
        "code": code,
        "open": float(open_),
        "high": float(high),
        "low": float(low),
        "close": float(close),
        "volume": float(volume),
        "tradestatus": 1,
        "isST": is_st,
        "preclose": float(preclose if preclose is not None else open_),
    }


class TestProtocolBoundaryAndTypeChecking(unittest.TestCase):
    """Area 1: Verify protocol instantiation, isinstance checking, and malformed mock errors."""

    def test_direct_protocol_instantiation_raises_type_error(self):
        """Protocols cannot be instantiated directly without implementation."""
        with self.assertRaises(TypeError):
            RankerProtocol()
        with self.assertRaises(TypeError):
            GatekeeperProtocol()
        with self.assertRaises(TypeError):
            OptimalStoppingProtocol()

    def test_ranker_protocol_isinstance_conformance(self):
        """RankerProtocol requires score_and_rank, rank_candidates, evaluate_turnover_damping."""
        class CompleteRanker:
            def score_and_rank(self, candidates, date, breadth, context, turnover_damping=False):
                return candidates

            def rank_candidates(self, day, candidates, context, positions=None, turnover_damping=False):
                return candidates

            def evaluate_turnover_damping(self, new_candidate, held_positions, hurdle=0.0120):
                return True

        self.assertTrue(isinstance(CompleteRanker(), RankerProtocol))

        # Missing score_and_rank fails isinstance check
        class MissingScore:
            def rank_candidates(self, day, candidates, context, positions=None, turnover_damping=False):
                return candidates

            def evaluate_turnover_damping(self, new_candidate, held_positions, hurdle=0.0120):
                return True

        self.assertFalse(isinstance(MissingScore(), RankerProtocol))

        # Missing rank_candidates fails isinstance check
        class MissingRank:
            def score_and_rank(self, candidates, date, breadth, context, turnover_damping=False):
                return candidates

            def evaluate_turnover_damping(self, new_candidate, held_positions, hurdle=0.0120):
                return True

        self.assertFalse(isinstance(MissingRank(), RankerProtocol))

        # Missing evaluate_turnover_damping fails isinstance check
        class MissingDamping:
            def score_and_rank(self, candidates, date, breadth, context, turnover_damping=False):
                return candidates

            def rank_candidates(self, day, candidates, context, positions=None, turnover_damping=False):
                return candidates

        self.assertFalse(isinstance(MissingDamping(), RankerProtocol))

    def test_non_callable_attribute_boundary_raises_type_error(self):
        """Boundary condition: Python's @runtime_checkable inspects hasattr() which passes,
        but ModularReplayEngine execution immediately raises TypeError on call.
        """
        class NonCallableAttrRanker:
            score_and_rank = 123
            rank_candidates = "not_callable"
            evaluate_turnover_damping = True

        # In standard typing.runtime_checkable, hasattr check passes
        self.assertTrue(isinstance(NonCallableAttrRanker(), RankerProtocol))

        # But executing run() with it raises TypeError
        calendar = ["2026-01-05", "2026-01-06"]
        bars = pd.DataFrame([
            make_test_bar("2026-01-05", "600000", 10.0, 10.5, 9.8, 10.2),
            make_test_bar("2026-01-06", "600000", 10.2, 10.6, 10.0, 10.4),
        ])
        signals = pd.DataFrame([{"date": "2026-01-05", "code": "600000", "score": 1.0}])

        engine = ModularReplayEngine(
            frames={"600000": bars},
            calendar=calendar,
            signals=signals,
            ranker=NonCallableAttrRanker(),  # type: ignore
        )
        with self.assertRaises(TypeError):
            engine.run()

    def test_gatekeeper_protocol_isinstance_conformance(self):
        """GatekeeperProtocol requires evaluate_candidate, evaluate_order."""
        class CompleteGatekeeper:
            def evaluate_candidate(self, code, date, features, context):
                return (True, 1.0, "pass")

            def evaluate_order(self, candidate_row, context):
                return (True, 1.0, "pass")

        self.assertTrue(isinstance(CompleteGatekeeper(), GatekeeperProtocol))

        class MissingOrder:
            def evaluate_candidate(self, code, date, features, context):
                return (True, 1.0, "pass")

        self.assertFalse(isinstance(MissingOrder(), GatekeeperProtocol))

        class MissingCandidate:
            def evaluate_order(self, candidate_row, context):
                return (True, 1.0, "pass")

        self.assertFalse(isinstance(MissingCandidate(), GatekeeperProtocol))

    def test_optimal_stopping_protocol_isinstance_conformance(self):
        """OptimalStoppingProtocol requires 4 specific methods."""
        class CompleteStopping:
            def evaluate_morning(self, position, day, holding_age, max_hold, context):
                return None

            def evaluate_intraday(self, position, bar, day, holding_age, context):
                return None

            def check_intraday_touch(self, position, bar, context):
                return ExitDecision(should_exit=False)

            def check_close_stopping(self, position, bar, context):
                return ExitDecision(should_exit=False)

        self.assertTrue(isinstance(CompleteStopping(), OptimalStoppingProtocol))

        class MissingTouch:
            def evaluate_morning(self, position, day, holding_age, max_hold, context):
                return None

            def evaluate_intraday(self, position, bar, day, holding_age, context):
                return None

            def check_close_stopping(self, position, bar, context):
                return ExitDecision(should_exit=False)

        self.assertFalse(isinstance(MissingTouch(), OptimalStoppingProtocol))

    def test_engine_runtime_raises_on_incomplete_ranker(self):
        """ModularReplayEngine raises AttributeError during run() when ranker lacks rank_candidates."""
        class BrokenRanker:
            pass

        calendar = ["2026-01-05", "2026-01-06"]
        bars = pd.DataFrame([
            make_test_bar("2026-01-05", "600000", 10.0, 10.5, 9.8, 10.2),
            make_test_bar("2026-01-06", "600000", 10.2, 10.6, 10.0, 10.4),
        ])
        signals = pd.DataFrame([{"date": "2026-01-05", "code": "600000", "score": 1.0}])

        engine = ModularReplayEngine(
            frames={"600000": bars},
            calendar=calendar,
            signals=signals,
            ranker=BrokenRanker(),  # type: ignore
        )
        with self.assertRaises(AttributeError):
            engine.run()

    def test_engine_runtime_raises_on_malformed_ranker_output(self):
        """ModularReplayEngine raises AttributeError when ranker output lacks .to_dict()."""
        class BadOutputRanker:
            def rank_candidates(self, day, candidates, context, positions=None, turnover_damping=False):
                return "not_a_dataframe"

        calendar = ["2026-01-05", "2026-01-06"]
        bars = pd.DataFrame([
            make_test_bar("2026-01-05", "600000", 10.0, 10.5, 9.8, 10.2),
            make_test_bar("2026-01-06", "600000", 10.2, 10.6, 10.0, 10.4),
        ])
        signals = pd.DataFrame([{"date": "2026-01-05", "code": "600000", "score": 1.0}])

        engine = ModularReplayEngine(
            frames={"600000": bars},
            calendar=calendar,
            signals=signals,
            ranker=BadOutputRanker(),  # type: ignore
        )
        with self.assertRaises(AttributeError):
            engine.run()

    def test_engine_runtime_raises_on_incomplete_gatekeeper(self):
        """ModularReplayEngine raises AttributeError when gatekeeper lacks evaluate_candidate."""
        class BrokenGatekeeper:
            pass

        calendar = ["2026-01-05", "2026-01-06"]
        bars = pd.DataFrame([
            make_test_bar("2026-01-05", "600000", 10.0, 10.5, 9.8, 10.2),
            make_test_bar("2026-01-06", "600000", 10.2, 10.6, 10.0, 10.4),
        ])
        signals = pd.DataFrame([{"date": "2026-01-05", "code": "600000", "score": 1.0}])

        engine = ModularReplayEngine(
            frames={"600000": bars},
            calendar=calendar,
            signals=signals,
            gatekeeper=BrokenGatekeeper(),  # type: ignore
        )
        with self.assertRaises(AttributeError):
            engine.run()

    def test_engine_runtime_raises_on_incomplete_stopping(self):
        """ModularReplayEngine raises AttributeError when stopping lacks evaluate_morning."""
        class BrokenStopping:
            pass

        calendar = ["2026-01-05", "2026-01-06", "2026-01-07"]
        bars = pd.DataFrame([
            make_test_bar("2026-01-05", "600000", 10.0, 10.5, 9.8, 10.2),
            make_test_bar("2026-01-06", "600000", 10.2, 10.6, 10.0, 10.4),
            make_test_bar("2026-01-07", "600000", 10.4, 10.8, 10.2, 10.5),
        ])
        signals = pd.DataFrame([{"date": "2026-01-05", "code": "600000", "score": 1.0}])

        engine = ModularReplayEngine(
            frames={"600000": bars},
            calendar=calendar,
            signals=signals,
            stopping=BrokenStopping(),  # type: ignore
        )
        with self.assertRaises(AttributeError):
            engine.run()


class TestBalanceSheetConservationUnderFloatingDividends(unittest.TestCase):
    """Area 2: Stress balance sheet conservation with floating dividends, tax withholding, odd lot cash yields."""

    def test_floating_dividend_and_fractional_withholding_conservation(self):
        """Validate exact conservation when dividend per share has 5 decimal places and 20% tax."""
        calendar = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"]
        code = "600000"

        # 5 days: buy on day 1 (fill day 2 open), record on day 2, ex-date on day 3, pay date on day 5
        bars = pd.DataFrame([
            make_test_bar("2026-01-05", code, 10.00, 10.20, 9.90, 10.00),
            make_test_bar("2026-01-06", code, 10.00, 10.30, 9.95, 10.10),
            # Day 3 is ex-date. Preclose adjusted by dividend before tax
            make_test_bar("2026-01-07", code, 9.86, 10.00, 9.80, 9.90, preclose=10.10 - 0.13789),
            make_test_bar("2026-01-08", code, 9.90, 10.10, 9.85, 10.00),
            make_test_bar("2026-01-09", code, 10.00, 10.20, 9.90, 10.10),
        ])

        signals = pd.DataFrame([{"date": "2026-01-05", "code": code, "score": 1.0}])

        # Floating dividend: 0.13789 RMB / share.
        event = {
            "code": code,
            "dividRegistDate": "2026-01-06",
            "dividOperateDate": "2026-01-07",
            "dividPayDate": "2026-01-09",
            "dividCashPsBeforeTax": 0.13789,
            "dividStocksPs": 0.0,
            "dividReserveToStockPs": 0.0,
        }

        # Set high breadth 0.65 to activate high_budget=7500 (700 shares)
        engine = ModularReplayEngine(
            frames={code: bars},
            calendar=calendar,
            signals=signals,
            dividends=[event],
            breadth_lookup={d: 0.65 for d in calendar},
            config=Config(capital=30000.0, order_budget=7500.0, dividend_tax=0.20),
            enable_cash_overlay=True,
        )
        engine.run()

        # 1. Assert daily balance sheet conservation held on all 5 days (< 1e-6)
        self.assertEqual(len(engine.equity), 5)
        for row in engine.equity:
            self.assertGreater(row["equity"], 0)

        # 2. Check position dividends received
        pos = engine.positions.get(code)
        self.assertIsNotNone(pos)
        self.assertEqual(pos.shares, 700)
        expected_net_div = 700 * 0.13789 * (1 - 0.20)
        self.assertAlmostEqual(pos.dividends, expected_net_div, places=6)

        # 3. Independent audit_ledger verification
        raw_bars = {code: bars.set_index("date").to_dict("index")}
        audit_res = audit_ledger(engine, raw_bars, [event])
        self.assertTrue(audit_res["passed"])
        self.assertEqual(audit_res["daily_nav_checked"], 5)

    def test_odd_lot_cash_yield_accumulated_precision(self):
        """Validate daily cash yield accrual does not produce float drift > 1e-6 over 60 trading days."""
        days = [f"2026-01-{i:02d}" for i in range(1, 32)] + [f"2026-02-{i:02d}" for i in range(1, 30)]
        code = "600000"
        bars = pd.DataFrame([
            make_test_bar(d, code, 10.0, 10.2, 9.8, 10.0) for d in days
        ])

        # Odd initial capital: 29876.54321
        engine = ModularReplayEngine(
            frames={code: bars},
            calendar=days,
            signals=pd.DataFrame(columns=["date", "code", "score"]),
            config=Config(capital=29876.54321),
            enable_cash_overlay=True,
        )
        engine.run()

        # Verify daily equity exactly matches compounded cash
        rate = 0.022 / 242.0
        expected_cash = 29876.54321
        for row in engine.equity:
            expected_cash += expected_cash * rate
            self.assertAlmostEqual(row["cash"], expected_cash, places=6)
            self.assertAlmostEqual(row["equity"], expected_cash, places=6)

        raw_bars = {code: bars.set_index("date").to_dict("index")}
        audit_res = audit_ledger(engine, raw_bars, [])
        self.assertTrue(audit_res["passed"])

    def test_stock_sold_between_ex_and_pay_date_preserves_receivable(self):
        """Dividend receivable accrued on ex-date must be safely paid to cash even if position is closed before pay-date."""
        calendar = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-12"]
        code = "600000"

        bars = pd.DataFrame([
            make_test_bar("2026-01-05", code, 10.00, 10.20, 9.90, 10.00),  # Buy queued
            make_test_bar("2026-01-06", code, 10.00, 10.30, 9.95, 10.10),  # Buy filled, record date
            make_test_bar("2026-01-07", code, 9.90, 10.00, 9.80, 9.90, preclose=10.10 - 0.10),  # Ex-date
            make_test_bar("2026-01-08", code, 9.90, 10.10, 9.85, 10.00),  # Position closed by stopping
            make_test_bar("2026-01-09", code, 10.00, 10.20, 9.90, 10.10),  # Receivable waiting
            make_test_bar("2026-01-12", code, 10.10, 10.30, 10.00, 10.20), # Pay date!
        ])

        signals = pd.DataFrame([{"date": "2026-01-05", "code": code, "score": 1.0}])

        event = {
            "code": code,
            "dividRegistDate": "2026-01-06",
            "dividOperateDate": "2026-01-07",
            "dividPayDate": "2026-01-12",
            "dividCashPsBeforeTax": 0.10,
            "dividStocksPs": 0.0,
            "dividReserveToStockPs": 0.0,
        }

        # Stopping protocol that forces exit on day 2026-01-08 morning
        class ForceExitOnDay:
            def evaluate_morning(self, position, day, holding_age, max_hold, context):
                if day == "2026-01-08":
                    return ExitDecision(should_exit=True, exit_type="morning_open", exit_reason="forced_exit")
                return ExitDecision(should_exit=False)

            def evaluate_intraday(self, position, bar, day, holding_age, context):
                return ExitDecision(should_exit=False)

            def check_intraday_touch(self, position, bar, context):
                return ExitDecision(should_exit=False)

            def check_close_stopping(self, position, bar, context):
                return ExitDecision(should_exit=False)

        # In neutral regime (breadth 0.5), budget is 5500 -> 500 shares
        engine = ModularReplayEngine(
            frames={code: bars},
            calendar=calendar,
            signals=signals,
            dividends=[event],
            stopping=ForceExitOnDay(),
            config=Config(capital=30000.0),
            enable_cash_overlay=False,
        )
        engine.run()

        # On 2026-01-08, position was sold.
        # Check that on 2026-01-08 and 2026-01-09, receivables > 0 while len(positions) == 0
        eq_08 = [r for r in engine.equity if r["date"] == "2026-01-08"][0]
        eq_09 = [r for r in engine.equity if r["date"] == "2026-01-09"][0]
        eq_12 = [r for r in engine.equity if r["date"] == "2026-01-12"][0]

        expected_div = 500 * 0.10 * 0.8  # 40.0 RMB
        self.assertEqual(eq_08["positions"], 0)
        self.assertAlmostEqual(eq_08["receivables"], expected_div, places=6)
        self.assertAlmostEqual(eq_09["receivables"], expected_div, places=6)
        # On pay date (2026-01-12), receivables cleared to 0 and cash increased
        self.assertEqual(eq_12["receivables"], 0.0)
        self.assertAlmostEqual(eq_12["cash"], eq_09["cash"] + expected_div, places=6)

        # Independent audit
        raw_bars = {code: bars.set_index("date").to_dict("index")}
        audit_res = audit_ledger(engine, raw_bars, [event])
        self.assertTrue(audit_res["passed"])


class TestConfigOptionPropagation(unittest.TestCase):
    """Area 3: Verify uniform propagation of config options without hardcoded constants."""

    def test_commission_propagation(self):
        """Commission rates scale execution fees uniformly."""
        calendar = ["2026-01-05", "2026-01-06"]
        code = "600000"
        bars = pd.DataFrame([
            make_test_bar("2026-01-05", code, 10.0, 10.2, 9.8, 10.0),
            make_test_bar("2026-01-06", code, 10.0, 10.2, 9.8, 10.0),
        ])
        signals = pd.DataFrame([{"date": "2026-01-05", "code": code, "score": 1.0}])

        # 1. Normal commission 万2.5 (0.00025) with min_commission=0 to check raw scaling
        cfg1 = Config(capital=30000.0, order_budget=7500.0, commission=0.00025, min_commission=0.0)
        eng1 = ModularReplayEngine(frames={code: bars}, calendar=calendar, signals=signals, config=cfg1)
        eng1.run()
        fill1 = eng1.fills[0]
        expected_comm1 = tick(fill1["notional"] * 0.00025)
        self.assertEqual(fill1["commission"], expected_comm1)

        # 2. Quadrupled commission 万10 (0.0010)
        cfg2 = Config(capital=30000.0, order_budget=7500.0, commission=0.0010, min_commission=0.0)
        eng2 = ModularReplayEngine(frames={code: bars}, calendar=calendar, signals=signals, config=cfg2)
        eng2.run()
        fill2 = eng2.fills[0]
        expected_comm2 = tick(fill2["notional"] * 0.0010)
        self.assertEqual(fill2["commission"], expected_comm2)
        self.assertAlmostEqual(fill2["commission"], fill1["commission"] * 4.0, delta=0.05)

    def test_min_commission_floor_propagation(self):
        """Min commission floor is enforced when notional * commission < min_commission."""
        calendar = ["2026-01-05", "2026-01-06"]
        code = "600000"
        bars = pd.DataFrame([
            make_test_bar("2026-01-05", code, 10.0, 10.2, 9.8, 10.0),
            make_test_bar("2026-01-06", code, 10.0, 10.2, 9.8, 10.0),
        ])
        signals = pd.DataFrame([{"date": "2026-01-05", "code": code, "score": 1.0}])

        # Order of ~5000 RMB: 5000 * 0.00025 = 1.25 RMB < 5.0 RMB min commission
        cfg_min5 = Config(capital=30000.0, order_budget=7500.0, commission=0.00025, min_commission=5.0)
        eng_min5 = ModularReplayEngine(frames={code: bars}, calendar=calendar, signals=signals, config=cfg_min5)
        eng_min5.run()
        self.assertEqual(eng_min5.fills[0]["commission"], 5.0)

        # Raise min_commission to 10.0 RMB
        cfg_min10 = Config(capital=30000.0, order_budget=7500.0, commission=0.00025, min_commission=10.0)
        eng_min10 = ModularReplayEngine(frames={code: bars}, calendar=calendar, signals=signals, config=cfg_min10)
        eng_min10.run()
        self.assertEqual(eng_min10.fills[0]["commission"], 10.0)

    def test_stamp_tax_propagation_sell_side(self):
        """Stamp tax applies only on sell side and scales proportionally."""
        calendar = ["2026-01-05", "2026-01-06", "2026-01-07"]
        code = "600000"
        bars = pd.DataFrame([
            make_test_bar("2026-01-05", code, 10.0, 10.2, 9.8, 10.0),
            make_test_bar("2026-01-06", code, 10.0, 10.2, 9.8, 10.0),
            make_test_bar("2026-01-07", code, 10.0, 10.2, 9.8, 10.0),
        ])
        signals = pd.DataFrame([{"date": "2026-01-05", "code": code, "score": 1.0}])

        class SellNextDay:
            def evaluate_morning(self, position, day, holding_age, max_hold, context):
                if day == "2026-01-07":
                    return ExitDecision(should_exit=True, exit_type="morning_open", exit_reason="exit")
                return ExitDecision(should_exit=False)

            def evaluate_intraday(self, position, bar, day, holding_age, context):
                return ExitDecision(should_exit=False)

            def check_intraday_touch(self, position, bar, context):
                return ExitDecision(should_exit=False)

            def check_close_stopping(self, position, bar, context):
                return ExitDecision(should_exit=False)

        # Standard stamp 千0.5 (0.0005)
        cfg_half = Config(capital=30000.0, order_budget=7500.0, stamp_tax=0.0005)
        eng_half = ModularReplayEngine(frames={code: bars}, calendar=calendar, signals=signals, config=cfg_half, stopping=SellNextDay())
        eng_half.run()
        sell_fill_half = [f for f in eng_half.fills if f["side"] == "sell"][0]
        self.assertEqual(sell_fill_half["stamp"], tick(sell_fill_half["notional"] * 0.0005))

        # Doubled stamp 千1.0 (0.0010)
        cfg_one = Config(capital=30000.0, order_budget=7500.0, stamp_tax=0.0010)
        eng_one = ModularReplayEngine(frames={code: bars}, calendar=calendar, signals=signals, config=cfg_one, stopping=SellNextDay())
        eng_one.run()
        sell_fill_one = [f for f in eng_one.fills if f["side"] == "sell"][0]
        self.assertEqual(sell_fill_one["stamp"], tick(sell_fill_one["notional"] * 0.0010))
        self.assertEqual(sell_fill_one["stamp"], sell_fill_half["stamp"] * 2.0)

    def test_slippage_propagation_normal_vs_double(self):
        """Slippage scales from 万5 to 万10 exactly."""
        calendar = ["2026-01-05", "2026-01-06"]
        code = "600000"
        bars = pd.DataFrame([
            make_test_bar("2026-01-05", code, 10.0, 10.2, 9.8, 10.0),
            make_test_bar("2026-01-06", code, 10.0, 10.2, 9.8, 10.0),
        ])
        signals = pd.DataFrame([{"date": "2026-01-05", "code": code, "score": 1.0}])

        cfg_slip5 = Config(capital=30000.0, order_budget=7500.0, slippage=0.0005)
        eng_slip5 = ModularReplayEngine(frames={code: bars}, calendar=calendar, signals=signals, config=cfg_slip5)
        eng_slip5.run()
        slip5_fee = eng_slip5.fills[0]["slippage"]
        self.assertEqual(slip5_fee, tick(eng_slip5.fills[0]["notional"] * 0.0005))

        cfg_slip10 = Config(capital=30000.0, order_budget=7500.0, slippage=0.0010)
        eng_slip10 = ModularReplayEngine(frames={code: bars}, calendar=calendar, signals=signals, config=cfg_slip10)
        eng_slip10.run()
        slip10_fee = eng_slip10.fills[0]["slippage"]
        self.assertEqual(slip10_fee, tick(eng_slip10.fills[0]["notional"] * 0.0010))
        self.assertEqual(slip10_fee, slip5_fee * 2.0)

    def test_dividend_tax_propagation(self):
        """Dividend tax withholding scales from 20% to 10% when configured."""
        calendar = ["2026-01-05", "2026-01-06", "2026-01-07"]
        code = "600000"
        bars = pd.DataFrame([
            make_test_bar("2026-01-05", code, 10.0, 10.2, 9.8, 10.0),
            make_test_bar("2026-01-06", code, 10.0, 10.2, 9.8, 10.0),
            make_test_bar("2026-01-07", code, 9.9, 10.1, 9.8, 10.0, preclose=9.90),
        ])
        signals = pd.DataFrame([{"date": "2026-01-05", "code": code, "score": 1.0}])
        event = {
            "code": code,
            "dividRegistDate": "2026-01-06",
            "dividOperateDate": "2026-01-07",
            "dividPayDate": "2026-01-07",
            "dividCashPsBeforeTax": 0.10,
            "dividStocksPs": 0.0,
            "dividReserveToStockPs": 0.0,
        }

        # 20% tax: net 0.08 / share -> 500 * 0.08 = 40.0
        cfg20 = Config(capital=30000.0, dividend_tax=0.20)
        eng20 = ModularReplayEngine(frames={code: bars}, calendar=calendar, signals=signals, dividends=[event], config=cfg20, enable_cash_overlay=False)
        eng20.run()
        self.assertAlmostEqual(eng20.positions[code].dividends, 40.0, places=6)

        # 10% tax: net 0.09 / share -> 500 * 0.09 = 45.0
        cfg10 = Config(capital=30000.0, dividend_tax=0.10)
        eng10 = ModularReplayEngine(frames={code: bars}, calendar=calendar, signals=signals, dividends=[event], config=cfg10, enable_cash_overlay=False)
        eng10.run()
        self.assertAlmostEqual(eng10.positions[code].dividends, 45.0, places=6)

    def test_cash_annual_rate_propagation_vulnerability(self):
        """CHALLENGE / VULNERABILITY AUDIT:
        ModularReplayEngine.__init__ accepts cash_annual_rate: float = 0.022.
        Does passing cash_annual_rate=0.05 update self.daily_cash_yield?
        Empirical finding: self.daily_cash_yield is NOT updated and remains 0.022 / 242.0.
        """
        engine_default = ModularReplayEngine(calendar=["2026-01-01"], enable_cash_overlay=True)
        self.assertAlmostEqual(engine_default.daily_cash_yield, 0.022 / 242.0, places=9)

        # Pass a non-default cash_annual_rate of 0.05 (5.0%)
        engine_custom = ModularReplayEngine(calendar=["2026-01-01"], enable_cash_overlay=True, cash_annual_rate=0.05)

        # Verify whether cash_annual_rate propagated:
        # Expected if properly propagated: 0.05 / 242.0
        # Actual observed: 0.022 / 242.0 (unpropagated defect)
        is_propagated = abs(engine_custom.daily_cash_yield - (0.05 / 242.0)) < 1e-9
        self.assertTrue(is_propagated, "cash_annual_rate must be properly propagated to daily_cash_yield")


if __name__ == "__main__":
    unittest.main()
