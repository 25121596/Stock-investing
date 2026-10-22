"""E2E Test Suite: Real Microstructure Rules & Ledger Balance Conservation (Tiers 1-4).

Requirements Enforced:
1. R3: Real microstructure rules from arena.py / engine.py:
   - Initial capital exactly 30,000 RMB.
   - T+1 settlement rule: bought on day T, cannot sell on day T.
   - 100-share round lots on buy orders and budget rejection if < 100 shares.
   - Price limit halts: no buy on limit-up open, no sell on limit-down open (10% main board, 5% ST).
   - Suspension handling: tradestatus != 1 or volume <= 0 blocks trading.
2. Fee Calculations:
   - Stamp duty: 0.05% (0.0005), sell only.
   - Commission: 0.025% (0.00025), minimum 5 RMB, buy & sell.
   - Transfer fee: 0.001% (0.00001), buy & sell.
   - Normal slippage: 5 bps (0.0005) vs double slippage: 10 bps (0.0010).
3. R4: Accounting Ledger Conservation:
   - Assert abs(NAV - (Cash + Position_Value + Receivables)) < 1e-6 daily and on exit.
   - Initial capital + closed_pnl + open_pnl == final_equity.
   - Cash >= -1e-7 and shares >= 0 at all times.
   - Independent verification via arena.audit_ledger.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import arena
from engine import Config, Position, Replay, tick
from v08_engine_verified import VerifiedV08Replay


class TestE2ELedgerConservation(unittest.TestCase):
    """Comprehensive Tier 1-4 tests for retail microstructure rules & accounting ledger conservation."""

    def setUp(self):
        self.default_cfg = Config(
            capital=30000.0,
            order_budget=5000.0,
            max_positions=3,
            max_exposure=0.5,
            hold_days=3,
            stop_loss=0.03,
            commission=0.00025,
            min_commission=5.0,
            transfer_fee=0.00001,
            stamp_tax=0.0005,
            slippage=0.0005,
        )

    # -------------------------------------------------------------------------
    # Tier 1: Core Feature Coverage (Happy Path)
    # -------------------------------------------------------------------------

    def test_tier1_initial_capital_invariant(self):
        """Tier 1: Verify 30,000 RMB initial capital requirement and baseline state."""
        cfg = Config()
        self.assertEqual(cfg.capital, 30000.0, "Default capital must be exactly 30,000 RMB")

        days = ["2025-01-02", "2025-01-03"]
        rep = Replay({}, days, pd.DataFrame(columns=["date", "code", "score"]), config=cfg)
        rep.run()

        # At start and finish with no trades, Cash == NAV == 30,000
        self.assertAlmostEqual(rep.cash, 30000.0, places=6)
        self.assertAlmostEqual(rep.nav(), 30000.0, places=6)
        summary = rep.summary()
        self.assertEqual(summary["initial_capital"], 30000.0)
        self.assertAlmostEqual(summary["final_equity"], 30000.0, places=6)
        self.assertAlmostEqual(summary["total_return"], 0.0, places=6)

    def test_tier1_fee_calculations_exactness(self):
        """Tier 1: Verify exact fee schedule (stamp 0.05%, commission 0.025% min 5, transfer 0.001%, slippage 5/10 bps)."""
        cfg = self.default_cfg

        # Case A: Small notional = 2,000 RMB (min commission 5 RMB triggers)
        # Notional * 0.00025 = 0.50 RMB < 5.0 RMB -> commission = 5.00
        # Transfer = 2000 * 0.00001 = 0.02
        # Stamp = 0 on buy, 2000 * 0.0005 = 1.00 on sell
        # Slippage = 2000 * 0.0005 = 1.00
        buy_fees = cfg.fees(2000.0, sell=False)
        self.assertEqual(buy_fees["commission"], 5.0)
        self.assertEqual(buy_fees["transfer"], 0.02)
        self.assertEqual(buy_fees["stamp"], 0.0)
        self.assertEqual(buy_fees["slippage"], 1.00)

        sell_fees = cfg.fees(2000.0, sell=True)
        self.assertEqual(sell_fees["commission"], 5.0)
        self.assertEqual(sell_fees["transfer"], 0.02)
        self.assertEqual(sell_fees["stamp"], 1.00)
        self.assertEqual(sell_fees["slippage"], 1.00)

        # Case B: Large notional = 40,000 RMB (commission > 5 RMB triggers)
        # 40000 * 0.00025 = 10.00 RMB
        # Transfer = 40000 * 0.00001 = 0.40 RMB
        # Stamp (sell) = 40000 * 0.0005 = 20.00 RMB
        # Slippage = 40000 * 0.0005 = 20.00 RMB
        large_sell_fees = cfg.fees(40000.0, sell=True)
        self.assertEqual(large_sell_fees["commission"], 10.00)
        self.assertEqual(large_sell_fees["transfer"], 0.40)
        self.assertEqual(large_sell_fees["stamp"], 20.00)
        self.assertEqual(large_sell_fees["slippage"], 20.00)

        # Case C: Double slippage (10 bps = 0.0010)
        stress_cfg = replace(cfg, slippage=0.0010)
        stress_fees = stress_cfg.fees(40000.0, sell=True)
        self.assertEqual(stress_fees["slippage"], 40.00, "Double slippage must be exactly 10 bps (40 RMB on 40k)")

    def test_tier1_round_lot_100_shares_enforced(self):
        """Tier 1: Buy orders must strictly be multiples of 100 shares."""
        days = ["2025-01-02", "2025-01-03"]
        code = "sh.600000"
        bar = {"date": "2025-01-02", "code": code, "open": 10.0, "high": 10.0, "low": 10.0,
               "close": 10.0, "preclose": 10.0, "volume": 100000.0, "tradestatus": 1, "isST": 0}
        frames = {code: pd.DataFrame([bar, {**bar, "date": "2025-01-03"}])}
        bars = {code: frames[code].set_index("date").to_dict("index")}

        # A: Buy 400 shares (round lot of 100) -> allowed
        rep = Replay(frames, days, pd.DataFrame(columns=["date", "code", "score"]))
        valid_order = {"code": code, "shares": 400, "limit": 10.5, "signal_date": "2025-01-02"}
        rep.buy(valid_order, "2025-01-02", 0)
        self.assertIn(code, rep.positions)
        self.assertEqual(rep.positions[code].shares, 400)

        # B: Budget insufficient for 100 shares rejects order with lot_size_or_budget
        rep_tiny = Replay(frames, days, pd.DataFrame([{"date": "2025-01-02", "code": code, "score": 1.0}]),
                          config=Config(capital=30000.0, order_budget=500.0))  # 500 RMB budget cannot buy 100 shares at 10 RMB
        rep_tiny.run()
        self.assertGreater(rep_tiny.rejected["lot_size_or_budget"], 0)

        # C: audit_ledger catches non-lot buy
        rep_fake_odd = Replay(frames, days, pd.DataFrame(columns=["date", "code", "score"]))
        odd_order = {"code": code, "shares": 250, "limit": 10.5, "signal_date": "2025-01-02"}
        rep_fake_odd.buy(odd_order, "2025-01-02", 0)
        rep_fake_odd.equity = [{"date": "2025-01-02", "cash": rep_fake_odd.cash, "receivables": 0, "equity": rep_fake_odd.nav()}]
        with self.assertRaisesRegex(ValueError, "Non-lot buy"):
            arena.audit_ledger(rep_fake_odd, bars, [])

    def test_tier1_t_plus_1_settlement_enforced(self):
        """Tier 1: Same-day exit (T+0) is strictly prohibited under A-share rules."""
        days = ["2025-01-02", "2025-01-03"]
        code = "sh.600000"
        bar = {"date": "2025-01-02", "code": code, "open": 10.0, "high": 10.0, "low": 10.0,
               "close": 10.0, "preclose": 10.0, "volume": 100000.0, "tradestatus": 1, "isST": 0}
        frames = {code: pd.DataFrame([bar, {**bar, "date": "2025-01-03"}])}

        rep = Replay(frames, days, pd.DataFrame(columns=["date", "code", "score"]))
        order = {"code": code, "shares": 100, "limit": 10.5, "signal_date": "2025-01-02"}
        rep.buy(order, "2025-01-02", 0)

        # Attempt to sell on day of entry (index 0 == entry_index 0)
        pos = rep.positions[code]
        rep.sell(pos, "2025-01-02", 0)
        self.assertEqual(rep.rejected["T_plus_1"], 1, "Must reject same-day sell with T_plus_1")
        self.assertEqual(pos.shares, 100, "Shares must not be sold on T+0")

        # Sell on next trading day (index 1 > entry_index 0) -> allowed
        rep.sell(pos, "2025-01-03", 1)
        self.assertEqual(pos.shares, 0, "Shares should be sold on T+1")

    def test_tier1_ledger_balance_conservation_daily(self):
        """Tier 1: Assert NAV == Cash + Position_Value + Receivables (< 1e-6) daily."""
        days = ["2025-01-02", "2025-01-03", "2025-01-06"]
        code = "sh.600000"
        prices = [10.0, 10.0, 10.0]
        bars_df = pd.DataFrame([
            {"date": d, "code": code, "open": p, "high": p, "low": p, "close": p,
             "preclose": 10.0, "volume": 100000.0, "tradestatus": 1, "isST": 0}
            for d, p in zip(days, prices)
        ])
        frames = {code: bars_df}
        bars = {code: bars_df.set_index("date").to_dict("index")}
        signals = pd.DataFrame([{"date": days[0], "code": code, "score": 1.0}])

        rep = arena.simulate(frames, bars, days, signals, [], self.default_cfg, {}, {}, {})
        audit = arena.audit_ledger(rep, bars, [])
        self.assertTrue(audit["passed"], "Independent audit must pass")

        # Track cumulative share balance from fills
        running_shares = defaultdict(int)
        fills_by_day = defaultdict(list)
        for f in rep.fills:
            fills_by_day[f["date"]].append(f)

        for row in rep.equity:
            day = row["date"]
            nav = row["equity"]
            cash = row["cash"]
            receivables = row["receivables"]

            for f in fills_by_day[day]:
                sign = 1 if f["side"] == "buy" else -1
                running_shares[f["code"]] += sign * f["shares"]

            pos_val = sum(q * float(bars[c][day]["close"]) for c, q in running_shares.items() if q)
            self.assertLess(
                abs(nav - (cash + pos_val + receivables)), 1e-6,
                f"Ledger conservation violated on {day}: NAV={nav}, Cash={cash}, Pos={pos_val}, Rec={receivables}"
            )

    # -------------------------------------------------------------------------
    # Tier 2: Boundary & Corner Cases
    # -------------------------------------------------------------------------

    def test_tier2_price_limit_halts_blocking(self):
        """Tier 2: No buy on limit-up open (+10%), no sell on limit-down open (-10%)."""
        days = ["2025-01-02", "2025-01-03"]
        code = "sh.600000"

        # Case 1: Limit-up open (+10%)
        # Preclose = 10.0, Open = 11.00 (limit up tick)
        limit_up_bar = {"date": "2025-01-02", "code": code, "open": 11.00, "high": 11.00,
                        "low": 11.00, "close": 11.00, "preclose": 10.0, "volume": 100000.0,
                        "tradestatus": 1, "isST": 0}
        frames_up = {code: pd.DataFrame([limit_up_bar, {**limit_up_bar, "date": "2025-01-03"}])}
        rep_up = Replay(frames_up, days, pd.DataFrame(columns=["date", "code", "score"]))
        rep_up.buy({"code": code, "shares": 100, "limit": 11.5, "signal_date": "2025-01-02"}, "2025-01-02", 0)
        self.assertEqual(rep_up.rejected["limit_up_open"], 1, "Must reject buy at limit-up open")

        # Case 2: Limit-down open (-10%)
        # Enter position on normal day, then next day opens at limit-down (9.00)
        normal_bar = {"date": "2025-01-02", "code": code, "open": 10.00, "high": 10.00,
                      "low": 10.00, "close": 10.00, "preclose": 10.0, "volume": 100000.0,
                      "tradestatus": 1, "isST": 0}
        limit_down_bar = {"date": "2025-01-03", "code": code, "open": 9.00, "high": 9.00,
                          "low": 9.00, "close": 9.00, "preclose": 10.0, "volume": 100000.0,
                          "tradestatus": 1, "isST": 0}
        frames_down = {code: pd.DataFrame([normal_bar, limit_down_bar])}
        rep_down = Replay(frames_down, days, pd.DataFrame(columns=["date", "code", "score"]))
        rep_down.buy({"code": code, "shares": 100, "limit": 10.5, "signal_date": "2025-01-02"}, "2025-01-02", 0)
        pos = rep_down.positions[code]
        # Attempt to sell at limit-down open
        rep_down.sell(pos, "2025-01-03", 1)
        self.assertEqual(rep_down.rejected["limit_down_open"], 1, "Must reject sell at limit-down open")
        self.assertEqual(pos.shares, 100, "Shares must not sell during limit-down halt")

    def test_tier2_suspension_handling(self):
        """Tier 2: Stocks with tradestatus != 1 or volume <= 0 cannot be traded."""
        days = ["2025-01-02"]
        code = "sh.600000"

        # Tradestatus = 0 (suspended)
        suspended_bar = {"date": "2025-01-02", "code": code, "open": 10.0, "high": 10.0,
                         "low": 10.0, "close": 10.0, "preclose": 10.0, "volume": 0.0,
                         "tradestatus": 0, "isST": 0}
        frames = {code: pd.DataFrame([suspended_bar])}
        rep = Replay(frames, days, pd.DataFrame(columns=["date", "code", "score"]))
        rep.buy({"code": code, "shares": 100, "limit": 10.5, "signal_date": "2025-01-02"}, "2025-01-02", 0)
        self.assertEqual(rep.rejected["suspended_or_missing"], 1)

    def test_tier2_st_stock_entry_blocked(self):
        """Tier 2: ST stocks (isST != 0) are strictly ineligible for purchase."""
        days = ["2025-01-02"]
        code = "sh.600000"
        st_bar = {"date": "2025-01-02", "code": code, "open": 10.0, "high": 10.0,
                  "low": 10.0, "close": 10.0, "preclose": 10.0, "volume": 100000.0,
                  "tradestatus": 1, "isST": 1}
        frames = {code: pd.DataFrame([st_bar])}
        rep = Replay(frames, days, pd.DataFrame(columns=["date", "code", "score"]))
        rep.buy({"code": code, "shares": 100, "limit": 10.5, "signal_date": "2025-01-02"}, "2025-01-02", 0)
        self.assertEqual(rep.rejected["ST_entry"], 1)

    def test_tier2_cash_exhaustion_and_negative_cash_guard(self):
        """Tier 2: Buy order exceeding available cash is rejected with cash_limit; cash never negative."""
        days = ["2025-01-02"]
        code = "sh.600000"
        bar = {"date": "2025-01-02", "code": code, "open": 100.0, "high": 100.0,
               "low": 100.0, "close": 100.0, "preclose": 100.0, "volume": 100000.0,
               "tradestatus": 1, "isST": 0}
        frames = {code: pd.DataFrame([bar])}
        rep = Replay(frames, days, pd.DataFrame(columns=["date", "code", "score"]),
                     config=Config(capital=30000.0, order_budget=50000.0, max_exposure=1.0))
        # 400 shares at 100 RMB = 40,000 RMB > 30,000 RMB cash
        rep.buy({"code": code, "shares": 400, "limit": 105.0, "signal_date": "2025-01-02"}, "2025-01-02", 0)
        self.assertEqual(rep.rejected["cash_limit"], 1)
        self.assertGreaterEqual(rep.cash, 0.0)

    # -------------------------------------------------------------------------
    # Tier 3: Cross-Feature Interactions & Corporate Actions
    # -------------------------------------------------------------------------

    def test_tier3_corporate_action_dividend_receivables_conservation(self):
        """Tier 3: Cash dividend in transit enters receivables; NAV perfectly conserved daily."""
        days = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"]
        code = "sh.600000"

        # Signal on 2025-01-02 -> Buy fills morning of 2025-01-03.
        # Registration date: 2025-01-03 (position is held at close).
        # Ex-date: 2025-01-06 (dividOperateDate). Preclose drops from 10.0 to 9.50.
        # Payment date: 2025-01-08 (dividPayDate).
        dividend_event = {
            "code": code,
            "dividRegistDate": "2025-01-03",  # Record date
            "dividOperateDate": "2025-01-06", # Ex-date
            "dividPayDate": "2025-01-08",     # Pay-date
            "dividCashPsBeforeTax": "0.50",
            "dividStocksPs": "0.0",
            "dividReserveToStockPs": "0.0",
        }

        # Market prices reflect 0.50 RMB dividend drop on 2025-01-06
        bars_df = pd.DataFrame([
            {"date": "2025-01-02", "code": code, "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "preclose": 10.0, "volume": 100000.0, "tradestatus": 1, "isST": 0},
            {"date": "2025-01-03", "code": code, "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "preclose": 10.0, "volume": 100000.0, "tradestatus": 1, "isST": 0},
            {"date": "2025-01-06", "code": code, "open": 9.50, "high": 9.50, "low": 9.50, "close": 9.50, "preclose": 9.50, "volume": 100000.0, "tradestatus": 1, "isST": 0},
            {"date": "2025-01-07", "code": code, "open": 9.50, "high": 9.50, "low": 9.50, "close": 9.50, "preclose": 9.50, "volume": 100000.0, "tradestatus": 1, "isST": 0},
            {"date": "2025-01-08", "code": code, "open": 9.50, "high": 9.50, "low": 9.50, "close": 9.50, "preclose": 9.50, "volume": 100000.0, "tradestatus": 1, "isST": 0},
        ])
        frames = {code: bars_df}
        bars = {code: bars_df.set_index("date").to_dict("index")}
        signals = pd.DataFrame([{"date": "2025-01-02", "code": code, "score": 1.0}])

        rep = arena.simulate(frames, bars, days, signals, [dividend_event], replace(self.default_cfg, hold_days=10), {}, {}, {})

        # Assert independent ledger verification passes
        audit = arena.audit_ledger(rep, bars, [dividend_event])
        self.assertTrue(audit["passed"], "Ledger audit must pass during corporate action life-cycle")

        # Check day-by-day conservation using dynamic share tracking
        running_shares = defaultdict(int)
        fills_by_day = defaultdict(list)
        for f in rep.fills:
            fills_by_day[f["date"]].append(f)

        for row in rep.equity:
            day = row["date"]
            nav = row["equity"]
            cash = row["cash"]
            receivables = row["receivables"]

            for f in fills_by_day[day]:
                sign = 1 if f["side"] == "buy" else -1
                running_shares[f["code"]] += sign * f["shares"]

            if day in ["2025-01-06", "2025-01-07"]:
                self.assertGreater(receivables, 0.0, f"Receivables must be held on {day}")
            elif day == "2025-01-08":
                self.assertEqual(receivables, 0.0, f"Receivables must clear on {day}")

            pos_val = sum(q * float(bars[c][day]["close"]) for c, q in running_shares.items() if q)
            self.assertLess(abs(nav - (cash + pos_val + receivables)), 1e-6)

    def test_tier3_adversarial_cash_tampering_caught(self):
        """Tier 3 Adversarial: Independent audit_ledger must detect cash tampering by even 0.01 RMB."""
        days = ["2025-01-02", "2025-01-03"]
        code = "sh.600000"
        bar = {"date": "2025-01-02", "code": code, "open": 10.0, "high": 10.0, "low": 10.0,
               "close": 10.0, "preclose": 10.0, "volume": 100000.0, "tradestatus": 1, "isST": 0}
        frames = {code: pd.DataFrame([bar, {**bar, "date": "2025-01-03"}])}
        bars = {code: frames[code].set_index("date").to_dict("index")}

        rep = arena.simulate(frames, bars, days, pd.DataFrame(columns=["date", "code", "score"]), [], self.default_cfg, {}, {}, {})
        self.assertTrue(arena.audit_ledger(rep, bars, [])["passed"])

        # Tamper with recorded equity cash
        rep.equity[-1]["cash"] += 0.01
        with self.assertRaisesRegex(ValueError, "ledger mismatch"):
            arena.audit_ledger(rep, bars, [])

    def test_tier3_adversarial_fill_price_tampering_caught(self):
        """Tier 3 Adversarial: Tampering with fill price violates price matching invariant."""
        days = ["2025-01-02", "2025-01-03"]
        code = "sh.600000"
        bar = {"date": "2025-01-02", "code": code, "open": 10.0, "high": 10.0, "low": 10.0,
               "close": 10.0, "preclose": 10.0, "volume": 100000.0, "tradestatus": 1, "isST": 0}
        frames = {code: pd.DataFrame([bar, {**bar, "date": "2025-01-03"}])}
        bars = {code: frames[code].set_index("date").to_dict("index")}
        signals = pd.DataFrame([{"date": "2025-01-02", "code": code, "score": 1.0}])

        rep = arena.simulate(frames, bars, days, signals, [], self.default_cfg, {}, {}, {})
        self.assertTrue(arena.audit_ledger(rep, bars, [])["passed"])

        # Falsify fill price
        rep.fills[0]["price"] += 0.05
        with self.assertRaisesRegex(ValueError, "Fill price/notional/fees mismatch"):
            arena.audit_ledger(rep, bars, [])

    # -------------------------------------------------------------------------
    # Tier 4: Scenario / Real-World Stress Test Runner
    # -------------------------------------------------------------------------

    def test_tier4_full_lifecycle_multi_trade_double_slippage_audit(self):
        """Tier 4: Multi-trade simulation under double slippage (10 bps) with full ledger audit."""
        days = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"]
        codes = ["sh.600000", "sh.600001"]

        # 2 stocks with consistent preclose matching previous day's close
        frames = {}
        for code, base_p in zip(codes, [10.0, 20.0]):
            records = []
            prev_c = base_p
            for i, d in enumerate(days):
                p = round(base_p * (1.0 + 0.01 * i), 2)
                records.append({
                    "date": d, "code": code, "open": p, "high": p * 1.01, "low": p * 0.99,
                    "close": p, "preclose": prev_c, "volume": 100000.0, "tradestatus": 1, "isST": 0
                })
                prev_c = p
            frames[code] = pd.DataFrame(records)

        bars = {c: frames[c].set_index("date").to_dict("index") for c in codes}
        signals = pd.DataFrame([
            {"date": "2025-01-02", "code": "sh.600000", "score": 0.9},
            {"date": "2025-01-03", "code": "sh.600001", "score": 0.8},
        ])

        # Run under double slippage (10 bps)
        double_slip_cfg = replace(self.default_cfg, slippage=0.0010)
        rep = arena.simulate(frames, bars, days, signals, [], double_slip_cfg, {}, {}, {})

        # 1. Independent ledger audit
        audit = arena.audit_ledger(rep, bars, [])
        self.assertTrue(audit["passed"], "Ledger audit under double slippage failed")
        self.assertGreater(audit["fills_checked"], 0)

        # 2. Verify all slippage fees were calculated at 10 bps
        for fill in rep.fills:
            expected_slippage = tick(fill["notional"] * 0.0010)
            self.assertEqual(fill["slippage"], expected_slippage)

        # 3. Final summary reconciliation check
        summary = rep.summary()
        closed_pnl = sum(trade["net_pnl"] for trade in rep.trades)
        open_pnl = sum(item["marked_pnl"] for item in rep.holdings())
        reconciliation = summary["final_equity"] - summary["initial_capital"] - closed_pnl - open_pnl
        self.assertLess(abs(reconciliation), 1e-6)
        self.assertEqual(summary["quality_warning_count"], 0)


if __name__ == "__main__":
    unittest.main()
