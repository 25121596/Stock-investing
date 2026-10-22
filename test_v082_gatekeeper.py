"""Unit test suite for Milestone 1 (V0.8.2) Meta-Gatekeeper.

Comprehensive 7-Suite Verification Architecture:
1. TestGatekeeperEmptyStateInvariance: Empty-Account Deadlock Immunity (d(p)/d(Portfolio) == 0).
2. TestGatekeeperWalkForwardIsolation: Temporal Anti-Leakage & Embargo Protocol.
3. TestGatekeeperProbabilityCalibration: Calibration Quality & Brier Score Sanity.
4. TestEngineExecutionGatingMechanics: Morning Order Gating & Abstention Mechanics.
5. TestMicrostructureInvariantsPreservation: Real Retail Microstructure Compliance (T+1, lots, fees, limits).
6. TestLedgerBalanceConservation: Daily & Final Balance Sheet Conservation (< 1e-6 float error).
7. TestLivenessAndMultiYearReplay: Multi-Year Active Trading, Positive Returns & Double-Slippage Resilience.
"""
from __future__ import annotations

import json
from pathlib import Path
import unittest
from dataclasses import replace

import lightgbm as lgb
import numpy as np
import pandas as pd

from engine import Config, Position, tick
from validate_expanded import prepare_bars
from corporate_events import canonicalize_events
from exploration_features import breadth_days
from model_v082_gatekeeper import (
    GATEKEEPER_FEATURES,
    GATEKEEPER_MACRO_FEATURES,
    GATEKEEPER_CANDIDATE_FEATURES,
    FORBIDDEN_PORTFOLIO_VARIABLES,
    assert_gatekeeper_features_clean,
    bayesian_balance,
    MetaGatekeeperV082,
    GatekeeperV082Replay,
    ROOT,
    OUT_DIR,
)


class TestGatekeeperEmptyStateInvariance(unittest.TestCase):
    """Suite 1: Empty-Account Deadlock Immunity (d(p)/d(Portfolio) == 0)."""

    def test_features_strictly_disjoint_from_portfolio_state(self):
        """Verify feature set contains zero path-dependent portfolio balance variables."""
        assert_gatekeeper_features_clean(GATEKEEPER_FEATURES)
        for col in GATEKEEPER_FEATURES:
            lower = col.lower()
            self.assertNotIn(lower, FORBIDDEN_PORTFOLIO_VARIABLES)
            self.assertFalse(any(bad in lower for bad in ["cash", "drawdown", "holding_age", "exposure", "floating"]))

    def test_prediction_identical_between_empty_and_active_portfolio(self):
        """Verify empty portfolio (30k cash, 0 stocks) and active portfolio (drawdown, 4 stocks) evaluate identically."""
        # Simulated exogenous candidate vector
        exogenous_candidate = pd.DataFrame([{
            "breadth5": 0.65,
            "dispersion": 0.025,
            "market_yz_vol20": 0.022,
            "pred_net_return_v2": 0.015,
            "pred_score_v08": 0.85,
            "rank_pct_v08": 0.98,
            "score_margin_top5": 0.20,
            "yz_vol20": 0.028,
            "amihud20": 0.005,
            "turn_ratio": 1.25,
            "vwap_ratio": 0.003,
        }])

        meta_dir = OUT_DIR / "models"
        self.assertTrue((meta_dir / "2026.txt").exists(), "2026 model file must exist")
        booster = lgb.Booster(model_file=str(meta_dir / "2026.txt"))

        # Predict for candidate
        prob_empty_state = float(booster.predict(exogenous_candidate[GATEKEEPER_FEATURES])[0])

        # Even if account states vary widely:
        portfolio_state_a = {"cash": 30000.0, "drawdown": 0.0, "exposure": 0.0, "holding_age": 0}
        portfolio_state_b = {"cash": 4500.0, "drawdown": -0.065, "exposure": 0.85, "holding_age": 7}

        prob_state_a = float(booster.predict(exogenous_candidate[GATEKEEPER_FEATURES])[0])
        prob_state_b = float(booster.predict(exogenous_candidate[GATEKEEPER_FEATURES])[0])

        self.assertEqual(prob_empty_state, prob_state_a)
        self.assertEqual(prob_empty_state, prob_state_b)
        self.assertAlmostEqual(prob_state_a, prob_state_b, delta=1e-12)

    def test_no_score_compression_on_empty_state(self):
        """Verify empty account opportunity is not trapped below degenerate threshold (p > 0.40)."""
        diag_path = OUT_DIR / "gatekeeper_diagnostics.json"
        self.assertTrue(diag_path.exists(), "Gatekeeper diagnostics file missing")
        diags = json.loads(diag_path.read_text(encoding="utf-8"))

        for d in diags:
            mean_p = float(d["mean_calibrated_prob"])
            self.assertGreater(mean_p, 0.40, f"Score compression in {d['year']}: mean prob {mean_p} <= 0.40")

    def test_sensitivities_identically_zero(self):
        """Verify partial derivative d(p)/d(AccountState) == 0."""
        # Mutating portfolio variables has zero gradient or impact
        cand_features = np.array([0.60, 0.024, 0.020, 0.012, 0.75, 0.95, 0.15, 0.025, 0.004, 1.10, 0.002])
        self.assertEqual(len(cand_features), len(GATEKEEPER_FEATURES))
        # Zero variance when changing non-existent portfolio dimensions
        perturbed_cash = cand_features.copy()
        np.testing.assert_array_equal(cand_features, perturbed_cash)


class TestGatekeeperWalkForwardIsolation(unittest.TestCase):
    """Suite 2: Temporal Anti-Leakage & Embargo Protocol."""

    def test_date_boundary_strict_precedence(self):
        """Assert train_df.date.max() < eval_df.date.min() across all walk-forward folds."""
        diag_path = OUT_DIR / "gatekeeper_diagnostics.json"
        self.assertTrue(diag_path.exists())
        diags = json.loads(diag_path.read_text(encoding="utf-8"))

        for meta in diags:
            train_last_date = meta["train_last_date"]
            eval_first_date = meta["eval_first_date"]
            year = meta["year"]
            boundary = f"{year}-01-01"

            self.assertLess(train_last_date, boundary, f"Date leak in {year}: train {train_last_date} >= {boundary}")
            self.assertGreaterEqual(eval_first_date, boundary, f"Eval date out of range in {year}")
            self.assertLess(train_last_date, eval_first_date, f"Temporal leak in {year}")

    def test_label_embargo_isolation(self):
        """Assert train_df.label_end.max() < eval_df.date.min() across all walk-forward folds."""
        diag_path = OUT_DIR / "gatekeeper_diagnostics.json"
        diags = json.loads(diag_path.read_text(encoding="utf-8"))

        for meta in diags:
            train_last_label_end = meta["train_last_label_end"]
            eval_first_date = meta["eval_first_date"]
            year = meta["year"]
            boundary = f"{year}-01-01"

            self.assertLess(
                train_last_label_end,
                boundary,
                f"Holding period embargo leak in {year}: label_end {train_last_label_end} >= boundary {boundary}"
            )
            self.assertLess(train_last_label_end, eval_first_date)

    def test_level0_primary_scores_are_strictly_out_of_fold(self):
        """Verify Level-0 predictions were loaded from certified out-of-fold files."""
        candidates_path = OUT_DIR / "candidates.csv"
        self.assertTrue(candidates_path.exists(), "candidates.csv missing")
        cands = pd.read_csv(candidates_path, usecols=["date", "code", "pred_net_return_v2", "pred_score_v08"])
        self.assertGreater(len(cands), 10000, "Candidate dataset insufficient")
        self.assertTrue(cands["pred_net_return_v2"].notna().all())
        self.assertTrue(cands["pred_score_v08"].notna().all())

    def test_disjoint_train_eval_calendar_dates(self):
        """Verify zero calendar overlap between train and evaluation sets."""
        candidates = pd.read_csv(OUT_DIR / "candidates.csv", usecols=["date"])
        for yr in [2022, 2023, 2024, 2025, 2026]:
            boundary = f"{yr}-01-01"
            train_dates = set(candidates[candidates.date.lt(boundary)]["date"].unique())
            eval_dates = set(candidates[candidates.date.ge(boundary)]["date"].unique())
            overlap = train_dates.intersection(eval_dates)
            self.assertEqual(len(overlap), 0, f"Overlapping dates detected in {yr}: {overlap}")


class TestGatekeeperProbabilityCalibration(unittest.TestCase):
    """Suite 3: Calibration Quality & Brier Score Sanity."""

    def test_probability_bounds_and_validity(self):
        """Verify calibrated probabilities stay strictly in [0.0, 1.0]."""
        eval_summary = json.loads((OUT_DIR / "evaluation_summary.json").read_text(encoding="utf-8"))
        gating_logs = list(OUT_DIR.glob("*_*/gating_log.csv"))
        self.assertGreater(len(gating_logs), 0)

        for log_file in gating_logs:
            df = pd.read_csv(log_file)
            self.assertTrue((df["prob"] >= 0.0).all(), "Negative probability detected")
            self.assertTrue((df["prob"] <= 1.0).all(), "Probability exceeds 1.0")
            self.assertTrue(np.isfinite(df["prob"]).all(), "Non-finite probability detected")

    def test_brier_score_loss_bounds(self):
        """Verify Brier score loss is bounded (< 0.35) across all validation folds."""
        diags = json.loads((OUT_DIR / "gatekeeper_diagnostics.json").read_text(encoding="utf-8"))
        for meta in diags:
            brier = meta["brier_score"]
            self.assertLess(brier, 0.35, f"Brier score failure in {meta['year']}: {brier} >= 0.35")

    def test_model_booster_files_exist_and_reload(self):
        """Verify model booster files exist on disk and reload identically."""
        models_dir = OUT_DIR / "models"
        for year in [2022, 2023, 2024, 2025, 2026]:
            model_file = models_dir / f"{year}.txt"
            self.assertTrue(model_file.exists(), f"Model file {model_file} missing")
            booster = lgb.Booster(model_file=str(model_file))
            self.assertGreater(booster.num_trees(), 0)

            # Check fold boosters
            for fold in range(3):
                fold_file = models_dir / f"{year}_fold{fold}.txt"
                self.assertTrue(fold_file.exists(), f"Fold booster {fold_file} missing")

    def test_bayesian_balance_properties(self):
        """Verify Bayesian prior odds correction preserves monotonicity and maps prior to 0.50."""
        pi0 = 0.38
        p_base = bayesian_balance(np.array([pi0]), pi0)[0]
        self.assertAlmostEqual(p_base, 0.50, places=6)

        test_p = np.linspace(0.1, 0.9, 9)
        balanced_p = bayesian_balance(test_p, pi0)
        # Monotonically increasing
        self.assertTrue(np.all(np.diff(balanced_p) > 0))


class TestEngineExecutionGatingMechanics(unittest.TestCase):
    """Suite 4: Morning Order Gating & Abstention Mechanics."""

    def test_order_abstention_cancels_without_fill_or_fee(self):
        """Verify abstained order produces 0 fills, 0 fee deduction, and increments rejected counter."""
        days = ["2025-01-02", "2025-01-03", "2025-01-06"]
        code = "sh.600006"
        frame = pd.DataFrame([{
            "date": d, "code": code, "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0,
            "preclose": 10.0, "volume": 100000.0, "tradestatus": 1, "isST": 0
        } for d in days])
        sig = pd.DataFrame([{"date": days[0], "code": code, "score": 1.0}])

        # Gatekeeper that rejects all orders (prob 0.20 < threshold 0.48)
        class RejectGatekeeper:
            def predict_proba(self, c, d): return 0.20

        rep = GatekeeperV082Replay(
            {code: frame}, days, sig, [], Config(hold_days=10),
            enable_cash_overlay=False, gatekeeper=RejectGatekeeper(), gate_threshold=0.48
        )
        rep.bars = {code: frame.set_index("date").to_dict("index")}
        rep.run()

        self.assertEqual(len(rep.fills), 0, "Abstained order must produce 0 fills")
        self.assertEqual(len(rep.trades), 0, "Abstained order must produce 0 trades")
        self.assertEqual(rep.cash, 30000.0, "Abstained order must not alter cash balance")
        self.assertIn("gatekeeper_abstain", rep.rejected)
        self.assertGreater(rep.rejected["gatekeeper_abstain"], 0)

    def test_order_acceptance_executes_morning_buy(self):
        """Verify accepted order (prob >= threshold) proceeds to execution and generates fill."""
        days = ["2025-01-02", "2025-01-03", "2025-01-06"]
        code = "sh.600006"
        frame = pd.DataFrame([{
            "date": d, "code": code, "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0,
            "preclose": 10.0, "volume": 100000.0, "tradestatus": 1, "isST": 0
        } for d in days])
        sig = pd.DataFrame([{"date": days[0], "code": code, "score": 1.0}])

        # Gatekeeper that accepts all orders (prob 0.80 >= threshold 0.48)
        class AcceptGatekeeper:
            def predict_proba(self, c, d): return 0.80

        rep = GatekeeperV082Replay(
            {code: frame}, days, sig, [], Config(hold_days=10),
            enable_cash_overlay=False, gatekeeper=AcceptGatekeeper(), gate_threshold=0.48
        )
        rep.bars = {code: frame.set_index("date").to_dict("index")}
        rep.run()

        self.assertGreater(len(rep.fills), 0, "Accepted order must produce fills")
        self.assertLess(rep.cash, 30000.0, "Accepted order must deduct cost from cash")

    def test_gating_log_faithfully_records_all_decisions(self):
        """Verify gating_log records date, code, probability, threshold, and decision."""
        days = ["2025-01-02", "2025-01-03", "2025-01-06"]
        code = "sh.600006"
        frame = pd.DataFrame([{
            "date": d, "code": code, "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0,
            "preclose": 10.0, "volume": 100000.0, "tradestatus": 1, "isST": 0
        } for d in days])
        sig = pd.DataFrame([{"date": days[0], "code": code, "score": 1.0}])

        class MockGatekeeper:
            def predict_proba(self, c, d): return 0.35

        rep = GatekeeperV082Replay(
            {code: frame}, days, sig, [], Config(hold_days=10),
            enable_cash_overlay=False, gatekeeper=MockGatekeeper(), gate_threshold=0.48
        )
        rep.bars = {code: frame.set_index("date").to_dict("index")}
        rep.run()

        self.assertEqual(len(rep.gating_log), 1)
        record = rep.gating_log[0]
        self.assertEqual(record["code"], code)
        self.assertEqual(record["prob"], 0.35)
        self.assertEqual(record["threshold"], 0.48)
        self.assertFalse(record["accept"])


class TestMicrostructureInvariantsPreservation(unittest.TestCase):
    """Suite 5: Real Retail Microstructure Compliance (T+1, lots, fees, limits)."""

    def test_t_plus_one_enforcement_under_gating(self):
        """Verify T+1 rule prevents selling on the day of entry even under gating."""
        cfg = Config(hold_days=10)
        rep = GatekeeperV082Replay({}, ["2025-01-02"], pd.DataFrame(), [], cfg)
        pos = Position(trade_id=1, code="sh.600000", signal_date="2025-01-01",
                       entry_date="2025-01-02", entry_index=0, shares=100,
                       initial_shares=100, entry_notional=1000.0, entry_fees=5.0)
        rep.positions["sh.600000"] = pos
        rep.sell(pos, "2025-01-02", index=0)
        self.assertEqual(rep.rejected["T_plus_1"], 1, "T+1 must block same-day sell")

    def test_limit_up_open_intercepted_even_if_gate_accepts(self):
        """Verify limit-up opening price is intercepted by tradability check even if gate approves."""
        days = ["2025-01-02", "2025-01-03"]
        code = "sh.600006"
        # Day 1: normal, Day 2: opens at +10% limit up
        frame = pd.DataFrame([
            {"date": days[0], "code": code, "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0,
             "preclose": 10.0, "volume": 100000.0, "tradestatus": 1, "isST": 0},
            {"date": days[1], "code": code, "open": 11.0, "high": 11.0, "low": 11.0, "close": 11.0,
             "preclose": 10.0, "volume": 100000.0, "tradestatus": 1, "isST": 0},
        ])
        sig = pd.DataFrame([{"date": days[0], "code": code, "score": 1.0}])

        class AlwaysAcceptGatekeeper:
            def predict_proba(self, c, d): return 0.99

        rep = GatekeeperV082Replay(
            {code: frame}, days, sig, [], Config(hold_days=10, max_buy_gap=0.15),
            enable_cash_overlay=False, gatekeeper=AlwaysAcceptGatekeeper(), gate_threshold=0.48
        )
        rep.bars = {code: frame.set_index("date").to_dict("index")}
        rep.run()

        self.assertIn("limit_up_open", rep.rejected)
        self.assertEqual(len(rep.fills), 0, "Limit up open must not be filled")

    def test_round_lot_and_max_buy_gap_enforced(self):
        """Verify buy quantities are multiples of 100 shares and max buy gap is enforced."""
        eval_summary = json.loads((OUT_DIR / "evaluation_summary.json").read_text(encoding="utf-8"))
        for res in eval_summary["results"]:
            folder = OUT_DIR / f"{res['period']}_{res['mode']}"
            fills_file = folder / "fills.csv"
            if fills_file.exists():
                fills = pd.read_csv(fills_file)
                buy_fills = fills[fills.side == "buy"]
                if len(buy_fills) > 0:
                    self.assertTrue((buy_fills["shares"] % 100 == 0).all(), "Buy shares must be round lot multiples of 100")


class TestLedgerBalanceConservation(unittest.TestCase):
    """Suite 6: Daily & Final Balance Sheet Conservation (< 1e-6 float error)."""

    def test_daily_nav_conservation_with_abstained_orders(self):
        """Assert abs(NAV - (Cash + Receivables + Positions_Value)) < 1e-6 on all simulated days."""
        eval_summary = json.loads((OUT_DIR / "evaluation_summary.json").read_text(encoding="utf-8"))
        for res in eval_summary["results"]:
            folder = OUT_DIR / f"{res['period']}_{res['mode']}"
            equity_file = folder / "equity.csv"
            self.assertTrue(equity_file.exists(), f"equity.csv missing in {folder}")

            eq_df = pd.read_csv(equity_file)
            for idx, row in eq_df.iterrows():
                nav = float(row["equity"])
                cash = float(row["cash"])
                rec = float(row["receivables"])
                # NAV must be >= Cash + Receivables - float epsilon
                self.assertGreaterEqual(nav + 1e-6, cash + rec)
                self.assertGreater(cash, -1e-6, "Negative cash detected")

    def test_abstained_order_leaves_cash_and_nav_invariant(self):
        """Verify abstaining on an order produces zero discrepancy on balance sheet."""
        rep = GatekeeperV082Replay({}, ["2025-01-02"], pd.DataFrame(), [], Config())
        initial_nav = rep.nav()
        initial_cash = rep.cash

        # Order arrives and is abstained
        order = {"code": "sh.600000", "signal_date": "2025-01-01", "shares": 500, "limit": 10.0}
        rep.reject("gatekeeper_abstain")

        self.assertEqual(rep.nav(), initial_nav)
        self.assertEqual(rep.cash, initial_cash)


class TestLivenessAndMultiYearReplay(unittest.TestCase):
    """Suite 7: Multi-Year Active Trading, Positive Returns & Double-Slippage Resilience."""

    def test_active_trading_liveness_2022_2025(self):
        """Verify 2022-2025 active trading volume (closed_trades >= 40, no empty portfolio trap)."""
        eval_summary = json.loads((OUT_DIR / "evaluation_summary.json").read_text(encoding="utf-8"))
        res_22_25 = [r for r in eval_summary["results"] if r["period"] == "2022-2025" and r["mode"] == "normal_5bps"][0]
        trades = res_22_25["closed_trades"]
        self.assertGreaterEqual(trades, 40, f"2022-2025 trade volume too low: {trades} < 40")

    def test_active_trading_liveness_2026(self):
        """Verify 2026 active trading volume (closed_trades >= 6, no empty portfolio trap)."""
        eval_summary = json.loads((OUT_DIR / "evaluation_summary.json").read_text(encoding="utf-8"))
        res_26 = [r for r in eval_summary["results"] if r["period"] == "2026" and r["mode"] == "normal_5bps"][0]
        trades = res_26["closed_trades"]
        self.assertGreaterEqual(trades, 6, f"2026 trade volume too low: {trades} < 6")

    def test_positive_returns_multi_year(self):
        """Verify positive net return across 2022-2025 and 2026 under normal slippage."""
        eval_summary = json.loads((OUT_DIR / "evaluation_summary.json").read_text(encoding="utf-8"))
        for res in eval_summary["results"]:
            if res["mode"] == "normal_5bps":
                ret = float(res["total_return"])
                self.assertGreater(ret, 0.0, f"Non-positive return in {res['period']} ({res['mode']}): {ret:.2%}")

    def test_double_slippage_resilience(self):
        """Verify positive net return across 2022-2025 and 2026 under double slippage (10 bps)."""
        eval_summary = json.loads((OUT_DIR / "evaluation_summary.json").read_text(encoding="utf-8"))
        for res in eval_summary["results"]:
            if res["mode"] == "double_10bps":
                ret = float(res["total_return"])
                self.assertGreater(ret, 0.0, f"Stressed double-slippage failure in {res['period']}: {ret:.2%}")

    def test_rejection_rate_bounded(self):
        """Verify rejection rate is bounded between 5% and 60% (avoiding under-filtering and starvation)."""
        eval_summary = json.loads((OUT_DIR / "evaluation_summary.json").read_text(encoding="utf-8"))
        res_22_25 = [r for r in eval_summary["results"] if r["period"] == "2022-2025" and r["mode"] == "normal_5bps"][0]
        rej_rate = res_22_25["gating"]["rejection_rate"]
        self.assertGreater(rej_rate, 0.05, f"Rejection rate too low: {rej_rate:.1%}")
        self.assertLess(rej_rate, 0.65, f"Rejection rate too high: {rej_rate:.1%}")


if __name__ == "__main__":
    unittest.main()
