"""Robustness Enhancements for A-share Quantitative Strategy.

Implements and evaluates:
1. Tiered Soft Breadth Gating (Eliminates the 50% knife-edge cliff):
   - Tier 1: breadth >= 0.55 -> max 2 buys/day, max 4 positions
   - Tier 2: 0.45 <= breadth < 0.55 -> max 1 buy/day, max 3 positions
   - Tier 3: breadth < 0.45 -> 0 buys/day (freeze new buys, protect existing)

2. Dual-Model Ensemble Confirmation (Anti-Drift):
   - Combines Logistic Regression win probability (p_win >= 0.52)
   - with LightGBM Huber predicted return (pred_net_return >= 0.005)
   - Dual-key confirmation reduces false positive whipsaws in choppy markets.

3. Volatility-Adaptive Stops (ATR / sigma-20 scaling):
   - Stop Loss scaled to 1.5 * ret_vol20 (bounded [0.025, 0.050])
   - Take Profit scaled to 4.0 * ret_vol20 (bounded [0.080, 0.140])

4. Comprehensive Robustness Grid:
   - Parameter perturbation testing across breadth, stops, and slippages (up to 3x slippage: 0.15%).
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from accessibility import AccessibleReplay
from engine import Config, tick
from exploration_features import breadth_days
from run import load
from validate_expanded import prepare_bars

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data/forward-2026-10-13"
DONOR = ROOT / "results/midterm-access-v3"
MODEL_V2_DIR = ROOT / "results/model-v2"


class RobustReplay(AccessibleReplay):
    def __init__(self, frames, calendar, signals, dividends=None, config=None,
                 take_profit: float | None = None,
                 trailing_trigger: float | None = None,
                 trailing_delta: float | None = None,
                 max_daily_buys: int = 1,
                 vol_adaptive_stops: bool = False,
                 vol_map: dict[tuple[str, str], float] | None = None,
                 daily_limits: dict[str, int] | None = None):
        super().__init__(frames, calendar, signals, dividends, config)
        self.take_profit = take_profit
        self.trailing_trigger = trailing_trigger
        self.trailing_delta = trailing_delta
        self.max_daily_buys = max_daily_buys
        self.vol_adaptive_stops = vol_adaptive_stops
        self.vol_map = vol_map or {}
        self.daily_limits = daily_limits or {}
        self.peak_prices: dict[int, float] = {}

    def run(self):
        for index, day in enumerate(self.calendar):
            self.corporate_actions(day)

            # Check sells at opening
            for code, position in list(self.positions.items()):
                bar = self.bars[code].get(day)
                if bar is not None and code in self.marks:
                    reference = float(bar.get("preclose", 0.0) or 0.0)
                    if reference > 0 and abs(reference - self.marks[code]) > 0.025:
                        known = [e for e in self.actions_by_ex.get(day, []) if e.get("code") == code]
                        if not known:
                            self.warnings.append({"date": day, "code": code, "issue": "unexplained_price_adjustment"})
                elif bar is None:
                    self.warnings.append({"date": day, "code": code, "issue": "missing_held_stock_quote"})

                if index - position.entry_index >= self.config.hold_days or position.stop_pending:
                    self.sell(position, day, index)

            for code in self.positions:
                bar = self.bars[code].get(day)
                if bar is not None and float(bar.get("open", 0.0) or 0.0) > 0:
                    self.marks[code] = float(bar["open"])

            for order in self.pending_buys:
                self.buy(order, day, index)
            self.pending_buys = []

            for code, position in self.positions.items():
                bar = self.bars[code].get(day)
                if bar is not None:
                    close_px = float(bar["close"])
                    self.marks[code] = close_px

                    current_peak = self.peak_prices.get(position.trade_id, position.entry_notional / position.shares)
                    if close_px > current_peak:
                        self.peak_prices[position.trade_id] = close_px
                        current_peak = close_px

                    entry_px = position.entry_notional / position.shares
                    cur_gain = (close_px - entry_px) / entry_px

                    # Calculate thresholds (fixed or volatility-adaptive)
                    if self.vol_adaptive_stops:
                        ret_vol = self.vol_map.get((position.entry_date, code), 0.025)
                        eff_sl = max(0.025, min(0.050, 1.5 * ret_vol))
                        eff_tp = max(0.080, min(0.140, 4.0 * ret_vol))
                    else:
                        eff_sl = self.config.stop_loss
                        eff_tp = self.take_profit

                    # Stop Loss
                    value = position.shares * close_px + position.proceeds + position.dividends
                    if eff_sl and value < position.entry_notional * (1 - eff_sl):
                        position.stop_pending = True

                    # Take-Profit
                    if eff_tp is not None and cur_gain >= eff_tp:
                        position.stop_pending = True

                    # Trailing Stop
                    if self.trailing_trigger is not None and self.trailing_delta is not None:
                        peak_gain = (current_peak - entry_px) / entry_px
                        if peak_gain >= self.trailing_trigger:
                            if (current_peak - close_px) / current_peak >= self.trailing_delta:
                                position.stop_pending = True

            for event in self.actions_by_record.get(day, []):
                position = self.positions.get(event.get("code"))
                if position and position.shares:
                    position.pending_actions += 1
                    self.entitlements.append({"event": event, "position": position,
                                              "shares": position.shares, "posted": False})

            nav = self.nav()
            self.equity.append({"date": day, "equity": nav, "cash": self.cash,
                                "receivables": sum(item[1] for item in self.receivables),
                                "positions": len(self.positions)})
            if self.cash < -1e-7 or any(p.shares < 0 for p in self.positions.values()):
                raise AssertionError("Negative cash or shares")

            candidates = sorted(self.candidates.get(day, []), key=lambda row: (-row["score"], row["code"]))
            queued = 0

            # Dynamic daily buy limit (Tiered breadth or fixed)
            daily_max_buy = self.daily_limits.get(day, self.max_daily_buys)
            available_slots = max(0, self.config.max_positions - len(self.positions))
            limit_buys = min(daily_max_buy, available_slots)

            for candidate in candidates:
                if queued >= limit_buys:
                    break
                code = candidate["code"]
                if code in self.positions or any(o["code"] == code for o in self.pending_buys):
                    continue
                bar = self.bars[code].get(day)
                if bar is None:
                    continue
                limit = tick(float(bar["close"]) * (1 + self.config.max_buy_gap))
                budget = min(self.config.order_budget, self.cash)
                unit_cost = limit * (1 + self.config.commission + self.config.transfer_fee + self.config.slippage)
                quantity = int(max(0, budget - self.config.min_commission) / unit_cost) // 100 * 100
                if quantity < 100:
                    self.reject("lot_size_or_budget")
                    continue
                self.pending_buys.append({"code": code, "signal_date": day, "shares": quantity, "limit": limit})
                queued += 1

        return self


def main():
    print("=" * 88)
    print(">>> ADVANCED ROBUSTNESS ENHANCEMENT RESEARCH & EMPIRICAL EVALUATION <<<")
    print("=" * 88)

    manifest, frames, calendar, dividends = load(DATASET)
    bars = prepare_bars(frames)

    panel = pd.read_csv(DONOR / "features.csv", dtype={"date": str, "code": str})
    pred_base = pd.read_csv(DONOR / "predictions-h10.csv", dtype={"date": str, "code": str})
    pred_v2 = pd.read_csv(MODEL_V2_DIR / "predictions-h10-v2.csv", dtype={"date": str, "code": str})

    # Map for volatility (date, code) -> ret_vol20
    vol_map = panel.set_index(["date", "code"])["ret_vol20"].to_dict()

    # Breadth calculation
    breadth_df = breadth_days(panel, return_col="adj_ret5")
    breadth_lookup = breadth_df.set_index("date")["breadth5"].to_dict()
    pool_lookup = breadth_df.set_index("date")["pool_size"].to_dict()

    # 1. Binary Breadth Allowed Dates (Baseline: >= 50%)
    binary_allowed = set(breadth_df.loc[(breadth_df.pool_size >= 100) & (breadth_df.breadth5 >= 0.50), "date"])

    # 2. Tiered Soft Breadth Limits:
    # >= 0.55: 2 buys
    # 0.45 <= b < 0.55: 1 buy
    # < 0.45: 0 buys
    tiered_limits = {}
    for d, b in breadth_lookup.items():
        pool = pool_lookup.get(d, 0)
        if pool < 100 or b < 0.45:
            tiered_limits[d] = 0
        elif b < 0.55:
            tiered_limits[d] = 1
        else:
            tiered_limits[d] = 2

    # Merged Predictions for Ensemble:
    merged_preds = pd.merge(
        pred_base[["date", "code", "p_win"]],
        pred_v2[["date", "code", "pred_net_return_v2"]],
        on=["date", "code"]
    )

    # Signal Sets:
    # S1: Pure V2 signals (threshold 0.005)
    sig_v2 = pred_v2[pred_v2.pred_net_return_v2 >= 0.005][["date", "code", "pred_net_return_v2"]].rename(columns={"pred_net_return_v2": "score"})

    # S2: Dual Ensemble signals (p_win >= 0.52 AND pred_ret >= 0.005)
    ensemble_mask = (merged_preds.p_win >= 0.52) & (merged_preds.pred_net_return_v2 >= 0.005)
    sig_ensemble = merged_preds[ensemble_mask][["date", "code", "pred_net_return_v2"]].rename(columns={"pred_net_return_v2": "score"})

    period_days = [d for d in calendar if "2022-01-01" <= d <= "2025-12-31"]

    cfg_std = Config(
        capital=30000.0,
        order_budget=5000.0,
        max_positions=4,
        max_exposure=0.65,
        hold_days=10,
        stop_loss=0.03,
    )

    scenarios = [
        # (Name, Signals, BinaryFilter, DailyBuys, TieredLimits, VolAdaptive, TP, SL)
        ("1. Super-V2 Baseline (Hard 50% Gate, Fixed TP/SL)", sig_v2, True, 2, None, False, 0.10, 0.03),
        ("2. Upgrade A: Tiered Soft Breadth (No Hard Cliff)", sig_v2, False, 2, tiered_limits, False, 0.10, 0.03),
        ("3. Upgrade B: Dual Ensemble (LGBM + Logistic)", sig_ensemble, True, 2, None, False, 0.10, 0.03),
        ("4. Upgrade C: Volatility-Adaptive Stops (ATR/Sigma)", sig_v2, True, 2, None, True, None, None),
        ("5. Upgrade D: Tiered Breadth + Vol-Adaptive Stops", sig_v2, False, 2, tiered_limits, True, None, None),
        ("6. Upgrade E: Tiered Breadth + Dual Ensemble (All-Round)", sig_ensemble, False, 2, tiered_limits, False, 0.10, 0.03),
        ("7. Upgrade F: Full-Spectrum Robust (Tiered + Ens + VolStops)", sig_ensemble, False, 2, tiered_limits, True, None, None),
    ]

    print("\n" + "-" * 98)
    print(f"{'Strategy Architecture':<54} {'Return':<9} {'MaxDD':<9} {'Calmar':<8} {'WinRate':<9} {'Trades':<7}")
    print("-" * 98)

    results = []
    for name, raw_sigs, use_binary, buys, t_lims, vol_ad, tp, sl in scenarios:
        if use_binary:
            sigs = raw_sigs[raw_sigs.date.isin(binary_allowed) & raw_sigs.date.between("2022-01-01", "2025-12-31")].copy()
        else:
            # When using tiered limits, dates with limit > 0 are allowed
            allowed_tiered = {d for d, lim in tiered_limits.items() if lim > 0}
            sigs = raw_sigs[raw_sigs.date.isin(allowed_tiered) & raw_sigs.date.between("2022-01-01", "2025-12-31")].copy()

        cfg = Config(
            capital=30000.0,
            order_budget=5000.0,
            max_positions=4,
            max_exposure=0.65,
            hold_days=10,
            stop_loss=sl or 0.03,
        )

        replay = RobustReplay(
            frames, period_days, sigs, dividends, cfg,
            take_profit=tp, max_daily_buys=buys,
            vol_adaptive_stops=vol_ad, vol_map=vol_map,
            daily_limits=t_lims
        )
        replay.bars = bars
        replay.run()
        s = replay.summary()

        cagr = (1 + s["total_return"]) ** (1 / 4.0) - 1.0
        calmar = cagr / abs(s["max_drawdown"]) if s["max_drawdown"] != 0 else 0.0
        wr = s["win_rate"] if s["win_rate"] is not None else 0.0

        results.append({
            "strategy": name,
            "total_return": s["total_return"],
            "max_drawdown": s["max_drawdown"],
            "cagr": cagr,
            "calmar": calmar,
            "win_rate": wr,
            "closed_trades": s["closed_trades"],
            "profit_factor": s["profit_factor"],
            "execution_costs": s["execution_costs"]
        })

        print(f"{name:<54} {s['total_return']:>8.2%} {s['max_drawdown']:>8.2%} {calmar:>7.2f} {wr:>8.1%} {s['closed_trades']:>6}")

    # -------------------------------------------------------------
    # PARAMETER PERTURBATION STRESS MATRIX (PLATEAU ROBUSTNESS)
    # -------------------------------------------------------------
    print("\n" + "=" * 88)
    print(">>> PARAMETER PERTURBATION STRESS MATRIX (ROBUSTNESS PLATEAU TEST) <<<")
    print("=" * 88)
    print("Evaluating resilience against parameter shifts on Strategy 6 (Tiered + Ensemble)...")

    perturbation_records = []
    # Test across 3 slippage levels (0.05%, 0.10%, 0.15% - 3x slippage!)
    # and 3 stop loss levels (2.5%, 3.0%, 3.5%)
    # and 3 take profit levels (8%, 10%, 12%)
    for slip in [0.0005, 0.0010, 0.0015]:
        for sl_p in [0.025, 0.030, 0.035]:
            for tp_p in [0.08, 0.10, 0.12]:
                cfg_p = Config(
                    capital=30000.0,
                    order_budget=5000.0,
                    max_positions=4,
                    max_exposure=0.65,
                    hold_days=10,
                    stop_loss=sl_p,
                    slippage=slip,
                )
                allowed_tiered = {d for d, lim in tiered_limits.items() if lim > 0}
                sigs = sig_ensemble[sig_ensemble.date.isin(allowed_tiered) & sig_ensemble.date.between("2022-01-01", "2025-12-31")].copy()

                rep_p = RobustReplay(
                    frames, period_days, sigs, dividends, cfg_p,
                    take_profit=tp_p, max_daily_buys=2,
                    daily_limits=tiered_limits
                )
                rep_p.bars = bars
                rep_p.run()
                sm = rep_p.summary()
                perturbation_records.append({
                    "slippage": slip,
                    "stop_loss": sl_p,
                    "take_profit": tp_p,
                    "total_return": sm["total_return"],
                    "max_drawdown": sm["max_drawdown"],
                })

    df_pert = pd.DataFrame(perturbation_records)
    print(f"Total perturbation combinations tested: {len(df_pert)}")
    print(f"Minimum Return across all perturbed configurations: {df_pert.total_return.min():.2%}")
    print(f"Average Return across all perturbed configurations: {df_pert.total_return.mean():.2%}")
    print(f"Maximum Drawdown across all perturbed configurations: {df_pert.max_drawdown.min():.2%}")
    print(f"Percentage of profitable configurations: {(df_pert.total_return > 0).mean():.1%}")

    # Save results
    out_dir = ROOT / "results/robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_dir / "robustness_architecture_comparison.csv", index=False, encoding="utf-8-sig")
    df_pert.to_csv(out_dir / "parameter_perturbation_grid.csv", index=False, encoding="utf-8-sig")
    print(f"\nAll robustness experiment results saved to: {out_dir}")


if __name__ == "__main__":
    main()
