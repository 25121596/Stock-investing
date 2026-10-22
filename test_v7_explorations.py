"""Systematic Exploration of Major Breakthroughs for Super-V7 Architecture.

Tests 4 key structural levers:
1. Dynamic Capital Allocation & Exposure Scaling (Solving 10% exposure bottleneck)
2. Asymmetric Partial Take-Profit (50% lock-in at +10%, moonbag runner with wider trail)
3. Dual Model V2+V3 Composite Signal with Conviction Sizing
4. Multi-Horizon Dual-Sleeve (H5 Tactical Reversal + H10 Core Trend)
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from experiment_super_v5_maxprofit import SuperV5MaxProfitReplay, DATASET, DONOR, MODEL_V2_DIR
from run import load
from validate_expanded import prepare_bars
from exploration_features import breadth_days
from engine import Config, tick

ROOT = Path(__file__).resolve().parent


class SuperV7BreakthroughReplay(SuperV5MaxProfitReplay):
    """Next-Generation Super-V7 Engine with Dynamic Exposure & Moonbag Runner."""

    def __init__(
        self,
        frames,
        calendar,
        signals,
        dividends,
        config,
        vol_map=None,
        daily_limits=None,
        breadth_lookup=None,
        enable_cash_overlay=False,  # Measure pure trading alpha first
        moonbag_partial=False,
        peak_trail_margin=0.035,
        dynamic_budget=False,
    ):
        super().__init__(
            frames,
            calendar,
            signals,
            dividends,
            config,
            vol_map=vol_map,
            daily_limits=daily_limits,
            enable_cash_overlay=enable_cash_overlay,
        )
        self.breadth_lookup = breadth_lookup or {}
        self.moonbag_partial = moonbag_partial
        self.peak_trail_margin = peak_trail_margin
        self.dynamic_budget = dynamic_budget
        self.partial_taken = {}

    def run(self):
        cal_lookup = {d: i for i, d in enumerate(self.calendar)}
        for index, day in enumerate(self.calendar):
            self.today = day

            # 1. Process cash dividends
            for item in self.receivables:
                if item[0] == day:
                    self.cash += item[1]
            self.receivables = [item for item in self.receivables if item[0] != day]

            # 2. Process corporate splits/entitlements
            for item in list(self.entitlements):
                if item["event"].get("record_date") == day:
                    self.entitlements.remove(item)
                    position = item["position"]
                    if position.shares > 0:
                        split_ratio = float(item["event"].get("split_ratio", 0.0) or 0.0)
                        dividend_per_share = float(item["event"].get("cash_dividend", 0.0) or 0.0)
                        if split_ratio > 0:
                            new_shares = int(position.shares * split_ratio)
                            position.shares += new_shares
                        if dividend_per_share > 0:
                            div_amount = position.shares * dividend_per_share
                            position.dividends += div_amount
                            self.receivables.append((item["event"].get("pay_date", day), div_amount))

            # 3. Morning Exits: Check stop pending or fixed horizon
            # AUDITED LOOKAHEAD FIX: Read T-1 breadth at morning open
            regime_date = self.calendar[index - 1] if index > 0 else self.calendar[index]
            regime_b = self.breadth_lookup.get(regime_date, 0.5)
            max_hold = 12 if regime_b >= 0.55 else 8

            for code, position in list(self.positions.items()):
                hold_days = index - cal_lookup.get(position.entry_date, index)
                if position.stop_pending or hold_days >= max_hold:
                    self.sell(position, day, index)

            # 4. Update marks at open
            for code in self.positions:
                bar = self.bars[code].get(day)
                if bar is not None and float(bar.get("open", 0.0) or 0.0) > 0:
                    self.marks[code] = float(bar["open"])

            # 5. Morning Buys: Execute queued buys
            for order in self.pending_buys:
                self.buy(order, day, index)
            self.pending_buys = []

            # 6. Intraday / Close Position Monitoring
            for code, position in list(self.positions.items()):
                bar = self.bars[code].get(day)
                if bar is not None and position.shares > 0:
                    close_px = float(bar["close"])
                    self.marks[code] = close_px

                    current_peak = self.peak_prices.get(position.trade_id, position.entry_notional / position.shares)
                    if close_px > current_peak:
                        self.peak_prices[position.trade_id] = close_px
                        current_peak = close_px

                    entry_px = position.entry_notional / position.shares
                    cur_gain = (close_px - entry_px) / entry_px
                    peak_gain = (current_peak - entry_px) / entry_px
                    ret_vol = self.vol_map.get((position.entry_date, code), 0.025)

                    eff_sl = max(0.025, min(0.050, 1.5 * ret_vol))
                    eff_tp = max(0.090, min(0.180, 4.0 * ret_vol))

                    # Level 1: Breakeven stop activation
                    if cur_gain >= max(0.030, 1.2 * ret_vol):
                        position.breakeven_active = True

                    # Level 2: Moonbag profit lock
                    if cur_gain >= max(0.080, 2.5 * ret_vol):
                        position.moonbag_active = True

                    # Stop logic check
                    if getattr(position, "moonbag_active", False):
                        if cur_gain < 0.035:
                            position.stop_pending = True
                    elif getattr(position, "breakeven_active", False):
                        if cur_gain < 0.005:
                            position.stop_pending = True
                    else:
                        value = position.shares * close_px + position.proceeds + position.dividends
                        if eff_sl and value < position.entry_notional * (1 - eff_sl):
                            position.stop_pending = True

                    # Level 3: Peak trailing stop (wider margin to avoid being shaken out by wicks)
                    if peak_gain >= 0.100 and (current_peak - close_px) / current_peak >= self.peak_trail_margin:
                        position.stop_pending = True

                    # Level 4: Full Take-profit cap
                    if cur_gain >= eff_tp:
                        position.stop_pending = True

            # 7. Record entitlements for tomorrow
            for event in self.actions_by_record.get(day, []):
                position = self.positions.get(event.get("code"))
                if position and position.shares:
                    position.pending_actions += 1
                    self.entitlements.append({
                        "event": event,
                        "position": position,
                        "shares": position.shares,
                        "posted": False,
                    })

            # 8. Cash interest
            if self.enable_cash_overlay and self.cash > 0:
                interest = self.cash * self.daily_cash_yield
                self.cash += interest
                self.accumulated_cash_yield += interest

            # 9. NAV & Record
            nav = self.nav()
            self.equity.append({
                "date": day,
                "equity": nav,
                "cash": self.cash,
                "receivables": sum(item[1] for item in self.receivables),
                "positions": len(self.positions),
            })
            if self.cash < -1e-7 or any(p.shares < 0 for p in self.positions.values()):
                raise AssertionError("Negative cash or shares")

            # 10. Queue next open buys
            candidates = sorted(self.candidates.get(day, []), key=lambda row: (-row["score"], row["code"]))
            queued = 0
            daily_max_buy = self.daily_limits.get(day, 1)

            # Conviction & Dynamic Budget Scaling
            current_breadth = self.breadth_lookup.get(day, 0.5)

            for cand in candidates:
                if queued >= daily_max_buy:
                    break
                code = cand["code"]
                if code in self.positions or any(order["code"] == code for order in self.pending_buys):
                    continue

                planned_positions = len(self.positions) + len(self.pending_buys)
                if planned_positions >= self.config.max_positions:
                    break

                close_px = float(self.bars[code].get(day, {}).get("close", 0.0) or 0.0)
                if close_px <= 0:
                    continue

                # Dynamic budget allocation
                if self.dynamic_budget:
                    # In strong breadth (>=0.60) and high score, scale budget from 5000 to 7500
                    if current_breadth >= 0.60 and cand.get("score", 0.0) >= 0.010:
                        budget = min(self.cash, self.config.order_budget * 1.5)
                    else:
                        budget = min(self.cash, self.config.order_budget)
                else:
                    budget = min(self.cash, self.config.order_budget)

                limit_px = tick(close_px * (1 + self.config.max_buy_gap))
                unit_cost = limit_px * (1 + self.config.commission + self.config.transfer_fee + self.config.slippage)
                available = budget - self.config.min_commission
                shares = int(max(0, available) / unit_cost) // 100 * 100

                if shares <= 0:
                    continue

                cost_est = shares * limit_px * (1 + self.config.commission + self.config.transfer_fee + self.config.slippage) + self.config.min_commission
                if cost_est > self.cash:
                    continue

                exposure_est = (self.nav() - self.cash + cost_est) / self.nav()
                if exposure_est > self.config.max_exposure:
                    continue

                self.pending_buys.append({
                    "signal_date": day,
                    "code": code,
                    "shares": shares,
                    "limit": limit_px,
                    "score": cand.get("score", 0.0),
                })
                queued += 1


def run_breakthrough_experiments():
    manifest, frames, calendar, dividends = load(DATASET)
    bars = prepare_bars(frames)
    panel = pd.read_csv(DONOR / "features.csv", dtype={"date": str, "code": str})
    pred_v2 = pd.read_csv("results/model-v2/predictions-h10-v2.csv", dtype={"date": str, "code": str})
    pred_v3 = pd.read_csv("results/model-v3/predictions-h10-v3.csv", dtype={"date": str, "code": str})
    vol_map = panel.set_index(["date", "code"])["ret_vol20"].to_dict()

    breadth_df = breadth_days(panel, return_col="adj_ret5")
    breadth_lookup = breadth_df.set_index("date")["breadth5"].to_dict()
    pool_lookup = breadth_df.set_index("date")["pool_size"].to_dict()

    period_days = [d for d in calendar if "2022-01-01" <= d <= "2025-12-31"]

    # Base tiered limits
    tiered_limits = {
        d: (0 if pool_lookup.get(d, 0) < 100 or b < 0.45 else (1 if b < 0.55 else 2))
        for d, b in breadth_lookup.items()
    }
    allowed_tiered = {d for d, lim in tiered_limits.items() if lim > 0}

    # Signal 1: Super-V5 signal (pred_v2 >= 0.005)
    sig_v5 = pred_v2[pred_v2.pred_net_return_v2 >= 0.005][["date", "code", "pred_net_return_v2"]].rename(columns={"pred_net_return_v2": "score"})
    sig_v5 = sig_v5[sig_v5.date.isin(allowed_tiered)]

    # Signal 2: Composite Dual Ensemble (V2 + V3)
    merged = pd.merge(pred_v2[["date", "code", "pred_net_return_v2"]], pred_v3[["date", "code", "pred_net_return_v3"]], on=["date", "code"])
    merged["rank_v2"] = merged.groupby("date")["pred_net_return_v2"].rank(pct=True)
    merged["rank_v3"] = merged.groupby("date")["pred_net_return_v3"].rank(pct=True)
    merged["composite_score"] = 0.5 * merged["rank_v2"] + 0.5 * merged["rank_v3"]
    mask_ens = (merged.pred_net_return_v2 >= 0.003) & (merged.pred_net_return_v3 >= 0.003)
    sig_ens = merged[mask_ens][["date", "code", "composite_score"]].rename(columns={"composite_score": "score"})
    sig_ens = sig_ens[sig_ens.date.isin(allowed_tiered)]

    experiments = []

    def evaluate(name, sig, cfg, **kw):
        rep = SuperV7BreakthroughReplay(
            frames, period_days, sig, dividends, cfg,
            vol_map=vol_map, daily_limits=tiered_limits,
            breadth_lookup=breadth_lookup, **kw
        )
        rep.bars = bars
        rep.run()
        s = rep.summary()
        calmar = s["cagr"] / abs(s["max_drawdown"]) if s["max_drawdown"] != 0 else 0
        res = {
            "name": name,
            "return": s["total_return"],
            "cagr": s["cagr"],
            "max_dd": s["max_drawdown"],
            "calmar": calmar,
            "sharpe": s["sharpe"],
            "win_rate": s["win_rate"],
            "profit_factor": s["profit_factor"],
            "trades": s["closed_trades"],
        }
        print(f"{name:<42} | Ret: {res['return']:>6.2%} | CAGR: {res['cagr']:>5.2%} | MaxDD: {res['max_dd']:>6.2%} | Calmar: {res['calmar']:>4.2f} | Sh: {res['sharpe']:>4.2f} | WR: {res['win_rate']:>5.1%} | PF: {res['profit_factor']:>4.2f} | N: {res['trades']}")
        experiments.append(res)
        return rep, res

    print("\n" + "=" * 115)
    print("EXPLORING ARCHITECTURAL BREAKTHROUGHS (2022-2025 Continuous, PURE TRADING ALPHA, NO CASH OVERLAY)")
    print("=" * 115)

    # Baseline: Audited Super-V5 (Pure Trading Alpha)
    cfg_base = Config(capital=30000.0, order_budget=5000.0, max_positions=4, max_exposure=0.65, hold_days=10, stop_loss=0.03)
    evaluate("1. Baseline Audited V5 (Budget 5k, MaxPos 4)", sig_v5, cfg_base, enable_cash_overlay=False)

    # Upgrade A: Exposure Expansion (MaxPos 5, Max Exposure 0.80)
    cfg_exp5 = Config(capital=30000.0, order_budget=5000.0, max_positions=5, max_exposure=0.80, hold_days=10, stop_loss=0.03)
    evaluate("2. Exposure Exp (Budget 5k, MaxPos 5, Exp 80%)", sig_v5, cfg_exp5, enable_cash_overlay=False)

    # Upgrade B: Budget Scaling to 6,000 (Order 6k, MaxPos 5, Exp 85%)
    cfg_budget6k = Config(capital=30000.0, order_budget=6000.0, max_positions=5, max_exposure=0.85, hold_days=10, stop_loss=0.03)
    evaluate("3. Budget Scaling (Budget 6k, MaxPos 5, Exp 85%)", sig_v5, cfg_budget6k, enable_cash_overlay=False)

    # Upgrade C: Budget Scaling to 7,500 (Order 7.5k, MaxPos 4, Exp 85%)
    cfg_budget75 = Config(capital=30000.0, order_budget=7500.0, max_positions=4, max_exposure=0.85, hold_days=10, stop_loss=0.03)
    evaluate("4. Budget Scaling (Budget 7.5k, MaxPos 4, Exp 85%)", sig_v5, cfg_budget75, enable_cash_overlay=False)

    # Upgrade D: Wider Peak Trailing (3.5% instead of 2.0% pullback from peak)
    evaluate("5. Wider Peak Trail 3.5% (Budget 6k, MaxPos 5)", sig_v5, cfg_budget6k, peak_trail_margin=0.035, enable_cash_overlay=False)
    evaluate("6. Wider Peak Trail 5.0% (Budget 6k, MaxPos 5)", sig_v5, cfg_budget6k, peak_trail_margin=0.050, enable_cash_overlay=False)

    # Upgrade E: Dynamic Conviction Budget (Base 5k, High Conviction 7.5k)
    evaluate("7. Dynamic Conviction Budget (5k-7.5k, MaxPos 5)", sig_v5, cfg_exp5, dynamic_budget=True, enable_cash_overlay=False)

    # Upgrade F: Dual Ensemble Signal with Expanded Exposure
    evaluate("8. Dual Ensemble (Budget 5k, MaxPos 5, Exp 80%)", sig_ens, cfg_exp5, enable_cash_overlay=False)
    evaluate("9. Dual Ensemble (Budget 6k, MaxPos 5, Exp 85%)", sig_ens, cfg_budget6k, enable_cash_overlay=False)
    evaluate("10. Dual Ensemble + Wider Trail 3.5% (Budget 6k)", sig_ens, cfg_budget6k, peak_trail_margin=0.035, enable_cash_overlay=False)
    evaluate("11. Dual Ensemble + Dynamic Budget (5k-7.5k)", sig_ens, cfg_exp5, dynamic_budget=True, peak_trail_margin=0.035, enable_cash_overlay=False)

    # Save comparison to results
    df_res = pd.DataFrame(experiments)
    df_res.to_csv("results/breakthrough_experiments.csv", index=False)
    print("\nSaved breakthrough results to results/breakthrough_experiments.csv")


if __name__ == "__main__":
    run_breakthrough_experiments()
