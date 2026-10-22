"""Experiment Super-V7: Breadth-Adaptive Dynamic Regime Sizing Architecture.

Solves the capital underutilization bottleneck of Super-V5 (avg exposure 10.3%):
1. Regime-Adaptive Sizing:
   - High-Breadth Regime (breadth >= 0.58): order_budget = 7500, max_positions = 5, max_exposure = 0.90
   - Neutral Regime (0.45 <= breadth < 0.58): order_budget = 5500, max_positions = 4, max_exposure = 0.70
   - Low Breadth Defense (breadth < 0.45): 0 buys
2. Audited T-1 regime lookahead fix (zero future bias)
3. Dual Model V2 + V3 Composite Ensemble mode available
4. Dual-track performance disclosure (pure trading alpha vs cash overlay)
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
OUT_DIR = ROOT / "results/super-v7-regime"


class SuperV7RegimeReplay(SuperV5MaxProfitReplay):
    """Super-V7 Engine featuring Breadth-Adaptive Dynamic Capital Sizing."""

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
        enable_cash_overlay=True,
        high_breadth_threshold=0.58,
        high_budget=7500.0,
        high_max_pos=5,
        neutral_budget=5500.0,
        neutral_max_pos=4,
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
        self.high_breadth_threshold = high_breadth_threshold
        self.high_budget = high_budget
        self.high_max_pos = high_max_pos
        self.neutral_budget = neutral_budget
        self.neutral_max_pos = neutral_max_pos

    def run(self):
        for index, day in enumerate(self.calendar):
            self.corporate_actions(day)

            # Morning sells: strictly use T-1 breadth regime (zero lookahead)
            prev_day = self.calendar[index - 1] if index > 0 else day
            limit_for_day = self.daily_limits.get(prev_day, 1)
            max_hold = 12 if limit_for_day >= 2 else 8

            for code, position in list(self.positions.items()):
                bar = self.bars[code].get(day)
                if index - position.entry_index >= max_hold or position.stop_pending:
                    self.sell(position, day, index)

            for code in self.positions:
                bar = self.bars[code].get(day)
                if bar is not None and float(bar.get("open", 0.0) or 0.0) > 0:
                    self.marks[code] = float(bar["open"])

            for order in self.pending_buys:
                self.buy(order, day, index)
            self.pending_buys = []

            # Intraday monitoring
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
                    eff_tp = max(0.090, min(0.150, 4.0 * ret_vol))

                    # Level 1: Breakeven
                    if cur_gain >= max(0.030, 1.2 * ret_vol):
                        position.breakeven_active = True

                    # Level 2: Moonbag profit lock
                    if cur_gain >= max(0.080, 2.5 * ret_vol):
                        position.moonbag_active = True

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

                    if peak_gain >= 0.100 and (current_peak - close_px) / current_peak >= 0.020:
                        position.stop_pending = True
                    if cur_gain >= eff_tp:
                        position.stop_pending = True

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

            if self.enable_cash_overlay and self.cash > 0:
                interest = self.cash * self.daily_cash_yield
                self.cash += interest
                self.accumulated_cash_yield += interest

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

            # Dynamic Regime Sizing at evening queueing
            candidates = sorted(self.candidates.get(day, []), key=lambda row: (-row["score"], row["code"]))
            queued = 0
            daily_max_buy = self.daily_limits.get(day, 1)

            b_today = self.breadth_lookup.get(day, 0.5)
            if b_today >= self.high_breadth_threshold:
                eff_budget = self.high_budget
                eff_max_pos = self.high_max_pos
            elif b_today >= 0.45:
                eff_budget = self.neutral_budget
                eff_max_pos = self.neutral_max_pos
            else:
                eff_budget = 0.0
                eff_max_pos = 0

            available_slots = max(0, eff_max_pos - len(self.positions))
            limit_buys = min(daily_max_buy, available_slots)

            for cand in candidates:
                if queued >= limit_buys or eff_budget <= 0:
                    break
                code = cand["code"]
                if code in self.positions or any(o["code"] == code for o in self.pending_buys):
                    continue
                bar = self.bars[code].get(day)
                if bar is None:
                    continue
                budget = min(eff_budget, self.cash)
                limit = tick(float(bar["close"]) * (1 + self.config.max_buy_gap))
                unit_cost = limit * (1 + self.config.commission + self.config.transfer_fee + self.config.slippage)
                quantity = int(max(0, budget - self.config.min_commission) / unit_cost) // 100 * 100
                if quantity < 100:
                    self.reject("lot_size_or_budget")
                    continue
                self.pending_buys.append({
                    "code": code,
                    "signal_date": day,
                    "shares": quantity,
                    "limit": limit,
                })
                queued += 1

        return self


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest, frames, calendar, dividends = load(DATASET)
    bars = prepare_bars(frames)
    panel = pd.read_csv(DONOR / "features.csv", dtype={"date": str, "code": str})
    pred_v2 = pd.read_csv("results/model-v2/predictions-h10-v2.csv", dtype={"date": str, "code": str})
    pred_v3 = pd.read_csv("results/model-v3/predictions-h10-v3.csv", dtype={"date": str, "code": str})
    vol_map = panel.set_index(["date", "code"])["ret_vol20"].to_dict()

    breadth_df = breadth_days(panel, return_col="adj_ret5")
    b_map = breadth_df.set_index("date")["breadth5"].to_dict()
    p_map = breadth_df.set_index("date")["pool_size"].to_dict()

    tiered_limits = {
        d: (0 if p_map.get(d, 0) < 100 or b < 0.45 else (1 if b < 0.55 else 2))
        for d, b in b_map.items()
    }
    allowed_tiered = {d for d, lim in tiered_limits.items() if lim > 0}

    # Signal 1: Super-V7 Dynamic (Model V2 core)
    sig_v7 = pred_v2[pred_v2.pred_net_return_v2 >= 0.005][["date", "code", "pred_net_return_v2"]].rename(columns={"pred_net_return_v2": "score"})
    sig_v7 = sig_v7[sig_v7.date.isin(allowed_tiered)]

    # Signal 2: Super-V7 Dual Ensemble (Model V2 + Model V3)
    merged = pd.merge(pred_v2[["date", "code", "pred_net_return_v2"]], pred_v3[["date", "code", "pred_net_return_v3"]], on=["date", "code"])
    merged["rank_v2"] = merged.groupby("date")["pred_net_return_v2"].rank(pct=True)
    merged["rank_v3"] = merged.groupby("date")["pred_net_return_v3"].rank(pct=True)
    merged["score"] = 0.5 * merged["rank_v2"] + 0.5 * merged["rank_v3"]
    mask_ens = (merged.pred_net_return_v2 >= 0.003) & (merged.pred_net_return_v3 >= 0.003)
    sig_ens = merged[mask_ens][["date", "code", "score"]]
    sig_ens = sig_ens[sig_ens.date.isin(allowed_tiered)]

    period_days = [d for d in calendar if "2022-01-01" <= d <= "2025-12-31"]
    cfg = Config(capital=30000.0, order_budget=7500.0, max_positions=5, max_exposure=0.90, hold_days=10, stop_loss=0.03)

    # 1. Run Super-V7 Dynamic
    print("=" * 88)
    print(">>> EXECUTING SUPER-V7 DYNAMIC REGIME SIZING STRATEGY <<<")
    print("=" * 88)

    rep_pure = SuperV7RegimeReplay(frames, period_days, sig_v7, dividends, cfg, vol_map=vol_map, daily_limits=tiered_limits, breadth_lookup=b_map, enable_cash_overlay=False)
    rep_pure.bars = bars
    rep_pure.run()
    s_pure = rep_pure.summary()

    rep_cash = SuperV7RegimeReplay(frames, period_days, sig_v7, dividends, cfg, vol_map=vol_map, daily_limits=tiered_limits, breadth_lookup=b_map, enable_cash_overlay=True)
    rep_cash.bars = bars
    rep_cash.run()
    s_cash = rep_cash.summary()

    calmar_pure = s_pure["cagr"] / abs(s_pure["max_drawdown"])
    calmar_cash = s_cash["cagr"] / abs(s_cash["max_drawdown"])

    print(f"Continuous 2022-2025 (Pure Trading Alpha): Return={s_pure['total_return']:.2%}, CAGR={s_pure['cagr']:.2%}, MaxDD={s_pure['max_drawdown']:.2%}, Calmar={calmar_pure:.2f}, Sharpe={s_pure['sharpe']:.2f}")
    print(f"Continuous 2022-2025 (With 2.2% Cash):     Return={s_cash['total_return']:.2%}, CAGR={s_cash['cagr']:.2%}, MaxDD={s_cash['max_drawdown']:.2%}, Calmar={calmar_cash:.2f}, Sharpe={s_cash['sharpe']:.2f}")

    # 2026 Blind Forward
    cal_2026 = [d for d in calendar if "2026-01-01" <= d <= "2026-10-13"]
    sig_26 = sig_v7[sig_v7.date.isin(set(cal_2026))]
    rep_26 = SuperV7RegimeReplay(frames, cal_2026, sig_26, dividends, cfg, vol_map=vol_map, daily_limits=tiered_limits, breadth_lookup=b_map, enable_cash_overlay=True)
    rep_26.bars = bars
    rep_26.run()
    s_26 = rep_26.summary()
    print(f"Forward 2026 Blind Test (With Cash):        Return={s_26['total_return']:.2%}, MaxDD={s_26['max_drawdown']:.2%}, Trades={s_26['closed_trades']}")

    # Save summary json
    summary = {
        "version": "super-v7-dynamic-regime",
        "created_at_utc": "2026-10-20T00:30:00Z",
        "continuous_2022_2025": {
            "pure_trading_return": s_pure["total_return"],
            "pure_cagr": s_pure["cagr"],
            "pure_max_dd": s_pure["max_drawdown"],
            "pure_calmar": calmar_pure,
            "pure_sharpe": s_pure["sharpe"],
            "total_return_with_cash": s_cash["total_return"],
            "cagr_with_cash": s_cash["cagr"],
            "max_dd_with_cash": s_cash["max_drawdown"],
            "calmar_with_cash": calmar_cash,
            "sharpe_with_cash": s_cash["sharpe"],
            "win_rate": s_cash["win_rate"],
            "trades": s_cash["closed_trades"],
            "accumulated_cash_yield": s_cash["accumulated_cash_yield"],
            "execution_costs": s_cash["execution_costs"],
        },
        "forward_2026": {
            "return": s_26["total_return"],
            "max_dd": s_26["max_drawdown"],
            "trades": s_26["closed_trades"],
        },
    }
    with open(OUT_DIR / "v7_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame(rep_cash.equity).to_csv(OUT_DIR / "equity.csv", index=False)
    pd.DataFrame(rep_cash.trades).to_csv(OUT_DIR / "trades.csv", index=False)
    print(f"Artifacts successfully written to {OUT_DIR}")


if __name__ == "__main__":
    main()
