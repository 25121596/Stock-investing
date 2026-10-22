"""Breadth-Adaptive Dynamic Sizing (Regime Sizing Engine):
When Market Breadth is strong (breadth >= 0.58):
  - order_budget = 7500
  - max_positions = 5
  - max_exposure = 0.90
When Market Breadth is neutral (0.45 <= breadth < 0.58):
  - order_budget = 5000
  - max_positions = 4
  - max_exposure = 0.70
When Market Breadth is weak (breadth < 0.45):
  - 0 buys (pure defense)
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from experiment_super_v5_maxprofit import SuperV5MaxProfitReplay, DATASET, DONOR
from run import load
from validate_expanded import prepare_bars
from exploration_features import breadth_days
from engine import Config, tick

manifest, frames, calendar, dividends = load(DATASET)
bars = prepare_bars(frames)
panel = pd.read_csv(DONOR / "features.csv", dtype={"date": str, "code": str})
pred_v2 = pd.read_csv("results/model-v2/predictions-h10-v2.csv", dtype={"date": str, "code": str})
pred_v3 = pd.read_csv("results/model-v3/predictions-h10-v3.csv", dtype={"date": str, "code": str})
vol_map = panel.set_index(["date", "code"])["ret_vol20"].to_dict()

breadth_df = breadth_days(panel, return_col="adj_ret5")
b_map = breadth_df.set_index("date")["breadth5"].to_dict()
p_map = breadth_df.set_index("date")["pool_size"].to_dict()

tiered_limits = {d: (0 if p_map.get(d, 0) < 100 or b < 0.45 else (1 if b < 0.55 else 2)) for d, b in b_map.items()}
allowed_tiered = {d for d, lim in tiered_limits.items() if lim > 0}
sig_v5 = pred_v2[pred_v2.pred_net_return_v2 >= 0.005][["date", "code", "pred_net_return_v2"]].rename(columns={"pred_net_return_v2": "score"})
sig_v5 = sig_v5[sig_v5.date.isin(allowed_tiered)]

period_days = [d for d in calendar if "2022-01-01" <= d <= "2025-12-31"]


class DynamicRegimeSizingReplay(SuperV5MaxProfitReplay):
    def __init__(self, frames, calendar, signals, dividends, config, vol_map=None, daily_limits=None, breadth_lookup=None, enable_cash_overlay=False):
        super().__init__(frames, calendar, signals, dividends, config, vol_map=vol_map, daily_limits=daily_limits, enable_cash_overlay=enable_cash_overlay)
        self.breadth_lookup = breadth_lookup or {}

    def run(self):
        for index, day in enumerate(self.calendar):
            self.corporate_actions(day)

            # Morning sells: use T-1 breadth
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

                    if cur_gain >= max(0.030, 1.2 * ret_vol):
                        position.breakeven_active = True
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
                    self.entitlements.append({"event": event, "position": position, "shares": position.shares, "posted": False})

            if self.enable_cash_overlay and self.cash > 0:
                interest = self.cash * self.daily_cash_yield
                self.cash += interest
                self.accumulated_cash_yield += interest

            nav = self.nav()
            self.equity.append({"date": day, "equity": nav, "cash": self.cash,
                                "receivables": sum(item[1] for item in self.receivables),
                                "positions": len(self.positions)})
            if self.cash < -1e-7 or any(p.shares < 0 for p in self.positions.values()):
                raise AssertionError("Negative cash or shares")

            # Dynamic Regime Sizing at evening queueing
            candidates = sorted(self.candidates.get(day, []), key=lambda row: (-row["score"], row["code"]))
            queued = 0
            daily_max_buy = self.daily_limits.get(day, 1)

            b_today = self.breadth_lookup.get(day, 0.5)
            if b_today >= 0.58:
                eff_budget = 7500.0
                eff_max_pos = 5
            elif b_today >= 0.45:
                eff_budget = 5500.0
                eff_max_pos = 4
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
                self.pending_buys.append({"code": code, "signal_date": day, "shares": quantity, "limit": limit})
                queued += 1

        return self


cfg_regime = Config(capital=30000.0, order_budget=7500.0, max_positions=5, max_exposure=0.90, hold_days=10, stop_loss=0.03)

print("=== BREADTH-ADAPTIVE DYNAMIC REGIME SIZING EVALUATION ===")
for cash in [False, True]:
    rep = DynamicRegimeSizingReplay(frames, period_days, sig_v5, dividends, cfg_regime, vol_map=vol_map, daily_limits=tiered_limits, breadth_lookup=b_map, enable_cash_overlay=cash)
    rep.bars = bars
    rep.run()
    s = rep.summary()
    calmar = s["cagr"] / abs(s["max_drawdown"]) if s["max_drawdown"] != 0 else 0
    cash_str = "With 2.2% Cash" if cash else "Pure Trading Alpha"
    print(f"Continuous 2022-2025 ({cash_str}): Return={s['total_return']:>7.2%}, CAGR={s['cagr']:>6.2%}, MaxDD={s['max_drawdown']:>6.2%}, Calmar={calmar:>5.2f}, Sharpe={s['sharpe']:>5.2f}, Trades={s['closed_trades']}")

print("\nYearly Breakdown (With Cash Overlay):")
for yr in [2022, 2023, 2024, 2025, 2026]:
    end_d = f"{yr}-12-31" if yr < 2026 else "2026-10-13"
    p_yr = [d for d in calendar if f"{yr}-01-01" <= d <= end_d]
    sig_yr = sig_v5[sig_v5.date.between(f"{yr}-01-01", end_d)].copy()
    r_yr = DynamicRegimeSizingReplay(frames, p_yr, sig_yr, dividends, cfg_regime, vol_map=vol_map, daily_limits=tiered_limits, breadth_lookup=b_map, enable_cash_overlay=True)
    r_yr.bars = bars
    r_yr.run()
    sm = r_yr.summary()
    calmar = sm["cagr"] / abs(sm["max_drawdown"]) if sm["max_drawdown"] != 0 else 0
    print(f"Year {yr}: Return={sm['total_return']:>7.2%}, MaxDD={sm['max_drawdown']:>6.2%}, Calmar={calmar:>5.2f}, Trades={sm['closed_trades']}")
