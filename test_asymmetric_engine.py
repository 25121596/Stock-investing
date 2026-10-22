"""Test Asymmetric Execution Engine:
- Intraday TP on surges (high_px >= tp_px)
- Close-Confirmed Stop Loss (avoids morning noise washout wicks)
- Cash Overlay (2.2% annual on uninvested cash)
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from architecture_stage1_and_stage2 import IntradayEnterpriseReplay
from engine import Config, tick
from exploration_features import breadth_days
from run import load
from validate_expanded import prepare_bars

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data/forward-2026-10-13"
DONOR = ROOT / "results/midterm-access-v3"
MODEL_V2_DIR = ROOT / "results/model-v2"


class AsymmetricReplay(IntradayEnterpriseReplay):
    def run(self):
        for index, day in enumerate(self.calendar):
            self.corporate_actions(day)

            # Sells from scheduled exits
            for code, position in list(self.positions.items()):
                bar = self.bars[code].get(day)
                limit_for_day = self.daily_limits.get(day, 1)
                max_hold = 12 if limit_for_day >= 2 else 8
                if index - position.entry_index >= max_hold or position.stop_pending:
                    self.sell(position, day, index)

            for code in self.positions:
                bar = self.bars[code].get(day)
                if bar is not None and float(bar.get("open", 0.0) or 0.0) > 0:
                    self.marks[code] = float(bar["open"])

            for order in self.pending_buys:
                self.buy(order, day, index)
            self.pending_buys = []

            for code, position in list(self.positions.items()):
                bar = self.bars[code].get(day)
                if bar is None:
                    continue
                open_px = float(bar.get("open", 0.0) or 0.0)
                high_px = float(bar.get("high", 0.0) or 0.0)
                close_px = float(bar.get("close", 0.0) or 0.0)
                if open_px <= 0 or high_px <= 0 or close_px <= 0:
                    continue
                self.marks[code] = close_px

                current_peak = self.peak_prices.get(position.trade_id, position.entry_notional / position.shares)
                if high_px > current_peak:
                    self.peak_prices[position.trade_id] = high_px
                    current_peak = high_px

                entry_px = position.entry_notional / position.shares
                ret_vol = self.vol_map.get((position.entry_date, code), 0.025)
                eff_sl_pct = max(0.025, min(0.050, 1.5 * ret_vol))
                eff_tp_pct = max(0.090, min(0.150, 4.0 * ret_vol))

                cur_gain = (close_px - entry_px) / entry_px
                peak_gain = (current_peak - entry_px) / entry_px
                tp_px = entry_px * (1 + eff_tp_pct)

                can_sell_today = (index > position.entry_index)

                # Asymmetric Execution: Intraday Take Profit on surges
                if can_sell_today and high_px >= tp_px:
                    fill_px = max(open_px, tp_px) * (1 - self.config.slippage)
                    self._execute_custom_sell(position, day, index, fill_px, "intraday_take_profit")
                    continue

                # Close-Confirmed Stop Escalation (avoids morning noise wicks)
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
                    if cur_gain < -eff_sl_pct:
                        position.stop_pending = True

                if peak_gain >= 0.100 and (current_peak - close_px) / current_peak >= 0.020:
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

            candidates = sorted(self.candidates.get(day, []), key=lambda row: (-row["score"], row["code"]))
            queued = 0
            daily_max_buy = self.daily_limits.get(day, 1)
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
                budget = min(self.config.order_budget, self.cash)
                limit = tick(float(bar["close"]) * (1 + self.config.max_buy_gap))
                unit_cost = limit * (1 + self.config.commission + self.config.transfer_fee + self.config.slippage)
                quantity = int(max(0, budget - self.config.min_commission) / unit_cost) // 100 * 100
                if quantity < 100:
                    self.reject("lot_size_or_budget")
                    continue
                self.pending_buys.append({"code": code, "signal_date": day, "shares": quantity, "limit": limit})
                queued += 1
        return self


def main():
    manifest, frames, calendar, dividends = load(DATASET)
    bars = prepare_bars(frames)

    panel = pd.read_csv(DONOR / "features.csv", dtype={"date": str, "code": str})
    pred_v2 = pd.read_csv(MODEL_V2_DIR / "predictions-h10-v2.csv", dtype={"date": str, "code": str})
    vol_map = panel.set_index(["date", "code"])["ret_vol20"].to_dict()

    breadth_df = breadth_days(panel, return_col="adj_ret5")
    breadth_lookup = breadth_df.set_index("date")["breadth5"].to_dict()
    pool_lookup = breadth_df.set_index("date")["pool_size"].to_dict()

    tiered_limits = {d: (0 if pool_lookup.get(d, 0) < 100 or b < 0.45 else (1 if b < 0.55 else 2)) for d, b in breadth_lookup.items()}
    allowed_tiered = {d for d, lim in tiered_limits.items() if lim > 0}
    sig_v2 = pred_v2[pred_v2.pred_net_return_v2 >= 0.005][["date", "code", "pred_net_return_v2"]].rename(columns={"pred_net_return_v2": "score"})
    sig_v2 = sig_v2[sig_v2.date.isin(allowed_tiered)]

    period_days = [d for d in calendar if "2022-01-01" <= d <= "2025-12-31"]
    cfg = Config(capital=30000.0, order_budget=5000.0, max_positions=4, max_exposure=0.65, hold_days=10, stop_loss=0.03)

    rep = AsymmetricReplay(frames, period_days, sig_v2, dividends, cfg, vol_map=vol_map, daily_limits=tiered_limits, enable_cash_overlay=True)
    rep.bars = bars
    rep.run()
    s = rep.summary()
    calmar = s["cagr"] / abs(s["max_drawdown"])

    print("=" * 80)
    print("ASYMMETRIC INTRADAY ENGINE RESULTS:")
    print(f"Total Return:           {s['total_return']:.2%}")
    print(f"CAGR:                   {s['cagr']:.2%}")
    print(f"Max Drawdown:           {s['max_drawdown']:.2%}")
    print(f"Calmar Ratio:           {calmar:.2f}")
    print(f"Win Rate:               {s['win_rate']:.1%}")
    print(f"Profit Factor:          {s['profit_factor']:.2f}")
    print(f"Trades:                 {s['closed_trades']}")
    print(f"Accumulated Cash Yield: {s['accumulated_cash_yield']:.2f} RMB")
    print("=" * 80)


if __name__ == "__main__":
    main()
