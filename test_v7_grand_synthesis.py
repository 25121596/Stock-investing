"""Grand Synthesis for Super-V7 Architecture:
Combining:
1. Extended Dynamic Horizon (10-14 days vs 8-12 days)
2. Capital Efficiency (Budget 6k, 7k, 8k with 4-5 positions)
3. Volatility Dynamic Stops: SL = min(2.5%, 1.2 * sigma_20)
4. Model V2 + V3 Composite Ranking
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

period_days = [d for d in calendar if "2022-01-01" <= d <= "2025-12-31"]

# 1. Model V2 Signal
sig_v2 = pred_v2[pred_v2.pred_net_return_v2 >= 0.005][["date", "code", "pred_net_return_v2"]].rename(columns={"pred_net_return_v2": "score"})
sig_v2 = sig_v2[sig_v2.date.isin(allowed_tiered)]

# 2. Dual Ensemble Signal
merged = pd.merge(pred_v2[["date", "code", "pred_net_return_v2"]], pred_v3[["date", "code", "pred_net_return_v3"]], on=["date", "code"])
merged["rank_v2"] = merged.groupby("date")["pred_net_return_v2"].rank(pct=True)
merged["rank_v3"] = merged.groupby("date")["pred_net_return_v3"].rank(pct=True)
merged["score"] = 0.5 * merged["rank_v2"] + 0.5 * merged["rank_v3"]
mask_ens = (merged.pred_net_return_v2 >= 0.003) & (merged.pred_net_return_v3 >= 0.003)
sig_ens = merged[mask_ens][["date", "code", "score"]]
sig_ens = sig_ens[sig_ens.date.isin(allowed_tiered)]


class SuperV7SynthesisReplay(SuperV5MaxProfitReplay):
    def __init__(self, frames, calendar, signals, dividends, config, vol_map=None, daily_limits=None, breadth_lookup=None, hold_strong=14, hold_neutral=10, sl_mult=1.2, enable_cash_overlay=False):
        super().__init__(frames, calendar, signals, dividends, config, vol_map=vol_map, daily_limits=daily_limits, enable_cash_overlay=enable_cash_overlay)
        self.breadth_lookup = breadth_lookup or {}
        self.hold_strong = hold_strong
        self.hold_neutral = hold_neutral
        self.sl_mult = sl_mult

    def run(self):
        cal_lookup = {d: i for i, d in enumerate(self.calendar)}
        for index, day in enumerate(self.calendar):
            self.today = day

            # 1. Dividends
            for item in self.receivables:
                if item[0] == day:
                    self.cash += item[1]
            self.receivables = [item for item in self.receivables if item[0] != day]

            # 2. Corporate actions
            for item in list(self.entitlements):
                if item["event"].get("record_date") == day:
                    self.entitlements.remove(item)
                    position = item["position"]
                    if position.shares > 0:
                        split_ratio = float(item["event"].get("split_ratio", 0.0) or 0.0)
                        div_ps = float(item["event"].get("cash_dividend", 0.0) or 0.0)
                        if split_ratio > 0:
                            position.shares += int(position.shares * split_ratio)
                        if div_ps > 0:
                            div_amount = position.shares * div_ps
                            position.dividends += div_amount
                            self.receivables.append((item["event"].get("pay_date", day), div_amount))

            # 3. Morning Exits: Check stop pending or horizon
            # AUDITED LOOKAHEAD FIX: T-1 breadth at morning open
            regime_date = self.calendar[index - 1] if index > 0 else self.calendar[index]
            regime_b = self.breadth_lookup.get(regime_date, 0.5)
            max_hold = self.hold_strong if regime_b >= 0.55 else self.hold_neutral

            for code, position in list(self.positions.items()):
                hold_days = index - cal_lookup.get(position.entry_date, index)
                if position.stop_pending or hold_days >= max_hold:
                    self.sell(position, day, index)

            # 4. Marks at open
            for code in self.positions:
                bar = self.bars[code].get(day)
                if bar is not None and float(bar.get("open", 0.0) or 0.0) > 0:
                    self.marks[code] = float(bar["open"])

            # 5. Morning Buys
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

                    eff_sl = max(0.020, min(0.045, self.sl_mult * ret_vol))
                    eff_tp = max(0.090, min(0.160, 4.0 * ret_vol))

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

                    if peak_gain >= 0.100 and (current_peak - close_px) / current_peak >= 0.025:
                        position.stop_pending = True
                    if cur_gain >= eff_tp:
                        position.stop_pending = True

            # 7. Entitlements
            for event in self.actions_by_record.get(day, []):
                position = self.positions.get(event.get("code"))
                if position and position.shares:
                    position.pending_actions += 1
                    self.entitlements.append({"event": event, "position": position, "shares": position.shares, "posted": False})

            # 8. Cash sweep
            if self.enable_cash_overlay and self.cash > 0:
                interest = self.cash * self.daily_cash_yield
                self.cash += interest
                self.accumulated_cash_yield += interest

            # 9. NAV
            nav = self.nav()
            self.equity.append({"date": day, "equity": nav, "cash": self.cash,
                                "receivables": sum(item[1] for item in self.receivables),
                                "positions": len(self.positions)})
            if self.cash < -1e-7 or any(p.shares < 0 for p in self.positions.values()):
                raise AssertionError("Negative cash or shares")

            # 10. Queue next buys
            candidates = sorted(self.candidates.get(day, []), key=lambda row: (-row["score"], row["code"]))
            queued = 0
            daily_max_buy = self.daily_limits.get(day, 1)
            available_slots = max(0, self.config.max_positions - len(self.positions))
            limit_buys = min(daily_max_buy, available_slots)

            for cand in candidates:
                if queued >= limit_buys:
                    break
                code = cand["code"]
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


def run_synthesis():
    grid = [
        # (Name, Signal, Config, hold_strong, hold_neutral, sl_mult, cash_overlay)
        ("1. Baseline Audited V5 (Pure Alpha)", sig_v2, Config(capital=30000.0, order_budget=5000.0, max_positions=4, max_exposure=0.65, hold_days=10, stop_loss=0.03), 12, 8, 1.5, False),
        ("2. Baseline Audited V5 (With 2.2% Cash)", sig_v2, Config(capital=30000.0, order_budget=5000.0, max_positions=4, max_exposure=0.65, hold_days=10, stop_loss=0.03), 12, 8, 1.5, True),

        ("3. V7-Alpha (Budget 6k, Exp 75%, Hold 12/8)", sig_v2, Config(capital=30000.0, order_budget=6000.0, max_positions=4, max_exposure=0.75, hold_days=10, stop_loss=0.03), 12, 8, 1.5, False),
        ("4. V7-Alpha + Extended Hold 14/10", sig_v2, Config(capital=30000.0, order_budget=6000.0, max_positions=4, max_exposure=0.75, hold_days=10, stop_loss=0.03), 14, 10, 1.5, False),
        ("5. V7-Alpha + Tighter SL 1.2*vol", sig_v2, Config(capital=30000.0, order_budget=6000.0, max_positions=4, max_exposure=0.75, hold_days=10, stop_loss=0.03), 14, 10, 1.2, False),

        ("6. V7-Max (Budget 7.5k, Exp 85%, Hold 14/10)", sig_v2, Config(capital=30000.0, order_budget=7500.0, max_positions=4, max_exposure=0.85, hold_days=10, stop_loss=0.03), 14, 10, 1.5, False),
        ("7. V7-Max (Budget 7.5k, Exp 85%, With Cash)", sig_v2, Config(capital=30000.0, order_budget=7500.0, max_positions=4, max_exposure=0.85, hold_days=10, stop_loss=0.03), 14, 10, 1.5, True),

        ("8. V7-Ensemble (Budget 6k, Exp 75%, Hold 14/10)", sig_ens, Config(capital=30000.0, order_budget=6000.0, max_positions=4, max_exposure=0.75, hold_days=10, stop_loss=0.03), 14, 10, 1.5, False),
        ("9. V7-Ensemble (Budget 7.5k, Exp 85%, Hold 14/10)", sig_ens, Config(capital=30000.0, order_budget=7500.0, max_positions=4, max_exposure=0.85, hold_days=10, stop_loss=0.03), 14, 10, 1.5, False),
        ("10. V7-Ensemble (Budget 7.5k, Exp 85%, With Cash)", sig_ens, Config(capital=30000.0, order_budget=7500.0, max_positions=4, max_exposure=0.85, hold_days=10, stop_loss=0.03), 14, 10, 1.5, True),
    ]

    print(f"\n{'Architecture Variant':<48} | {'TotalRet':<9} | {'MaxDD':<7} | {'Calmar':<7} | {'Sharpe':<7} | {'WR':<6} | {'Trades':<6}")
    print("=" * 105)

    res_list = []
    for name, sig, cfg, h_str, h_neu, sl_m, cash in grid:
        rep = SuperV7SynthesisReplay(frames, period_days, sig, dividends, cfg, vol_map=vol_map, daily_limits=tiered_limits, breadth_lookup=b_map, hold_strong=h_str, hold_neutral=h_neu, sl_mult=sl_m, enable_cash_overlay=cash)
        rep.bars = bars
        rep.run()
        s = rep.summary()
        calmar = s["cagr"] / abs(s["max_drawdown"]) if s["max_drawdown"] != 0 else 0
        ret_s = f"{s['total_return']:.2%}"
        dd_s = f"{s['max_drawdown']:.2%}"
        wr_s = f"{s['win_rate']:.1%}"
        print(f"{name:<48} | {ret_s:>9} | {dd_s:>7} | {calmar:>7.2f} | {s['sharpe']:>7.2f} | {wr_s:>6} | {s['closed_trades']:>6}")
        res_list.append({
            "name": name,
            "return": s["total_return"],
            "max_dd": s["max_drawdown"],
            "calmar": calmar,
            "sharpe": s["sharpe"],
            "win_rate": s["win_rate"],
            "trades": s["closed_trades"],
            "profit_factor": s["profit_factor"],
            "cash_overlay": cash,
        })

    pd.DataFrame(res_list).to_csv("results/v7_synthesis_results.csv", index=False)
    print("\nSaved synthesis results to results/v7_synthesis_results.csv")


if __name__ == "__main__":
    run_synthesis()
