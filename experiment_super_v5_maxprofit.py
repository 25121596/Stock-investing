"""Super-V5 MaxProfit: Super-V4 AlphaMax + Institutional Cash Sweep Overlay.

Combines the highest-alpha trading engine (Super-V4 AlphaMax) with
institutional idle cash yield harvesting (2.20% annual cash sweep):
- Super-V4 AlphaMax:
  * Tiered Soft Breadth (>=55% 2 buys/d, 45-55% 1 buy/d, <45% 0 buy)
  * Volatility-Adaptive Base Stops: SL = clip(1.5 * sigma_20, 2.5%, 5.0%), TP = clip(4.0 * sigma_20, 9.0%, 15.0%)
  * Two-Tier Profit Escalator: Level 1 Breakeven (+0.5% stop at >=+3% gain), Level 2 Moonbag Lock (+3.5% stop at >=+8% gain)
  * Parabolic Peak Trailing (2.0% delta above +10.0%)
  * Regime Horizon: 12 days in strong bull, 8 days in moderate
- Cash Sweep Overlay:
  * 2.20% annual institutional cash yield automatically accrued on uninvested cash.
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


class SuperV5MaxProfitReplay(AccessibleReplay):
    def __init__(self, frames, calendar, signals, dividends=None, config=None,
                 vol_map: dict[tuple[str, str], float] | None = None,
                 daily_limits: dict[str, int] | None = None,
                 enable_cash_overlay: bool = True,
                 cash_annual_rate: float = 0.022):
        super().__init__(frames, calendar, signals, dividends, config)
        self.vol_map = vol_map or {}
        self.daily_limits = daily_limits or {}
        self.enable_cash_overlay = enable_cash_overlay
        self.daily_cash_yield = cash_annual_rate / 242.0 if enable_cash_overlay else 0.0
        self.accumulated_cash_yield = 0.0
        self.peak_prices: dict[int, float] = {}

    def run(self):
        for index, day in enumerate(self.calendar):
            self.corporate_actions(day)

            # Check sells at opening
            for code, position in list(self.positions.items()):
                bar = self.bars[code].get(day)
                # Fix look-ahead: at morning open of day t, use breadth regime from previous session t-1
                prev_day = self.calendar[index - 1] if index > 0 else day
                limit_for_day = self.daily_limits.get(prev_day, 1)
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
                    peak_gain = (current_peak - entry_px) / entry_px
                    ret_vol = self.vol_map.get((position.entry_date, code), 0.025)

                    eff_sl = max(0.025, min(0.050, 1.5 * ret_vol))
                    eff_tp = max(0.090, min(0.150, 4.0 * ret_vol))

                    # Level 1: Breakeven at +3.0%
                    if cur_gain >= max(0.030, 1.2 * ret_vol):
                        position.breakeven_active = True

                    # Level 2: Moonbag profit lock at +8.0%
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

                    # Level 3: Peak trailing when peak gain >= 10.0%
                    if peak_gain >= 0.100 and (current_peak - close_px) / current_peak >= 0.020:
                        position.stop_pending = True

                    # Cap take profit
                    if cur_gain >= eff_tp:
                        position.stop_pending = True

            for event in self.actions_by_record.get(day, []):
                position = self.positions.get(event.get("code"))
                if position and position.shares:
                    position.pending_actions += 1
                    self.entitlements.append({"event": event, "position": position,
                                              "shares": position.shares, "posted": False})

            # Cash Management Sweep (Earn daily interest on uninvested cash)
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

    def summary(self):
        equity = pd.DataFrame(self.equity)
        curve = pd.concat([pd.Series([self.config.capital]), equity.equity], ignore_index=True)
        pnl = [trade["net_pnl"] for trade in self.trades]
        wins = sum(value > 0 for value in pnl)
        positive = sum(value for value in pnl if value > 0)
        negative = -sum(value for value in pnl if value < 0)
        final = float(curve.iloc[-1])
        open_pnl = sum(item["marked_pnl"] for item in self.holdings())
        reconciliation = final - self.config.capital - sum(pnl) - open_pnl - self.accumulated_cash_yield
        if abs(reconciliation) > 1e-5:
            raise AssertionError(f"NAV does not reconcile to closed/open P&L: {reconciliation}")

        eq_series = equity["equity"]
        daily_ret = eq_series.pct_change().dropna()
        excess_ret = daily_ret - (0.02 / 242.0)
        sharpe = float(np.sqrt(242.0) * excess_ret.mean() / (daily_ret.std() + 1e-8))
        downside = daily_ret[daily_ret < 0]
        sortino = float(np.sqrt(242.0) * excess_ret.mean() / (downside.std() + 1e-8)) if len(downside) > 0 else 0.0

        cagr = (final / self.config.capital) ** (1.0 / 4.0) - 1.0
        max_dd = float((curve / curve.cummax() - 1).min())
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

        return {
            "initial_capital": self.config.capital,
            "final_equity": final,
            "total_return": final / self.config.capital - 1,
            "cagr": cagr,
            "max_drawdown": max_dd,
            "calmar": calmar,
            "sharpe": sharpe,
            "sortino": sortino,
            "closed_trades": len(pnl),
            "win_rate": wins / len(pnl) if pnl else None,
            "profit_factor": positive / negative if negative else None,
            "accumulated_cash_yield": self.accumulated_cash_yield,
            "execution_costs": sum(sum(fill[key] for key in ("commission", "transfer", "stamp", "slippage")) for fill in self.fills),
        }


def main():
    print("=" * 88)
    print(">>> SUPER-V5 MAXPROFIT (V4-ALPHAMAX + CASH SWEEP OVERLAY) EVALUATION <<<")
    print("=" * 88)

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

    cfg = Config(
        capital=30000.0,
        order_budget=5000.0,
        max_positions=4,
        max_exposure=0.65,
        hold_days=10,
        stop_loss=0.03,
    )

    print("\n[Backtest 1] Running Super-V5 MaxProfit (2022-2025 Continuous)...")
    rep = SuperV5MaxProfitReplay(
        frames, period_days, sig_v2, dividends, cfg,
        vol_map=vol_map, daily_limits=tiered_limits,
        enable_cash_overlay=True, cash_annual_rate=0.022
    )
    rep.bars = bars
    rep.run()
    s = rep.summary()

    print(f"  V5 MaxProfit Total Return: {s['total_return']:.2%}")
    print(f"  V5 MaxProfit CAGR:         {s['cagr']:.2%}")
    print(f"  V5 MaxProfit Max Drawdown: {s['max_drawdown']:.2%}")
    print(f"  V5 MaxProfit Calmar:       {s['calmar']:.2f}")
    print(f"  V5 MaxProfit Sharpe:       {s['sharpe']:.2f}")
    print(f"  V5 MaxProfit Sortino:      {s['sortino']:.2f}")
    print(f"  V5 MaxProfit Win Rate:     {s['win_rate']:.1%}")
    print(f"  V5 MaxProfit Profit Factor:{s['profit_factor']:.2f}")
    print(f"  V5 MaxProfit Trades:       {s['closed_trades']}")
    print(f"  V5 MaxProfit Costs:        {s['execution_costs']:.2f} RMB")
    print(f"  Accumulated Cash Yield:    {s['accumulated_cash_yield']:.2f} RMB")

    # Double Slippage Stress
    print("\n[Backtest 2] Running Super-V5 MaxProfit Double Slippage Stress (0.10% one-way)...")
    cfg_stress = Config(
        capital=30000.0,
        order_budget=5000.0,
        max_positions=4,
        max_exposure=0.65,
        hold_days=10,
        stop_loss=0.03,
        slippage=0.0010,
    )
    rep_stress = SuperV5MaxProfitReplay(
        frames, period_days, sig_v2, dividends, cfg_stress,
        vol_map=vol_map, daily_limits=tiered_limits,
        enable_cash_overlay=True, cash_annual_rate=0.022
    )
    rep_stress.bars = bars
    rep_stress.run()
    s_stress = rep_stress.summary()
    print(f"  Stress Total Return: {s_stress['total_return']:.2%}")
    print(f"  Stress Max Drawdown: {s_stress['max_drawdown']:.2%}")
    print(f"  Stress Calmar:       {s_stress['calmar']:.2f}")

    # Forward 2026 Test
    print("\n[Backtest 3] Running Super-V5 MaxProfit 2026 Forward Blind Test...")
    period_2026 = [d for d in calendar if "2026-01-01" <= d <= "2026-10-13"]
    sig_2026 = sig_v2[sig_v2.date.between("2026-01-01", "2026-10-13")].copy()
    rep_2026 = SuperV5MaxProfitReplay(
        frames, period_2026, sig_2026, dividends, cfg,
        vol_map=vol_map, daily_limits=tiered_limits,
        enable_cash_overlay=True, cash_annual_rate=0.022
    )
    rep_2026.bars = bars
    rep_2026.run()
    s_2026 = rep_2026.summary()
    print(f"  2026 Return: {s_2026['total_return']:.2%}, MaxDD: {s_2026['max_drawdown']:.2%}, Trades: {s_2026['closed_trades']}")

    # Calendar Year Breakdown
    print("\n[Backtest 4] Calendar Year Breakdown (Super-V5 MaxProfit):")
    yearly = []
    for yr in [2022, 2023, 2024, 2025, 2026]:
        end_d = f"{yr}-12-31" if yr < 2026 else "2026-10-13"
        p_yr = [d for d in calendar if f"{yr}-01-01" <= d <= end_d]
        sig_yr = sig_v2[sig_v2.date.between(f"{yr}-01-01", end_d)].copy()
        r_yr = SuperV5MaxProfitReplay(
            frames, p_yr, sig_yr, dividends, cfg,
            vol_map=vol_map, daily_limits=tiered_limits,
            enable_cash_overlay=True, cash_annual_rate=0.022
        )
        r_yr.bars = bars
        r_yr.run()
        sm_yr = r_yr.summary()
        yearly.append({"year": yr, "return": sm_yr["total_return"], "max_dd": sm_yr["max_drawdown"], "trades": sm_yr["closed_trades"]})
        print(f"  Year {yr}: Return={sm_yr['total_return']:>6.2%}, MaxDD={sm_yr['max_drawdown']:>6.2%}, Trades={sm_yr['closed_trades']}")

    out_dir = ROOT / "results/super-v5-maxprofit"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "continuous_2022_2025": {
            "total_return": s["total_return"],
            "cagr": s["cagr"],
            "max_drawdown": s["max_drawdown"],
            "calmar": s["calmar"],
            "sharpe": s["sharpe"],
            "sortino": s["sortino"],
            "win_rate": s["win_rate"],
            "profit_factor": s["profit_factor"],
            "trades": s["closed_trades"],
            "costs": s["execution_costs"],
            "accumulated_cash_yield": s["accumulated_cash_yield"],
            "stress_return": s_stress["total_return"],
            "stress_max_dd": s_stress["max_drawdown"],
            "stress_calmar": s_stress["calmar"]
        },
        "forward_2026": {
            "return": s_2026["total_return"],
            "max_dd": s_2026["max_drawdown"],
            "trades": s_2026["closed_trades"]
        },
        "yearly": yearly
    }
    with open(out_dir / "v5_maxprofit_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSaved Super-V5 MaxProfit results to: {out_dir / 'v5_maxprofit_summary.json'}")


if __name__ == "__main__":
    main()
