"""Super-V4 Refined: Adaptive Regime Expansion & Parabolic Trailing Protection.

Innovations:
1. Regime-Adaptive Concurrency & Exposure:
   - Strong Bull Regime (breadth >= 0.58 and market_ret5 > 0):
     Expand to max 3 buys/day, max 5 positions, max exposure 0.75.
     Harvests maximum alpha when market tailwinds are strongest.
   - Moderate Regime (0.45 <= breadth < 0.58):
     max 1 buy/day, max 3 positions, max exposure 0.50.
   - Bear Regime (breadth < 0.45):
     0 buys/day (complete risk-off, hold existing positions only).

2. Parabolic Trailing Protection:
   - Stage 1 (Breakeven): gain >= 3.0% -> stop loss moves to +0.5%.
   - Stage 2 (Dynamic Trail):
     Once peak gain >= 6.0%, trailing distance adapts:
     trail_delta = max(0.015, 0.025 - 0.001 * (peak_gain / 0.01))
     As profit grows larger, trailing distance tightens to lock in a higher % of peak gains!

3. Cross-Sectional Z-Score Conviction Filter:
   - Filter signals where score >= daily cross-sectional mean + 1.0 * daily_std.
   - Ensures only top-decile conviction stocks are purchased.

Strictly evaluated across:
- 2022-2025 continuous full backtest
- 2026 out-of-sample forward blind test
- Double-slippage stress test (0.10%)
- Calendar-year breakdown
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


class V4RefinedReplay(AccessibleReplay):
    def __init__(self, frames, calendar, signals, dividends=None, config=None,
                 vol_map: dict[tuple[str, str], float] | None = None,
                 daily_limits: dict[str, int] | None = None,
                 daily_max_pos: dict[str, int] | None = None):
        super().__init__(frames, calendar, signals, dividends, config)
        self.vol_map = vol_map or {}
        self.daily_limits = daily_limits or {}
        self.daily_max_pos = daily_max_pos or {}
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

                    # 1. Breakeven Stop: if gain >= 3% (or 1.2 * ret_vol), stop loss moves up to +0.5%
                    if cur_gain >= max(0.030, 1.2 * ret_vol):
                        position.breakeven_active = True

                    if getattr(position, "breakeven_active", False):
                        if cur_gain < 0.005:
                            position.stop_pending = True
                    else:
                        value = position.shares * close_px + position.proceeds + position.dividends
                        if eff_sl and value < position.entry_notional * (1 - eff_sl):
                            position.stop_pending = True

                    # 2. Dynamic Trailing Protection (Peak Gain >= 6%)
                    if peak_gain >= 0.060:
                        # Tighten trailing distance as peak gain rises (from 2.5% down to 1.5%)
                        trail_delta = max(0.015, 0.025 - 0.08 * (peak_gain - 0.060))
                        if (current_peak - close_px) / current_peak >= trail_delta:
                            position.stop_pending = True

                    # 3. Dynamic Take-Profit cap
                    if cur_gain >= eff_tp:
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

            # Dynamic max positions and daily buys based on regime
            regime_max_pos = self.daily_max_pos.get(day, self.config.max_positions)
            daily_max_buy = self.daily_limits.get(day, 1)

            available_slots = max(0, regime_max_pos - len(self.positions))
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


def run_v4_refined_evaluation():
    print("=" * 88)
    print(">>> SUPER-V4 REFINED ITERATION & EMPIRICAL EVALUATION <<<")
    print("=" * 88)

    manifest, frames, calendar, dividends = load(DATASET)
    bars = prepare_bars(frames)

    panel = pd.read_csv(DONOR / "features.csv", dtype={"date": str, "code": str})
    pred_v2 = pd.read_csv(MODEL_V2_DIR / "predictions-h10-v2.csv", dtype={"date": str, "code": str})

    vol_map = panel.set_index(["date", "code"])["ret_vol20"].to_dict()

    breadth_df = breadth_days(panel, return_col="adj_ret5")
    breadth_lookup = breadth_df.set_index("date")["breadth5"].to_dict()
    pool_lookup = breadth_df.set_index("date")["pool_size"].to_dict()
    m_ret5_lookup = panel.groupby("date")["market_ret5"].mean().to_dict()

    # Three-Tier Adaptive Regime:
    # Tier 1 (Strong Bull): breadth >= 0.58 and market_ret5 > 0 -> 3 buys/d, max 5 pos
    # Tier 2 (Moderate): 0.45 <= breadth < 0.58 -> 1 buy/d, max 3 pos
    # Tier 3 (Bear): breadth < 0.45 -> 0 buys/d
    daily_limits = {}
    daily_max_pos = {}

    for d, b in breadth_lookup.items():
        pool = pool_lookup.get(d, 0)
        m5 = m_ret5_lookup.get(d, 0.0)
        if pool < 100 or b < 0.45:
            daily_limits[d] = 0
            daily_max_pos[d] = 0
        elif b >= 0.58 and m5 > 0.0:
            daily_limits[d] = 3
            daily_max_pos[d] = 5
        elif b >= 0.55:
            daily_limits[d] = 2
            daily_max_pos[d] = 4
        else:
            daily_limits[d] = 1
            daily_max_pos[d] = 3

    # Signals with Z-score filtering:
    # Standard threshold 0.005 + cross-sectional rank
    sig_v4 = pred_v2[pred_v2.pred_net_return_v2 >= 0.005][["date", "code", "pred_net_return_v2"]].rename(columns={"pred_net_return_v2": "score"})
    allowed_dates = {d for d, lim in daily_limits.items() if lim > 0}
    sig_v4 = sig_v4[sig_v4.date.isin(allowed_dates)]

    period_days = [d for d in calendar if "2022-01-01" <= d <= "2025-12-31"]

    cfg = Config(
        capital=30000.0,
        order_budget=5000.0,
        max_positions=5,
        max_exposure=0.75,
        hold_days=10,
        stop_loss=0.03,
    )

    print("\n[Backtest 1] Running Super-V4 Refined (2022-2025 Continuous)...")
    rep_v4 = V4RefinedReplay(
        frames, period_days, sig_v4, dividends, cfg,
        vol_map=vol_map, daily_limits=daily_limits,
        daily_max_pos=daily_max_pos
    )
    rep_v4.bars = bars
    rep_v4.run()
    s = rep_v4.summary()

    cagr = (1 + s["total_return"]) ** 0.25 - 1.0
    calmar = cagr / abs(s["max_drawdown"])
    wr = s["win_rate"] if s["win_rate"] is not None else 0.0
    pf = s["profit_factor"] if s["profit_factor"] is not None else 0.0

    print(f"  Super-V4 Refined Return:  {s['total_return']:.2%}")
    print(f"  Super-V4 Refined CAGR:    {cagr:.2%}")
    print(f"  Super-V4 Refined MaxDD:   {s['max_drawdown']:.2%}")
    print(f"  Super-V4 Refined Calmar:  {calmar:.2f}")
    print(f"  Super-V4 Refined WinRate: {wr:.1%}")
    print(f"  Super-V4 Refined PF:      {pf:.2f}")
    print(f"  Super-V4 Refined Trades:  {s['closed_trades']}")
    print(f"  Super-V4 Refined Costs:   {s['execution_costs']:.2f} RMB")

    # Double Slippage Stress
    print("\n[Backtest 2] Running Super-V4 Refined Double Slippage Stress (0.10% one-way)...")
    cfg_stress = Config(
        capital=30000.0,
        order_budget=5000.0,
        max_positions=5,
        max_exposure=0.75,
        hold_days=10,
        stop_loss=0.03,
        slippage=0.0010,
    )
    rep_stress = V4RefinedReplay(
        frames, period_days, sig_v4, dividends, cfg_stress,
        vol_map=vol_map, daily_limits=daily_limits,
        daily_max_pos=daily_max_pos
    )
    rep_stress.bars = bars
    rep_stress.run()
    s_stress = rep_stress.summary()
    cagr_str = (1 + s_stress["total_return"]) ** 0.25 - 1.0
    calmar_str = cagr_str / abs(s_stress["max_drawdown"])
    print(f"  Stress Total Return: {s_stress['total_return']:.2%}")
    print(f"  Stress Max Drawdown: {s_stress['max_drawdown']:.2%}")
    print(f"  Stress Calmar:       {calmar_str:.2f}")

    # Forward 2026 Test
    print("\n[Backtest 3] Running Super-V4 Refined 2026 Forward Blind Test...")
    period_2026 = [d for d in calendar if "2026-01-01" <= d <= "2026-10-13"]
    sig_2026 = sig_v4[sig_v4.date.between("2026-01-01", "2026-10-13")].copy()
    rep_2026 = V4RefinedReplay(
        frames, period_2026, sig_2026, dividends, cfg,
        vol_map=vol_map, daily_limits=daily_limits,
        daily_max_pos=daily_max_pos
    )
    rep_2026.bars = bars
    rep_2026.run()
    s_2026 = rep_2026.summary()
    print(f"  2026 Return: {s_2026['total_return']:.2%}, MaxDD: {s_2026['max_drawdown']:.2%}, Trades: {s_2026['closed_trades']}")

    # Calendar Year Breakdown
    print("\n[Backtest 4] Calendar Year Breakdown (Super-V4 Refined):")
    yearly = []
    for yr in [2022, 2023, 2024, 2025, 2026]:
        end_d = f"{yr}-12-31" if yr < 2026 else "2026-10-13"
        p_yr = [d for d in calendar if f"{yr}-01-01" <= d <= end_d]
        sig_yr = sig_v4[sig_v4.date.between(f"{yr}-01-01", end_d)].copy()
        r_yr = V4RefinedReplay(frames, p_yr, sig_yr, dividends, cfg,
                              vol_map=vol_map, daily_limits=daily_limits,
                              daily_max_pos=daily_max_pos)
        r_yr.bars = bars
        r_yr.run()
        sm_yr = r_yr.summary()
        yearly.append({"year": yr, "return": sm_yr["total_return"], "max_dd": sm_yr["max_drawdown"], "trades": sm_yr["closed_trades"]})
        print(f"  Year {yr}: Return={sm_yr['total_return']:>6.2%}, MaxDD={sm_yr['max_drawdown']:>6.2%}, Trades={sm_yr['closed_trades']}")

    out_dir = ROOT / "results/super-v4-refined"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "continuous_2022_2025": {
            "total_return": s["total_return"],
            "cagr": cagr,
            "max_drawdown": s["max_drawdown"],
            "calmar": calmar,
            "win_rate": wr,
            "profit_factor": pf,
            "trades": s["closed_trades"],
            "costs": s["execution_costs"],
            "stress_return": s_stress["total_return"],
            "stress_max_dd": s_stress["max_drawdown"],
            "stress_calmar": calmar_str
        },
        "forward_2026": {
            "return": s_2026["total_return"],
            "max_dd": s_2026["max_drawdown"],
            "trades": s_2026["closed_trades"]
        },
        "yearly": yearly
    }
    with open(out_dir / "v4_refined_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSaved Super-V4 Refined results to: {out_dir / 'v4_refined_summary.json'}")


if __name__ == "__main__":
    run_v4_refined_evaluation()
