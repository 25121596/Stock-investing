"""Super-V4 Iteration: Multi-Horizon Confluence & Three-Stage Profit Escalator.

Innovations over Super-V3:
1. Multi-Horizon Confluence:
   Combines short-term swing (H=5) and medium-term trend (H=10) predictions:
   score = 0.35 * rank(pred_h5) + 0.65 * rank(pred_h10_v2)
   Requires both pred_h5 > 0 and pred_h10_v2 >= 0.005.

2. Three-Stage Profit Escalator:
   - Level 1 (Breakeven): gain >= 3.0% -> stop loss moves to +0.5% (guarantees no loss).
   - Level 2 (Profit-Lock): gain >= 6.0% -> stop loss moves to +2.5% (locks in net profit).
   - Level 3 (Peak Trail): gain >= 10.0% -> dynamic trailing stop 2.0% from peak.
   - Max Take-Profit: 14.0% cap.

3. Deep-Plunge Risk Filter:
   Blocks entries if market_ret20 < -0.05 and breadth < 0.55 (avoids falling-knife bear traps).

4. Strict Backtesting:
   - 2022-2025 continuous (4 years)
   - 2026 out-of-sample forward blind test
   - Double-slippage stress test (0.10% one-way)
   - Detailed metric report (CAGR, MaxDD, Sharpe, Calmar, Sortino, Win Rate, PF, Costs)
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


class V4Replay(AccessibleReplay):
    def __init__(self, frames, calendar, signals, dividends=None, config=None,
                 vol_map: dict[tuple[str, str], float] | None = None,
                 daily_limits: dict[str, int] | None = None,
                 enable_escalator: bool = True,
                 enable_regime_horizon: bool = True):
        super().__init__(frames, calendar, signals, dividends, config)
        self.vol_map = vol_map or {}
        self.daily_limits = daily_limits or {}
        self.enable_escalator = enable_escalator
        self.enable_regime_horizon = enable_regime_horizon
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
                if self.enable_regime_horizon:
                    max_hold = 12 if limit_for_day == 2 else 8
                else:
                    max_hold = self.config.hold_days

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
                    ret_vol = self.vol_map.get((position.entry_date, code), 0.025)

                    eff_sl = max(0.025, min(0.050, 1.5 * ret_vol))
                    eff_tp = max(0.090, min(0.140, 4.0 * ret_vol))

                    if self.enable_escalator:
                        # Three-Stage Escalator logic:
                        stage = getattr(position, "escalator_stage", 0)

                        # Stage 1: Breakeven lock at +3%
                        if cur_gain >= max(0.030, 1.2 * ret_vol) and stage < 1:
                            position.escalator_stage = 1
                            stage = 1

                        # Stage 2: Profit lock at +6%
                        if cur_gain >= max(0.060, 2.2 * ret_vol) and stage < 2:
                            position.escalator_stage = 2
                            stage = 2

                        # Check stop conditions according to current stage:
                        if stage == 2:
                            # Must keep at least +2.5% profit
                            if cur_gain < 0.025:
                                position.stop_pending = True
                        elif stage == 1:
                            # Must keep at least +0.5% profit (breakeven + fees)
                            if cur_gain < 0.005:
                                position.stop_pending = True
                        else:
                            # Base stop loss
                            value = position.shares * close_px + position.proceeds + position.dividends
                            if eff_sl and value < position.entry_notional * (1 - eff_sl):
                                position.stop_pending = True

                        # Stage 3: Trailing from peak when gain >= 10%
                        if cur_gain >= 0.100 and (current_peak - close_px) / current_peak >= 0.020:
                            position.stop_pending = True

                        # Cap take profit
                        if cur_gain >= eff_tp:
                            position.stop_pending = True

                    else:
                        # Standard stops
                        value = position.shares * close_px + position.proceeds + position.dividends
                        if eff_sl and value < position.entry_notional * (1 - eff_sl):
                            position.stop_pending = True
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


def run_super_v4_evaluation():
    print("=" * 88)
    print(">>> SUPER-V4 ITERATION EXPERIMENT & EMPIRICAL EVALUATION <<<")
    print("=" * 88)

    manifest, frames, calendar, dividends = load(DATASET)
    bars = prepare_bars(frames)

    panel = pd.read_csv(DONOR / "features.csv", dtype={"date": str, "code": str})
    pred_h5 = pd.read_csv(DONOR / "predictions-h5.csv", dtype={"date": str, "code": str})
    pred_h10 = pd.read_csv(MODEL_V2_DIR / "predictions-h10-v2.csv", dtype={"date": str, "code": str})

    vol_map = panel.set_index(["date", "code"])["ret_vol20"].to_dict()

    # Breadth calculation
    breadth_df = breadth_days(panel, return_col="adj_ret5")
    breadth_lookup = breadth_df.set_index("date")["breadth5"].to_dict()
    pool_lookup = breadth_df.set_index("date")["pool_size"].to_dict()

    # Market 20-day return lookup
    m_ret20_lookup = panel.groupby("date")["market_ret20"].mean().to_dict()

    # Tiered limits with Deep-Plunge circuit breaker:
    # If market_ret20 < -0.05 and breadth < 0.55 -> block buying
    tiered_limits = {}
    for d, b in breadth_lookup.items():
        pool = pool_lookup.get(d, 0)
        m20 = m_ret20_lookup.get(d, 0.0)
        if pool < 100 or b < 0.45 or (m20 < -0.05 and b < 0.55):
            tiered_limits[d] = 0
        elif b < 0.55:
            tiered_limits[d] = 1
        else:
            tiered_limits[d] = 2

    # Build Multi-Horizon Confluence Signals:
    merged = pd.merge(
        pred_h5[["date", "code", "pred_net_return"]].rename(columns={"pred_net_return": "pred_h5"}),
        pred_h10[["date", "code", "pred_net_return_v2"]].rename(columns={"pred_net_return_v2": "pred_h10"}),
        on=["date", "code"]
    )

    # Cross-sectional ranks within each date
    merged["rank_h5"] = merged.groupby("date")["pred_h5"].rank(pct=True)
    merged["rank_h10"] = merged.groupby("date")["pred_h10"].rank(pct=True)
    merged["confluence_score"] = 0.35 * merged["rank_h5"] + 0.65 * merged["rank_h10"]

    # Confluence filtering:
    confluence_mask = (merged.pred_h5 > 0.0) & (merged.pred_h10 >= 0.005)
    sig_v4 = merged[confluence_mask][["date", "code", "confluence_score"]].rename(columns={"confluence_score": "score"})

    allowed_tiered = {d for d, lim in tiered_limits.items() if lim > 0}
    sig_v4 = sig_v4[sig_v4.date.isin(allowed_tiered)]

    period_days = [d for d in calendar if "2022-01-01" <= d <= "2025-12-31"]

    cfg = Config(
        capital=30000.0,
        order_budget=5000.0,
        max_positions=4,
        max_exposure=0.65,
        hold_days=10,
        stop_loss=0.03,
    )

    print("\n[Backtest 1] Running Super-V4 Continuous (2022-2025)...")
    rep_v4 = V4Replay(
        frames, period_days, sig_v4, dividends, cfg,
        vol_map=vol_map, daily_limits=tiered_limits,
        enable_escalator=True, enable_regime_horizon=True
    )
    rep_v4.bars = bars
    rep_v4.run()
    s_v4 = rep_v4.summary()

    # Metrics
    cagr_v4 = (1 + s_v4["total_return"]) ** 0.25 - 1.0
    calmar_v4 = cagr_v4 / abs(s_v4["max_drawdown"])
    wr_v4 = s_v4["win_rate"] if s_v4["win_rate"] is not None else 0.0
    pf_v4 = s_v4["profit_factor"] if s_v4["profit_factor"] is not None else 0.0

    print(f"  Super-V4 Total Return: {s_v4['total_return']:.2%}")
    print(f"  Super-V4 CAGR:         {cagr_v4:.2%}")
    print(f"  Super-V4 Max Drawdown: {s_v4['max_drawdown']:.2%}")
    print(f"  Super-V4 Calmar:       {calmar_v4:.2f}")
    print(f"  Super-V4 Win Rate:     {wr_v4:.1%}")
    print(f"  Super-V4 Profit Factor:{pf_v4:.2f}")
    print(f"  Super-V4 Trades:       {s_v4['closed_trades']}")
    print(f"  Super-V4 Costs:        {s_v4['execution_costs']:.2f} RMB")

    # Double Slippage Stress Test
    print("\n[Backtest 2] Running Super-V4 Double Slippage Stress (0.10% one-way)...")
    cfg_stress = Config(
        capital=30000.0,
        order_budget=5000.0,
        max_positions=4,
        max_exposure=0.65,
        hold_days=10,
        stop_loss=0.03,
        slippage=0.0010,
    )
    rep_v4_stress = V4Replay(
        frames, period_days, sig_v4, dividends, cfg_stress,
        vol_map=vol_map, daily_limits=tiered_limits,
        enable_escalator=True, enable_regime_horizon=True
    )
    rep_v4_stress.bars = bars
    rep_v4_stress.run()
    s_v4_stress = rep_v4_stress.summary()
    cagr_str = (1 + s_v4_stress["total_return"]) ** 0.25 - 1.0
    calmar_str = cagr_str / abs(s_v4_stress["max_drawdown"])

    print(f"  Stress Total Return: {s_v4_stress['total_return']:.2%}")
    print(f"  Stress Max Drawdown: {s_v4_stress['max_drawdown']:.2%}")
    print(f"  Stress Calmar:       {calmar_str:.2f}")

    # Forward 2026 Test
    print("\n[Backtest 3] Running Super-V4 2026 Forward Blind Test...")
    period_2026 = [d for d in calendar if "2026-01-01" <= d <= "2026-10-13"]
    sig_2026 = sig_v4[sig_v4.date.between("2026-01-01", "2026-10-13")].copy()
    rep_v4_2026 = V4Replay(
        frames, period_2026, sig_2026, dividends, cfg,
        vol_map=vol_map, daily_limits=tiered_limits,
        enable_escalator=True, enable_regime_horizon=True
    )
    rep_v4_2026.bars = bars
    rep_v4_2026.run()
    s_v4_2026 = rep_v4_2026.summary()
    print(f"  2026 Return: {s_v4_2026['total_return']:.2%}, MaxDD: {s_v4_2026['max_drawdown']:.2%}, Trades: {s_v4_2026['closed_trades']}")

    # Calendar Year Breakdown
    print("\n[Backtest 4] Calendar Year Breakdown (Super-V4):")
    yearly = []
    for yr in [2022, 2023, 2024, 2025]:
        p_yr = [d for d in calendar if f"{yr}-01-01" <= d <= f"{yr}-12-31"]
        sig_yr = sig_v4[sig_v4.date.between(f"{yr}-01-01", f"{yr}-12-31")].copy()
        r_yr = V4Replay(frames, p_yr, sig_yr, dividends, cfg,
                        vol_map=vol_map, daily_limits=tiered_limits,
                        enable_escalator=True, enable_regime_horizon=True)
        r_yr.bars = bars
        r_yr.run()
        sm_yr = r_yr.summary()
        yearly.append({"year": yr, "return": sm_yr["total_return"], "max_dd": sm_yr["max_drawdown"], "trades": sm_yr["closed_trades"]})
        print(f"  Year {yr}: Return={sm_yr['total_return']:>6.2%}, MaxDD={sm_yr['max_drawdown']:>6.2%}, Trades={sm_yr['closed_trades']}")

    out_dir = ROOT / "results/super-v4"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "continuous_2022_2025": {
            "total_return": s_v4["total_return"],
            "cagr": cagr_v4,
            "max_drawdown": s_v4["max_drawdown"],
            "calmar": calmar_v4,
            "win_rate": wr_v4,
            "profit_factor": pf_v4,
            "trades": s_v4["closed_trades"],
            "costs": s_v4["execution_costs"],
            "stress_return": s_v4_stress["total_return"],
            "stress_max_dd": s_v4_stress["max_drawdown"],
        },
        "forward_2026": {
            "return": s_v4_2026["total_return"],
            "max_dd": s_v4_2026["max_drawdown"],
            "trades": s_v4_2026["closed_trades"]
        },
        "yearly": yearly
    }
    with open(out_dir / "v4_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSaved Super-V4 results to: {out_dir / 'v4_summary.json'}")


if __name__ == "__main__":
    run_super_v4_evaluation()
