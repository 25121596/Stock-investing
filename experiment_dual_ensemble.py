"""Dual Model V2 + V3 Ensemble Experiment:
Combines Model V2 (Trend & Macro Technicals) + Model V3 (A-Share Turnover & Microstructure).
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
from engine import Config

ROOT = Path(__file__).resolve().parent


def main():
    manifest, frames, calendar, dividends = load(DATASET)
    bars = prepare_bars(frames)
    panel = pd.read_csv(DONOR / "features.csv", dtype={"date": str, "code": str})
    pred_v2 = pd.read_csv("results/model-v2/predictions-h10-v2.csv", dtype={"date": str, "code": str})
    pred_v3 = pd.read_csv("results/model-v3/predictions-h10-v3.csv", dtype={"date": str, "code": str})
    vol_map = panel.set_index(["date", "code"])["ret_vol20"].to_dict()

    breadth_df = breadth_days(panel, return_col="adj_ret5")
    breadth_lookup = breadth_df.set_index("date")["breadth5"].to_dict()
    pool_lookup = breadth_df.set_index("date")["pool_size"].to_dict()
    tiered_limits = {d: (0 if pool_lookup.get(d, 0) < 100 or b < 0.45 else (1 if b < 0.55 else 2)) for d, b in breadth_lookup.items()}
    allowed_tiered = {d for d, lim in tiered_limits.items() if lim > 0}

    merged = pd.merge(pred_v2[["date", "code", "pred_net_return_v2"]], pred_v3[["date", "code", "pred_net_return_v3"]], on=["date", "code"])
    merged["rank_v2"] = merged.groupby("date")["pred_net_return_v2"].rank(pct=True)
    merged["rank_v3"] = merged.groupby("date")["pred_net_return_v3"].rank(pct=True)
    merged["ensemble_score"] = 0.5 * merged["rank_v2"] + 0.5 * merged["rank_v3"]

    mask = (merged.pred_net_return_v2 >= 0.003) & (merged.pred_net_return_v3 >= 0.003)
    sig_ens = merged[mask][["date", "code", "ensemble_score"]].rename(columns={"ensemble_score": "score"})
    sig_ens = sig_ens[sig_ens.date.isin(allowed_tiered)]

    period_days = [d for d in calendar if "2022-01-01" <= d <= "2025-12-31"]
    cfg = Config(capital=30000.0, order_budget=5000.0, max_positions=4, max_exposure=0.65, hold_days=10, stop_loss=0.03)

    rep = SuperV5MaxProfitReplay(frames, period_days, sig_ens, dividends, cfg, vol_map=vol_map, daily_limits=tiered_limits, enable_cash_overlay=True)
    rep.bars = bars
    rep.run()
    s = rep.summary()
    calmar = s["cagr"] / abs(s["max_drawdown"])

    print("=" * 80)
    print("DUAL MODEL V2 + V3 ENSEMBLE RESULTS:")
    print(f"Total Return:           {s['total_return']:.2%}")
    print(f"CAGR:                   {s['cagr']:.2%}")
    print(f"Max Drawdown:           {s['max_drawdown']:.2%}")
    print(f"Calmar Ratio:           {calmar:.2f}")
    print(f"Sharpe Ratio:           {s['sharpe']:.2f}")
    print(f"Sortino Ratio:          {s['sortino']:.2f}")
    print(f"Win Rate:               {s['win_rate']:.1%}")
    print(f"Profit Factor:          {s['profit_factor']:.2f}")
    print(f"Trades:                 {s['closed_trades']}")
    print(f"Accumulated Cash Yield: {s['accumulated_cash_yield']:.2f} RMB")
    print("=" * 80)

    # Calendar Year Breakdown
    print("\nYearly Breakdown:")
    for yr in [2022, 2023, 2024, 2025, 2026]:
        end_d = f"{yr}-12-31" if yr < 2026 else "2026-10-13"
        p_yr = [d for d in calendar if f"{yr}-01-01" <= d <= end_d]
        sig_yr = sig_ens[sig_ens.date.between(f"{yr}-01-01", end_d)].copy()
        r_yr = SuperV5MaxProfitReplay(frames, p_yr, sig_yr, dividends, cfg, vol_map=vol_map, daily_limits=tiered_limits, enable_cash_overlay=True)
        r_yr.bars = bars
        r_yr.run()
        sm_yr = r_yr.summary()
        print(f"  Year {yr}: Return={sm_yr['total_return']:>6.2%}, MaxDD={sm_yr['max_drawdown']:>6.2%}, Trades={sm_yr['closed_trades']}")


if __name__ == "__main__":
    main()
