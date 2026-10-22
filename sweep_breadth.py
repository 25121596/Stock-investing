"""Sensitivity analysis of market breadth gating parameters.

Sweeps:
- Breadth thresholds: 35%, 40%, 45%, 50%, 55%, 60%, 65%
- Return lookback columns: 'adj_ret5' (and other available returns)
- Minimum pool sizes: 50, 100, 150

Evaluates impact on:
- Total Return
- Max Drawdown
- Win Rate
- Closed Trades count
- Exposure / Execution costs
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from accessibility import AccessibleReplay
from engine import Config
from exploration_features import breadth_days
from run import load
from validate_expanded import prepare_bars

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data/forward-2026-10-13"
DONOR = ROOT / "results/midterm-access-v3"


def simulate(frames, bars, days, signals, dividends, config):
    replay = AccessibleReplay({}, days, signals, dividends, config)
    replay.frames = frames
    replay.bars = bars
    return replay.run()


def run_sweep():
    print("Loading dataset...")
    manifest, frames, calendar, dividends = load(DATASET)
    bars = prepare_bars(frames)

    print("Loading candidate panel and predictions...")
    panel = pd.read_csv(DONOR / "features.csv", dtype={"date": str, "code": str})
    predictions = pd.read_csv(DONOR / "predictions-h10.csv", dtype={"date": str, "code": str})

    # Base candidate pool: LGB selective (pred_net_return >= 0.005)
    base_candidates = predictions[predictions.pred_net_return >= 0.005][["date", "code", "pred_net_return"]].copy()
    base_candidates = base_candidates.rename(columns={"pred_net_return": "score"})

    # Evaluate across periods: 2022-2025 (in-sample) and 2026 (out-of-sample forward)
    periods = [
        ("2022-2025", "2022-01-01", "2025-12-31"),
        ("2026", "2026-01-01", "2026-10-13"),
    ]

    thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    results = []

    print("\nStarting Breadth Threshold Sensitivity Sweep...")
    print(f"{'Period':<12} {'Threshold':<10} {'AllowDays':<10} {'Return':<10} {'MaxDD':<10} {'WinRate':<10} {'Trades':<8} {'PF':<8}")
    print("-" * 78)

    for period_name, start_date, end_date in periods:
        period_days = [d for d in calendar if start_date <= d <= end_date]
        period_panel = panel[panel.date.ge(start_date) & panel.date.le(end_date)].copy()
        period_signals = base_candidates[base_candidates.date.ge(start_date) & base_candidates.date.le(end_date)].copy()

        # Benchmark: No breadth gating (threshold = 0.0)
        config = Config(hold_days=10)
        bench_replay = simulate(frames, bars, period_days, period_signals, dividends, config)
        bench_summary = bench_replay.summary()
        results.append({
            "period": period_name,
            "threshold": 0.0,
            "allow_days": len(period_days),
            "total_return": bench_summary["total_return"],
            "max_drawdown": bench_summary["max_drawdown"],
            "win_rate": bench_summary["win_rate"],
            "closed_trades": bench_summary["closed_trades"],
            "profit_factor": bench_summary["profit_factor"],
            "execution_costs": bench_summary["execution_costs"],
        })
        print(f"{period_name:<12} {'None(0%)':<10} {len(period_days):<10} {bench_summary['total_return']:>9.2%} "
              f"{bench_summary['max_drawdown']:>9.2%} {bench_summary['win_rate']:>9.2%} "
              f"{bench_summary['closed_trades']:>7} {bench_summary['profit_factor']:>7.2f}")

        # Compute daily breadth
        breadth_df = breadth_days(period_panel, return_col="adj_ret5")

        for th in thresholds:
            # Custom allow entry logic
            allowed_dates = set(
                breadth_df.loc[
                    (breadth_df.pool_size >= 100) & (breadth_df.breadth5 >= th), "date"
                ]
            )
            gated_signals = period_signals[period_signals.date.isin(allowed_dates)].copy()

            replay = simulate(frames, bars, period_days, gated_signals, dividends, config)
            summary = replay.summary()

            win_rate = summary["win_rate"] if summary["win_rate"] is not None else 0.0
            pf = summary["profit_factor"] if summary["profit_factor"] is not None else 0.0

            results.append({
                "period": period_name,
                "threshold": th,
                "allow_days": len(allowed_dates),
                "total_return": summary["total_return"],
                "max_drawdown": summary["max_drawdown"],
                "win_rate": win_rate,
                "closed_trades": summary["closed_trades"],
                "profit_factor": pf,
                "execution_costs": summary["execution_costs"],
            })

            print(f"{period_name:<12} {th:>9.0%} {len(allowed_dates):<10} {summary['total_return']:>9.2%} "
                  f"{summary['max_drawdown']:>9.2%} {win_rate:>9.2%} "
                  f"{summary['closed_trades']:>7} {pf:>7.2f}")

    df_res = pd.DataFrame(results)
    out_dir = ROOT / "results/sweep-breadth"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_dir / "breadth_sensitivity.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved results to {out_dir / 'breadth_sensitivity.csv'}")
    return df_res


if __name__ == "__main__":
    run_sweep()
