"""Backtest evaluation of Model Ensemble (LR + LGBM) vs Single Models.

Compares:
1. base_lgb10: LightGBM selective (pred_return >= 0.005)
2. breadth_lgb10: LightGBM selective + Market Breadth >= 50%
3. ensemble_intersection: LR >= 0.55 AND LGBM >= 0.005 (no breadth gate)
4. breadth_ensemble_int: Ensemble Intersection + Market Breadth >= 50%
5. breadth_ensemble_avg: Ensemble Average score >= 0.5 + Market Breadth >= 50%
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from accessibility import AccessibleReplay
from engine import Config
from ensemble import ensemble_signals
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


def main():
    print("Loading data...")
    manifest, frames, calendar, dividends = load(DATASET)
    bars = prepare_bars(frames)

    panel = pd.read_csv(DONOR / "features.csv", dtype={"date": str, "code": str})
    predictions = pd.read_csv(DONOR / "predictions-h10.csv", dtype={"date": str, "code": str})

    periods = [
        ("2022-2025", "2022-01-01", "2025-12-31"),
        ("2026", "2026-01-01", "2026-10-13"),
    ]

    config = Config(hold_days=10)

    # 1. Base LGB
    sig_lgb = predictions[predictions.pred_net_return >= 0.005][["date", "code", "pred_net_return"]].rename(columns={"pred_net_return": "score"})

    # 2. Ensemble Intersection (LR >= 0.55 AND LGB >= 0.005)
    sig_ens_int = ensemble_signals(predictions, method="intersection", p_win_threshold=0.55, return_threshold=0.005)

    # 3. Ensemble Average (score >= 0.5)
    sig_ens_avg = ensemble_signals(predictions, method="average")

    all_strategies = []

    print("\n" + "=" * 88)
    print(f"{'Period':<12} {'Strategy':<26} {'Return':<10} {'MaxDD':<10} {'WinRate':<10} {'Trades':<8} {'PF':<8} {'Costs':<8}")
    print("=" * 88)

    for period_name, start_date, end_date in periods:
        period_days = [d for d in calendar if start_date <= d <= end_date]
        period_panel = panel[panel.date.ge(start_date) & panel.date.le(end_date)].copy()

        # Compute Breadth for this period
        breadth_df = breadth_days(period_panel, return_col="adj_ret5")
        allowed_dates = set(breadth_df.loc[(breadth_df.pool_size >= 100) & (breadth_df.breadth5 >= 0.50), "date"])

        cases = [
            ("base_lgb10", "1. 原LGBM单模型", sig_lgb[sig_lgb.date.between(start_date, end_date)]),
            ("breadth_lgb10", "2. 广度+LGBM (冠军策略)", sig_lgb[sig_lgb.date.between(start_date, end_date) & sig_lgb.date.isin(allowed_dates)]),
            ("ensemble_intersection", "3. 双模型交集(无广度)", sig_ens_int[sig_ens_int.date.between(start_date, end_date)]),
            ("breadth_ensemble_int", "4. 广度+双模型交集", sig_ens_int[sig_ens_int.date.between(start_date, end_date) & sig_ens_int.date.isin(allowed_dates)]),
            ("breadth_ensemble_avg", "5. 广度+双模型均值", sig_ens_avg[sig_ens_avg.date.between(start_date, end_date) & sig_ens_avg.date.isin(allowed_dates)]),
        ]

        for case_id, name, sigs in cases:
            replay = simulate(frames, bars, period_days, sigs, dividends, config)
            summary = replay.summary()
            win_rate = summary["win_rate"] if summary["win_rate"] is not None else 0.0
            pf = summary["profit_factor"] if summary["profit_factor"] is not None else 0.0

            all_strategies.append({
                "period": period_name,
                "case_id": case_id,
                "strategy_name": name,
                "signals_count": len(sigs),
                "total_return": summary["total_return"],
                "max_drawdown": summary["max_drawdown"],
                "win_rate": win_rate,
                "closed_trades": summary["closed_trades"],
                "profit_factor": pf,
                "execution_costs": summary["execution_costs"],
            })

            print(f"{period_name:<12} {name:<26} {summary['total_return']:>9.2%} "
                  f"{summary['max_drawdown']:>9.2%} {win_rate:>9.2%} "
                  f"{summary['closed_trades']:>7} {pf:>7.2f} {summary['execution_costs']:>7.0f}")

    df_out = pd.DataFrame(all_strategies)
    out_dir = ROOT / "results/ensemble-backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_dir / "ensemble_comparison.csv", index=False, encoding="utf-8-sig")
    print("\n" + "=" * 88)
    print(f"Results saved to: {out_dir / 'ensemble_comparison.csv'}")


if __name__ == "__main__":
    main()
