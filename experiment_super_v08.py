"""Production Runner for Super-V0.8 Architecture.

Name: Super-V0.8 (LambdaMART Cross-Sectional Ranking + Breadth-Adaptive Regime Sizing)
Core Innovations:
1. Signal: Model V0.8 Listwise LambdaMART (37 features including Amihud Illiquidity & Yang-Zhang Volatility)
2. Hurdle & Ranking: Absolute Positive Return Hurdle (pred_return >= +0.005) + Cross-Sectional LambdaRank Sorting
3. Sizing: Breadth-Adaptive Dynamic Regime Sizing (5.5k-7.5k, 4-5 positions, 90% exposure)
4. Audited Reality: T-1 Close Breadth Regime for morning executions (zero lookahead)
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from experiment_super_v7_regime import SuperV7RegimeReplay
from run import load
from validate_expanded import prepare_bars
from exploration_features import breadth_days
from engine import Config, tick

ROOT = Path("d:/Benefits/stock-research")
OUT_DIR = ROOT / "results/super-v08"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest, frames, calendar, dividends = load(ROOT / "data/forward-2026-10-13")
    bars = prepare_bars(frames)
    panel = pd.read_csv("results/midterm-access-v3/features.csv", dtype={"date": str, "code": str})
    pred_v08 = pd.read_csv("results/model-v08/predictions-h10-v08.csv", dtype={"date": str, "code": str})
    pred_v2 = pd.read_csv("results/model-v2/predictions-h10-v2.csv", dtype={"date": str, "code": str})
    vol_map = panel.set_index(["date", "code"])["ret_vol20"].to_dict()

    breadth_df = breadth_days(panel, return_col="adj_ret5")
    b_map = breadth_df.set_index("date")["breadth5"].to_dict()
    p_map = breadth_df.set_index("date")["pool_size"].to_dict()

    tiered_limits = {
        d: (0 if p_map.get(d, 0) < 100 or b < 0.45 else (1 if b < 0.55 else 2))
        for d, b in b_map.items()
    }
    allowed_tiered = {d for d, lim in tiered_limits.items() if lim > 0}

    # Signal: Hurdle on V2 expected return >= 0.005, sorted by LambdaMART V0.8 cross-sectional score
    merged = pd.merge(pred_v08[["date", "code", "pred_score_v08"]], pred_v2[["date", "code", "pred_net_return_v2"]], on=["date", "code"])
    sig_v08 = merged[merged.pred_net_return_v2 >= 0.005].assign(score=lambda df: df.pred_score_v08)[["date", "code", "score"]]
    sig_v08 = sig_v08[sig_v08.date.isin(allowed_tiered)]

    period_days = [d for d in calendar if "2022-01-01" <= d <= "2025-12-31"]
    cfg = Config(capital=30000.0, order_budget=7500.0, max_positions=5, max_exposure=0.90, hold_days=10, stop_loss=0.03)

    print("=" * 88)
    print(">>> EXECUTING SUPER-V0.8 (LAMBDAMART RANKING + REGIME SIZING) <<<")
    print("=" * 88)

    # 1. 2022-2025 Continuous Pure Trading Alpha
    rep_pure = SuperV7RegimeReplay(frames, period_days, sig_v08, dividends, cfg, vol_map=vol_map, daily_limits=tiered_limits, breadth_lookup=b_map, enable_cash_overlay=False)
    rep_pure.bars = bars
    rep_pure.run()
    s_pure = rep_pure.summary()

    # 2. 2022-2025 Continuous With 2.2% Cash Overlay
    rep_cash = SuperV7RegimeReplay(frames, period_days, sig_v08, dividends, cfg, vol_map=vol_map, daily_limits=tiered_limits, breadth_lookup=b_map, enable_cash_overlay=True)
    rep_cash.bars = bars
    rep_cash.run()
    s_cash = rep_cash.summary()

    calmar_pure = s_pure["cagr"] / abs(s_pure["max_drawdown"]) if s_pure["max_drawdown"] != 0 else 0
    calmar_cash = s_cash["cagr"] / abs(s_cash["max_drawdown"]) if s_cash["max_drawdown"] != 0 else 0

    print(f"Continuous 2022-2025 (Pure Trading Alpha): Return={s_pure['total_return']:>7.2%}, CAGR={s_pure['cagr']:>6.2%}, MaxDD={s_pure['max_drawdown']:>6.2%}, Calmar={calmar_pure:>5.2f}, Sharpe={s_pure['sharpe']:>5.2f}, Trades={s_pure['closed_trades']}")
    print(f"Continuous 2022-2025 (With 2.2% Cash):     Return={s_cash['total_return']:>7.2%}, CAGR={s_cash['cagr']:>6.2%}, MaxDD={s_cash['max_drawdown']:>6.2%}, Calmar={calmar_cash:>5.2f}, Sharpe={s_cash['sharpe']:>5.2f}, Trades={s_cash['closed_trades']}")

    # 3. 2026 Blind Forward Test
    cal_26 = [d for d in calendar if "2026-01-01" <= d <= "2026-10-13"]
    sig_26 = sig_v08[sig_v08.date.isin(set(cal_26))]
    r26 = SuperV7RegimeReplay(frames, cal_26, sig_26, dividends, cfg, vol_map=vol_map, daily_limits=tiered_limits, breadth_lookup=b_map, enable_cash_overlay=True)
    r26.bars = bars
    r26.run()
    s26 = r26.summary()
    print(f"Forward 2026 Blind Test (With Cash):        Return={s26['total_return']:>7.2%}, MaxDD={s26['max_drawdown']:>6.2%}, Trades={s26['closed_trades']}")

    # 4. Yearly Breakdown
    print("\nYearly Breakdown (With Cash Overlay):")
    yearly_results = []
    for yr in [2022, 2023, 2024, 2025, 2026]:
        end_d = f"{yr}-12-31" if yr < 2026 else "2026-10-13"
        p_yr = [d for d in calendar if f"{yr}-01-01" <= d <= end_d]
        sig_yr = sig_v08[sig_v08.date.between(f"{yr}-01-01", end_d)].copy()
        r_yr = SuperV7RegimeReplay(frames, p_yr, sig_yr, dividends, cfg, vol_map=vol_map, daily_limits=tiered_limits, breadth_lookup=b_map, enable_cash_overlay=True)
        r_yr.bars = bars
        r_yr.run()
        sm = r_yr.summary()
        calm = sm["cagr"] / abs(sm["max_drawdown"]) if sm["max_drawdown"] != 0 else 0
        print(f"Year {yr}: Return={sm['total_return']:>7.2%}, MaxDD={sm['max_drawdown']:>6.2%}, Calmar={calm:>5.2f}, Trades={sm['closed_trades']}")
        yearly_results.append({
            "year": yr,
            "return": sm["total_return"],
            "max_dd": sm["max_drawdown"],
            "calmar": calm,
            "trades": sm["closed_trades"],
        })

    # Save summary and artifacts
    summary = {
        "version": "super-v0.8-frozen",
        "name": "Super-V0.8 (LambdaMART Cross-Sectional Ranking + Breadth-Adaptive Regime Sizing)",
        "created_at_utc": "2026-10-20T13:00:00Z",
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
            "return": s26["total_return"],
            "max_dd": s26["max_drawdown"],
            "trades": s26["closed_trades"],
        },
        "yearly": yearly_results,
    }
    with open(OUT_DIR / "v08_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame(rep_cash.equity).to_csv(OUT_DIR / "equity.csv", index=False)
    pd.DataFrame(rep_cash.trades).to_csv(OUT_DIR / "trades.csv", index=False)
    print(f"\nArtifacts successfully written to {OUT_DIR}")


if __name__ == "__main__":
    main()
