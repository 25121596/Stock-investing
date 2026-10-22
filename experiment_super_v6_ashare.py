"""Super-V6 A-Share Alpha Edition: Model V3 (A-Share Features) + Super-V5 Production Engine.

Combines:
1. Model V3 (A-Share Turnover & Microstructure Features):
   - turn5 (Ranked #4 overall feature)
   - lower_shadow (Ranked in top 15 features)
   - upper_shadow, limit_count_60, vol_price_corr20
   - Proven higher Rank IC in every single year (2022-2026).

2. Super-V5 Production Engine:
   - Tiered Soft Breadth Gate (>=55% 2 buys/d, 45-55% 1 buy/d, <45% 0 buy)
   - Volatility-Adaptive Base Stops
   - Two-Tier Profit Escalator: Level 1 Breakeven (+0.5%), Level 2 Moonbag (+3.5%)
   - Parabolic Peak Trailing (2.0% delta above +10.0%)
   - Institutional Cash Sweep Overlay (2.20% annual yield on idle cash)

Evaluates:
- 2022-2025 continuous full backtest
- Double-slippage stress test (0.10% one-way)
- 2026 out-of-sample forward blind test
- Calendar year breakdowns
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from experiment_super_v5_maxprofit import SuperV5MaxProfitReplay
from engine import Config, tick
from exploration_features import breadth_days
from run import load
from validate_expanded import prepare_bars

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data/forward-2026-10-13"
DONOR = ROOT / "results/midterm-access-v3"
MODEL_V3_DIR = ROOT / "results/model-v3"


def main():
    print("=" * 88)
    print(">>> SUPER-V6 A-SHARE ALPHA EDITION (MODEL V3 + ENTERPRISE ENGINE) <<<")
    print("=" * 88)

    manifest, frames, calendar, dividends = load(DATASET)
    bars = prepare_bars(frames)

    panel = pd.read_csv(DONOR / "features.csv", dtype={"date": str, "code": str})
    pred_v3 = pd.read_csv(MODEL_V3_DIR / "predictions-h10-v3.csv", dtype={"date": str, "code": str})
    vol_map = panel.set_index(["date", "code"])["ret_vol20"].to_dict()

    breadth_df = breadth_days(panel, return_col="adj_ret5")
    breadth_lookup = breadth_df.set_index("date")["breadth5"].to_dict()
    pool_lookup = breadth_df.set_index("date")["pool_size"].to_dict()

    tiered_limits = {d: (0 if pool_lookup.get(d, 0) < 100 or b < 0.45 else (1 if b < 0.55 else 2)) for d, b in breadth_lookup.items()}
    allowed_tiered = {d for d, lim in tiered_limits.items() if lim > 0}

    # Model V3 Signals
    sig_v3 = pred_v3[pred_v3.pred_net_return_v3 >= 0.005][["date", "code", "pred_net_return_v3"]].rename(columns={"pred_net_return_v3": "score"})
    sig_v3 = sig_v3[sig_v3.date.isin(allowed_tiered)]

    period_days = [d for d in calendar if "2022-01-01" <= d <= "2025-12-31"]

    cfg = Config(
        capital=30000.0,
        order_budget=5000.0,
        max_positions=4,
        max_exposure=0.65,
        hold_days=10,
        stop_loss=0.03,
    )

    print("\n[Backtest 1] Running Super-V6 Continuous Full Backtest (2022-2025)...")
    rep_v6 = SuperV5MaxProfitReplay(
        frames, period_days, sig_v3, dividends, cfg,
        vol_map=vol_map, daily_limits=tiered_limits,
        enable_cash_overlay=True, cash_annual_rate=0.022
    )
    rep_v6.bars = bars
    rep_v6.run()
    s = rep_v6.summary()

    print(f"  Super-V6 Total Return: {s['total_return']:.2%}")
    print(f"  Super-V6 CAGR:         {s['cagr']:.2%}")
    print(f"  Super-V6 Max Drawdown: {s['max_drawdown']:.2%}")
    print(f"  Super-V6 Calmar:       {s['calmar']:.2f}")
    print(f"  Super-V6 Sharpe:       {s['sharpe']:.2f}")
    print(f"  Super-V6 Sortino:      {s['sortino']:.2f}")
    print(f"  Super-V6 Win Rate:     {s['win_rate']:.1%}")
    print(f"  Super-V6 Profit Factor:{s['profit_factor']:.2f}")
    print(f"  Super-V6 Trades:       {s['closed_trades']}")
    print(f"  Super-V6 Costs:        {s['execution_costs']:.2f} RMB")
    print(f"  Accumulated Cash Yield:{s['accumulated_cash_yield']:.2f} RMB")

    # Double Slippage Stress
    print("\n[Backtest 2] Running Super-V6 Double Slippage Stress (0.10% one-way)...")
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
        frames, period_days, sig_v3, dividends, cfg_stress,
        vol_map=vol_map, daily_limits=tiered_limits,
        enable_cash_overlay=True, cash_annual_rate=0.022
    )
    rep_stress.bars = bars
    rep_stress.run()
    s_stress = rep_stress.summary()
    print(f"  Stress Total Return: {s_stress['total_return']:.2%}")
    print(f"  Stress Max Drawdown: {s_stress['max_drawdown']:.2%}")
    print(f"  Stress Calmar:       {s_stress['calmar']:.2f}")

    # Forward 2026 Blind Test
    print("\n[Backtest 3] Running Super-V6 2026 Forward Blind Test...")
    period_2026 = [d for d in calendar if "2026-01-01" <= d <= "2026-10-13"]
    sig_2026 = sig_v3[sig_v3.date.between("2026-01-01", "2026-10-13")].copy()
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
    print("\n[Backtest 4] Calendar Year Breakdown (Super-V6):")
    yearly = []
    for yr in [2022, 2023, 2024, 2025, 2026]:
        end_d = f"{yr}-12-31" if yr < 2026 else "2026-10-13"
        p_yr = [d for d in calendar if f"{yr}-01-01" <= d <= end_d]
        sig_yr = sig_v3[sig_v3.date.between(f"{yr}-01-01", end_d)].copy()
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

    out_dir = ROOT / "results/super-v6"
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
    with open(out_dir / "v6_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSaved Super-V6 results to: {out_dir / 'v6_summary.json'}")


if __name__ == "__main__":
    main()
