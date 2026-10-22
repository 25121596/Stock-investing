import pandas as pd
from experiment_super_v5_maxprofit import SuperV5MaxProfitReplay, DATASET, DONOR, MODEL_V2_DIR
from run import load
from validate_expanded import prepare_bars
from exploration_features import breadth_days
from engine import Config

manifest, frames, calendar, dividends = load(DATASET)
bars = prepare_bars(frames)
panel = pd.read_csv(DONOR / "features.csv", dtype={"date": str, "code": str})
pred_v2 = pd.read_csv("results/model-v2/predictions-h10-v2.csv", dtype={"date": str, "code": str})
vol_map = panel.set_index(["date", "code"])["ret_vol20"].to_dict()

breadth_df = breadth_days(panel, return_col="adj_ret5")
b_map = breadth_df.set_index("date")["breadth5"].to_dict()
p_map = breadth_df.set_index("date")["pool_size"].to_dict()

tiered_limits = {d: (0 if p_map.get(d, 0) < 100 or b < 0.45 else (1 if b < 0.55 else 2)) for d, b in b_map.items()}
allowed_tiered = {d for d, lim in tiered_limits.items() if lim > 0}
sig_v5 = pred_v2[pred_v2.pred_net_return_v2 >= 0.005][["date", "code", "pred_net_return_v2"]].rename(columns={"pred_net_return_v2": "score"})
sig_v5 = sig_v5[sig_v5.date.isin(allowed_tiered)]

period_days = [d for d in calendar if "2022-01-01" <= d <= "2025-12-31"]

candidates = [
    ("Super-V5 Frozen Baseline (Budget 5k, Exp 65%)", Config(capital=30000.0, order_budget=5000.0, max_positions=4, max_exposure=0.65, hold_days=10, stop_loss=0.03)),
    ("Super-V7 Scaled (Budget 6k, Exp 75%)",           Config(capital=30000.0, order_budget=6000.0, max_positions=4, max_exposure=0.75, hold_days=10, stop_loss=0.03)),
    ("Super-V7 High-Exposure (Budget 7k, Exp 90%)",     Config(capital=30000.0, order_budget=7000.0, max_positions=5, max_exposure=0.90, hold_days=10, stop_loss=0.03)),
    ("Super-V7 MaxAlpha (Budget 8k, Exp 90%)",          Config(capital=30000.0, order_budget=8000.0, max_positions=4, max_exposure=0.90, hold_days=10, stop_loss=0.03)),
]

print("=" * 110)
print(f"{'Strategy Version':<45} | {'Cash?':<6} | {'TotalRet':<9} | {'CAGR':<7} | {'MaxDD':<7} | {'Calmar':<7} | {'Sharpe':<7} | {'Trades':<6}")
print("=" * 110)

for name, cfg in candidates:
    for cash in [False, True]:
        rep = SuperV5MaxProfitReplay(frames, period_days, sig_v5, dividends, cfg, vol_map=vol_map, daily_limits=tiered_limits, enable_cash_overlay=cash)
        rep.bars = bars
        rep.run()
        s = rep.summary()
        calmar = s["cagr"] / abs(s["max_drawdown"]) if s["max_drawdown"] != 0 else 0
        cash_str = "Yes" if cash else "No"
        print(f"{name:<45} | {cash_str:<6} | {s['total_return']:>8.2%} | {s['cagr']:>6.2%} | {s['max_drawdown']:>6.2%} | {calmar:>6.2f} | {s['sharpe']:>6.2f} | {s['closed_trades']:>6}")

    # Also run 2026 blind forward
    cal_2026 = [d for d in calendar if "2026-01-01" <= d <= "2026-10-13"]
    sig_2026 = sig_v5[sig_v5.date.isin(set(cal_2026))]
    rep26 = SuperV5MaxProfitReplay(frames, cal_2026, sig_2026, dividends, cfg, vol_map=vol_map, daily_limits=tiered_limits, enable_cash_overlay=True)
    rep26.bars = bars
    rep26.run()
    s26 = rep26.summary()
    print(f"  --> 2026 Blind Forward: Return={s26['total_return']:>6.2%}, MaxDD={s26['max_drawdown']:>6.2%}, Trades={s26['closed_trades']}")
    print("-" * 110)
