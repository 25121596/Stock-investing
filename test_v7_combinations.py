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

# 2. Model V3 Signal
sig_v3 = pred_v3[pred_v3.pred_net_return_v3 >= 0.005][["date", "code", "pred_net_return_v3"]].rename(columns={"pred_net_return_v3": "score"})
sig_v3 = sig_v3[sig_v3.date.isin(allowed_tiered)]

# 3. Dual Ensemble Signal
merged = pd.merge(pred_v2[["date", "code", "pred_net_return_v2"]], pred_v3[["date", "code", "pred_net_return_v3"]], on=["date", "code"])
merged["rank_v2"] = merged.groupby("date")["pred_net_return_v2"].rank(pct=True)
merged["rank_v3"] = merged.groupby("date")["pred_net_return_v3"].rank(pct=True)
merged["ensemble_score"] = 0.5 * merged["rank_v2"] + 0.5 * merged["rank_v3"]
mask_ens = (merged.pred_net_return_v2 >= 0.003) & (merged.pred_net_return_v3 >= 0.003)
sig_ens = merged[mask_ens][["date", "code", "ensemble_score"]].rename(columns={"ensemble_score": "score"})
sig_ens = sig_ens[sig_ens.date.isin(allowed_tiered)]

test_cases = [
    # Model V2
    ("V2 Base (Budget 5k, Pos 4, Exp 65%)", sig_v2, Config(capital=30000.0, order_budget=5000.0, max_positions=4, max_exposure=0.65, hold_days=10, stop_loss=0.03)),
    ("V2 Opt-A (Budget 6k, Pos 4, Exp 75%)", sig_v2, Config(capital=30000.0, order_budget=6000.0, max_positions=4, max_exposure=0.75, hold_days=10, stop_loss=0.03)),
    ("V2 Opt-D (Budget 8k, Pos 4, Exp 90%)", sig_v2, Config(capital=30000.0, order_budget=8000.0, max_positions=4, max_exposure=0.90, hold_days=10, stop_loss=0.03)),
    ("V2 Opt-E (Budget 7k, Pos 5, Exp 90%)", sig_v2, Config(capital=30000.0, order_budget=7000.0, max_positions=5, max_exposure=0.90, hold_days=10, stop_loss=0.03)),

    # Model V3 (A-Share Features)
    ("V3 Base (Budget 5k, Pos 4, Exp 65%)", sig_v3, Config(capital=30000.0, order_budget=5000.0, max_positions=4, max_exposure=0.65, hold_days=10, stop_loss=0.03)),
    ("V3 Opt-A (Budget 6k, Pos 4, Exp 75%)", sig_v3, Config(capital=30000.0, order_budget=6000.0, max_positions=4, max_exposure=0.75, hold_days=10, stop_loss=0.03)),
    ("V3 Opt-D (Budget 8k, Pos 4, Exp 90%)", sig_v3, Config(capital=30000.0, order_budget=8000.0, max_positions=4, max_exposure=0.90, hold_days=10, stop_loss=0.03)),
    ("V3 Opt-E (Budget 7k, Pos 5, Exp 90%)", sig_v3, Config(capital=30000.0, order_budget=7000.0, max_positions=5, max_exposure=0.90, hold_days=10, stop_loss=0.03)),

    # Dual Ensemble
    ("Ens Base (Budget 5k, Pos 4, Exp 65%)", sig_ens, Config(capital=30000.0, order_budget=5000.0, max_positions=4, max_exposure=0.65, hold_days=10, stop_loss=0.03)),
    ("Ens Opt-A (Budget 6k, Pos 4, Exp 75%)", sig_ens, Config(capital=30000.0, order_budget=6000.0, max_positions=4, max_exposure=0.75, hold_days=10, stop_loss=0.03)),
    ("Ens Opt-D (Budget 8k, Pos 4, Exp 90%)", sig_ens, Config(capital=30000.0, order_budget=8000.0, max_positions=4, max_exposure=0.90, hold_days=10, stop_loss=0.03)),
    ("Ens Opt-E (Budget 7k, Pos 5, Exp 90%)", sig_ens, Config(capital=30000.0, order_budget=7000.0, max_positions=5, max_exposure=0.90, hold_days=10, stop_loss=0.03)),
]

print(f"{'Configuration':<40} | {'PureRet':<8} | {'MaxDD':<7} | {'Calmar':<7} | {'Sharpe':<7} | {'WR':<6} | {'Trades':<6}")
print("-" * 98)

results = []
for name, sig, cfg in test_cases:
    rep = SuperV5MaxProfitReplay(frames, period_days, sig, dividends, cfg, vol_map=vol_map, daily_limits=tiered_limits, enable_cash_overlay=False)
    rep.bars = bars
    rep.run()
    s = rep.summary()
    calmar = s["cagr"] / abs(s["max_drawdown"]) if s["max_drawdown"] != 0 else 0
    row = {
        "name": name,
        "total_return": s["total_return"],
        "max_drawdown": s["max_drawdown"],
        "calmar": calmar,
        "sharpe": s["sharpe"],
        "win_rate": s["win_rate"],
        "trades": s["closed_trades"],
        "profit_factor": s["profit_factor"],
    }
    results.append(row)
    print(f"{name:<40} | {s['total_return']:>7.2%} | {s['max_drawdown']:>6.2%} | {calmar:>6.2f} | {s['sharpe']:>6.2f} | {s['win_rate']:>5.1%} | {s['closed_trades']:>6}")

pd.DataFrame(results).to_csv("results/v7_combination_grid.csv", index=False)
print("\nGrid search complete. Results saved to results/v7_combination_grid.csv")
