import pandas as pd
from experiment_super_v7_regime import SuperV7RegimeReplay
from run import load
from validate_expanded import prepare_bars
from exploration_features import breadth_days
from engine import Config
from pathlib import Path

manifest, frames, calendar, dividends = load(Path("data/forward-2026-10-13"))
bars = prepare_bars(frames)
panel = pd.read_csv("results/midterm-access-v3/features.csv", dtype={"date": str, "code": str})
pred_v08 = pd.read_csv("results/model-v08/predictions-h10-v08.csv", dtype={"date": str, "code": str})
pred_v2 = pd.read_csv("results/model-v2/predictions-h10-v2.csv", dtype={"date": str, "code": str})
vol_map = panel.set_index(["date", "code"])["ret_vol20"].to_dict()

breadth_df = breadth_days(panel, return_col="adj_ret5")
b_map = breadth_df.set_index("date")["breadth5"].to_dict()
p_map = breadth_df.set_index("date")["pool_size"].to_dict()

tiered_limits = {d: (0 if p_map.get(d, 0) < 100 or b < 0.45 else (1 if b < 0.55 else 2)) for d, b in b_map.items()}
allowed_tiered = {d for d, lim in tiered_limits.items() if lim > 0}

merged = pd.merge(pred_v08[["date", "code", "rank_pct_v08", "pred_score_v08"]], pred_v2[["date", "code", "pred_net_return_v2"]], on=["date", "code"])
merged["rank_v2"] = merged.groupby("date")["pred_net_return_v2"].rank(pct=True)

# Keep exact V2 hurdle >= 0.005, but test different ranking scores:
ranking_cases = [
    # Baseline: V7 (pure V2 return score)
    ("Baseline V7 (Rank by V2 expected return)",
     merged[merged.pred_net_return_v2 >= 0.005].assign(score=lambda df: df.pred_net_return_v2)),

    # V0.8 Option 1: Rank purely by LambdaRank score
    ("V0.8 Option 1 (Hurdle V2>=0.005, Rank by Pure LambdaRank)",
     merged[merged.pred_net_return_v2 >= 0.005].assign(score=lambda df: df.pred_score_v08)),

    # V0.8 Option 2: Rank by 50% LambdaRank + 50% V2 rank
    ("V0.8 Option 2 (Hurdle V2>=0.005, Rank by 50% LambdaRank + 50% V2)",
     merged[merged.pred_net_return_v2 >= 0.005].assign(score=lambda df: 0.5 * df.rank_pct_v08 + 0.5 * df.rank_v2)),

    # V0.8 Option 3: Rank by 70% LambdaRank + 30% V2 rank
    ("V0.8 Option 3 (Hurdle V2>=0.005, Rank by 70% LambdaRank + 30% V2)",
     merged[merged.pred_net_return_v2 >= 0.005].assign(score=lambda df: 0.7 * df.rank_pct_v08 + 0.3 * df.rank_v2)),

    # V0.8 Option 4: Rank by 30% LambdaRank + 70% V2 rank
    ("V0.8 Option 4 (Hurdle V2>=0.005, Rank by 30% LambdaRank + 70% V2)",
     merged[merged.pred_net_return_v2 >= 0.005].assign(score=lambda df: 0.3 * df.rank_pct_v08 + 0.7 * df.rank_v2)),
]

period_days = [d for d in calendar if "2022-01-01" <= d <= "2025-12-31"]
cfg_regime = Config(capital=30000.0, order_budget=7500.0, max_positions=5, max_exposure=0.90, hold_days=10, stop_loss=0.03)

print("=" * 115)
print(f"{'Ranking Key Case':<52} | {'PureRet':<8} | {'WithCash':<8} | {'MaxDD':<7} | {'Calmar':<7} | {'Sharpe':<7} | {'Trades':<6}")
print("=" * 115)

for name, sig_df in ranking_cases:
    sig = sig_df[["date", "code", "score"]].copy()
    sig = sig[sig.date.isin(allowed_tiered)]
    
    rep_pure = SuperV7RegimeReplay(frames, period_days, sig, dividends, cfg_regime, vol_map=vol_map, daily_limits=tiered_limits, breadth_lookup=b_map, enable_cash_overlay=False)
    rep_pure.bars = bars
    rep_pure.run()
    s_pure = rep_pure.summary()

    rep_cash = SuperV7RegimeReplay(frames, period_days, sig, dividends, cfg_regime, vol_map=vol_map, daily_limits=tiered_limits, breadth_lookup=b_map, enable_cash_overlay=True)
    rep_cash.bars = bars
    rep_cash.run()
    s_cash = rep_cash.summary()
    
    calmar = s_cash["cagr"] / abs(s_cash["max_drawdown"]) if s_cash["max_drawdown"] != 0 else 0
    print(f"{name:<52} | {s_pure['total_return']:>7.2%} | {s_cash['total_return']:>7.2%} | {s_cash['max_drawdown']:>6.2%} | {calmar:>6.2f} | {s_cash['sharpe']:>6.2f} | {s_cash['closed_trades']:>6}")

    cal_26 = [d for d in calendar if "2026-01-01" <= d <= "2026-10-13"]
    sig_26 = sig[sig.date.isin(set(cal_26))]
    r26 = SuperV7RegimeReplay(frames, cal_26, sig_26, dividends, cfg_regime, vol_map=vol_map, daily_limits=tiered_limits, breadth_lookup=b_map, enable_cash_overlay=True)
    r26.bars = bars
    r26.run()
    s26 = r26.summary()
    print(f"  --> 2026 Blind: Return={s26['total_return']:>6.2%}, MaxDD={s26['max_drawdown']:>6.2%}, Trades={s26['closed_trades']}")
    print("-" * 115)
