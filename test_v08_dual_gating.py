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
pred_v3 = pd.read_csv("results/model-v3/predictions-h10-v3.csv", dtype={"date": str, "code": str})
vol_map = panel.set_index(["date", "code"])["ret_vol20"].to_dict()

breadth_df = breadth_days(panel, return_col="adj_ret5")
b_map = breadth_df.set_index("date")["breadth5"].to_dict()
p_map = breadth_df.set_index("date")["pool_size"].to_dict()

tiered_limits = {d: (0 if p_map.get(d, 0) < 100 or b < 0.45 else (1 if b < 0.55 else 2)) for d, b in b_map.items()}
allowed_tiered = {d for d, lim in tiered_limits.items() if lim > 0}

# Merge LambdaRank V0.8 with Model V2/V3
merged = pd.merge(pred_v08[["date", "code", "rank_pct_v08", "pred_score_v08"]], pred_v2[["date", "code", "pred_net_return_v2"]], on=["date", "code"])
merged = pd.merge(merged, pred_v3[["date", "code", "pred_net_return_v3"]], on=["date", "code"])

# Dual Gating:
# 1. Absolute positive hurdle: pred_return >= 0.005
# 2. Score = 0.5 * LambdaRank_pct + 0.5 * Model_V2/V3_rank
merged["rank_v2"] = merged.groupby("date")["pred_net_return_v2"].rank(pct=True)
merged["rank_v3"] = merged.groupby("date")["pred_net_return_v3"].rank(pct=True)

# Test different combinations of gating
gating_cases = [
    # Case 1: LambdaRank filtering on V2 positive hurdle
    ("V0.8 Dual-Gated (V2 >= 0.005 & RankV08 >= 0.85)", 
     merged[(merged.pred_net_return_v2 >= 0.005) & (merged.rank_pct_v08 >= 0.85)].assign(score=lambda df: df.rank_pct_v08)),

    # Case 2: LambdaRank filtering on V2 >= 0.005, sorted by V0.8 + V2 composite
    ("V0.8 Composite (V2 >= 0.005, Score = 0.6*V08 + 0.4*V2)", 
     merged[merged.pred_net_return_v2 >= 0.005].assign(score=lambda df: 0.6 * df.rank_pct_v08 + 0.4 * df.rank_v2)),

    # Case 3: V2+V3 positive hurdle, sorted by LambdaRank V0.8
    ("V0.8 Tri-Ensemble (V2>=0.003 & V3>=0.003, Score = 0.5*V08 + 0.25*V2 + 0.25*V3)",
     merged[(merged.pred_net_return_v2 >= 0.003) & (merged.pred_net_return_v3 >= 0.003)].assign(score=lambda df: 0.5 * df.rank_pct_v08 + 0.25 * df.rank_v2 + 0.25 * df.rank_v3)),

    # Case 4: Pure LambdaRank Top 2% with positive return hurdle
    ("V0.8 Top 2% LambdaRank with Hurdle (V2 >= 0.003 & V08 >= 0.95)",
     merged[(merged.pred_net_return_v2 >= 0.003) & (merged.rank_pct_v08 >= 0.95)].assign(score=lambda df: df.rank_pct_v08)),
]

period_days = [d for d in calendar if "2022-01-01" <= d <= "2025-12-31"]
cfg_regime = Config(capital=30000.0, order_budget=7500.0, max_positions=5, max_exposure=0.90, hold_days=10, stop_loss=0.03)

print("=" * 115)
print(f"{'V0.8 Candidate Configuration':<52} | {'PureRet':<8} | {'WithCash':<8} | {'MaxDD':<7} | {'Calmar':<7} | {'Sharpe':<7} | {'Trades':<6}")
print("=" * 115)

for name, sig_df in gating_cases:
    sig = sig_df[["date", "code", "score"]].copy()
    sig = sig[sig.date.isin(allowed_tiered)]
    
    # Pure
    rep_pure = SuperV7RegimeReplay(frames, period_days, sig, dividends, cfg_regime, vol_map=vol_map, daily_limits=tiered_limits, breadth_lookup=b_map, enable_cash_overlay=False)
    rep_pure.bars = bars
    rep_pure.run()
    s_pure = rep_pure.summary()

    # Cash
    rep_cash = SuperV7RegimeReplay(frames, period_days, sig, dividends, cfg_regime, vol_map=vol_map, daily_limits=tiered_limits, breadth_lookup=b_map, enable_cash_overlay=True)
    rep_cash.bars = bars
    rep_cash.run()
    s_cash = rep_cash.summary()
    
    calmar = s_cash["cagr"] / abs(s_cash["max_drawdown"]) if s_cash["max_drawdown"] != 0 else 0
    print(f"{name:<52} | {s_pure['total_return']:>7.2%} | {s_cash['total_return']:>7.2%} | {s_cash['max_drawdown']:>6.2%} | {calmar:>6.2f} | {s_cash['sharpe']:>6.2f} | {s_cash['closed_trades']:>6}")

    # Also test 2026 blind
    cal_26 = [d for d in calendar if "2026-01-01" <= d <= "2026-10-13"]
    sig_26 = sig[sig.date.isin(set(cal_26))]
    r26 = SuperV7RegimeReplay(frames, cal_26, sig_26, dividends, cfg_regime, vol_map=vol_map, daily_limits=tiered_limits, breadth_lookup=b_map, enable_cash_overlay=True)
    r26.bars = bars
    r26.run()
    s26 = r26.summary()
    print(f"  --> 2026 Blind: Return={s26['total_return']:>6.2%}, MaxDD={s26['max_drawdown']:>6.2%}, Trades={s26['closed_trades']}")
    print("-" * 115)
