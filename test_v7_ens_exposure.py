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

merged = pd.merge(pred_v2[["date", "code", "pred_net_return_v2"]], pred_v3[["date", "code", "pred_net_return_v3"]], on=["date", "code"])
merged["rank_v2"] = merged.groupby("date")["pred_net_return_v2"].rank(pct=True)
merged["rank_v3"] = merged.groupby("date")["pred_net_return_v3"].rank(pct=True)
merged["score"] = 0.5 * merged["rank_v2"] + 0.5 * merged["rank_v3"]
mask_ens = (merged.pred_net_return_v2 >= 0.003) & (merged.pred_net_return_v3 >= 0.003)
sig_ens = merged[mask_ens][["date", "code", "score"]]
sig_ens = sig_ens[sig_ens.date.isin(allowed_tiered)]

period_days = [d for d in calendar if "2022-01-01" <= d <= "2025-12-31"]

cfg_v7_ens = Config(capital=30000.0, order_budget=7000.0, max_positions=5, max_exposure=0.90, hold_days=10, stop_loss=0.03)

print("=== DUAL ENSEMBLE (MODEL V2 + V3) WITH V7 HIGH-EXPOSURE (BUDGET 7K, EXP 90%) ===")
for cash in [False, True]:
    rep = SuperV5MaxProfitReplay(frames, period_days, sig_ens, dividends, cfg_v7_ens, vol_map=vol_map, daily_limits=tiered_limits, enable_cash_overlay=cash)
    rep.bars = bars
    rep.run()
    s = rep.summary()
    calmar = s["cagr"] / abs(s["max_drawdown"]) if s["max_drawdown"] != 0 else 0
    cash_str = "With 2.2% Cash" if cash else "Pure Trading Alpha"
    print(f"Continuous 2022-2025 ({cash_str}): Return={s['total_return']:>7.2%}, CAGR={s['cagr']:>6.2%}, MaxDD={s['max_drawdown']:>6.2%}, Calmar={calmar:>5.2f}, Sharpe={s['sharpe']:>5.2f}, Trades={s['closed_trades']}")

print("\nYearly Breakdown (With Cash Overlay):")
for yr in [2022, 2023, 2024, 2025, 2026]:
    end_d = f"{yr}-12-31" if yr < 2026 else "2026-10-13"
    p_yr = [d for d in calendar if f"{yr}-01-01" <= d <= end_d]
    sig_yr = sig_ens[sig_ens.date.between(f"{yr}-01-01", end_d)].copy()
    r_yr = SuperV5MaxProfitReplay(frames, p_yr, sig_yr, dividends, cfg_v7_ens, vol_map=vol_map, daily_limits=tiered_limits, enable_cash_overlay=True)
    r_yr.bars = bars
    r_yr.run()
    sm = r_yr.summary()
    calmar = sm["cagr"] / abs(sm["max_drawdown"]) if sm["max_drawdown"] != 0 else 0
    print(f"Year {yr}: Return={sm['total_return']:>7.2%}, MaxDD={sm['max_drawdown']:>6.2%}, Calmar={calmar:>5.2f}, Trades={sm['closed_trades']}")
