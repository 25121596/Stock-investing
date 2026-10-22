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

configs = [
    ("Super-V7 Scaled (Budget 6k, Pos 4, Exp 75%)", Config(capital=30000.0, order_budget=6000.0, max_positions=4, max_exposure=0.75, hold_days=10, stop_loss=0.03)),
    ("Super-V7 High-Exposure (Budget 7k, Pos 5, Exp 90%)", Config(capital=30000.0, order_budget=7000.0, max_positions=5, max_exposure=0.90, hold_days=10, stop_loss=0.03)),
]

for name, cfg in configs:
    print(f"\n{'='*35} {name} {'='*35}")
    for yr in [2022, 2023, 2024, 2025, 2026]:
        end_d = f"{yr}-12-31" if yr < 2026 else "2026-10-13"
        p_yr = [d for d in calendar if f"{yr}-01-01" <= d <= end_d]
        sig_yr = sig_v5[sig_v5.date.between(f"{yr}-01-01", end_d)].copy()
        r_yr = SuperV5MaxProfitReplay(frames, p_yr, sig_yr, dividends, cfg, vol_map=vol_map, daily_limits=tiered_limits, enable_cash_overlay=True)
        r_yr.bars = bars
        r_yr.run()
        sm = r_yr.summary()
        calmar = sm["cagr"] / abs(sm["max_drawdown"]) if sm["max_drawdown"] != 0 else 0
        print(f"Year {yr}: Return={sm['total_return']:>7.2%}, MaxDD={sm['max_drawdown']:>6.2%}, Calmar={calmar:>5.2f}, Trades={sm['closed_trades']}")
