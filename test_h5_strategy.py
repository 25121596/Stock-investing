import pandas as pd
from experiment_super_v5_maxprofit import SuperV5MaxProfitReplay, DATASET, DONOR, MODEL_V2_DIR
from run import load
from validate_expanded import prepare_bars
from exploration_features import breadth_days
from engine import Config

manifest, frames, calendar, dividends = load(DATASET)
bars = prepare_bars(frames)
panel = pd.read_csv(DONOR / "features.csv", dtype={"date": str, "code": str})
pred_h5 = pd.read_csv("results/midterm-access-v3/predictions-h5.csv", dtype={"date": str, "code": str})
vol_map = panel.set_index(["date", "code"])["ret_vol20"].to_dict()

breadth_df = breadth_days(panel, return_col="adj_ret5")
b_map = breadth_df.set_index("date")["breadth5"].to_dict()
p_map = breadth_df.set_index("date")["pool_size"].to_dict()

tiered_limits = {d: (0 if p_map.get(d, 0) < 100 or b < 0.45 else (1 if b < 0.55 else 2)) for d, b in b_map.items()}
allowed_tiered = {d for d, lim in tiered_limits.items() if lim > 0}

period_days = [d for d in calendar if "2022-01-01" <= d <= "2025-12-31"]

sig_h5 = pred_h5[pred_h5.pred_net_return >= 0.005][["date", "code", "pred_net_return"]].rename(columns={"pred_net_return": "score"})
sig_h5 = sig_h5[sig_h5.date.isin(allowed_tiered)]

print(f"H5 signals count: {len(sig_h5)}")

# Test H5 with hold_days=5
cfg_h5 = Config(capital=30000.0, order_budget=6000.0, max_positions=4, max_exposure=0.75, hold_days=5, stop_loss=0.03)

rep = SuperV5MaxProfitReplay(frames, period_days, sig_h5, dividends, cfg_h5, vol_map=vol_map, daily_limits=tiered_limits, enable_cash_overlay=False)
rep.bars = bars
rep.run()
s = rep.summary()
calmar = s["cagr"] / abs(s["max_drawdown"]) if s["max_drawdown"] != 0 else 0
print(f"H5 Pure Trading Return: {s['total_return']:.2%}")
print(f"H5 Max Drawdown:        {s['max_drawdown']:.2%}")
print(f"H5 Calmar:              {calmar:.2f}")
print(f"H5 Sharpe:              {s['sharpe']:.2f}")
print(f"H5 Win Rate:            {s['win_rate']:.1%}")
print(f"H5 Trades:              {s['closed_trades']}")
