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
sig_v5 = pred_v2[pred_v2.pred_net_return_v2 >= 0.005][["date", "code", "pred_net_return_v2"]].rename(columns={"pred_net_return_v2": "score"})
sig_v5 = sig_v5[sig_v5.date.isin(allowed_tiered)]

period_days = [d for d in calendar if "2022-01-01" <= d <= "2025-12-31"]

configs = [
    ("V5 Base (Budget 5k, Pos 4, Exp 65%)", Config(capital=30000.0, order_budget=5000.0, max_positions=4, max_exposure=0.65, hold_days=10, stop_loss=0.03)),
    ("Opt A (Budget 6k, Pos 4, Exp 75%)",   Config(capital=30000.0, order_budget=6000.0, max_positions=4, max_exposure=0.75, hold_days=10, stop_loss=0.03)),
    ("Opt B (Budget 6k, Pos 5, Exp 80%)",   Config(capital=30000.0, order_budget=6000.0, max_positions=5, max_exposure=0.80, hold_days=10, stop_loss=0.03)),
    ("Opt C (Budget 7.5k, Pos 4, Exp 85%)", Config(capital=30000.0, order_budget=7500.0, max_positions=4, max_exposure=0.85, hold_days=10, stop_loss=0.03)),
    ("Opt D (Budget 8k, Pos 4, Exp 90%)",   Config(capital=30000.0, order_budget=8000.0, max_positions=4, max_exposure=0.90, hold_days=10, stop_loss=0.03)),
    ("Opt E (Budget 7k, Pos 5, Exp 90%)",   Config(capital=30000.0, order_budget=7000.0, max_positions=5, max_exposure=0.90, hold_days=10, stop_loss=0.03)),
]

print(f"{'Config':<38} | {'PureRet':<8} | {'MaxDD':<7} | {'Calmar':<7} | {'Sharpe':<7} | {'WR':<6} | {'Trades':<6}")
print("-" * 95)
for name, cfg in configs:
    rep = SuperV5MaxProfitReplay(frames, period_days, sig_v5, dividends, cfg, vol_map=vol_map, daily_limits=tiered_limits, enable_cash_overlay=False)
    rep.bars = bars
    rep.run()
    s = rep.summary()
    calmar = s["cagr"] / abs(s["max_drawdown"]) if s["max_drawdown"] != 0 else 0
    ret_str = f"{s['total_return']:.2%}"
    dd_str = f"{s['max_drawdown']:.2%}"
    wr_str = f"{s['win_rate']:.1%}"
    print(f"{name:<38} | {ret_str:>8} | {dd_str:>7} | {calmar:>7.2f} | {s['sharpe']:>7.2f} | {wr_str:>6} | {s['closed_trades']:>6}")
