"""Comprehensive Risk, Metrics, Stress Test, and Yearly Breakdown.

Evaluates the new Super-Strategy:
'Model V2 + Breadth Gate + TP10% + Multi-buy (2/day)' vs 'Champion Baseline'

Computes:
1. CAGR, Sharpe Ratio, Sortino Ratio, Calmar Ratio
2. Yearly independent performance (2022, 2023, 2024, 2025, 2026)
3. Double Slippage Stress Test (0.10% one-way slippage)
4. Liquidity & Trading Friction Drag
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import numpy as np

from accessibility import AccessibleReplay
from engine import Config, tick
from exploration_features import breadth_days
from run import load
from validate_expanded import prepare_bars

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data/forward-2026-10-13"
DONOR = ROOT / "results/midterm-access-v3"
MODEL_V2_DIR = ROOT / "results/model-v2"


class MetricsReplay(AccessibleReplay):
    def __init__(self, frames, calendar, signals, dividends=None, config=None,
                 take_profit: float | None = None,
                 trailing_trigger: float | None = None,
                 trailing_delta: float | None = None,
                 max_daily_buys: int = 1):
        super().__init__(frames, calendar, signals, dividends, config)
        self.take_profit = take_profit
        self.trailing_trigger = trailing_trigger
        self.trailing_delta = trailing_delta
        self.max_daily_buys = max_daily_buys
        self.peak_prices: dict[int, float] = {}

    def run(self):
        for index, day in enumerate(self.calendar):
            self.corporate_actions(day)

            for code, position in list(self.positions.items()):
                bar = self.bars[code].get(day)
                if bar is not None and code in self.marks:
                    reference = float(bar.get("preclose", 0.0) or 0.0)
                    if reference > 0 and abs(reference - self.marks[code]) > 0.025:
                        known = [e for e in self.actions_by_ex.get(day, []) if e.get("code") == code]
                        if not known:
                            self.warnings.append({"date": day, "code": code, "issue": "unexplained_price_adjustment"})
                elif bar is None:
                    self.warnings.append({"date": day, "code": code, "issue": "missing_held_stock_quote"})

                if index - position.entry_index >= self.config.hold_days or position.stop_pending:
                    self.sell(position, day, index)

            for code in self.positions:
                bar = self.bars[code].get(day)
                if bar is not None and float(bar.get("open", 0.0) or 0.0) > 0:
                    self.marks[code] = float(bar["open"])

            for order in self.pending_buys:
                self.buy(order, day, index)
            self.pending_buys = []

            for code, position in self.positions.items():
                bar = self.bars[code].get(day)
                if bar is not None:
                    close_px = float(bar["close"])
                    self.marks[code] = close_px

                    current_peak = self.peak_prices.get(position.trade_id, position.entry_notional / position.shares)
                    if close_px > current_peak:
                        self.peak_prices[position.trade_id] = close_px
                        current_peak = close_px

                    entry_px = position.entry_notional / position.shares
                    cur_gain = (close_px - entry_px) / entry_px

                    value = position.shares * close_px + position.proceeds + position.dividends
                    if self.config.stop_loss and value < position.entry_notional * (1 - self.config.stop_loss):
                        position.stop_pending = True

                    if self.take_profit is not None and cur_gain >= self.take_profit:
                        position.stop_pending = True

                    if self.trailing_trigger is not None and self.trailing_delta is not None:
                        peak_gain = (current_peak - entry_px) / entry_px
                        if peak_gain >= self.trailing_trigger:
                            if (current_peak - close_px) / current_peak >= self.trailing_delta:
                                position.stop_pending = True

            for event in self.actions_by_record.get(day, []):
                position = self.positions.get(event.get("code"))
                if position and position.shares:
                    position.pending_actions += 1
                    self.entitlements.append({"event": event, "position": position,
                                              "shares": position.shares, "posted": False})

            nav = self.nav()
            self.equity.append({"date": day, "equity": nav, "cash": self.cash,
                                "receivables": sum(item[1] for item in self.receivables),
                                "positions": len(self.positions)})
            if self.cash < -1e-7 or any(p.shares < 0 for p in self.positions.values()):
                raise AssertionError("Negative cash or shares")

            candidates = sorted(self.candidates.get(day, []), key=lambda row: (-row["score"], row["code"]))
            queued = 0
            available_slots = max(0, self.config.max_positions - len(self.positions))
            limit_buys = min(self.max_daily_buys, available_slots)

            for candidate in candidates:
                if queued >= limit_buys:
                    break
                code = candidate["code"]
                if code in self.positions or any(o["code"] == code for o in self.pending_buys):
                    continue
                bar = self.bars[code].get(day)
                if bar is None:
                    continue
                limit = tick(float(bar["close"]) * (1 + self.config.max_buy_gap))
                budget = min(self.config.order_budget, self.cash)
                unit_cost = limit * (1 + self.config.commission + self.config.transfer_fee + self.config.slippage)
                quantity = int(max(0, budget - self.config.min_commission) / unit_cost) // 100 * 100
                if quantity < 100:
                    self.reject("lot_size_or_budget")
                    continue
                self.pending_buys.append({"code": code, "signal_date": day, "shares": quantity, "limit": limit})
                queued += 1

        return self


def compute_advanced_metrics(equity_series: pd.Series, rf_annual: float = 0.02) -> dict[str, float]:
    """Calculate CAGR, Sharpe, Sortino, Calmar ratios from daily equity curve."""
    daily_returns = equity_series.pct_change().dropna()
    if daily_returns.empty:
        return {}

    n_days = len(daily_returns)
    years = n_days / 244.0  # ~244 trading days per year in A-shares

    total_ret = equity_series.iloc[-1] / equity_series.iloc[0] - 1
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 and total_ret > -1 else 0.0

    # Drawdown
    cummax = equity_series.cummax()
    dd = (equity_series - cummax) / cummax
    max_dd = float(dd.min())

    # Volatility & Downside Deviation
    ann_vol = float(daily_returns.std() * np.sqrt(244))
    downside = daily_returns[daily_returns < 0]
    downside_vol = float(downside.std() * np.sqrt(244)) if len(downside) > 1 else ann_vol

    # Sharpe & Sortino
    rf_daily = rf_annual / 244.0
    excess_mean = float((daily_returns - rf_daily).mean() * 244)
    sharpe = excess_mean / ann_vol if ann_vol > 0 else 0.0
    sortino = excess_mean / downside_vol if downside_vol > 0 else 0.0

    # Calmar
    calmar = abs(cagr / max_dd) if max_dd < 0 else 0.0

    return {
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "total_return": total_ret,
    }


def main():
    print("Loading data for advanced metrics and stress testing...")
    manifest, frames, calendar, dividends = load(DATASET)
    bars = prepare_bars(frames)

    panel = pd.read_csv(DONOR / "features.csv", dtype={"date": str, "code": str})
    pred_base = pd.read_csv(DONOR / "predictions-h10.csv", dtype={"date": str, "code": str})
    pred_v2 = pd.read_csv(MODEL_V2_DIR / "predictions-h10-v2.csv", dtype={"date": str, "code": str})

    breadth_df = breadth_days(panel, return_col="adj_ret5")
    allowed_dates = set(breadth_df.loc[(breadth_df.pool_size >= 100) & (breadth_df.breadth5 >= 0.50), "date"])

    sig_base = pred_base[pred_base.pred_net_return >= 0.005][["date", "code", "pred_net_return"]].rename(columns={"pred_net_return": "score"})
    sig_base = sig_base[sig_base.date.isin(allowed_dates)]

    sig_v2 = pred_v2[pred_v2.pred_net_return_v2 >= 0.005][["date", "code", "pred_net_return_v2"]].rename(columns={"pred_net_return_v2": "score"})
    sig_v2 = sig_v2[sig_v2.date.isin(allowed_dates)]

    strategies = [
        ("Champion Baseline (Fixed 10d, 1buy/d)", sig_base, None, 1, 0.0005, 3, 0.55),
        ("Baseline + Double Slippage (Stress)", sig_base, None, 1, 0.0010, 3, 0.55),
        ("Super-V2: TP10% + 2Buys/d", sig_v2, 0.10, 2, 0.0005, 4, 0.65),
        ("Super-V2 + Double Slippage (Stress)", sig_v2, 0.10, 2, 0.0010, 4, 0.65),
        ("Super-V2 + Trail(+5%/2%) + 2Buys/d", sig_v2, 0.10, 2, 0.0005, 4, 0.65, 0.05, 0.02),
        ("Super-V2 + Trail + Double Slippage (Stress)", sig_v2, 0.10, 2, 0.0010, 4, 0.65, 0.05, 0.02),
    ]

    periods = [
        ("2022-2025 (4yr Continuous)", "2022-01-01", "2025-12-31"),
        ("2026 (Out-of-Sample Forward)", "2026-01-01", "2026-10-13"),
    ]

    all_metrics = []

    print("\n" + "=" * 105)
    print(f"{'Period':<14} {'Strategy':<38} {'TotalRet':<9} {'CAGR':<8} {'MaxDD':<8} {'Sharpe':<7} {'Sortino':<8} {'Calmar':<7} {'PF':<6}")
    print("=" * 105)

    for p_name, start, end in periods:
        p_days = [d for d in calendar if start <= d <= end]

        for item in strategies:
            strat_name, sigs_all, tp, buys, slip, max_p, max_e = item[:7]
            t_trig = item[7] if len(item) > 7 else None
            t_delta = item[8] if len(item) > 8 else None

            p_sigs = sigs_all[sigs_all.date.between(start, end)].copy()
            cfg = Config(
                capital=30000.0,
                order_budget=5000.0,
                max_positions=max_p,
                max_exposure=max_e,
                hold_days=10,
                stop_loss=0.03,
                slippage=slip,
            )
            replay = MetricsReplay(frames, p_days, p_sigs, dividends, cfg,
                                   take_profit=tp, trailing_trigger=t_trig,
                                   trailing_delta=t_delta, max_daily_buys=buys)
            replay.bars = bars
            replay.run()
            summary = replay.summary()

            eq_series = pd.DataFrame(replay.equity)["equity"]
            adv = compute_advanced_metrics(eq_series)

            pf = summary["profit_factor"] if summary["profit_factor"] is not None else 0.0
            win = summary["win_rate"] if summary["win_rate"] is not None else 0.0

            all_metrics.append({
                "period": p_name,
                "strategy": strat_name,
                "total_return": summary["total_return"],
                "cagr": adv.get("cagr", 0.0),
                "max_drawdown": summary["max_drawdown"],
                "sharpe": adv.get("sharpe", 0.0),
                "sortino": adv.get("sortino", 0.0),
                "calmar": adv.get("calmar", 0.0),
                "profit_factor": pf,
                "win_rate": win,
                "trades": summary["closed_trades"],
                "costs": summary["execution_costs"],
            })

            print(f"{p_name:<14} {strat_name:<38} {summary['total_return']:>8.2%} "
                  f"{adv.get('cagr', 0.0):>7.2%} {summary['max_drawdown']:>7.2%} "
                  f"{adv.get('sharpe', 0.0):>6.2f} {adv.get('sortino', 0.0):>7.2f} "
                  f"{adv.get('calmar', 0.0):>6.2f} {pf:>5.2f}")

    # Detailed Year-by-Year breakdown for Super-V2 vs Baseline
    print("\n" + "=" * 80)
    print("Detailed Year-by-Year Independent Account Returns (30,000 RMB Fresh Capital)")
    print("=" * 80)
    print(f"{'Year':<8} {'Baseline Return':<18} {'Super-V2 Return':<18} {'Alpha (Diff)':<14} {'Super-V2 MaxDD':<14}")
    print("-" * 80)

    yearly_records = []
    years = [2022, 2023, 2024, 2025, 2026]

    for yr in years:
        y_start = f"{yr}-01-01"
        y_end = f"{yr}-12-31" if yr < 2026 else "2026-10-13"
        y_days = [d for d in calendar if y_start <= d <= y_end]

        # 1. Baseline
        b_sigs = sig_base[sig_base.date.between(y_start, y_end)].copy()
        b_cfg = Config(capital=30000.0, order_budget=5000.0, max_positions=3, max_exposure=0.55, hold_days=10, stop_loss=0.03)
        b_rep = MetricsReplay(frames, y_days, b_sigs, dividends, b_cfg, max_daily_buys=1)
        b_rep.bars = bars
        b_rep.run()
        b_sum = b_rep.summary()

        # 2. Super-V2
        s_sigs = sig_v2[sig_v2.date.between(y_start, y_end)].copy()
        s_cfg = Config(capital=30000.0, order_budget=5000.0, max_positions=4, max_exposure=0.65, hold_days=10, stop_loss=0.03)
        s_rep = MetricsReplay(frames, y_days, s_sigs, dividends, s_cfg, take_profit=0.10, trailing_trigger=0.05, trailing_delta=0.02, max_daily_buys=2)
        s_rep.bars = bars
        s_rep.run()
        s_sum = s_rep.summary()

        diff = s_sum["total_return"] - b_sum["total_return"]
        print(f"{yr:<8} {b_sum['total_return']:>16.2%} {s_sum['total_return']:>16.2%} {diff:>+13.2%} {s_sum['max_drawdown']:>13.2%}")

        yearly_records.append({
            "year": yr,
            "baseline_return": b_sum["total_return"],
            "baseline_max_dd": b_sum["max_drawdown"],
            "baseline_trades": b_sum["closed_trades"],
            "super_v2_return": s_sum["total_return"],
            "super_v2_max_dd": s_sum["max_drawdown"],
            "super_v2_trades": s_sum["closed_trades"],
            "alpha": diff,
        })

    out_dir = ROOT / "results/stress-and-metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_metrics).to_csv(out_dir / "metrics_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(yearly_records).to_csv(out_dir / "yearly_breakdown.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved metrics to: {out_dir}")


if __name__ == "__main__":
    main()
