"""Research Script: Testing Portfolio Capacity, Concurrency, and Exposure Optimization.

Examines the impact of:
1. max_daily_buys: 1 (baseline) vs up to available slots (multi-buy on high breadth days)
2. max_positions: 3 (baseline) vs 4 vs 5
3. max_exposure: 0.50 (baseline) vs 0.70 vs 0.85
4. order_budget: 5000 (baseline) vs 6000 vs dynamic (nav / max_positions)

Tests against the champion breadth_lgb10 strategy on:
- 2022-2025 (in-sample 4-year continuous)
- 2026 (out-of-sample forward blind test)
"""
from __future__ import annotations

from dataclasses import replace
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


class FlexibleReplay(AccessibleReplay):
    """Replay engine with configurable daily buy concurrency and dynamic sizing."""

    def __init__(self, frames, calendar, signals, dividends=None, config=None,
                 max_daily_buys: int = 1, dynamic_sizing: bool = False):
        super().__init__(frames, calendar, signals, dividends, config)
        self.max_daily_buys = max_daily_buys
        self.dynamic_sizing = dynamic_sizing

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
                    self.marks[code] = float(bar["close"])
                value = position.shares * self.marks[code] + position.proceeds + position.dividends
                if self.config.stop_loss and value < position.entry_notional * (1 - self.config.stop_loss):
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

            # Selection logic with max_daily_buys support
            candidates = sorted(self.candidates.get(day, []), key=lambda row: (-row["score"], row["code"]))
            queued = 0
            # How many positions could we realistically add?
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

                if self.dynamic_sizing:
                    # Allocate nav / max_positions, bounded by cash
                    target_budget = (nav * self.config.max_exposure) / self.config.max_positions
                    budget = min(target_budget, self.cash)
                else:
                    budget = min(self.config.order_budget, self.cash)

                unit_cost = limit * (1 + self.config.commission + self.config.transfer_fee + self.config.slippage)
                quantity = int(max(0, budget - self.config.min_commission) / unit_cost) // 100 * 100
                if quantity < 100:
                    self.reject("lot_size_or_budget")
                    continue
                self.pending_buys.append({"code": code, "signal_date": day, "shares": quantity, "limit": limit})
                queued += 1

        return self


def main():
    print("Loading datasets...")
    manifest, frames, calendar, dividends = load(DATASET)
    bars = prepare_bars(frames)

    panel = pd.read_csv(DONOR / "features.csv", dtype={"date": str, "code": str})
    predictions = pd.read_csv(DONOR / "predictions-h10.csv", dtype={"date": str, "code": str})

    # Prepare champion breadth signals
    chosen = predictions[predictions.pred_net_return >= 0.005][["date", "code", "pred_net_return"]].copy()
    chosen = chosen.rename(columns={"pred_net_return": "score"})

    breadth_df = breadth_days(panel, return_col="adj_ret5")
    allowed_dates = set(breadth_df.loc[(breadth_df.pool_size >= 100) & (breadth_df.breadth5 >= 0.50), "date"])
    gated_signals = chosen[chosen.date.isin(allowed_dates)].copy()

    periods = [
        ("2022-2025", "2022-01-01", "2025-12-31"),
        ("2026", "2026-01-01", "2026-10-13"),
    ]

    scenarios = [
        # (name, max_pos, max_exp, budget, max_daily_buys, dynamic_sizing)
        ("Baseline (3pos, 50%exp, 1buy/d)", 3, 0.50, 5000.0, 1, False),
        ("Fix Exposure Bug (3pos, 55%exp, 1buy/d)", 3, 0.55, 5000.0, 1, False),
        ("Multi-buy (3pos, 55%exp, 2buy/d)", 3, 0.55, 5000.0, 2, False),
        ("Multi-buy (3pos, 55%exp, 3buy/d)", 3, 0.55, 5000.0, 3, False),
        ("Expand Capacity (4pos, 70%exp, 5000, 2buy/d)", 4, 0.70, 5000.0, 2, False),
        ("Expand Capacity (5pos, 80%exp, 5000, 2buy/d)", 5, 0.80, 5000.0, 2, False),
        ("Expand Capacity (5pos, 80%exp, 5000, 3buy/d)", 5, 0.80, 5000.0, 3, False),
        ("Dynamic Sizing (4pos, 70%exp, 2buy/d)", 4, 0.70, 5000.0, 2, True),
        ("Dynamic Sizing (5pos, 80%exp, 3buy/d)", 5, 0.80, 5000.0, 3, True),
    ]

    records = []
    print("\n" + "=" * 95)
    print(f"{'Period':<12} {'Scenario':<42} {'Return':<10} {'MaxDD':<10} {'WinRate':<9} {'Trades':<7} {'PF':<7} {'AvgExp':<8}")
    print("=" * 95)

    for period_name, start, end in periods:
        period_days = [d for d in calendar if start <= d <= end]
        sigs = gated_signals[gated_signals.date.between(start, end)].copy()

        for name, max_p, max_e, budget, daily_buys, dyn in scenarios:
            cfg = Config(
                capital=30000.0,
                order_budget=budget,
                max_positions=max_p,
                max_exposure=max_e,
                hold_days=10,
                stop_loss=0.03,
            )
            replay = FlexibleReplay(frames, period_days, sigs, dividends, cfg,
                                    max_daily_buys=daily_buys, dynamic_sizing=dyn)
            replay.bars = bars
            replay.run()
            summary = replay.summary()

            eq_df = pd.DataFrame(replay.equity)
            avg_exposure = float(((eq_df.equity - eq_df.cash - eq_df.receivables) / eq_df.equity).mean())

            win_rate = summary["win_rate"] if summary["win_rate"] is not None else 0.0
            pf = summary["profit_factor"] if summary["profit_factor"] is not None else 0.0

            records.append({
                "period": period_name,
                "scenario": name,
                "max_positions": max_p,
                "max_exposure": max_e,
                "max_daily_buys": daily_buys,
                "dynamic_sizing": dyn,
                "total_return": summary["total_return"],
                "max_drawdown": summary["max_drawdown"],
                "win_rate": win_rate,
                "closed_trades": summary["closed_trades"],
                "profit_factor": pf,
                "avg_exposure": avg_exposure,
                "execution_costs": summary["execution_costs"],
            })

            print(f"{period_name:<12} {name:<42} {summary['total_return']:>9.2%} "
                  f"{summary['max_drawdown']:>9.2%} {win_rate:>8.1%} "
                  f"{summary['closed_trades']:>6} {pf:>6.2f} {avg_exposure:>7.1%}")

    df = pd.DataFrame(records)
    out_dir = ROOT / "results/capacity-opt"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "capacity_optimization.csv", index=False, encoding="utf-8-sig")
    print("\nResults saved to: " + str(out_dir / "capacity_optimization.csv"))


if __name__ == "__main__":
    main()
