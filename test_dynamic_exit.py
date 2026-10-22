"""Research Script: Testing Dynamic Exits, Take-Profit, and Trailing Stops.

Hypothesis:
Fixed 10-day holding lets profitable trades round-trip back to losses.
Introducing dynamic exits (Take-Profit and Trailing Stop):
1. Captures outsized gains quickly.
2. Frees up capital slots for fresh signals.
3. Reduces downside volatility and tail drawdowns.

Tests:
- Baseline: 10-day hold, 3% stop loss, no take-profit, no trailing stop
- Take-Profit sweeps: +6%, +8%, +10%, +12%
- Trailing Stop sweeps: (trigger, delta) = (+4%, 2%), (+5%, 2%), (+6%, 2.5%), (+8%, 3%)
- Combined: TP + Trailing Stop + Stop Loss
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import numpy as np

from accessibility import AccessibleReplay
from engine import Config, Position, tick
from exploration_features import breadth_days
from run import load
from validate_expanded import prepare_bars

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data/forward-2026-10-13"
DONOR = ROOT / "results/midterm-access-v3"


class DynamicExitReplay(AccessibleReplay):
    """Replay engine with Take-Profit, Trailing Stop, and Multi-buy capability."""

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
        # Track peak prices per trade: trade_id -> peak_close
        self.peak_prices: dict[int, float] = {}

    def run(self):
        for index, day in enumerate(self.calendar):
            self.corporate_actions(day)

            # Check sells at opening
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

            # Update marks to open
            for code in self.positions:
                bar = self.bars[code].get(day)
                if bar is not None and float(bar.get("open", 0.0) or 0.0) > 0:
                    self.marks[code] = float(bar["open"])

            # Execute pending buys
            for order in self.pending_buys:
                self.buy(order, day, index)
            self.pending_buys = []

            # Post-market evaluation: Update marks to close & check exit conditions
            for code, position in self.positions.items():
                bar = self.bars[code].get(day)
                if bar is not None:
                    close_px = float(bar["close"])
                    self.marks[code] = close_px

                    # Update peak price for trailing stop
                    current_peak = self.peak_prices.get(position.trade_id, position.entry_notional / position.shares)
                    if close_px > current_peak:
                        self.peak_prices[position.trade_id] = close_px
                        current_peak = close_px

                    entry_px = position.entry_notional / position.shares
                    cur_gain = (close_px - entry_px) / entry_px

                    # 1. Standard Stop Loss
                    value = position.shares * close_px + position.proceeds + position.dividends
                    if self.config.stop_loss and value < position.entry_notional * (1 - self.config.stop_loss):
                        position.stop_pending = True

                    # 2. Take-Profit
                    if self.take_profit is not None and cur_gain >= self.take_profit:
                        position.stop_pending = True

                    # 3. Trailing Stop
                    if self.trailing_trigger is not None and self.trailing_delta is not None:
                        peak_gain = (current_peak - entry_px) / entry_px
                        if peak_gain >= self.trailing_trigger:
                            # If retraced from peak by trailing_delta
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

            # Candidate queuing
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


def main():
    print("Loading data for dynamic exit evaluation...")
    manifest, frames, calendar, dividends = load(DATASET)
    bars = prepare_bars(frames)

    panel = pd.read_csv(DONOR / "features.csv", dtype={"date": str, "code": str})
    predictions = pd.read_csv(DONOR / "predictions-h10.csv", dtype={"date": str, "code": str})

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
        # (name, tp, trail_trig, trail_delta, max_daily_buys)
        ("1. Baseline (Fixed 10d, Stop 3%)", None, None, None, 1),
        ("2. Take Profit +6%", 0.06, None, None, 1),
        ("3. Take Profit +8%", 0.08, None, None, 1),
        ("4. Take Profit +10%", 0.10, None, None, 1),
        ("5. Take Profit +12%", 0.12, None, None, 1),
        ("6. Trailing Stop (Trig +4%, Delta 2%)", None, 0.04, 0.02, 1),
        ("7. Trailing Stop (Trig +5%, Delta 2%)", None, 0.05, 0.02, 1),
        ("8. Trailing Stop (Trig +6%, Delta 2.5%)", None, 0.06, 0.025, 1),
        ("9. Trailing Stop (Trig +8%, Delta 3%)", None, 0.08, 0.03, 1),
        ("10. Combined: TP+10% + Trail(+5%, 2%)", 0.10, 0.05, 0.02, 1),
        ("11. Combined: TP+8% + Trail(+4%, 2%)", 0.08, 0.04, 0.02, 1),
        ("12. Combined + 2Buys/day (TP+10%, Trail +5%/2%)", 0.10, 0.05, 0.02, 2),
    ]

    records = []
    print("\n" + "=" * 92)
    print(f"{'Period':<12} {'Exit Strategy':<40} {'Return':<10} {'MaxDD':<10} {'WinRate':<9} {'Trades':<7} {'PF':<7}")
    print("=" * 92)

    for period_name, start, end in periods:
        period_days = [d for d in calendar if start <= d <= end]
        sigs = gated_signals[gated_signals.date.between(start, end)].copy()

        for name, tp, t_trig, t_delta, buys in scenarios:
            cfg = Config(
                capital=30000.0,
                order_budget=5000.0,
                max_positions=3,
                max_exposure=0.55,
                hold_days=10,
                stop_loss=0.03,
            )
            replay = DynamicExitReplay(frames, period_days, sigs, dividends, cfg,
                                       take_profit=tp, trailing_trigger=t_trig,
                                       trailing_delta=t_delta, max_daily_buys=buys)
            replay.bars = bars
            replay.run()
            summary = replay.summary()

            win_rate = summary["win_rate"] if summary["win_rate"] is not None else 0.0
            pf = summary["profit_factor"] if summary["profit_factor"] is not None else 0.0

            records.append({
                "period": period_name,
                "strategy": name,
                "take_profit": tp,
                "trailing_trigger": t_trig,
                "trailing_delta": t_delta,
                "total_return": summary["total_return"],
                "max_drawdown": summary["max_drawdown"],
                "win_rate": win_rate,
                "closed_trades": summary["closed_trades"],
                "profit_factor": pf,
                "execution_costs": summary["execution_costs"],
            })

            print(f"{period_name:<12} {name:<40} {summary['total_return']:>9.2%} "
                  f"{summary['max_drawdown']:>9.2%} {win_rate:>8.1%} "
                  f"{summary['closed_trades']:>6} {pf:>6.2f}")

    df = pd.DataFrame(records)
    out_dir = ROOT / "results/dynamic-exit"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "dynamic_exit_comparison.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved results to: {out_dir / 'dynamic_exit_comparison.csv'}")


if __name__ == "__main__":
    main()
