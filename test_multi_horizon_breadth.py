"""Research Script: Testing Multi-Horizon Strategies under Breadth Gating.

Evaluates:
1. breadth_h5: 5-day LGBM selective + Breadth >= 50%
2. breadth_h10: 10-day LGBM selective + Breadth >= 50% (Champion Baseline)
3. breadth_h20: 20-day LGBM selective + Breadth >= 50%
4. breadth_blended_score: Cross-horizon ensemble score (0.2*h5 + 0.6*h10 + 0.2*h20)
5. breadth_consensus: High-conviction consensus (both h5 and h10 agree pred >= 0.005)
6. breadth_h10_tp10: Champion h10 + Take-Profit 10% + 2 buys/day
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


class MultiHorizonReplay(AccessibleReplay):
    def __init__(self, frames, calendar, signals, dividends=None, config=None,
                 take_profit: float | None = None, max_daily_buys: int = 1):
        super().__init__(frames, calendar, signals, dividends, config)
        self.take_profit = take_profit
        self.max_daily_buys = max_daily_buys

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

                    value = position.shares * close_px + position.proceeds + position.dividends
                    if self.config.stop_loss and value < position.entry_notional * (1 - self.config.stop_loss):
                        position.stop_pending = True

                    if self.take_profit is not None:
                        entry_px = position.entry_notional / position.shares
                        if (close_px - entry_px) / entry_px >= self.take_profit:
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


def main():
    print("Loading datasets...")
    manifest, frames, calendar, dividends = load(DATASET)
    bars = prepare_bars(frames)

    panel = pd.read_csv(DONOR / "features.csv", dtype={"date": str, "code": str})
    pred_h5 = pd.read_csv(DONOR / "predictions-h5.csv", dtype={"date": str, "code": str})
    pred_h10 = pd.read_csv(DONOR / "predictions-h10.csv", dtype={"date": str, "code": str})
    pred_h20 = pd.read_csv(DONOR / "predictions-h20.csv", dtype={"date": str, "code": str})

    # Breadth calculation
    breadth_df = breadth_days(panel, return_col="adj_ret5")
    allowed_dates = set(breadth_df.loc[(breadth_df.pool_size >= 100) & (breadth_df.breadth5 >= 0.50), "date"])

    # 1. H5 signals
    s_h5 = pred_h5[pred_h5.pred_net_return >= 0.005][["date", "code", "pred_net_return"]].rename(columns={"pred_net_return": "score"})
    s_h5 = s_h5[s_h5.date.isin(allowed_dates)]

    # 2. H10 signals
    s_h10 = pred_h10[pred_h10.pred_net_return >= 0.005][["date", "code", "pred_net_return"]].rename(columns={"pred_net_return": "score"})
    s_h10 = s_h10[s_h10.date.isin(allowed_dates)]

    # 3. H20 signals
    s_h20 = pred_h20[pred_h20.pred_net_return >= 0.005][["date", "code", "pred_net_return"]].rename(columns={"pred_net_return": "score"})
    s_h20 = s_h20[s_h20.date.isin(allowed_dates)]

    # 4. Blended Score (0.2*h5 + 0.6*h10 + 0.2*h20)
    m = pred_h10[["date", "code", "pred_net_return"]].merge(
        pred_h5[["date", "code", "pred_net_return"]], on=["date", "code"], suffixes=("_h10", "_h5")
    ).merge(
        pred_h20[["date", "code", "pred_net_return"]], on=["date", "code"]
    ).rename(columns={"pred_net_return": "pred_net_return_h20"})

    m["score"] = 0.20 * m["pred_net_return_h5"] + 0.60 * m["pred_net_return_h10"] + 0.20 * m["pred_net_return_h20"]
    s_blend = m[(m.pred_net_return_h10 >= 0.005) & (m.score >= 0.005)][["date", "code", "score"]]
    s_blend = s_blend[s_blend.date.isin(allowed_dates)]

    # 5. Consensus: Both H5 and H10 >= 0.005
    s_consensus = m[(m.pred_net_return_h5 >= 0.005) & (m.pred_net_return_h10 >= 0.005)][["date", "code", "pred_net_return_h10"]].rename(columns={"pred_net_return_h10": "score"})
    s_consensus = s_consensus[s_consensus.date.isin(allowed_dates)]

    periods = [
        ("2022-2025", "2022-01-01", "2025-12-31"),
        ("2026", "2026-01-01", "2026-10-13"),
    ]

    configs_to_test = [
        ("1. breadth_h5 (5d hold)", s_h5, 5, None, 1),
        ("2. breadth_h10 (10d hold, Champion Base)", s_h10, 10, None, 1),
        ("3. breadth_h20 (20d hold)", s_h20, 20, None, 1),
        ("4. breadth_blend_h10 (Multi-horizon score)", s_blend, 10, None, 1),
        ("5. breadth_consensus_h10 (h5+h10 agree)", s_consensus, 10, None, 1),
        ("6. breadth_h10 + TP10% (Take-Profit 10%)", s_h10, 10, 0.10, 1),
        ("7. breadth_blend_h10 + TP10%", s_blend, 10, 0.10, 1),
        ("8. breadth_consensus + TP10%", s_consensus, 10, 0.10, 1),
        ("9. SUPER-STRATEGY: Blend + TP10% + 2Buys/d", s_blend, 10, 0.10, 2),
        ("10. SUPER-STRATEGY: Base_h10 + TP10% + 2Buys/d", s_h10, 10, 0.10, 2),
    ]

    records = []
    print("\n" + "=" * 96)
    print(f"{'Period':<12} {'Strategy':<46} {'Return':<10} {'MaxDD':<10} {'WinRate':<9} {'Trades':<7} {'PF':<7}")
    print("=" * 96)

    for period_name, start, end in periods:
        period_days = [d for d in calendar if start <= d <= end]

        for name, sigs_all, hold_d, tp, daily_buys in configs_to_test:
            period_sigs = sigs_all[sigs_all.date.between(start, end)].copy()
            cfg = Config(
                capital=30000.0,
                order_budget=5000.0,
                max_positions=4 if daily_buys > 1 else 3,
                max_exposure=0.65 if daily_buys > 1 else 0.55,
                hold_days=hold_d,
                stop_loss=0.03,
            )
            replay = MultiHorizonReplay(frames, period_days, period_sigs, dividends, cfg,
                                        take_profit=tp, max_daily_buys=daily_buys)
            replay.bars = bars
            replay.run()
            summary = replay.summary()

            win_rate = summary["win_rate"] if summary["win_rate"] is not None else 0.0
            pf = summary["profit_factor"] if summary["profit_factor"] is not None else 0.0

            records.append({
                "period": period_name,
                "strategy": name,
                "hold_days": hold_d,
                "take_profit": tp,
                "daily_buys": daily_buys,
                "total_return": summary["total_return"],
                "max_drawdown": summary["max_drawdown"],
                "win_rate": win_rate,
                "closed_trades": summary["closed_trades"],
                "profit_factor": pf,
                "execution_costs": summary["execution_costs"],
            })

            print(f"{period_name:<12} {name:<46} {summary['total_return']:>9.2%} "
                  f"{summary['max_drawdown']:>9.2%} {win_rate:>8.1%} "
                  f"{summary['closed_trades']:>6} {pf:>6.2f}")

    df = pd.DataFrame(records)
    out_dir = ROOT / "results/multi-horizon-breadth"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "multi_horizon_comparison.csv", index=False, encoding="utf-8-sig")
    print(f"\nResults saved to: {out_dir / 'multi_horizon_comparison.csv'}")


if __name__ == "__main__":
    main()
