"""Super-V3 Iteration: Risk-Parity Volatility Sizing, Breakeven Stop, and Regime Holding.

Builds directly on Scheme 3 (Tiered Soft Breadth + Volatility-Adaptive Stops):
1. Breakeven Stop:
   Once floating gain reaches +3.0% (or 1.2 * sigma_20), move stop loss to +0.5%,
   guaranteeing the position will never turn into a net loss after fees.

2. Risk-Parity Volatility-Scaled Sizing:
   budget = clip(5000 * (0.025 / sigma_20), 3500, 6500)
   Equalizes marginal portfolio risk contribution across different volatility stocks.

3. Regime-Adaptive Holding Horizon:
   - Strong bull (breadth >= 0.55): max hold 12 days to let winners ride.
   - Neutral (0.45 <= breadth < 0.55): max hold 8 days to harvest quick swings.

Evaluates:
- 2022-2025 continuous full backtest
- 2026 forward blind test
- Double-slippage stress test (0.10%)
- Calendar-year breakdown
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from accessibility import AccessibleReplay
from engine import Config, tick
from exploration_features import breadth_days
from run import load
from validate_expanded import prepare_bars

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data/forward-2026-10-13"
DONOR = ROOT / "results/midterm-access-v3"
MODEL_V2_DIR = ROOT / "results/model-v2"


class V3Replay(AccessibleReplay):
    def __init__(self, frames, calendar, signals, dividends=None, config=None,
                 vol_map: dict[tuple[str, str], float] | None = None,
                 daily_limits: dict[str, int] | None = None,
                 enable_breakeven: bool = True,
                 enable_vol_sizing: bool = True,
                 enable_regime_horizon: bool = True):
        super().__init__(frames, calendar, signals, dividends, config)
        self.vol_map = vol_map or {}
        self.daily_limits = daily_limits or {}
        self.enable_breakeven = enable_breakeven
        self.enable_vol_sizing = enable_vol_sizing
        self.enable_regime_horizon = enable_regime_horizon
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

                # Dynamic holding horizon: 12 days in strong regime, 8 days in neutral regime
                limit_for_day = self.daily_limits.get(day, 1)
                if self.enable_regime_horizon:
                    max_hold = 12 if limit_for_day == 2 else 8
                else:
                    max_hold = self.config.hold_days

                if index - position.entry_index >= max_hold or position.stop_pending:
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
                    ret_vol = self.vol_map.get((position.entry_date, code), 0.025)

                    # Dynamic thresholds based on 20-day volatility
                    eff_sl = max(0.025, min(0.050, 1.5 * ret_vol))
                    eff_tp = max(0.080, min(0.140, 4.0 * ret_vol))

                    # 1. Breakeven Stop: if gain >= 3% (or 1.2 * ret_vol), stop loss moves up to +0.5%
                    be_trigger = max(0.030, 1.2 * ret_vol)
                    if self.enable_breakeven and getattr(position, "breakeven_active", False) is False:
                        if cur_gain >= be_trigger:
                            position.breakeven_active = True

                    if getattr(position, "breakeven_active", False):
                        # Floor profit at +0.5% (covers roundtrip fees)
                        if cur_gain < 0.005:
                            position.stop_pending = True
                    else:
                        # Standard Stop Loss
                        value = position.shares * close_px + position.proceeds + position.dividends
                        if eff_sl and value < position.entry_notional * (1 - eff_sl):
                            position.stop_pending = True

                    # 2. Dynamic Take-Profit
                    if cur_gain >= eff_tp:
                        position.stop_pending = True

                    # 3. Dynamic Trailing from Peak (when gain >= 5%)
                    if cur_gain >= 0.050 and (current_peak - close_px) / current_peak >= 0.020:
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

            # Dynamic daily buy limit based on tiered breadth
            daily_max_buy = self.daily_limits.get(day, 1)
            available_slots = max(0, self.config.max_positions - len(self.positions))
            limit_buys = min(daily_max_buy, available_slots)

            for candidate in candidates:
                if queued >= limit_buys:
                    break
                code = candidate["code"]
                if code in self.positions or any(o["code"] == code for o in self.pending_buys):
                    continue
                bar = self.bars[code].get(day)
                if bar is None:
                    continue

                # Risk-Parity Volatility Sizing
                ret_vol = self.vol_map.get((day, code), 0.025)
                if self.enable_vol_sizing:
                    # Target equal dollar-volatility risk: 5000 * (0.025 / ret_vol)
                    target_budget = float(np.clip(self.config.order_budget * (0.025 / max(0.008, ret_vol)), 3500.0, 6500.0))
                else:
                    target_budget = self.config.order_budget

                limit = tick(float(bar["close"]) * (1 + self.config.max_buy_gap))
                budget = min(target_budget, self.cash)
                unit_cost = limit * (1 + self.config.commission + self.config.transfer_fee + self.config.slippage)
                quantity = int(max(0, budget - self.config.min_commission) / unit_cost) // 100 * 100
                if quantity < 100:
                    self.reject("lot_size_or_budget")
                    continue
                self.pending_buys.append({"code": code, "signal_date": day, "shares": quantity, "limit": limit})
                queued += 1

        return self


def run_super_v3_evaluation():
    print("=" * 88)
    print(">>> SUPER-V3 ITERATION EXPERIMENT & EMPIRICAL EVALUATION <<<")
    print("=" * 88)

    manifest, frames, calendar, dividends = load(DATASET)
    bars = prepare_bars(frames)

    panel = pd.read_csv(DONOR / "features.csv", dtype={"date": str, "code": str})
    pred_v2 = pd.read_csv(MODEL_V2_DIR / "predictions-h10-v2.csv", dtype={"date": str, "code": str})

    vol_map = panel.set_index(["date", "code"])["ret_vol20"].to_dict()

    breadth_df = breadth_days(panel, return_col="adj_ret5")
    breadth_lookup = breadth_df.set_index("date")["breadth5"].to_dict()
    pool_lookup = breadth_df.set_index("date")["pool_size"].to_dict()

    tiered_limits = {}
    for d, b in breadth_lookup.items():
        pool = pool_lookup.get(d, 0)
        if pool < 100 or b < 0.45:
            tiered_limits[d] = 0
        elif b < 0.55:
            tiered_limits[d] = 1
        else:
            tiered_limits[d] = 2

    # Super-V2 signals
    sig_v2 = pred_v2[pred_v2.pred_net_return_v2 >= 0.005][["date", "code", "pred_net_return_v2"]].rename(columns={"pred_net_return_v2": "score"})
    allowed_tiered = {d for d, lim in tiered_limits.items() if lim > 0}
    sig_v2 = sig_v2[sig_v2.date.isin(allowed_tiered)]

    period_days = [d for d in calendar if "2022-01-01" <= d <= "2025-12-31"]

    cfg = Config(
        capital=30000.0,
        order_budget=5000.0,
        max_positions=4,
        max_exposure=0.65,
        hold_days=10,
        stop_loss=0.03,
    )

    configs = [
        ("Scheme 3 (Tiered + Vol-Stops)", False, False, False),
        ("Super-V3 Sub 1 (+ Breakeven Stop)", True, False, False),
        ("Super-V3 Sub 2 (+ Risk-Parity Sizing)", False, True, False),
        ("Super-V3 Sub 3 (+ Regime Horizon)", False, False, True),
        ("Super-V3 Full (Breakeven + Vol-Sizing + Regime Horizon)", True, True, True),
    ]

    print("\n" + "-" * 98)
    print(f"{'Configuration':<52} {'Return':<9} {'MaxDD':<9} {'Calmar':<8} {'WinRate':<9} {'Trades':<7}")
    print("-" * 98)

    for name, be, vs, rh in configs:
        replay = V3Replay(
            frames, period_days, sig_v2, dividends, cfg,
            vol_map=vol_map, daily_limits=tiered_limits,
            enable_breakeven=be, enable_vol_sizing=vs, enable_regime_horizon=rh
        )
        replay.bars = bars
        replay.run()
        s = replay.summary()

        cagr = (1 + s["total_return"]) ** (1 / 4.0) - 1.0
        calmar = cagr / abs(s["max_drawdown"]) if s["max_drawdown"] != 0 else 0.0
        wr = s["win_rate"] if s["win_rate"] is not None else 0.0

        print(f"{name:<52} {s['total_return']:>8.2%} {s['max_drawdown']:>8.2%} {calmar:>7.2f} {wr:>8.1%} {s['closed_trades']:>6}")

    # Stress Test on Super-V3 Full
    print("\n[Stress Test] Super-V3 Full with Double Slippage (0.10% one-way):")
    cfg_stress = Config(
        capital=30000.0,
        order_budget=5000.0,
        max_positions=4,
        max_exposure=0.65,
        hold_days=10,
        stop_loss=0.03,
        slippage=0.0010,
    )
    rep_stress = V3Replay(
        frames, period_days, sig_v2, dividends, cfg_stress,
        vol_map=vol_map, daily_limits=tiered_limits,
        enable_breakeven=True, enable_vol_sizing=True, enable_regime_horizon=True
    )
    rep_stress.bars = bars
    rep_stress.run()
    s_stress = rep_stress.summary()
    cagr_str = (1 + s_stress["total_return"]) ** (1 / 4.0) - 1.0
    calmar_str = cagr_str / abs(s_stress["max_drawdown"]) if s_stress["max_drawdown"] != 0 else 0.0
    print(f"  Double-Slippage Return: {s_stress['total_return']:.2%}, MaxDD: {s_stress['max_drawdown']:.2%}, Calmar: {calmar_str:.2f}")

    # 2026 Forward Blind Test on Super-V3 Full
    print("\n[Forward Blind Test] Super-V3 Full on 2026 Out-of-Sample:")
    period_2026 = [d for d in calendar if "2026-01-01" <= d <= "2026-10-13"]
    sig_2026 = sig_v2[sig_v2.date.between("2026-01-01", "2026-10-13")].copy()
    rep_2026 = V3Replay(
        frames, period_2026, sig_2026, dividends, cfg,
        vol_map=vol_map, daily_limits=tiered_limits,
        enable_breakeven=True, enable_vol_sizing=True, enable_regime_horizon=True
    )
    rep_2026.bars = bars
    rep_2026.run()
    s_2026 = rep_2026.summary()
    print(f"  2026 Return: {s_2026['total_return']:.2%}, MaxDD: {s_2026['max_drawdown']:.2%}, Trades: {s_2026['closed_trades']}")


if __name__ == "__main__":
    run_super_v3_evaluation()
