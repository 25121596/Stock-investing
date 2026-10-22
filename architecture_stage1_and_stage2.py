"""Architecture Upgrade Stage 1 & Stage 2: Intraday Conditional Execution & Cash Overlay.

1. Stage 1: Cash Management Overlay (全天候现金流协同)
   - Overnight uninvested cash earns institutional cash yield (annualized 2.20%, 0.009% per session).
   - Modeled after standard A-share broker cash sweeps (券商保证金自动理财 / 国债逆回购 GC001).

2. Stage 2: Intraday Conditional Order Engine (日内条件单事件驱动撮合)
   - In real A-share trading, conditional orders (条件单) execute intraday once shares are T+1 eligible.
   - If Low_t <= StopPrice:
     * If Open_t <= StopPrice: fills at Open_t * (1 - slippage) [gap-down fill]
     * If Open_t > StopPrice: fills at StopPrice * (1 - slippage) [intraday stop touch]
   - If High_t >= TakeProfitPrice:
     * If Open_t >= TakeProfitPrice: fills at Open_t * (1 - slippage) [gap-up fill]
     * If Open_t < TakeProfitPrice: fills at TakeProfitPrice * (1 - slippage) [intraday limit touch]
   - Eliminates overnight gap risk from waiting for next morning's open!

Evaluates:
- Baseline Super-V4 AlphaMax
- + Cash Management Overlay
- + Intraday Conditional Execution
- + Both Combined (Enterprise Engine V5 Alpha)
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from accessibility import AccessibleReplay
from engine import Config, tick, number
from exploration_features import breadth_days
from run import load
from validate_expanded import prepare_bars

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data/forward-2026-10-13"
DONOR = ROOT / "results/midterm-access-v3"
MODEL_V2_DIR = ROOT / "results/model-v2"


class IntradayEnterpriseReplay(AccessibleReplay):
    def __init__(self, frames, calendar, signals, dividends=None, config=None,
                 vol_map: dict[tuple[str, str], float] | None = None,
                 daily_limits: dict[str, int] | None = None,
                 enable_cash_overlay: bool = True,
                 cash_annual_rate: float = 0.022,  # 2.2% annual cash yield
                 enable_intraday_stops: bool = True):
        super().__init__(frames, calendar, signals, dividends, config)
        self.vol_map = vol_map or {}
        self.daily_limits = daily_limits or {}
        self.enable_cash_overlay = enable_cash_overlay
        self.daily_cash_yield = cash_annual_rate / 242.0 if enable_cash_overlay else 0.0
        self.accumulated_cash_yield = 0.0
        self.enable_intraday_stops = enable_intraday_stops
        self.peak_prices: dict[int, float] = {}

    def run(self):
        for index, day in enumerate(self.calendar):
            self.corporate_actions(day)

            # -------------------------------------------------------------
            # STEP 1: Process scheduled exits from previous day (e.g. max hold reached)
            # -------------------------------------------------------------
            for code, position in list(self.positions.items()):
                bar = self.bars[code].get(day)
                limit_for_day = self.daily_limits.get(day, 1)
                max_hold = 12 if limit_for_day >= 2 else 8

                # If holding period expired, sell at open
                if index - position.entry_index >= max_hold or position.stop_pending:
                    self.sell(position, day, index)

            # Update marks to open for remaining positions
            for code in self.positions:
                bar = self.bars[code].get(day)
                if bar is not None and float(bar.get("open", 0.0) or 0.0) > 0:
                    self.marks[code] = float(bar["open"])

            # -------------------------------------------------------------
            # STEP 2: Execute queued buys at opening
            # -------------------------------------------------------------
            for order in self.pending_buys:
                self.buy(order, day, index)
            self.pending_buys = []

            # -------------------------------------------------------------
            # STEP 3: Intraday Conditional Order Execution & Stop Evaluation
            # -------------------------------------------------------------
            for code, position in list(self.positions.items()):
                bar = self.bars[code].get(day)
                if bar is None:
                    continue

                open_px = float(bar.get("open", 0.0) or 0.0)
                high_px = float(bar.get("high", 0.0) or 0.0)
                low_px = float(bar.get("low", 0.0) or 0.0)
                close_px = float(bar.get("close", 0.0) or 0.0)
                preclose_px = float(bar.get("preclose", 0.0) or 0.0)

                if open_px <= 0 or high_px <= 0 or low_px <= 0 or close_px <= 0:
                    continue

                self.marks[code] = close_px

                # Update peak price
                current_peak = self.peak_prices.get(position.trade_id, position.entry_notional / position.shares)
                if high_px > current_peak:
                    self.peak_prices[position.trade_id] = high_px
                    current_peak = high_px

                entry_px = position.entry_notional / position.shares
                ret_vol = self.vol_map.get((position.entry_date, code), 0.025)

                # Thresholds
                eff_sl_pct = max(0.025, min(0.050, 1.5 * ret_vol))
                eff_tp_pct = max(0.090, min(0.150, 4.0 * ret_vol))

                cur_gain = (close_px - entry_px) / entry_px
                peak_gain = (current_peak - entry_px) / entry_px

                # State escalation
                if cur_gain >= max(0.030, 1.2 * ret_vol):
                    position.breakeven_active = True
                if cur_gain >= max(0.080, 2.5 * ret_vol):
                    position.moonbag_active = True

                # Determine effective stop price
                if getattr(position, "moonbag_active", False):
                    stop_px = entry_px * (1 + 0.035)
                elif getattr(position, "breakeven_active", False):
                    stop_px = entry_px * (1 + 0.005)
                else:
                    stop_px = entry_px * (1 - eff_sl_pct)

                tp_px = entry_px * (1 + eff_tp_pct)

                # T+1 Check: only positions entered before today are eligible for intraday selling!
                can_sell_today = (index > position.entry_index)

                if self.enable_intraday_stops and can_sell_today:
                    intraday_sold = False

                    # Check Take Profit touch
                    if high_px >= tp_px:
                        # Filled at max(open, tp_px) with slippage
                        fill_px = max(open_px, tp_px) * (1 - self.config.slippage)
                        self._execute_custom_sell(position, day, index, fill_px, "intraday_take_profit")
                        intraday_sold = True

                    # Check Stop Loss touch (if not already sold by TP)
                    elif low_px <= stop_px:
                        # Filled at min(open, stop_px) with slippage
                        fill_px = min(open_px, stop_px) * (1 - self.config.slippage)
                        # Check limit down floor
                        limit_down = tick(preclose_px * 0.90)
                        fill_px = max(fill_px, limit_down)
                        self._execute_custom_sell(position, day, index, fill_px, "intraday_stop_loss")
                        intraday_sold = True

                    # Check Peak Trailing touch
                    elif peak_gain >= 0.100 and (current_peak - close_px) / current_peak >= 0.020:
                        fill_px = close_px * (1 - self.config.slippage)
                        self._execute_custom_sell(position, day, index, fill_px, "intraday_peak_trail")
                        intraday_sold = True

                elif not self.enable_intraday_stops:
                    # Naive close evaluation (sets stop_pending for tomorrow)
                    if getattr(position, "moonbag_active", False) and cur_gain < 0.035:
                        position.stop_pending = True
                    elif getattr(position, "breakeven_active", False) and cur_gain < 0.005:
                        position.stop_pending = True
                    elif cur_gain < -eff_sl_pct:
                        position.stop_pending = True
                    elif peak_gain >= 0.100 and (current_peak - close_px) / current_peak >= 0.020:
                        position.stop_pending = True
                    elif cur_gain >= eff_tp_pct:
                        position.stop_pending = True

            # -------------------------------------------------------------
            # STEP 4: Corporate Actions Entitlements
            # -------------------------------------------------------------
            for event in self.actions_by_record.get(day, []):
                position = self.positions.get(event.get("code"))
                if position and position.shares:
                    position.pending_actions += 1
                    self.entitlements.append({"event": event, "position": position,
                                              "shares": position.shares, "posted": False})

            # -------------------------------------------------------------
            # STEP 5: Cash Management Sweep (Earn daily interest on uninvested cash)
            # -------------------------------------------------------------
            if self.enable_cash_overlay and self.cash > 0:
                interest = self.cash * self.daily_cash_yield
                self.cash += interest
                self.accumulated_cash_yield += interest

            # Record daily equity
            nav = self.nav()
            self.equity.append({"date": day, "equity": nav, "cash": self.cash,
                                "receivables": sum(item[1] for item in self.receivables),
                                "positions": len(self.positions)})
            if self.cash < -1e-7 or any(p.shares < 0 for p in self.positions.values()):
                raise AssertionError("Negative cash or shares")

            # -------------------------------------------------------------
            # STEP 6: Select candidates for tomorrow
            # -------------------------------------------------------------
            candidates = sorted(self.candidates.get(day, []), key=lambda row: (-row["score"], row["code"]))
            queued = 0

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

                budget = min(self.config.order_budget, self.cash)
                limit = tick(float(bar["close"]) * (1 + self.config.max_buy_gap))
                unit_cost = limit * (1 + self.config.commission + self.config.transfer_fee + self.config.slippage)
                quantity = int(max(0, budget - self.config.min_commission) / unit_cost) // 100 * 100
                if quantity < 100:
                    self.reject("lot_size_or_budget")
                    continue
                self.pending_buys.append({"code": code, "signal_date": day, "shares": quantity, "limit": limit})
                queued += 1

        return self

    def summary(self):
        equity = pd.DataFrame(self.equity)
        curve = pd.concat([pd.Series([self.config.capital]), equity.equity], ignore_index=True)
        pnl = [trade["net_pnl"] for trade in self.trades]
        wins = sum(value > 0 for value in pnl)
        positive = sum(value for value in pnl if value > 0)
        negative = -sum(value for value in pnl if value < 0)
        final = float(curve.iloc[-1])
        duration = (pd.Timestamp(self.calendar[-1]) - pd.Timestamp(self.calendar[0])).days
        open_pnl = sum(item["marked_pnl"] for item in self.holdings())
        reconciliation = final - self.config.capital - sum(pnl) - open_pnl - self.accumulated_cash_yield
        if abs(reconciliation) > 1e-5:
            raise AssertionError(f"NAV does not reconcile to closed/open P&L: {reconciliation}")
        return {
            "initial_capital": self.config.capital,
            "final_equity": final,
            "total_return": final / self.config.capital - 1,
            "cagr": (final / self.config.capital) ** (1.0 / 4.0) - 1.0,
            "max_drawdown": float((curve / curve.cummax() - 1).min()),
            "closed_trades": len(pnl),
            "win_rate": wins / len(pnl) if pnl else None,
            "profit_factor": positive / negative if negative else None,
            "accumulated_cash_yield": self.accumulated_cash_yield,
            "execution_costs": sum(sum(fill[key] for key in ("commission", "transfer", "stamp", "slippage")) for fill in self.fills)
        }

    def _execute_custom_sell(self, position, day, index, fill_px, reason):
        """Execute intraday sell at simulated conditional fill price."""
        code = position.code
        bar = self.bars[code].get(day)
        # Fees
        notional = position.shares * fill_px
        commission = max(self.config.min_commission, notional * self.config.commission)
        transfer_fee = notional * self.config.transfer_fee
        stamp_tax = notional * self.config.stamp_tax
        fees = commission + transfer_fee + stamp_tax

        position.proceeds += notional - fees
        position.exit_fees += fees

        # Finish trade
        pnl = position.proceeds + position.dividends - position.entry_notional - position.entry_fees
        self.trades.append({
            "trade_id": position.trade_id,
            "code": position.code,
            "signal_date": position.signal_date,
            "entry_date": position.entry_date,
            "exit_date": day,
            "initial_shares": position.initial_shares,
            "entry_notional": position.entry_notional,
            "entry_fees": position.entry_fees,
            "exit_fees": position.exit_fees,
            "net_dividends": position.dividends,
            "net_pnl": pnl,
            "net_return": pnl / (position.entry_notional + position.entry_fees),
            "exit_reason": reason
        })
        self.positions.pop(code, None)
        self.cash += (notional - fees)


def run_stage1_and_stage2_evaluation():
    print("=" * 88)
    print(">>> ARCHITECTURAL EVOLUTION: STAGE 1 (CASH OVERLAY) & STAGE 2 (INTRADAY ENGINE) <<<")
    print("=" * 88)

    manifest, frames, calendar, dividends = load(DATASET)
    bars = prepare_bars(frames)

    panel = pd.read_csv(DONOR / "features.csv", dtype={"date": str, "code": str})
    pred_v2 = pd.read_csv(MODEL_V2_DIR / "predictions-h10-v2.csv", dtype={"date": str, "code": str})

    vol_map = panel.set_index(["date", "code"])["ret_vol20"].to_dict()

    breadth_df = breadth_days(panel, return_col="adj_ret5")
    breadth_lookup = breadth_df.set_index("date")["breadth5"].to_dict()
    pool_lookup = breadth_df.set_index("date")["pool_size"].to_dict()

    tiered_limits = {d: (0 if pool_lookup.get(d, 0) < 100 or b < 0.45 else (1 if b < 0.55 else 2)) for d, b in breadth_lookup.items()}
    allowed_tiered = {d for d, lim in tiered_limits.items() if lim > 0}

    sig_v2 = pred_v2[pred_v2.pred_net_return_v2 >= 0.005][["date", "code", "pred_net_return_v2"]].rename(columns={"pred_net_return_v2": "score"})
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

    architectures = [
        ("1. Super-V4 AlphaMax (Baseline Architecture)", False, False),
        ("2. Architecture + Cash Sweep Overlay (Stage 1)", True, False),
        ("3. Architecture + Intraday Event Engine (Stage 2)", False, True),
        ("4. Enterprise Engine V5 Alpha (Stage 1 + Stage 2)", True, True),
    ]

    print("\n" + "-" * 98)
    print(f"{'Architecture Version':<52} {'Return':<9} {'MaxDD':<9} {'Calmar':<8} {'WinRate':<9} {'Trades':<7}")
    print("-" * 98)

    for name, cash_on, intra_on in architectures:
        replay = IntradayEnterpriseReplay(
            frames, period_days, sig_v2, dividends, cfg,
            vol_map=vol_map, daily_limits=tiered_limits,
            enable_cash_overlay=cash_on, enable_intraday_stops=intra_on
        )
        replay.bars = bars
        replay.run()
        s = replay.summary()

        cagr = (1 + s["total_return"]) ** 0.25 - 1.0
        calmar = cagr / abs(s["max_drawdown"]) if s["max_drawdown"] != 0 else 0.0
        wr = s["win_rate"] if s["win_rate"] is not None else 0.0

        print(f"{name:<52} {s['total_return']:>8.2%} {s['max_drawdown']:>8.2%} {calmar:>7.2f} {wr:>8.1%} {s['closed_trades']:>6}")


if __name__ == "__main__":
    run_stage1_and_stage2_evaluation()
