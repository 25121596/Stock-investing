"""Stage 3 Architectural Upgrade: Cross-Asset Correlation Dispersion & Orthogonalization.

Core Innovation:
When selecting daily buy candidates, before queuing an order:
Compute rolling 60-day price return correlation with all currently held positions.
If correlation(candidate, position_i) >= 0.65:
    Reject candidate for "sector_clustering_risk", pick the next highest-conviction uncorrelated candidate!

Ensures true multi-sector diversification, preventing correlated drawdowns.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from test_asymmetric_engine import AsymmetricReplay, DATASET, DONOR, MODEL_V2_DIR
from engine import Config, tick
from exploration_features import breadth_days
from run import load
from validate_expanded import prepare_bars

ROOT = Path(__file__).resolve().parent


class CorrelationDiversifiedReplay(AsymmetricReplay):
    def __init__(self, frames, calendar, signals, dividends=None, config=None,
                 vol_map: dict[tuple[str, str], float] | None = None,
                 daily_limits: dict[str, int] | None = None,
                 enable_cash_overlay: bool = True,
                 max_correlation: float = 0.65,
                 ret_matrix: pd.DataFrame | None = None):
        super().__init__(frames, calendar, signals, dividends, config,
                         vol_map=vol_map, daily_limits=daily_limits,
                         enable_cash_overlay=enable_cash_overlay)
        self.max_correlation = max_correlation
        self.ret_matrix = ret_matrix  # Precomputed matrix of daily returns: index=date, columns=codes
        self.rejected_correlation_count = 0

    def run(self):
        # We need quick rolling correlation access
        for index, day in enumerate(self.calendar):
            self.corporate_actions(day)

            # Sells from scheduled exits
            for code, position in list(self.positions.items()):
                bar = self.bars[code].get(day)
                limit_for_day = self.daily_limits.get(day, 1)
                max_hold = 12 if limit_for_day >= 2 else 8
                if index - position.entry_index >= max_hold or position.stop_pending:
                    self.sell(position, day, index)

            for code in self.positions:
                bar = self.bars[code].get(day)
                if bar is not None and float(bar.get("open", 0.0) or 0.0) > 0:
                    self.marks[code] = float(bar["open"])

            for order in self.pending_buys:
                self.buy(order, day, index)
            self.pending_buys = []

            for code, position in list(self.positions.items()):
                bar = self.bars[code].get(day)
                if bar is None:
                    continue
                open_px = float(bar.get("open", 0.0) or 0.0)
                high_px = float(bar.get("high", 0.0) or 0.0)
                close_px = float(bar.get("close", 0.0) or 0.0)
                if open_px <= 0 or high_px <= 0 or close_px <= 0:
                    continue
                self.marks[code] = close_px

                current_peak = self.peak_prices.get(position.trade_id, position.entry_notional / position.shares)
                if high_px > current_peak:
                    self.peak_prices[position.trade_id] = high_px
                    current_peak = high_px

                entry_px = position.entry_notional / position.shares
                ret_vol = self.vol_map.get((position.entry_date, code), 0.025)
                eff_sl_pct = max(0.025, min(0.050, 1.5 * ret_vol))
                eff_tp_pct = max(0.090, min(0.150, 4.0 * ret_vol))

                cur_gain = (close_px - entry_px) / entry_px
                peak_gain = (current_peak - entry_px) / entry_px
                tp_px = entry_px * (1 + eff_tp_pct)

                can_sell_today = (index > position.entry_index)

                # Intraday TP on surges
                if can_sell_today and high_px >= tp_px:
                    fill_px = max(open_px, tp_px) * (1 - self.config.slippage)
                    self._execute_custom_sell(position, day, index, fill_px, "intraday_take_profit")
                    continue

                # Close-Confirmed Stop Escalation
                if cur_gain >= max(0.030, 1.2 * ret_vol):
                    position.breakeven_active = True
                if cur_gain >= max(0.080, 2.5 * ret_vol):
                    position.moonbag_active = True

                if getattr(position, "moonbag_active", False):
                    if cur_gain < 0.035:
                        position.stop_pending = True
                elif getattr(position, "breakeven_active", False):
                    if cur_gain < 0.005:
                        position.stop_pending = True
                else:
                    if cur_gain < -eff_sl_pct:
                        position.stop_pending = True

                if peak_gain >= 0.100 and (current_peak - close_px) / current_peak >= 0.020:
                    position.stop_pending = True

            for event in self.actions_by_record.get(day, []):
                position = self.positions.get(event.get("code"))
                if position and position.shares:
                    position.pending_actions += 1
                    self.entitlements.append({"event": event, "position": position, "shares": position.shares, "posted": False})

            if self.enable_cash_overlay and self.cash > 0:
                interest = self.cash * self.daily_cash_yield
                self.cash += interest
                self.accumulated_cash_yield += interest

            nav = self.nav()
            self.equity.append({"date": day, "equity": nav, "cash": self.cash,
                                "receivables": sum(item[1] for item in self.receivables),
                                "positions": len(self.positions)})

            candidates = sorted(self.candidates.get(day, []), key=lambda row: (-row["score"], row["code"]))
            queued = 0
            daily_max_buy = self.daily_limits.get(day, 1)
            available_slots = max(0, self.config.max_positions - len(self.positions))
            limit_buys = min(daily_max_buy, available_slots)

            # Active codes currently held or already queued today
            active_codes = list(self.positions.keys()) + [o["code"] for o in self.pending_buys]

            # Correlation lookback window (60 days prior to day)
            window_slice = None
            if self.ret_matrix is not None and len(active_codes) > 0 and index >= 60:
                start_i = max(0, index - 60)
                sub_dates = self.calendar[start_i:index]
                window_slice = self.ret_matrix.loc[self.ret_matrix.index.isin(sub_dates)]

            for candidate in candidates:
                if queued >= limit_buys:
                    break
                code = candidate["code"]
                if code in active_codes:
                    continue

                # Check correlation with currently active positions
                is_too_correlated = False
                if window_slice is not None and code in window_slice.columns:
                    c_ret = window_slice[code]
                    for pos_c in active_codes:
                        if pos_c in window_slice.columns:
                            p_ret = window_slice[pos_c]
                            # Compute pearson correlation
                            corr = c_ret.corr(p_ret)
                            if pd.notna(corr) and corr >= self.max_correlation:
                                is_too_correlated = True
                                self.rejected_correlation_count += 1
                                break

                if is_too_correlated:
                    # Skip this candidate to prevent sector clustering risk
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
                active_codes.append(code)
                queued += 1

        return self


def run_stage3_evaluation():
    print("=" * 88)
    print(">>> ARCHITECTURAL UPGRADE STAGE 3: CORRELATION DIVERSIFICATION <<<")
    print("=" * 88)

    manifest, frames, calendar, dividends = load(DATASET)
    bars = prepare_bars(frames)

    panel = pd.read_csv(DONOR / "features.csv", dtype={"date": str, "code": str})
    pred_v2 = pd.read_csv(MODEL_V2_DIR / "predictions-h10-v2.csv", dtype={"date": str, "code": str})
    vol_map = panel.set_index(["date", "code"])["ret_vol20"].to_dict()

    # Build return matrix for correlation (pivot on date and code using adj_ret1)
    print("Constructing 60-day cross-sectional return matrix...")
    ret_matrix = panel.pivot(index="date", columns="code", values="adj_ret1")

    breadth_df = breadth_days(panel, return_col="adj_ret5")
    breadth_lookup = breadth_df.set_index("date")["breadth5"].to_dict()
    pool_lookup = breadth_df.set_index("date")["pool_size"].to_dict()

    tiered_limits = {d: (0 if pool_lookup.get(d, 0) < 100 or b < 0.45 else (1 if b < 0.55 else 2)) for d, b in breadth_lookup.items()}
    allowed_tiered = {d for d, lim in tiered_limits.items() if lim > 0}
    sig_v2 = pred_v2[pred_v2.pred_net_return_v2 >= 0.005][["date", "code", "pred_net_return_v2"]].rename(columns={"pred_net_return_v2": "score"})
    sig_v2 = sig_v2[sig_v2.date.isin(allowed_tiered)]

    period_days = [d for d in calendar if "2022-01-01" <= d <= "2025-12-31"]
    cfg = Config(capital=30000.0, order_budget=5000.0, max_positions=4, max_exposure=0.65, hold_days=10, stop_loss=0.03)

    for corr_thresh in [1.0, 0.70, 0.60, 0.50]:
        rep = CorrelationDiversifiedReplay(
            frames, period_days, sig_v2, dividends, cfg,
            vol_map=vol_map, daily_limits=tiered_limits,
            enable_cash_overlay=True, max_correlation=corr_thresh,
            ret_matrix=ret_matrix
        )
        rep.bars = bars
        rep.run()
        s = rep.summary()
        calmar = s["cagr"] / abs(s["max_drawdown"])
        desc = "No Corr Filter (Threshold=1.0)" if corr_thresh == 1.0 else f"Corr Filter Threshold < {corr_thresh:.2f}"
        print(f"{desc:<36} Return: {s['total_return']:>6.2%} | MaxDD: {s['max_drawdown']:>6.2%} | Calmar: {calmar:>5.2f} | WinRate: {s['win_rate']:>5.1%} | Filtered: {rep.rejected_correlation_count:>3d} | Trades: {s['closed_trades']:>3d}")


if __name__ == "__main__":
    run_stage3_evaluation()
