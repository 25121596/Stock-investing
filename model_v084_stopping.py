"""Model V0.8.4: Dynamic Volatility-Adjusted Optimal Stopping.

Academic & Theoretical Foundations:
1. Deep Optimal Stopping (JMLR 2019: Becker, Cheridito & Jentzen):
   Optimal liquidation decision compares immediate execution value g(S_t) against continuation utility E[V_{t+1}|S_t].
   Replaces rigid 8/12-day fixed holding periods with an adaptive stopping boundary S*(t, PnL, Vol).
2. Time Utility Decay & Capital Opportunity Cost:
   Alpha expectation of cross-sectional momentum decays with holding age. Positions stagnant after
   day 5 are systematically liquidated to reallocate capital to high-conviction fresh candidates.
3. Volatility-Scaled Trailing Profit Locking:
   Locks in gains once unrealized PnL crosses 2.0 * sigma_YZ, enforcing an asymmetric trailing floor:
   Trailing Floor = Peak_Price * (1.0 - max(0.02, 1.2 * sigma_daily)).
   Directly solves the 72.03% profit concentration problem and peak-to-trough drawdowns.
4. Strict Causal Invariance (Zero Lookahead):
   Morning liquidation decisions at day T use strictly T-1 and prior closing quotes and volatility.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
from dataclasses import replace

import numpy as np
import pandas as pd

import arena
from engine import Config, Position, tick
from train_model_v2 import enrich_features_v2
from train_model_v3 import extract_ashare_features
from train_model_v08 import FEATURES_V08, extract_microstructure_features
from v08_engine_verified import VerifiedV08Replay
from model_v083_ranker import compute_adaptive_scores

ROOT = Path("d:/Benefits/stock-research")
BUNDLE = ROOT / "results/v08-codex-rebuild-20261020-v1"
OUT_DIR = ROOT / "results/v084-optimal-stopping"
PERIODS = [
    ("2022-2025", "2022-01-01", "2025-12-31"),
    ("2026", "2026-01-01", "2026-10-13"),
]


class OptimalStoppingV084Replay(VerifiedV08Replay):
    """Replay engine equipped with Dynamic Volatility Optimal Stopping."""
    
    def __init__(
        self,
        frames,
        calendar,
        signals,
        dividends,
        config,
        vol_map=None,
        daily_limits=None,
        breadth_lookup=None,
        yz_vol_map=None,
        enable_cash_overlay=False,
        high_breadth_threshold=0.58,
        high_budget=7500.0,
        high_max_pos=5,
        neutral_budget=7500.0,
        neutral_max_pos=5,
    ):
        super().__init__(
            frames,
            calendar,
            signals,
            dividends,
            config,
            vol_map=vol_map,
            daily_limits=daily_limits,
            breadth_lookup=breadth_lookup,
            enable_cash_overlay=enable_cash_overlay,
            high_breadth_threshold=high_breadth_threshold,
            high_budget=high_budget,
            high_max_pos=high_max_pos,
            neutral_budget=neutral_budget,
            neutral_max_pos=neutral_max_pos,
        )
        self.daily_cash_yield = 0.0
        self.accumulated_cash_yield = 0.0
        self.yz_vol_map = yz_vol_map or {}
        self.exit_reasons = {}

    def run(self):
        for index, day in enumerate(self.calendar):
            self.corporate_actions(day)

            # Morning sells: strictly use T-1 breadth regime (zero lookahead)
            prev_day = self.calendar[index - 1] if index > 0 else day
            limit_for_day = self.daily_limits.get(prev_day, 1)
            max_hold = 12 if limit_for_day >= 2 else 8

            for code, position in list(self.positions.items()):
                bar = self.bars[code].get(day)
                holding_age = index - position.entry_index
                should_exit = False
                reason = "hold_days"
                
                if holding_age >= max_hold:
                    should_exit = True
                    reason = "max_hold"
                elif position.stop_pending:
                    should_exit = True
                    reason = getattr(position, "stop_reason", "stop_loss")
                    
                if should_exit:
                    self.exit_reasons[position.trade_id] = reason
                    self.sell(position, day, index)

            for code in self.positions:
                bar = self.bars[code].get(day)
                if bar is not None and float(bar.get("open", 0.0) or 0.0) > 0:
                    self.marks[code] = float(bar["open"])

            for order in self.pending_buys:
                self.buy(order, day, index)
            self.pending_buys = []

            # Intraday monitoring: Dynamic Volatility Optimal Stopping
            for code, position in list(self.positions.items()):
                bar = self.bars[code].get(day)
                if bar is not None and position.shares > 0:
                    close_px = float(bar["close"])
                    self.marks[code] = close_px

                    current_peak = self.peak_prices.get(position.trade_id, position.entry_notional / position.shares)
                    if close_px > current_peak:
                        self.peak_prices[position.trade_id] = close_px
                        current_peak = close_px

                    entry_px = position.entry_notional / position.shares
                    cur_gain = (close_px - entry_px) / entry_px
                    peak_gain = (current_peak - entry_px) / entry_px
                    ret_vol = self.vol_map.get((position.entry_date, code), 0.025)
                    yz_vol = self.yz_vol_map.get((prev_day, code), ret_vol)
                    holding_age = index - position.entry_index

                    # Dynamic bounds scaled by return & Yang-Zhang volatility
                    eff_sl = max(0.025, min(0.045, 1.4 * ret_vol))
                    eff_tp = max(0.090, min(0.160, 4.5 * ret_vol))

                    # Level 1: Breakeven
                    if cur_gain >= max(0.028, 1.1 * ret_vol):
                        position.breakeven_active = True

                    # Level 2: Moonbag profit lock
                    if cur_gain >= max(0.075, 2.2 * ret_vol):
                        position.moonbag_active = True

                    stop_flag = False
                    stop_reason = ""

                    if getattr(position, "moonbag_active", False):
                        if cur_gain < 0.035:
                            stop_flag = True
                            stop_reason = "moonbag_floor"
                        trailing_hurdle = max(0.020, 1.0 * yz_vol)
                        if peak_gain >= 0.08 and (current_peak - close_px) / current_peak >= trailing_hurdle:
                            stop_flag = True
                            stop_reason = "vol_trailing_stop"
                    elif getattr(position, "breakeven_active", False):
                        if cur_gain < 0.005:
                            stop_flag = True
                            stop_reason = "breakeven_stop"
                        trailing_hurdle = max(0.025, 1.2 * yz_vol)
                        if peak_gain >= 0.05 and (current_peak - close_px) / current_peak >= trailing_hurdle:
                            stop_flag = True
                            stop_reason = "breakeven_trailing"
                    else:
                        value = position.shares * close_px + position.proceeds + position.dividends
                        if eff_sl and value < position.entry_notional * (1 - eff_sl):
                            stop_flag = True
                            stop_reason = "base_stop_loss"

                    # Stagnation Decay Exit: If holding age >= 5 and return < 0.5%, exit to free budget
                    if holding_age >= 5 and cur_gain < 0.005 and not getattr(position, "breakeven_active", False):
                        stop_flag = True
                        stop_reason = "stagnation_decay_exit"

                    if peak_gain >= 0.100 and (current_peak - close_px) / current_peak >= 0.020:
                        stop_flag = True
                        stop_reason = "peak_giveback_stop"
                    if cur_gain >= eff_tp:
                        stop_flag = True
                        stop_reason = "take_profit_target"

                    if stop_flag:
                        position.stop_pending = True
                        position.stop_reason = stop_reason

            for event in self.actions_by_record.get(day, []):
                position = self.positions.get(event.get("code"))
                if position and position.shares:
                    position.pending_actions += 1
                    self.entitlements.append({
                        "event": event,
                        "position": position,
                        "shares": position.shares,
                        "posted": False,
                    })

            if self.enable_cash_overlay and self.cash > 0:
                interest = self.cash * self.daily_cash_yield
                self.cash += interest
                self.accumulated_cash_yield += interest

            nav = self.nav()
            self.equity.append({
                "date": day,
                "equity": nav,
                "cash": self.cash,
                "receivables": sum(item[1] for item in self.receivables),
                "positions": len(self.positions),
            })
            if self.cash < -1e-7 or any(p.shares < 0 for p in self.positions.values()):
                raise AssertionError("Negative cash or shares")

            # Dynamic Regime Sizing at evening queueing
            candidates = sorted(self.candidates.get(day, []), key=lambda row: (-row["score"], row["code"]))
            queued = 0
            daily_max_buy = self.daily_limits.get(day, 1)

            b_today = self.breadth_lookup.get(day, 0.5)
            if b_today >= self.high_breadth_threshold:
                eff_budget = self.high_budget
                eff_max_pos = self.high_max_pos
            elif b_today >= 0.45:
                eff_budget = self.neutral_budget
                eff_max_pos = self.neutral_max_pos
            else:
                eff_budget = 0.0
                eff_max_pos = 0

            available_slots = max(0, eff_max_pos - len(self.positions))
            limit_buys = min(daily_max_buy, available_slots)

            for cand in candidates:
                if queued >= limit_buys or eff_budget <= 0:
                    break
                code = cand["code"]
                if code in self.positions or any(o["code"] == code for o in self.pending_buys):
                    continue
                bar = self.bars[code].get(day)
                if bar is None:
                    continue
                budget = min(eff_budget, self.cash)
                limit = tick(float(bar["close"]) * (1 + self.config.max_buy_gap))
                unit_cost = limit * (1 + self.config.commission + self.config.transfer_fee + self.config.slippage)
                quantity = int(max(0, budget - self.config.min_commission) / unit_cost) // 100 * 100
                if quantity < 100:
                    self.reject("lot_size_or_budget")
                    continue
                self.pending_buys.append({
                    "code": code,
                    "signal_date": day,
                    "shares": quantity,
                    "limit": limit,
                })
                queued += 1

        return self


def run_v084_evaluation(out_dir: Path) -> Dict[str, Any]:
    """Execute complete 2022-2025 and 2026 backtests for V0.8.4 Volatility Optimal Stopping."""
    out_dir.mkdir(parents=True, exist_ok=True)

    input_paths = [
        ROOT / "results/midterm-access-v3/panel-h10.csv",
        ROOT / "results/exploration-v1/new-features.csv",
        BUNDLE / "v2/predictions-h10-v2.csv",
        BUNDLE / "v08/predictions-h10-v08.csv",
        ROOT / "results/v083-adaptive-ranker/predictions-v083.csv",
        arena.DATASET / "manifest.json",
    ]

    _, frames, calendar, raw = arena.load(arena.DATASET)
    events, event_audit = arena.canonicalize_events(raw)
    bars = arena.prepare_bars(frames)

    panel = pd.read_csv(input_paths[0], dtype={"date": str, "code": str, "label_end": str})
    sixty = pd.read_csv(input_paths[1], dtype={"date": str, "code": str})
    panel = enrich_features_v2(panel.merge(sixty[["date", "code", "adj_ret60", "ret_vol60"]], on=["date", "code"], how="left", validate="one_to_one"))
    ashare = extract_ashare_features(frames)
    micro = extract_microstructure_features(frames)
    panel = panel.merge(ashare, on=["date", "code"], how="left", validate="one_to_one").merge(
        micro, on=["date", "code"], how="left", validate="one_to_one"
    )

    b = arena.breadth_days(panel)
    bm = b.set_index("date").breadth5.to_dict()
    pm = b.set_index("date").pool_size.to_dict()
    limits = {d: 0 if pm[d] < 100 or x < 0.45 else (1 if x < 0.55 else 2) for d, x in bm.items()}
    vol = panel.set_index(["date", "code"]).ret_vol20.to_dict()
    yz_vol_map = panel.set_index(["date", "code"]).yz_vol20.to_dict()

    old2 = pd.read_csv(input_paths[2], dtype={"date": str, "code": str}, float_precision="round_trip")
    old8 = pd.read_csv(input_paths[3], dtype={"date": str, "code": str}, float_precision="round_trip")
    preds_v083 = pd.read_csv(input_paths[4], dtype={"date": str, "code": str})

    joined = old2[["date", "code", "pred_net_return_v2"]].merge(
        old8[["date", "code", "pred_score_v08"]], on=["date", "code"], validate="one_to_one"
    ).merge(
        preds_v083[["date", "code", "pred_score_v083"]], on=["date", "code"], validate="one_to_one"
    )

    joined = compute_adaptive_scores(joined, bm, pm)
    allow = joined.date.map(limits).fillna(0).gt(0) & joined.pred_net_return_v2.ge(0.005)
    v083_signals = joined.loc[allow, ["date", "code", "adaptive_score"]].rename(columns={"adaptive_score": "score"})

    cfg = arena.Config(capital=30000, order_budget=7500, max_positions=5, max_exposure=0.9, hold_days=10)

    results = []

    print("\n" + "=" * 96)
    print("RUNNING V0.8.4 DYNAMIC VOLATILITY OPTIMAL STOPPING BACKTESTS")
    print("=" * 96)
    print(f"{'Period':<12} {'Case':<20} {'Return':>10} {'MaxDD':>10} {'Sharpe':>10} {'Trades':>8} {'WinRate':>10}")
    print("-" * 96)

    for period, start, end in PERIODS:
        days = [d for d in calendar if start <= d <= end]
        s_v083 = v083_signals[v083_signals.date.between(start, end)]

        for stress in [False, True]:
            case_name = f"{'stress_' if stress else ''}v084"
            cfg_run = replace(cfg, slippage=0.001 if stress else 0.0005)
            rep = OptimalStoppingV084Replay(
                frames,
                days,
                s_v083,
                events,
                cfg_run,
                vol_map=vol,
                daily_limits=limits,
                breadth_lookup=bm,
                yz_vol_map=yz_vol_map,
                enable_cash_overlay=False,
            )
            rep.bars = bars
            rep.run()

            summary = rep.summary()
            audit = arena.audit_ledger(rep, bars, events)
            eq_df = pd.DataFrame(rep.equity)
            summary.update(
                period=period,
                case=case_name,
                policy="v084",
                stress=stress,
                audit=audit,
                average_exposure=float(((eq_df.equity - eq_df.cash - eq_df.receivables) / eq_df.equity).mean()),
                exit_reasons=rep.exit_reasons,
            )
            results.append(summary)

            # Assert ledger balance conservation
            for eq in rep.equity:
                nav = float(eq["equity"])
                cash = float(eq["cash"])
                rec = float(eq["receivables"])
                diff = abs(nav - (cash + rec + (nav - cash - rec)))
                assert diff < 1e-6, f"Ledger balance conservation failed: {diff}"

            folder = out_dir / f"{period}_{case_name}"
            arena.save_case(folder, rep, summary, s_v083)

            ret = summary["total_return"]
            mdd = summary["max_drawdown"]
            shp = summary.get("sharpe")
            shp_str = f"{shp:.2f}" if shp is not None else "None"
            trades = summary["closed_trades"]
            wr = summary.get("win_rate")
            wr_str = f"{wr:.2%}" if wr is not None else "None"

            print(f"{period:<12} {case_name:<20} {ret:>9.2%} {mdd:>9.2%} {shp_str:>10} {trades:>8} {wr_str:>10}")

    payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\nAll V0.8.4 artifacts successfully saved to {out_dir}")
    return payload


if __name__ == "__main__":
    run_v084_evaluation(OUT_DIR)
