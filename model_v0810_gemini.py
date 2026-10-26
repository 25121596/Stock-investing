"""model_v0810_gemini.py - V0.8.10 Stability-Enhanced Quantitative Model (built by Gemini).

Stability Optimization Mandate:
  1. Mitigate compounding risk during low-breadth regimes:
     - Breadth-Adaptive Compounding Budget: Dynamic compounding budget (up to 15,000 RMB)
       only when market breadth >= 0.58 (strong bull / expansion).
     - When breadth < 0.58 (defensive/sideways), clamp budget strictly to base (7,500 RMB)
       to prevent oversized positions from taking huge RMB losses in choppy markets (fixing the 2025 drawdown).
  2. Integrate Broker Auto Cash Sweep (1.85% annual yield):
     - Completely automated overnight yield on idle cash.
  3. Target Metrics:
     - Expand 2025 return from +2.01% to >= +5.00%.
     - Maintain 2022-2025 cumulative net return >= +85.00%.
     - Reduce Max Drawdown from -8.38% to <= -7.50%.
     - Boost Sharpe ratio from 1.59 to >= 1.70+.
     - Maintain 100% bit-for-bit balance sheet conservation (|NAV - (Cash+Recv+PosVal)| < 1e-6).
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import arena
from engine import Config, Position, number, tick
from engine_core import (
    BaselineRanker,
    ExitDecision,
    GatekeeperProtocol,
    MarketContext,
    ModularReplayEngine,
    OptimalStoppingProtocol,
    PositionState,
    audit_ledger,
)
from engine_intraday import IntradayTouchStopping
from model_v083_ranker import compute_adaptive_scores
from model_v089_gemini import (
    BreadthGatekeeperGemini,
    ReplayEngineV089,
    TouchModeStoppingGemini,
    build_dynamic_daily_limits,
    compute_dynamic_compounding_budget,
)
from train_model_v08 import extract_microstructure_features
from train_model_v2 import enrich_features_v2
from train_model_v3 import extract_ashare_features

BUNDLE = ROOT / "results/v08-codex-rebuild-20261020-v1"
OUT_DIR = ROOT / "results/v0810-gemini-v1"


class ReplayEngineV0810(ReplayEngineV089):
    """V0.8.10 Replay Engine with Breadth-Adaptive Compounding and Stability Controls."""

    def __init__(
        self,
        *args,
        adaptive_budget_scaling: bool = True,
        cash_annual_rate: float = 0.0185, # Broker Auto Sweep 1.85%
        **kwargs,
    ):
        super().__init__(
            *args,
            cash_annual_rate=cash_annual_rate,
            **kwargs,
        )
        self.adaptive_budget_scaling = adaptive_budget_scaling

    def run(self):
        """Unified daily event-driven execution loop with adaptive compounding budget."""
        for index, day in enumerate(self.calendar):
            # 1. Corporate Actions
            self.corporate_actions(day)

            # 2. Market Context & Dynamic Horizons
            b_today = self.breadth_lookup.get(day, 0.5)
            prev_day = self.calendar[index - 1] if index > 0 else day
            limit_for_day = self.daily_limits.get(prev_day, 1)
            max_hold = self.hold_strong if limit_for_day >= 2 else self.hold_normal
            daily_max_buy = self.daily_limits.get(day, 1)
            pool_size = self.pool_sizes.get(day, 300)

            # Effective slots by regime
            regime = self._determine_regime(b_today)
            if regime == "high":
                eff_max_pos = self.target_slots
            elif regime == "neutral":
                eff_max_pos = min(self.recovery_max_slots, self.target_slots)
            else:
                eff_max_pos = 0

            if self.gatekeeper is not None:
                if isinstance(self.gatekeeper, BreadthGatekeeperGemini) and not self.gatekeeper.recovery_enabled:
                    if b_today < self.gatekeeper.breadth_floor:
                        eff_max_pos = 0

            raw_bars_today = {code: self.bars[code].get(day, {}) for code in self.bars if day in self.bars[code]}
            context = MarketContext(
                date=day,
                calendar_index=index,
                breadth5=b_today,
                pool_size=pool_size,
                daily_limit=daily_max_buy,
                regime=regime,
                budget=0.0,
                max_positions=eff_max_pos,
                vol_map=self.vol_map,
                yz_vol_map=self.yz_vol_map,
                raw_bars=raw_bars_today,
            )

            # 3. Morning Sells
            for code, position in list(self.positions.items()):
                bar = self.bars[code].get(day)
                holding_age = index - position.entry_index
                decision = self.stopping.evaluate_morning(position, day, holding_age, max_hold, context)
                if decision and decision.should_exit:
                    self.exit_reasons[position.trade_id] = decision.exit_reason
                    self.sell(position, day, index)
                elif holding_age >= max_hold or position.stop_pending:
                    self.sell(position, day, index)

            # 4. Morning Marks
            for code in self.positions:
                bar = self.bars[code].get(day)
                if bar is not None and float(bar.get("open", 0.0) or 0.0) > 0:
                    self.marks[code] = float(bar["open"])

            # 5. Morning Buys
            for order in self.pending_buys:
                self.buy(order, day, index)
            self.pending_buys = []

            # 6. Intraday Touch Execution
            if self.enable_intraday_touch:
                for code, position in list(self.positions.items()):
                    if index <= position.entry_index:
                        continue
                    bar = self.bars[code].get(day)
                    if bar is not None and position.shares > 0:
                        touch_decision = self.stopping.check_intraday_touch(position, bar, context)
                        if touch_decision and touch_decision.should_exit:
                            self._execute_intraday_exit(position, bar, day, index, touch_decision)

            # 7. Close Monitoring & Peak Price Tracking
            for code, position in list(self.positions.items()):
                bar = self.bars[code].get(day)
                if bar is not None and position.shares > 0:
                    close_px = float(bar["close"])
                    self.marks[code] = close_px
                    current_peak = self.peak_prices.get(position.trade_id, position.entry_notional / position.shares)
                    if close_px > current_peak:
                        self.peak_prices[position.trade_id] = close_px
                        current_peak = close_px
                    position.peak_price = current_peak
                    self.stopping.check_close_stopping(position, bar, context)

            # 8. Corporate Actions Entitlements
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

            # 9. Cash Management Overlay (Broker Auto Sweep)
            if self.enable_cash_overlay and self.cash > 0:
                interest = self.cash * self.daily_cash_yield
                self.cash += interest
                self.accumulated_cash_yield += interest

            # 10. Balance Sheet Conservation Identity Assertion
            nav = self.nav()
            rec_val = sum(item[1] for item in self.receivables)
            pos_val = sum(p.shares * self.marks.get(p.code, 0.0) for p in self.positions.values())

            if abs(nav - (self.cash + rec_val + pos_val)) >= 1e-6:
                raise AssertionError(
                    f"Daily balance sheet conservation violated on {day}: "
                    f"NAV={nav}, Cash={self.cash}, Receivables={rec_val}, Positions={pos_val}, "
                    f"Diff={abs(nav - (self.cash + rec_val + pos_val))}"
                )
            if self.cash < -1e-7 or any(p.shares < 0 for p in self.positions.values()):
                raise AssertionError(f"Negative cash or shares on {day}: cash={self.cash}")

            self.equity.append({
                "date": day,
                "equity": nav,
                "cash": self.cash,
                "receivables": rec_val,
                "positions": len(self.positions),
            })

            # 11. Evening Candidate Ranking, Filtering & Sizing for Tomorrow
            candidates_raw = self.candidates.get(day, [])
            if candidates_raw:
                cand_df = pd.DataFrame(candidates_raw)
                pos_states = {c: self._make_position_state(p, day, index) for c, p in self.positions.items()}
                ranked_df = self.ranker.rank_candidates(
                    day,
                    cand_df,
                    context,
                    positions=pos_states,
                    turnover_damping=self.enable_turnover_damping,
                )
                candidates = ranked_df.to_dict("records")
            else:
                candidates = []

            available_slots = max(0, eff_max_pos - len(self.positions))
            limit_buys = min(daily_max_buy, available_slots)
            queued = 0

            # Formulate order budget with Stability Adaptive Budget Scaling
            if eff_max_pos > 0:
                if self.enable_nav_compounding:
                    if self.compounding_mode == "floor_capped":
                        if self.adaptive_budget_scaling and b_today < self.high_breadth_threshold:
                            # Clamp to defensive baseline budget during weak/sideways regimes
                            eff_budget = self.min_budget
                        else:
                            eff_budget = compute_dynamic_compounding_budget(
                                nav,
                                slots=self.target_slots,
                                target_exposure=self.target_exposure,
                                budget_multiplier=self.budget_multiplier,
                                min_budget=self.min_budget,
                                max_budget=self.max_budget,
                            )
                    else:
                        eff_budget = (nav * self.target_exposure) / eff_max_pos
                else:
                    eff_budget = self.config.order_budget
            else:
                eff_budget = 0.0

            for cand in candidates:
                if queued >= limit_buys or eff_budget <= 0:
                    break
                code = cand["code"]
                if code in self.positions or any(o["code"] == code for o in self.pending_buys):
                    continue

                if self.gatekeeper is not None:
                    allowed, prob, reason = self.gatekeeper.evaluate_candidate(code, day, cand, context)
                    if not allowed:
                        self.gating_abstentions += 1
                        self.reject(reason or "gatekeeper_abstain")
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


def run_benchmarks():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _, frames, calendar, raw = arena.load(arena.DATASET)
    events, _ = arena.canonicalize_events(raw)
    bars = arena.prepare_bars(frames)

    panel = pd.read_csv(ROOT / "results/midterm-access-v3/panel-h10.csv", dtype={"date": str, "code": str, "label_end": str})
    sixty = pd.read_csv(ROOT / "results/exploration-v1/new-features.csv", dtype={"date": str, "code": str})
    panel = enrich_features_v2(panel.merge(sixty[["date", "code", "adj_ret60", "ret_vol60"]], on=["date", "code"], how="left", validate="one_to_one"))
    panel = panel.merge(extract_ashare_features(frames), on=["date", "code"], how="left", validate="one_to_one").merge(
        extract_microstructure_features(frames), on=["date", "code"], how="left", validate="one_to_one"
    )

    b = arena.breadth_days(panel)
    bm = b.set_index("date").breadth5.to_dict()
    pm = b.set_index("date").pool_size.to_dict()
    vol = panel.set_index(["date", "code"]).ret_vol20.to_dict()
    yz_vol_map = panel.set_index(["date", "code"]).yz_vol20.to_dict()

    old2 = pd.read_csv(BUNDLE / "v2/predictions-h10-v2.csv", dtype={"date": str, "code": str})
    old8 = pd.read_csv(BUNDLE / "v08/predictions-h10-v08.csv", dtype={"date": str, "code": str})
    preds_v083 = pd.read_csv(ROOT / "results/v083-adaptive-ranker/predictions-v083.csv", dtype={"date": str, "code": str})

    joined = old2[["date", "code", "pred_net_return_v2"]].merge(
        old8[["date", "code", "pred_score_v08"]], on=["date", "code"], validate="one_to_one"
    ).merge(
        preds_v083[["date", "code", "pred_score_v083"]], on=["date", "code"], validate="one_to_one"
    )
    joined = compute_adaptive_scores(joined, bm, pm)

    limits_boost = build_dynamic_daily_limits(bm, pm, explosive_boost=True)
    allow = joined.date.map(limits_boost).fillna(0).gt(0) & joined.pred_net_return_v2.ge(0.0030)
    signals = joined.loc[allow, ["date", "code", "adaptive_score", "pred_net_return_v2"]].rename(
        columns={"adaptive_score": "score"}
    )

    cfg_base = Config(capital=30000, order_budget=7500, max_positions=5, max_exposure=0.9, hold_days=10)

    models_to_test = [
        ("V0.8.9-AutoSweep (Base Compounding)", False),
        ("V0.8.10-Gemini (Adaptive Stability)", True),
    ]

    all_res = {}
    for name, adapt_flag in models_to_test:
        for period in ["2022-2025", "2026"]:
            start = "2022-01-01" if period == "2022-2025" else "2026-01-01"
            end = "2025-12-31" if period == "2022-2025" else "2026-10-13"
            days = [d for d in calendar if start <= d <= end]
            s_period = signals[signals.date.between(start, end)]

            rep = ReplayEngineV0810(
                frames=frames,
                calendar=days,
                signals=s_period,
                dividends=events,
                config=cfg_base,
                vol_map=vol,
                daily_limits=limits_boost,
                breadth_lookup=bm,
                yz_vol_map=yz_vol_map,
                ranker=BaselineRanker(),
                gatekeeper=BreadthGatekeeperGemini(0.58),
                stopping=TouchModeStoppingGemini(plan_mode="plan_a", intraday_mode="sl"),
                enable_cash_overlay=True,
                cash_annual_rate=0.0185, # Broker Auto Sweep
                enable_nav_compounding=True,
                adaptive_budget_scaling=adapt_flag,
                target_slots=4,
                hold_strong=11,
                hold_normal=7,
                recovery_max_slots=0,
            )
            rep.bars = bars
            rep.run()
            summ = rep.summary()
            audit = audit_ledger(rep, bars, events)

            # Yearly stats
            eq_df = pd.DataFrame(rep.equity)
            eq_df["year"] = eq_df["date"].str[:4]
            yearly = {}
            for yr, y_df in eq_df.groupby("year"):
                y_ret = (y_df["equity"].iloc[-1] / y_df["equity"].iloc[0]) - 1.0
                roll_peak = y_df["equity"].cummax()
                y_dd = float(((y_df["equity"] - roll_peak) / roll_peak).min())
                yearly[yr] = {"return": y_ret, "maxdd": y_dd}

            key = f"{name}_{period}"
            all_res[key] = {
                "model": name,
                "period": period,
                "total_return": summ["total_return"],
                "max_drawdown": summ["max_drawdown"],
                "sharpe": summ["sharpe"],
                "calmar": summ["total_return"] / abs(summ["max_drawdown"]),
                "trades": summ["closed_trades"],
                "win_rate": summ["win_rate"],
                "final_equity": summ["final_equity"],
                "accumulated_cash_yield": rep.accumulated_cash_yield,
                "stock_profit": summ["final_equity"] - 30000.0 - rep.accumulated_cash_yield,
                "audit_passed": audit["passed"],
                "yearly": yearly,
            }

    with open(OUT_DIR / "v0810_benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 115)
    print("V0.8.10-GEMINI STABILITY BENCHMARK COMPARISON")
    print("=" * 115)
    for k, v in all_res.items():
        print(
            f"{k:<38} | Ret: {v['total_return']:>7.2%} | MaxDD: {v['max_drawdown']:>6.2%} | Sharpe: {v['sharpe']:>4.2f} | "
            f"Calmar: {v['calmar']:>5.2f} | Trades: {v['trades']} | WinRate: {v['win_rate']:>6.2%} | Audit: {v['audit_passed']}"
        )
        for yr, yst in v["yearly"].items():
            print(f"   Year {yr}: Return {yst['return']:>6.2%}, MaxDD {yst['maxdd']:>6.2%}")
        print("-" * 115)


if __name__ == "__main__":
    run_benchmarks()
