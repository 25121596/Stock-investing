"""model_v089_gemini.py - V0.8.9 Research Model (built by Gemini).

Primary mandate:
  1. Overcome the ~85% cash-drag bottleneck and expand 2022-2025 cumulative net return
     from +68.49% to >= +90.00% (target: +90% to +120%) while strictly maintaining
     Maximum Drawdown within <= -12.00% (2022-2025) and <= -3.99% (2026 OOS).
  2. Implement and benchmark all 4 candidate optimization pathways:
     - Variant 1 (Cash Overlay): Automated 2.2% annual rate daily interest on idle cash.
     - Variant 2 (Multi-Regime Adaptive Sizing): Recovery zone 0.48 <= breadth < 0.58.
     - Variant 3 (Concentrated Sizing & Dynamic Compounding): 4 concentrated leader slots
       with floor-capped dynamic NAV compounding budget.
     - Compact Dynamic Holding Horizon: 11-day holding on strong breadth (>=0.58),
       7-day holding on normal/recovery/defensive regimes.
     - Variant 4 (Grand Synthesis V0.8.9-Gemini): Synthesizes Cash Overlay + 4 Concentrated
       Compounding Slots + Compact 11/7 Horizons + Intraday SL Touch Stopping.
  3. Execute 18-case ablation matrix, double-slippage stress (10 bps), 3 random control seeds.
  4. 100% Bit-for-bit Ledger Balance Sheet Conservation:
     |NAV - (Cash + Receivables + Position Value)| < 1e-6 on all 1,154 days.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
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
from train_model_v2 import enrich_features_v2
from train_model_v3 import extract_ashare_features
from train_model_v08 import extract_microstructure_features

OUT_DIR = ROOT / "results/v089-gemini-v1"
BUNDLE = ROOT / "results/v08-codex-rebuild-20261020-v1"

PERIODS = [
    ("2022-2025", "2022-01-01", "2025-12-31"),
    ("2026", "2026-01-01", "2026-10-13"),
]

CANDIDATE_CASE = "v089_gemini"
RANDOM_SEEDS = [2026101401, 2026101402, 2026101403]
INTRADAY_REASON_PREFIX = "intraday_"

# Pre-registered acceptance hurdles for V0.8.9-Gemini
HURDLES = {
    "p1_return_ge": 0.9000,
    "p1_maxdd_ge": -0.1200,
    "p2_return_ge": 0.1500,
    "p2_maxdd_ge": -0.0399,
    "stress_p1_return_ge": 0.8000,
    "stress_p2_return_ge": 0.1400,
    "p1_stressed_return_ge": 0.8000,
    "p2_stressed_return_ge": 0.1400,
    "p1_trades_ge": 120,
    "p2_trades_ge": 14,
    "balance_sheet_conservation": "|NAV - (Cash + Recv + PosVal)| < 1e-6",
}


# =============================================================================
# 1. Breadth Gatekeeper - Gemini
# =============================================================================

class BreadthGatekeeperGemini(GatekeeperProtocol):
    """Breadth gatekeeper conforming to GatekeeperProtocol."""

    def __init__(
        self,
        standard_floor: float = 0.58,
        recovery_floor: float = 0.48,
        recovery_min_er: float = 0.0050,
        recovery_enabled: bool = True,
        breadth_floor: Optional[float] = None,
        **kwargs,
    ):
        if breadth_floor is not None:
            self.standard_floor = float(breadth_floor)
        else:
            self.standard_floor = float(standard_floor)
        self.recovery_floor = float(recovery_floor)
        self.recovery_min_er = float(recovery_min_er)
        self.recovery_enabled = bool(recovery_enabled)
        self.breadth_floor = self.standard_floor

    def evaluate_candidate(
        self,
        code: str,
        date: str,
        features: Optional[Dict[str, Any]],
        context: MarketContext,
    ) -> Tuple[bool, float, str]:
        breadth = float(context.breadth5)
        if breadth < 0:
            return False, breadth, "negative_breadth_block"

        if breadth >= self.standard_floor:
            return True, breadth, "standard_pass"

        if self.recovery_enabled and self.recovery_floor <= breadth < self.standard_floor:
            er = 0.0
            if features:
                if isinstance(features, dict):
                    er = float(features.get("pred_net_return_v2", 0.0) or features.get("score", 0.0) or features.get("er", 0.0))
                else:
                    er = float(getattr(features, "pred_net_return_v2", 0.0))
            if er >= self.recovery_min_er:
                return True, breadth, "recovery_pass"
            return False, breadth, "recovery_insufficient_conviction"

        return False, breadth, "defensive_block"

    def evaluate_order(
        self,
        candidate_row: Any,
        context: MarketContext,
    ) -> Tuple[bool, float, str]:
        if isinstance(candidate_row, dict):
            code = candidate_row.get("code", "")
        else:
            code = getattr(candidate_row, "code", "")
        return self.evaluate_candidate(code, context.date, candidate_row, context)


class AdaptiveGatekeeperGemini(BreadthGatekeeperGemini):
    """Adaptive gatekeeper with recovery zone enabled by default."""

    def __init__(
        self,
        standard_floor: float = 0.58,
        recovery_floor: float = 0.48,
        recovery_min_er: float = 0.0050,
        recovery_enabled: bool = True,
        breadth_floor: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            standard_floor=standard_floor,
            recovery_floor=recovery_floor,
            recovery_min_er=recovery_min_er,
            recovery_enabled=recovery_enabled,
            breadth_floor=breadth_floor,
            **kwargs,
        )


AdaptiveGatekeeperV089 = AdaptiveGatekeeperGemini


# =============================================================================
# 2. Mode-Restricted Intraday Touch Stopping - Gemini
# =============================================================================

class TouchModeStoppingGemini(IntradayTouchStopping):
    """Mode-restricted intraday touch stopping for Gemini architecture.

    'sl': Only the stop-loss touch leg executes intraday (with 10 bps penalty).
          Take-profit and plan_a floor legs remain close-based, exiting next morning
          at open to allow multi-day winners to compound fully.
    """

    _SL_REASONS = ("stop_loss", "stop_loss_intraday_priority")

    def __init__(
        self,
        plan_mode: str = "plan_a",
        intraday_mode: str = "sl",
        hold_strong: int = 11,
        hold_normal: int = 7,
    ):
        super().__init__(plan_mode=plan_mode)
        if intraday_mode not in ("sl", "tp", "all"):
            raise ValueError(f"Unknown intraday_mode: {intraday_mode!r}. Must be 'sl', 'tp' or 'all'.")
        self.intraday_mode = intraday_mode
        self.hold_strong = int(hold_strong)
        self.hold_normal = int(hold_normal)

    def check_intraday_touch(self, position, bar, context) -> ExitDecision:
        decision = self.state_machine.evaluate_bar(position, bar, context)
        if decision is None or not decision.should_exit:
            return ExitDecision(should_exit=False)

        reason = decision.exit_reason
        is_sl = reason in self._SL_REASONS
        if self.intraday_mode == "all":
            keep = True
        elif self.intraday_mode == "sl":
            keep = is_sl
        else:
            keep = not is_sl

        if not keep:
            return ExitDecision(should_exit=False)

        decision.exit_reason = INTRADAY_REASON_PREFIX + reason
        return decision


# =============================================================================
# 3. Dynamic Daily Limits & Compounding Budget Helpers
# =============================================================================

def build_dynamic_daily_limits(bm: Dict[str, float], pm: Dict[str, int], explosive_boost: bool = True) -> Dict[str, int]:
    """Builds daily buy limits with dynamic capacity scaling.

    If explosive_boost is True:
      - pool_size < 100 or breadth < 0.45: 0 buys
      - 0.45 <= breadth < 0.55: 1 buy
      - 0.55 <= breadth < 0.65: 2 buys
      - breadth >= 0.65: 3 buys (dynamic expansion for explosive momentum days)
    """
    limits: Dict[str, int] = {}
    for d, x in bm.items():
        if pm.get(d, 0) < 100 or x < 0.45:
            limits[d] = 0
        elif x < 0.55:
            limits[d] = 1
        elif explosive_boost and x >= 0.65:
            limits[d] = 3
        else:
            limits[d] = 2
    return limits


def compute_dynamic_compounding_budget(
    nav: float,
    slots: int = 4,
    target_exposure: float = 0.90,
    budget_multiplier: float = 1.04,
    min_budget: float = 7500.0,
    max_budget: float = 15000.0,
) -> float:
    """Calculates floor-capped dynamic compounding order budget.

    Formula:
      eff_budget = max(min_budget, min(max_budget, (nav * target_exposure / slots) * budget_multiplier))
    """
    if slots <= 0:
        return 0.0
    raw_budget = (float(nav) * float(target_exposure) / float(slots)) * float(budget_multiplier)
    return float(max(min_budget, min(max_budget, raw_budget)))


class DynamicCompoundingBudget:
    """Class wrapper for dynamic compounding budget calculation."""

    def __init__(
        self,
        slots: int = 4,
        target_exposure: float = 0.90,
        budget_multiplier: float = 1.04,
        min_budget: float = 7500.0,
        max_budget: float = 15000.0,
    ):
        self.slots = int(slots)
        self.target_exposure = float(target_exposure)
        self.budget_multiplier = float(budget_multiplier)
        self.min_budget = float(min_budget)
        self.max_budget = float(max_budget)

    def calculate(self, nav: float) -> float:
        return compute_dynamic_compounding_budget(
            nav=nav,
            slots=self.slots,
            target_exposure=self.target_exposure,
            budget_multiplier=self.budget_multiplier,
            min_budget=self.min_budget,
            max_budget=self.max_budget,
        )

    def __call__(self, nav: float) -> float:
        return self.calculate(nav)

    @classmethod
    def compute(
        cls,
        nav: float,
        slots: int = 4,
        target_exposure: float = 0.90,
        budget_multiplier: float = 1.04,
        min_budget: float = 7500.0,
        max_budget: float = 15000.0,
    ) -> float:
        return compute_dynamic_compounding_budget(
            nav=nav,
            slots=slots,
            target_exposure=target_exposure,
            budget_multiplier=budget_multiplier,
            min_budget=min_budget,
            max_budget=max_budget,
        )


compute_dynamic_budget = compute_dynamic_compounding_budget


# =============================================================================
# 4. Modular Replay Engine - V0.8.9-Gemini Architecture
# =============================================================================

class ReplayEngineV089(ModularReplayEngine):
    """Replay engine integrating:
      - 2.2% annual rate daily cash overlay (r/242 per trading day).
      - Multi-regime capacity & selective recovery zone entries.
      - 4 concentrated slots with floor-capped dynamic compounding budget.
      - Compact holding horizons (11d strong / 7d normal).
      - Bit-for-bit balance sheet conservation identity |NAV - (Cash + Recv + PosVal)| < 1e-6.
    """

    def __init__(
        self,
        *args,
        hold_strong: int = 11,
        hold_normal: int = 7,
        enable_nav_compounding: bool = True,
        compounding_mode: str = "floor_capped",
        target_slots: int = 4,
        target_exposure: float = 0.90,
        budget_multiplier: float = 1.04,
        min_budget: float = 7500.0,
        max_budget: float = 15000.0,
        recovery_max_slots: int = 2,
        high_breadth_threshold: float = 0.58,
        neutral_breadth_threshold: float = 0.48,
        enable_cash_overlay: bool = True,
        cash_annual_rate: float = 0.022,
        **kwargs,
    ):
        super().__init__(
            *args,
            enable_cash_overlay=enable_cash_overlay,
            cash_annual_rate=cash_annual_rate,
            **kwargs,
        )
        self.hold_strong = hold_strong
        self.hold_normal = hold_normal
        self.enable_nav_compounding = enable_nav_compounding
        self.compounding_mode = compounding_mode
        self.target_slots = target_slots
        self.target_exposure = target_exposure
        self.budget_multiplier = budget_multiplier
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.recovery_max_slots = recovery_max_slots
        self.high_breadth_threshold = high_breadth_threshold
        self.neutral_breadth_threshold = neutral_breadth_threshold

    def run(self):
        """Unified daily event-driven execution loop."""
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

            # If gatekeeper does not allow recovery, clamp eff_max_pos to 0 on breadth < floor
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

            # 3. Morning Sells (Holding limit or pending stop at 09:30 open)
            for code, position in list(self.positions.items()):
                bar = self.bars[code].get(day)
                holding_age = index - position.entry_index
                decision = self.stopping.evaluate_morning(position, day, holding_age, max_hold, context)
                if decision and decision.should_exit:
                    self.exit_reasons[position.trade_id] = decision.exit_reason
                    self.sell(position, day, index)
                elif holding_age >= max_hold or position.stop_pending:
                    self.sell(position, day, index)

            # 4. Morning Marks (Update marks to open price for held stocks)
            for code in self.positions:
                bar = self.bars[code].get(day)
                if bar is not None and float(bar.get("open", 0.0) or 0.0) > 0:
                    self.marks[code] = float(bar["open"])

            # 5. Morning Buys (Execute pending buy orders from prior evening)
            for order in self.pending_buys:
                self.buy(order, day, index)
            self.pending_buys = []

            # 6. Intraday Touch Execution (T+1 enforced)
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

            # 9. Cash Management Overlay
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

            # Formulate order budget
            if eff_max_pos > 0:
                if self.enable_nav_compounding:
                    if self.compounding_mode == "floor_capped":
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

        return self


HorizonReplay = ReplayEngineV089


# =============================================================================
# 5. Acceptance Verdict Function
# =============================================================================

def compute_acceptance(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluates pre-registered hurdles against real replay result rows."""

    def find_row(period: str, case: str, stress: bool) -> Optional[Dict[str, Any]]:
        eff_case = f"stress_{case}" if (stress and not case.startswith("stress_")) else case
        for r in rows:
            if r.get("period") == period and r.get("case") == eff_case:
                return r
        return None

    cand_p1 = find_row("2022-2025", CANDIDATE_CASE, False)
    cand_p2 = find_row("2026", CANDIDATE_CASE, False)
    stress_p1 = find_row("2022-2025", CANDIDATE_CASE, True)
    stress_p2 = find_row("2026", CANDIDATE_CASE, True)

    p1_ret = cand_p1.get("total_return", -99.0) if cand_p1 else -99.0
    p1_mdd = cand_p1.get("max_drawdown", -99.0) if cand_p1 else -99.0
    p2_ret = cand_p2.get("total_return", -99.0) if cand_p2 else -99.0
    p2_mdd = cand_p2.get("max_drawdown", -99.0) if cand_p2 else -99.0
    s_p1_ret = stress_p1.get("total_return", -99.0) if stress_p1 else -99.0
    s_p2_ret = stress_p2.get("total_return", -99.0) if stress_p2 else -99.0

    aud_p1 = bool(cand_p1 and cand_p1.get("audit", {}).get("passed", False))
    aud_p2 = bool(cand_p2 and cand_p2.get("audit", {}).get("passed", False))
    aud_pass = aud_p1 and aud_p2

    criteria: Dict[str, Dict[str, Any]] = {
        "p1_return_expansion": {
            "passed": bool(p1_ret >= HURDLES["p1_return_ge"]),
            "actual": float(p1_ret),
            "hurdle": f">= {HURDLES['p1_return_ge']:.4f}",
        },
        "p1_maxdd_safety": {
            "passed": bool(p1_mdd >= HURDLES["p1_maxdd_ge"]),
            "actual": float(p1_mdd),
            "hurdle": f">= {HURDLES['p1_maxdd_ge']:.4f}",
        },
        "p2_return_preservation": {
            "passed": bool(p2_ret >= HURDLES["p2_return_ge"]),
            "actual": float(p2_ret),
            "hurdle": f">= {HURDLES['p2_return_ge']:.4f}",
        },
        "p2_maxdd_safety": {
            "passed": bool(p2_mdd >= HURDLES["p2_maxdd_ge"]),
            "actual": float(p2_mdd),
            "hurdle": f">= {HURDLES['p2_maxdd_ge']:.4f}",
        },
        "stress_p1_return_hurdle": {
            "passed": bool(s_p1_ret >= HURDLES.get("stress_p1_return_ge", HURDLES.get("p1_stressed_return_ge", 0.8000))),
            "actual": float(s_p1_ret),
            "hurdle": f">= {HURDLES.get('stress_p1_return_ge', HURDLES.get('p1_stressed_return_ge', 0.8000)):.4f}",
        },
        "stress_p2_return_hurdle": {
            "passed": bool(s_p2_ret >= HURDLES.get("stress_p2_return_ge", HURDLES.get("p2_stressed_return_ge", 0.1400))),
            "actual": float(s_p2_ret),
            "hurdle": f">= {HURDLES.get('stress_p2_return_ge', HURDLES.get('p2_stressed_return_ge', 0.1400)):.4f}",
        },
        "balance_sheet_conservation": {
            "passed": aud_pass,
            "actual": "100% passed" if aud_pass else "failed",
            "hurdle": HURDLES.get("balance_sheet_conservation", "|NAV - (Cash + Recv + PosVal)| < 1e-6"),
        },
    }

    if cand_p1 and "closed_trades" in cand_p1 and "p1_trades_ge" in HURDLES:
        criteria["p1_trades_expansion"] = {
            "passed": bool(int(cand_p1["closed_trades"]) >= HURDLES["p1_trades_ge"]),
            "actual": int(cand_p1["closed_trades"]),
            "hurdle": f">= {HURDLES['p1_trades_ge']}",
        }
    if cand_p2 and "closed_trades" in cand_p2 and "p2_trades_ge" in HURDLES:
        criteria["p2_trades_expansion"] = {
            "passed": bool(int(cand_p2["closed_trades"]) >= HURDLES["p2_trades_ge"]),
            "actual": int(cand_p2["closed_trades"]),
            "hurdle": f">= {HURDLES['p2_trades_ge']}",
        }

    overall = all(c["passed"] for c in criteria.values())
    return {
        "candidate_case": CANDIDATE_CASE,
        "cost_basis": "normal (5 bps slippage) unless criterion name says stressed (10 bps)",
        "criteria": criteria,
        "overall_pass": bool(overall),
        "pre_registered": True,
        "registered_at": "2026-10-26 (V0.8.9-Gemini pre-registration)",
        "made_by_checkpoint": "Gemini",
    }


# =============================================================================
# 6. Full Evaluation Pipeline - Gemini
# =============================================================================

def run_v089_gemini_evaluation(out_dir: Path = OUT_DIR) -> Dict[str, Any]:
    """Runs the complete V0.8.9-Gemini evaluation across periods, ablations, and stress conditions."""
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("V0.8.9-GEMINI EVALUATION PIPELINE")
    print(f"Output Directory: {out_dir}")
    print("=" * 80)

    # 1. Load data
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

    # Dynamic limits
    limits_boost = build_dynamic_daily_limits(bm, pm, explosive_boost=True)

    # Base candidate signals
    allow_gemini = joined.date.map(limits_boost).fillna(0).gt(0) & joined.pred_net_return_v2.ge(0.0030)
    sig_gemini = joined.loc[allow_gemini, ["date", "code", "adaptive_score", "pred_net_return_v2"]].rename(
        columns={"adaptive_score": "score"}
    )

    cfg = Config(capital=30000, order_budget=7500, max_positions=5, max_exposure=0.9, hold_days=10)
    results: List[Dict[str, Any]] = []

    print("\n" + "=" * 122)
    print(f"{'Period':<12} {'Case':<32} {'Return':>9} {'MaxDD':>9} {'Sharpe':>8} {'Trades':>8} {'WinRate':>9} {'Audit':>7}")
    print("-" * 122)

    # 18-Case Comparative Ablation Suite
    # Period 1: 2022-2025 (13 cases)
    # Period 2: 2026 (5 cases)
    p1_cases = [
        # (case_name, stress, cash_overlay, gatekeeper, dyn_nav, comp_mode, slots, h_s, h_n, rec_slots)
        ("v088_baseline", False, False, BreadthGatekeeperGemini(0.58), False, "pure_nav", 5, 12, 8, 0),
        ("variant_1_cash_overlay", False, True, BreadthGatekeeperGemini(0.58), False, "pure_nav", 5, 12, 8, 0),
        ("variant_2_multi_regime", False, False, AdaptiveGatekeeperGemini(0.58, 0.48, 0.0045), False, "pure_nav", 5, 12, 8, 2),
        ("variant_3_concentrated", False, False, BreadthGatekeeperGemini(0.58), True, "pure_nav", 3, 12, 8, 0),
        ("combo_v1_v2", False, True, AdaptiveGatekeeperGemini(0.58, 0.48, 0.0045), False, "pure_nav", 5, 12, 8, 2),
        ("combo_v1_v3", False, True, BreadthGatekeeperGemini(0.58), True, "floor_capped", 4, 12, 8, 0),
        ("combo_v2_v3", False, False, AdaptiveGatekeeperGemini(0.58, 0.48, 0.0045), True, "pure_nav", 3, 12, 8, 2),
        ("v089_gemini", False, True, BreadthGatekeeperGemini(0.58), True, "floor_capped", 4, 11, 7, 0),
        ("stress_v088_baseline", True, False, BreadthGatekeeperGemini(0.58), False, "pure_nav", 5, 12, 8, 0),
        ("stress_variant_1_cash_overlay", True, True, BreadthGatekeeperGemini(0.58), False, "pure_nav", 5, 12, 8, 0),
        ("stress_variant_2_multi_regime", True, False, AdaptiveGatekeeperGemini(0.58, 0.48, 0.0045), False, "pure_nav", 5, 12, 8, 2),
        ("stress_variant_3_concentrated", True, False, BreadthGatekeeperGemini(0.58), True, "pure_nav", 3, 12, 8, 0),
        ("stress_v089_gemini", True, True, BreadthGatekeeperGemini(0.58), True, "floor_capped", 4, 11, 7, 0),
    ]

    p2_cases = [
        ("v088_baseline", False, False, BreadthGatekeeperGemini(0.58), False, "pure_nav", 5, 12, 8, 0),
        ("variant_1_cash_overlay", False, True, BreadthGatekeeperGemini(0.58), False, "pure_nav", 5, 12, 8, 0),
        ("v089_gemini", False, True, BreadthGatekeeperGemini(0.58), True, "floor_capped", 4, 11, 7, 0),
        ("stress_v088_baseline", True, False, BreadthGatekeeperGemini(0.58), False, "pure_nav", 5, 12, 8, 0),
        ("stress_v089_gemini", True, True, BreadthGatekeeperGemini(0.58), True, "floor_capped", 4, 11, 7, 0),
    ]

    case_plan = [("2022-2025", p1_cases), ("2026", p2_cases)]

    for period, cases in case_plan:
        start = "2022-01-01" if period == "2022-2025" else "2026-01-01"
        end = "2025-12-31" if period == "2022-2025" else "2026-10-13"
        days = [d for d in calendar if start <= d <= end]
        s_period = sig_gemini[sig_gemini.date.between(start, end)]

        for c_name, stress, cash, gate, dyn_nav, comp_mode, slots, h_s, h_n, rec_slots in cases:
            slip = 0.0010 if stress else 0.0005
            cfg_run = replace(cfg, slippage=slip)

            rep = ReplayEngineV089(
                frames=frames,
                calendar=days,
                signals=s_period,
                dividends=events,
                config=cfg_run,
                vol_map=vol,
                daily_limits=limits_boost,
                breadth_lookup=bm,
                yz_vol_map=yz_vol_map,
                ranker=BaselineRanker(),
                gatekeeper=gate,
                stopping=TouchModeStoppingGemini(plan_mode="plan_a", intraday_mode="sl"),
                enable_cash_overlay=cash,
                enable_intraday_touch=True,
                enable_turnover_damping=False,
                enable_nav_compounding=dyn_nav,
                compounding_mode=comp_mode,
                target_slots=slots,
                hold_strong=h_s,
                hold_normal=h_n,
                recovery_max_slots=rec_slots,
            )
            rep.bars = bars
            rep.run()
            summary = rep.summary()
            audit = audit_ledger(rep, bars, events)
            summary.update(period=period, case=c_name, stress=stress, audit=audit)
            results.append(summary)

            case_folder = out_dir / f"{period}_{c_name}"
            if case_folder.exists():
                shutil.rmtree(case_folder)
            arena.save_case(case_folder, rep, summary, s_period)

            # Output comprehensive ledger.csv
            ledger_rows = [
                {
                    "date": eq["date"],
                    "cash": eq["cash"],
                    "receivables": eq["receivables"],
                    "positions": eq["positions"],
                    "equity": eq["equity"],
                    "accumulated_cash_yield": rep.accumulated_cash_yield,
                }
                for eq in rep.equity
            ]
            pd.DataFrame(ledger_rows).to_csv(case_folder / "ledger.csv", index=False)

            print(
                f"{period:<12} {c_name:<32} {summary['total_return']:>8.2%} {summary['max_drawdown']:>8.2%} "
                f"{summary['sharpe']:>8.2f} {summary['closed_trades']:>8} {summary['win_rate']:>8.2%} {str(audit['passed']):>7}"
            )

    # Multi-seed random controls for candidate (normal cost 5 bps)
    random_controls = []
    print("\n" + "=" * 90)
    print("RUNNING V0.8.9-GEMINI CANDIDATE MULTI-SEED RANDOM CONTROLS (normal cost 5 bps)")
    print("=" * 90)

    for period, start, end in PERIODS:
        days = [d for d in calendar if start <= d <= end]
        pool = joined[joined.date.between(start, end)]
        s_period = sig_gemini[sig_gemini.date.between(start, end)]

        for seed in RANDOM_SEEDS:
            s_rnd = arena.matched_candidates(s_period, arena.stable_random(pool, seed))
            rep_rnd = ReplayEngineV089(
                frames=frames,
                calendar=days,
                signals=s_rnd,
                dividends=events,
                config=cfg,
                vol_map=vol,
                daily_limits=limits_boost,
                breadth_lookup=bm,
                yz_vol_map=yz_vol_map,
                ranker=BaselineRanker(),
                gatekeeper=BreadthGatekeeperGemini(0.58),
                stopping=TouchModeStoppingGemini(plan_mode="plan_a", intraday_mode="sl"),
                enable_cash_overlay=True,
                enable_intraday_touch=True,
                enable_turnover_damping=False,
                enable_nav_compounding=True,
                compounding_mode="floor_capped",
                target_slots=4,
                hold_strong=11,
                hold_normal=7,
            )
            rep_rnd.bars = bars
            rep_rnd.run()
            r_sum = rep_rnd.summary()
            r_aud = audit_ledger(rep_rnd, bars, events)
            r_sum.update(period=period, seed=seed, audit=r_aud)
            random_controls.append(r_sum)

            rnd_folder = out_dir / f"{period}_random_seed_{seed}"
            if rnd_folder.exists():
                shutil.rmtree(rnd_folder)
            arena.save_case(rnd_folder, rep_rnd, r_sum, s_rnd)
            ledger_rows_rnd = [
                {
                    "date": eq["date"],
                    "cash": eq["cash"],
                    "receivables": eq["receivables"],
                    "positions": eq["positions"],
                    "equity": eq["equity"],
                    "accumulated_cash_yield": rep_rnd.accumulated_cash_yield,
                }
                for eq in rep_rnd.equity
            ]
            pd.DataFrame(ledger_rows_rnd).to_csv(rnd_folder / "ledger.csv", index=False)

            print(
                f"Random Control {period} Seed {seed:<10}: Return={r_sum['total_return']:>7.2%}, "
                f"MaxDD={r_sum['max_drawdown']:>7.2%}, Trades={r_sum['closed_trades']:>3}, Sharpe={r_sum['sharpe']:>5.2f}, Audit={str(r_aud['passed'])}"
            )

    acceptance = compute_acceptance(results)

    payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "display_version": "V0.8.9-Gemini研究模型",
        "made_by_checkpoint": "Gemini",
        "provenance": "Google Deepmind Antigravity Advanced Agentic Coding",
        "architecture": {
            "signal_admission": "pred_net_return_v2 >= 0.0030 on breadth >= 0.58; recovery entries restricted to ER >= 0.0050",
            "capacity_limits": "Dynamic explosive breadth capacity: 3 buys/day on breadth >= 0.65; 2 buys on 0.55-0.65; 1 buy on 0.45-0.55",
            "slots_and_allocation": "4 concentrated leader slots with floor-capped dynamic NAV compounding: eff_budget = max(7500.0, min(15000.0, (NAV * 0.90 / 4) * 1.04))",
            "stopping_and_horizon": "TouchModeStoppingGemini (intraday SL touch with 10 bps stress; compact dynamic horizon 11d strong / 7d normal)",
            "cash_management_overlay": "Automated 2.2% annual yield overnight cash overlay on idle cash balances",
            "capital_microstructure": "30,000 RMB retail account, 100-share round lots, T+1 settlement, 0.05% stamp tax, 0.001% transfer fee, min 5 RMB commission floor",
        },
        "pre_registered_hurdles": HURDLES,
        "results": results,
        "random_controls": random_controls,
        "acceptance": acceptance,
    }

    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out_dir / "acceptance.json").write_text(json.dumps(acceptance, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out_dir / "display-version.json").write_text(
        json.dumps(
            {
                "display_version": payload["display_version"],
                "made_by_checkpoint": "Gemini",
                "acceptance_pass": acceptance["overall_pass"],
                "note": "V0.8.9模型由 Gemini 检查点自主完成大型改造，验收以 acceptance.json 真实回放数字为准。",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    comp_df = pd.DataFrame(results)[
        ["period", "case", "stress", "total_return", "max_drawdown", "sharpe", "closed_trades", "win_rate", "cagr", "calmar"]
    ]
    comp_df.to_csv(out_dir / "comparison.csv", index=False)

    print("\n" + "=" * 80)
    print(f"ACCEPTANCE VERDICT (real replay rows, pre-registered hurdles): {'PASS' if acceptance['overall_pass'] else 'FAIL'}")
    for name, c in acceptance["criteria"].items():
        actual = "missing" if c["actual"] is None else (f"{c['actual']:.4f}" if isinstance(c["actual"], float) else str(c["actual"]))
        print(f"  {'[PASS]' if c['passed'] else '[FAIL]'} {name:<35} actual={actual} hurdle={c['hurdle']}")
    print(f"\nSaved artifacts to {out_dir}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run V0.8.9-Gemini research model evaluation")
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()
    run_v089_gemini_evaluation(out_dir=Path(args.out_dir))
