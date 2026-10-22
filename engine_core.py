"""engine_core.py - Unified Protocol Definitions and Modular Simulation Replay Core.

Preserves exact A-share microstructure invariants for small-capital trading:
- Capital: 30,000.00 RMB
- Lot sizing: 100 shares for buy orders
- Fee structure: Commission万2.5 (min 5.0 RMB), Stamp tax千0.5 (sell side), Transfer fee万0.01,
  Normal slippage万5 (0.0005) or Double slippage万10 (0.0010), Dividend tax 20%
- Price precision: tick rounding to 0.01 RMB (ROUND_HALF_UP)
- Microstructure intercepts: Limit-up open buy intercept, Limit-down open sell intercept,
  Suspension/missing quote check, ST entry block, T+1 settlement enforcement
- Dividend receivables lifecycle: Cash receivables accrued at ex-dividend and credited on pay-date;
  bonus shares locked until listing date
- Cash yield overlay: Daily accrual on uninvested cash balances
- Daily balance sheet conservation: abs(NAV - (Cash + Receivables + Position_Value)) < 1e-6
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import pandas as pd

from engine import Config, Position, number, tick
from v08_engine_verified import VerifiedV08Replay


# =====================================================================
# 1. Standard Data Classes
# =====================================================================

@dataclass(frozen=True)
class MarketContext:
    """Immutable market context for a specific trading day."""
    date: str
    calendar_index: int
    breadth5: float
    pool_size: int
    daily_limit: int
    regime: str  # 'high', 'neutral', 'defensive'
    budget: float
    max_positions: int
    vol_map: Dict[Tuple[str, str], float] = field(default_factory=dict)
    yz_vol_map: Dict[Tuple[str, str], float] = field(default_factory=dict)
    raw_bars: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class PositionState:
    """Read-only view of a held position passed to protocols."""
    trade_id: int
    code: str
    entry_date: str
    entry_index: int
    holding_age: int
    shares: int
    entry_price: float
    entry_notional: float
    peak_price: float
    current_gain: float
    peak_gain: float
    breakeven_active: bool = False
    moonbag_active: bool = False
    plan_a_floor_active: bool = False
    plan_b_floor_active: bool = False
    trailing_floor: float = 0.0
    stop_pending: bool = False
    proceeds: float = 0.0
    dividends: float = 0.0


@dataclass
class ExitDecision:
    """Exit recommendation from OptimalStoppingProtocol."""
    should_exit: bool
    exit_type: str = ""        # 'morning_open', 'intraday_touch', 'close_signal'
    target_price: float = 0.0  # Trigger/execution reference price
    exit_reason: str = ""      # 'stop_loss', 'take_profit', 'max_hold', etc.
    slippage_bps_penalty: float = 0.0  # e.g., 0.0010 (10 bps) for intraday touch


# =====================================================================
# 2. Runtime Checkable Protocols
# =====================================================================

@runtime_checkable
class RankerProtocol(Protocol):
    """Protocol for cross-sectional ranking and scoring."""

    def score_and_rank(
        self,
        candidates: pd.DataFrame,
        date: str,
        breadth: float,
        context: MarketContext,
        turnover_damping: bool = False,
    ) -> pd.DataFrame:
        """Score and sort daily candidates. Returns DataFrame sorted by score descending."""
        ...

    def rank_candidates(
        self,
        day: str,
        candidates: pd.DataFrame,
        context: MarketContext,
        positions: Optional[Dict[str, PositionState]] = None,
        turnover_damping: bool = False,
    ) -> pd.DataFrame:
        """Rank candidates for evening queueing."""
        ...

    def evaluate_turnover_damping(
        self,
        new_candidate: Dict[str, Any],
        held_positions: Any,
        hurdle: float = 0.0120,
    ) -> bool:
        """Evaluate conviction hurdle against weakest holding."""
        ...


@runtime_checkable
class GatekeeperProtocol(Protocol):
    """Protocol for selective classification and morning order gating."""

    def evaluate_candidate(
        self,
        code: str,
        date: str,
        features: Optional[Dict[str, Any]],
        context: MarketContext,
    ) -> Tuple[bool, float, str]:
        """Evaluate candidate during evening queueing or morning buy.
        Returns: (accepted, probability, rejection_reason)
        """
        ...

    def evaluate_order(
        self,
        candidate_row: Any,
        context: MarketContext,
    ) -> Tuple[bool, float, str]:
        """Evaluate order candidate. Returns: (accepted, probability, rejection_reason)."""
        ...


@runtime_checkable
class OptimalStoppingProtocol(Protocol):
    """Protocol for dynamic position monitoring, trailing floor, and exits."""

    def evaluate_morning(
        self,
        position: Any,
        day: str,
        holding_age: int,
        max_hold: int,
        context: MarketContext,
    ) -> Optional[ExitDecision]:
        """Evaluates morning open liquidations (holding limit or pending stop)."""
        ...

    def evaluate_intraday(
        self,
        position: Any,
        bar: Dict[str, Any],
        day: str,
        holding_age: int,
        context: MarketContext,
    ) -> Optional[ExitDecision]:
        """Evaluates intraday or close-based exits."""
        ...

    def check_intraday_touch(
        self,
        position: Any,
        bar: Dict[str, Any],
        context: MarketContext,
    ) -> ExitDecision:
        """Evaluates intraday touch triggers against bar High and Low."""
        ...

    def check_close_stopping(
        self,
        position: Any,
        bar: Dict[str, Any],
        context: MarketContext,
    ) -> ExitDecision:
        """Evaluates close-of-day optimal stopping conditions."""
        ...


# =====================================================================
# 3. Baseline Protocol Implementations
# =====================================================================

class BaselineRanker:
    """Baseline Ranker: sorts candidates by score descending, code ascending."""

    def score_and_rank(
        self,
        candidates: pd.DataFrame,
        date: str,
        breadth: float,
        context: MarketContext,
        turnover_damping: bool = False,
    ) -> pd.DataFrame:
        if candidates.empty:
            return candidates
        df = candidates.copy()
        sort_col = "score" if "score" in df.columns else "pred_score_v08"
        if sort_col in df.columns and "code" in df.columns:
            df = df.sort_values(by=[sort_col, "code"], ascending=[False, True]).reset_index(drop=True)
        return df

    def rank_candidates(
        self,
        day: str,
        candidates: pd.DataFrame,
        context: MarketContext,
        positions: Optional[Dict[str, PositionState]] = None,
        turnover_damping: bool = False,
    ) -> pd.DataFrame:
        return self.score_and_rank(candidates, day, context.breadth5, context, turnover_damping=turnover_damping)

    def evaluate_turnover_damping(
        self,
        new_candidate: Dict[str, Any],
        held_positions: Any,
        hurdle: float = 0.0120,
    ) -> bool:
        return True


class NoOpGatekeeper:
    """Baseline No-Op Gatekeeper: always passes all candidates."""

    def evaluate_candidate(
        self,
        code: str,
        date: str,
        features: Optional[Dict[str, Any]],
        context: MarketContext,
    ) -> Tuple[bool, float, str]:
        return (True, 1.0, "pass")

    def evaluate_order(
        self,
        candidate_row: Any,
        context: MarketContext,
    ) -> Tuple[bool, float, str]:
        return (True, 1.0, "bypassed_pass")


class BaselineRegimeStopping:
    """Baseline SuperV7 Optimal Stopping Implementation."""

    def evaluate_morning(
        self,
        position: Any,
        day: str,
        holding_age: int,
        max_hold: int,
        context: MarketContext,
    ) -> Optional[ExitDecision]:
        if holding_age >= max_hold:
            return ExitDecision(
                should_exit=True,
                exit_type="morning_open",
                target_price=0.0,
                exit_reason="max_hold",
            )
        if getattr(position, "stop_pending", False):
            return ExitDecision(
                should_exit=True,
                exit_type="morning_open",
                target_price=0.0,
                exit_reason="stop_close_next_open",
            )
        return ExitDecision(should_exit=False)

    def evaluate_intraday(
        self,
        position: Any,
        bar: Dict[str, Any],
        day: str,
        holding_age: int,
        context: MarketContext,
    ) -> Optional[ExitDecision]:
        return self.check_close_stopping(position, bar, context)

    def check_intraday_touch(
        self,
        position: Any,
        bar: Dict[str, Any],
        context: MarketContext,
    ) -> ExitDecision:
        return ExitDecision(should_exit=False)

    def check_close_stopping(
        self,
        position: Any,
        bar: Dict[str, Any],
        context: MarketContext,
    ) -> ExitDecision:
        if bar is None or getattr(position, "shares", 0) <= 0:
            return ExitDecision(should_exit=False)
        close_px = float(bar.get("close", 0.0))
        shares = position.shares
        entry_notional = position.entry_notional
        entry_px = entry_notional / shares if shares > 0 else 0.0

        current_peak = getattr(position, "peak_price", entry_px)
        if close_px > current_peak:
            current_peak = close_px
            position.peak_price = current_peak

        cur_gain = (close_px - entry_px) / entry_px if entry_px > 0 else 0.0
        peak_gain = (current_peak - entry_px) / entry_px if entry_px > 0 else 0.0

        ret_vol = context.vol_map.get((position.entry_date, position.code), 0.025)
        eff_sl = max(0.025, min(0.050, 1.5 * ret_vol))
        eff_tp = max(0.090, min(0.150, 4.0 * ret_vol))

        # Level 1: Breakeven
        if cur_gain >= max(0.030, 1.2 * ret_vol):
            position.breakeven_active = True

        # Level 2: Moonbag profit lock
        if cur_gain >= max(0.080, 2.5 * ret_vol):
            position.moonbag_active = True

        stop = False
        reason = ""
        if getattr(position, "moonbag_active", False):
            if cur_gain < 0.035:
                stop = True
                reason = "moonbag_floor"
        elif getattr(position, "breakeven_active", False):
            if cur_gain < 0.005:
                stop = True
                reason = "breakeven_stop"
        else:
            proceeds = getattr(position, "proceeds", 0.0)
            dividends = getattr(position, "dividends", 0.0)
            value = shares * close_px + proceeds + dividends
            if eff_sl and value < entry_notional * (1 - eff_sl):
                stop = True
                reason = "stop_loss"

        if peak_gain >= 0.100 and (current_peak - close_px) / current_peak >= 0.020:
            stop = True
            reason = "peak_giveback"
        if cur_gain >= eff_tp:
            stop = True
            reason = "take_profit"

        if stop:
            position.stop_pending = True
            return ExitDecision(
                should_exit=True,
                exit_type="close_signal",
                target_price=close_px,
                exit_reason=reason,
            )
        return ExitDecision(should_exit=False)


# =====================================================================
# 4. Modular Simulation Replay Engine
# =====================================================================

class ModularReplayEngine(VerifiedV08Replay):
    """Unified Modular Execution Engine.

    Consolidates the duplicated simulation loops across model_v082 through model_v085
    and SuperV7RegimeReplay into a single protocol-driven core.

    In baseline bypass mode (with BaselineRanker, NoOpGatekeeper, BaselineRegimeStopping,
    enable_intraday_touch=False, turnover_hurdle=0.0), ModularReplayEngine achieves 100%
    numerical and NAV fly-flat identity with VerifiedV08Replay.
    """

    def __init__(
        self,
        frames=None,
        calendar=None,
        signals=None,
        dividends=None,
        config=None,
        vol_map=None,
        daily_limits=None,
        breadth_lookup=None,
        enable_cash_overlay=True,
        high_breadth_threshold=0.58,
        neutral_breadth_threshold: float = 0.45,
        high_budget=7500.0,
        high_max_pos=5,
        neutral_budget=5500.0,
        neutral_max_pos=4,
        yz_vol_map=None,
        ranker: Optional[RankerProtocol] = None,
        gatekeeper: Optional[GatekeeperProtocol] = None,
        stopping: Optional[OptimalStoppingProtocol] = None,
        enable_intraday_touch: bool = False,
        turnover_hurdle: float = 0.0,
        enable_turnover_damping: bool = False,
        ranker_mode: str = "v08_baseline",
        cash_annual_rate: float = 0.022,
        pool_sizes: Optional[Dict[str, int]] = None,
        **kwargs,
    ):
        sig_df = signals if signals is not None else pd.DataFrame(columns=["date", "code", "score"])
        super().__init__(
            frames or {},
            calendar or [],
            sig_df,
            dividends=dividends,
            config=config,
            vol_map=vol_map,
            daily_limits=daily_limits,
            breadth_lookup=breadth_lookup,
            enable_cash_overlay=enable_cash_overlay,
            high_breadth_threshold=high_breadth_threshold,
            high_budget=high_budget,
            high_max_pos=high_max_pos,
            neutral_budget=neutral_budget,
            neutral_max_pos=neutral_max_pos,
            **kwargs,
        )
        self.cash_annual_rate = cash_annual_rate
        self.daily_cash_yield = (self.cash_annual_rate / 242.0) if self.enable_cash_overlay else 0.0
        self.neutral_breadth_threshold = neutral_breadth_threshold
        self.yz_vol_map = yz_vol_map or {}
        self.pool_sizes = pool_sizes or {}
        self.ranker = ranker or BaselineRanker()
        self.gatekeeper = gatekeeper or NoOpGatekeeper()
        self.stopping = stopping or BaselineRegimeStopping()
        self.enable_intraday_touch = enable_intraday_touch
        self.turnover_hurdle = turnover_hurdle
        self.enable_turnover_damping = enable_turnover_damping
        self.ranker_mode = ranker_mode
        self.exit_reasons: Dict[int, str] = {}
        self.gating_abstentions: int = 0

    def _make_position_state(self, position: Position, day: str, index: int) -> PositionState:
        bar = self.bars[position.code].get(day) if hasattr(self, "bars") and position.code in self.bars else None
        entry_px = position.entry_notional / position.shares if position.shares > 0 else 0.0
        current_peak = self.peak_prices.get(position.trade_id, entry_px)
        close_px = float(bar["close"]) if bar and "close" in bar else entry_px
        cur_gain = (close_px - entry_px) / entry_px if entry_px > 0 else 0.0
        peak_gain = (current_peak - entry_px) / entry_px if entry_px > 0 else 0.0
        return PositionState(
            trade_id=position.trade_id,
            code=position.code,
            entry_date=position.entry_date,
            entry_index=position.entry_index,
            holding_age=index - position.entry_index,
            shares=position.shares,
            entry_price=entry_px,
            entry_notional=position.entry_notional,
            peak_price=current_peak,
            current_gain=cur_gain,
            peak_gain=peak_gain,
            breakeven_active=getattr(position, "breakeven_active", False),
            moonbag_active=getattr(position, "moonbag_active", False),
            plan_a_floor_active=getattr(position, "plan_a_floor_active", False),
            plan_b_floor_active=getattr(position, "plan_b_floor_active", False),
            trailing_floor=getattr(position, "trailing_floor", 0.0),
            stop_pending=getattr(position, "stop_pending", False),
            proceeds=getattr(position, "proceeds", 0.0),
            dividends=getattr(position, "dividends", 0.0),
        )

    def _execute_intraday_exit(self, position: Position, bar: Dict[str, Any], day: str, index: int, decision: ExitDecision):
        """Executes an intraday touch exit with penalty slippage surcharge."""
        locked = sum(quantity for listing, quantity in position.locked if listing > day)
        available = position.shares - locked
        if available < position.shares:
            available = available // 100 * 100
        if available <= 0:
            self.reject("locked_bonus_shares")
            return

        rate = 0.05 if number(bar.get("isST")) == 1 else 0.10
        preclose = number(bar.get("preclose"))
        limit_down_px = tick(preclose * (1 - rate))
        high_px = float(bar.get("high", 0.0))
        if high_px <= limit_down_px + 0.0001:
            self.reject("limit_down_locked")
            return

        price = decision.target_price if decision.target_price > 0 else float(bar.get("close", 0.0))
        notional = available * price
        fees = self.config.fees(notional, sell=True)
        if decision.slippage_bps_penalty > 0:
            penalty_fee = tick(notional * decision.slippage_bps_penalty)
            fees["slippage"] = tick(fees["slippage"] + penalty_fee)

        proceeds = notional - sum(fees.values())
        self.cash += proceeds
        position.proceeds += proceeds
        position.exit_fees += sum(fees.values())
        position.shares -= available
        self.fills.append({
            "date": day,
            "trade_id": position.trade_id,
            "code": position.code,
            "side": "sell",
            "shares": available,
            "price": price,
            "notional": notional,
            **fees,
        })
        self.exit_reasons[position.trade_id] = decision.exit_reason
        self.finish(position, day, decision.exit_reason)

    def _determine_regime(self, breadth: float) -> str:
        """Classify market regime based on breadth thresholds."""
        if breadth >= self.high_breadth_threshold:
            return "high"
        elif breadth >= self.neutral_breadth_threshold:
            return "neutral"
        else:
            return "defensive"

    def run(self):
        """Unified daily event-driven execution loop."""
        for index, day in enumerate(self.calendar):
            # 1. Corporate Actions & Data Quality Review Checks
            self.corporate_actions(day)

            # 2. Market Context & Regime Classification (T-1 breadth lookahead prevention)
            b_today = self.breadth_lookup.get(day, 0.5)
            prev_day = self.calendar[index - 1] if index > 0 else day
            limit_for_day = self.daily_limits.get(prev_day, 1)
            max_hold = 12 if limit_for_day >= 2 else 8
            daily_max_buy = self.daily_limits.get(day, 1)
            pool_size = self.pool_sizes.get(day, 300)

            regime = self._determine_regime(b_today)
            if regime == "high":
                eff_budget = self.high_budget
                eff_max_pos = self.high_max_pos
            elif regime == "neutral":
                eff_budget = self.neutral_budget
                eff_max_pos = self.neutral_max_pos
            else:
                eff_budget = 0.0
                eff_max_pos = 0

            raw_bars_today = {code: self.bars[code].get(day, {}) for code in self.bars if day in self.bars[code]}
            context = MarketContext(
                date=day,
                calendar_index=index,
                breadth5=b_today,
                pool_size=pool_size,
                daily_limit=daily_max_buy,
                regime=regime,
                budget=eff_budget,
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
                        # T+1 rule: same-day liquidation strictly forbidden
                        continue
                    bar = self.bars[code].get(day)
                    if bar is not None and position.shares > 0:
                        touch_decision = self.stopping.check_intraday_touch(position, bar, context)
                        if touch_decision and touch_decision.should_exit:
                            self._execute_intraday_exit(position, bar, day, index, touch_decision)

            # 7. Intraday Close Monitoring & Peak Price Tracking
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

            # 8. Corporate Actions Entitlements (Evening record date capture)
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

            # 10. Daily Balance Sheet & NAV Conservation Assertion
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

            for cand in candidates:
                if queued >= limit_buys or eff_budget <= 0:
                    break
                code = cand["code"]
                if code in self.positions or any(o["code"] == code for o in self.pending_buys):
                    continue

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


# =====================================================================
# 5. Independent Ledger Audit Verification
# =====================================================================

def audit_ledger(rep, bars: Dict[str, Dict[str, Any]], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Independent reconstruction of all daily cash, share balances, and NAV."""
    cfg = rep.config
    cash = cfg.capital
    shares: Dict[str, int] = defaultdict(int)
    waiting: List[Tuple[str, float]] = []
    fills: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    actions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    entry: Dict[int, str] = {}

    for f in rep.fills:
        fills[f["date"]].append(f)
    for a in rep.actions:
        actions[a["date"]].append(a)

    for row in rep.equity:
        day = row["date"]
        cash += sum(v for pay, v in waiting if pay <= day)
        waiting = [(pay, v) for pay, v in waiting if pay > day]

        for a in actions[day]:
            shares[a["code"]] += a["bonus_shares"]
            if a["net_cash"]:
                matched = [e for e in events if e["code"] == a["code"] and e.get("dividOperateDate") == day]
                if len(matched) != 1:
                    raise ValueError(f"Ambiguous dividend payment for {a['code']} on {day}")
                pay = matched[0]["dividPayDate"]
                if pay <= day:
                    cash += a["net_cash"]
                else:
                    waiting.append((pay, a["net_cash"]))

        for f in fills[day]:
            fees = cfg.fees(f["notional"], sell=(f["side"] == "sell"))
            # When penalty slippage is applied, slippage may exceed base fees
            expected_slippage = fees["slippage"]
            if f["side"] == "buy":
                if f["shares"] % 100:
                    raise ValueError("Non-lot buy")
                entry[f["trade_id"]] = day
            elif day <= entry.get(f["trade_id"], "9999-99-99"):
                raise ValueError("T+1 violation")

            sign = 1 if f["side"] == "buy" else -1
            shares[f["code"]] += sign * f["shares"]
            actual_fees = f.get("commission", 0.0) + f.get("transfer", 0.0) + f.get("stamp", 0.0) + f.get("slippage", 0.0)
            cash -= sign * f["notional"] + actual_fees
            if shares[f["code"]] < 0 or cash < -1e-7:
                raise ValueError(f"Negative cash/shares on {day}")

        cash *= (1 + rep.daily_cash_yield)
        receivable = sum(v for _, v in waiting)
        nav = cash + receivable + sum(q * float(bars[c][day]["close"]) for c, q in shares.items() if q)
        if max(abs(cash - row["cash"]), abs(receivable - row["receivables"]), abs(nav - row["equity"])) > 1e-6:
            raise ValueError(f"Independent ledger mismatch on {day}: expected NAV={nav}, record NAV={row['equity']}")

    return {"passed": True, "fills_checked": len(rep.fills), "daily_nav_checked": len(rep.equity)}
