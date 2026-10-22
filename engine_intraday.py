"""engine_intraday.py - Intraday Touch Execution State Machine and Optimal Stopping Protocol.

Implements real-time intraday price touch execution for A-share small-capital quantitative strategies:
- High/Low bar touch checking:
  * Stop-loss: P_SL = P_entry * (1.0 - eff_sl), triggered when Low <= P_SL
  * Take-profit: P_TP = P_entry * (1.0 + eff_tp), triggered when High >= P_TP
  * Plan A (Dynamic 50% Floating Profit Protection): activated when peak floating gain > 2.5 * sigma_YZ.
    Dynamic floor: P_floor_A = 0.5 * (P_entry + P_peak), triggered when Low <= P_floor_A
  * Plan B (Tight 3.5% Moonbag Floor Lock): activated when Moonbag active (gain >= max(0.075, 2.2 * sigma_ret)).
    Hard floor: P_floor_B = 1.035 * P_entry, triggered when Low <= P_floor_B
    Trailing bound if peak gain >= 0.080: P_trail = P_peak * (1.0 - max(0.020, 1.0 * sigma_YZ))
- Strict Anti-Hindsight Stop-Loss Priority: if BOTH take-profit and stop-loss lines are touched in the same day
  (High >= P_TP and Low <= P_SL), MANDATORY Stop-Loss priority is enforced and Take-Profit is discarded.
- 10 bps additional penalty slippage: execution price gets an additional 0.0010 (10 bps) slippage on all intraday touch exits.
- Strict T+1 locking: positions entered today (calendar_index <= position.entry_index) CANNOT exit intraday.
- Microstructure limit rules: locked at limit down (High == Low == limit_down_price) cannot sell (status: limit_down_locked).
- Gap handling: gap down open below SL fills at Open; gap up open above TP fills at Open (both subject to 10 bps penalty slippage).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

from engine import tick
from engine_core import (
    ExitDecision,
    MarketContext,
    OptimalStoppingProtocol,
    PositionState,
)


def compute_intraday_stop_levels(
    entry_price: float,
    ret_vol: float,
) -> Tuple[float, float, float, float]:
    """Computes stop-loss and take-profit ratios and prices given return volatility.

    Returns:
        (eff_sl, eff_tp, p_sl, p_tp)
    """
    eff_sl = max(0.025, min(0.045, 1.4 * ret_vol))
    eff_tp = max(0.090, min(0.160, 4.5 * ret_vol))
    p_sl = tick(entry_price * (1.0 - eff_sl))
    p_tp = tick(entry_price * (1.0 + eff_tp))
    return eff_sl, eff_tp, p_sl, p_tp


class IntradayTouchStateMachine:
    """Intraday Touch Execution State Machine.

    Evaluates OHLC bars against position state, enforcing strict anti-hindsight
    stop-loss priority, 10 bps execution penalty slippage, T+1 locking, and limit-down locking.
    """

    def __init__(self, plan_mode: str = "plan_a"):
        if plan_mode not in ("plan_a", "plan_b"):
            raise ValueError(f"Unknown plan_mode: {plan_mode!r}. Must be 'plan_a' or 'plan_b'.")
        self.plan_mode = plan_mode

    def evaluate_bar(
        self,
        position: Any,
        bar: Dict[str, Any],
        context: MarketContext,
        plan_mode: Optional[str] = None,
    ) -> Optional[ExitDecision]:
        """Evaluates an OHLC bar for intraday touch exits.

        Args:
            position: PositionState or Position instance.
            bar: Today's OHLC bar dictionary.
            context: MarketContext instance for today.
            plan_mode: Optional override for 'plan_a' or 'plan_b'.

        Returns:
            ExitDecision if an intraday liquidation is triggered, None otherwise.
        """
        # Rule 1: Strict T+1 locking. Fresh entry on day T cannot exit intraday on day T
        entry_index = getattr(position, "entry_index", 0)
        if context.calendar_index <= entry_index:
            return None

        # Bar quote extraction & validity checks
        if bar is None:
            return None
        tradestatus = bar.get("tradestatus", 1)
        volume = float(bar.get("volume", 1.0) or 0.0)
        if tradestatus != 1 or volume <= 0:
            return None

        p_open = float(bar["open"])
        p_high = float(bar["high"])
        p_low = float(bar["low"])
        p_close = float(bar.get("close", p_open))
        p_preclose = float(bar.get("preclose", p_open))
        is_st = bool(bar.get("isST", 0))

        # Rule 2: Limit down open locked microstructure check (cannot sell)
        rate = 0.05 if is_st else 0.10
        limit_down_price = tick(p_preclose * (1.0 - rate))
        if p_high == p_low and abs(p_low - limit_down_price) < 1e-4:
            return None

        # Position cost and peak tracking
        entry_notional = getattr(position, "entry_notional", 0.0)
        shares = getattr(position, "shares", 0)
        p_entry = getattr(position, "entry_price", None)
        if p_entry is None or p_entry <= 0:
            p_entry = entry_notional / shares if shares > 0 else 0.0
        if p_entry <= 0:
            return None

        p_peak = getattr(position, "peak_price", None)
        if p_peak is None or p_peak <= 0:
            p_peak = p_entry
        p_peak = max(p_peak, p_high)
        if hasattr(position, "peak_price"):
            position.peak_price = p_peak

        # Volatility parameters
        entry_date = getattr(position, "entry_date", "")
        code = getattr(position, "code", "")
        ret_vol = context.vol_map.get((entry_date, code), 0.02)
        prev_date = bar.get("prev_date", entry_date)
        yz_vol = context.yz_vol_map.get((prev_date, code))
        if yz_vol is None:
            yz_vol = context.yz_vol_map.get((context.date, code), ret_vol)
        if yz_vol is None:
            yz_vol = ret_vol

        eff_sl, eff_tp, p_sl, p_tp = compute_intraday_stop_levels(p_entry, ret_vol)

        # Determine price touch occurrences
        touched_sl = p_low <= p_sl
        touched_tp = p_high >= p_tp

        # Rule 3: Anti-hindsight Stop-Loss Priority on Sky-and-Earth dual-touch
        if touched_sl and touched_tp:
            exec_price = min(p_open, p_sl)
            return ExitDecision(
                should_exit=True,
                exit_type="intraday_touch",
                target_price=exec_price,
                exit_reason="stop_loss_intraday_priority",
                slippage_bps_penalty=0.0010,  # 10 bps additional slippage penalty
            )

        # Rule 4: Single stop-loss touch
        if touched_sl:
            exec_price = min(p_open, p_sl)
            return ExitDecision(
                should_exit=True,
                exit_type="intraday_touch",
                target_price=exec_price,
                exit_reason="stop_loss",
                slippage_bps_penalty=0.0010,
            )

        # Rule 5: Single take-profit touch
        if touched_tp:
            exec_price = max(p_open, p_tp)
            return ExitDecision(
                should_exit=True,
                exit_type="intraday_touch",
                target_price=exec_price,
                exit_reason="take_profit",
                slippage_bps_penalty=0.0010,
            )

        # Rule 6: Trailing floor exit (Plan A vs Plan B)
        mode = plan_mode or self.plan_mode
        peak_gain = (p_peak - p_entry) / p_entry

        if mode == "plan_a":
            # Plan A: Dynamic 50% floating profit protection
            # Active when peak floating gain > 2.5 * yz_vol
            if peak_gain > 2.5 * yz_vol:
                p_floor_a = tick(0.5 * (p_entry + p_peak))
                if p_low <= p_floor_a:
                    exec_price = min(p_open, p_floor_a)
                    return ExitDecision(
                        should_exit=True,
                        exit_type="intraday_touch",
                        target_price=exec_price,
                        exit_reason="plan_a_floor",
                        slippage_bps_penalty=0.0010,
                    )
        elif mode == "plan_b":
            # Plan B: Tight 3.5% floor lock once Moonbag is activated
            moonbag_threshold = max(0.075, 2.2 * ret_vol)
            is_moonbag = getattr(position, "moonbag_active", False) or (peak_gain >= moonbag_threshold)
            if is_moonbag:
                if hasattr(position, "moonbag_active"):
                    position.moonbag_active = True
                p_floor_b = tick(1.035 * p_entry)
                if p_low <= p_floor_b:
                    exec_price = min(p_open, p_floor_b)
                    return ExitDecision(
                        should_exit=True,
                        exit_type="intraday_touch",
                        target_price=exec_price,
                        exit_reason="plan_b_floor",
                        slippage_bps_penalty=0.0010,
                    )
                # Trailing bound if peak gain >= 0.080
                if peak_gain >= 0.080:
                    p_trail = tick(p_peak * (1.0 - max(0.020, 1.0 * yz_vol)))
                    if p_low <= p_trail:
                        exec_price = min(p_open, p_trail)
                        return ExitDecision(
                            should_exit=True,
                            exit_type="intraday_touch",
                            target_price=exec_price,
                            exit_reason="vol_trailing_stop",
                            slippage_bps_penalty=0.0010,
                        )

        return None


class IntradayTouchStopping:
    """Optimal Stopping Protocol implementation using IntradayTouchStateMachine.

    Satisfies OptimalStoppingProtocol runtime conformance:
    - evaluate_morning: morning open liquidations (max hold or pending stop)
    - evaluate_intraday: intraday touch evaluation
    - check_intraday_touch: intraday touch execution check
    - check_close_stopping: close-of-day stop monitoring and state maintenance
    """

    def __init__(
        self,
        plan_mode: str = "plan_a",
        state_machine: Optional[IntradayTouchStateMachine] = None,
    ):
        self.plan_mode = plan_mode
        self.state_machine = state_machine or IntradayTouchStateMachine(plan_mode=plan_mode)

    def evaluate_morning(
        self,
        position: Any,
        day: str,
        holding_age: int,
        max_hold: int,
        context: MarketContext,
    ) -> Optional[ExitDecision]:
        """Evaluates morning open liquidations."""
        if holding_age >= max_hold:
            return ExitDecision(
                should_exit=True,
                exit_type="morning_open",
                target_price=0.0,
                exit_reason="max_hold",
            )
        if getattr(position, "stop_pending", False):
            reason = getattr(position, "stop_reason", "") or "stop_close_next_open"
            return ExitDecision(
                should_exit=True,
                exit_type="morning_open",
                target_price=0.0,
                exit_reason=reason,
            )
        return ExitDecision(should_exit=False, exit_type="none", target_price=0.0, exit_reason="none")

    def evaluate_intraday(
        self,
        position: Any,
        bar: Dict[str, Any],
        day: str,
        holding_age: int,
        context: MarketContext,
    ) -> Optional[ExitDecision]:
        """Evaluates intraday touch exit."""
        return self.check_intraday_touch(position, bar, context)

    def check_intraday_touch(
        self,
        position: Any,
        bar: Dict[str, Any],
        context: MarketContext,
    ) -> ExitDecision:
        """Evaluates intraday touch triggers against bar High and Low."""
        decision = self.state_machine.evaluate_bar(position, bar, context)
        if decision is not None:
            return decision
        return ExitDecision(
            should_exit=False,
            exit_type="none",
            target_price=0.0,
            exit_reason="none",
            slippage_bps_penalty=0.0,
        )

    def check_close_stopping(
        self,
        position: Any,
        bar: Dict[str, Any],
        context: MarketContext,
    ) -> ExitDecision:
        """Evaluates close-of-day optimal stopping conditions and state updates."""
        if bar is None or getattr(position, "shares", 0) <= 0:
            return ExitDecision(should_exit=False, exit_type="none", target_price=0.0, exit_reason="none")

        close_px = float(bar.get("close", 0.0))
        shares = position.shares
        entry_notional = getattr(position, "entry_notional", 0.0)
        p_entry = getattr(position, "entry_price", None)
        if p_entry is None or p_entry <= 0:
            p_entry = entry_notional / shares if shares > 0 else 0.0

        current_peak = getattr(position, "peak_price", p_entry)
        if close_px > current_peak:
            current_peak = close_px
            position.peak_price = current_peak

        cur_gain = (close_px - p_entry) / p_entry if p_entry > 0 else 0.0
        peak_gain = (current_peak - p_entry) / p_entry if p_entry > 0 else 0.0

        entry_date = getattr(position, "entry_date", "")
        code = getattr(position, "code", "")
        ret_vol = context.vol_map.get((entry_date, code), 0.025)
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
            if hasattr(position, "stop_reason"):
                position.stop_reason = reason
            return ExitDecision(
                should_exit=True,
                exit_type="close_signal",
                target_price=close_px,
                exit_reason=reason,
            )
        return ExitDecision(should_exit=False, exit_type="none", target_price=0.0, exit_reason="none")
