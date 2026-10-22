"""Independent longer-horizon labels for the frozen ML candidate panel.

Candidate selection and features are copied unchanged from ``panel``. Each
trade buys at the next calendar session's open and follows Replay's daily
stop/fixed-horizon rules, with up to 15 additional sessions to obtain a sell.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from engine import tick
from engine_2026 import Replay2026
from ml_features import _date, _event_dates, _n, _quantity


def _label(code, signal_i, rows, calendar, replay, config, action_dates):
    signal = rows[signal_i]
    shares = _quantity(_n(signal.get("close")), config)
    if shares < 100:
        return None, None, None, "insufficient_shares"
    entry_i = signal_i + 1
    if entry_i >= len(calendar):
        return None, None, None, "missing_next_session"
    entry_day = calendar[entry_i]
    entry = rows[entry_i]
    replay.execution_day = entry_day
    allowed, _ = replay.tradable(entry, "buy")
    if not allowed:
        return None, None, None, "entry_not_tradable"
    limit = tick(_n(signal.get("close")) * (1 + config.max_buy_gap))
    entry_open = _n(entry.get("open"))
    if not math.isfinite(entry_open) or entry_open > limit + .0001:
        return None, None, None, "entry_gap_limit"

    entry_notional = shares * entry_open
    buy_fees = sum(config.fees(entry_notional).values())
    stop_value = entry_notional * (1 - config.stop_loss)
    due_i = entry_i + config.hold_days
    last_i = min(len(calendar) - 1, due_i + 15)
    stop_pending = False
    exit_open = None
    exit_day = None
    for i in range(entry_i, last_i + 1):
        day, bar = calendar[i], rows[i]
        # Replay attempts sales at the start of the day, before marking its close.
        if i > entry_i and (stop_pending or i >= due_i):
            replay.execution_day = day
            allowed, _ = replay.tradable(bar, "sell")
            if allowed:
                exit_open, exit_day = _n(bar.get("open")), day
                break
        close = _n(bar.get("close"))
        if math.isfinite(close) and config.stop_loss and shares * close < stop_value:
            stop_pending = True

    if exit_day is None:
        reason = "right_censored" if len(calendar) - 1 < due_i + 15 else "no_exit_after_delay_window"
        return None, None, None, reason

    if any(_date(signal.get("date")) <= d <= exit_day for d in action_dates.get(code, set())):
        return None, None, None, "corporate_action_interval"
    previous_close = _n(signal.get("close"))
    exit_i = i
    for i in range(entry_i, exit_i + 1):
        bar = rows[i]
        preclose = _n(bar.get("preclose"))
        if math.isfinite(previous_close) and math.isfinite(preclose) and previous_close > 0:
            if abs(preclose - previous_close) > .025:
                return None, None, None, "preclose_discontinuity"
        close = _n(bar.get("close"))
        if math.isfinite(close):
            previous_close = close
    sell_fees = sum(config.fees(shares * exit_open, sell=True).values())
    net = (shares * exit_open - sell_fees - entry_notional - buy_fees) / (entry_notional + buy_fees)
    return float(net), bool(net > 0), exit_day, "labeled"


def relabel_panel(panel, frames, calendar, dividends, config, replay_class=Replay2026):
    """Return a copy of the candidate panel with labels rebuilt for ``config``.

    Rows, candidate membership, feature values, and their order are preserved.
    ``label_end`` is the actual sell date; unresolved and masked outcomes remain
    NaN/None with an explanatory ``label_reason``.
    """
    cal = [_date(day) for day in calendar]
    if not cal or cal != sorted(set(cal)):
        raise ValueError("Calendar must be nonempty, increasing, and unique")
    events = list(dividends or [])
    dates = _event_dates(events)
    dense = {}
    for code, raw in frames.items():
        f = raw.copy()
        f["date"] = f["date"].map(_date)
        if "code" not in f:
            f["code"] = code
        f = f.sort_values("date").drop_duplicates("date", keep="last")
        for col in ("open", "high", "low", "close", "preclose", "volume", "tradestatus", "isST"):
            if col in f:
                f[col] = pd.to_numeric(f[col], errors="coerce")
        lookup = f.set_index("date").to_dict("index")
        dense[code] = [{"date": day, **lookup.get(day, {"code": code})} for day in cal]
    # Replay2026 applies the calendar-specific ST sell limit only on/after its
    # effective date; using it for every panel date keeps one per-day path.
    replay = replay_class({}, cal, pd.DataFrame(columns=["date", "code"]), events, config)

    result = panel.copy()
    for column in ("net_return", "y_win"):
        result[column] = pd.Series(np.nan, index=result.index, dtype=object if column == "y_win" else float)
    for column in ("label_end", "label_reason"):
        result[column] = None
    date_to_i = {day: i for i, day in enumerate(cal)}
    for idx, code_value, day_value in zip(result.index, result.code, result.date):
        code, day = str(code_value), _date(day_value)
        if code not in dense or day not in date_to_i:
            reason = "missing_candidate_data"
            outcome = (None, None, None, reason)
        else:
            try:
                outcome = _label(code, date_to_i[day], dense[code], cal, replay, config, dates)
            except (KeyError, TypeError, ValueError, IndexError):
                outcome = (None, None, None, "label_error")
        net, win, end, reason = outcome
        result.at[idx, "label_reason"] = reason
        if end is not None:
            result.at[idx, "net_return"] = net
            result.at[idx, "y_win"] = win
            result.at[idx, "label_end"] = end
    return result
