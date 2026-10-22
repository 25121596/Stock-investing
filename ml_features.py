"""Past-only feature panel and conservative single-trade labels for ML research.

This module is deliberately separate from the account-level replay engine.  It
creates one candidate row per eligible close and labels that row using the same
order sizing, fees, and opening tradability checks as :mod:`engine`.
"""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np
import pandas as pd

from engine import Config, Replay, tick


FEATURES = [
    "adj_ret1", "adj_ret3", "adj_ret5", "adj_ret10", "adj_ret20",
    "close_ma5", "close_ma20", "close_ma60", "ret_vol20",
    "volume_vs_prev5", "amount_vs_prev20", "intraday_return",
    "range_vs_preclose", "drawdown20", "high_distance20",
    "log_amount_mean20", "market_ret1", "market_ret5", "market_ret20",
    "excess_ret5",
]
_STOCK_FEATURES = [name for name in FEATURES if name not in
                   {"market_ret1", "market_ret5", "market_ret20", "excess_ret5"}]

OUTPUT_COLUMNS = ["date", "code", *FEATURES, "eligible", "net_return", "y_win", "label_end", "label_reason"]


def _n(value, default=np.nan):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _quantity(close: float, config: Config) -> int:
    """Match Replay's precommitted 100-share sizing at signal close."""
    if not math.isfinite(close) or close <= 0:
        return 0
    limit = tick(close * (1 + config.max_buy_gap))
    budget = config.order_budget
    unit_cost = limit * (1 + config.commission + config.transfer_fee + config.slippage)
    return int(max(0, budget - config.min_commission) / unit_cost) // 100 * 100


def _date(value) -> str:
    return str(value)[:10]


def _event_dates(events: list[dict]) -> dict[str, set[str]]:
    dates: dict[str, set[str]] = defaultdict(set)
    for event in events:
        code = str(event.get("code", ""))
        for key in ("dividRegistDate", "dividOperateDate", "dividPayDate", "dividStockMarketDate"):
            value = str(event.get(key, "") or "")
            if value and value.lower() != "nan":
                dates[code].add(value[:10])
    return dates


def _label(code: str, signal_i: int, rows: list[dict], date_to_i: dict[str, int],
           replay: Replay, config: Config, action_dates: dict[str, set[str]]):
    """Return (net_return, y_win, exit_date); unresolved cases stay unlabeled."""
    signal = rows[signal_i]
    shares = _quantity(_n(signal.get("close")), config)
    if shares < 100:
        return None, None, None, "insufficient_shares"
    entry_i = signal_i + 1
    if entry_i >= len(rows):
        return None, None, None, "missing_next_session"
    entry = rows[entry_i]
    entry_allowed, _ = replay.tradable(entry, "buy")
    if not entry_allowed:
        return None, None, None, "entry_not_tradable"
    signal_limit = tick(_n(signal.get("close")) * (1 + config.max_buy_gap))
    entry_open = _n(entry.get("open"))
    if not math.isfinite(entry_open) or entry_open > signal_limit + 0.0001:
        return None, None, None, "entry_gap_limit"

    entry_notional = shares * entry_open
    buy_fees = sum(config.fees(entry_notional).values())
    stop_value = entry_notional * (1 - config.stop_loss)
    max_i = min(len(rows) - 1, entry_i + config.hold_days + 14)  # hold period + 15-session exit buffer
    stop_pending = False
    last_price = entry_open
    exit_open = None
    exit_date = None

    for i in range(entry_i, max_i + 1):
        bar = rows[i]
        # Like Replay, a pending stop or elapsed fixed horizon is attempted at
        # the open, before observing this day's close.
        if i > entry_i and (stop_pending or i - entry_i >= config.hold_days):
            allowed, _ = replay.tradable(bar, "sell")
            if allowed:
                exit_open = _n(bar.get("open"))
                exit_date = _date(bar.get("date"))
                break
        close = _n(bar.get("close"))
        if math.isfinite(close):
            last_price = close
        if config.stop_loss and shares * last_price < stop_value:
            stop_pending = True

    if exit_open is None:
        return None, None, None, "no_exit_by_15_sessions"

    # Any known registration, ex, payment, or bonus-listing event touching the
    # realized holding interval makes the price-only label ambiguous.
    d0, d1 = _date(signal.get("date")), exit_date
    affected = any(d0 <= day <= d1 for day in action_dates.get(code, set()))
    if affected:
        return None, None, None, "corporate_action_interval"

    # Unexplained preclose discontinuities indicate a possible unknown action.
    previous_close = _n(signal.get("close"))
    for i in range(entry_i, date_to_i[exit_date] + 1):
        bar = rows[i]
        preclose = _n(bar.get("preclose"))
        if math.isfinite(previous_close) and math.isfinite(preclose) and previous_close > 0:
            if abs(preclose - previous_close) > 0.025:
                return None, None, None, "preclose_discontinuity"
        close = _n(bar.get("close"))
        if math.isfinite(close):
            previous_close = close

    exit_notional = shares * exit_open
    sell_fees = sum(config.fees(exit_notional, sell=True).values())
    net_return = (exit_notional - sell_fees - entry_notional - buy_fees) / (entry_notional + buy_fees)
    return float(net_return), bool(net_return > 0), exit_date, "labeled"


def build_panel(frames: dict[str, pd.DataFrame], calendar, dividends=None,
                config: Config = Config(hold_days=3)) -> pd.DataFrame:
    """Build eligible close-time candidates, past-only features, and labels.

    Future execution failures and corporate actions affect labels only. They do
    not remove rows or alter ``eligible``, which is determined from information
    available at the signal close.
    """
    cal = [_date(day) for day in calendar]
    if not cal or cal != sorted(set(cal)):
        raise ValueError("Calendar must be nonempty, increasing, and unique")
    date_to_i = {day: i for i, day in enumerate(cal)}
    events = list(dividends or [])
    action_dates = _event_dates(events)
    replay = Replay(frames, cal, pd.DataFrame(columns=["date", "code", "score"]), events, config)

    prepared: dict[str, pd.DataFrame] = {}
    market_daily: dict[str, list[float]] = defaultdict(list)
    for code, raw in frames.items():
        frame = raw.copy()
        if "code" not in frame:
            frame["code"] = code
        frame["date"] = frame["date"].map(_date)
        frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        for col in ("open", "high", "low", "close", "preclose", "volume", "amount", "tradestatus", "isST", "adj_close"):
            if col in frame:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
            else:
                frame[col] = np.nan
        frame["_valid"] = (frame.tradestatus.eq(1) & frame.volume.gt(0)
                           & frame.close.gt(0) & frame.adj_close.gt(0))
        history = frame.loc[frame._valid].copy().reset_index(drop=True)
        if history.empty:
            prepared[code] = frame
            continue
        # Every rolling operation is on valid past/current bars only. Ratios
        # from adjusted close are invariant to later uniform adjustment factors.
        adj = history.adj_close
        raw_close = history.close
        amount = history.amount
        volume = history.volume
        f = pd.DataFrame(index=history.index)
        for period in (1, 3, 5, 10, 20):
            f[f"adj_ret{period}"] = adj.pct_change(period, fill_method=None)
        for period in (5, 20, 60):
            f[f"close_ma{period}"] = adj / adj.rolling(period, min_periods=period).mean() - 1
        daily_ret = adj.pct_change(fill_method=None)
        f["ret_vol20"] = daily_ret.rolling(20, min_periods=20).std(ddof=0)
        f["volume_vs_prev5"] = volume / volume.shift(1).rolling(5, min_periods=5).mean()
        f["amount_vs_prev20"] = amount / amount.shift(1).rolling(20, min_periods=20).mean()
        f["intraday_return"] = history.close / history.open - 1
        f["range_vs_preclose"] = (history.high - history.low) / history.preclose
        adjusted_high = history.high * adj / raw_close
        rolling_high = adjusted_high.rolling(20, min_periods=20).max()
        f["high_distance20"] = adj / rolling_high - 1

        def max_close_drawdown(window):
            peaks = np.maximum.accumulate(window)
            return float(np.min(window / peaks - 1))

        f["drawdown20"] = adj.rolling(20, min_periods=20).apply(max_close_drawdown, raw=True)
        f["log_amount_mean20"] = np.log(amount.rolling(20, min_periods=20).mean().where(lambda x: x > 0))
        history_features = history[["date"]].copy()
        for name in _STOCK_FEATURES:
            history_features[name] = f[name].to_numpy()
        history_features["_valid_count"] = np.arange(1, len(history) + 1)
        history_features["_amount_mean20"] = amount.rolling(20, min_periods=20).mean().to_numpy()
        # Rejoin computed features on each actual valid bar's date.
        feature_by_date = history_features.set_index("date")
        for name in _STOCK_FEATURES + ["_valid_count", "_amount_mean20"]:
            frame[name] = frame.date.map(feature_by_date[name])
        prepared[code] = frame
        for day, value in zip(history.date, daily_ret):
            if math.isfinite(_n(value)):
                market_daily[day].append(float(value))

    market = pd.Series({day: float(np.mean(values)) for day, values in market_daily.items() if values}, dtype=float)
    market = market.reindex(cal)
    market_index = (1 + market.fillna(0)).cumprod()
    mret1 = market
    mret5 = market_index / market_index.shift(5) - 1
    mret20 = market_index / market_index.shift(20) - 1

    output = []
    label_jobs = []
    for code, frame in prepared.items():
        frame_rows = frame.to_dict("records")
        bydate = {row["date"]: row for row in frame_rows}
        row_lookup = bydate
        dense_rows = [row_lookup.get(day, {"date": day, "code": code}) for day in cal]
        for day, row in bydate.items():
            i = date_to_i.get(day)
            if i is None or not row.get("_valid", False):
                continue
            if row.get("_valid_count", 0) < 250:
                continue
            if row.get("isST") != 0 or _n(row.get("amount"), 0) < 0:
                continue
            if not math.isfinite(_n(row.get("_amount_mean20"))) or row["_amount_mean20"] < 20_000_000:
                continue
            shares = _quantity(_n(row.get("close")), config)
            if shares < 100:
                continue
            rec = {"date": day, "code": code, **{name: row.get(name, np.nan) for name in FEATURES}}
            rec["market_ret1"] = _n(mret1.get(day))
            rec["market_ret5"] = _n(mret5.get(day))
            rec["market_ret20"] = _n(mret20.get(day))
            rec["excess_ret5"] = _n(row.get("adj_ret5")) - rec["market_ret5"] if math.isfinite(rec["market_ret5"]) else np.nan
            rec["eligible"] = True
            rec.update(net_return=np.nan, y_win=np.nan, label_end=None, label_reason=None)
            output.append(rec)
            label_jobs.append((len(output) - 1, code, i, dense_rows))

    # Labels are computed after the full inference candidate set is frozen.
    # Global calendar indexing preserves skipped/suspended sessions for purging.
    for out_i, code, sig_i, rows in label_jobs:
        row_date = output[out_i]["date"]
        try:
            net, win, end, reason = _label(code, date_to_i[row_date], rows, date_to_i,
                                   replay, config, action_dates)
        except (KeyError, TypeError, ValueError):
            net, win, end, reason = None, None, None, "label_error"
        output[out_i]["label_reason"] = reason
        if end is not None:
            output[out_i].update(net_return=net, y_win=win, label_end=end)

    return pd.DataFrame(output, columns=OUTPUT_COLUMNS)
