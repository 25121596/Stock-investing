"""Small, past-only features and deterministic masks for hypothesis studies."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _day(value) -> str:
    """Normalize the date keys used by the research panels and price frames."""
    if pd.isna(value):
        return ""
    return str(value)[:10]


def extend_features(panel: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Add 60-session return and volatility to panel rows, using past valid bars.

    A valid bar has ``tradestatus == 1`` and positive volume, raw close, and
    adjusted close. Rolling periods count valid bars for that stock. The input
    rows, order, and existing values are retained exactly; rows without a
    matching price history receive NaN for both added features.
    """
    result = panel.copy()
    if panel.empty:
        result["adj_ret60"] = pd.Series(index=panel.index, dtype=float)
        result["ret_vol60"] = pd.Series(index=panel.index, dtype=float)
        return result
    required_panel = {"date", "code"}
    missing = required_panel.difference(panel.columns)
    if missing:
        raise ValueError(f"panel missing required columns: {sorted(missing)}")

    feature_maps: dict[str, dict[str, dict[str, float]]] = {}
    for code, source in frames.items():
        frame = source.copy()
        if "code" not in frame:
            frame["code"] = code
        required = {"date", "tradestatus", "volume", "close", "adj_close"}
        absent = required.difference(frame.columns)
        if absent:
            raise ValueError(f"frame {code} missing required columns: {sorted(absent)}")
        frame["_day"] = frame["date"].map(_day)
        for col in ("tradestatus", "volume", "close", "adj_close"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.sort_values("_day", kind="stable").drop_duplicates("_day", keep="last")
        valid = (frame["tradestatus"].eq(1) & frame["volume"].gt(0)
                 & frame["close"].gt(0) & frame["adj_close"].gt(0))
        history = frame.loc[valid, ["_day", "adj_close"]].copy()
        adj = history["adj_close"]
        daily_return = adj.pct_change(fill_method=None)
        features = pd.DataFrame({
            "adj_ret60": adj.pct_change(60, fill_method=None),
            "ret_vol60": daily_return.rolling(60, min_periods=60).std(ddof=0),
        }, index=history.index)
        features["_day"] = history["_day"]
        feature_maps[str(code)] = {
            "adj_ret60": dict(zip(features["_day"], features["adj_ret60"])),
            "ret_vol60": dict(zip(features["_day"], features["ret_vol60"])),
        }

    codes = panel["code"].astype(str).to_numpy()
    days = panel["date"].map(_day).to_numpy()
    for name in ("adj_ret60", "ret_vol60"):
        maps = feature_maps
        result[name] = [maps.get(code, {}).get(name, {}).get(day, np.nan)
                        for code, day in zip(codes, days)]
    return result


def breadth_days(panel: pd.DataFrame, return_col: str = 'adj_ret5') -> pd.DataFrame:
    """Summarize positive five-session breadth among that day's candidates.

    ``pool_size`` counts candidate rows. ``positive5_count`` counts rows with
    a known positive ``adj_ret5``; the breadth denominator includes only rows
    with a known value. At least 100 known values are required; missing returns
    are never treated as negative/zero or used to meet that minimum.
    """
    required = {"date", return_col}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"panel missing required columns: {sorted(missing)}")
    if panel.empty:
        return pd.DataFrame(columns=["date", "pool_size", "positive5_count", "breadth5", "allow_entry"])
    work = panel[["date", return_col]].copy()
    work["_day"] = work["date"].map(_day)
    work[return_col] = pd.to_numeric(work[return_col], errors="coerce")
    rows = []
    for day, group in work.groupby("_day", sort=True):
        known = group[return_col].dropna()
        pool_size = len(group)
        positive_count = int(known.gt(0).sum())
        breadth = positive_count / len(known) if len(known) else np.nan
        rows.append({"date": day, "pool_size": pool_size,
                     "positive5_count": positive_count, "breadth5": breadth,
                     "allow_entry": bool(len(known) >= 100 and pd.notna(breadth) and breadth >= 0.5)})
    return pd.DataFrame(rows, columns=["date", "pool_size", "positive5_count", "breadth5", "allow_entry"])


# Deprecated: use week_last_day_mask instead; hardcoding dayofweek == 4 misses holiday weeks.
def friday_mask(dates) -> pd.Series:
    """Return True only for calendar dates that are actually Fridays."""
    if isinstance(dates, pd.Series):
        index = dates.index
        values = dates
    else:
        values = pd.Series(list(dates))
        index = values.index
    parsed = pd.to_datetime(values, errors="coerce")
    return pd.Series(parsed.dt.dayofweek.eq(4).to_numpy(), index=index, dtype=bool)


def week_last_day_mask(dates, calendar) -> pd.Series:
    """Return True for dates that are the last trading day of their week in calendar.

    Correctly handles holiday-shortened weeks (e.g. before Spring Festival,
    National Day, etc.) where the final trading session of the week may be
    Thursday or Wednesday instead of Friday.
    """
    cal_clean = pd.Series([_day(d) for d in calendar if _day(d)]).drop_duplicates().sort_values()
    cal_dt = pd.to_datetime(cal_clean, errors="coerce").dropna()
    if cal_clean.empty or cal_dt.empty:
        if isinstance(dates, pd.Series):
            return pd.Series(False, index=dates.index, dtype=bool)
        return pd.Series([False] * len(dates), dtype=bool)
    cal_clean = cal_clean.loc[cal_dt.index]
    iso = cal_dt.dt.isocalendar()
    last_trading_days = set(cal_clean.groupby([iso["year"], iso["week"]]).last().values)

    if isinstance(dates, pd.Series):
        index = dates.index
        str_dates = dates.map(_day)
    else:
        str_dates = pd.Series([_day(d) for d in dates])
        index = str_dates.index
    return pd.Series(str_dates.isin(last_trading_days).to_numpy(), index=index, dtype=bool)
