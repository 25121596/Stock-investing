"""Call the downloaded project's actual rule functions with past-only inputs."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys

import pandas as pd

REPO = Path(__file__).resolve().parent.parent / "stock"
STRATEGIES = ("volume", "volume_trend", "turtle")
LABELS = {"volume": "原放量上涨", "volume_trend": "放量上涨＋趋势过滤", "turtle": "原海龟突破"}


def source_hashes():
    return {name: hashlib.sha256((REPO / "instock/core/strategy" / name).read_bytes()).hexdigest()
            for name in ("enter.py", "turtle_trade.py")}


def past_frame(raw: pd.DataFrame) -> pd.DataFrame:
    # Zero-volume suspension placeholders are retained in the account calendar,
    # but are not actual trading bars for indicator lookback periods.
    data = raw[(raw.tradestatus == 1) & (raw.volume > 0)].copy()
    data["volume"] = data["volume"].astype("float64")
    # Re-anchor adjusted history to the signal day's actual price. Future
    # corporate actions then cannot change today's price/amount thresholds.
    anchor = float(data.iloc[-1]["close"]) / float(data.iloc[-1]["adj_close"])
    for key in ("open", "high", "low", "close"):
        data[key] = data[f"adj_{key}"] * anchor
    data["p_change"] = data.close.pct_change(fill_method=None).fillna(0.0) * 100
    return data


def evaluate(raw: pd.DataFrame) -> dict[str, bool]:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    volume = importlib.import_module("instock.core.strategy.enter").check_volume
    turtle = importlib.import_module("instock.core.strategy.turtle_trade").check_enter
    frame = past_frame(raw)
    if frame.empty:
        return dict.fromkeys(STRATEGIES, False)
    day = str(frame.iloc[-1]["date"])
    code = str(frame.iloc[-1]["code"])
    key = (day, code.split(".")[-1], code)
    vol = bool(volume(key, frame.copy()))
    trend = (len(frame) >= 60 and frame.close.iloc[-1] >= frame.close.tail(20).mean()
             and frame.close.tail(20).mean() > frame.close.tail(60).mean())
    return {"volume": vol, "volume_trend": bool(vol and trend),
            "turtle": bool(turtle(key, frame.copy()))}


def generate(folder: Path, frames: dict[str, pd.DataFrame], start: str, end: str):
    identity = {"source": source_hashes(), "adapter_version": 2,
                "manifest": hashlib.sha256((folder / "manifest.json").read_bytes()).hexdigest()}
    path, meta = folder / "signals.csv", folder / "signals.meta.json"
    if path.exists() and meta.exists() and json.loads(meta.read_text()) == identity:
        return pd.read_csv(path, dtype={"code": str, "date": str})
    rows = []
    for n, (code, frame) in enumerate(frames.items(), 1):
        mature = (frame.tradestatus == 1).cumsum()
        for index, bar in frame.iterrows():
            if (not start <= bar.date <= end or bar.tradestatus != 1
                    or bar.isST != 0 or mature.iloc[index] < 250):
                continue
            history = frame.iloc[:index + 1]
            flags = evaluate(history)
            # Tie-breaking uses historical liquidity, never subsequent returns.
            score = float(history.amount.tail(20).mean())
            for strategy, selected in flags.items():
                if selected:
                    rows.append({"date": bar.date, "code": code, "strategy": strategy, "score": score})
        print(f"signals [{n}/{len(frames)}] {code}", flush=True)
    result = pd.DataFrame(rows, columns=["date", "code", "strategy", "score"])
    result.to_csv(path, index=False)
    meta.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    return result

