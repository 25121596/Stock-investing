"""Cache a historical, reproducible A-share pilot; no brokerage access."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import socket
import time

import baostock as bs
import pandas as pd

ROOT = Path(__file__).resolve().parent
FIELDS = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"


def read_result(result):
    # BaoStock.get_data() still uses removed DataFrame.append on paginated data.
    rows = []
    while result.error_code == "0" and result.next():
        rows.append(result.get_row_data())
    if result.error_code != "0":
        raise RuntimeError(f"BaoStock {result.error_code}: {result.error_msg}")
    return pd.DataFrame(rows, columns=result.fields)


def query(call, **kwargs):
    last = None
    for attempt in range(3):
        try:
            return read_result(call(**kwargs))
        except Exception as error:
            last = error
            if attempt < 2:
                time.sleep(1 + attempt)
    raise RuntimeError(str(last))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20201009)
    parser.add_argument("--start", default="2020-01-02")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--warmup", default="2018-10-01")
    args = parser.parse_args()
    if args.count < 1 or not args.warmup < args.start < args.end:
        parser.error("Require count >= 1 and warmup < start < end")
    folder = ROOT / "data" / f"pilot-{args.start}-{args.end}-n{args.count}-s{args.seed}"
    folder.mkdir(parents=True, exist_ok=True)
    socket.setdefaulttimeout(35)
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(login.error_msg)
    try:
        snapshot_path = folder / "universe.csv"
        if snapshot_path.exists():
            universe = pd.read_csv(snapshot_path, dtype=str)
        else:
            universe = query(bs.query_all_stock, day=args.start)
            universe = universe[universe.code.str.match(r"^(sh\.60\d{4}|sz\.00\d{4})$")]
            universe = universe.sort_values("code").reset_index(drop=True)
            if universe.empty:
                raise RuntimeError("No historical main-board universe returned")
            chosen = sorted(random.Random(args.seed).sample(range(len(universe)), min(args.count, len(universe))))
            universe = universe.iloc[chosen].copy()
            universe.to_csv(snapshot_path, index=False, encoding="utf-8-sig")
        calendar_path = folder / "calendar.csv"
        if not calendar_path.exists():
            calendar = query(bs.query_trade_dates, start_date=args.start, end_date=args.end)
            calendar = calendar[calendar.is_trading_day == "1"]
            if calendar.empty:
                raise RuntimeError("Empty exchange calendar")
            calendar.to_csv(calendar_path, index=False)

        failures, inventory = [], []
        for n, row in enumerate(universe.to_dict("records"), 1):
            code = row["code"]
            path = folder / f"{code}.csv"
            dividend_path = folder / f"{code}.dividends.csv"
            try:
                if not path.exists():
                    raw = query(bs.query_history_k_data_plus, code=code, fields=FIELDS,
                                start_date=args.warmup, end_date=args.end, frequency="d", adjustflag="3")
                    adj = query(bs.query_history_k_data_plus, code=code,
                                fields="date,open,high,low,close", start_date=args.warmup,
                                end_date=args.end, frequency="d", adjustflag="1")
                    if raw.empty or adj.empty:
                        raise RuntimeError("Empty history; not silently replacing historical constituent")
                    adj = adj.rename(columns={key: f"adj_{key}" for key in ("open", "high", "low", "close")})
                    data = raw.merge(adj, on="date", how="left", validate="one_to_one").sort_values("date")
                    for key in data.columns.difference(["date", "code"]):
                        data[key] = pd.to_numeric(data[key], errors="coerce")
                    if data[["open", "close", "adj_close"]].isna().any().any():
                        raise RuntimeError("Incomplete raw/adjusted OHLC data")
                    if (data.close <= 0).any() or (data.adj_close <= 0).any():
                        raise RuntimeError("Non-positive prices")
                    data.to_csv(path, index=False)
                if not dividend_path.exists():
                    parts = [query(bs.query_dividend_data, code=code, year=str(year), yearType="operate")
                             for year in range(int(args.start[:4]), int(args.end[:4]) + 1)]
                    dividends = pd.concat(parts, ignore_index=True).drop_duplicates()
                    dividends.to_csv(dividend_path, index=False, encoding="utf-8-sig")
                inventory.append({"code": code, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                  "dividends_sha256": hashlib.sha256(dividend_path.read_bytes()).hexdigest()})
                print(f"[{n}/{len(universe)}] cached {code}", flush=True)
            except Exception as error:
                failures.append({"code": code, "error": str(error)})
                print(f"[{n}/{len(universe)}] FAILED {code}: {error}", flush=True)
        manifest = {"provider": "BaoStock", "start": args.start, "end": args.end, "warmup": args.warmup,
                    "seed": args.seed, "requested_count": args.count, "universe_as_of": args.start,
                    "universe_policy": "fixed historical main-board snapshot; seeded sample; no replacements",
                    "scope": "pilot sample, excludes later IPOs; not full-market proof",
                    "inventory": inventory, "failures": failures,
                    "downloaded_utc": pd.Timestamp.now(tz="UTC").isoformat()}
        (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"DATASET {folder}", flush=True)
        if failures:
            raise RuntimeError("Incomplete downloads recorded in manifest; simulator refuses incomplete sample")
    finally:
        bs.logout()


if __name__ == "__main__":
    main()
