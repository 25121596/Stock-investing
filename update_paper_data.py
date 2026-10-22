"""Manually extend the frozen forward paper-trading snapshot.

This is an offline-by-default module: network access occurs only from the CLI
or an explicit build_snapshot call. It never places orders.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import socket
import time

import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_DONOR = ROOT / "data" / "forward-2026-10-13"
RAW_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"
ADJ_FIELDS = ("adj_open", "adj_high", "adj_low", "adj_close")
RAW_NUMERIC = ("open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg")
RAW_EXACT = ("adjustflag", "tradestatus", "isST")
FINAL_COLUMNS = RAW_FIELDS.split(",") + list(ADJ_FIELDS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _now_shanghai() -> datetime:
    return datetime.now(timezone.utc).astimezone(__import__("zoneinfo").ZoneInfo("Asia/Shanghai"))


def _query_call(name: str):
    def unavailable(**_kwargs):
        raise RuntimeError("live provider call is unavailable in offline fetcher mode")
    unavailable.__name__ = name
    return unavailable


def _validate_donor(donor: Path, expected_count: int) -> tuple[dict, list[str], str]:
    manifest_path = donor / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("provider") != "BaoStock" or manifest.get("end") != "2026-10-13":
        raise ValueError("donor is not the frozen forward-2026-10-13 BaoStock snapshot")
    inventory = manifest.get("inventory", [])
    codes = [str(x["code"]).lower() for x in inventory]
    if len(codes) != expected_count or len(set(codes)) != expected_count:
        raise ValueError(f"donor inventory must contain {expected_count} unique stocks")
    for item, code in zip(inventory, codes):
        for key, suffix in (("sha256", ".csv"), ("dividends_sha256", ".dividends.csv")):
            path = donor / f"{code}{suffix}"
            if not path.is_file() or sha256(path) != item.get(key):
                raise ValueError(f"donor hash mismatch: {path.name}")
        frame = pd.read_csv(donor / f"{code}.csv", dtype={"date": str, "code": str})
        if not set(FINAL_COLUMNS).issubset(frame.columns) or frame.empty:
            raise ValueError(f"invalid donor history: {code}")
        if frame.date.duplicated().any() or not frame.date.is_monotonic_increasing:
            raise ValueError(f"donor dates are not unique/increasing: {code}")
        if set(frame.code.astype(str).str.lower()) != {code}:
            raise ValueError(f"donor code mismatch: {code}")
    cal = pd.read_csv(donor / "calendar.csv", dtype={"calendar_date": str})
    if not {"calendar_date", "is_trading_day"}.issubset(cal.columns) or cal.calendar_date.duplicated().any():
        raise ValueError("invalid donor trade calendar")
    universe = pd.read_csv(donor / "universe.csv", dtype={"code": str})
    if set(universe.code.astype(str).str.lower()) != set(codes):
        raise ValueError("universe and manifest do not cover the same stocks")
    return manifest, codes, sha256(manifest_path)


def _fetch(call, method: str, code: str, params: dict, attempt: int):
    """Call one BaoStock query with serial retry and a bounded socket timeout."""
    import baostock as bs
    from baostock.common import context as bs_context

    if getattr(bs_context, "default_socket", None) is None:
        login = bs.login()
        if login.error_code != "0":
            raise RuntimeError(f"BaoStock login failed: {login.error_code} {login.error_msg}")
    active_socket = getattr(bs_context, "default_socket", None)
    if active_socket is not None:
        active_socket.settimeout(socket.getdefaulttimeout() or 30)
    response = call(**params)
    if response is None or getattr(response, "error_code", None) != "0":
        raise RuntimeError(f"BaoStock query failed: {getattr(response, 'error_code', '?')} "
                           f"{getattr(response, 'error_msg', '')}")
    actual_method = getattr(response, "method", "")
    expected_method = ("query_trade_dates" if method == "query_trade_dates" else
                       "query_dividend_data" if method == "query_dividend_data" else
                       "query_history_k_data_plus")
    if actual_method and actual_method != expected_method:
        raise RuntimeError(f"response method mismatch: expected {expected_method}, got {actual_method}")
    for key in ("code", "start_date", "end_date", "frequency", "adjustflag", "year", "yearType"):
        expected = params.get(key)
        if expected is None or not hasattr(response, key):
            continue
        actual = getattr(response, key)
        if key == "adjustflag" and actual in (None, ""):
            continue
        if actual is None or str(actual).strip().lower() != str(expected).strip().lower():
            raise RuntimeError(f"response {key} mismatch: expected {expected}, got {actual}")
    requested_fields = params.get("fields")
    if requested_fields and hasattr(response, "fields"):
        expected_fields = [x.strip() for x in requested_fields.split(",")]
        if list(response.fields) != expected_fields:
            raise RuntimeError(f"response fields mismatch: expected {expected_fields}, got {list(response.fields)}")
    fields = list(getattr(response, "fields", []))
    rows = []
    while response.next():
        rows.append(dict(zip(fields, response.get_row_data())))
    if getattr(response, "error_code", None) != "0":
        raise RuntimeError(f"BaoStock pagination {response.error_code}: {response.error_msg}")
    try:
        if int(response.per_page_count) > 0 and len(response.data) >= int(response.per_page_count):
            raise RuntimeError(f"unverified pagination end: final page contains {len(response.data)} rows")
    except (AttributeError, TypeError, ValueError):
        pass
    return pd.DataFrame(rows, columns=fields), {"method": actual_method, "fields": fields, "rows": rows}


def _request(call, *, code: str, kind: str, params: dict, log_dir: Path,
             fetcher=None, retries: int = 3) -> pd.DataFrame:
    method = getattr(call, "__name__", "query")
    log_path = log_dir / f"{code}.{kind}.json"
    attempts = []
    for attempt in range(1, retries + 1):
        started = datetime.now(timezone.utc).isoformat()
        try:
            if fetcher:
                result = fetcher(kind=kind, code=code, params=params)
                if isinstance(result, tuple):
                    frame, raw = result
                else:
                    frame, raw = result, result.to_dict(orient="records")
            else:
                socket.setdefaulttimeout(30)
                frame, raw = _fetch(call, method, code, params, attempt)
            attempts.append({"attempt": attempt, "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
                             "params": params, "response": raw})
            _json(log_path, {"code": code, "kind": kind, "method": method, "attempts": attempts})
            return frame.copy()
        except Exception as exc:
            attempts.append({"attempt": attempt, "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
                             "params": params, "error": f"{type(exc).__name__}: {exc}"})
            _json(log_path, {"code": code, "kind": kind, "method": method, "attempts": attempts})
            if not fetcher:
                try:
                    from baostock.common import context as bs_context
                    active_socket = getattr(bs_context, "default_socket", None)
                    if active_socket is not None:
                        active_socket.close()
                    bs_context.default_socket = None
                except Exception:
                    pass
            if attempt < retries:
                time.sleep(min(attempt, 2))
    raise RuntimeError(f"{kind} request failed after {retries} attempts; see {log_path.name}")


def _check_flags(frame: pd.DataFrame, expected: str, code: str) -> None:
    if frame.empty or not {"code", "date", "adjustflag"}.issubset(frame.columns):
        raise ValueError(f"empty or incomplete {expected}-adjusted response for {code}")
    if set(frame.code.astype(str).str.lower()) != {code}:
        raise ValueError(f"response code mismatch for {code}")
    if set(frame.adjustflag.astype(str).str.strip()) != {expected}:
        raise ValueError(f"row-level adjustflag must be {expected} for {code}")
    if frame.date.duplicated().any():
        raise ValueError(f"duplicate response dates for {code}")


def _validate_joined_prices(frame: pd.DataFrame, code: str) -> None:
    required = set(FINAL_COLUMNS)
    if not required.issubset(frame.columns):
        raise ValueError(f"combined price fields missing for {code}")
    numeric = frame[list(RAW_NUMERIC) + list(ADJ_FIELDS)].apply(pd.to_numeric, errors="coerce")
    required_prices = ("open", "high", "low", "close", "preclose", *ADJ_FIELDS)
    if numeric[list(required_prices)].isna().any().any() or not numeric[list(required_prices)].map(lambda x: math.isfinite(float(x))).all().all():
        raise ValueError(f"missing/non-finite required prices for {code}")
    if (numeric[list(required_prices)] <= 0).any().any():
        raise ValueError(f"non-positive price for {code}")
    factor = numeric.adj_close / numeric.close
    for raw_key, adj_key in zip(("open", "high", "low", "close"), ADJ_FIELDS):
        if ((numeric[adj_key] - numeric[raw_key] * factor).abs() > .001).any():
            raise ValueError(f"adjusted/raw daily factor conflict for {code} ({adj_key})")
    status = pd.to_numeric(frame.tradestatus, errors="coerce")
    if status.isna().any() or not status.isin((0, 1)).all():
        raise ValueError(f"invalid tradestatus for {code}")
    flags = pd.to_numeric(frame.adjustflag, errors="coerce")
    if flags.isna().any() or not flags.eq(3).all():
        raise ValueError(f"combined history row adjustflag is not 3 for {code}")


def _compare_overlap(old: pd.Series, new: pd.Series, code: str) -> None:
    def missing(value) -> bool:
        return value is None or (not isinstance(value, str) and pd.isna(value)) or str(value).strip().lower() in ("", "nan", "none")

    old_status = pd.to_numeric(pd.Series([old.get("tradestatus")]), errors="coerce").iloc[0]
    new_status = pd.to_numeric(pd.Series([new.get("tradestatus")]), errors="coerce").iloc[0]
    for key in RAW_NUMERIC:
        old_missing, new_missing = missing(old.get(key)), missing(new.get(key))
        if old_missing or new_missing:
            if old_missing and new_missing and key in ("volume", "amount", "turn", "pctChg") and old_status == new_status == 0:
                continue
            raise ValueError(f"overlap missing/value conflict for {code} {key}")
        try:
            a, b = float(old[key]), float(new[key])
        except (TypeError, ValueError):
            if str(old.get(key, "")) != str(new.get(key, "")):
                raise ValueError(f"overlap missing/value conflict for {code} {key}")
            continue
        tolerance = .5 if key == "volume" else (.02 if key in ("turn", "pctChg") else .011)
        if not math.isfinite(a) or not math.isfinite(b) or abs(a - b) > tolerance:
            raise ValueError(f"raw overlap conflict for {code} {key}: {a} != {b}")
    for key in RAW_EXACT:
        if str(old.get(key, "")).strip() != str(new.get(key, "")).strip():
            raise ValueError(f"raw overlap conflict for {code} {key}")


def _update_stock(code: str, donor: Path, output: Path, end: str, expected_dates: set[str],
                  fetcher=None, retries: int = 3, evidence_subdir: str = "raw-responses") -> dict:
    if fetcher:
        history_call = _query_call("query_history_k_data_plus")
        dividend_call = _query_call("query_dividend_data")
    else:
        import baostock as bs
        history_call, dividend_call = bs.query_history_k_data_plus, bs.query_dividend_data
    old = pd.read_csv(donor / f"{code}.csv", dtype={"date": str, "code": str})
    prior_manifest = json.loads((donor / "manifest.json").read_text(encoding="utf-8"))
    prior_ext = prior_manifest.get("forward_extension", {})
    cache_meta = prior_ext.get("cache_metadata", {}).get(code, {})
    inventory_item = next(x for x in prior_manifest["inventory"] if str(x["code"]).lower() == code)
    cache_bound = (cache_meta.get("raw_sha256") == inventory_item.get("sha256")
                   and cache_meta.get("dividends_sha256") == inventory_item.get("dividends_sha256")
                   and cache_meta.get("donor_manifest_sha256") == prior_ext.get("donor_manifest_sha256"))
    early_codes = set(prior_ext.get("early_ended_codes_reused_without_forward_query", []))
    if cache_bound and cache_meta.get("status") == "verified_delisted_donor" and code in early_codes:
        return {"code": code, "status": "verified_delisted_preserved"}
    start = "2026-10-13"
    raw = _request(history_call, code=code, kind="raw", fetcher=fetcher, retries=retries,
                   log_dir=output / evidence_subdir, params={"code": code, "fields": RAW_FIELDS,
                   "start_date": start, "end_date": end, "frequency": "d", "adjustflag": "3"})
    adj = _request(history_call, code=code, kind="adjusted", fetcher=fetcher, retries=retries,
                   log_dir=output / evidence_subdir, params={"code": code, "fields": "date,code,open,high,low,close,adjustflag",
                   "start_date": start, "end_date": end, "frequency": "d", "adjustflag": "1"})
    _check_flags(raw, "3", code)
    _check_flags(adj, "1", code)
    if set(raw.date.astype(str)) != set(adj.date.astype(str)):
        raise ValueError(f"raw/adjusted dates differ for {code}")
    raw_dates = raw.date.astype(str)
    if raw_dates.duplicated().any() or not raw_dates.is_monotonic_increasing:
        raise ValueError(f"raw dates are duplicated or unordered for {code}")
    anchor = old[old.date == start]
    overlap = raw[raw.date.astype(str) == start]
    if len(anchor) != 1 or len(overlap) != 1:
        raise ValueError(f"required overlap {start} missing for {code}")
    _compare_overlap(anchor.iloc[0], overlap.iloc[0], code)
    adj_keep = adj[["date", "open", "high", "low", "close"]].rename(columns={k: f"adj_{k}" for k in ("open", "high", "low", "close")})
    combined_new = raw.merge(adj_keep, on="date", validate="one_to_one")
    ratio = float(combined_new.loc[combined_new.date.astype(str) == start, "adj_close"].iloc[0]) / float(anchor.adj_close.iloc[0])
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError(f"invalid adjusted-price anchor ratio for {code}")
    for field in ADJ_FIELDS:
        old[field] = pd.to_numeric(old[field], errors="raise") * ratio
    future = combined_new[combined_new.date.astype(str) > start].copy()
    actual_dates = set(future.date.astype(str))
    if actual_dates != expected_dates:
        raise ValueError(f"coverage mismatch for {code}; missing={sorted(expected_dates-actual_dates)[:5]}, extra={sorted(actual_dates-expected_dates)[:5]}")
    result = pd.concat([old, future], ignore_index=True).sort_values("date").reset_index(drop=True)[FINAL_COLUMNS]
    if result.date.duplicated().any():
        raise ValueError(f"duplicate combined dates for {code}")
    _validate_joined_prices(result, code)
    result.to_csv(output / f"{code}.csv", index=False)

    old_div_path = donor / f"{code}.dividends.csv"
    old_div = pd.read_csv(old_div_path, dtype=str, encoding="utf-8-sig").fillna("")
    new_div = _request(dividend_call, code=code, kind="dividends-2026", fetcher=fetcher, retries=retries,
                       log_dir=output / evidence_subdir, params={"code": code, "year": "2026", "yearType": "operate"}).fillna("")
    if not new_div.empty and "code" in new_div and set(new_div.code.astype(str).str.lower()) != {code}:
        raise ValueError(f"dividend response code mismatch for {code}")
    dividends = pd.concat([old_div, new_div], ignore_index=True, sort=False).fillna("").drop_duplicates()
    dividends.to_csv(output / f"{code}.dividends.csv", index=False, encoding="utf-8-sig")
    _json(output / f"{code}.cache.json", {"code": code, "status": "manual_update", "through": end,
          "parent_manifest_sha256": sha256(donor / "manifest.json"), "raw_sha256": sha256(output / f"{code}.csv"),
          "dividends_sha256": sha256(output / f"{code}.dividends.csv"), "anchor_date": start, "anchor_ratio": ratio})
    return {"code": code, "status": "updated", "anchor_ratio": ratio, "last_quote_date": str(max(actual_dates) if actual_dates else start)}


def build_snapshot(donor=DEFAULT_DONOR, output=None, end=None, *, fetcher=None, retries: int = 3,
                   timeout: int = 30, expected_count: int = 300) -> Path:
    """Extend a frozen snapshot through inclusive `end`; fetcher enables offline tests.

    fetcher(kind=..., code=..., params=...) returns a DataFrame (or (DataFrame, raw evidence)).
    """
    donor, output = Path(donor), Path(output) if output is not None else None
    if end is None:
        raise ValueError("end is required (YYYY-MM-DD)")
    end = pd.Timestamp(end).strftime("%Y-%m-%d")
    if end <= "2026-10-13":
        raise ValueError("end must be after the donor cutoff 2026-10-13")
    if not (len(end) == 10 and end[:4] == "2026"):
        raise ValueError("end must be a valid date in 2026")
    shanghai_now = _now_shanghai()
    if shanghai_now.strftime("%Y-%m-%d") < end or (shanghai_now.strftime("%Y-%m-%d") == end and shanghai_now.hour * 60 + shanghai_now.minute < 930):
        raise ValueError("end must have completed its Beijing session (at least 15:30 Asia/Shanghai)")
    if output is None or output.resolve() == donor.resolve() or output.exists():
        raise ValueError("output must be a new, separate directory")
    manifest, codes, donor_manifest_hash = _validate_donor(donor, expected_count)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(donor, output)
    socket.setdefaulttimeout(timeout)
    errors, statuses = [], []
    try:
        if fetcher:
            calendar_call, history_call = _query_call("query_trade_dates"), _query_call("query_history_k_data_plus")
        else:
            import baostock as bs
            calendar_call, history_call = bs.query_trade_dates, bs.query_history_k_data_plus
        calendar = _request(calendar_call, code="market", kind="calendar", fetcher=fetcher, retries=retries,
                            log_dir=output / "raw-responses", params={"start_date": "2026-10-13", "end_date": end})
        if not {"calendar_date", "is_trading_day"}.issubset(calendar.columns):
            raise ValueError("calendar response is missing required columns")
        donor_calendar = pd.read_csv(donor / "calendar.csv", dtype={"calendar_date": str})
        calendar.calendar_date = calendar.calendar_date.astype(str)
        through_end = calendar[calendar.calendar_date <= end]
        extra = through_end[through_end.calendar_date > "2026-10-13"]
        if extra.empty or str(extra.calendar_date.max()) != end:
            raise ValueError("trade calendar does not reach requested end")
        end_row = extra[extra.calendar_date == end]
        if len(end_row) != 1 or pd.to_numeric(end_row.is_trading_day, errors="coerce").iloc[0] != 1:
            raise ValueError(f"end date {end} is not a verified trading day")
        extra_trading = extra[pd.to_numeric(extra.is_trading_day, errors="coerce").eq(1)]
        merged_calendar = pd.concat([donor_calendar, extra_trading], ignore_index=True).drop_duplicates("calendar_date", keep="last").sort_values("calendar_date")
        merged_calendar.to_csv(output / "calendar.csv", index=False)
        expected_dates = set(extra_trading.calendar_date.astype(str))
        if not expected_dates:
            raise ValueError("requested period contains no trading days")
        # A separate forward calendar supports next-session labels while keeping
        # the evaluation calendar strictly bounded at the requested end.
        exchange = _request(calendar_call, code="market", kind="exchange-calendar", fetcher=fetcher, retries=retries,
                            log_dir=output / "raw-responses", params={"start_date": end, "end_date": (pd.Timestamp(end) + pd.Timedelta(days=30)).strftime("%Y-%m-%d")})
        if not {"calendar_date", "is_trading_day"}.issubset(exchange.columns):
            raise ValueError("forward exchange calendar response is missing required columns")
        exchange.calendar_date = exchange.calendar_date.astype(str)
        exchange.to_csv(output / "exchange-calendar.csv", index=False)
        next_days = exchange.loc[(exchange.calendar_date > end) & pd.to_numeric(exchange.is_trading_day, errors="coerce").eq(1), "calendar_date"].astype(str)
        if next_days.empty:
            raise ValueError("exchange calendar has no published next trading day within 30 days")
        next_trading_day = str(next_days.min())

        # Cheap freshness probe prevents 300 empty per-stock attempts when the
        # requested final quote is not yet published. The full stock pass still
        # performs its own identity and overlap checks.
        probe = _request(history_call, code="sh.600006", kind="freshness-probe", fetcher=fetcher, retries=retries,
                         log_dir=output / "raw-responses", params={"code": "sh.600006", "fields": RAW_FIELDS,
                         "start_date": end, "end_date": end, "frequency": "d", "adjustflag": "3"})
        _check_flags(probe, "3", "sh.600006")
        if set(probe.date.astype(str)) != {end}:
            raise ValueError(f"freshness probe lacks complete requested end-date quote: {end}")
        missing_probe = set(RAW_FIELDS.split(",")) - set(probe.columns)
        if missing_probe:
            raise ValueError(f"freshness probe missing raw fields: {sorted(missing_probe)}")
        for field in ("open", "high", "low", "close", "preclose"):
            val = pd.to_numeric(probe[field], errors="coerce")
            if val.isna().any() or not val.map(lambda x: math.isfinite(float(x)) and float(x) > 0).all():
                raise ValueError(f"freshness probe has invalid {field} on {end}")
        for index, code in enumerate(codes, start=1):
            try:
                statuses.append(_update_stock(code, donor, output, end, expected_dates, fetcher, retries))
            except Exception as exc:
                failure = {"code": code, "error": f"{type(exc).__name__}: {exc}"}
                errors.append(failure)
                statuses.append({"code": code, "status": "failed"})
            if index % 20 == 0:
                succeeded = sum(item["status"] not in ("failed", "not_attempted_due_preflight") for item in statuses)
                failed = sum(item["status"] == "failed" for item in statuses)
                print(f"Progress {index}/{len(codes)}: success={succeeded}, failed={failed}", flush=True)
    except Exception as exc:
        errors.append({"code": "__calendar__", "error": f"{type(exc).__name__}: {exc}"})
    known = {x["code"] for x in statuses}
    statuses.extend({"code": code, "status": "not_attempted_due_preflight"} for code in codes if code not in known)
    for item in manifest.get("inventory", []):
        code = str(item["code"]).lower()
        item["sha256"] = sha256(output / f"{code}.csv")
        item["dividends_sha256"] = sha256(output / f"{code}.dividends.csv")
    manifest.update({"end": end, "parent_snapshot": str(donor.resolve()), "parent_manifest_sha256": donor_manifest_hash,
                     "manual_update_utc": datetime.now(timezone.utc).isoformat(), "complete": not errors,
                     "update_statuses": statuses, "failures": errors,
                     "next_trading_day": locals().get("next_trading_day"),
                     "calendar_sha256": sha256(output / "calendar.csv"),
                     "universe_sha256": sha256(output / "universe.csv"),
                     "exchange_calendar_sha256": sha256(output / "exchange-calendar.csv") if (output / "exchange-calendar.csv").is_file() else None,
                     "request_evidence": "raw-responses/*.json; includes UTC attempt time, parameters, raw rows or error"})
    _json(output / "manifest.json", manifest)
    return output


def _validate_repair_source(source: Path, expected_count: int) -> tuple[dict, list[str], list[str], str]:
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not False or not manifest.get("failures"):
        raise ValueError("resume source must be an incomplete snapshot with recorded failures")
    if not str(manifest.get("end", "")).startswith("2026-"):
        raise ValueError("resume source cutoff must be in 2026")
    inventory = manifest.get("inventory", [])
    codes = [str(x["code"]).lower() for x in inventory]
    if len(codes) != expected_count or len(set(codes)) != expected_count:
        raise ValueError(f"resume source inventory must contain {expected_count} unique stocks")
    for item, code in zip(inventory, codes):
        for key, suffix in (("sha256", ".csv"), ("dividends_sha256", ".dividends.csv")):
            path = source / f"{code}{suffix}"
            if not path.is_file() or sha256(path) != item.get(key):
                raise ValueError(f"resume source inventory hash mismatch: {path.name}")
    for filename, field in (("calendar.csv", "calendar_sha256"),
                            ("universe.csv", "universe_sha256"),
                            ("exchange-calendar.csv", "exchange_calendar_sha256")):
        path = source / filename
        if not path.is_file() or not manifest.get(field) or sha256(path) != manifest[field]:
            raise ValueError(f"resume source {filename} hash mismatch")
    universe = pd.read_csv(source / "universe.csv", dtype={"code": str})
    if set(universe.code.astype(str).str.lower()) != set(codes):
        raise ValueError("resume source universe and inventory differ")
    calendar = pd.read_csv(source / "calendar.csv", dtype={"calendar_date": str})
    exchange = pd.read_csv(source / "exchange-calendar.csv", dtype={"calendar_date": str})
    if not {"calendar_date", "is_trading_day"}.issubset(calendar.columns):
        raise ValueError("resume source calendar schema mismatch")
    if not {"calendar_date", "is_trading_day"}.issubset(exchange.columns):
        raise ValueError("resume source exchange-calendar schema mismatch")
    parent_path = Path(manifest.get("parent_snapshot", ""))
    parent_manifest = parent_path / "manifest.json"
    if not parent_path.is_dir() or not parent_manifest.is_file() or sha256(parent_manifest) != manifest.get("parent_manifest_sha256"):
        raise ValueError("resume source original donor manifest hash mismatch")
    # Validate every frozen parent inventory file as well as its manifest
    # fingerprint before allowing repair to rebuild a failed stock from it.
    _validate_donor(parent_path, expected_count)
    failed = [str(x.get("code", "")).lower() for x in manifest["failures"] if x.get("code") != "__calendar__"]
    failed = list(dict.fromkeys(failed))
    if not failed or set(failed) - set(codes):
        raise ValueError("resume source has no repairable stock failures or includes unknown codes")
    return manifest, codes, failed, sha256(manifest_path)


def repair_snapshot(source, output, *, fetcher=None, retries: int = 3, timeout: int = 30,
                    expected_count: int = 300) -> Path:
    """Retry only the failed stock codes from an integrity-checked failed snapshot."""
    source, output = Path(source), Path(output)
    manifest, codes, failed, failed_manifest_hash = _validate_repair_source(source, expected_count)
    if output.resolve() == source.resolve() or output.exists():
        raise ValueError("repair output must be a new directory separate from its source")
    shutil.copytree(source, output)
    end = str(manifest["end"])
    cal = pd.read_csv(source / "calendar.csv", dtype={"calendar_date": str})
    expected_dates = set(cal.loc[(cal.calendar_date > "2026-10-13") &
                                 (pd.to_numeric(cal.is_trading_day, errors="coerce") == 1) &
                                 (cal.calendar_date <= end), "calendar_date"].astype(str))
    if not expected_dates or end not in expected_dates:
        raise ValueError("resume source calendar does not confirm the cutoff trading day")
    socket.setdefaulttimeout(timeout)
    errors, result_by_code = [], {}
    original_donor = Path(manifest["parent_snapshot"])
    for index, code in enumerate(failed, start=1):
        try:
            # A failed stock may already have had its price file written before
            # a later query (for example dividends) failed. Rebuild it from the
            # immutable 2026-10-13 donor to avoid appending duplicate dates.
            result_by_code[code] = _update_stock(code, original_donor, output, end, expected_dates,
                                                 fetcher, retries, evidence_subdir="repair-responses")
        except Exception as exc:
            errors.append({"code": code, "error": f"{type(exc).__name__}: {exc}"})
            result_by_code[code] = {"code": code, "status": "failed"}
        if index % 20 == 0:
            succeeded = sum(x["status"] != "failed" for x in result_by_code.values())
            print(f"Repair progress {index}/{len(failed)}: success={succeeded}, failed={len(errors)}", flush=True)
    previous_statuses = {x["code"]: x for x in manifest.get("update_statuses", [])}
    previous_statuses.update(result_by_code)
    prior_history = list(manifest.get("repair_history", []))
    prior_history.append({"parent_failed_manifest_sha256": failed_manifest_hash,
                          "failures": manifest.get("failures", []),
                          "recorded_utc": datetime.now(timezone.utc).isoformat()})
    new_manifest = dict(manifest)
    for item in new_manifest.get("inventory", []):
        code = str(item["code"]).lower()
        item["sha256"] = sha256(output / f"{code}.csv")
        item["dividends_sha256"] = sha256(output / f"{code}.dividends.csv")
    new_manifest.update({"parent_failed_manifest_sha256": failed_manifest_hash,
                         "repair_history": prior_history,
                         "update_statuses": [previous_statuses[c] for c in codes],
                         "failures": errors, "complete": not errors,
                         "manual_repair_utc": datetime.now(timezone.utc).isoformat(),
                         "calendar_sha256": sha256(output / "calendar.csv"),
                         "universe_sha256": sha256(output / "universe.csv"),
                         "exchange_calendar_sha256": sha256(output / "exchange-calendar.csv")})
    _json(output / "manifest.json", new_manifest)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", help="inclusive cutoff YYYY-MM-DD")
    parser.add_argument("--output", required=True, type=Path, help="new output directory")
    parser.add_argument("--donor", type=Path, default=DEFAULT_DONOR)
    parser.add_argument("--resume-from", type=Path, help="integrity-checked failed snapshot to repair")
    args = parser.parse_args()
    if args.resume_from:
        path = repair_snapshot(args.resume_from, args.output)
    else:
        if not args.end:
            parser.error("--end is required unless --resume-from is used")
        path = build_snapshot(args.donor, args.output, args.end)
    print(path)
    result = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if result.get("failures"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
