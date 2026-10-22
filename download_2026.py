"""Extend the frozen 300-stock sample through 2026-10-13 without replacements."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import socket

import baostock.common.context as bs_context
import pandas as pd

import download as source
import expand_download as prior

ROOT = Path(__file__).resolve().parent
START = "2020-01-02"
WARMUP = "2018-10-01"
DONOR_END = "2025-12-31"
END = "2026-10-13"  # Inclusive; the current session (2026-10-14) is excluded.
COUNT = 300
SEED = 20201009
CODE_RE = re.compile(r"^(sh\.60\d{4}|sz\.00\d{4})$")
RAW_FIELDS = source.FIELDS.split(",")
ADJ_FIELDS = ("adj_open", "adj_high", "adj_low", "adj_close")
FINAL_COLUMNS = RAW_FIELDS + list(ADJ_FIELDS)
RAW_NUMERIC = ("open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg")
RAW_EXACT = ("adjustflag", "tradestatus", "isST")
PRICE_TOL = 0.011
FROZEN_DIR = ROOT / "results" / "forward-2026-frozen"
FROZEN_JSON = FROZEN_DIR / "freeze.json"
EXPECTED_MODEL_HASHES = {
    "2026-logistic.joblib": "b431468e333eefb88977de818ccbaebd484d19c7e33271706ff908c269f5e1db",
    "2026-lgb.txt": "c8da745e730f668ac04cb70f3b0163f267360030af5fa7b538457d794aa83075",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path, encoding: str | None = None) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False, encoding=encoding)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(value: dict, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _close_socket() -> None:
    sock = getattr(bs_context, "default_socket", None)
    if sock is not None:
        try:
            sock.close()
        except Exception:
            pass
        finally:
            bs_context.default_socket = None


def _row_flags(frame: pd.DataFrame, expected: str, code: str) -> None:
    if frame.empty or "code" not in frame or set(frame.code.astype(str).str.lower()) != {code.lower()}:
        raise RuntimeError(f"empty data or code mismatch for {code}")
    if "adjustflag" not in frame:
        raise RuntimeError(f"missing row-level adjustflag for {code}")
    flags = set(frame.adjustflag.astype(str).str.strip())
    if flags != {expected}:
        raise RuntimeError(f"row adjustflag mismatch for {code}: expected {expected}, got {sorted(flags)}")
    if frame.date.duplicated().any():
        raise RuntimeError(f"duplicate query dates for {code}")


def _compare_raw_overlap(old: pd.Series, new: pd.Series, code: str) -> dict:
    mismatches = []
    def value(item, key):
        raw = item.get(key)
        try:
            missing = raw is None or pd.isna(raw) or str(raw).strip() == ""
        except (TypeError, ValueError):
            missing = False
        if missing:
            return None, True
        try:
            result = float(raw)
            return (result, False) if math.isfinite(result) else (None, True)
        except (TypeError, ValueError):
            return None, False

    old_status, _ = value(old, "tradestatus")
    new_status, _ = value(new, "tradestatus")
    for key in RAW_NUMERIC:
        left, old_missing = value(old, key)
        right, new_missing = value(new, key)
        if left is None or right is None:
            both_missing_allowed = (old_missing and new_missing and
                                    (key in ("turn", "pctChg") or
                                     (key in ("volume", "amount") and old_status == new_status == 0)))
            if not both_missing_allowed:
                mismatches.append({"field": key, "old": str(old.get(key)), "new": str(new.get(key)),
                                   "reason": "one_sided_or_required_missing"})
            continue
        tolerance = 0.5 if key == "volume" else (0.011 if key in ("open", "high", "low", "close", "preclose", "amount") else 0.02)
        if abs(left - right) > tolerance:
            mismatches.append({"field": key, "old": left, "new": right, "tolerance": tolerance})
    for key in RAW_EXACT:
        left, right = str(old.get(key, "")).strip(), str(new.get(key, "")).strip()
        if left != right:
            mismatches.append({"field": key, "old": left, "new": right, "reason": "value_mismatch"})
    if mismatches:
        raise RuntimeError(f"2025-12-31 raw overlap mismatch for {code}: {mismatches}")
    return {"date": DONOR_END, "checked_fields": list(RAW_NUMERIC + RAW_EXACT),
            "status": "matched_with_tolerance"}


def _validate_raw_adjusted(frame: pd.DataFrame, code: str) -> None:
    required = set(FINAL_COLUMNS)
    if not required.issubset(frame.columns):
        raise RuntimeError(f"combined columns missing for {code}: {sorted(required-set(frame.columns))}")
    if frame.empty or frame.code.nunique() != 1 or str(frame.code.iloc[0]).lower() != code.lower():
        raise RuntimeError(f"combined code mismatch for {code}")
    if frame.date.duplicated().any() or not frame.date.is_monotonic_increasing:
        raise RuntimeError(f"combined dates are not strictly increasing for {code}")
    required_price = ("open", "high", "low", "close", "preclose", *ADJ_FIELDS)
    required = frame[list(required_price)].apply(pd.to_numeric, errors="coerce")
    if required.isna().any().any() or not required.map(lambda x: math.isfinite(float(x))).all().all():
        raise RuntimeError(f"missing or non-finite required price for {code}")
    if (required[list(("open", "high", "low", "close", *ADJ_FIELDS))] <= 0).any().any() or (required["preclose"] <= 0).any():
        raise RuntimeError(f"invalid raw or adjusted OHLC values for {code}")
    status = pd.to_numeric(frame.tradestatus, errors="coerce")
    st = pd.to_numeric(frame.isST, errors="coerce")
    adjust = pd.to_numeric(frame.adjustflag, errors="coerce")
    if status.isna().any() or not status.isin((0, 1)).all():
        raise RuntimeError(f"invalid tradestatus flags for {code}")
    if st.isna().any() or not st.isin((0, 1)).all():
        raise RuntimeError(f"invalid isST flags for {code}")
    if adjust.isna().any() or not adjust.eq(3).all():
        raise RuntimeError(f"invalid raw adjustflag for {code}")
    for key in ("volume", "amount"):
        values = pd.to_numeric(frame[key], errors="coerce")
        nonempty = frame[key].notna() & frame[key].astype(str).str.strip().ne("")
        if (nonempty & values.isna()).any() or not values[nonempty].map(lambda x: math.isfinite(float(x))).all():
            raise RuntimeError(f"invalid {key} values for {code}")
        active = status.eq(1)
        if values[active].isna().any() or (values[active] < 0).any():
            raise RuntimeError(f"active rows require finite non-negative {key} for {code}")
    for key in ("turn", "pctChg"):
        values = pd.to_numeric(frame[key], errors="coerce")
        nonempty = frame[key].notna() & frame[key].astype(str).str.strip().ne("")
        if (nonempty & values.isna()).any() or not values[nonempty].map(lambda x: math.isfinite(float(x))).all():
            raise RuntimeError(f"invalid optional {key} values for {code}")
    factor = frame.adj_close.astype(float) / frame.close.astype(float)
    for raw_key, adj_key in zip(("open", "high", "low", "close"), ADJ_FIELDS):
        discrepancy = (frame[adj_key].astype(float) - frame[raw_key].astype(float) * factor).abs()
        if (discrepancy > 0.001).any():
            row = int(discrepancy.idxmax())
            raise RuntimeError(f"adjusted/raw factor mismatch for {code} {frame.loc[row,'date']} {adj_key}: "
                               f"{discrepancy.loc[row]}")


def _validate_forward_coverage(frame: pd.DataFrame, expected_future_dates: list[str],
                               verified_out_date: str | None = None,
                               last_quote_date: str | None = None) -> None:
    actual = set(frame.loc[frame.date.astype(str) >= DONOR_END, "date"].astype(str))
    if verified_out_date:
        if not last_quote_date or not ("2026-01-01" <= verified_out_date <= END):
            raise RuntimeError("invalid verified 2026 outDate")
        if abs((pd.Timestamp(verified_out_date) - pd.Timestamp(last_quote_date)).days) > 7:
            raise RuntimeError("verified outDate is more than 7 days from the last quote")
        if last_quote_date > END or last_quote_date not in actual:
            raise RuntimeError("verified last quote is absent or beyond the cutoff")
        required = {day for day in expected_future_dates if day <= last_quote_date}
        if actual != required:
            missing = sorted(required - actual)[:8]
            extra = sorted(actual - required)[:8]
            raise RuntimeError(f"coverage gap before verified outDate; missing={missing}, extra={extra}")
        return
    required = set(expected_future_dates)
    if actual != required:
        missing = sorted(required - actual)[:8]
        extra = sorted(actual - required)[:8]
        raise RuntimeError(f"forward coverage gap; missing={missing}, extra={extra}")


def _valid_combined(code: str, raw_path: Path, dividend_path: Path, expected_future_dates: list[str],
                    verified_out_date: str | None = None, last_quote_date: str | None = None) -> bool:
    try:
        if not raw_path.is_file() or not dividend_path.is_file():
            return False
        frame = pd.read_csv(raw_path, dtype={"date": str, "code": str})
        if not set(FINAL_COLUMNS).issubset(frame.columns):
            return False
        _validate_raw_adjusted(frame, code)
        _validate_forward_coverage(frame, expected_future_dates, verified_out_date, last_quote_date)
        dividends = pd.read_csv(dividend_path, dtype=str, encoding="utf-8-sig").fillna("")
        if not dividends.empty and "code" in dividends:
            if set(dividends.code.astype(str).str.lower()) - {code.lower()}:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, pd.errors.ParserError, pd.errors.EmptyDataError, RuntimeError):
        return False


def _valid_cached(code: str, folder: Path, donor_hash: str, expected_future_dates: list[str], status: str) -> bool:
    raw_path, dividend_path = folder / f"{code}.csv", folder / f"{code}.dividends.csv"
    meta_path = folder / f"{code}.cache.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cached_status = meta.get("status")
        status_ok = (cached_status == status or
                     (status == "forward_download" and cached_status == "forward_delisted"))
        return (status_ok and meta.get("donor_manifest_sha256") == donor_hash
                and meta.get("raw_sha256") == sha256(raw_path)
                and meta.get("dividends_sha256") == sha256(dividend_path)
                and _valid_combined(code, raw_path, dividend_path, expected_future_dates,
                                    meta.get("verified_forward_outDate"), meta.get("last_quote_date")))
    except (OSError, json.JSONDecodeError, FileNotFoundError):
        return False


def _cache_metadata(folder: Path, code: str, donor_hash: str, status: str, anchor_ratio=None,
                    overlap_check=None, verified_out_date=None, last_quote_date=None) -> None:
    raw_path, dividend_path = folder / f"{code}.csv", folder / f"{code}.dividends.csv"
    _atomic_json({"code": code, "status": status, "donor_manifest_sha256": donor_hash,
                  "raw_sha256": sha256(raw_path), "dividends_sha256": sha256(dividend_path),
                  "anchor_date": DONOR_END if anchor_ratio is not None else None,
                  "anchor_ratio": anchor_ratio, "overlap_check": overlap_check,
                  "verified_forward_outDate": verified_out_date, "last_quote_date": last_quote_date},
                 folder / f"{code}.cache.json")


def _query_stock_basic(code: str) -> dict:
    """Fresh serial identity-checked lookup used only to verify a truncated tail."""
    last_error = None
    need_login = getattr(bs_context, "default_socket", None) is None
    for attempt in range(1, 4):
        if need_login:
            try:
                login = source.bs.login()
                if login.error_code != "0":
                    raise RuntimeError(f"BaoStock relogin failed: {login.error_code} {login.error_msg}")
                need_login = False
            except Exception as error:
                last_error = error
                _close_socket()
                continue
        try:
            response = source.bs.query_stock_basic(code=code)
            if response is None or getattr(response, "error_code", None) != "0":
                raise RuntimeError(f"query_stock_basic failed: {getattr(response, 'error_code', '?')} "
                                   f"{getattr(response, 'error_msg', '')}")
            if getattr(response, "method", None) != "query_stock_basic":
                raise RuntimeError(f"stock-basic method mismatch: {getattr(response, 'method', None)}")
            response_code = getattr(response, "code", None)
            if response_code is None or str(response_code).lower() != code.lower():
                raise RuntimeError(f"stock-basic response code mismatch: {response_code}")
            frame = source.read_result(response)
            if getattr(response, "error_code", None) != "0":
                raise RuntimeError(f"stock-basic pagination failed: {response.error_code} {response.error_msg}")
            if "code" not in frame or frame.empty or set(frame.code.astype(str).str.lower()) != {code.lower()}:
                raise RuntimeError("stock-basic row code mismatch or empty result")
            if len(frame) != 1 or "outDate" not in frame:
                raise RuntimeError("stock-basic result is not one row or has no outDate")
            return frame.iloc[0].to_dict()
        except Exception as error:
            last_error = error
            _close_socket()
            need_login = True
    raise RuntimeError(f"query_stock_basic failed after 3 attempts for {code}: {last_error}")


def _update_one(code: str, old_folder: Path, output: Path, expected_future_dates: list[str]) -> dict:
    old_path = old_folder / f"{code}.csv"
    old_div_path = old_folder / f"{code}.dividends.csv"
    old = pd.read_csv(old_path, dtype={"date": str, "code": str}).sort_values("date").reset_index(drop=True)
    if old.empty or old.code.nunique() != 1 or str(old.code.iloc[0]).lower() != code.lower():
        raise RuntimeError(f"invalid donor history for {code}")
    if old.date.duplicated().any():
        raise RuntimeError(f"duplicate donor dates for {code}")

    raw_new = prior.safe_query(source.bs.query_history_k_data_plus, code=code, call_type="history-raw-forward",
                              fields=source.FIELDS, start_date=DONOR_END, end_date=END,
                              frequency="d", adjustflag="3")
    adj_new = prior.safe_query(source.bs.query_history_k_data_plus, code=code, call_type="history-adjusted-forward",
                              fields="date,code,open,high,low,close,adjustflag", start_date=DONOR_END,
                              end_date=END, frequency="d", adjustflag="1")
    _row_flags(raw_new, "3", code)
    _row_flags(adj_new, "1", code)
    raw_new = raw_new.copy().sort_values("date").reset_index(drop=True)
    adj_new = adj_new.copy().sort_values("date").reset_index(drop=True)
    if set(raw_new.date.astype(str)) != set(adj_new.date.astype(str)):
        missing_adj = sorted(set(raw_new.date.astype(str)) - set(adj_new.date.astype(str)))[:5]
        missing_raw = sorted(set(adj_new.date.astype(str)) - set(raw_new.date.astype(str)))[:5]
        raise RuntimeError(f"raw/adjusted query date mismatch for {code}; missing adj={missing_adj}, raw={missing_raw}")

    overlap_raw = raw_new[raw_new.date.astype(str) == DONOR_END]
    overlap_old = old[old.date.astype(str) == DONOR_END]
    if len(overlap_raw) != 1 or len(overlap_old) != 1:
        raise RuntimeError(f"required {DONOR_END} raw overlap absent for {code} (old={len(overlap_old)}, new={len(overlap_raw)})")
    overlap_check = _compare_raw_overlap(overlap_old.iloc[0], overlap_raw.iloc[0], code)

    adj_keep = adj_new[["date", "open", "high", "low", "close"]].rename(
        columns={key: f"adj_{key}" for key in ("open", "high", "low", "close")})
    new_rows = raw_new.merge(adj_keep, on="date", how="left", validate="one_to_one")
    for key in new_rows.columns.difference(("date", "code")):
        new_rows[key] = pd.to_numeric(new_rows[key], errors="coerce")
    _validate_raw_adjusted(new_rows, code)
    overlap_new = new_rows[new_rows.date.astype(str) == DONOR_END]
    ratio = float(overlap_new.adj_close.iloc[0]) / float(overlap_old.adj_close.iloc[0])
    if not math.isfinite(ratio) or ratio <= 0:
        raise RuntimeError(f"invalid adjusted-price anchor ratio for {code}: {ratio}")

    old = old.copy()
    for key in ADJ_FIELDS:
        old[key] = pd.to_numeric(old[key], errors="coerce") * ratio
    future = new_rows[new_rows.date.astype(str) > DONOR_END].copy()
    combined = pd.concat([old, future], ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)
    combined = combined[FINAL_COLUMNS]
    if combined.date.duplicated().any():
        raise RuntimeError(f"duplicate combined dates for {code}")
    _validate_raw_adjusted(combined, code)
    last_quote_date = str(raw_new.date.astype(str).max())
    verified_out_date = None
    if last_quote_date != expected_future_dates[-1]:
        basic = _query_stock_basic(code)
        verified_out_date = str(basic.get("outDate", "")).strip()
        if not (verified_out_date.startswith("2026-") and "2026-01-01" <= verified_out_date <= END):
            raise RuntimeError(f"tail ends before cutoff without verified 2026 outDate: {verified_out_date!r}")
        _validate_forward_coverage(combined, expected_future_dates, verified_out_date, last_quote_date)
    else:
        _validate_forward_coverage(combined, expected_future_dates)

    old_div = pd.read_csv(old_div_path, dtype=str, encoding="utf-8-sig").fillna("")
    new_div = prior.safe_query(source.bs.query_dividend_data, code=code, call_type="dividend-2026",
                               year="2026", yearType="operate").fillna("")
    if not new_div.empty and "code" in new_div:
        wrong = set(new_div.code.astype(str).str.lower()) - {code.lower()}
        if wrong:
            raise RuntimeError(f"2026 dividend code mismatch for {code}: {sorted(wrong)}")
    if not new_div.empty and "dividOperateDate" in new_div:
        future_ops = new_div.dividOperateDate.astype(str).str.strip()
        new_div = new_div[(future_ops == "") | (future_ops <= END)].copy()
    dividend_frames = [part for part in (old_div, new_div) if len(part.columns)]
    if dividend_frames:
        dividends = pd.concat(dividend_frames, ignore_index=True, sort=False).fillna("").drop_duplicates()
    else:
        dividends = old_div.copy()
    if not dividends.empty and "code" in dividends:
        wrong = set(dividends.code.astype(str).str.lower()) - {code.lower()}
        if wrong:
            raise RuntimeError(f"combined dividend code mismatch for {code}: {sorted(wrong)}")
    raw_path, dividend_path = output / f"{code}.csv", output / f"{code}.dividends.csv"
    _atomic_csv(combined, raw_path)
    _atomic_csv(dividends, dividend_path, encoding="utf-8-sig")
    if not _valid_combined(code, raw_path, dividend_path, expected_future_dates):
        raise RuntimeError(f"combined files failed post-write validation for {code}")
    return {"code": code, "anchor_ratio": ratio, "overlap_check": overlap_check,
            "verified_forward_outDate": verified_out_date, "last_quote_date": last_quote_date,
            "status": "forward_delisted" if verified_out_date else "forward_download"}


def _worker(output_text: str, old_text: str, codes: list[tuple[int, str]],
            expected_future_dates: list[str], donor_hash: str) -> dict:
    """Each spawned process owns one login and processes its chunk in order."""
    output, old_folder = Path(output_text), Path(old_text)
    socket.setdefaulttimeout(35)
    logged_in = False
    successes, failures = [], []
    try:
        login = source.bs.login()
        if login.error_code != "0":
            message = f"BaoStock login failed: {login.error_code} {login.error_msg}"
            return {"successes": successes,
                    "failures": [{"code": code, "error": message} for _, code in codes]}
        logged_in = True
        for ordinal, code in codes:
            try:
                result = _update_one(code, old_folder, output, expected_future_dates)
                _cache_metadata(output, code, donor_hash, result["status"],
                                result["anchor_ratio"], result["overlap_check"],
                                result["verified_forward_outDate"], result["last_quote_date"])
                successes.append(result)
                print(f"[{ordinal}/{COUNT}] updated {code}", flush=True)
            except Exception as error:
                failures.append({"code": code, "error": str(error)})
                print(f"[{ordinal}/{COUNT}] FAILED {code}: {error}", flush=True)
    except Exception as error:
        done = {item["code"] for item in successes}
        done.update(item["code"] for item in failures)
        for ordinal, code in codes:
            if code not in done:
                failures.append({"code": code, "error": f"worker setup failed: {error}"})
                print(f"[{ordinal}/{COUNT}] FAILED {code}: {error}", flush=True)
    finally:
        # Closing only this child's TCP socket avoids server-side logout affecting peers.
        if logged_in or getattr(bs_context, "default_socket", None) is not None:
            _close_socket()
    return {"successes": successes, "failures": failures}


def _load_sources(old_folder: Path, source_check_path: Path):
    manifest_path = old_folder / "manifest.json"
    if not manifest_path.is_file() or not (old_folder / "universe.csv").is_file() or not (old_folder / "calendar.csv").is_file():
        raise RuntimeError(f"Missing frozen source dataset files in {old_folder}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("start"), manifest.get("end"), manifest.get("warmup"), manifest.get("requested_count")) != (START, DONOR_END, WARMUP, COUNT):
        raise RuntimeError("Donor manifest does not match frozen 300-stock 2020-2025 dataset")
    if manifest.get("failures"):
        raise RuntimeError("Donor dataset is incomplete")
    universe = pd.read_csv(old_folder / "universe.csv", dtype=str)
    if "code" not in universe or len(universe) != COUNT or universe.code.duplicated().any():
        raise RuntimeError("Frozen universe must contain exactly 300 unique stock codes")
    codes = universe.code.astype(str).tolist()
    if any(not CODE_RE.fullmatch(code) for code in codes):
        raise RuntimeError("Frozen universe contains an out-of-scope code")
    entries = {item.get("code"): item for item in manifest.get("inventory", [])}
    if set(entries) != set(codes):
        raise RuntimeError("Donor manifest inventory does not exactly match frozen universe")
    for code in codes:
        for suffix, hash_key in ((".csv", "sha256"), (".dividends.csv", "dividends_sha256")):
            path = old_folder / f"{code}{suffix}"
            if not path.is_file() or sha256(path) != entries[code].get(hash_key):
                raise RuntimeError(f"Donor file hash mismatch: {code}{suffix}")

    source_check = json.loads(source_check_path.read_text(encoding="utf-8"))
    early_items = source_check.get("early_end", [])
    early_codes = [item.get("code") for item in early_items]
    if len(early_codes) != 10 or len(set(early_codes)) != 10 or not set(early_codes).issubset(set(codes)):
        raise RuntimeError("Expected exactly the 10 previously verified early-ended constituents")
    if any(item.get("last_quote_date", DONOR_END) >= DONOR_END for item in early_items):
        raise RuntimeError("Early-ended constituent metadata conflicts with the 2025-12-31 cutoff")
    return manifest, universe, codes, early_codes, source_check


def _verify_freeze() -> tuple[dict, str, dict]:
    """Validate the pre-registered freeze and model artifacts before any network call."""
    if not FROZEN_JSON.is_file():
        raise RuntimeError(f"Required model freeze is missing: {FROZEN_JSON}")
    freeze = json.loads(FROZEN_JSON.read_text(encoding="utf-8"))
    if freeze.get("evaluation_data_loaded") is not False:
        raise RuntimeError("Freeze must attest that forward evaluation data has not been loaded")
    if freeze.get("training_label_end_max") != DONOR_END:
        raise RuntimeError(f"Freeze label boundary must be {DONOR_END}")
    if freeze.get("evaluation_range_end") != END:
        raise RuntimeError(f"Freeze evaluation boundary must be {END}")
    try:
        frozen_at = datetime.fromisoformat(str(freeze["frozen_at_utc"]).replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Freeze has no valid frozen_at_utc timestamp") from error
    if frozen_at.tzinfo is None:
        raise RuntimeError("Freeze timestamp must include a UTC offset")
    started_at = datetime.now(timezone.utc)
    if frozen_at.astimezone(timezone.utc) > started_at:
        raise RuntimeError("Freeze timestamp is later than this download attempt")
    hashes = freeze.get("hashes", {})
    checked = {}
    for filename, expected in EXPECTED_MODEL_HASHES.items():
        recorded = hashes.get(filename)
        artifact = FROZEN_DIR / "models" / filename
        if recorded != expected:
            raise RuntimeError(f"Freeze is missing the expected hash record for {filename}")
        if not artifact.is_file() or sha256(artifact) != expected:
            raise RuntimeError(f"Frozen model artifact is missing or hash-mismatched: {filename}")
        checked[filename] = expected
    return freeze, started_at.isoformat(timespec="seconds").replace("+00:00", "Z"), checked


def _query_calendar() -> pd.DataFrame:
    socket.setdefaulttimeout(35)
    last_error = None
    try:
        need_login = True
        for attempt in range(1, 4):
            if need_login:
                try:
                    login = source.bs.login()
                    if login.error_code != "0":
                        raise RuntimeError(f"BaoStock login failed: {login.error_code} {login.error_msg}")
                except Exception as error:
                    last_error = error
                    _close_socket()
                    continue
                need_login = False
            try:
                response = source.bs.query_trade_dates(start_date=START, end_date=END)
                if response is None or getattr(response, "error_code", None) != "0":
                    raise RuntimeError(f"BaoStock calendar response failed: {getattr(response, 'error_code', '?')} "
                                       f"{getattr(response, 'error_msg', '')}")
                if getattr(response, "method", "") not in ("", "query_trade_dates"):
                    raise RuntimeError(f"Calendar response method mismatch: {response.method}")
                for key, expected in (("start_date", START), ("end_date", END)):
                    actual = getattr(response, key, None)
                    if actual not in (None, "") and str(actual) != expected:
                        raise RuntimeError(f"Calendar response {key} mismatch: expected {expected}, got {actual}")
                expected_fields = ["calendar_date", "is_trading_day"]
                if list(getattr(response, "fields", [])) != expected_fields:
                    raise RuntimeError(f"Calendar fields mismatch: {getattr(response, 'fields', [])}")
                calendar = source.read_result(response)
                if getattr(response, "error_code", None) != "0":
                    raise RuntimeError(f"Calendar pagination failed: {response.error_code} {response.error_msg}")
                try:
                    page_size = int(response.per_page_count)
                    if page_size > 0 and len(response.data) >= page_size:
                        raise RuntimeError("Calendar pagination terminal page was not verified")
                except (AttributeError, TypeError, ValueError):
                    pass
                break
            except Exception as error:
                last_error = error
                _close_socket()
                need_login = True
        else:
            raise RuntimeError(f"Calendar query failed after 3 attempts: {last_error}")
    finally:
        _close_socket()
    required = {"calendar_date", "is_trading_day"}
    if not required.issubset(calendar.columns):
        raise RuntimeError(f"Trade calendar missing fields: {sorted(required-set(calendar.columns))}")
    calendar = calendar.copy()
    calendar["calendar_date"] = calendar.calendar_date.astype(str)
    calendar["is_trading_day"] = calendar.is_trading_day.astype(str)
    if calendar.calendar_date.duplicated().any() or not calendar.calendar_date.is_monotonic_increasing:
        raise RuntimeError("BaoStock trade calendar is not unique and increasing")
    if ((calendar.calendar_date < START) | (calendar.calendar_date > END)).any():
        raise RuntimeError("Trade calendar includes dates outside the frozen period")
    if set(calendar.is_trading_day) - {"0", "1"}:
        raise RuntimeError("Trade calendar contains an invalid trading-day flag")
    selected = calendar[calendar.is_trading_day == "1"].reset_index(drop=True)
    if selected.empty or selected.calendar_date.iloc[-1] > END:
        raise RuntimeError("Trade calendar has no valid sessions through cutoff")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Build forward data for the frozen 300-stock sample")
    parser.add_argument("--workers", type=int, default=3, choices=(1, 2, 3))
    args = parser.parse_args()
    # The frozen models and explicit no-forward-data attestation are checked before
    # the first BaoStock login/query, so any evaluation fetch necessarily follows freeze.
    freeze, download_started_utc, model_hashes = _verify_freeze()
    old_folder = ROOT / "data" / f"pilot-{START}-{DONOR_END}-n{COUNT}-s{SEED}"
    output = ROOT / "data" / "forward-2026-10-13"
    source_check_path = ROOT / "results" / "expanded-300-final" / "source_verification.json"
    donor_manifest, universe, codes, early_codes, source_check = _load_sources(old_folder, source_check_path)
    donor_manifest_path = old_folder / "manifest.json"
    donor_hash = sha256(donor_manifest_path)
    old_calendar_hash = sha256(old_folder / "calendar.csv")
    old_universe_hash = sha256(old_folder / "universe.csv")

    output.mkdir(parents=True, exist_ok=True)
    universe_out = output / "universe.csv"
    if universe_out.exists() and sha256(universe_out) != old_universe_hash:
        raise RuntimeError("Existing forward universe differs from frozen donor; refusing to replace it")
    if not universe_out.exists():
        temporary = universe_out.with_name(f"{universe_out.name}.{os.getpid()}.tmp")
        try:
            shutil.copyfile(old_folder / "universe.csv", temporary)
            os.replace(temporary, universe_out)
        finally:
            if temporary.exists():
                temporary.unlink()

    selected_calendar = _query_calendar()
    donor_calendar = pd.read_csv(old_folder / "calendar.csv", dtype=str)
    donor_dates = donor_calendar.calendar_date.astype(str).tolist()
    new_history_dates = selected_calendar.loc[
        (selected_calendar.calendar_date >= DONOR_END) & (selected_calendar.calendar_date <= END),
        "calendar_date"].tolist()
    # Verify the historical calendar portion byte-for-date against the frozen source.
    queried_history = selected_calendar[selected_calendar.calendar_date <= DONOR_END].calendar_date.tolist()
    if queried_history != donor_dates:
        raise RuntimeError("BaoStock's 2020-2025 calendar differs from the frozen donor calendar")
    _atomic_csv(selected_calendar, output / "calendar.csv")
    expected_future_dates = new_history_dates
    if not expected_future_dates or expected_future_dates[0] != DONOR_END:
        raise RuntimeError("Forward calendar must contain the overlap session 2025-12-31")

    resumed, copied_delisted, todo = [], [], []
    for ordinal, code in enumerate(codes, 1):
        status = "verified_delisted_donor" if code in early_codes else "forward_download"
        code_expected_dates = [] if code in early_codes else expected_future_dates
        if _valid_cached(code, output, donor_hash, code_expected_dates, status):
            resumed.append(code)
            print(f"[{ordinal}/{COUNT}] resumed {code}", flush=True)
            continue
        if code in early_codes:
            for suffix in (".csv", ".dividends.csv"):
                src, dst = old_folder / f"{code}{suffix}", output / f"{code}{suffix}"
                temporary = dst.with_name(f"{dst.name}.{os.getpid()}.tmp")
                try:
                    shutil.copyfile(src, temporary)
                    os.replace(temporary, dst)
                finally:
                    if temporary.exists():
                        temporary.unlink()
            if not _valid_combined(code, output / f"{code}.csv", output / f"{code}.dividends.csv", []):
                raise RuntimeError(f"Verified early-ended donor files failed validation for {code}")
            _cache_metadata(output, code, donor_hash, status)
            copied_delisted.append(code)
            print(f"[{ordinal}/{COUNT}] reused verified delisted history {code}", flush=True)
        else:
            todo.append((ordinal, code))

    chunks = []
    if todo:
        size = math.ceil(len(todo) / min(args.workers, 3))
        chunks = [todo[i:i + size] for i in range(0, len(todo), size)]
    outcomes = []
    if chunks:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(args.workers, 3), mp_context=context) as pool:
            future_chunks = {pool.submit(_worker, str(output), str(old_folder), chunk,
                                         expected_future_dates, donor_hash): chunk for chunk in chunks}
            for future in as_completed(future_chunks):
                try:
                    outcomes.append(future.result())
                except Exception as error:
                    # A crashed worker still becomes an explicit per-code failure.
                    outcomes.append({"successes": [], "failures": [
                        {"code": code, "error": f"worker process failed: {error}"}
                        for _, code in future_chunks[future]]})

    worker_successes = [item for result in outcomes for item in result["successes"]]
    current_failures = [item for result in outcomes for item in result["failures"]]
    success_codes = {item["code"] for item in worker_successes}
    failure_codes = {item["code"] for item in current_failures}
    inventory = []
    for code in codes:
        status = "verified_delisted_donor" if code in early_codes else "forward_download"
        code_expected_dates = [] if code in early_codes else expected_future_dates
        if code in failure_codes or not _valid_cached(code, output, donor_hash, code_expected_dates, status):
            if code not in failure_codes:
                current_failures.append({"code": code, "error": "Missing or invalid final cache pair"})
            continue
        raw_path, dividend_path = output / f"{code}.csv", output / f"{code}.dividends.csv"
        inventory.append({"code": code, "sha256": sha256(raw_path),
                          "dividends_sha256": sha256(dividend_path)})
    current_failures.sort(key=lambda item: codes.index(item["code"]))

    # Preserve failures from earlier attempts as history while exposing only unresolved failures
    # in the run.load-compatible `failures` field.
    previous_history = []
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        try:
            previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            previous_history = list(previous_manifest.get("failure_history", []))
            for failure in previous_manifest.get("failures", []):
                previous_history.append({**failure, "attempt": "previous_manifest"})
        except (OSError, json.JSONDecodeError):
            previous_history.append({"error": "Previous manifest could not be parsed; retained only as a note"})
    failure_history = previous_history + [{**item, "attempt": "2026-forward-refresh"} for item in current_failures]
    cache_meta = {}
    for code in codes:
        try:
            cache_meta[code] = json.loads((output / f"{code}.cache.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    anchors = {code: item.get("anchor_ratio") for code, item in cache_meta.items()
               if item.get("anchor_ratio") is not None}
    forward_delisted_records = [
        {"code": code, "outDate": item.get("verified_forward_outDate"),
         "last_quote_date": item.get("last_quote_date")}
        for code, item in cache_meta.items() if item.get("status") == "forward_delisted"
    ]
    manifest = {"provider": "BaoStock", "start": START, "end": END, "warmup": WARMUP,
                "seed": SEED, "requested_count": COUNT, "universe_as_of": START,
                "universe_policy": "fixed historical main-board snapshot; seeded sample; no replacements",
                "scope": "forward evaluation of the frozen 300-stock sample; excludes the 10 verified early-ended constituents from new requests",
                "inventory": inventory, "failures": current_failures,
                "downloaded_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "forward_extension": {"donor_dataset": str(old_folder),
                                      "donor_manifest_sha256": donor_hash,
                                      "donor_calendar_sha256": old_calendar_hash,
                                      "donor_universe_sha256": old_universe_hash,
                                      "forward_cutoff_inclusive": END,
                                      "overlap_anchor_date": DONOR_END,
                                      "overlap_raw_tolerance": {"prices": PRICE_TOL, "volume": 0.5,
                                                                 "amount": 0.011, "turn_pctChg": 0.02,
                                                                 "discrete": "exact"},
                                      "adjusted_history_rebase": "multiply each donor adj_OHLC by new_adj_close/old_adj_close on 2025-12-31, then append new dates after overlap",
                                      "anchor_ratios": anchors,
                                      "early_ended_codes_reused_without_forward_query": early_codes,
                                      "forward_delisted_codes": forward_delisted_records,
                                      "early_end_source_verification_sha256": sha256(source_check_path),
                                      "current_reused_cache_codes": resumed,
                                      "donor_early_end_files_copied": copied_delisted,
                                      "forward_downloaded_codes": sorted(success_codes),
                                      "calendar_source": "BaoStock query_trade_dates via download.read_result; validated trading_day=1; historical dates matched donor",
                                      "failure_history": failure_history,
                                      "cache_metadata": cache_meta},
                "model_freeze": {"path": str(FROZEN_JSON),
                                 "sha256": sha256(FROZEN_JSON),
                                 "frozen_at_utc": freeze["frozen_at_utc"],
                                 "download_started_utc": download_started_utc,
                                 "evaluation_data_loaded_at_freeze": freeze["evaluation_data_loaded"],
                                 "training_label_end_max": freeze["training_label_end_max"],
                                 "model_hashes_verified": model_hashes},
                "parallel": {"workers_requested": args.workers, "workers_used": len(chunks),
                             "process_isolation": True, "chunking": "contiguous universe-order chunks",
                             "worker_cleanup": "close_default_socket_without_server_logout"},
                "query_validation": {"response_identity_and_fields_checked": True,
                                     "row_level_adjustflag_required": True,
                                     "raw_adjustflag": "3", "adjusted_adjustflag": "1",
                                     "history_pagination_terminal_checked": True,
                                     "reconnect_after_query_failure": True,
                                     "max_query_attempts": 3,
                                     "raw_overlap_checked": True,
                                     "adjusted_raw_daily_factors_checked": True,
                                     "future_dividend_operate_dates_filtered": True}}
    _atomic_json(manifest, manifest_path)
    print(f"DATASET {output}", flush=True)
    print(f"Complete inventory: {len(inventory)}/{COUNT}; unresolved failures: {len(current_failures)}", flush=True)
    if current_failures:
        raise SystemExit("Incomplete forward dataset recorded in manifest; do not replace or ignore failed constituents")


if __name__ == "__main__":
    main()
