"""Resume the fixed 300-stock BaoStock sample with isolated worker processes."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import socket

import baostock.common.context as bs_context
import pandas as pd

import download as source


ROOT = Path(__file__).resolve().parent
START = "2020-01-02"
END = "2025-12-31"
WARMUP = "2018-10-01"
SEED = 20201009
COUNT = 300
CODE_RE = re.compile(r"^(sh\.60\d{4}|sz\.00\d{4})$")
RAW_COLUMNS = set(source.FIELDS.split(",")) | {
    "adj_open", "adj_high", "adj_low", "adj_close"
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_pair(code: str, raw_path: Path, dividend_path: Path) -> bool:
    """Accept only a complete, readable raw/adjusted bar file and action file."""
    if not raw_path.is_file() or not dividend_path.is_file():
        return False
    try:
        frame = pd.read_csv(raw_path, dtype={"date": str, "code": str})
        if frame.empty or not RAW_COLUMNS.issubset(frame.columns):
            return False
        if frame.code.nunique() != 1 or frame.code.iloc[0] != code or frame.date.duplicated().any():
            return False
        numeric = frame[["open", "high", "low", "close", "adj_open", "adj_high", "adj_low", "adj_close"]]
        numeric = numeric.apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any() or (numeric <= 0).any().any():
            return False
        pd.read_csv(dividend_path, dtype=str, encoding="utf-8-sig")
        return True
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return False


def _atomic_csv(frame: pd.DataFrame, path: Path, encoding: str | None = None) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False, encoding=encoding)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _close_default_socket() -> None:
    sock = getattr(bs_context, "default_socket", None)
    if sock is not None:
        try:
            sock.close()
        except Exception:
            pass
        finally:
            bs_context.default_socket = None


def _check_response(response, call_type: str, code: str, kwargs: dict) -> None:
    if response is None:
        raise RuntimeError("BaoStock returned no response object")
    if getattr(response, "error_code", None) != "0":
        raise RuntimeError(f"BaoStock {getattr(response, 'error_code', '?')}: "
                           f"{getattr(response, 'error_msg', '')}")

    expected_method = "query_dividend_data" if call_type.startswith("dividend") else "query_history_k_data_plus"
    actual_method = getattr(response, "method", "")
    if actual_method and actual_method != expected_method:
        raise RuntimeError(f"response method mismatch: expected {expected_method}, got {actual_method}")

    # History responses echo all request metadata. Verify each echoed value so
    # a misframed response cannot be accepted as another stock's/query's data.
    if expected_method == "query_history_k_data_plus":
        expected = {"code": code, "adjustflag": kwargs.get("adjustflag"),
                    "frequency": kwargs.get("frequency"), "start_date": kwargs.get("start_date"),
                    "end_date": kwargs.get("end_date")}
        for key, value in expected.items():
            actual = getattr(response, key, None)
            # The live public server returns an empty header adjustflag even
            # when row-level adjustflag is present. Verify it on every row.
            if key == "adjustflag" and actual in (None, ""):
                continue
            if value is not None and hasattr(response, key):
                if actual is None or str(actual).strip().lower() != str(value).strip().lower():
                    raise RuntimeError(f"response {key} mismatch: expected {value}, got {actual}")
        requested_fields = kwargs.get("fields")
        if requested_fields and hasattr(response, "fields"):
            expected_fields = [field.strip() for field in requested_fields.split(",")]
            actual_fields = list(getattr(response, "fields", []))
            if actual_fields != expected_fields:
                raise RuntimeError(f"response fields mismatch: expected {expected_fields}, got {actual_fields}")
    else:
        for key, value in (("code", code), ("year", kwargs.get("year")),
                           ("yearType", kwargs.get("yearType"))):
            actual = getattr(response, key, None)
            if value is not None and hasattr(response, key):
                if actual is None or str(actual).strip().lower() != str(value).strip().lower():
                    raise RuntimeError(f"response {key} mismatch: expected {value}, got {actual}")


def safe_query(call, *, code: str, call_type: str, **kwargs) -> pd.DataFrame:
    """Run one query with response identity checks and reconnect after any failure."""
    need_login = getattr(bs_context, "default_socket", None) is None
    last_error = None
    for attempt in range(1, 4):
        if need_login:
            try:
                login = source.bs.login()
                if login.error_code != "0":
                    raise RuntimeError(f"BaoStock relogin {login.error_code}: {login.error_msg}")
                need_login = False
            except Exception as error:
                last_error = error
                print(f"[{code}/{call_type}] attempt {attempt}/3 relogin failed: {error}", flush=True)
                _close_default_socket()
                need_login = True
                continue
        try:
            response = call(code=code, **kwargs)
            _check_response(response, call_type, code, kwargs)
            frame = source.read_result(response)
            if "code" in frame and not frame.empty:
                if set(frame.code.astype(str).str.lower()) != {code.lower()}:
                    raise RuntimeError("row stock code differs from requested code")
            if kwargs.get("adjustflag") is not None:
                if "adjustflag" not in frame:
                    raise RuntimeError("Require row-level adjustflag to verify price adjustment")
                if not frame.empty and set(frame.adjustflag.astype(str)) != {str(kwargs["adjustflag"])}:
                    raise RuntimeError("row adjustflag differs from requested adjustment")
            if getattr(response, "error_code", None) != "0":
                raise RuntimeError(f"BaoStock pagination {response.error_code}: {response.error_msg}")
            # ResultData.next() returns False on a socket receive/decode failure
            # without changing error_code. A full final page means pagination
            # did not reach a verified terminal page, so reject and refetch.
            try:
                page_size = int(response.per_page_count)
                final_page_rows = len(response.data)
                if page_size > 0 and final_page_rows >= page_size:
                    raise RuntimeError(f"unverified pagination end: final page has {final_page_rows} rows")
            except (AttributeError, TypeError, ValueError):
                pass
            return frame
        except Exception as error:
            last_error = error
            print(f"[{code}/{call_type}] attempt {attempt}/3 failed: {error}", flush=True)
            _close_default_socket()
            need_login = True
    raise RuntimeError(f"[{code}/{call_type}] failed after 3 attempts: {last_error}")


def _load_donor(donor: Path):
    manifest_path = donor / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected = {"start": START, "end": END, "warmup": WARMUP}
    if any(manifest.get(key) != value for key, value in expected.items()):
        return None
    entries = {}
    for item in manifest.get("inventory", []):
        code = item.get("code")
        if isinstance(code, str) and CODE_RE.fullmatch(code):
            entries[code] = item
    return manifest, entries


def _copy_verified_donor(code: str, donor: Path, item: dict) -> bool:
    source_raw = donor / f"{code}.csv"
    source_div = donor / f"{code}.dividends.csv"
    if not source_raw.is_file() or not source_div.is_file():
        return False
    if sha256(source_raw) != item.get("sha256") or sha256(source_div) != item.get("dividends_sha256"):
        return False
    if not _valid_pair(code, source_raw, source_div):
        return False
    target_raw = ROOT / "data" / f"pilot-{START}-{END}-n{COUNT}-s{SEED}" / source_raw.name
    target_div = ROOT / "data" / f"pilot-{START}-{END}-n{COUNT}-s{SEED}" / source_div.name
    for source_path, target_path in ((source_raw, target_raw), (source_div, target_div)):
        temporary = target_path.with_name(f"{target_path.name}.{os.getpid()}.tmp")
        try:
            shutil.copyfile(source_path, temporary)
            os.replace(temporary, target_path)
        finally:
            if temporary.exists():
                temporary.unlink()
    return _valid_pair(code, target_raw, target_div)


def _download_chunk(folder_text: str, codes: list[tuple[int, str]]) -> dict:
    """One process owns one BaoStock session and handles its codes in order."""
    folder = Path(folder_text)
    socket.setdefaulttimeout(35)
    inventory, failures = {}, []
    logged_in = False
    try:
        login = source.bs.login()
        if login.error_code != "0":
            message = f"BaoStock login failed: {login.error_msg}"
            for ordinal, code in codes:
                failures.append({"code": code, "error": message})
                print(f"[{ordinal}/300] FAILED {code}: {message}", flush=True)
            return {"inventory": inventory, "failures": failures}
        logged_in = True
        for ordinal, code in codes:
            raw_path = folder / f"{code}.csv"
            dividend_path = folder / f"{code}.dividends.csv"
            try:
                raw = safe_query(source.bs.query_history_k_data_plus, code=code, call_type="history-raw",
                                 fields=source.FIELDS, start_date=WARMUP, end_date=END,
                                 frequency="d", adjustflag="3")
                adj = safe_query(source.bs.query_history_k_data_plus, code=code, call_type="history-adjusted",
                                 fields="date,code,open,high,low,close,adjustflag", start_date=WARMUP,
                                 end_date=END, frequency="d", adjustflag="1")
                if raw.empty or adj.empty:
                    raise RuntimeError("Empty history; not silently replacing historical constituent")
                adj = adj.drop(columns=["code", "adjustflag"])
                adj = adj.rename(columns={key: f"adj_{key}" for key in ("open", "high", "low", "close")})
                data = raw.merge(adj, on="date", how="left", validate="one_to_one").sort_values("date")
                for key in data.columns.difference(["date", "code"]):
                    data[key] = pd.to_numeric(data[key], errors="coerce")
                if data[["open", "close", "adj_close"]].isna().any().any():
                    raise RuntimeError("Incomplete raw/adjusted OHLC data")
                if (data.close <= 0).any() or (data.adj_close <= 0).any():
                    raise RuntimeError("Non-positive prices")

                parts = [safe_query(source.bs.query_dividend_data, code=code,
                                    call_type=f"dividend-{year}", year=str(year), yearType="operate")
                         for year in range(int(START[:4]), int(END[:4]) + 1)]
                dividends = pd.concat(parts, ignore_index=True).drop_duplicates()
                if "code" in dividends and not dividends.empty:
                    event_codes = set(dividends.code.dropna().astype(str).str.lower()) - {code.lower()}
                    if event_codes:
                        raise RuntimeError(f"dividend row code mismatch: {sorted(event_codes)}")
                _atomic_csv(data, raw_path)
                _atomic_csv(dividends, dividend_path, encoding="utf-8-sig")
                if not _valid_pair(code, raw_path, dividend_path):
                    raise RuntimeError("Downloaded files failed post-write validation")
                inventory[code] = {"code": code, "sha256": sha256(raw_path),
                                   "dividends_sha256": sha256(dividend_path)}
                print(f"[{ordinal}/300] downloaded {code}", flush=True)
            except Exception as error:
                failures.append({"code": code, "error": str(error)})
                print(f"[{ordinal}/300] FAILED {code}: {error}", flush=True)
    except Exception as error:
        # Login/network setup errors are recorded for every code assigned here.
        for ordinal, code in codes:
            if code not in inventory and all(item["code"] != code for item in failures):
                failures.append({"code": code, "error": str(error)})
                print(f"[{ordinal}/300] FAILED {code}: {error}", flush=True)
    finally:
        # BaoStock's anonymous login may share a server-side identity across
        # processes. Closing this process's TCP socket avoids a logout message
        # invalidating a peer worker's session.
        if logged_in or getattr(bs_context, "default_socket", None) is not None:
            _close_default_socket()
    return {"inventory": inventory, "failures": failures}


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume the fixed 300-stock historical sample in isolated processes")
    parser.add_argument("--workers", type=int, default=3, choices=(1, 2, 3))
    args = parser.parse_args()

    folder = ROOT / "data" / f"pilot-{START}-{END}-n{COUNT}-s{SEED}"
    universe_path, calendar_path = folder / "universe.csv", folder / "calendar.csv"
    if not universe_path.is_file() or not calendar_path.is_file():
        parser.error(f"Expected existing universe.csv and calendar.csv in {folder}")
    universe = pd.read_csv(universe_path, dtype=str)
    if "code" not in universe or len(universe) != COUNT or universe.code.duplicated().any():
        parser.error("universe.csv must contain exactly 300 unique codes")
    if not all(isinstance(code, str) and CODE_RE.fullmatch(code) for code in universe.code):
        parser.error("universe.csv contains a code outside the fixed main-board sample")
    calendar = pd.read_csv(calendar_path, dtype=str)
    if "calendar_date" not in calendar or calendar.empty:
        parser.error("calendar.csv has no calendar_date rows")

    donor = ROOT / "data" / f"pilot-{START}-{END}-n24-s{SEED}"
    donor_data = _load_donor(donor)
    donor_manifest_hash = sha256(donor / "manifest.json") if donor_data else None
    donor_entries = donor_data[1] if donor_data else {}
    current_reused, donor_copied, todo = [], [], []
    for ordinal, code in enumerate(universe.code.tolist(), 1):
        raw_path, dividend_path = folder / f"{code}.csv", folder / f"{code}.dividends.csv"
        if _valid_pair(code, raw_path, dividend_path):
            current_reused.append(code)
            print(f"[{ordinal}/300] reused {code}", flush=True)
        elif code in donor_entries and _copy_verified_donor(code, donor, donor_entries[code]):
            donor_copied.append(code)
            print(f"[{ordinal}/300] copied verified n24 cache {code}", flush=True)
        else:
            todo.append((ordinal, code))

    chunks = [todo[index::args.workers] for index in range(args.workers)]
    chunks = [chunk for chunk in chunks if chunk]
    outcomes = []
    if chunks:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(args.workers, 3), mp_context=context) as pool:
            futures = [pool.submit(_download_chunk, str(folder), chunk) for chunk in chunks]
            for future in as_completed(futures):
                outcomes.append(future.result())

    downloaded = {}
    failures = []
    for outcome in outcomes:
        downloaded.update(outcome["inventory"])
        failures.extend(outcome["failures"])

    inventory = []
    for code in universe.code.tolist():
        raw_path, dividend_path = folder / f"{code}.csv", folder / f"{code}.dividends.csv"
        if _valid_pair(code, raw_path, dividend_path):
            inventory.append({"code": code, "sha256": sha256(raw_path),
                              "dividends_sha256": sha256(dividend_path)})
        elif not any(item["code"] == code for item in failures):
            failures.append({"code": code, "error": "Missing or invalid final cache pair"})
    failures.sort(key=lambda item: universe.code.tolist().index(item["code"]))
    manifest = {"provider": "BaoStock", "start": START, "end": END, "warmup": WARMUP,
                "seed": SEED, "requested_count": COUNT, "universe_as_of": START,
                "universe_policy": "fixed historical main-board snapshot; seeded sample; no replacements",
                "scope": "pilot sample, excludes later IPOs; not full-market proof",
                "inventory": inventory, "failures": failures,
                "downloaded_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "parallel": {"workers_requested": args.workers, "workers_used": len(chunks),
                             "process_isolation": True, "chunking": "universe-order round robin",
                             "worker_cleanup": "close_default_socket_without_server_logout"},
                "cache": {"current_reused": len(current_reused), "verified_n24_copied": len(donor_copied),
                          "downloaded": len(downloaded), "donor_dataset": str(donor) if donor_data else None,
                          "donor_manifest_sha256": donor_manifest_hash,
                          "current_reused_codes": current_reused,
                          "verified_n24_copied_codes": donor_copied,
                          "downloaded_codes": sorted(downloaded)},
                "query_validation": {"response_identity_and_fields_checked": True,
                                     "history_pagination_terminal_checked": True,
                                     "reconnect_after_query_failure": True,
                                     "max_query_attempts": 3}}
    _atomic_json(manifest, folder / "manifest.json")
    print(f"DATASET {folder}", flush=True)
    print(f"Complete inventory: {len(inventory)}/{COUNT}; failures: {len(failures)}", flush=True)
    if failures:
        raise SystemExit("Incomplete downloads recorded in manifest; simulator refuses incomplete sample")


if __name__ == "__main__":
    main()
