"""Refresh four fixed BaoStock samples and verify identities of early-ending histories."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
from pathlib import Path

import expand_download
from run import load


ROOT = Path(__file__).resolve().parent
START = "2020-01-02"
END = "2025-12-31"
SEED = 20201009
COUNT = 300
DATASET = ROOT / "data" / f"pilot-{START}-{END}-n{COUNT}-s{SEED}"
REFRESH_CODES = ("sh.603089", "sh.603180", "sz.000779", "sz.000800")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def refresh_codes(folder: Path, inventory: list[dict]) -> tuple[list[dict], list[dict]]:
    index_by_code = {item["code"]: index for index, item in enumerate(inventory, start=1)}
    missing = [code for code in REFRESH_CODES if code not in index_by_code]
    if missing:
        raise ValueError(f"Fixed refresh codes are absent from manifest: {missing}")
    before = {}
    for code in REFRESH_CODES:
        before[code] = {
            "raw_sha256": sha256(folder / f"{code}.csv"),
            "dividends_sha256": sha256(folder / f"{code}.dividends.csv"),
        }

    errors = []
    attempted = []
    error_by_code = {}
    # Call the existing chunk worker directly in this process, one code at a
    # time, so a failed refresh prevents the remaining network checks.
    for code in REFRESH_CODES:
        attempted.append(code)
        try:
            outcome = expand_download._download_chunk(
                str(folder), [(index_by_code[code], code)])
        except Exception as error:
            failure_record = {"code": code, "error": str(error)}
            errors.append(failure_record)
            error_by_code[code] = failure_record["error"]
            break
        code_errors = outcome.get("failures", [])
        if code_errors:
            errors.extend(code_errors)
            error_by_code.update({item.get("code"): item.get("error") for item in code_errors})
            break
    comparisons = []
    for code in REFRESH_CODES:
        raw_path = folder / f"{code}.csv"
        dividend_path = folder / f"{code}.dividends.csv"
        after = {
            "raw_sha256": sha256(raw_path) if raw_path.is_file() else None,
            "dividends_sha256": sha256(dividend_path) if dividend_path.is_file() else None,
        }
        comparisons.append({
            "code": code,
            "before": before[code],
            "after": after,
            "attempted": code in attempted,
            "raw_byte_identical": before[code]["raw_sha256"] == after["raw_sha256"],
            "dividends_byte_identical": before[code]["dividends_sha256"] == after["dividends_sha256"],
            "error": error_by_code.get(code),
        })
    return comparisons, errors


def query_stock_basic(code: str) -> dict:
    source = expand_download.source
    context = expand_download.bs_context
    need_login = getattr(context, "default_socket", None) is None
    last_error = None
    for attempt in range(1, 4):
        if need_login:
            try:
                login = source.bs.login()
                if login.error_code != "0":
                    raise RuntimeError(f"BaoStock login failed: {login.error_code}: {login.error_msg}")
                need_login = False
            except Exception as error:
                last_error = error
                expand_download._close_default_socket()
                need_login = True
                continue
        try:
            response = source.bs.query_stock_basic(code=code)
            if getattr(response, "error_code", None) != "0":
                raise RuntimeError(f"BaoStock {getattr(response, 'error_code', '?')}: "
                                   f"{getattr(response, 'error_msg', '')}")
            method = getattr(response, "method", "")
            response_code = getattr(response, "code", "")
            if method != "query_stock_basic":
                raise RuntimeError(f"response method mismatch: {method!r}")
            if str(response_code).strip().lower() != code.lower():
                raise RuntimeError(f"response code mismatch: expected {code}, got {response_code!r}")
            frame = source.read_result(response)
            if getattr(response, "error_code", None) != "0":
                raise RuntimeError(f"BaoStock pagination {response.error_code}: {response.error_msg}")
            if "code" not in frame.columns:
                raise RuntimeError("stock_basic result has no code field")
            result_codes = sorted(set(frame["code"].dropna().astype(str).str.lower()))
            if result_codes != [code.lower()]:
                raise RuntimeError(f"result code mismatch: expected {code}, got {result_codes}")
            return {
                "code": code,
                "response_method": method,
                "response_code": response_code,
                "result_codes": result_codes,
                "records": frame.to_dict("records"),
            }
        except Exception as error:
            last_error = error
            expand_download._close_default_socket()
            need_login = True
    raise RuntimeError(f"[{code}/stock_basic] failed after 3 attempts: {last_error}")


def main() -> None:
    folder = DATASET
    manifest, frames, _calendar, _dividends = load(folder)
    if (manifest.get("start"), manifest.get("end"), manifest.get("seed"),
            manifest.get("requested_count")) != (START, END, SEED, COUNT):
        raise ValueError("Dataset manifest does not match the fixed n300 experiment")
    if len(manifest.get("inventory", [])) != COUNT:
        raise ValueError("Dataset manifest does not contain exactly 300 inventory entries")
    del frames, _calendar, _dividends
    initial_backup = folder / "manifest.initial.json"
    if not initial_backup.exists():
        shutil.copyfile(folder / "manifest.json", initial_backup)

    queried_utc = datetime.now(timezone.utc).isoformat()
    report = {
        "dataset": str(folder),
        "query_utc": queried_utc,
        "script_sha256": sha256(Path(__file__)),
        "refresh": [],
        "early_end": [],
        "status": "failed",
    }
    comparison, refresh_errors = refresh_codes(folder, manifest["inventory"])
    report["refresh"] = comparison

    # Keep the live manifest aligned with any rewritten files, even if a later
    # refresh or stock-basic check fails. The initial snapshot remains untouched.
    entries = {item["code"]: item for item in manifest["inventory"]}
    for code in REFRESH_CODES:
        entries[code]["sha256"] = sha256(folder / f"{code}.csv")
        entries[code]["dividends_sha256"] = sha256(folder / f"{code}.dividends.csv")
    atomic_json(folder / "manifest.json", manifest)

    if refresh_errors:
        report["errors"] = refresh_errors
    else:
        try:
            refreshed_manifest, refreshed_frames, _calendar, _dividends = load(folder)
            if refreshed_manifest["failures"]:
                raise ValueError("Dataset has manifest failures after refresh")
            early = []
            for code, frame in refreshed_frames.items():
                last_date = str(frame.date.iloc[-1])
                if last_date < END:
                    early.append((code, last_date))
            if early:
                source = expand_download.source
                logout_needed = False
                try:
                    login = source.bs.login()
                    if login.error_code != "0":
                        raise RuntimeError(f"BaoStock login failed: {login.error_code}: {login.error_msg}")
                    logout_needed = True
                    for code, last_date in early:
                        detail = query_stock_basic(code)
                        detail["last_quote_date"] = last_date
                        report["early_end"].append(detail)
                finally:
                    if logout_needed:
                        try:
                            source.bs.logout()
                        except Exception:
                            expand_download._close_default_socket()
            report["status"] = "passed"
        except Exception as error:
            report["errors"] = [str(error)]

    verification_path = folder / "source_verification.json"
    atomic_json(verification_path, report)
    verification_hash = sha256(verification_path)
    manifest["source_verification_sha256"] = verification_hash
    atomic_json(folder / "manifest.json", manifest)

    if report["status"] == "passed":
        try:
            load(folder)
        except Exception as error:
            report["status"] = "failed"
            report.setdefault("errors", []).append(f"Final manifest/data validation failed: {error}")
            atomic_json(verification_path, report)
            manifest["source_verification_sha256"] = sha256(verification_path)
            atomic_json(folder / "manifest.json", manifest)

    if report["status"] != "passed":
        raise SystemExit("Source verification failed; see source_verification.json")
    print(f"SOURCE_VERIFICATION {verification_path}", flush=True)
    print(f"Early-ending histories checked: {len(report['early_end'])}; refresh comparisons: {len(report['refresh'])}", flush=True)


if __name__ == "__main__":
    main()
