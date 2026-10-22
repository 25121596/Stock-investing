"""Build per-stock signal caches in parallel, then merge in manifest order."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

from run import load
from signals import generate, source_hashes


ROOT = Path(__file__).resolve().parent
PARTIAL_START = "2020-01-02"
PARTIAL_END = "2025-12-31"
RAW_COLUMNS = {
    "date", "code", "high", "low", "close", "adj_close", "adj_open", "adj_high",
    "adj_low", "volume", "amount", "tradestatus", "isST",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_one(code: str, csv_path: str, part_folder: str, start: str, end: str,
                expected_csv_hash: str) -> tuple[str, int]:
    """Read one validated stock file and use the unchanged signal generator."""
    path = Path(csv_path)
    if sha256(path) != expected_csv_hash:
        raise ValueError(f"Data changed while preparing signals: {code}")
    frame = pd.read_csv(path, dtype={"date": str, "code": str})
    frame = frame.sort_values("date").reset_index(drop=True)
    result = generate(Path(part_folder), {code: frame}, start, end)
    return code, len(result)


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def atomic_write_bytes(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def stock_part_identity(code: str, start: str, end: str, quote_hash: str,
                        adapter_hash: str) -> dict:
    return {
        "code": code,
        "start": start,
        "end": end,
        "quote_sha256": quote_hash,
        "source_hashes": source_hashes(),
        "signal_adapter_sha256": adapter_hash,
    }


def partial_candidates(dataset: Path) -> tuple[list[dict], list[str]]:
    universe_path = dataset / "universe.csv"
    if not universe_path.is_file():
        raise ValueError(f"Partial mode requires universe.csv: {universe_path}")
    universe = pd.read_csv(universe_path, dtype=str)
    if "code" not in universe.columns:
        raise ValueError("universe.csv must contain a code column")
    candidates, skipped = [], []
    for code in universe.code.dropna().tolist():
        if not re.fullmatch(r"(?:sh\.60\d{4}|sz\.00\d{4})", code):
            skipped.append(f"{code}: invalid main-board code")
            continue
        quote_path = dataset / f"{code}.csv"
        dividends_path = dataset / f"{code}.dividends.csv"
        if not quote_path.is_file() or not dividends_path.is_file():
            skipped.append(f"{code}: raw or dividends file missing")
            continue
        try:
            quote_hash = sha256(quote_path)
            dividends_hash = sha256(dividends_path)
            frame = pd.read_csv(quote_path, dtype={"date": str, "code": str})
            if not RAW_COLUMNS.issubset(frame.columns):
                raise ValueError("raw columns incomplete")
            if frame.empty or frame.date.duplicated().any() or frame.code.nunique() != 1 or frame.code.iloc[0] != code:
                raise ValueError("raw dates/codes invalid")
            numeric = {}
            for column in ("high", "low", "close", "adj_close", "adj_open", "adj_high",
                           "adj_low", "volume", "amount", "tradestatus", "isST"):
                numeric[column] = pd.to_numeric(frame[column], errors="coerce")
            for column in ("high", "low", "close", "adj_close", "adj_open", "adj_high",
                           "adj_low", "tradestatus", "isST"):
                if numeric[column].isna().any():
                    raise ValueError(f"raw {column} is not numeric")
            active = numeric["tradestatus"] == 1
            if numeric["volume"].isna()[active].any():
                raise ValueError("raw volume is missing on a trading day")
            signal_rows = active & (numeric["volume"] > 0)
            if numeric["amount"].isna()[signal_rows].any():
                raise ValueError("raw amount is missing on a signal-eligible day")
            if ((frame.high < frame.low) | (frame.close <= 0) | (frame.adj_close <= 0)).any():
                raise ValueError("raw OHLC invalid")
            dividend_events = pd.read_csv(dividends_path, dtype=str).fillna("")
            if len(dividend_events.columns) == 0:
                raise ValueError("dividends structure invalid")
            if sha256(quote_path) != quote_hash or sha256(dividends_path) != dividends_hash:
                raise ValueError("file changed during validation")
            candidates.append({"code": code, "sha256": quote_hash,
                               "dividends_sha256": dividends_hash})
        except Exception as error:
            skipped.append(f"{code}: {error}")
    return candidates, skipped


def prepare(dataset: Path, workers: int = 3, partial: bool = False) -> int:
    if workers < 1 or workers > 3:
        raise ValueError("--workers must be between 1 and 3")
    dataset = dataset.resolve()
    if not dataset.is_dir():
        raise ValueError(f"Dataset directory is missing: {dataset}")
    manifest_path = dataset / "manifest.json"
    if partial:
        inventory, skipped = partial_candidates(dataset)
        if skipped:
            print(f"partial mode skipped {len(skipped)} incomplete/invalid stock files", flush=True)
            for reason in skipped:
                print(f"skip {reason}", flush=True)
        start, end = PARTIAL_START, PARTIAL_END
        original_manifest_hash = None
    else:
        if not manifest_path.is_file():
            raise ValueError(f"manifest.json is missing: {manifest_path}")
        # This is the repository's existing strict validation for hashes and CSV structure.
        manifest, frames, _calendar, _dividends = load(dataset)
        if manifest.get("failures"):
            raise ValueError("Incomplete historical sample: fix downloads before preparing signals")
        inventory = manifest["inventory"]
        if len(frames) != len(inventory):
            raise ValueError("Manifest contains duplicate stock codes")
        del frames, _calendar, _dividends
        start, end = manifest["start"], manifest["end"]
        original_manifest_hash = sha256(manifest_path)

    adapter_hash = sha256(ROOT / "signals.py")
    tasks = []
    for item in inventory:
        code = item["code"]
        csv_path = dataset / f"{code}.csv"
        part_folder = dataset / ".signal-parts" / code
        part_folder.mkdir(parents=True, exist_ok=True)
        part_manifest = stock_part_identity(code, start, end, item["sha256"], adapter_hash)
        atomic_write(part_folder / "manifest.json",
                     json.dumps(part_manifest, ensure_ascii=False, indent=2))
        tasks.append((code, str(csv_path), str(part_folder), start, end, item["sha256"]))

    completed = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(prepare_one, *task): task[0] for task in tasks}
        for future in as_completed(futures):
            code, count = future.result()
            completed[code] = count
            print(f"prepared [{len(completed)}/{len(tasks)}] {code}: {count} signals", flush=True)

    total_signals = sum(completed.values())
    if partial:
        print(f"Partial signal cache ready for {len(completed)} stocks; {total_signals} signal rows.", flush=True)
        return total_signals

    # Preserve the exact stock iteration order used by signals.generate.
    merged = bytearray()
    for item in inventory:
        code = item["code"]
        path = dataset / ".signal-parts" / code / "signals.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Signal part missing for {code}: {path}")
        rows = path.read_bytes().splitlines(keepends=True)
        if not rows:
            raise ValueError(f"Signal part is empty, including its header: {code}")
        if not merged:
            merged.extend(b"".join(rows))
        else:
            merged.extend(b"".join(rows[1:]))

    identity = {
        "source": source_hashes(),
        "adapter_version": 2,
        "manifest": original_manifest_hash,
    }
    atomic_write_bytes(dataset / "signals.csv", bytes(merged))
    atomic_write(dataset / "signals.meta.json", json.dumps(identity, indent=2))
    return sum(completed.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--partial", action="store_true",
                        help="prepare caches for complete existing universe members without merging top-level signals")
    args = parser.parse_args()
    result = prepare(args.dataset, args.workers, args.partial)
    action = "Prepared partial caches for" if args.partial else "Merged"
    print(f"{action} {result} signal rows.", flush=True)


if __name__ == "__main__":
    main()
