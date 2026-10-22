"""Small offline tests for frozen-data integrity and conflict handling."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch

import pandas as pd

import update_paper_data as updater


class PaperDataUpdateTests(unittest.TestCase):
    def test_donor_inventory_hashes_are_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code = "sh.600006"
            frame = pd.DataFrame([{key: value for key, value in {
                "date": "2026-10-13", "code": code, "open": 10, "high": 10, "low": 10,
                "close": 10, "preclose": 10, "volume": 1, "amount": 10, "adjustflag": 3,
                "turn": 1, "tradestatus": 1, "pctChg": 0, "isST": 0,
                "adj_open": 10, "adj_high": 10, "adj_low": 10, "adj_close": 10,
            }.items()}])
            history = root / f"{code}.csv"
            dividends = root / f"{code}.dividends.csv"
            frame.to_csv(history, index=False)
            pd.DataFrame(columns=["code"]).to_csv(dividends, index=False)
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            (root / "manifest.json").write_text(json.dumps({
                "provider": "BaoStock", "end": "2026-10-13",
                "inventory": [{"code": code, "sha256": digest(history), "dividends_sha256": digest(dividends)}],
            }), encoding="utf-8")
            pd.DataFrame([{"calendar_date": "2026-10-13", "is_trading_day": 1}]).to_csv(root / "calendar.csv", index=False)
            pd.DataFrame([{"code": code, "tradeStatus": 1}]).to_csv(root / "universe.csv", index=False)

            _, codes, _ = updater._validate_donor(root, expected_count=1)
            self.assertEqual(codes, [code])
            history.write_text(history.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                updater._validate_donor(root, expected_count=1)

    def test_raw_overlap_conflicts_stop_instead_of_being_repaired(self):
        old = pd.Series({"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "preclose": 10.0,
                         "volume": 100.0, "amount": 1000.0, "turn": 1.0, "pctChg": 0.0,
                         "adjustflag": 3, "tradestatus": 1, "isST": 0})
        same = old.copy()
        same["close"] = 10.005
        updater._compare_overlap(old, same, "sh.600006")
        conflicting = same.copy()
        conflicting["close"] = 10.2
        with self.assertRaisesRegex(ValueError, "raw overlap conflict"):
            updater._compare_overlap(old, conflicting, "sh.600006")

    def test_both_missing_suspension_metrics_are_allowed_only_for_stopped_rows(self):
        old = pd.Series({"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "preclose": 10.0,
                         "volume": float("nan"), "amount": float("nan"), "turn": float("nan"),
                         "pctChg": float("nan"), "adjustflag": 3, "tradestatus": 0, "isST": 0})
        new = old.astype(object).copy()
        new["volume"] = ""
        new["amount"] = ""
        new["turn"] = ""
        new["pctChg"] = ""
        updater._compare_overlap(old, new, "sh.600825")

        active = new.copy()
        active["tradestatus"] = 1
        with self.assertRaisesRegex(ValueError, "overlap missing/value conflict.*volume"):
            updater._compare_overlap(old, active, "sh.600825")

        one_sided = new.copy()
        one_sided["volume"] = 100
        with self.assertRaisesRegex(ValueError, "overlap missing/value conflict.*volume"):
            updater._compare_overlap(old, one_sided, "sh.600825")

    def test_adjustment_flags_are_checked_per_row(self):
        correct = pd.DataFrame([{"code": "sh.600006", "date": "2026-10-14", "adjustflag": "3"}])
        updater._check_flags(correct, "3", "sh.600006")
        wrong = correct.copy()
        wrong.loc[0, "adjustflag"] = "1"
        with self.assertRaisesRegex(ValueError, "row-level adjustflag"):
            updater._check_flags(wrong, "3", "sh.600006")

    def test_offline_build_filters_weekends_and_preserves_raw_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            donor, failed_snapshot, output = root / "donor", root / "failed", root / "repaired"
            donor.mkdir()
            code = "sh.600006"
            base = {"date": "2026-10-13", "code": code, "open": 10.0, "high": 10.0, "low": 10.0,
                    "close": 10.0, "preclose": 9.9, "volume": 100.0, "amount": 1000.0,
                    "adjustflag": 3, "turn": 1.0, "tradestatus": 1, "pctChg": 1.0, "isST": 0,
                    "adj_open": 10.0, "adj_high": 10.0, "adj_low": 10.0, "adj_close": 10.0}
            donor_frame = pd.DataFrame([base])
            donor_frame.to_csv(donor / f"{code}.csv", index=False)
            donor_frame[["code"]].iloc[0:0].to_csv(donor / f"{code}.dividends.csv", index=False)
            donor_calendar = pd.DataFrame([{"calendar_date": "2026-10-13", "is_trading_day": 1}])
            donor_calendar.to_csv(donor / "calendar.csv", index=False)
            pd.DataFrame([{"code": code, "tradeStatus": 1}]).to_csv(donor / "universe.csv", index=False)
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            inventory = [{"code": code, "sha256": digest(donor / f"{code}.csv"),
                          "dividends_sha256": digest(donor / f"{code}.dividends.csv")}]
            (donor / "manifest.json").write_text(json.dumps({
                "provider": "BaoStock", "end": "2026-10-13", "inventory": inventory,
                "forward_extension": {"donor_manifest_sha256": "source-manifest-hash",
                                      "cache_metadata": {code: {"status": "forward_download"}}},
            }), encoding="utf-8")

            def quote(day, close):
                row = dict(base, date=day, open=close, high=close, low=close, close=close,
                           preclose=close - .1, amount=close * 100, adjustflag=3)
                return {key: row[key] for key in updater.RAW_FIELDS.split(",")}

            raw_rows = [quote("2026-10-13", 10.0), quote("2026-10-14", 10.1), quote("2026-10-15", 10.2)]
            def adjusted(row):
                return {"date": row["date"], "code": code, "open": row["close"], "high": row["close"],
                        "low": row["close"], "close": row["close"], "adjustflag": 1}
            end_quote = pd.DataFrame([raw_rows[-1]])
            calendar = pd.DataFrame([
                {"calendar_date": day, "is_trading_day": flag}
                for day, flag in (("2026-10-13", 1), ("2026-10-14", 1), ("2026-10-15", 1),
                                  ("2026-10-16", 0), ("2026-10-17", 0), ("2026-10-18", 0))
            ])
            exchange = pd.DataFrame([
                {"calendar_date": day, "is_trading_day": flag}
                for day, flag in (("2026-10-15", 1), ("2026-10-16", 0), ("2026-10-17", 0),
                                  ("2026-10-18", 0), ("2026-10-19", 1))
            ])

            def fetcher(*, kind, code: str, params):
                if kind == "calendar":
                    return calendar
                if kind == "exchange-calendar":
                    return exchange
                if kind == "freshness-probe":
                    return end_quote
                if kind == "raw":
                    return pd.DataFrame(raw_rows)
                if kind == "adjusted":
                    return pd.DataFrame([adjusted(row) for row in raw_rows])
                if kind == "dividends-2026":
                    return pd.DataFrame(columns=["code"])
                raise AssertionError(f"unexpected request {kind} for {code}")

            def failing_fetcher(*, kind, code: str, params):
                if kind == "dividends-2026":
                    raise RuntimeError("synthetic dividend endpoint outage")
                return fetcher(kind=kind, code=code, params=params)

            with patch.object(updater, "_now_shanghai", return_value=datetime(2026, 10, 16, 16, 0)):
                updater.build_snapshot(donor, failed_snapshot, "2026-10-15", fetcher=failing_fetcher, expected_count=1)
                failed_manifest_hash = digest(failed_snapshot / "manifest.json")
                partially_written = pd.read_csv(failed_snapshot / f"{code}.csv", dtype={"date": str})
                self.assertEqual(sum(partially_written.date == "2026-10-15"), 1)
                result = updater.repair_snapshot(failed_snapshot, output, fetcher=fetcher, expected_count=1)

            manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["complete"], manifest["failures"])
            self.assertEqual(manifest["next_trading_day"], "2026-10-19")
            self.assertEqual(manifest["parent_failed_manifest_sha256"], failed_manifest_hash)
            self.assertEqual(manifest["repair_history"][0]["failures"][0]["code"], code)
            self.assertNotIn("2026-10-17", set(pd.read_csv(result / "calendar.csv").calendar_date.astype(str)))
            self.assertEqual(manifest["exchange_calendar_sha256"], digest(result / "exchange-calendar.csv"))
            self.assertEqual(manifest["calendar_sha256"], digest(result / "calendar.csv"))
            result_frame = pd.read_csv(result / f"{code}.csv", dtype={"date": str, "code": str})
            pd.testing.assert_series_equal(result_frame.loc[0, donor_frame.columns].astype(str),
                                           donor_frame.loc[0].astype(str), check_names=False)
            self.assertEqual(set(result_frame.date), {"2026-10-13", "2026-10-14", "2026-10-15"})
            self.assertEqual(len(result_frame), 3)
            self.assertFalse(result_frame.date.duplicated().any())


if __name__ == "__main__":
    unittest.main()
