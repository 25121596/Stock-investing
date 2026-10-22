"""E2E Test Suite: Walk-Forward Temporal Isolation & Anti-Leakage Guard (Tiers 1-4).

Requirements Enforced:
1. R1: Strict Walk-Forward temporal isolation across 2022 to 2026:
   - train_df.date.max() < eval_df.date.min()
   - train_df.label_end.max() < eval_df.date.min()
2. Label Embargo:
   - Holding periods crossing the boundary (e.g., entered in Dec with label_end in Jan)
     must be strictly barred from training masks.
3. Zero In-Sample Training Replay:
   - Any evaluation attempt overlapping with training calendar is detected and blocked.
4. Feature Column Cleanliness & Interface Invariants:
   - Forbidden future labels ('net_return', 'y_win', 'label_end', 'label_reason') must
     never appear in feature sets or inference candidate matrices.
   - Gatekeeper features must be decoupled from path-dependent portfolio states (cash, drawdown).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_model_v08 import FEATURES_V08
from v10_rank_objectives import fit_rows


class LeakageGuardError(AssertionError):
    """Raised when temporal isolation, label embargo, or data leakage rules are violated."""
    pass


class LeakageGuard:
    """Authoritative audit engine for Walk-Forward temporal isolation and anti-leakage."""

    FORBIDDEN_LABEL_COLUMNS = {"net_return", "y_win", "label_end", "label_reason"}
    FORBIDDEN_PORTFOLIO_STATE_COLUMNS = {"cash", "portfolio_cash", "drawdown", "max_drawdown",
                                         "portfolio_drawdown", "nav", "equity", "positions_count"}

    @classmethod
    def assert_temporal_isolation(cls, train_df: pd.DataFrame, eval_df: pd.DataFrame,
                                  eval_start: str | None = None) -> dict[str, str]:
        """Assert strict Walk-Forward date and label_end inequalities."""
        if train_df.empty:
            raise LeakageGuardError("Training dataframe is empty; cannot establish temporal isolation")
        if eval_df.empty:
            raise LeakageGuardError("Evaluation dataframe is empty; cannot establish temporal isolation")

        if "date" not in train_df.columns or "date" not in eval_df.columns:
            raise LeakageGuardError("Both train_df and eval_df must contain 'date' column")

        train_date_max = str(train_df["date"].max())
        eval_date_min = str(eval_df["date"].min())
        effective_eval_start = eval_start or eval_date_min

        # 1. Date boundary enforcement
        if not (train_date_max < eval_date_min):
            raise LeakageGuardError(
                f"Temporal leak detected: train_df.date.max() ({train_date_max}) "
                f"is not strictly earlier than eval_df.date.min() ({eval_date_min})"
            )

        if eval_start and not (train_date_max < eval_start):
            raise LeakageGuardError(
                f"Temporal leak detected: train_df.date.max() ({train_date_max}) "
                f"reaches into evaluation start boundary ({eval_start})"
            )

        # 2. Label Embargo enforcement (holding period must complete before evaluation start)
        if "label_end" not in train_df.columns:
            raise LeakageGuardError("train_df missing 'label_end' column; cannot verify label embargo")

        valid_label_ends = train_df["label_end"].dropna()
        if valid_label_ends.empty:
            raise LeakageGuardError("train_df contains no valid non-null label_end values")

        train_label_end_max = str(valid_label_ends.max())
        if not (train_label_end_max < eval_date_min):
            raise LeakageGuardError(
                f"Label embargo breach: train_df.label_end.max() ({train_label_end_max}) "
                f"is not strictly earlier than eval_df.date.min() ({eval_date_min}). "
                f"Holding periods cross the evaluation boundary!"
            )

        if eval_start and not (train_label_end_max < eval_start):
            raise LeakageGuardError(
                f"Label embargo breach: train_df.label_end.max() ({train_label_end_max}) "
                f"reaches into evaluation start boundary ({eval_start})"
            )

        # 3. Disjoint calendar dates check (Zero in-sample training replay)
        train_dates = set(train_df["date"].unique())
        eval_dates = set(eval_df["date"].unique())
        overlap = train_dates.intersection(eval_dates)
        if overlap:
            sample_overlap = sorted(list(overlap))[:5]
            raise LeakageGuardError(
                f"Zero in-sample training replay violation: evaluation dataset contains {len(overlap)} "
                f"dates overlapping with training data. Overlap samples: {sample_overlap}"
            )

        return {
            "status": "PASSED",
            "train_date_max": train_date_max,
            "train_label_end_max": train_label_end_max,
            "eval_date_min": eval_date_min,
            "effective_eval_start": effective_eval_start,
        }

    @classmethod
    def assert_feature_columns_clean(cls, feature_columns: list[str]) -> None:
        """Assert that no label columns sneak into feature inputs."""
        intersection = cls.FORBIDDEN_LABEL_COLUMNS.intersection(feature_columns)
        if intersection:
            raise LeakageGuardError(
                f"Future label leakage in features: columns {intersection} are forward labels "
                f"and must never be used as predictive features."
            )

    @classmethod
    def assert_gatekeeper_features_decoupled(cls, gatekeeper_feature_columns: list[str]) -> None:
        """Assert that Meta-Gatekeeper features do not depend on portfolio state."""
        intersection = cls.FORBIDDEN_PORTFOLIO_STATE_COLUMNS.intersection(
            {col.lower() for col in gatekeeper_feature_columns}
        )
        if intersection:
            raise LeakageGuardError(
                f"Meta-Gatekeeper interface violation: features {intersection} are path-dependent "
                f"portfolio metrics. Gatekeeper must only use market/candidate signals."
            )


class TestE2ELeakageGuard(unittest.TestCase):
    """Tier 1-4 comprehensive test suite for Leakage Guard & Temporal Isolation."""

    # -------------------------------------------------------------------------
    # Tier 1: Core Feature Coverage (Happy Path)
    # -------------------------------------------------------------------------

    def test_tier1_historical_rebuild_metadata_isolation(self):
        """Tier 1: Verify all rebuild training metadata JSONs (2022-2026) satisfy strict isolation."""
        json_pattern = str(ROOT / "results/v08-codex-rebuild-20261020-v1/*/*-training.json")
        training_files = glob.glob(json_pattern)
        self.assertGreater(len(training_files), 0, "No historical training metadata JSONs found")

        for file_path in training_files:
            meta = json.loads(Path(file_path).read_text(encoding="utf-8"))
            year = int(meta["year"])
            boundary = f"{year}-01-01"

            last_signal = meta.get("training_last_signal")
            last_label_end = meta.get("training_last_label_end")

            self.assertIsNotNone(last_signal, f"{file_path} missing training_last_signal")
            self.assertIsNotNone(last_label_end, f"{file_path} missing training_last_label_end")

            # Must be strictly earlier than Jan 1 of that year
            self.assertLess(
                last_signal, boundary,
                f"{file_path}: training_last_signal ({last_signal}) >= boundary ({boundary})"
            )
            self.assertLess(
                last_label_end, boundary,
                f"{file_path}: training_last_label_end ({last_label_end}) >= boundary ({boundary})"
            )

    def test_tier1_leakage_guard_validator_happy_path(self):
        """Tier 1: Happy path validation on synthetic walk-forward split."""
        train_df = pd.DataFrame({
            "date": ["2021-12-20", "2021-12-24", "2021-12-28"],
            "code": ["sh.600000", "sh.600001", "sh.600002"],
            "label_end": ["2021-12-25", "2021-12-29", "2021-12-31"],
        })
        eval_df = pd.DataFrame({
            "date": ["2022-01-04", "2022-01-05"],
            "code": ["sh.600000", "sh.600001"],
            "score": [0.8, 0.9],
        })

        result = LeakageGuard.assert_temporal_isolation(train_df, eval_df, eval_start="2022-01-01")
        self.assertEqual(result["status"], "PASSED")
        self.assertEqual(result["train_date_max"], "2021-12-28")
        self.assertEqual(result["train_label_end_max"], "2021-12-31")
        self.assertEqual(result["eval_date_min"], "2022-01-04")

    def test_tier1_feature_columns_cleanliness(self):
        """Tier 1: Verify FEATURES_V08 definition contains zero target labels."""
        LeakageGuard.assert_feature_columns_clean(FEATURES_V08)
        for forbidden in ["net_return", "y_win", "label_end", "label_reason"]:
            self.assertNotIn(forbidden, FEATURES_V08)

    # -------------------------------------------------------------------------
    # Tier 2: Boundary Value Analysis & Corner Cases
    # -------------------------------------------------------------------------

    def test_tier2_cross_year_holding_boundary_embargo(self):
        """Tier 2: Trade entered in Dec with holding period into Jan MUST be blocked."""
        # Entry date is in 2021, but holding period ends on 2022-01-05
        train_df_leaking = pd.DataFrame({
            "date": ["2021-12-20", "2021-12-28"],
            "code": ["sh.600000", "sh.600001"],
            "label_end": ["2021-12-24", "2022-01-05"],  # Crosses boundary!
        })
        eval_df = pd.DataFrame({
            "date": ["2022-01-04", "2022-01-05"],
            "code": ["sh.600000", "sh.600001"],
        })

        with self.assertRaisesRegex(LeakageGuardError, "Label embargo breach"):
            LeakageGuard.assert_temporal_isolation(train_df_leaking, eval_df, eval_start="2022-01-01")

    def test_tier2_nan_and_unclosed_position_handling(self):
        """Tier 2: Trades with NaN or null label_end must not bypass embargo."""
        # If all label_ends are NaN, guard must reject
        train_all_nan = pd.DataFrame({
            "date": ["2021-12-20", "2021-12-28"],
            "code": ["sh.600000", "sh.600001"],
            "label_end": [None, np.nan],
        })
        eval_df = pd.DataFrame({"date": ["2022-01-04"], "code": ["sh.600000"]})

        with self.assertRaisesRegex(LeakageGuardError, "contains no valid non-null label_end"):
            LeakageGuard.assert_temporal_isolation(train_all_nan, eval_df)

    def test_tier2_fit_rows_strictly_filters_cross_year_and_nan(self):
        """Tier 2: Test fit_rows filtering on synthetic dataset with edge cases."""
        synthetic_panel = pd.DataFrame([
            # Valid 2021 trade (enters 2021-12-10, exits 2021-12-20)
            {"date": "2021-12-10", "code": "A", "label_end": "2021-12-20", "net_return": 0.02, **{f: 1.0 for f in FEATURES_V08}},
            # Boundary trade crossing into 2022 (enters 2021-12-28, exits 2022-01-05)
            {"date": "2021-12-28", "code": "B", "label_end": "2022-01-05", "net_return": 0.05, **{f: 1.0 for f in FEATURES_V08}},
            # Unclosed trade (label_end is NaN)
            {"date": "2021-12-29", "code": "C", "label_end": np.nan, "net_return": np.nan, **{f: 1.0 for f in FEATURES_V08}},
            # 2022 trade
            {"date": "2022-01-04", "code": "D", "label_end": "2022-01-14", "net_return": -0.01, **{f: 1.0 for f in FEATURES_V08}},
        ])

        # Filter for year 2022 training fold
        train_2022 = fit_rows(synthetic_panel, FEATURES_V08, 2022)

        # Assert only stock A is included; B (cross-year), C (unclosed), and D (2022) must be excluded!
        self.assertEqual(len(train_2022), 1)
        self.assertEqual(train_2022.iloc[0]["code"], "A")
        self.assertLess(train_2022["date"].max(), "2022-01-01")
        self.assertLess(train_2022["label_end"].max(), "2022-01-01")

    def test_tier2_empty_dataframes_raise_error(self):
        """Tier 2: Empty dataframes must trigger explicit errors."""
        empty_df = pd.DataFrame(columns=["date", "label_end"])
        valid_df = pd.DataFrame({"date": ["2022-01-04"], "label_end": ["2022-01-10"]})

        with self.assertRaisesRegex(LeakageGuardError, "Training dataframe is empty"):
            LeakageGuard.assert_temporal_isolation(empty_df, valid_df)

        with self.assertRaisesRegex(LeakageGuardError, "Evaluation dataframe is empty"):
            LeakageGuard.assert_temporal_isolation(valid_df, empty_df)

    def test_tier2_leap_year_boundary_ordering(self):
        """Tier 2: Ensure leap year dates (2020-02-29, 2024-02-29) respect chronological order."""
        train_df = pd.DataFrame({
            "date": ["2024-02-28", "2024-02-29"],
            "code": ["sh.600000", "sh.600000"],
            "label_end": ["2024-02-29", "2024-03-01"],
        })
        eval_df = pd.DataFrame({
            "date": ["2024-03-04"],
            "code": ["sh.600000"],
        })
        result = LeakageGuard.assert_temporal_isolation(train_df, eval_df, eval_start="2024-03-04")
        self.assertEqual(result["status"], "PASSED")

    # -------------------------------------------------------------------------
    # Tier 3: Cross-Feature Interactions & Adversarial Verification
    # -------------------------------------------------------------------------

    def test_tier3_adversarial_in_sample_training_replay_blocked(self):
        """Tier 3 Adversarial: Block any attempt to evaluate on training period."""
        # Simulated cheating attempt: Model trained on 2023 is evaluated on 2023 dates
        train_df = pd.DataFrame({
            "date": ["2022-01-05", "2022-06-15", "2022-12-20"],
            "code": ["sh.600000", "sh.600001", "sh.600002"],
            "label_end": ["2022-01-15", "2022-06-25", "2022-12-30"],
        })
        eval_df_cheating = pd.DataFrame({
            "date": ["2022-06-15", "2022-06-16"],  # Replaying training dates!
            "code": ["sh.600000", "sh.600001"],
        })

        with self.assertRaisesRegex(LeakageGuardError, "Zero in-sample training replay violation|Temporal leak"):
            LeakageGuard.assert_temporal_isolation(train_df, eval_df_cheating)

    def test_tier3_adversarial_future_label_in_features_detected(self):
        """Tier 3 Adversarial: Contaminating features with future labels must fail."""
        contaminated_features = [*FEATURES_V08[:5], "net_return", *FEATURES_V08[5:10]]
        with self.assertRaisesRegex(LeakageGuardError, "Future label leakage in features"):
            LeakageGuard.assert_feature_columns_clean(contaminated_features)

    def test_tier3_meta_gatekeeper_feature_decoupling_enforced(self):
        """Tier 3: Ensure Meta-Gatekeeper features do not depend on portfolio state."""
        # Valid features: market breadth, volatility, candidate score
        valid_gatekeeper_features = ["pred_net_return_v2", "breadth5", "yz_vol20", "market_dispersion"]
        LeakageGuard.assert_gatekeeper_features_decoupled(valid_gatekeeper_features)

        # Contaminated features: adding portfolio cash or portfolio drawdown
        contaminated_gatekeeper_features = ["pred_net_return_v2", "breadth5", "cash", "drawdown"]
        with self.assertRaisesRegex(LeakageGuardError, "Meta-Gatekeeper interface violation"):
            LeakageGuard.assert_gatekeeper_features_decoupled(contaminated_gatekeeper_features)

    # -------------------------------------------------------------------------
    # Tier 4: Scenario / Real-World Panel Walk-Forward Audit
    # -------------------------------------------------------------------------

    def test_tier4_real_panel_walk_forward_audit_2022_to_2026(self):
        """Tier 4: Audit actual panel data across all 5 walk-forward years (2022-2026)."""
        panel_path = ROOT / "results/midterm-access-v3/panel-h10.csv"
        if not panel_path.exists():
            panel_path = ROOT / "data/ml-panel-v1.csv"

        if not panel_path.exists():
            self.skipTest("No panel CSV available in results or data for Tier 4 audit")

        # Read only required columns for speed (< 1s)
        panel = pd.read_csv(panel_path, usecols=["date", "code", "label_end"], dtype=str)

        for year in range(2022, 2027):
            start = f"{year}-01-01"
            end = f"{year}-12-31" if year < 2026 else "2026-10-13"

            # Replicate canonical split mask:
            # 1. date < boundary
            # 2. label_end.notna() & label_end < boundary
            train_mask = panel.date.lt(start) & panel.label_end.notna() & panel.label_end.lt(start)
            eval_mask = panel.date.between(start, end)

            train_df = panel[train_mask]
            eval_df = panel[eval_mask]

            self.assertGreater(len(train_df), 1000, f"Year {year} training set too small")
            self.assertGreater(len(eval_df), 100, f"Year {year} evaluation set too small")

            audit_result = LeakageGuard.assert_temporal_isolation(train_df, eval_df, eval_start=start)
            self.assertEqual(audit_result["status"], "PASSED")
            self.assertLess(audit_result["train_date_max"], start)
            self.assertLess(audit_result["train_label_end_max"], start)
            self.assertGreaterEqual(audit_result["eval_date_min"], start)


if __name__ == "__main__":
    unittest.main()
