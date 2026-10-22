"""Train and freeze fixed 2026-forward models without loading 2026 market data."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from engine import Config
from ml_features import FEATURES
from run_ml import training_mask


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data" / "pilot-2020-01-02-2025-12-31-n300-s20201009"
PANEL = ROOT / "data" / "ml-panel-v1.csv"
PANEL_META = ROOT / "data" / "ml-panel-v1.meta.json"
OUTPUT = ROOT / "results" / "forward-2026-frozen"
BOUNDARY = "2026-01-01"
EVALUATION_RANGE_END = "2026-10-13"
MODEL_SOURCES = (
    "engine.py", "ml_features.py", "run.py", "signals.py", "validate_expanded.py",
    "run_ml.py", "ml_report.py", "ml-plan-v1.md", "freeze_2026.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def expected_input_hashes() -> dict[str, str]:
    return {
        "ml_features.py": sha256(ROOT / "ml_features.py"),
        "engine.py": sha256(ROOT / "engine.py"),
        "run_ml.py": sha256(ROOT / "run_ml.py"),
        "ml-plan-v1.md": sha256(ROOT / "ml-plan-v1.md"),
        "freeze_2026.py": sha256(ROOT / "freeze_2026.py"),
        "manifest.json": sha256(DATASET / "manifest.json"),
        "ml-panel-v1.csv": sha256(PANEL),
    }


def validate_inputs(meta: dict, hashes: dict[str, str]) -> None:
    required = {
        "source_sha256": hashes["ml_features.py"],
        "engine_sha256": hashes["engine.py"],
        "manifest_sha256": hashes["manifest.json"],
        "panel_sha256": hashes["ml-panel-v1.csv"],
    }
    for key, actual in required.items():
        if meta.get(key) != actual:
            raise ValueError(f"Stale or changed panel input: {key}")
    if meta.get("features") != FEATURES:
        raise ValueError("Feature panel schema does not match FEATURES")
    if meta.get("config") != asdict(Config(hold_days=3)):
        raise ValueError("Feature panel execution config does not match fixed Config")


def verify_existing(freeze_path: Path) -> dict:
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    current = expected_input_hashes()
    recorded = frozen.get("hashes", {})
    for name, value in current.items():
        if recorded.get(name) != value:
            raise ValueError(f"Existing freeze input hash differs; preserving freeze: {name}")
    for filename, key in (("2026-logistic.joblib", "2026-logistic.joblib"),
                           ("2026-lgb.txt", "2026-lgb.txt")):
        actual = sha256(OUTPUT / "models" / filename)
        if recorded.get(key) != actual:
            raise ValueError(f"Existing frozen model hash differs; preserving freeze: {filename}")
    print(f"EXISTING FREEZE VERIFIED {freeze_path}", flush=True)
    return frozen


def main() -> dict:
    freeze_path = OUTPUT / "freeze.json"
    if freeze_path.exists():
        return verify_existing(freeze_path)

    for path in (PANEL, PANEL_META, DATASET / "manifest.json", ROOT / "ml-plan-v1.md"):
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes = expected_input_hashes()
    meta = json.loads(PANEL_META.read_text(encoding="utf-8"))
    validate_inputs(meta, hashes)

    panel = pd.read_csv(PANEL, dtype={"date": str, "code": str, "label_end": str})
    if len(panel) != meta.get("rows") or panel.duplicated(["date", "code"]).any():
        raise ValueError("Panel row count or key uniqueness does not match metadata")
    if not panel.eligible.fillna(False).all():
        raise ValueError("Panel includes an ineligible prediction row")

    train = panel.loc[training_mask(panel, BOUNDARY)].copy()
    if train.empty or train.y_win.nunique() != 2:
        raise ValueError("Frozen training set is empty or contains only one class")
    if train.date.max() >= BOUNDARY or train.label_end.max() >= BOUNDARY:
        raise AssertionError("Training rows cross the frozen date boundary")
    if not np.isfinite(train[FEATURES].to_numpy(dtype=float)).all():
        raise AssertionError("Training features contain non-finite values")

    classifier = make_pipeline(StandardScaler(), LogisticRegression(
        C=.1, max_iter=300, random_state=20261014))
    regressor = lgb.LGBMRegressor(
        n_estimators=150, learning_rate=.03, num_leaves=7, max_depth=3,
        min_child_samples=300, reg_lambda=10, random_state=20261014,
        n_jobs=3, deterministic=True, force_col_wise=True, verbosity=-1)
    with threadpool_limits(limits=3):
        classifier.fit(train[FEATURES], train.y_win.astype(int))
        regressor.fit(train[FEATURES], train.net_return.clip(-.2, .2))

    models = OUTPUT / "models"
    models.mkdir(parents=True, exist_ok=True)
    sources = OUTPUT / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    logistic_path = models / "2026-logistic.joblib"
    lgb_path = models / "2026-lgb.txt"
    joblib.dump(classifier, logistic_path)
    regressor.booster_.save_model(str(lgb_path))
    source_hashes = {}
    for filename in MODEL_SOURCES:
        shutil.copy2(ROOT / filename, sources / filename)
        source_hashes[filename] = sha256(sources / filename)
    # The copied source tree must exactly match what was hashed before training.
    if source_hashes["freeze_2026.py"] != hashes["freeze_2026.py"]:
        raise AssertionError("Copied freeze script changed during the run")

    all_hashes = {
        **hashes,
        "2026-logistic.joblib": sha256(logistic_path),
        "2026-lgb.txt": sha256(lgb_path),
        "copied_sources": source_hashes,
    }
    payload = {
        "version": "forward-2026-frozen-v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "evaluation_range_end": EVALUATION_RANGE_END,
        "evaluation_data_loaded": False,
        "training_boundary_exclusive": BOUNDARY,
        "training_rows": int(len(train)),
        "training_positive_rows": int((train.y_win.astype(int) == 1).sum()),
        "training_negative_rows": int((train.y_win.astype(int) == 0).sum()),
        "training_date_min": str(train.date.min()),
        "training_date_max": str(train.date.max()),
        "training_label_end_max": str(train.label_end.max()),
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "config": asdict(Config(hold_days=3)),
        "decision_thresholds": {"logistic_p_win_min": 0.55, "lgb_pred_net_return_min": 0.005},
        "model_parameters": {
            "logistic": {"pipeline": ["StandardScaler", "LogisticRegression"],
                         "C": 0.1, "max_iter": 300, "random_state": 20261014},
            "lightgbm_regressor": {"n_estimators": 150, "learning_rate": 0.03,
                                   "num_leaves": 7, "max_depth": 3,
                                   "min_child_samples": 300, "reg_lambda": 10,
                                   "random_state": 20261014, "n_jobs": 3,
                                   "deterministic": True, "force_col_wise": True,
                                   "target_clip": [-0.2, 0.2]},
        },
        "versions": {"sklearn": sklearn.__version__, "lightgbm": lgb.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "source_dataset": DATASET.name,
        "panel_metadata": {"version": meta.get("version"), "rows": meta.get("rows"),
                           "labeled_rows": meta.get("labeled_rows")},
        "hashes": all_hashes,
        "notes": [
            "Trained only on labels with date and label_end strictly before 2026-01-01.",
            "No 2026 market data was loaded, predicted, evaluated, or replayed.",
            "evaluation_range_end is a planned scope boundary only; data through that date was not accessed.",
            "Model files are frozen artifacts; any later fitting requires a separate version and timestamp.",
        ],
    }
    write_json(freeze_path, payload)
    print(f"FROZEN {freeze_path}", flush=True)
    print(json.dumps({"training_rows": payload["training_rows"],
                      "training_date_max": payload["training_date_max"],
                      "training_label_end_max": payload["training_label_end_max"],
                      "logistic_sha256": all_hashes["2026-logistic.joblib"],
                      "lgb_sha256": all_hashes["2026-lgb.txt"]}, ensure_ascii=False), flush=True)
    return payload


if __name__ == "__main__":
    main()
