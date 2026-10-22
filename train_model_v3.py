"""Train Model V3: A-Share Specific Features + Huber Loss Objective.

New A-Share Microstructure & Behavioral Features:
1. turn5: 5-day rolling average turnover rate
2. turn_ratio: turn5 / (turn20 + 1e-4) (volume surge vs consolidation ratio)
3. upper_shadow: (high - max(open, close)) / preclose (selling pressure / overhead resistance)
4. lower_shadow: (min(open, close) - low) / preclose (support / dip-buying absorption)
5. close_strength: (close - low) / (high - low + 1e-4) (intraday close strength / late-day accumulation)
6. limit_count_60: 60-day rolling sum of days touching 9.5%+ limit-up (A-share momentum gene)
7. vol_price_corr20: 20-day rolling correlation between return and volume (volume-price harmony)

Combined with:
- 20 Base Technical Features
- 7 Features V2 (cross-sectional rank_vol20, rank_ret20, ret_vol60, vol_ratio, etc.)
Total: 34 features.

Walk-Forward Training with 5-day embargo across 2022-2026.
Outputs:
- results/model-v3/predictions-h10-v3.csv
- results/model-v3/feature_importance_v3.csv
"""

from __future__ import annotations

import json
from pathlib import Path
import time
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from run import load
from train_model_v2 import BASE_FEATURES, NEW_FEATURES, enrich_features_v2

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data/forward-2026-10-13"
DONOR = ROOT / "results/midterm-access-v3"
MODEL_V2_DIR = ROOT / "results/model-v2"
OUT_DIR = ROOT / "results/model-v3"

ASHARE_FEATURES = [
    "upper_shadow",
    "lower_shadow",
    "close_strength",
    "turn5",
    "turn_ratio",
    "limit_count_60",
    "vol_price_corr20",
]

FEATURES_V3 = BASE_FEATURES + NEW_FEATURES + ASHARE_FEATURES


def extract_ashare_features(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Extract A-share unique microstructure and turnover features."""
    feature_rows = []
    for code, df in frames.items():
        preclose = df["preclose"].replace(0, np.nan)
        hl_range = df["high"] - df["low"]

        # Shadows & Close position
        upper_s = (df["high"] - df[["open", "close"]].max(axis=1)) / preclose
        lower_s = (df[["open", "close"]].min(axis=1) - df["low"]) / preclose
        close_str = (df["close"] - df["low"]) / (hl_range + 1e-4)

        # Turnover dynamics
        t5 = df["turn"].rolling(5).mean()
        t20 = df["turn"].rolling(20).mean()
        turn_rat = t5 / (t20 + 1e-4)

        # Limit-up gene
        limit_touch = (df["high"] >= np.round(df["preclose"] * 1.095, 2)).astype(float)
        limit_cnt = limit_touch.rolling(60).sum()

        # Volume-price correlation
        vol_ret_corr = df["pctChg"].rolling(20).corr(df["volume"])

        sub = pd.DataFrame({
            "date": df["date"].astype(str),
            "code": code,
            "upper_shadow": upper_s.fillna(0.0),
            "lower_shadow": lower_s.fillna(0.0),
            "close_strength": close_str.fillna(0.5),
            "turn5": t5.fillna(0.5),
            "turn_ratio": turn_rat.fillna(1.0),
            "limit_count_60": limit_cnt.fillna(0.0),
            "vol_price_corr20": vol_ret_corr.fillna(0.0),
        })
        feature_rows.append(sub)

    return pd.concat(feature_rows, ignore_index=True)


def train_model_v3():
    print("=" * 88)
    print(">>> TRAINING MODEL V3: A-SHARE SPECIFIC FEATURES + HUBER REGRESSOR <<<")
    print("=" * 88)

    t0 = time.time()
    manifest, frames, calendar, dividends = load(DATASET)
    print(f"Loaded raw frames for {len(frames)} stocks in {time.time() - t0:.2f}s")

    print("Extracting A-share specific features from raw tick/bar frames...")
    ashare_df = extract_ashare_features(frames)
    print(f"A-share features extracted: {ashare_df.shape}")

    print("Loading base midterm panel-h10 dataset...")
    panel = pd.read_csv(DONOR / "panel-h10.csv", dtype={"date": str, "code": str, "label_end": str})

    # Merge 60-day features from exploration
    new_feat_path = ROOT / "results/exploration-v1/new-features.csv"
    if new_feat_path.exists():
        feat60 = pd.read_csv(new_feat_path, dtype={"date": str, "code": str})
        panel = panel.merge(feat60[["date", "code", "adj_ret60", "ret_vol60"]], on=["date", "code"], how="left")
    else:
        panel["adj_ret60"] = panel["adj_ret20"]
        panel["ret_vol60"] = panel["ret_vol20"]

    print("Enriching with Features V2...")
    panel = enrich_features_v2(panel)

    print("Merging A-share features into panel...")
    panel = panel.merge(ashare_df, on=["date", "code"], how="left")

    # Fill any remaining NaNs in A-share features
    for col in ASHARE_FEATURES:
        panel[col] = panel[col].fillna(panel[col].median() if not panel[col].dropna().empty else 0.0)

    print(f"Enriched panel shape: {panel.shape}, Total Features: {len(FEATURES_V3)}")

    # Load baseline V2 predictions for side-by-side comparison
    preds_v2 = pd.read_csv(MODEL_V2_DIR / "predictions-h10-v2.csv", dtype={"date": str, "code": str})
    v2_lookup = preds_v2.set_index(["date", "code"])["pred_net_return_v2"].to_dict()

    years = [2022, 2023, 2024, 2025, 2026]
    predictions_v3 = []
    diagnostics = []

    print("\nStarting Walk-Forward Training across 2022-2026...")
    print(f"{'Year':<8} {'TrainRows':<10} {'EvalRows':<10} {'RankIC(V3)':<12} {'RankIC(V2)':<12} {'Alpha Gain':<12}")
    print("-" * 68)

    for year in years:
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31" if year < 2026 else "2026-10-13"

        # Embargo of 5 trading days: label_end must be before start_date
        train_mask = (
            panel.date.lt(start_date) &
            panel.label_end.notna() &
            panel.label_end.lt(start_date) &
            panel.net_return.notna() &
            np.isfinite(panel[FEATURES_V3].to_numpy(dtype=float)).all(axis=1)
        )
        train_df = panel[train_mask]

        eval_mask = (
            panel.date.ge(start_date) &
            panel.date.le(end_date) &
            np.isfinite(panel[FEATURES_V3].to_numpy(dtype=float)).all(axis=1)
        )
        eval_df = panel[eval_mask].copy()

        if len(train_df) < 1000 or eval_df.empty:
            continue

        regressor = lgb.LGBMRegressor(
            objective="huber",
            alpha=0.90,
            n_estimators=220,
            learning_rate=0.02,
            num_leaves=15,
            max_depth=4,
            min_child_samples=250,
            reg_lambda=15.0,
            colsample_bytree=0.80,
            subsample=0.85,
            random_state=20261019,
            n_jobs=4,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )

        X_train = train_df[FEATURES_V3]
        y_train = train_df.net_return.clip(-0.20, 0.20)

        regressor.fit(X_train, y_train)

        eval_df["pred_net_return_v3"] = regressor.predict(eval_df[FEATURES_V3])
        eval_df["pred_net_return_v2"] = eval_df.apply(
            lambda r: v2_lookup.get((r["date"], r["code"]), np.nan), axis=1
        )

        labeled = eval_df[eval_df.net_return.notna()]
        if not labeled.empty:
            ic_v3, _ = spearmanr(labeled["pred_net_return_v3"], labeled["net_return"])
            ic_v2, _ = spearmanr(labeled["pred_net_return_v2"].dropna(),
                                 labeled.loc[labeled.pred_net_return_v2.notna(), "net_return"])
            gain = ic_v3 - ic_v2
            print(f"{year:<8} {len(train_df):<10} {len(eval_df):<10} {ic_v3:>11.4f} {ic_v2:>11.4f} {gain:>+11.4f}")
            diagnostics.append({"year": year, "ic_v3": ic_v3, "ic_v2": ic_v2, "gain": gain})

        predictions_v3.append(eval_df[["date", "code", "pred_net_return_v3", "net_return", "y_win", "label_end"]])

    all_preds_v3 = pd.concat(predictions_v3, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save predictions
    all_preds_v3.to_csv(OUT_DIR / "predictions-h10-v3.csv", index=False, encoding="utf-8-sig")

    # Feature Importance
    gain = regressor.booster_.feature_importance(importance_type="gain")
    feat_imp = pd.DataFrame({"feature": FEATURES_V3, "gain": gain}).sort_values("gain", ascending=False)
    feat_imp.to_csv(OUT_DIR / "feature_importance_v3.csv", index=False, encoding="utf-8-sig")

    print("\nTop 15 Most Important Features in Model V3 (by Gain):")
    print(feat_imp.head(15).to_string(index=False))

    ashare_ranks = feat_imp[feat_imp.feature.isin(ASHARE_FEATURES)]
    print("\nA-Share Specific Features Performance & Rankings:")
    print(ashare_ranks.to_string(index=False))

    print(f"\nModel V3 predictions saved to: {OUT_DIR / 'predictions-h10-v3.csv'}")


if __name__ == "__main__":
    train_model_v3()
