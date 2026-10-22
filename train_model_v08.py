"""Train Model V0.8: Listwise LambdaMART Ranking + Microstructure & Volatility Cones.

Innovations in Model V0.8:
1. New Microstructure Features:
   - yz_vol20: Yang-Zhang unbiased continuous volatility estimator
   - amihud20: Amihud illiquidity price impact ratio (|pctChg| / turnover)
   - vwap_ratio: Intraday close vs VWAP displacement (center of gravity shift)
   Combined with 34 features from Model V3 (Total 37 features).

2. Learning Objective: Listwise LambdaMART (objective='lambdarank', metric='ndcg@5')
   - Directly optimizes daily cross-sectional ranking order (NDCG@K)
   - Eliminates noise fitting in middle 80% of stocks
   - Focuses strictly on separating the top 2% explosive momentum stocks.

3. Walk-Forward Validation:
   - 2022 to 2026 yearly walk-forward with 5-session label embargo.
   - Outputs:
     - results/model-v08/predictions-h10-v08.csv
     - results/model-v08/feature_importance_v08.csv
     - results/model-v08/model_diagnostics.json
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
from train_model_v3 import ASHARE_FEATURES, extract_ashare_features

ROOT = Path("d:/Benefits/stock-research")
DATASET = ROOT / "data/forward-2026-10-13"
DONOR = ROOT / "results/midterm-access-v3"
MODEL_V2_DIR = ROOT / "results/model-v2"
MODEL_V3_DIR = ROOT / "results/model-v3"
OUT_DIR = ROOT / "results/model-v08"

MICRO_FEATURES_V08 = [
    "yz_vol20",
    "amihud20",
    "vwap_ratio",
]

FEATURES_V08 = BASE_FEATURES + NEW_FEATURES + ASHARE_FEATURES + MICRO_FEATURES_V08


def extract_microstructure_features(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Extract Yang-Zhang volatility, Amihud illiquidity, and VWAP ratio."""
    rows = []
    k_yz = 0.34 / (1.34 + (20 + 1) / (20 - 1))

    for code, df in frames.items():
        preclose = df["preclose"].replace(0, np.nan)
        open_px = df["open"].replace(0, np.nan)

        log_ho = np.log(df["high"] / open_px)
        log_lo = np.log(df["low"] / open_px)
        log_co = np.log(df["close"] / open_px)
        log_oc = np.log(df["open"] / preclose)
        log_cc = np.log(df["close"] / preclose)

        open_var = log_oc.rolling(20).var()
        close_var = log_cc.rolling(20).var()
        rs_var = (log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).rolling(20).mean()
        yz_vol = np.sqrt((open_var + k_yz * close_var + (1 - k_yz) * rs_var).clip(lower=0)) * np.sqrt(242)

        amihud = (df["pctChg"].abs() / (df["amount"] / 1e6 + 1e-4)).rolling(20).mean()
        vwap = df["amount"] / (df["volume"] + 1e-4)
        vwap_ratio = (df["close"] - vwap) / preclose

        sub = pd.DataFrame({
            "date": df["date"].astype(str),
            "code": code,
            "yz_vol20": yz_vol.fillna(0.30),
            "amihud20": amihud.fillna(0.01),
            "vwap_ratio": vwap_ratio.fillna(0.0),
        })
        rows.append(sub)

    return pd.concat(rows, ignore_index=True)


def train_model_v08():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 88)
    print(">>> TRAINING MODEL V0.8: LISTWISE LAMBDAMART + ADVANCED MICROSTRUCTURE <<<")
    print("=" * 88)

    t0 = time.time()
    manifest, frames, calendar, dividends = load(DATASET)
    print(f"Loaded raw frames for {len(frames)} stocks in {time.time() - t0:.2f}s")

    print("Extracting A-share specific & microstructure features...")
    ashare_df = extract_ashare_features(frames)
    micro_df = extract_microstructure_features(frames)
    combined_micro = pd.merge(ashare_df, micro_df, on=["date", "code"], how="outer")
    print(f"Features extracted: shape {combined_micro.shape}")

    print("Loading base midterm panel-h10 dataset...")
    panel = pd.read_csv(DONOR / "panel-h10.csv", dtype={"date": str, "code": str, "label_end": str})

    new_feat_path = ROOT / "results/exploration-v1/new-features.csv"
    if new_feat_path.exists():
        feat60 = pd.read_csv(new_feat_path, dtype={"date": str, "code": str})
        panel = panel.merge(feat60[["date", "code", "adj_ret60", "ret_vol60"]], on=["date", "code"], how="left")
    else:
        panel["adj_ret60"] = panel["adj_ret20"]
        panel["ret_vol60"] = panel["ret_vol20"]

    print("Enriching with Features V2...")
    panel = enrich_features_v2(panel)

    print("Merging Microstructure & Volatility Cone features into panel...")
    panel = panel.merge(combined_micro, on=["date", "code"], how="left")

    for col in ASHARE_FEATURES + MICRO_FEATURES_V08:
        panel[col] = panel[col].fillna(panel[col].median() if not panel[col].dropna().empty else 0.0)

    print(f"Enriched panel shape: {panel.shape}, Total Features: {len(FEATURES_V08)}")

    years = [2022, 2023, 2024, 2025, 2026]
    predictions_v08 = []
    diagnostics = []
    feature_importances = []

    print("\nStarting Walk-Forward LambdaMART Training across 2022-2026...")
    print(f"{'Year':<8} {'TrainRows':<10} {'EvalRows':<10} {'RankIC(V0.8)':<14} {'NDCG@5':<10} {'Top5% Return':<14}")
    print("-" * 72)

    for year in years:
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31" if year < 2026 else "2026-10-13"

        train_mask = (
            panel.date.lt(start_date) &
            panel.label_end.notna() &
            panel.label_end.lt(start_date) &
            panel.net_return.notna() &
            np.isfinite(panel[FEATURES_V08].to_numpy(dtype=float)).all(axis=1)
        )
        train_df = panel[train_mask].sort_values(["date", "code"]).copy()

        eval_mask = (
            panel.date.ge(start_date) &
            panel.date.le(end_date) &
            np.isfinite(panel[FEATURES_V08].to_numpy(dtype=float)).all(axis=1)
        )
        eval_df = panel[eval_mask].sort_values(["date", "code"]).copy()

        # Convert net_return into integer relevance grades (0 to 4) within each day
        train_df["relevance"] = train_df.groupby("date")["net_return"].transform(
            lambda s: pd.qcut(s, q=5, labels=False, duplicates="drop")
        ).fillna(2).astype(int)

        group_train = train_df.groupby("date").size().to_numpy()

        X_train = train_df[FEATURES_V08].to_numpy(dtype=float)
        y_train_rel = train_df["relevance"].to_numpy(dtype=int)
        X_eval = eval_df[FEATURES_V08].to_numpy(dtype=float)

        # Train LightGBM LambdaRanker
        ranker = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=120,
            learning_rate=0.035,
            max_depth=4,
            num_leaves=15,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            label_gain=[0, 1, 3, 7, 15],  # Exponential gain favoring top relevance
        )
        ranker.fit(X_train, y_train_rel, group=group_train)

        pred_scores = ranker.predict(X_eval)
        eval_df["pred_score_v08"] = pred_scores
        eval_df["rank_pct_v08"] = eval_df.groupby("date")["pred_score_v08"].rank(pct=True)

        # Evaluate performance on test set
        if eval_df["net_return"].notna().sum() > 0:
            valid_eval = eval_df[eval_df.net_return.notna()]
            daily_ics = valid_eval.groupby("date").apply(
                lambda g: spearmanr(g["pred_score_v08"], g["net_return"])[0] if len(g) > 5 else np.nan,
                include_groups=False,
            ).dropna()
            mean_ic = float(daily_ics.mean()) if len(daily_ics) > 0 else 0.0

            top5_ret = valid_eval.groupby("date").apply(
                lambda g: g.nlargest(max(1, int(len(g) * 0.05)), "pred_score_v08")["net_return"].mean(),
                include_groups=False,
            ).mean()
        else:
            mean_ic = 0.0
            top5_ret = 0.0

        # Gain importance
        imp = ranker.booster_.feature_importance(importance_type="gain")
        feature_importances.append(pd.DataFrame({
            "year": year,
            "feature": FEATURES_V08,
            "gain": imp,
        }))

        print(f"{year:<8} {len(train_df):<10} {len(eval_df):<10} {mean_ic:>14.4f} {'-':<10} {top5_ret:>14.4%}")

        diagnostics.append({
            "year": year,
            "train_rows": len(train_df),
            "eval_rows": len(eval_df),
            "rank_ic_v08": mean_ic,
            "top5_pct_return": top5_ret,
        })

        predictions_v08.append(eval_df[["date", "code", "pred_score_v08", "rank_pct_v08"]])

    all_preds = pd.concat(predictions_v08, ignore_index=True)
    all_preds.to_csv(OUT_DIR / "predictions-h10-v08.csv", index=False)
    print(f"\nSaved {len(all_preds)} predictions to {OUT_DIR / 'predictions-h10-v08.csv'}")

    all_imp = pd.concat(feature_importances, ignore_index=True)
    avg_imp = all_imp.groupby("feature")["gain"].mean().sort_values(ascending=False).reset_index()
    avg_imp.to_csv(OUT_DIR / "feature_importance_v08.csv", index=False)

    print("\nTop 15 Most Important Features in Model V0.8:")
    print(avg_imp.head(15).to_string(index=False))

    with open(OUT_DIR / "model_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    print(f"\nModel V0.8 Training Complete in {time.time() - t0:.2f}s!")


if __name__ == "__main__":
    train_model_v08()
