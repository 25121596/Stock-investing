"""Prototype test comparing Pointwise Huber vs Listwise LambdaRank in LightGBM for A-share factor ranking.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

ROOT = Path("d:/Benefits/stock-research")
DONOR = ROOT / "results/midterm-access-v3"
MODEL_V2_DIR = ROOT / "results/model-v2"

# Load panel-h10
panel = pd.read_csv(DONOR / "panel-h10.csv", dtype={"date": str, "code": str, "label_end": str})
new_feat_path = ROOT / "results/exploration-v1/new-features.csv"
if new_feat_path.exists():
    feat60 = pd.read_csv(new_feat_path, dtype={"date": str, "code": str})
    panel = panel.merge(feat60[["date", "code", "adj_ret60", "ret_vol60"]], on=["date", "code"], how="left")
else:
    panel["adj_ret60"] = panel["adj_ret20"]
    panel["ret_vol60"] = panel["ret_vol20"]

from train_model_v2 import BASE_FEATURES, NEW_FEATURES, enrich_features_v2
panel = enrich_features_v2(panel)

from train_model_v3 import ASHARE_FEATURES, extract_ashare_features, FEATURES_V3
from run import load
manifest, frames, calendar, dividends = load(ROOT / "data/forward-2026-10-13")
ashare_df = extract_ashare_features(frames)
panel = panel.merge(ashare_df, on=["date", "code"], how="left")

for col in ASHARE_FEATURES:
    panel[col] = panel[col].fillna(panel[col].median() if not panel[col].dropna().empty else 0.0)

print(f"Panel ready. Shape: {panel.shape}, Features: {len(FEATURES_V3)}")

# Test on 2024 (evaluation on 2024, train on < 2024)
train_mask = (panel.date.lt("2024-01-01") & panel.label_end.notna() & panel.label_end.lt("2024-01-01") & panel.net_return.notna())
eval_mask = (panel.date.ge("2024-01-01") & panel.date.le("2024-12-31") & panel.net_return.notna())

train_df = panel[train_mask].sort_values(["date", "code"])
eval_df = panel[eval_mask].sort_values(["date", "code"])

X_train = train_df[FEATURES_V3].to_numpy(dtype=float)
y_train = train_df["net_return"].to_numpy(dtype=float)
X_eval = eval_df[FEATURES_V3].to_numpy(dtype=float)
y_eval = eval_df["net_return"].to_numpy(dtype=float)

# 1. Pointwise Huber Regressor
reg_huber = lgb.LGBMRegressor(
    objective="huber", n_estimators=100, learning_rate=0.03,
    max_depth=4, num_leaves=15, min_child_samples=50,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
)
reg_huber.fit(X_train, y_train)
pred_huber = reg_huber.predict(X_eval)

# Compute daily Rank IC for Huber
eval_df = eval_df.copy()
eval_df["pred_huber"] = pred_huber
ic_huber = eval_df.groupby("date").apply(lambda g: spearmanr(g["pred_huber"], g["net_return"])[0]).mean()
print(f"2024 Pointwise Huber Mean Daily Rank IC: {ic_huber:.4f}")

# 2. LambdaMART / Pairwise Ranking
# Discretize net_return into 5 integer relevance grades (0 to 4) per date
train_df = train_df.copy()
train_df["relevance"] = train_df.groupby("date")["net_return"].transform(lambda s: pd.qcut(s, q=5, labels=False, duplicates="drop")).fillna(2).astype(int)
group_train = train_df.groupby("date").size().to_numpy()

ranker = lgb.LGBMRanker(
    objective="lambdarank", n_estimators=100, learning_rate=0.03,
    max_depth=4, num_leaves=15, min_child_samples=50,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
)
ranker.fit(X_train, train_df["relevance"].to_numpy(), group=group_train)
pred_rank = ranker.predict(X_eval)

eval_df["pred_rank"] = pred_rank
ic_rank = eval_df.groupby("date").apply(lambda g: spearmanr(g["pred_rank"], g["net_return"])[0]).mean()
print(f"2024 Listwise LambdaRank Mean Daily Rank IC: {ic_rank:.4f}")

# Compare Top 5% precision (actual return of top 5% selected stocks)
top_huber = eval_df.groupby("date").apply(lambda g: g.nlargest(max(1, int(len(g)*0.05)), "pred_huber")["net_return"].mean()).mean()
top_rank = eval_df.groupby("date").apply(lambda g: g.nlargest(max(1, int(len(g)*0.05)), "pred_rank")["net_return"].mean()).mean()
print(f"2024 Top 5% Stocks Forward Return: Huber = {top_huber:.4%}, LambdaRank = {top_rank:.4%}")
