"""Model ensemble scoring and signal generation for Logistic Regression and LightGBM.

This module provides methods to combine probability predictions (p_win from LogisticRegression)
and return predictions (pred_net_return from LGBMRegressor) to form more robust signals:
1. 'intersection': Strict conservative gating (p_win >= 0.55 AND pred_return >= 0.005).
2. 'average': Symmetrically normalized arithmetic average score.
3. 'weighted': Symmetrically normalized weighted score with configurable model weights.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _normalize_series(series: pd.Series) -> pd.Series:
    """Robust min-max normalization of a numeric Series to [0.0, 1.0]."""
    clean = series.dropna()
    if clean.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    min_val, max_val = clean.min(), clean.max()
    if max_val == min_val:
        return pd.Series(0.5, index=series.index, dtype=float)
    return (series - min_val) / (max_val - min_val)


def ensemble_score(predictions: pd.DataFrame,
                   method: str = "intersection",
                   p_win_threshold: float = 0.55,
                   return_threshold: float = 0.005,
                   weights: tuple[float, float] = (0.5, 0.5)) -> pd.DataFrame:
    """Calculate ensemble score and selection flag from model predictions.

    Parameters:
        predictions: DataFrame with columns 'date', 'code', 'p_win', 'pred_net_return'.
        method: 'intersection', 'average', or 'weighted'.
        p_win_threshold: Minimum win probability for intersection method (default 0.55).
        return_threshold: Minimum predicted return for intersection method (default 0.005).
        weights: (lr_weight, lgb_weight) for weighted method (default (0.5, 0.5)).

    Returns:
        DataFrame with ['date', 'code', 'ensemble_score', 'selected'].
    """
    req = {"date", "code", "p_win", "pred_net_return"}
    missing = req.difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions DataFrame missing required columns: {sorted(missing)}")

    result = predictions[["date", "code"]].copy()
    p_win = pd.to_numeric(predictions["p_win"], errors="coerce")
    pred_ret = pd.to_numeric(predictions["pred_net_return"], errors="coerce")

    # Normalized representations for combining scales
    norm_p = _normalize_series(p_win)
    norm_ret = _normalize_series(pred_ret)

    if method == "intersection":
        selected = (p_win >= p_win_threshold) & (pred_ret >= return_threshold)
        # Score is combined normalized confidence for ranking among selected
        score = 0.5 * norm_p + 0.5 * norm_ret
    elif method == "average":
        score = 0.5 * norm_p + 0.5 * norm_ret
        # Average threshold: both normalized components average above 0.5
        selected = score >= 0.5
    elif method == "weighted":
        w1, w2 = weights
        total_w = w1 + w2
        if total_w <= 0:
            raise ValueError("Sum of weights must be positive")
        w1, w2 = w1 / total_w, w2 / total_w
        score = w1 * norm_p + w2 * norm_ret
        selected = score >= 0.5
    else:
        raise ValueError(f"Unknown ensemble method: {method}. Choose from 'intersection', 'average', 'weighted'")

    result["ensemble_score"] = score
    result["selected"] = selected.fillna(False).astype(bool)
    return result


def ensemble_signals(predictions: pd.DataFrame,
                     method: str = "intersection",
                     p_win_threshold: float = 0.55,
                     return_threshold: float = 0.005,
                     weights: tuple[float, float] = (0.5, 0.5)) -> pd.DataFrame:
    """Generate engine-compatible ranked signals from ensemble model.

    Returns:
        DataFrame with columns: ['date', 'code', 'score'], filtered to selected==True,
        sorted descending by score and ascending by code.
    """
    scored = ensemble_score(predictions, method=method,
                            p_win_threshold=p_win_threshold,
                            return_threshold=return_threshold,
                            weights=weights)

    selected = scored[scored.selected].copy()
    if selected.empty:
        return pd.DataFrame(columns=["date", "code", "score"])

    signals = selected.rename(columns={"ensemble_score": "score"})[["date", "code", "score"]]
    signals["date"] = signals["date"].astype(str).str[:10]
    signals["code"] = signals["code"].astype(str)
    signals["score"] = signals["score"].astype(float)

    if signals.duplicated(["date", "code"]).any():
        raise ValueError("Duplicate signals generated for the same date and code")

    return signals.sort_values(["date", "score", "code"], ascending=[True, False, True]).reset_index(drop=True)


def compare_with_singles(predictions: pd.DataFrame,
                         p_win_threshold: float = 0.55,
                         return_threshold: float = 0.005) -> dict[str, Any]:
    """Compare candidate selections between single models and ensemble variants.

    Returns:
        Dictionary summarizing signal counts, date coverage, and overlap statistics.
    """
    lr_mask = (pd.to_numeric(predictions["p_win"], errors="coerce") >= p_win_threshold)
    lgb_mask = (pd.to_numeric(predictions["pred_net_return"], errors="coerce") >= return_threshold)

    ens_intersect = ensemble_score(predictions, method="intersection",
                                   p_win_threshold=p_win_threshold,
                                   return_threshold=return_threshold)["selected"]
    ens_avg = ensemble_score(predictions, method="average")["selected"]

    def set_of_pairs(mask):
        sub = predictions[mask]
        return set(zip(sub["date"].astype(str).str[:10], sub["code"].astype(str)))

    set_lr = set_of_pairs(lr_mask)
    set_lgb = set_of_pairs(lgb_mask)
    set_int = set_of_pairs(ens_intersect)
    set_avg = set_of_pairs(ens_avg)

    def jaccard(s1, s2):
        union = len(s1.union(s2))
        return len(s1.intersection(s2)) / union if union else 0.0

    return {
        "counts": {
            "logistic_selective": len(set_lr),
            "lgb_selective": len(set_lgb),
            "ensemble_intersection": len(set_int),
            "ensemble_average": len(set_avg),
        },
        "signal_dates": {
            "logistic_selective": predictions.loc[lr_mask, "date"].nunique(),
            "lgb_selective": predictions.loc[lgb_mask, "date"].nunique(),
            "ensemble_intersection": predictions.loc[ens_intersect, "date"].nunique(),
            "ensemble_average": predictions.loc[ens_avg, "date"].nunique(),
        },
        "jaccard_similarity": {
            "lr_vs_lgb": round(jaccard(set_lr, set_lgb), 4),
            "intersection_vs_lgb": round(jaccard(set_int, set_lgb), 4),
            "intersection_vs_lr": round(jaccard(set_int, set_lr), 4),
            "average_vs_lgb": round(jaccard(set_avg, set_lgb), 4),
            "average_vs_lr": round(jaccard(set_avg, set_lr), 4),
        },
    }
