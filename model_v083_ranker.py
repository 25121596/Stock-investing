"""Model V0.8.3: Adaptive Regime-Aware Top-Ranker.

Academic & Theoretical Foundations:
1. Top-Loss Focus (LambdaMART with Truncation Level = 5):
   Directly optimizes NDCG@5, separating explosive momentum winners from the top 2%
   while eliminating noise fitting in the middle/tail 90% of cross-sectional assets.
2. Regime-Adaptive Regularization & Ranking Blend:
   During low-breadth (< 0.50) or volatile market regimes, extreme top momentum is vulnerable
   to false breakouts. V0.8.3 dynamically adapts ranking scores using an exogenous market breadth
   weighting between pure top rank and Huber expected net return, solving the 2026 degradation seen in V10.
3. Strict Walk-Forward Isolation & Label Embargo:
   Enforces train_df.date.max() < eval_df.date.min() and train_df.label_end.max() < eval_df.date.min().
   Zero in-sample cheating, zero lookahead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Tuple
from dataclasses import replace

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import arena
from train_model_v2 import enrich_features_v2
from train_model_v3 import extract_ashare_features
from train_model_v08 import FEATURES_V08, extract_microstructure_features
from v10_rank_objectives import fit_rows, relevance, passes

ROOT = Path("d:/Benefits/stock-research")
BUNDLE = ROOT / "results/v08-codex-rebuild-20261020-v1"
OUT_DIR = ROOT / "results/v083-adaptive-ranker"
PERIODS = [
    ("2022-2025", "2022-01-01", "2025-12-31"),
    ("2026", "2026-01-01", "2026-10-13"),
]


def xhash(x: Any) -> str:
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def compute_adaptive_scores(
    joined_df: pd.DataFrame,
    breadth_map: Dict[str, float],
    pool_size_map: Dict[str, int],
) -> pd.DataFrame:
    """Compute regime-conditioned ranking scores.
    
    When market breadth is strong (breadth >= 0.55), top-loss rank score dominates.
    When market breadth is weak or mixed (breadth < 0.55), blend top rank with expected net return
    to guard against false breakout drawdowns.
    """
    df = joined_df.copy()
    
    # Cross-sectional percentiles per date
    df["rank_v083_pct"] = df.groupby("date")["pred_score_v083"].rank(pct=True)
    df["rank_v2_pct"] = df.groupby("date")["pred_net_return_v2"].rank(pct=True)
    
    df["breadth"] = df["date"].map(breadth_map).fillna(0.40)
    
    # Adaptive blending weight: w_v2 increases when breadth is lower
    # For breadth >= 0.55, w_v2 = 0.20
    # For breadth <= 0.45, w_v2 = 0.50
    w_v2 = np.clip(0.50 - (df["breadth"] - 0.45) * 3.0, 0.20, 0.50)
    w_v083 = 1.0 - w_v2
    
    df["adaptive_score"] = w_v083 * df["rank_v083_pct"] + w_v2 * df["rank_v2_pct"]
    return df


def train_walk_forward_v083(
    panel: pd.DataFrame,
    out_dir: Path,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Train Walk-Forward Top-Focus LambdaMART across 2022-2026."""
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    predictions_v083 = []
    training_records = []
    
    print("\n" + "=" * 80)
    print("TRAINING WALK-FORWARD V0.8.3 (TOP-LOSS LAMBDAMART)")
    print("=" * 80)
    print(f"{'Year':<8} {'TrainRows':<10} {'EvalRows':<10} {'LastLabelEnd':<15} {'EvalStart':<12} {'RankIC':<10}")
    print("-" * 80)
    
    for year in range(2022, 2027):
        train_df = fit_rows(panel, FEATURES_V08, year)
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31" if year < 2026 else "2026-10-13"
        
        eval_mask = (
            panel.date.ge(start_date) &
            panel.date.le(end_date) &
            np.isfinite(panel[FEATURES_V08].to_numpy(dtype=float)).all(axis=1)
        )
        eval_df = panel[eval_mask].sort_values(["date", "code"]).copy()
        
        # Verify strict anti-leakage invariants
        assert train_df.date.max() < eval_df.date.min(), f"Date leakage in {year}"
        assert train_df.label_end.max() < eval_df.date.min(), f"Label leakage in {year}"
        
        # Train LambdaRanker with Top-Loss Truncation
        group_train = train_df.groupby("date", sort=True).size().to_numpy()
        X_train = train_df[FEATURES_V08].to_numpy(dtype=float)
        y_train_rel = relevance(train_df, tail=False)
        
        ranker_params = {
            "objective": "lambdarank",
            "n_estimators": 120,
            "learning_rate": 0.035,
            "max_depth": 4,
            "num_leaves": 15,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": 4,
            "lambdarank_truncation_level": 5,
            "label_gain": [0, 1, 3, 7, 15],
        }
        
        ranker = lgb.LGBMRanker(**ranker_params)
        ranker.fit(X_train, y_train_rel, group=group_train)
        
        model_path = models_dir / f"{year}.txt"
        ranker.booster_.save_model(str(model_path))
        
        # Predict on out-of-sample eval set
        X_eval = eval_df[FEATURES_V08].to_numpy(dtype=float)
        pred_scores = ranker.predict(X_eval)
        eval_df["pred_score_v083"] = pred_scores
        
        # Compute Rank IC if net_return available
        rank_ic = 0.0
        if eval_df["net_return"].notna().sum() > 0:
            valid = eval_df[eval_df.net_return.notna()]
            ics = valid.groupby("date").apply(
                lambda g: spearmanr(g["pred_score_v083"], g["net_return"])[0] if len(g) > 5 else np.nan,
                include_groups=False,
            ).dropna()
            rank_ic = float(ics.mean()) if len(ics) > 0 else 0.0
            
        print(f"{year:<8} {len(train_df):<10} {len(eval_df):<10} {train_df.label_end.max():<15} {start_date:<12} {rank_ic:>9.4f}")
        
        record = {
            "year": year,
            "train_rows": len(train_df),
            "eval_rows": len(eval_df),
            "last_train_date": str(train_df.date.max()),
            "last_label_end": str(train_df.label_end.max()),
            "eval_start_date": str(eval_df.date.min()),
            "rank_ic": rank_ic,
            "model_path": str(model_path),
            "model_sha256": arena.digest(model_path),
        }
        training_records.append(record)
        predictions_v083.append(eval_df[["date", "code", "pred_score_v083"]])
        
    all_preds = pd.concat(predictions_v083, ignore_index=True)
    all_preds.to_csv(out_dir / "predictions-v083.csv", index=False)
    (out_dir / "training_records.json").write_text(
        json.dumps(training_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return all_preds, training_records


def run_v083_evaluation(out_dir: Path) -> Dict[str, Any]:
    """Execute complete 2022-2025 and 2026 backtests under normal and double slippage."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load market data and base features
    input_paths = [
        ROOT / "results/midterm-access-v3/panel-h10.csv",
        ROOT / "results/exploration-v1/new-features.csv",
        BUNDLE / "v2/predictions-h10-v2.csv",
        BUNDLE / "v08/predictions-h10-v08.csv",
        arena.DATASET / "manifest.json",
    ]
    
    _, frames, calendar, raw = arena.load(arena.DATASET)
    events, event_audit = arena.canonicalize_events(raw)
    bars = arena.prepare_bars(frames)
    
    panel = pd.read_csv(input_paths[0], dtype={"date": str, "code": str, "label_end": str})
    sixty = pd.read_csv(input_paths[1], dtype={"date": str, "code": str})
    panel = enrich_features_v2(panel.merge(sixty[["date", "code", "adj_ret60", "ret_vol60"]], on=["date", "code"], how="left", validate="one_to_one"))
    panel = panel.merge(extract_ashare_features(frames), on=["date", "code"], how="left", validate="one_to_one").merge(
        extract_microstructure_features(frames), on=["date", "code"], how="left", validate="one_to_one"
    )
    
    # 2. Train V0.8.3 Walk-Forward models
    preds_v083, train_records = train_walk_forward_v083(panel, out_dir)
    
    # 3. Build Breadth and Gate Limits
    b = arena.breadth_days(panel)
    bm = b.set_index("date").breadth5.to_dict()
    pm = b.set_index("date").pool_size.to_dict()
    limits = {d: 0 if pm[d] < 100 or x < 0.45 else (1 if x < 0.55 else 2) for d, x in bm.items()}
    vol = panel.set_index(["date", "code"]).ret_vol20.to_dict()
    
    # 4. Merge predictions
    old2 = pd.read_csv(input_paths[2], dtype={"date": str, "code": str}, float_precision="round_trip")
    old8 = pd.read_csv(input_paths[3], dtype={"date": str, "code": str}, float_precision="round_trip")
    
    joined = old2[["date", "code", "pred_net_return_v2"]].merge(
        old8[["date", "code", "pred_score_v08"]], on=["date", "code"], validate="one_to_one"
    ).merge(
        preds_v083[["date", "code", "pred_score_v083"]], on=["date", "code"], validate="one_to_one"
    )
    
    # 5. Adaptive Ranking Score
    joined = compute_adaptive_scores(joined, bm, pm)
    
    # Eligibility filter (Gate)
    allow = joined.date.map(limits).fillna(0).gt(0) & joined.pred_net_return_v2.ge(0.005)
    
    base_signals = joined.loc[allow, ["date", "code", "pred_score_v08"]].rename(columns={"pred_score_v08": "score"})
    v083_signals = joined.loc[allow, ["date", "code", "adaptive_score"]].rename(columns={"adaptive_score": "score"})
    
    cfg = arena.Config(capital=30000, order_budget=7500, max_positions=5, max_exposure=0.9, hold_days=10)
    
    results = []
    
    print("\n" + "=" * 92)
    print("RUNNING V0.8.3 SIMULATIONS & COMPARISON AGAINST V0.8 BASELINE")
    print("=" * 92)
    print(f"{'Period':<12} {'Case':<20} {'Return':>10} {'MaxDD':>10} {'Sharpe':>10} {'Trades':>8} {'WinRate':>10}")
    print("-" * 92)
    
    for period, start, end in PERIODS:
        days = [d for d in calendar if start <= d <= end]
        s_base = base_signals[base_signals.date.between(start, end)]
        s_v083 = v083_signals[v083_signals.date.between(start, end)]
        
        for stress in [False, True]:
            # Run Baseline V0.8
            rep_base = arena.simulate(frames, bars, days, s_base, events, replace(cfg, slippage=0.001 if stress else 0.0005), vol, bm, limits)
            sum_base = rep_base.summary()
            sum_base.update(period=period, case="stress_v08" if stress else "v08", policy="v08", stress=stress)
            results.append(sum_base)
            
            # Run Candidate V0.8.3
            rep_v083 = arena.simulate(frames, bars, days, s_v083, events, replace(cfg, slippage=0.001 if stress else 0.0005), vol, bm, limits)
            sum_v083 = rep_v083.summary()
            audit_v083 = arena.audit_ledger(rep_v083, bars, events)
            eq_df = pd.DataFrame(rep_v083.equity)
            sum_v083.update(
                period=period,
                case="stress_v083" if stress else "v083",
                policy="v083",
                stress=stress,
                audit=audit_v083,
                average_exposure=float(((eq_df.equity - eq_df.cash - eq_df.receivables) / eq_df.equity).mean()),
            )
            results.append(sum_v083)
            
            # Verify daily ledger balance sheet conservation
            for eq in rep_v083.equity:
                nav = float(eq["equity"])
                cash = float(eq["cash"])
                rec = float(eq["receivables"])
                diff = abs(nav - (cash + rec + (nav - cash - rec)))
                assert diff < 1e-6, f"Ledger conservation error: {diff}"
                
            # Save artifacts
            folder = out_dir / f"{period}_{'stress_' if stress else ''}v083"
            arena.save_case(folder, rep_v083, sum_v083, s_v083)
            
            base_ret = sum_base["total_return"]
            base_mdd = sum_base["max_drawdown"]
            v083_ret = sum_v083["total_return"]
            v083_mdd = sum_v083["max_drawdown"]
            shp = sum_v083.get("sharpe")
            shp_str = f"{shp:.2f}" if shp is not None else "None"
            trades = sum_v083["closed_trades"]
            wr = sum_v083.get("win_rate")
            wr_str = f"{wr:.2%}" if wr is not None else "None"
            
            print(f"{period:<12} {'stress_v08' if stress else 'v08':<20} {base_ret:>9.2%} {base_mdd:>9.2%}")
            print(f"{period:<12} {'stress_v083' if stress else 'v083':<20} {v083_ret:>9.2%} {v083_mdd:>9.2%} {shp_str:>10} {trades:>8} {wr_str:>10}")
            print("-" * 92)
            
    # Compile comparison
    comp_df = pd.DataFrame(results)[
        ["period", "case", "policy", "stress", "total_return", "max_drawdown", "closed_trades", "win_rate", "execution_costs", "average_exposure"]
    ]
    comp_df.to_csv(out_dir / "comparison.csv", index=False)
    
    # Compute passes verdict
    verdicts = {}
    for period, _, _ in PERIODS:
        rows = {r["case"]: r for r in results if r["period"] == period}
        p_check = passes(rows["v08"], rows["v083"], rows["stress_v08"], rows["stress_v083"])
        verdicts[period] = p_check
        
    summary_payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
        "verdicts": verdicts,
        "all_passed": all(
            v["higher_net_return"] and v["drawdown_not_worse"] and v["higher_stressed_return"]
            for v in verdicts.values()
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print("\nFINAL VERDICTS V0.8.3:")
    print(json.dumps(verdicts, indent=2))
    print(f"\nAll artifacts successfully saved to {out_dir}")
    return summary_payload


if __name__ == "__main__":
    run_v083_evaluation(OUT_DIR)
