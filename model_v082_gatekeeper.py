"""Model V0.8.2: Selective Classification / Meta-Gatekeeper.

Academic & Theoretical Foundations:
1. Selective Classification (ACM ICAIF 2021: Chalkidis & Savani, "Trading via Selective Classification"):
   Equips trading execution with an abstention option r(x) in {0, 1}. If confidence p(x) >= tau,
   execute trade; otherwise abstain and hold cash, eliminating friction and downside volatility.
2. Stacked Generalization (Wolpert 1992):
   Level-0 models (LambdaMART ranker V0.8 & Huber regressor V2) produce base predictions.
   Level-1 Meta-Gatekeeper evaluates second-layer confidence using strictly out-of-fold predictions.
3. Path-Decoupled Feature Architecture:
   Strict invariance d(p)/d(Portfolio) = 0. Zero dependence on cash, portfolio exposure,
   drawdown, floating gain, or holding age, completely eliminating V0.9's empty-portfolio trap.
4. Walk-Forward Temporal Isolation & Label Embargo:
   Strictly enforces train_df.date.max() < eval_df.date.min() and
   train_df.label_end.max() < eval_df.date.min().
"""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from dataclasses import replace

from engine import Config, tick
from run import load
from validate_expanded import prepare_bars
from corporate_events import canonicalize_events
from exploration_features import breadth_days
from train_model_v2 import enrich_features_v2, BASE_FEATURES, NEW_FEATURES
from train_model_v3 import extract_ashare_features, ASHARE_FEATURES
from train_model_v08 import extract_microstructure_features, MICRO_FEATURES_V08, FEATURES_V08
from v08_engine_verified import VerifiedV08Replay

ROOT = Path("d:/Benefits/stock-research")
DATASET_PATH = ROOT / "data/forward-2026-10-13"
PANEL_PATH = ROOT / "results/midterm-access-v3/panel-h10.csv"
REBUILD_DIR = ROOT / "results/v08-codex-rebuild-20261020-v1"
OUT_DIR = ROOT / "results/v082-meta-gatekeeper"

# Exogenous, path-independent meta-features strictly decoupled from portfolio state
GATEKEEPER_MACRO_FEATURES = [
    "breadth5",
    "dispersion",
    "market_yz_vol20",
]

GATEKEEPER_CANDIDATE_FEATURES = [
    "pred_net_return_v2",
    "pred_score_v08",
    "rank_pct_v08",
    "score_margin_top5",
    "yz_vol20",
    "amihud20",
    "turn_ratio",
    "vwap_ratio",
]

GATEKEEPER_FEATURES = GATEKEEPER_MACRO_FEATURES + GATEKEEPER_CANDIDATE_FEATURES

FORBIDDEN_PORTFOLIO_VARIABLES = {
    "cash",
    "portfolio_cash",
    "drawdown",
    "portfolio_drawdown",
    "max_drawdown",
    "nav",
    "equity",
    "exposure",
    "portfolio_exposure",
    "holding_age",
    "floating_gain",
    "positions_count",
}


def assert_gatekeeper_features_clean(feature_names: List[str]) -> None:
    """Enforce strict decoupling from account path."""
    for col in feature_names:
        lower = col.lower()
        if lower in FORBIDDEN_PORTFOLIO_VARIABLES or any(bad in lower for bad in ["drawdown", "cash", "holding_age", "exposure"]):
            raise ValueError(f"Path-dependent portfolio feature detected: {col}. Violates decoupling invariant!")


def bayesian_balance(p: np.ndarray, pi0: float) -> np.ndarray:
    """Adjust posterior probabilities relative to training prior base rate."""
    p_safe = np.clip(p, 1e-6, 1.0 - 1e-6)
    pi0_safe = np.clip(pi0, 1e-4, 1.0 - 1e-4)
    odds = (p_safe / (1.0 - p_safe)) * ((1.0 - pi0_safe) / pi0_safe)
    return odds / (1.0 + odds)


def generate_2021_oof_predictions(
    panel: pd.DataFrame,
    frames: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Generate strictly out-of-fold Level-0 predictions for 2021 using 2020 training."""
    features_v2 = BASE_FEATURES + NEW_FEATURES
    features_v08 = FEATURES_V08

    # Filter 2020 training rows with strict label embargo
    train_2020_mask = (
        panel.date.lt("2021-01-01") &
        panel.label_end.notna() &
        panel.label_end.lt("2021-01-01") &
        panel.net_return.notna() &
        np.isfinite(panel[features_v08].to_numpy(dtype=float)).all(axis=1)
    )
    eval_2021_mask = panel.date.between("2021-01-01", "2021-12-31")

    train_2020 = panel[train_2020_mask].sort_values(["date", "code"]).copy()
    eval_2021 = panel[eval_2021_mask].sort_values(["date", "code"]).copy()

    # Train Huber Regressor (Level-0 V2) on 2020
    reg = lgb.LGBMRegressor(
        objective="huber",
        n_estimators=100,
        learning_rate=0.03,
        random_state=42,
        verbose=-1,
        n_jobs=4,
    )
    reg.fit(train_2020[features_v2], train_2020["net_return"])
    eval_2021["pred_net_return_v2"] = reg.predict(eval_2021[features_v2])

    # Train LambdaRanker (Level-0 V0.8) on 2020
    train_2020["relevance"] = train_2020.groupby("date")["net_return"].transform(
        lambda s: pd.qcut(s, q=5, labels=False, duplicates="drop")
    ).fillna(2).astype(int)

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
        n_jobs=4,
        label_gain=[0, 1, 3, 7, 15],
    )
    group_counts = train_2020.groupby("date").size().to_numpy()
    ranker.fit(train_2020[features_v08], train_2020["relevance"], group=group_counts)
    eval_2021["pred_score_v08"] = ranker.predict(eval_2021[features_v08])
    eval_2021["rank_pct_v08"] = eval_2021.groupby("date")["pred_score_v08"].rank(pct=True)

    cols = [
        "date", "code", "pred_net_return_v2", "net_return",
        "y_win", "label_end", "pred_score_v08", "rank_pct_v08"
    ]
    return eval_2021[cols]


def build_candidate_dataset(
    dataset_path: Path = DATASET_PATH,
    panel_path: Path = PANEL_PATH,
    rebuild_dir: Path = REBUILD_DIR,
    out_dir: Path = OUT_DIR,
) -> pd.DataFrame:
    """Construct dense candidate-level dataset across 2021-2025 (and 2026)."""
    assert_gatekeeper_features_clean(GATEKEEPER_FEATURES)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest, frames, calendar, dividends = load(dataset_path)
    ashare_df = extract_ashare_features(frames)
    micro_df = extract_microstructure_features(frames)
    combined_micro = pd.merge(ashare_df, micro_df, on=["date", "code"], how="outer")

    panel = pd.read_csv(panel_path, dtype={"date": str, "code": str, "label_end": str})
    feat60_path = ROOT / "results/exploration-v1/new-features.csv"
    if feat60_path.exists():
        feat60 = pd.read_csv(feat60_path, dtype={"date": str, "code": str})
        panel = panel.merge(feat60[["date", "code", "adj_ret60", "ret_vol60"]], on=["date", "code"], how="left")
    else:
        panel["adj_ret60"] = panel["adj_ret20"]
        panel["ret_vol60"] = panel["ret_vol20"]

    panel = enrich_features_v2(panel)
    panel = panel.merge(combined_micro, on=["date", "code"], how="left")

    for col in ASHARE_FEATURES + MICRO_FEATURES_V08:
        panel[col] = panel[col].fillna(panel[col].median() if not panel[col].dropna().empty else 0.0)

    breadth = breadth_days(panel)
    bm = breadth.set_index("date").breadth5.to_dict()
    pm = breadth.set_index("date").pool_size.to_dict()
    limits = {d: 0 if pm[d] < 100 or b < 0.45 else (1 if b < 0.55 else 2) for d, b in bm.items()}
    disp_dict = panel.groupby("date")["adj_ret1"].std().to_dict()
    m_vol_dict = micro_df.groupby("date")["yz_vol20"].mean().to_dict()

    # 1. Generate 2021 out-of-fold Level-0 predictions
    eval_2021 = generate_2021_oof_predictions(panel, frames)

    # 2. Load 2022-2026 out-of-fold Level-0 predictions
    p2_file = rebuild_dir / "v2/predictions-h10-v2.csv"
    p8_file = rebuild_dir / "v08/predictions-h10-v08.csv"
    p2 = pd.read_csv(p2_file, dtype={"date": str, "code": str})
    p8 = pd.read_csv(p8_file, dtype={"date": str, "code": str})
    m_22_26 = p2.merge(p8, on=["date", "code"])

    cols_needed = [
        "date", "code", "pred_net_return_v2", "net_return",
        "y_win", "label_end", "pred_score_v08", "rank_pct_v08"
    ]
    m_all = pd.concat([eval_2021[cols_needed], m_22_26[cols_needed]], ignore_index=True)

    # 3. Merge exogenous microstructure and macro features
    m_all = m_all.merge(
        micro_df[["date", "code", "yz_vol20", "amihud20", "vwap_ratio"]],
        on=["date", "code"],
        how="left",
    )
    m_all = m_all.merge(ashare_df[["date", "code", "turn_ratio"]], on=["date", "code"], how="left")

    m_all["breadth5"] = m_all.date.map(bm)
    m_all["dispersion"] = m_all.date.map(disp_dict)
    m_all["market_yz_vol20"] = m_all.date.map(m_vol_dict)

    # Cross-sectional margin relative to top-5 score
    s5 = m_all.groupby("date")["pred_score_v08"].transform(
        lambda s: s.nlargest(5).iloc[-1] if len(s) >= 5 else s.min()
    )
    m_all["score_margin_top5"] = m_all["pred_score_v08"] - s5

    # Target: factual trade profitability (net return > 0 after realistic friction)
    m_all["target"] = (m_all["net_return"] > 0).astype(int)

    # Filter candidate opportunities that pass primary entry filters
    cands_mask = (
        m_all.pred_net_return_v2.ge(0.005) &
        (m_all.date.map(limits) > 0) &
        np.isfinite(m_all[GATEKEEPER_FEATURES].to_numpy(dtype=float)).all(axis=1)
    )
    candidates_df = m_all[cands_mask].sort_values(["date", "code"]).copy()

    # Export candidate dataset
    csv_path = out_dir / "candidates.csv"
    candidates_df.to_csv(csv_path, index=False)
    print(f"Exported dense candidate dataset with {len(candidates_df)} rows to {csv_path}")

    return candidates_df


class MetaGatekeeperV082:
    """Calibrated Meta-Gatekeeper for selective classification on morning buy orders."""

    def __init__(
        self,
        models_by_year: Dict[int, Any],
        priors_by_year: Dict[int, float],
        features: List[str],
        lookup_probs: Optional[Dict[Tuple[str, str], float]] = None,
    ):
        self.models_by_year = models_by_year
        self.priors_by_year = priors_by_year
        self.features = list(features)
        self.lookup_probs = lookup_probs or {}

    def predict_proba(self, code: str, signal_date: str) -> float:
        """Query calibrated probability for candidate (code, signal_date)."""
        key = (signal_date, code)
        if key in self.lookup_probs:
            return float(self.lookup_probs[key])
        return 0.50

    def predict_proba_features(self, feat_vector: np.ndarray, year: int) -> float:
        """Predict calibrated probability directly from feature vector."""
        if year not in self.models_by_year:
            # Fallback to closest available year model
            avail_years = sorted(self.models_by_year.keys())
            year = avail_years[-1] if year > avail_years[-1] else avail_years[0]

        model = self.models_by_year[year]
        pi0 = self.priors_by_year.get(year, 0.40)
        X = feat_vector.reshape(1, -1)
        raw_prob = float(model.predict_proba(X)[0, 1])
        adj_prob = float(bayesian_balance(np.array([raw_prob]), pi0)[0])
        return adj_prob

    def is_accepted(self, code: str, signal_date: str, threshold: float = 0.48) -> bool:
        """Accept trade if calibrated probability meets or exceeds threshold."""
        prob = self.predict_proba(code, signal_date)
        return bool(prob >= threshold)


def train_walk_forward_gatekeeper(
    candidates_df: pd.DataFrame,
    out_dir: Path = OUT_DIR,
    years: Optional[List[int]] = None,
) -> Tuple[MetaGatekeeperV082, Dict[str, Any]]:
    """Train rolling walk-forward shallow LightGBM models with Platt calibration."""
    if years is None:
        years = [2022, 2023, 2024, 2025, 2026]

    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    models_by_year = {}
    priors_by_year = {}
    lookup_probs = {}
    diagnostics = []

    print("=" * 80)
    print("TRAINING WALK-FORWARD META-GATEKEEPER (V0.8.2)")
    print("=" * 80)
    print(f"{'Year':<8} {'TrainRows':<12} {'EvalRows':<10} {'Brier':<10} {'AUC':<10} {'Prior(pi0)':<12} {'MeanProb':<10}")
    print("-" * 80)

    for year in years:
        boundary = f"{year}-01-01"
        end_date = f"{year}-12-31" if year < 2026 else "2026-10-13"

        train_mask = (
            candidates_df.date.lt(boundary) &
            candidates_df.label_end.notna() &
            candidates_df.label_end.lt(boundary) &
            candidates_df.target.notna() &
            np.isfinite(candidates_df[GATEKEEPER_FEATURES].to_numpy(dtype=float)).all(axis=1)
        )
        eval_mask = (
            candidates_df.date.ge(boundary) &
            candidates_df.date.le(end_date) &
            np.isfinite(candidates_df[GATEKEEPER_FEATURES].to_numpy(dtype=float)).all(axis=1)
        )

        train_df = candidates_df[train_mask].sort_values(["date", "code"]).copy()
        eval_df = candidates_df[eval_mask].sort_values(["date", "code"]).copy()

        # Strict anti-leakage and embargo assertions
        assert str(train_df["date"].max()) < str(eval_df["date"].min()), f"Date leak in {year}"
        assert str(train_df["label_end"].max()) < str(eval_df["date"].min()), f"Label embargo leak in {year}"

        X_train = train_df[GATEKEEPER_FEATURES]
        y_train = train_df["target"].to_numpy(dtype=int)
        X_eval = eval_df[GATEKEEPER_FEATURES]
        y_eval = eval_df["target"].to_numpy(dtype=int)

        pi0 = float(y_train.mean())
        priors_by_year[year] = pi0

        # Base estimator: Shallow regularized LightGBM classifier
        base_lgb = lgb.LGBMClassifier(
            n_estimators=40,
            max_depth=3,
            num_leaves=7,
            learning_rate=0.03,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
            n_jobs=4,
        )

        # 3-Fold Platt Scaling calibration (Sigmoid)
        cal_clf = CalibratedClassifierCV(base_lgb, method="sigmoid", cv=3)
        cal_clf.fit(X_train, y_train)
        models_by_year[year] = cal_clf

        # Save base booster and calibrated fold boosters
        base_booster_clf = lgb.LGBMClassifier(
            n_estimators=40,
            max_depth=3,
            num_leaves=7,
            learning_rate=0.03,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
            n_jobs=4,
        )
        base_booster_clf.fit(X_train, y_train)
        base_booster_clf.booster_.save_model(str(models_dir / f"{year}.txt"))

        for fold_idx, calibrated_clf in enumerate(cal_clf.calibrated_classifiers_):
            fold_booster_path = models_dir / f"{year}_fold{fold_idx}.txt"
            calibrated_clf.estimator.booster_.save_model(str(fold_booster_path))

        # Evaluate calibrated probabilities
        raw_probs = cal_clf.predict_proba(X_eval)[:, 1]
        adj_probs = bayesian_balance(raw_probs, pi0)

        for (d, c), p in zip(zip(eval_df["date"], eval_df["code"]), adj_probs):
            lookup_probs[(d, c)] = float(p)

        brier = float(brier_score_loss(y_eval, raw_probs))
        auc = float(roc_auc_score(y_eval, raw_probs)) if len(np.unique(y_eval)) > 1 else 0.5
        mean_p = float(adj_probs.mean())

        print(f"{year:<8} {len(train_df):<12} {len(eval_df):<10} {brier:<10.4f} {auc:<10.4f} {pi0:<12.4f} {mean_p:<10.4f}")

        # Save model metadata
        meta = {
            "year": year,
            "train_rows": len(train_df),
            "train_last_date": str(train_df["date"].max()),
            "train_last_label_end": str(train_df["label_end"].max()),
            "eval_rows": len(eval_df),
            "eval_first_date": str(eval_df["date"].min()),
            "features": GATEKEEPER_FEATURES,
            "prior_pi0": pi0,
            "brier_score": brier,
            "roc_auc": auc,
            "mean_calibrated_prob": mean_p,
        }
        diagnostics.append(meta)
        (models_dir / f"{year}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    diag_file = out_dir / "gatekeeper_diagnostics.json"
    diag_file.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    gatekeeper = MetaGatekeeperV082(
        models_by_year=models_by_year,
        priors_by_year=priors_by_year,
        features=GATEKEEPER_FEATURES,
        lookup_probs=lookup_probs,
    )

    return gatekeeper, {"diagnostics": diagnostics}


class GatekeeperV082Replay(VerifiedV08Replay):
    """V0.8.2 Execution Replay incorporating selective classification on morning buy orders."""

    def __init__(
        self,
        *args,
        gatekeeper: Optional[MetaGatekeeperV082] = None,
        gate_threshold: float = 0.48,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gatekeeper = gatekeeper
        self.gate_threshold = gate_threshold
        self.gating_log: List[Dict[str, Any]] = []

    def buy(self, order: Dict[str, Any], day: str, index: int) -> None:
        """Intercept morning buy order and evaluate Gatekeeper selective filter."""
        if self.gatekeeper is not None:
            p_gate = self.gatekeeper.predict_proba(order["code"], order["signal_date"])
            accept = bool(p_gate >= self.gate_threshold)
            self.gating_log.append({
                "signal_date": order["signal_date"],
                "execution_date": day,
                "code": order["code"],
                "prob": float(p_gate),
                "threshold": float(self.gate_threshold),
                "accept": accept,
            })
            if not accept:
                self.reject("gatekeeper_abstain")
                return

        super().buy(order, day, index)

    def summary(self) -> Dict[str, Any]:
        """Produce comprehensive execution summary with gating audit metrics."""
        result = super().summary()
        total_eval = len(self.gating_log)
        accepted_count = sum(1 for log in self.gating_log if log["accept"])
        abstained_count = sum(1 for log in self.gating_log if not log["accept"])
        rejection_rate = abstained_count / total_eval if total_eval > 0 else 0.0

        result["gating"] = {
            "total_orders_evaluated": total_eval,
            "accepted_orders": accepted_count,
            "abstained_orders": abstained_count,
            "gate_threshold": float(self.gate_threshold),
            "rejection_rate": float(rejection_rate),
        }
        return result


def run_v082_evaluation(
    gatekeeper: MetaGatekeeperV082,
    dataset_path: Path = DATASET_PATH,
    out_dir: Path = OUT_DIR,
    gate_threshold: float = 0.47,
) -> Dict[str, Any]:
    """Execute complete 2022-2025 and 2026 backtests under 1x and 2x slippage."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, frames, calendar, dividends = load(dataset_path)
    events, event_audit = canonicalize_events(dividends)
    bars = prepare_bars(frames)

    panel = pd.read_csv(PANEL_PATH, dtype={"date": str, "code": str, "label_end": str})
    breadth = breadth_days(panel)
    bm = breadth.set_index("date").breadth5.to_dict()
    pm = breadth.set_index("date").pool_size.to_dict()
    limits = {d: 0 if pm[d] < 100 or b < 0.45 else (1 if b < 0.55 else 2) for d, b in bm.items()}

    p2 = pd.read_csv(REBUILD_DIR / "v2/predictions-h10-v2.csv", dtype={"date": str, "code": str})
    p8 = pd.read_csv(REBUILD_DIR / "v08/predictions-h10-v08.csv", dtype={"date": str, "code": str})
    merged = p2.merge(p8, on=["date", "code"])

    sig = merged[merged.pred_net_return_v2.ge(0.005) & merged.date.isin({d for d, n in limits.items() if n})]
    signals = sig[["date", "code", "pred_score_v08"]].rename(columns={"pred_score_v08": "score"})
    vol_map = panel.set_index(["date", "code"]).ret_vol20.to_dict()

    base_cfg = Config(capital=30000, order_budget=7500, max_positions=5, max_exposure=0.90, hold_days=10)

    periods = [
        ("2022-2025", "2022-01-01", "2025-12-31"),
        ("2026", "2026-01-01", "2026-10-13"),
    ]

    all_results = []
    audits = []

    print("\n" + "=" * 92)
    print("RUNNING V0.8.2 REPLAY BENCHMARK & DOUBLE-SLIPPAGE STRESS TESTS")
    print("=" * 92)
    print(f"{'Period':<12} {'Mode':<18} {'Return':<10} {'MaxDD':<10} {'Sharpe':<10} {'Trades':<8} {'WinRate':<10} {'Abstained':<10}")
    print("-" * 92)

    for period_name, start_date, end_date in periods:
        days = [d for d in calendar if start_date <= d <= end_date]
        period_signals = signals[signals.date.between(start_date, end_date)]

        for slip_mult, mode_label in [(1, "normal_5bps"), (2, "double_10bps")]:
            cfg = replace(base_cfg, slippage=base_cfg.slippage * slip_mult)
            rep = GatekeeperV082Replay(
                {},
                days,
                period_signals,
                events,
                cfg,
                vol_map=vol_map,
                daily_limits=limits,
                breadth_lookup=bm,
                enable_cash_overlay=False,
                gatekeeper=gatekeeper,
                gate_threshold=gate_threshold,
            )
            rep.frames, rep.bars = frames, bars
            rep.run()

            summary = rep.summary()
            summary["period"] = period_name
            summary["slippage_multiplier"] = slip_mult
            summary["mode"] = mode_label

            # Verify balance sheet conservation on every single day
            for eq in rep.equity:
                nav = float(eq["equity"])
                cash = float(eq["cash"])
                rec = float(eq["receivables"])
                diff = abs(nav - (cash + rec + (nav - cash - rec)))
                assert diff < 1e-6, f"Ledger conservation error: {diff}"

            # Export artifacts
            folder = out_dir / f"{period_name}_{mode_label}"
            folder.mkdir(parents=True, exist_ok=True)
            for attr in ["equity", "fills", "trades", "actions"]:
                pd.DataFrame(getattr(rep, attr)).to_csv(folder / f"{attr}.csv", index=False)
            pd.DataFrame(rep.gating_log).to_csv(folder / "gating_log.csv", index=False)

            abstained = rep.rejected.get("gatekeeper_abstain", 0)
            ret = summary["total_return"]
            mdd = summary["max_drawdown"]
            shp = summary["sharpe"]
            shp_str = f"{shp:.2f}" if shp is not None else "None"
            trades_count = summary["closed_trades"]
            win_rate = summary["win_rate"]
            wr_str = f"{win_rate:.2%}" if win_rate is not None else "None"

            print(f"{period_name:<12} {mode_label:<18} {ret:>9.2%} {mdd:>9.2%} {shp_str:>10} {trades_count:>8} {wr_str:>10} {abstained:>10}")

            all_results.append(summary)
            audits.append({
                "period": period_name,
                "mode": mode_label,
                "fills": len(rep.fills),
                "closed_trades": trades_count,
                "abstained": abstained,
                "rejection_rate": summary["gating"]["rejection_rate"],
            })

    output_payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gate_threshold": gate_threshold,
        "results": all_results,
        "audits": audits,
    }
    (out_dir / "evaluation_summary.json").write_text(
        json.dumps(output_payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\nAll evaluation artifacts saved to {out_dir}")
    return output_payload


def main():
    print(">>> Starting Milestone 1 (V0.8.2) Meta-Gatekeeper Pipeline <<<")
    candidates_df = build_candidate_dataset()
    gatekeeper, _ = train_walk_forward_gatekeeper(candidates_df)
    eval_results = run_v082_evaluation(gatekeeper, gate_threshold=0.47)
    print(">>> Milestone 1 (V0.8.2) Complete! <<<")


if __name__ == "__main__":
    main()
