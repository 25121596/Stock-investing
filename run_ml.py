"""Finite, annually purged walk-forward experiment with execution baselines."""
from __future__ import annotations

import argparse
from dataclasses import asdict
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
from sklearn.metrics import brier_score_loss, mean_squared_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from engine import Config
from ml_features import FEATURES
from run import load
from validate_expanded import prepare_bars, save_ledger, simulate

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data/pilot-2020-01-02-2025-12-31-n300-s20201009"
YEARS = (2022, 2023, 2024, 2025)
SEEDS = tuple(range(2026101401, 2026101411))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def training_mask(panel, start):
    # A row is trainable only after the entire realized trade is observable.
    return (panel.date.lt(start) & panel.label_end.notna() & panel.label_end.lt(start)
            & panel.net_return.notna() & panel.y_win.notna()
            & np.isfinite(panel[FEATURES].to_numpy(dtype=float)).all(axis=1))


def rank_signals(rows, scores):
    result = rows[["date", "code"]].copy()
    result["score"] = np.asarray(scores, dtype=float)
    if not np.isfinite(result.score).all() or result.duplicated(["date", "code"]).any():
        raise ValueError("Invalid/nonunique signal scores")
    return result


def verify_labels(panel, frames, bars, calendar, dividends, count=200):
    candidates = panel[panel.net_return.notna()].sample(n=min(count, panel.net_return.notna().sum()), random_state=20261014)
    index = {day: number for number, day in enumerate(calendar)}
    errors = []
    for row in candidates.itertuples():
        start = index[row.date]
        days = calendar[start:min(len(calendar), start + 17)]
        signals = pd.DataFrame([{"date": row.date, "code": row.code, "score": 1.0}])
        replay = simulate({row.code: frames[row.code]}, bars, days, signals, dividends, Config(hold_days=3))
        if len(replay.trades) != 1:
            errors.append({"date": row.date, "code": row.code, "issue": "isolated_ledger_not_closed"})
            continue
        trade = replay.trades[0]
        if abs(trade["net_return"] - row.net_return) > 1e-10 or trade["exit_date"] != row.label_end or replay.warnings:
            errors.append({"date": row.date, "code": row.code, "label_net_return": row.net_return,
                           "ledger_net_return": trade["net_return"], "label_end": row.label_end,
                           "ledger_exit": trade["exit_date"], "warnings": replay.warnings})
    if errors:
        raise AssertionError(f"Training labels differ from account engine: {errors[:5]}")
    return {"sampled_labels": len(candidates), "matched": len(candidates), "errors": errors,
            "scope": "Fixed seeded sample of labeled rows; not a full proof of all source data/labels"}


def learn(panel, output):
    predictions, training = [], []
    models_folder = output / "models"
    models_folder.mkdir()
    for year in YEARS:
        start, end = f"{year}-01-01", f"{year}-12-31"
        train = panel.loc[training_mask(panel, start)]
        evaluation = panel[(panel.date >= start) & (panel.date <= end)].copy()
        finite = np.isfinite(evaluation[FEATURES].to_numpy(dtype=float)).all(axis=1)
        evaluation = evaluation.loc[finite].copy()
        if len(train) < 1000 or evaluation.empty or train.y_win.nunique() != 2:
            raise ValueError(f"Insufficient train/evaluation rows for {year}")
        assert train.date.max() < start and train.label_end.max() < start
        classifier = make_pipeline(StandardScaler(), LogisticRegression(
            C=.1, max_iter=300, random_state=20261014))
        regressor = lgb.LGBMRegressor(n_estimators=150, learning_rate=.03, num_leaves=7,
            max_depth=3, min_child_samples=300, reg_lambda=10, random_state=20261014,
            n_jobs=3, deterministic=True, force_col_wise=True, verbosity=-1)
        with threadpool_limits(limits=3):
            classifier.fit(train[FEATURES], train.y_win.astype(int))
            regressor.fit(train[FEATURES], train.net_return.clip(-.2, .2))
            evaluation["p_win"] = classifier.predict_proba(evaluation[FEATURES])[:, 1]
            evaluation["pred_net_return"] = regressor.predict(evaluation[FEATURES])
        joblib.dump(classifier, models_folder / f"{year}-logistic.joblib")
        regressor.booster_.save_model(str(models_folder / f"{year}-lgb.txt"))
        labeled = evaluation[evaluation.net_return.notna()]
        observed_win = labeled.y_win.astype(int)
        rank_ic = labeled.pred_net_return.corr(labeled.net_return, method="spearman")
        stats = {"year": year, "train_rows": len(train), "train_date_max": train.date.max(),
                 "train_label_end_max": train.label_end.max(), "train_win_rate": float(train.y_win.mean()),
                 "eval_rows": len(evaluation), "eval_labeled_rows": len(labeled),
                 "excluded_nonfinite_feature_rows": int((~finite).sum()),
                 "logistic_auc": float(roc_auc_score(observed_win, labeled.p_win)) if observed_win.nunique() == 2 else None,
                 "logistic_brier": float(brier_score_loss(observed_win, labeled.p_win)),
                 "lgb_mse": float(mean_squared_error(labeled.net_return, labeled.pred_net_return)),
                 "lgb_rank_ic": float(rank_ic) if np.isfinite(rank_ic) else None}
        training.append(stats)
        predictions.append(evaluation)
        pd.DataFrame({"feature": FEATURES, "logistic_weight": classifier[-1].coef_[0],
                      "lgb_gain": regressor.booster_.feature_importance(importance_type="gain")
                     }).to_csv(models_folder / f"{year}-features.csv", index=False, encoding="utf-8-sig")
        print(f"TRAIN {year}: train={len(train)}, predict={len(evaluation)}, "
              f"last_label={train.label_end.max()}, AUC={stats['logistic_auc']:.4f}", flush=True)
    return pd.concat(predictions, ignore_index=True).sort_values(["date", "code"]).reset_index(drop=True), training


def experiment_signals(predictions, legacy):
    cases = []
    def add(case, label, group, rows, scores):
        cases.append((case, label, group, rank_signals(rows, scores)))
    add("logistic_rank", "逻辑回归每日排名", "model", predictions, predictions.p_win)
    selective_lr = predictions[predictions.p_win >= .55]
    add("logistic_selective", "逻辑回归≥55%才选", "model", selective_lr, selective_lr.p_win)
    add("lgb_rank", "收益模型每日排名", "model", predictions, predictions.pred_net_return)
    selective_lgb = predictions[predictions.pred_net_return >= .005]
    add("lgb_selective", "收益模型预测净收益≥0.5%", "model", selective_lgb, selective_lgb.pred_net_return)
    for seed in SEEDS:
        add(f"random_seed{seed}", f"随机选股 {seed}", "random", predictions,
            np.random.default_rng(seed).random(len(predictions)))
    add("simple_reversal", "过去5日跌幅优先", "simple", predictions, -predictions.adj_ret5)
    add("simple_momentum", "过去20日涨幅优先", "simple", predictions, predictions.adj_ret20)
    for strategy, label in (("volume", "原放量上涨"), ("volume_trend", "原放量加趋势"), ("turtle", "原海龟突破")):
        rows = legacy[legacy.strategy == strategy].merge(predictions[["date", "code"]], on=["date", "code"], how="inner", validate="one_to_one")
        add(f"legacy_{strategy}", label, "legacy", rows, rows.score)
    for model, chosen in (("logistic_selective", selective_lr), ("lgb_selective", selective_lgb)):
        opportunity_rows = predictions[predictions.date.isin(chosen.date.unique())]
        for seed in SEEDS[:3]:
            add(f"matched_{model}_seed{seed}", f"{model}同日期随机 {seed}", "matched_random", opportunity_rows,
                np.random.default_rng(seed).random(len(opportunity_rows)))
    return cases


def replay_cases(cases, frames, bars, calendar, dividends, output):
    results = []
    config = Config(hold_days=3)
    days = [day for day in calendar if "2022-01-01" <= day <= "2025-12-31"]
    for case, label, group, signals in cases:
        replay = simulate(frames, bars, days, signals, dividends, config)
        equity = pd.DataFrame(replay.equity)
        summary = {**replay.summary(), "case": case, "label": label, "group": group,
                   "quality_warnings": replay.warnings, "signal_dates": signals.date.nunique(),
                   "buy_fills": sum(item["side"] == "buy" for item in replay.fills),
                   "invested_days": int((equity.positions > 0).sum()),
                   "avg_exposure": float(((equity.equity - equity.cash - equity.receivables) / equity.equity).mean()),
                   "periods": {}}
        save_ledger(output / case, replay)
        for year in YEARS:
            period_days = [day for day in days if day.startswith(str(year))]
            selected = signals[signals.date.str.startswith(str(year))]
            period_replay = simulate(frames, bars, period_days, selected, dividends, config)
            summary["periods"][str(year)] = period_replay.summary()
        results.append(summary)
        print(f"CASE {case}: net={summary['total_return']:.2%}, win="
              f"{summary['win_rate'] if summary['win_rate'] is not None else 'none'}, "
              f"trades={summary['closed_trades']}, warnings={summary['quality_warning_count']}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/ml-walkforward-v1")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Choose a new empty output folder; previous experiments are preserved")
    manifest, frames, calendar, dividends = load(DATASET)
    if len(frames) != 300 or manifest["end"] != "2025-12-31":
        raise ValueError("Expected the fixed complete 300-stock dataset")
    panel_path = ROOT / "data/ml-panel-v1.csv"
    panel_meta = json.loads(panel_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    checks = {"source_sha256": ROOT / "ml_features.py", "engine_sha256": ROOT / "engine.py",
              "manifest_sha256": DATASET / "manifest.json", "panel_sha256": panel_path}
    for key, path in checks.items():
        if panel_meta[key] != digest(path):
            raise ValueError(f"Stale or changed ML input: {key}")
    if panel_meta["config"] != asdict(Config(hold_days=3)) or panel_meta["features"] != FEATURES:
        raise ValueError("Feature cache does not match this fixed experiment")
    panel = pd.read_csv(panel_path, dtype={"date": str, "code": str, "label_end": str}).sort_values(["date", "code"]).reset_index(drop=True)
    if len(panel) != panel_meta["rows"] or panel.duplicated(["date", "code"]).any() or not panel.eligible.all():
        raise ValueError("Invalid feature panel")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "ml-plan-v1.md", output / "protocol.md")
    bars = prepare_bars(frames)
    label_audit = verify_labels(panel, frames, bars, calendar, dividends)
    write_json(output / "label_audit.json", label_audit)
    print(f"LABEL AUDIT {label_audit['matched']}/{label_audit['sampled_labels']} matched engine", flush=True)
    predictions, training = learn(panel, output)
    predictions[["date", "code", "p_win", "pred_net_return", "net_return", "y_win", "label_end"]].to_csv(
        output / "predictions.csv", index=False, encoding="utf-8-sig")
    legacy = pd.read_csv(DATASET / "signals.csv", dtype={"date": str, "code": str})
    results = replay_cases(experiment_signals(predictions, legacy), frames, bars, calendar, dividends, output)
    hashes = {name: digest(ROOT / name) for name in
              ("engine.py", "ml_features.py", "run_ml.py", "ml_report.py", "ml-plan-v1.md")}
    payload = {"version": "ml-walkforward-v1", "dataset": str(DATASET), "manifest": manifest,
               "panel_meta": panel_meta, "features": FEATURES, "training": training, "hashes": hashes,
               "versions": {"sklearn": sklearn.__version__, "lightgbm": lgb.__version__, "numpy": np.__version__, "pandas": pd.__version__},
               "time_scope": "2022-2025, annually refit on earlier finished labels; exploratory previously viewed years",
               "results": results}
    write_json(output / "summary.json", payload)
    from ml_report import make_report
    report_path = make_report(output, payload)
    (ROOT / "results/latest.txt").write_text(str(report_path), encoding="utf-8")
    print(f"REPORT {report_path}", flush=True)


if __name__ == "__main__":
    main()
