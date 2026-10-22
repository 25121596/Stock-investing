"""Feature drift monitoring and Rank IC time series analysis / 特征漂移监控与Rank IC时序分析.

This module provides Population Stability Index (PSI) distribution drift detection
and daily Spearman Rank IC / Rolling IC_IR evaluation for A-share quantitative models.
本模块提供基于群体稳定性指标(PSI)的特征分布漂移监控与基于Spearman Rank IC的时序衰减分析功能。
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.stats

from ml_features import FEATURES

ROOT = Path(__file__).resolve().parent


def population_stability_index(
    expected: Sequence[float] | np.ndarray | pd.Series,
    actual: Sequence[float] | np.ndarray | pd.Series,
    bins: int = 10,
) -> float:
    """Compute PSI between expected and actual feature distributions.

    PSI < 0.10: no significant shift
    PSI 0.10-0.25: moderate shift, monitor
    PSI > 0.25: significant shift, investigate/retrain

    计算基准分布(expected)与实际分布(actual)之间的群体稳定性指标(PSI)。
    - PSI < 0.10: 无显著漂移 (状态正常)
    - 0.10 <= PSI <= 0.25: 中度漂移 (需保持关注)
    - PSI > 0.25: 显著漂移 (需介入排查或重新训练模型)

    Parameters
    ----------
    expected : Sequence[float] | np.ndarray | pd.Series
        Baseline or training feature observations / 基准分布数据 (如训练集特征)
    actual : Sequence[float] | np.ndarray | pd.Series
        Current or live feature observations / 实际监控数据 (如实盘或前瞻特征)
    bins : int, default 10
        Number of quantile bins to partition expected data / 分箱数量 (默认10等分分位数)

    Returns
    -------
    float
        Computed Population Stability Index. Returns np.nan if either dataset is empty
        or lacks finite values. / 计算得到的PSI指标值。若数据为空或无有效数值则返回np.nan。
    """
    if bins < 1:
        raise ValueError("bins must be at least 1")

    exp_arr = np.asarray(expected, dtype=float).ravel()
    act_arr = np.asarray(actual, dtype=float).ravel()

    exp_clean = exp_arr[np.isfinite(exp_arr)]
    act_clean = act_arr[np.isfinite(act_arr)]

    if len(exp_clean) == 0 or len(act_clean) == 0:
        return float("nan")

    # Generate quantile bin edges based on expected distribution
    quantiles = np.linspace(0, 100, bins + 1)
    raw_edges = np.percentile(exp_clean, quantiles)
    unique_edges = np.unique(raw_edges)

    # Handle edge cases: constant or near-constant features
    if len(unique_edges) < 2:
        val = float(unique_edges[0])
        if np.allclose(act_clean, val):
            return 0.0
        # Expected is constant but actual deviates
        edges = np.array([-np.inf, val - 1e-6, val + 1e-6, np.inf])
    elif len(unique_edges) == 2:
        mid = float((unique_edges[0] + unique_edges[1]) / 2.0)
        edges = np.array([-np.inf, mid, np.inf])
    else:
        # Use internal unique quantile cuts bounded by [-inf, inf]
        # to ensure all values (including actual out-of-bounds) are captured
        edges = np.r_[-np.inf, unique_edges[1:-1], np.inf]

    exp_counts, _ = np.histogram(exp_clean, bins=edges)
    act_counts, _ = np.histogram(act_clean, bins=edges)

    exp_pct = exp_counts.astype(float) / len(exp_clean)
    act_pct = act_counts.astype(float) / len(act_clean)

    # Laplace / epsilon smoothing to avoid log(0) or division by zero
    eps = 1e-4
    exp_pct = np.clip(exp_pct, eps, None)
    act_pct = np.clip(act_pct, eps, None)
    exp_pct = exp_pct / np.sum(exp_pct)
    act_pct = act_pct / np.sum(act_pct)

    psi_val = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return max(0.0, psi_val)


def feature_drift_report(
    train_panel: pd.DataFrame,
    live_panel: pd.DataFrame,
    features: Sequence[str] | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Compute PSI for each feature between training and live data.

    Returns a DataFrame with columns: feature, psi, status (ok/warning/alert).
    Optionally saves to CSV.

    计算训练集与当前监控集各特征的PSI漂移报告。
    返回包含 feature, psi, status 列的 DataFrame。
    可选将报告以 utf-8-sig 编码持久化到 CSV 文件。

    Parameters
    ----------
    train_panel : pd.DataFrame
        Training or historical reference panel / 训练集或历史基准面板
    live_panel : pd.DataFrame
        Current or live monitoring panel / 当前或实盘监控面板
    features : Sequence[str] | None, optional
        List of feature column names. Defaults to FEATURES from ml_features.
        要评估的特征名列表，默认采用 ml_features 中的 FEATURES。
    output_path : str | Path | None, optional
        Destination CSV path. Saved with utf-8-sig encoding if provided.
        可选的输出文件路径，若指定则保存为utf-8-sig CSV。

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['feature', 'psi', 'status'].
    """
    if features is None:
        target_features = list(FEATURES)
    else:
        target_features = list(features)

    records: list[dict[str, object]] = []
    for feat in target_features:
        if feat not in train_panel.columns or feat not in live_panel.columns:
            records.append({
                "feature": feat,
                "psi": float("nan"),
                "status": "missing",
            })
            continue

        psi = population_stability_index(train_panel[feat], live_panel[feat])
        if math.isnan(psi):
            status = "insufficient_data"
        elif psi < 0.10:
            status = "ok"
        elif psi <= 0.25:
            status = "warning"
        else:
            status = "alert"

        records.append({
            "feature": feat,
            "psi": psi,
            "status": status,
        })

    report = pd.DataFrame(records, columns=["feature", "psi", "status"])

    if output_path is not None:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(dest, index=False, encoding="utf-8-sig")

    return report


def rank_ic_series(
    predictions: pd.DataFrame,
    actual_col: str = "net_return",
    pred_col: str = "pred_net_return",
    min_obs: int = 10,
) -> pd.Series:
    """Compute daily Spearman rank IC between predicted and realized returns.

    Returns a Series indexed by date.

    计算每日预测收益与实际收益之间的 Spearman 秩相关系数 (Rank IC)。
    返回以日期为索引的 pd.Series。

    Parameters
    ----------
    predictions : pd.DataFrame
        DataFrame containing date, actual returns, and predicted returns.
        包含日期、真实收益与预测收益的数据框。
    actual_col : str, default 'net_return'
        Column name for realized returns / 真实收益列名
    pred_col : str, default 'pred_net_return'
        Column name for model predictions / 预测得分或预测收益列名
    min_obs : int, default 10
        Minimum number of valid observations required per date / 单日最小有效样本数

    Returns
    -------
    pd.Series
        Daily Spearman rank IC indexed by date string.
        以日期字符串为索引的每日Rank IC序列。
    """
    if "date" in predictions.columns:
        date_series = predictions["date"].astype(str)
    elif getattr(predictions.index, "name", None) in ("date", "Date"):
        date_series = predictions.index.astype(str)
    else:
        raise ValueError("predictions DataFrame must contain a 'date' column or date index")

    if actual_col not in predictions.columns:
        raise KeyError(f"actual_col '{actual_col}' not found in predictions columns")
    if pred_col not in predictions.columns:
        raise KeyError(f"pred_col '{pred_col}' not found in predictions columns")

    threshold = max(2, int(min_obs))

    work = pd.DataFrame({
        "date": date_series,
        "pred": pd.to_numeric(predictions[pred_col], errors="coerce"),
        "actual": pd.to_numeric(predictions[actual_col], errors="coerce"),
    })
    work = work.dropna(subset=["pred", "actual"])

    daily_ic: dict[str, float] = {}
    for dt, group in work.groupby("date", sort=True):
        if len(group) < threshold:
            continue
        x = group["pred"].to_numpy(dtype=float)
        y = group["actual"].to_numpy(dtype=float)

        # Check for zero variance / constant values
        if np.all(x == x[0]) or np.all(y == y[0]):
            continue

        res = scipy.stats.spearmanr(x, y)
        stat = getattr(res, "statistic", None)
        if stat is None:
            stat = getattr(res, "correlation", None)
        if stat is None:
            stat = res[0]
        val = float(stat)

        if math.isfinite(val):
            daily_ic[str(dt)] = val

    result = pd.Series(daily_ic, dtype=float, name="rank_ic")
    result.index.name = "date"
    return result.sort_index()


def rolling_ic_summary(
    ic_series: pd.Series,
    window: int = 60,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Compute rolling mean IC, IC_IR (mean/std), and percentage of positive IC days.

    Returns a DataFrame with columns: date, rolling_ic, ic_ir, pct_positive.

    计算滚动平均 IC、IC_IR (均值/标准差) 以及正 IC 交易日占比。
    返回包含 date, rolling_ic, ic_ir, pct_positive 列的 DataFrame。

    Parameters
    ----------
    ic_series : pd.Series
        Daily Rank IC series indexed by date / 每日Rank IC序列
    window : int, default 60
        Rolling window length in sessions / 滚动窗口交易日数 (默认60日)
    min_periods : int | None, optional
        Minimum number of observations in window required. Defaults to min(window, 5).
        窗口内最小观测样本数，默认取 min(window, 5)。

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['date', 'rolling_ic', 'ic_ir', 'pct_positive'].
    """
    cols = ["date", "rolling_ic", "ic_ir", "pct_positive"]
    if ic_series.empty:
        return pd.DataFrame(columns=cols)

    if window < 1:
        raise ValueError("window must be a positive integer")

    clean_s = ic_series.dropna().sort_index().astype(float)
    if clean_s.empty:
        return pd.DataFrame(columns=cols)

    effective_min = min(window, 5) if min_periods is None else max(1, int(min_periods))

    rolling_mean = clean_s.rolling(window=window, min_periods=effective_min).mean()
    rolling_std = clean_s.rolling(window=window, min_periods=effective_min).std(ddof=1)

    # Protect against division by zero or extremely small standard deviations
    std_arr = rolling_std.to_numpy(dtype=float)
    mean_arr = rolling_mean.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ic_ir_arr = np.where(std_arr > 1e-12, mean_arr / std_arr, np.nan)

    is_pos = (clean_s > 0).astype(float)
    pct_pos = is_pos.rolling(window=window, min_periods=effective_min).mean()

    return pd.DataFrame({
        "date": clean_s.index.astype(str),
        "rolling_ic": mean_arr,
        "ic_ir": ic_ir_arr,
        "pct_positive": pct_pos.to_numpy(dtype=float),
    }, columns=cols)


def main() -> None:
    """Run drift monitoring on the latest predictions vs training data.

    运行特征漂移与预测IC监控：
    1. 读取历史基准特征集 (data/ml-panel-v1.csv)
    2. 读取最新前瞻预测或回测预测 (results/forward-2026-v1/predictions.csv 或 results/midterm-access-v3/predictions-h10.csv)
    3. 输出各特征的 PSI 漂移报告与状态 (ok / warning / alert)
    4. 计算并输出 Rank IC 时序与滚动 IC_IR 统计
    """
    parser = argparse.ArgumentParser(
        description="Feature drift monitoring and Rank IC evaluation / 特征漂移与Rank IC监控",
    )
    parser.add_argument(
        "--train-panel",
        type=Path,
        default=ROOT / "data" / "ml-panel-v1.csv",
        help="Path to training feature panel CSV (default: data/ml-panel-v1.csv)",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT / "results" / "forward-2026-v1" / "predictions.csv",
        help="Path to predictions CSV (default: results/forward-2026-v1/predictions.csv)",
    )
    parser.add_argument(
        "--live-panel",
        type=Path,
        default=None,
        help="Optional path to live feature panel CSV (e.g. results/midterm-access-v3/features.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to save drift report and rolling IC summary",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=60,
        help="Rolling window for IC summary (default: 60)",
    )
    parser.add_argument(
        "--min-obs",
        type=int,
        default=10,
        help="Minimum observations per day for Rank IC (default: 10)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("A-Share Feature Drift & Rank IC Monitor / 特征漂移与Rank IC监控系统")
    print("=" * 70)

    # 1. Load training panel
    train_path = args.train_panel.resolve()
    if not train_path.is_file():
        raise FileNotFoundError(f"Training panel not found: {train_path}")

    print(f"[1/4] Loading training panel: {train_path.name}")
    # Inspect columns first to selectively load relevant columns
    header_cols = pd.read_csv(train_path, nrows=0).columns.tolist()
    load_cols = [c for c in ["date", "code", *FEATURES, "net_return", "eligible"] if c in header_cols]
    train_df = pd.read_csv(train_path, usecols=load_cols, dtype={"date": str, "code": str})
    print(f"      Rows: {len(train_df):,}, Date range: {train_df['date'].min()} ~ {train_df['date'].max()}")

    # 2. Load predictions
    pred_path = args.predictions.resolve()
    if not pred_path.is_file():
        # Fallback to midterm-access-v3 predictions if forward-2026 is missing
        fallback = ROOT / "results" / "midterm-access-v3" / "predictions-h10.csv"
        if fallback.is_file():
            print(f"      Predictions at {pred_path.name} not found, falling back to {fallback.name}")
            pred_path = fallback
        else:
            raise FileNotFoundError(f"Predictions file not found at {pred_path} or {fallback}")

    print(f"[2/4] Loading predictions: {pred_path.name}")
    pred_df = pd.read_csv(pred_path, dtype={"date": str, "code": str})
    print(f"      Rows: {len(pred_df):,}, Date range: {pred_df['date'].min()} ~ {pred_df['date'].max()}")

    # 3. Determine live panel for PSI calculation
    if args.live_panel and args.live_panel.resolve().is_file():
        live_path = args.live_panel.resolve()
        print(f"[3/4] Loading live feature panel from: {live_path.name}")
        live_df = pd.read_csv(live_path, dtype={"date": str, "code": str})
    else:
        # Check if full features.csv exists in midterm-access-v3
        candidate_features = ROOT / "results" / "midterm-access-v3" / "features.csv"
        if candidate_features.is_file():
            print(f"[3/4] Using live feature panel from: {candidate_features.name} (filtered to >= 2026-01-01)")
            live_df = pd.read_csv(candidate_features, dtype={"date": str, "code": str})
            if "date" in live_df.columns:
                live_df = live_df[live_df["date"] >= "2026-01-01"].copy()
        else:
            print("[3/4] Evaluating features available directly in predictions file")
            live_df = pred_df

    # Features present in live data
    available_features = [f for f in FEATURES if f in live_df.columns and f in train_df.columns]
    if not available_features:
        # Fallback to any common numeric columns
        available_features = [c for c in FEATURES if c in train_df.columns]

    print(f"      Computing PSI on {len(available_features)} feature(s)...")
    drift_dest = (args.output_dir / "feature_drift_report.csv") if args.output_dir else None
    report_df = feature_drift_report(train_df, live_df, available_features, output_path=drift_dest)

    print("\n--- Feature Drift Report (Population Stability Index) ---")
    print(f"{'Feature':<20} {'PSI':>10}   {'Status':<8}")
    print("-" * 42)
    for _, row in report_df.iterrows():
        psi_str = f"{row['psi']:.4f}" if math.isfinite(row["psi"]) else "NaN"
        print(f"{row['feature']:<20} {psi_str:>10}   {row['status']:<8}")

    status_counts = report_df["status"].value_counts().to_dict()
    print("-" * 42)
    print(f"Summary: OK={status_counts.get('ok', 0)}, "
          f"WARNING={status_counts.get('warning', 0)}, "
          f"ALERT={status_counts.get('alert', 0)}")

    # 4. Rank IC Series and Rolling Summary
    print("\n[4/4] Evaluating Rank IC and Rolling IC_IR...")
    working_preds = pred_df.copy()
    if "net_return" not in working_preds.columns:
        # Try merging net_return from training panel or midterm panels
        if "net_return" in train_df.columns:
            merged = working_preds.merge(
                train_df[["date", "code", "net_return"]].dropna(subset=["net_return"]),
                on=["date", "code"],
                how="inner",
            )
            if not merged.empty:
                working_preds = merged
                print(f"      Merged {len(working_preds):,} labeled trades from training panel")

        if "net_return" not in working_preds.columns or working_preds["net_return"].dropna().empty:
            # Check midterm panel-h10.csv if available
            midterm_panel = ROOT / "results" / "midterm-access-v3" / "panel-h10.csv"
            if midterm_panel.is_file():
                print(f"      Merging realized labels from {midterm_panel.name}...")
                m_labels = pd.read_csv(
                    midterm_panel,
                    usecols=["date", "code", "net_return"],
                    dtype={"date": str, "code": str},
                ).dropna(subset=["net_return"])
                merged = pred_df.merge(m_labels, on=["date", "code"], how="inner")
                if not merged.empty:
                    working_preds = merged
                    print(f"      Successfully linked {len(working_preds):,} labeled observations")

    if "net_return" in working_preds.columns and not working_preds["net_return"].dropna().empty:
        ic_s = rank_ic_series(
            working_preds,
            actual_col="net_return",
            pred_col="pred_net_return",
            min_obs=args.min_obs,
        )

        if not ic_s.empty:
            mean_ic = float(ic_s.mean())
            std_ic = float(ic_s.std(ddof=1))
            ic_ir = mean_ic / std_ic if std_ic > 1e-12 else float("nan")
            pct_pos = float((ic_s > 0).mean())

            print(f"      Total trading sessions with valid IC: {len(ic_s)}")
            print(f"      Mean Rank IC : {mean_ic:+.4f}")
            print(f"      Rank IC Std  : {std_ic:.4f}")
            print(f"      IC IR        : {ic_ir:+.4f}")
            print(f"      Positive IC% : {pct_pos:.2%}")

            rolling_dest = (args.output_dir / "rolling_ic_summary.csv") if args.output_dir else None
            summary_df = rolling_ic_summary(ic_s, window=args.window)
            if rolling_dest is not None:
                rolling_dest.parent.mkdir(parents=True, exist_ok=True)
                summary_df.to_csv(rolling_dest, index=False, encoding="utf-8-sig")

            print(f"\n--- Rolling IC Summary (Window={args.window}) ---")
            print(summary_df.tail(10).to_string(index=False))
        else:
            print("      Notice: No trading sessions met the minimum observation threshold.")
    else:
        print("      Notice: Predictions do not contain realized 'net_return' labels.")
        print("      Rank IC evaluation requires closed trades with realized outcomes.")

    if args.output_dir:
        print(f"\nDrift and IC artifacts saved to: {args.output_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
