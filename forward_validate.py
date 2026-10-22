"""Frozen-model 2026 historical forward check; predictions are never refit here."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
from statistics import median

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from engine import Config
from engine_2026 import Replay2026
from ml_features import FEATURES, build_panel
from run import load
from run_ml import experiment_signals
from signals import generate, source_hashes
from validate_expanded import prepare_bars, save_ledger


ROOT = Path(__file__).resolve().parent
OLD_DATASET = ROOT / "data" / "pilot-2020-01-02-2025-12-31-n300-s20201009"
DATASET = ROOT / "data" / "forward-2026-10-13"
FREEZE_DIR = ROOT / "results" / "forward-2026-frozen"
FREEZE_PATH = FREEZE_DIR / "freeze.json"
PANEL = ROOT / "data" / "ml-panel-v1.csv"
PANEL_META_PATH = ROOT / "data" / "ml-panel-v1.meta.json"
OUTPUT_DEFAULT = ROOT / "results" / "forward-2026-v1"
END_DATE = "2026-10-13"
BOUNDARY = "2026-01-01"
EXPECTED_CASES = 27
SOURCE_COPIES = ("forward_validate.py", "engine_2026.py", "engine.py", "ml_features.py",
                 "run_ml.py", "forward-plan-v1.md")
RAW_OVERLAP_COLUMNS = ("open", "high", "low", "close", "preclose", "volume", "amount",
                       "tradestatus", "isST")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def check_frozen_inputs() -> tuple[dict, dict[str, str]]:
    if not FREEZE_PATH.is_file():
        raise FileNotFoundError(FREEZE_PATH)
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    freeze_hash = digest(FREEZE_PATH)
    recorded = freeze.get("hashes", {})
    paths = {
        "ml_features.py": ROOT / "ml_features.py",
        "engine.py": ROOT / "engine.py",
        "run_ml.py": ROOT / "run_ml.py",
        "freeze_2026.py": ROOT / "freeze_2026.py",
        "ml-plan-v1.md": ROOT / "ml-plan-v1.md",
        "manifest.json": OLD_DATASET / "manifest.json",
        "ml-panel-v1.csv": PANEL,
        "2026-logistic.joblib": FREEZE_DIR / "models" / "2026-logistic.joblib",
        "2026-lgb.txt": FREEZE_DIR / "models" / "2026-lgb.txt",
    }
    current = {name: digest(path) for name, path in paths.items()}
    for name, actual in current.items():
        if recorded.get(name) != actual:
            raise ValueError(f"Frozen input/model hash mismatch: {name}")
    if freeze.get("features") != FEATURES or freeze.get("feature_count") != len(FEATURES):
        raise ValueError("Frozen feature list does not match current FEATURES")
    if freeze.get("config") != Config(hold_days=3).__dict__:
        raise ValueError("Frozen execution Config does not match the forward protocol")
    if freeze.get("training_boundary_exclusive") != BOUNDARY or freeze.get("evaluation_data_loaded") is not False:
        raise ValueError("Freeze record has an unexpected boundary or evaluation-data state")
    if freeze.get("decision_thresholds") != {"logistic_p_win_min": 0.55,
                                               "lgb_pred_net_return_min": 0.005}:
        raise ValueError("Frozen thresholds do not match the forward protocol")
    return freeze, {**current, "freeze.json": freeze_hash}


def verify_dataset_freeze_link(manifest: dict, freeze_hash: str) -> None:
    freeze_record = manifest.get("model_freeze")
    recorded = freeze_record.get("sha256") if isinstance(freeze_record, dict) else None
    if recorded != freeze_hash:
        raise ValueError("Forward dataset manifest does not match the frozen freeze.json SHA256")


def check_overlap(old_frames: dict[str, pd.DataFrame], new_frames: dict[str, pd.DataFrame]) -> dict:
    if set(old_frames) != set(new_frames):
        raise ValueError("Forward dataset stock codes differ from the historical 300-stock universe")
    checked = 0
    overlap_rows = 0
    for code, old in old_frames.items():
        new = new_frames[code]
        old = old.sort_values("date").drop_duplicates("date", keep="last")
        new = new.sort_values("date").drop_duplicates("date", keep="last")
        old_end = old.iloc[-1]
        same_day = new.loc[new.date.eq(old_end.date)]
        if same_day.empty:
            raise ValueError(f"Forward data lost the old terminal row for {code}: {old_end.date}")
        current = same_day.iloc[-1]
        for column in RAW_OVERLAP_COLUMNS:
            a, b = float(old_end[column]), float(current[column])
            same_missing = math.isnan(a) and math.isnan(b)
            same_value = (math.isfinite(a) and math.isfinite(b)
                          and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6))
            if not (same_missing or same_value):
                raise ValueError(f"Raw overlap mismatch for {code} {old_end.date} {column}: {a} vs {b}")
        shared = old[["date", "adj_close"]].merge(new[["date", "adj_close"]], on="date",
                                                       suffixes=("_old", "_new"), validate="one_to_one")
        valid = (np.isfinite(shared.adj_close_old) & np.isfinite(shared.adj_close_new)
                 & shared.adj_close_old.gt(0) & shared.adj_close_new.gt(0))
        shared = shared.loc[valid]
        if shared.empty:
            raise ValueError(f"No usable adjusted-price overlap for {code}")
        factors = shared.adj_close_new / shared.adj_close_old
        factor = float(factors.median())
        if not np.isfinite(factor) or not np.allclose(factors.to_numpy(), factor, rtol=1e-5, atol=1e-8):
            raise ValueError(f"Historical adjustment factor is not a uniform scale for {code}")
        checked += 1
        overlap_rows += len(shared)
    return {"codes_checked": checked, "adjusted_overlap_rows": overlap_rows,
            "raw_fields": list(RAW_OVERLAP_COLUMNS),
            "adjusted_history_policy": "new/old adj_close factor must be uniform across each stock's overlap"}


def build_predictions(frames: dict[str, pd.DataFrame], calendar: list[str], dividends: list[dict],
                      freeze: dict, freeze_hash: str) -> tuple[pd.DataFrame, dict[str, str]]:
    model_paths = {"logistic": FREEZE_DIR / "models" / "2026-logistic.joblib",
                   "lgb": FREEZE_DIR / "models" / "2026-lgb.txt"}
    before = {key: digest(path) for key, path in model_paths.items()}
    config = Config(hold_days=3)
    panel = build_panel(frames, calendar, dividends, config)
    if "eligible" not in panel or not panel.eligible.fillna(False).all():
        raise ValueError("build_panel returned an ineligible prediction row")
    date_mask = panel.date.ge(BOUNDARY) & panel.date.le(END_DATE)
    finite = np.isfinite(panel[FEATURES].to_numpy(dtype=float)).all(axis=1)
    # Labels are deliberately not selected or consulted in the inference table.
    inference = panel.loc[date_mask.to_numpy() & finite,
                          ["date", "code", *FEATURES]].copy().sort_values(["date", "code"]).reset_index(drop=True)
    if inference.empty:
        raise ValueError("No 2026-01-01 through 2026-10-13 candidates with finite frozen features")

    classifier = joblib.load(model_paths["logistic"])
    booster = lgb.Booster(model_file=str(model_paths["lgb"]))
    if list(getattr(classifier, "feature_names_in_", [])) != FEATURES:
        raise ValueError("Frozen Logistic feature names do not match FEATURES")
    if list(booster.feature_name()) != FEATURES:
        raise ValueError("Frozen LightGBM feature names do not match FEATURES")
    predictions = pd.DataFrame({
        "date": inference.date.to_numpy(),
        "code": inference.code.to_numpy(),
        "p_win": classifier.predict_proba(inference[FEATURES])[:, 1],
        "pred_net_return": booster.predict(inference[FEATURES]),
        "adj_ret5": inference.adj_ret5.to_numpy(),
        "adj_ret20": inference.adj_ret20.to_numpy(),
    })
    if predictions.duplicated(["date", "code"]).any() or not np.isfinite(
            predictions[["p_win", "pred_net_return", "adj_ret5", "adj_ret20"]].to_numpy(dtype=float)).all():
        raise ValueError("Frozen model produced duplicate or non-finite predictions")
    after = {key: digest(path) for key, path in model_paths.items()}
    if before != after:
        raise AssertionError("Frozen model file hashes changed during prediction")
    return predictions, {"before": before, "after": after}


def generate_legacy(DATASET_PATH: Path, frames: dict[str, pd.DataFrame], manifest: dict,
                    end_date: str) -> pd.DataFrame:
    child = DATASET_PATH / ".legacy-2026"
    child.mkdir(parents=True, exist_ok=True)
    source_manifest_hash = digest(DATASET_PATH / "manifest.json")
    child_manifest = {"source_manifest_sha256": source_manifest_hash,
                      "scope": {"start": BOUNDARY, "end": end_date, "purpose": "legacy rule signals for frozen 2026 comparison"}}
    write_json(child / "manifest.json", child_manifest)
    return generate(child, frames, BOUNDARY, end_date)


def enrich_case(case: str, label: str, group: str, signals: pd.DataFrame,
                frames: dict[str, pd.DataFrame], bars: dict, calendar: list[str],
                dividends: list[dict], output: Path, config: Config) -> dict:
    replay = Replay2026({}, calendar, signals, dividends, config)
    replay.frames = frames
    replay.bars = {code: bars[code] for code in frames}
    replay.run()
    equity = pd.DataFrame(replay.equity)
    if equity.empty:
        avg_exposure = 0.0
    else:
        exposure = (equity.equity - equity.cash - equity.receivables) / equity.equity.replace(0, np.nan)
        avg_exposure = float(exposure.fillna(0).mean())
    save_ledger(output / case, replay)
    return {
        **replay.summary(), "case": case, "label": label, "group": group,
        "signal_dates": int(signals.date.nunique()) if len(signals) else 0,
        "buy_fills": int(sum(fill["side"] == "buy" for fill in replay.fills)),
        "avg_exposure": avg_exposure,
        "quality_warnings": replay.warnings,
    }


def add_stress_cases(cases: list, frames, bars, calendar, dividends, output: Path) -> list[dict]:
    results = []
    by_case = {case: (label, group, signals) for case, label, group, signals in cases}
    for name, label in (("logistic_selective", "逻辑回归选择性·双倍滑点"),
                        ("lgb_selective", "收益模型选择性·双倍滑点")):
        _base_label, _group, signals = by_case[name]
        config = Config(hold_days=3)
        config.slippage *= 2
        case = f"stress_{name}"
        result = enrich_case(case, label, "stress", signals, frames, bars, calendar, dividends, output, config)
        results.append(result)
    return results


def run_cases(cases: list, frames, calendar, dividends, output: Path) -> list[dict]:
    bars = prepare_bars(frames)
    results = []
    for case, label, group, signals in cases:
        result = enrich_case(case, label, group, signals, frames, bars, calendar, dividends, output,
                             Config(hold_days=3))
        results.append(result)
        print(f"CASE {case}: return={result['total_return']:.2%}, trades={result['closed_trades']}, "
              f"warnings={result['quality_warning_count']}", flush=True)
    results.extend(add_stress_cases(cases, frames, bars, calendar, dividends, output))
    if len(results) != EXPECTED_CASES:
        raise AssertionError(f"Expected {EXPECTED_CASES} cases, produced {len(results)}")
    return results


def comparison_csv(output: Path, results: list[dict]) -> None:
    columns = ["case", "group", "label", "total_return", "max_drawdown", "win_rate",
               "profit_factor", "closed_trades", "execution_costs", "avg_exposure",
               "quality_warning_count", "open_positions", "signal_dates", "buy_fills"]
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def _fmt(value, pct=False) -> str:
    try:
        number = float(value)
        if not math.isfinite(number):
            return "—"
        return f"{number:.2%}" if pct else f"{number:,.2f}"
    except (TypeError, ValueError):
        return "—"


def _metrics_table(results: list[dict]) -> str:
    rows = []
    for item in results:
        links = " · ".join(f'<a href="{html.escape(item["case"], quote=True)}/{name}.csv">{name}</a>'
                           for name in ("trades", "fills", "equity", "holdings", "warnings"))
        warning_count = int(item.get("quality_warning_count") or 0)
        cls = ' class="warn"' if warning_count else ""
        rows.append(f"<tr{cls}><td>{html.escape(item['group'])}</td><td>{html.escape(item['case'])}</td>"
                    f"<td>{html.escape(item['label'])}</td><td>{_fmt(item['total_return'], True)}</td>"
                    f"<td>{_fmt(item['max_drawdown'], True)}</td><td>{_fmt(item['win_rate'], True)}</td>"
                    f"<td>{_fmt(item['profit_factor'])}</td><td>{int(item['closed_trades'])}</td>"
                    f"<td>{_fmt(item['execution_costs'])}</td><td>{_fmt(item['avg_exposure'], True)}</td>"
                    f"<td>{int(item['quality_warning_count'])}</td><td>{int(item['open_positions'])}</td>"
                    f"<td>{links}</td></tr>")
    return "".join(rows)


def _random_distribution(results: list[dict], group: str, case_prefix: str | None,
                         group_label: str) -> str:
    selected = [item for item in results if item["group"] == group
                and (case_prefix is None or item["case"].startswith(case_prefix))]
    clean = [item for item in selected if not int(item.get("quality_warning_count") or 0)]
    rows = []
    for sample_name, sample in (("全部案例", selected), ("无警告子集", clean)):
        for key, metric, pct in (("total_return", "累计收益", True),
                                 ("max_drawdown", "最大回撤", True),
                                 ("win_rate", "净胜率", True),
                                 ("closed_trades", "平仓数", False)):
            values = [float(item[key]) for item in sample if item.get(key) is not None
                      and math.isfinite(float(item[key]))]
            stats = ((sum(values) / len(values), median(values), min(values), max(values))
                     if values else (None, None, None, None))
            cells = [group_label, sample_name, f"{len(sample)}/{len(selected)}",
                     f"{len(clean)}/{len(selected)}", metric, *(_fmt(value, pct) for value in stats)]
            rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    return "".join(rows)


def make_report(output: Path, results: list[dict], freeze: dict, hashes: dict) -> Path:
    model_rows = [item for item in results if item["group"] == "model"]
    highlighted = "".join(f"<li><strong>{html.escape(item['label'])}</strong>: "
                          f"净收益 {_fmt(item['total_return'], True)}，平仓 {item['closed_trades']} 笔，"
                          f"净胜率 {_fmt(item['win_rate'], True)}。</li>" for item in model_rows)
    doc = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>2026冻结模型历史回放</title><style>
body{{font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif;max-width:1500px;margin:28px auto;padding:0 18px;color:#202936;background:#f5f7f9}}
h1{{font-size:26px}}h2{{margin-top:30px}}.notice{{background:#fff5df;border-left:5px solid #bd7924;padding:14px;margin:16px 0}}
.wrap{{overflow:auto;background:white;border:1px solid #d9e0e7;border-radius:8px}}table{{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}}
th,td{{text-align:right;padding:7px 9px;border-bottom:1px solid #e5e9ed}}th{{background:#eaf0f5}}td:first-child,th:first-child,td:nth-child(2),th:nth-child(2),td:nth-child(3),th:nth-child(3){{text-align:left}}
.warn{{background:#fff0ed}}a{{color:#145da0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:12px}}
</style><body><h1>2026冻结模型历史回放</h1>
<p><strong>回放窗口：</strong>2026年首个交易日至2026-10-13 · 累计非年化收益 · 初始本金30,000元</p>
<p><a href="summary.json">summary.json</a> · <a href="comparison.csv">comparison.csv</a> · <a href="protocol.md">forward-plan-v1.md</a> · <a href="freeze.json">冻结记录</a> · <a href="dataset-manifest.json">2026数据清单</a> · <a href="historical-manifest.json">历史数据清单</a></p>
<div class="notice"><strong>解释边界：</strong>这是冻结后新载入年份的历史检验，不是2026年初已记录的事前实盘预测。结果不承诺60%胜率或未来盈利。期末未平仓按最后可得价格计入净值，可能仍有未实现损益。2022–2025模型结果（包括此前报告的+22.21%）属于探索性先验，不是独立验证。</div>
<h2>四个冻结模型政策</h2><ul>{highlighted}</ul>
<h2>全部27个案例</h2><p>随机十次、同日期随机六次、简单规则、旧规则、双倍滑点压力测试均完整保留。浅红色行表示存在质量警告。</p>
<div class="wrap"><table><thead><tr><th>类别</th><th>案例</th><th>说明</th><th>累计收益</th><th>最大回撤</th><th>净胜率</th><th>PF</th><th>平仓数</th><th>费用</th><th>平均仓位</th><th>警告</th><th>期末持仓</th><th>逐笔记录</th></tr></thead><tbody>{_metrics_table(results)}</tbody></table></div>
<h2>随机分布（逐项案例亦见上表）</h2>
<div class="wrap"><table><thead><tr><th>基准与日期匹配范围</th><th>样本</th><th>案例数</th><th>无警告数</th><th>指标</th><th>均值</th><th>中位数</th><th>最小</th><th>最大</th></tr></thead><tbody>{_random_distribution(results,'random',None,'随机排名（10个完整候选池）')}{_random_distribution(results,'matched_random','matched_logistic_selective_','Logistic选择信号日期匹配随机（3次）')}{_random_distribution(results,'matched_random','matched_lgb_selective_','LightGBM选择信号日期匹配随机（3次）')}</tbody></table></div>
<h2>冻结与数据哈希</h2><pre>{html.escape(json.dumps(hashes,ensure_ascii=False,indent=2))}</pre>
<p>逐笔账本位于各案例目录；模型和特征在2026预测前已冻结。报告不调参、不重训。</p></body></html>'''
    path = output / "forward-report.html"
    path.write_text(doc, encoding="utf-8")
    return path


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and not output.is_dir():
        parser.error(f"Output path is not a directory: {output}")
    if output.exists() and any(output.iterdir()):
        parser.error(f"Refusing to overwrite nonempty output directory: {output}")

    freeze, frozen_hashes = check_frozen_inputs()
    freeze_hash = frozen_hashes["freeze.json"]
    manifest, frames, full_calendar, dividends = load(DATASET)
    if len(frames) != 300 or manifest.get("end") != END_DATE:
        raise ValueError("Expected complete 300-stock forward dataset ending 2026-10-13")
    verify_dataset_freeze_link(manifest, freeze_hash)
    old_manifest = json.loads((OLD_DATASET / "manifest.json").read_text(encoding="utf-8"))
    _old_manifest_again, old_frames, _old_calendar, _old_dividends = load(OLD_DATASET)
    overlap = check_overlap(old_frames, frames)

    calendar = [day for day in full_calendar if BOUNDARY <= day <= END_DATE]
    if not calendar:
        raise ValueError("Forward dataset has no market dates within the frozen evaluation interval")
    model_hashes_before = {name: frozen_hashes[filename] for name, filename in
                           (("logistic", "2026-logistic.joblib"), ("lgb", "2026-lgb.txt"))}
    predictions, model_hashes = build_predictions(frames, full_calendar, dividends, freeze, freeze_hash)
    if model_hashes["before"] != model_hashes_before or model_hashes["after"] != model_hashes_before:
        raise AssertionError("Prediction changed or substituted a frozen model")

    # The prediction file intentionally contains no realized target fields.
    if {"net_return", "y_win", "label_end", "label_reason"}.intersection(predictions.columns):
        raise AssertionError("Future target leaked into predictions.csv")
    legacy = generate_legacy(DATASET, frames, manifest, END_DATE)
    cases = experiment_signals(predictions, legacy)
    expected_base = {"logistic_rank", "logistic_selective", "lgb_rank", "lgb_selective",
                     "simple_reversal", "simple_momentum", "legacy_volume", "legacy_volume_trend",
                     "legacy_turtle", *(f"random_seed{seed}" for seed in range(2026101401, 2026101411)),
                     *(f"matched_{model}_seed{seed}" for model in ("logistic_selective", "lgb_selective")
                       for seed in range(2026101401, 2026101404))}
    if len(cases) != 25 or {case for case, _, _, _ in cases} != expected_base:
        raise AssertionError("run_ml.experiment_signals returned an unexpected fixed case set")

    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(FREEZE_PATH, output / "freeze.json")
    shutil.copy2(ROOT / "forward-plan-v1.md", output / "protocol.md")
    shutil.copy2(DATASET / "manifest.json", output / "dataset-manifest.json")
    shutil.copy2(OLD_DATASET / "manifest.json", output / "historical-manifest.json")
    sources_dir = output / "sources"
    sources_dir.mkdir()
    source_hashes_for_output = {}
    for name in SOURCE_COPIES:
        shutil.copy2(ROOT / name, sources_dir / name)
        source_hashes_for_output[name] = digest(sources_dir / name)
    if source_hashes_for_output["forward_validate.py"] != digest(ROOT / "forward_validate.py"):
        raise AssertionError("Copied runner does not match the executed runner")

    predictions.to_csv(output / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": FEATURES}).to_csv(output / "feature-list.csv", index=False, encoding="utf-8-sig")
    # Legacy cache is scoped under a child directory with its own manifest and identity.
    legacy_meta_path = DATASET / ".legacy-2026" / "signals.meta.json"
    legacy_cache_meta = json.loads(legacy_meta_path.read_text(encoding="utf-8")) if legacy_meta_path.exists() else {}
    legacy_source_hashes = source_hashes()
    results = run_cases(cases, frames, calendar, dividends, output)

    hashes = {
        **frozen_hashes,
        "freeze.json": freeze_hash,
        "forward_dataset_manifest.json": digest(DATASET / "manifest.json"),
        "forward_plan": digest(ROOT / "forward-plan-v1.md"),
        "forward_validate.py": digest(ROOT / "forward_validate.py"),
        "engine_2026.py": digest(ROOT / "engine_2026.py"),
        "legacy_source_files": legacy_source_hashes,
        "legacy_cache_identity": legacy_cache_meta,
        "model_hashes_before_prediction": model_hashes["before"],
        "model_hashes_after_prediction": model_hashes["after"],
        "copied_sources": source_hashes_for_output,
        "overlap_audit": overlap,
    }
    payload = {
        "version": "forward-2026-v1",
        "evaluation_scope": {"start_inclusive": calendar[0], "end_inclusive": END_DATE,
                             "sessions": len(calendar), "interpretation": "post-freeze historical year test; not an as-of-2026-01-01 live forecast"},
        "freeze": freeze,
        "dataset_manifest": manifest,
        "historical_source_manifest": old_manifest,
        "features": FEATURES,
        "thresholds": freeze["decision_thresholds"],
        "config": Config(hold_days=3).__dict__,
        "hashes": hashes,
        "prediction_rows": int(len(predictions)),
        "prediction_date_min": str(predictions.date.min()),
        "prediction_date_max": str(predictions.date.max()),
        "prediction_columns": list(predictions.columns),
        "case_count": len(results),
        "results": results,
        "notes": [
            "Frozen models were loaded and used for prediction only; no model fitting or threshold changes occurred.",
            "Feature panel labels were discarded before prediction selection; candidate eligibility used only build_panel rows and finite FEATURES.",
            "Comparison year 2026 was loaded after the frozen model record was written; this is a historical holdout analysis, not a live forecast made at year start.",
            "All 27 cases are retained, including those with quality warnings and the two doubled-slippage stress cases.",
            "Terminal open positions are marked to the final available price and are not force-closed.",
            "The earlier 2022-2025 +22.21% outcome is exploratory prior evidence, not an independent validation result.",
        ],
    }
    write_json(output / "summary.json", payload)
    comparison_csv(output, results)
    report_path = make_report(output, results, freeze, hashes)
    (ROOT / "results" / "latest.txt").write_text(str(report_path), encoding="utf-8")
    print(f"FORWARD REPORT {report_path}", flush=True)
    return payload


if __name__ == "__main__":
    main()
