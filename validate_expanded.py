"""Frozen-rule expansion, replication and disjoint-pool sensitivity checks."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random

import pandas as pd

from engine import Config, Replay
from run import comparison_frame, load, percent, report
from signals import LABELS, STRATEGIES, generate, source_hashes

ROOT = Path(__file__).resolve().parent
PERIODS = (("research", "2020-01-01", "2023-12-31"),
           ("validation", "2024-01-01", "2024-12-31"),
           ("holdout", "2025-01-01", "2025-12-31"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def assert_replication(expected, actual, location="summary"):
    if isinstance(expected, dict):
        for key, value in expected.items():
            if key not in actual:
                raise AssertionError(f"Missing replicated field: {location}.{key}")
            assert_replication(value, actual[key], f"{location}.{key}")
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            raise AssertionError(f"Replicated list length differs: {location}")
        for index, value in enumerate(expected):
            assert_replication(value, actual[index], f"{location}[{index}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or abs(expected - actual) > 1e-7:
            raise AssertionError(f"Replicated number differs: {location}: {expected} vs {actual}")
    elif expected != actual:
        raise AssertionError(f"Replicated value differs: {location}: {expected} vs {actual}")


def prepare_bars(frames):
    # Replay only reads these dictionaries. Reuse identical immutable inputs
    # between runs instead of rebuilding hundreds of thousands of row dicts.
    return {code: frame.set_index("date").to_dict("index") for code, frame in frames.items()}


def simulate(frames, bars, calendar, signals, dividends, config):
    replay = Replay({}, calendar, signals, dividends, config)
    replay.frames = frames
    replay.bars = {code: bars[code] for code in frames}
    return replay.run()


def save_ledger(folder, replay):
    folder.mkdir(parents=True, exist_ok=True)
    for key, rows in (("trades", replay.trades), ("fills", replay.fills), ("equity", replay.equity),
                      ("actions", replay.actions), ("warnings", replay.warnings), ("holdings", replay.holdings())):
        pd.DataFrame(rows).to_csv(folder / f"{key}.csv", index=False, encoding="utf-8-sig")


def run_cases(frames, bars, calendar, signals, dividends, label, folder=None, periods=False):
    selected_codes = set(frames)
    pool_signals = signals[signals.code.isin(selected_codes)]
    results = []
    for strategy in STRATEGIES:
        selected = pool_signals[pool_signals.strategy == strategy]
        for holding in (1, 3, 5):
            config = Config(hold_days=holding)
            case = f"{strategy}-{holding}d"
            replay = simulate(frames, bars, calendar, selected, dividends, config)
            summary = {**replay.summary(), "strategy": strategy, "hold_days": holding,
                       "case": case, "pool": label, "pool_size": len(frames),
                       "signal_candidates": len(selected), "signal_dates": selected.date.nunique(),
                       "quality_warnings": replay.warnings,
                       "buy_fills": sum(fill["side"] == "buy" for fill in replay.fills)}
            if folder:
                save_ledger(folder / case, replay)
            if periods:
                summary["periods"] = {}
                for name, start, end in PERIODS:
                    days = [day for day in calendar if start <= day <= end]
                    period_signals = selected[(selected.date >= start) & (selected.date <= end)]
                    period_replay = simulate(frames, bars, days, period_signals, dividends, replace(config))
                    summary["periods"][name] = period_replay.summary()
            results.append(summary)
            print(f"{label}/{case}: return={percent(summary['total_return'])}, "
                  f"win={percent(summary['win_rate'])}, trades={summary['closed_trades']}, "
                  f"warnings={summary['quality_warning_count']}", flush=True)
    return results


def data_audit(frames, calendar):
    records = []
    days = set(calendar)
    for code, frame in frames.items():
        observed = set(frame.date)
        gaps = sorted(day for day in days if frame.date.iloc[0] <= day <= frame.date.iloc[-1] and day not in observed)
        active_zero = frame[(frame.tradestatus == 1) & (frame.volume <= 0)
                            & (frame.date >= calendar[0]) & (frame.date <= calendar[-1])]
        adjustment_mismatches = 0
        for key in ("open", "high", "low"):
            expected = frame[key] * frame.adj_close / frame.close
            adjustment_mismatches += int((abs(frame[f"adj_{key}"] - expected) > .001).sum())
        records.append({"code": code, "first_date": frame.date.iloc[0], "last_date": frame.date.iloc[-1],
                        "rows": len(frame), "ends_before_period": frame.date.iloc[-1] < calendar[-1],
                        "missing_inside_coverage": len(gaps), "missing_dates": gaps,
                        "active_zero_volume_rows": len(active_zero),
                        "raw_adjusted_ohlc_mismatches": adjustment_mismatches})
    return {"stock_count": len(frames), "rows": sum(row["rows"] for row in records),
            "early_ending_stocks": sum(row["ends_before_period"] for row in records),
            "missing_inside_coverage": sum(row["missing_inside_coverage"] for row in records),
            "active_zero_volume_rows": sum(row["active_zero_volume_rows"] for row in records),
            "raw_adjusted_ohlc_mismatches": sum(row["raw_adjusted_ohlc_mismatches"] for row in records),
            "stocks": records}


def subgroup_table(results):
    rows = []
    for strategy in STRATEGIES:
        for holding in (1, 3, 5):
            items = [row for row in results if row["strategy"] == strategy and row["hold_days"] == holding]
            clean = [row for row in items if not row["quality_warning_count"]]
            returns = pd.Series([row["total_return"] for row in clean], dtype=float)
            wins = pd.Series([row["win_rate"] for row in clean if row["win_rate"] is not None], dtype=float)
            rows.append({"策略": LABELS[strategy], "持有日数": holding, "无警告子池": len(clean),
                         "无警告且有平仓子池": len(wins),
                         "盈利子池": sum(value > 0 for value in returns),
                         "收益中位数": percent(float(returns.median())) if len(returns) else "—",
                         "收益最小值": percent(float(returns.min())) if len(returns) else "—",
                         "收益最大值": percent(float(returns.max())) if len(returns) else "—",
                         "胜率中位数": percent(float(wins.median())) if len(wins) else "—",
                         "达到60%胜率子池": sum(value >= .60 for value in wins)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/pilot-2020-01-02-2025-12-31-n300-s20201009")
    parser.add_argument("--output", type=Path, default=ROOT / "results/expanded-300-v1")
    args = parser.parse_args()
    manifest, frames, calendar, dividends = load(args.dataset)
    expected_manifest = {"start": "2020-01-02", "end": "2025-12-31", "warmup": "2018-10-01",
                         "seed": 20201009, "requested_count": 300, "universe_as_of": "2020-01-02"}
    universe_codes = pd.read_csv(args.dataset / "universe.csv", dtype=str).code.tolist()
    if (len(frames) != 300 or any(manifest.get(key) != value for key, value in expected_manifest.items())
            or universe_codes != [item["code"] for item in manifest["inventory"]]):
        parser.error("This frozen expansion requires the 300-stock 2020-2025 dataset")
    source_check_path = args.dataset / "source_verification.json"
    source_check = json.loads(source_check_path.read_text(encoding="utf-8"))
    if source_check.get("status") != "passed" or digest(source_check_path) != manifest.get("source_verification_sha256"):
        raise ValueError("Source verification did not pass or its recorded hash changed")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new empty output directory to preserve completed experiments")
    signals = generate(args.dataset, frames, manifest["start"], manifest["end"])
    audit = data_audit(frames, calendar)
    if audit["raw_adjusted_ohlc_mismatches"]:
        raise ValueError("Raw/adjusted candle shapes differ: investigate network response or source data")
    bars = prepare_bars(frames)
    old_folder = ROOT / "data/pilot-2020-01-02-2025-12-31-n24-s20201009"
    old_manifest, old_frames, old_calendar, old_dividends = load(old_folder)
    old_signals = generate(old_folder, old_frames, old_manifest["start"], old_manifest["end"])
    old_results = json.loads((ROOT / "results/replay-v1/summary.json").read_text(encoding="utf-8"))["results"]
    old_by_case = {result["case"]: result for result in old_results}
    overlap = sorted(set(old_frames) & set(frames))
    if overlap != sorted(old_frames):
        raise ValueError("Expected all old 24 stocks to be nested in the larger sample")
    for code in overlap:
        for suffix in (".csv", ".dividends.csv"):
            if digest(args.dataset / f"{code}{suffix}") != digest(old_folder / f"{code}{suffix}"):
                raise ValueError(f"Overlapping raw data changed: {code}{suffix}")
    # Check both the old signal cache and shared-bar execution against v1.
    nested_signals = signals[signals.code.isin(overlap)].reset_index(drop=True)
    pd.testing.assert_frame_equal(nested_signals, old_signals.reset_index(drop=True), check_dtype=False)
    replicated = run_cases(old_frames, bars, old_calendar, old_signals, old_dividends, "replicate-24", periods=True)
    checks = []
    for result in replicated:
        previous = old_by_case[result["case"]]
        deltas = {key: result[key] - previous[key] for key in
                  ("final_equity", "total_return", "win_rate", "closed_trades", "execution_costs", "max_drawdown")}
        assert_replication(previous, result, result["case"])
        checks.append({"case": result["case"], "deltas": deltas, "matched": True,
                       "checked_fields": list(previous), "includes_all_period_fields": True})
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "replication.json", checks)
    write_json(args.output / "data_audit.json", audit)
    write_json(args.output / "source_verification.json", source_check)
    results = run_cases(frames, bars, calendar, signals, dividends, "full-300", args.output, periods=True)
    new_frames = {code: frame for code, frame in frames.items() if code not in old_frames}
    new_results = run_cases(new_frames, bars, calendar, signals, dividends, "new-276", args.output / "new-276")
    write_json(args.output / "new_stock_results.json", new_results)
    shuffled = sorted(frames)
    random.Random(4821).shuffle(shuffled)
    subgroup_results, memberships = [], []
    for index in range(10):
        codes = shuffled[index * 30:(index + 1) * 30]
        name = f"pool-{index + 1:02d}"
        memberships.extend({"pool": name, "code": code} for code in codes)
        subset = {code: frames[code] for code in sorted(codes)}
        subgroup_results.extend(run_cases(subset, bars, calendar, signals, dividends, name))
    write_json(args.output / "subpool_results.json", subgroup_results)
    warning_details = [{"pool": row["pool"], "case": row["case"], **warning}
                       for row in subgroup_results for warning in row["quality_warnings"]]
    write_json(args.output / "subpool_warning_details.json", warning_details)
    pd.DataFrame(memberships).to_csv(args.output / "subpool_members.csv", index=False, encoding="utf-8-sig")
    subgroups = subgroup_table(subgroup_results)
    subgroups.to_csv(args.output / "subpool_comparison.csv", index=False, encoding="utf-8-sig")
    # Metadata explicitly covers this runner and its fixed protocol as well.
    metadata = {"dataset": str(args.dataset), "manifest": manifest, "source_hashes": source_hashes(),
                "simulator_hash": digest(ROOT / "engine.py"), "signal_adapter_hash": digest(ROOT / "signals.py"),
                "validation_runner_hash": digest(__file__), "protocol_hash": digest(ROOT / "expanded-plan.md"),
                "run_module_hash": digest(ROOT / "run.py"), "calendar_hash": digest(args.dataset / "calendar.csv"),
                "universe_hash": digest(args.dataset / "universe.csv"), "config_base": asdict(Config()),
                "source_verification_hash": digest(source_check_path),
                "subpool_seed": 4821, "old_stock_overlap": len(overlap), "replication_matched": True,
                "audit": {key: value for key, value in audit.items() if key != "stocks"}, "results": results}
    write_json(args.output / "summary.json", metadata)
    report(args.output, manifest, results, frames)
    report_path = args.output / "report.html"
    document = report_path.read_text(encoding="utf-8").replace("首轮小样本", "300只扩样验证")
    document = document.replace("预先定义规则的首轮比较", "冻结上一轮规则的扩样比较")
    overview = ("<h2>这轮验证了什么</h2><p>规则、持有期和交易成本全部沿用上一轮；扩大到300只，"
                "包含旧24只和新增276只。旧24只的9组结果已完整复现，重叠原始文件和信号一致。"
                "时间仍是2020–2025，2025已经看过，因此这是股票范围扩展，不是新的未见时间测试。"
                "没有训练或调参。股票池扩大后，同日候选增多，成交额排序和仓位上限会改变实际成交股票。</p>")
    positive_full = sum(row["total_return"] > 0 for row in results)
    positive_new = sum(row["total_return"] > 0 for row in new_results)
    overview += (f"<p><strong>六年累计收益，非年化：300只池9组中{positive_full}组净盈利；"
                 f"新增276只池9组中{positive_new}组净盈利。</strong>胜率按扣费后已平仓交易计算。</p>")
    overview += (f"<p>共{audit['rows']:,}条日线（含暖身）；{audit['early_ending_stocks']}只在期末前结束行情，仍保留；"
                 f"覆盖区间内部缺失{audit['missing_inside_coverage']}行，标记可交易但零成交{audit['active_zero_volume_rows']}行。"
                 f"原价与复权价的当日K线比例不一致{audit['raw_adjusted_ohlc_mismatches']}项。"
                 "持仓期间缺行情和公司行动疑点见各组警告。带警告的回测不能作为已确认收益。</p>")
    identical_refresh = sum(item["raw_byte_identical"] and item["dividends_byte_identical"]
                            for item in source_check["refresh"])
    overview += (f"<p>接口超时附近4只股票已重新下载，{identical_refresh}只的行情及分红文件前后逐字节一致；"
                 f"所有{len(source_check['early_end'])}只提前结束行情的股票另查询了证券基本资料，详见来源核查记录。</p>")
    document = document.replace("<h2>全部时期</h2>", overview + "<h2>全部时期（300只）</h2>", 1)
    additions = "<h2>新增276只股票，去掉旧24只</h2>" + comparison_frame(new_results).to_html(index=False)
    additions += ("<h2>10个固定的30只子池</h2><p>固定种子4821，300只随机分为10个互不重叠子池，"
                  "每池运行全部9组。以下统计仅使用无质量警告子池，不能把有警告的池默认为亏损或盈利；"
                  "完整90格和成员见附表。它用于检查池子敏感性，不是90次独立实盘检验。</p>"
                  + subgroups.to_html(index=False))
    crowded = pd.DataFrame([{"策略": LABELS[result["strategy"]], "持有日数": result["hold_days"],
                              "信号候选数": result["signal_candidates"], "有信号日期数": result["signal_dates"],
                              "实际买入次数": result["buy_fills"],
                              "拒单原因": json.dumps(result["rejected"], ensure_ascii=False)} for result in results])
    additions += ("<h2>信号与实际成交</h2><p>每日只安排一个候选订单；其他未排到的候选不计拒单。"
                  "以下拒单只统计实际安排的订单尝试和延后卖出的尝试，同一持仓可能反复尝试。</p>"
                  + crowded.to_html(index=False))
    additions += ("<h2>完整审计记录</h2><ul>"
                  '<li><a href="summary.json">全部300只结果与配置</a></li>'
                  '<li><a href="new_stock_results.json">新增276只结果</a></li>'
                  '<li><a href="subpool_results.json">全部90格子池结果</a> · <a href="subpool_members.csv">子池成员</a></li>'
                  '<li><a href="subpool_warning_details.json">子池质量警告详情</a></li>'
                  '<li><a href="replication.json">旧24只复现核对</a> · <a href="data_audit.json">行情覆盖核查</a> · '
                  '<a href="source_verification.json">接口复拉与证券资料</a></li></ul>')
    document = document.replace("<h2>模拟条件与局限</h2>", additions + "<h2>模拟条件与局限</h2>", 1)
    report_path.write_text(document, encoding="utf-8")
    (args.output / "README.md").write_text(
        "# 300只扩样验证\n\n冻结原3种规则和1/3/5日持有，不训练或调参。完整报告见 report.html。\n\n"
        "包含300只主实验、新增276只实验、10个30只子池、旧24只复现和数据审计。\n\n"
        "时间段仍为2020–2025，不能视为全新未见时间样本。质量警告与局限详见报告。\n", encoding="utf-8")
    (ROOT / "results/latest.txt").write_text(str(report_path.resolve()), encoding="utf-8")
    print(f"REPORT {report_path.resolve()}", flush=True)
    print(f"PRIMARY profitable: {sum(row['total_return'] > 0 for row in results)}/9; "
          f"NEW profitable: {sum(row['total_return'] > 0 for row in new_results)}/9", flush=True)


if __name__ == "__main__":
    main()
