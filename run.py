"""Run preregistered rule comparisons and export a transparent research report."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import html
import json
from pathlib import Path

import pandas as pd

from engine import Config, Replay
from signals import STRATEGIES, LABELS, generate, source_hashes

ROOT = Path(__file__).resolve().parent


def load(folder):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["failures"]:
        raise ValueError("Incomplete historical sample: fix downloads before evaluating")
    frames, dividends = {}, []
    for item in manifest["inventory"]:
        code = item["code"]
        path = folder / f"{code}.csv"
        dpath = folder / f"{code}.dividends.csv"
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Data changed since download: {code}")
        if hashlib.sha256(dpath.read_bytes()).hexdigest() != item["dividends_sha256"]:
            raise ValueError(f"Corporate actions changed since download: {code}")
        frame = pd.read_csv(path, dtype={"date": str, "code": str}).sort_values("date").reset_index(drop=True)
        if frame.date.duplicated().any() or frame.code.nunique() != 1 or frame.code.iloc[0] != code:
            raise ValueError(f"Invalid dates/codes: {code}")
        if ((frame.high < frame.low) | (frame.close <= 0) | (frame.adj_close <= 0)).any():
            raise ValueError(f"Invalid OHLC: {code}")
        frames[code] = frame
        events = pd.read_csv(dpath, dtype=str).fillna("")
        dividends.extend(events.to_dict("records"))
    calendar = pd.read_csv(folder / "calendar.csv", dtype=str).calendar_date.tolist()
    return manifest, frames, calendar, dividends


def percent(value):
    return "—" if value is None else f"{value * 100:.2f}%"


def comparison_frame(results):
    rows = []
    for result in results:
        rows.append({"策略": LABELS[result["strategy"]], "持有日数": result["hold_days"],
                     "总收益": percent(result["total_return"]), "最大回撤": percent(result["max_drawdown"]),
                     "平仓笔数": result["closed_trades"], "扣费后胜率": percent(result["win_rate"]),
                     "盈利因子PF": "—" if result["profit_factor"] is None else f'{result["profit_factor"]:.2f}',
                     "费用含模拟滑点": f'{result["execution_costs"]:.2f}',
                     "期末持仓": result["open_positions"], "数据警告": result["quality_warning_count"]})
    return pd.DataFrame(rows)


def report(output, manifest, results, frames):
    title = "A股选股规则历史回放 · 首轮小样本"
    summary = comparison_frame(results)
    coverage = pd.DataFrame([{"股票": code, "首日": frame.date.iloc[0], "末日": frame.date.iloc[-1],
                              "日线条数": len(frame)} for code, frame in frames.items()])
    assumptions = [
        f"{manifest['start']} 至 {manifest['end']}；股票池是起始日历史主板列表的固定随机样本，种子 {manifest['seed']}，共 {len(frames)} 只；没有按当前存活股票筛选，也不包含后来上市股票。",
        "直接调用下载项目的放量上涨、海龟突破规则；另预先定义一个放量上涨＋MA20高于MA60且收盘不低于MA20的趋势过滤版本。源文件不修改。",
        "所有策略先要求至少250个有交易状态的历史日。指标输入剔除零成交停牌占位行，成交量统一为股且转成浮点数。原规则输入采用复权历史，并按信号日原价锚定；成交和账户市值使用不复权价格。",
        "收盘生成信号，次日开盘尝试买入；数量和价格上限提前按信号日价格确定。每天最多新买1只，最多3只，单笔预算5000元，最大计划持仓50%，模拟本金30000元。",
        "按起始买入后的第1、3、5个市场交易日开盘退出；收盘触发3%亏损退出条件，下一交易日开盘尝试退出。停牌和开盘涨跌停导致退出延后，损失并非封顶3%。",
        "统一使用当前成本情景：佣金万2.5、每次最低5元；双边过户费万分之0.1；卖出印花税万5；每边额外计入0.05%模拟滑点费用。并非历史税费政策逐日复刻。",
        "现金分红按登记日持股确认、除权日计入应收、支付日转现金；短线分红税统一按20%保守情景。送转股在上市日前锁定。特殊分红、配股、退市结算、分红税差异等不能完整复原时必须标记警告。",
        "开盘位于涨停价不买、位于跌停价不卖，采用保守近似。日线不能还原排队、部分成交及盘中先后顺序，不运行盘中止盈止损。",
        "期末持仓按最后可得价格计入净值，不强行平仓；胜率仅统计真正平仓的交易。缺失持仓行情会警告，不能当成零损失或认定研究已通过。",
        "按2020–2023研究、2024验证、2025留出评估分别列账。这里是预先定义规则的首轮比较，无机器学习训练、参数搜索或实盘下单。看过结果后修改策略，需要新的独立检验。",
        f"{len(frames)}只样本只能验证数据与账本、发现明显问题；不能证明全市场盈利能力、长期胜率超过60%，也不能把各交易当作相互独立。",
        "质量警告为0只表示已实现的检查项没有发现疑点，不表示原始数据、特殊公司行动或模拟成交已获全面验证。"
    ]
    body = f"<h1>{title}</h1><p>真实历史数据 · 仅研究回放 · 不连接券商</p>"
    body += "<h2>全部时期</h2>" + summary.to_html(index=False, escape=True)
    period_names = {"research": "研究段 2020–2023", "validation": "验证段 2024", "holdout": "留出段 2025"}
    for period, label in period_names.items():
        period_results = []
        for result in results:
            values = result["periods"][period]
            period_results.append({**values, "strategy": result["strategy"], "hold_days": result["hold_days"]})
        body += f"<h2>{label}（各段重新以3万元启动）</h2>" + comparison_frame(period_results).to_html(index=False)
    diagnostic = pd.DataFrame([{"策略": LABELS[result["strategy"]], "持有日数": result["hold_days"],
                               "原净收益": percent(result["total_return"]),
                               "原成交路径加回费用": percent(result["total_return"] + result["execution_costs"] / 30000)}
                              for result in results])
    body += "<h2>费用诊断</h2><p>把原成交路径已经支付的费用加回期末净值，仅用于区分费用拖累与价格表现；这不是重新运行的零费用策略，买入数量和成交路径保持原样。</p>" + diagnostic.to_html(index=False)
    body += "<h2>模拟条件与局限</h2><ul>" + "".join(f"<li>{html.escape(text)}</li>" for text in assumptions) + "</ul>"
    body += "<h2>数据覆盖</h2>" + coverage.to_html(index=False)
    body += "<h2>逐笔与每日记录</h2><ul>"
    for result in results:
        case = result["case"]
        body += f'<li>{html.escape(case)}：<a href="{case}/trades.csv">平仓交易</a> · <a href="{case}/fills.csv">买卖与费用</a> · <a href="{case}/equity.csv">每日净值</a> · <a href="{case}/holdings.csv">期末持仓</a> · <a href="{case}/warnings.csv">数据警告</a></li>'
    body += "</ul><p>配置、原规则源码校验值、数据清单与结果见同目录 summary.json。</p>"
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + title + '</title><style>body{font:16px/1.7 system-ui,sans-serif;max-width:1150px;margin:32px auto;padding:0 18px;color:#202020;background:#fafafa}h1{font-size:26px}h2{font-size:20px;margin-top:32px}table{border-collapse:collapse;width:100%;font-size:14px;display:block;overflow-x:auto}th,td{border-bottom:1px solid #ddd;padding:8px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}a{color:#164f99}li{margin:8px 0}</style>' + body + '</html>'
    (output / "report.html").write_text(document, encoding="utf-8")
    (output / "README.md").write_text("# " + title + "\n\n" + "\n".join("- " + text for text in assumptions) + "\n\n完整结果见 report.html 和 summary.json。\n", encoding="utf-8")
    summary.to_csv(output / "comparison.csv", index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "replay-v1")
    parser.add_argument("--capital", type=float, default=30000)
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    args = parser.parse_args()
    if args.capital != 30000:
        parser.error("First-pass report is preregistered for 30000; create a separate experiment for other capital")
    folder = args.dataset or ROOT / "data" / "pilot-2020-01-02-2025-12-31-n24-s20201009"
    manifest, frames, calendar, dividends = load(folder)
    if manifest["start"] != "2020-01-02" or manifest["end"] != "2025-12-31":
        parser.error("This experiment's date segments and historical trading rules are fixed to 2020-2025")
    signals = generate(folder, frames, manifest["start"], manifest["end"])
    output = args.output
    existing_summary = output / "summary.json"
    if existing_summary.exists():
        existing = json.loads(existing_summary.read_text(encoding="utf-8"))
        if (existing["simulator_hash"] != hashlib.sha256((ROOT / "engine.py").read_bytes()).hexdigest()
                or existing["signal_adapter_hash"] != hashlib.sha256((ROOT / "signals.py").read_bytes()).hexdigest()
                or existing["source_hashes"] != source_hashes()
                or existing["manifest"] != manifest):
            parser.error("Experiment inputs changed; choose a new --output to preserve previous results")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for strategy in args.strategies:
        selected = signals[signals.strategy == strategy]
        for hold in (1, 3, 5):
            case = f"{strategy}-{hold}d"
            config = Config(hold_days=hold)
            sim = Replay(frames, calendar, selected, dividends, config).run()
            summary = {**sim.summary(), "strategy": strategy, "hold_days": hold, "case": case}
            destination = output / case
            destination.mkdir(exist_ok=True)
            for key, rows in (("trades", sim.trades), ("fills", sim.fills), ("equity", sim.equity),
                              ("actions", sim.actions), ("warnings", sim.warnings), ("holdings", sim.holdings())):
                pd.DataFrame(rows).to_csv(destination / f"{key}.csv", index=False, encoding="utf-8-sig")
            periods = {}
            for name, start, end in (("research", "2020-01-01", "2023-12-31"),
                                      ("validation", "2024-01-01", "2024-12-31"),
                                      ("holdout", "2025-01-01", "2025-12-31")):
                days = [day for day in calendar if start <= day <= end]
                period_signals = selected[(selected.date >= start) & (selected.date <= end)]
                period_sim = Replay(frames, days, period_signals, dividends, replace(config)).run()
                periods[name] = period_sim.summary()
            summary["periods"] = periods
            results.append(summary)
            print(f"{case}: return={percent(summary['total_return'])}, trades={summary['closed_trades']}, warnings={summary['quality_warning_count']}", flush=True)
    payload = {"dataset": str(folder), "manifest": manifest, "source_hashes": source_hashes(),
               "simulator_hash": hashlib.sha256((ROOT / "engine.py").read_bytes()).hexdigest(),
               "signal_adapter_hash": hashlib.sha256((ROOT / "signals.py").read_bytes()).hexdigest(),
               "results": results}
    (output / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    report(output, manifest, results, frames)
    (ROOT / "results" / "latest.txt").write_text(str(output.resolve() / "report.html"), encoding="utf-8")
    print(f"REPORT {output / 'report.html'}", flush=True)


if __name__ == "__main__":
    main()
