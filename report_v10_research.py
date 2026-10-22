"""Build a Chinese, read-only research report from a completed V10 run directory.

Usage: python report_v10_research.py --run results/v10-ranking-objectives-YYYYMMDD-vN
The script only writes research-report.md and report.html inside --run.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import re
from statistics import mean


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def number(value, default=None):
    try:
        x = float(value)
        return x if x == x and abs(x) != float("inf") else default
    except (TypeError, ValueError):
        return default


def pct(value):
    return "—" if value is None else f"{value:+.2%}"


def pct_abs(value):
    return "—" if value is None else f"{value:.2%}"


def fmt(value, digits=2):
    return "—" if value is None else f"{value:,.{digits}f}"


def case_summary(root: Path, period: str, case: str):
    path = root / f"{period}_{case}" / "summary.json"
    if not path.is_file():
        return None
    try:
        return read_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def annual_returns(root: Path, period: str, case: str, initial: float):
    path = root / f"{period}_{case}" / "equity.csv"
    if not path.is_file():
        return {}
    rows = read_csv(path)
    values = []
    for row in rows:
        date = str(row.get("date", ""))
        equity = number(row.get("equity"))
        if len(date) >= 10 and equity is not None:
            values.append((date, equity))
    out = {}
    years = sorted({d[:4] for d, _ in values})
    for year in years:
        current = [(d, e) for d, e in values if d.startswith(year)]
        if not current:
            continue
        prior = [e for d, e in values if d < f"{year}-01-01"]
        start = prior[-1] if prior else initial
        end = current[-1][1]
        out[year] = end / start - 1 if start else None
    return out


def trade_concentration(root: Path, period: str, case: str, summary: dict):
    path = root / f"{period}_{case}" / "trades.csv"
    if not path.is_file():
        return None
    trades = read_csv(path)
    gains = sorted((number(r.get("net_pnl"), 0.0) or 0.0 for r in trades), reverse=True)
    positive = [x for x in gains if x > 0]
    top3 = sum(positive[:3])
    initial = number(summary.get("initial_capital"), 0.0) or 0.0
    ending = number(summary.get("final_equity"), initial) or initial
    account_profit = ending - initial
    share = top3 / account_profit if account_profit > 0 else None
    adjusted_return = (account_profit - top3) / initial if initial else None
    return {"top3": top3, "share": share, "adjusted_return": adjusted_return,
            "trade_count": len(trades), "positive_count": len(positive)}


def is_random(name: str):
    return "random" in name.lower()


def html_from_markdown(md: str):
    """Small dependency-free renderer for this script's headings, tables and lists."""
    out = ["<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">",
           "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
           "<title>模型研究报告</title><style>",
           "body{font:16px/1.65 system-ui,'Microsoft YaHei',sans-serif;max-width:1180px;margin:36px auto;padding:0 22px;color:#202631}",
           "h1,h2,h3{line-height:1.3;color:#142b46}table{border-collapse:collapse;width:100%;margin:14px 0 24px;display:block;overflow-x:auto}th,td{border:1px solid #ccd4dd;padding:7px 10px;text-align:left;white-space:nowrap}th{background:#edf2f7}blockquote{border-left:4px solid #557da5;padding:4px 14px;color:#46566a}code{background:#f2f4f6;padding:2px 5px;border-radius:3px}.muted{color:#586575}</style></head><body>"]
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i]); i += 1
            parsed = [[c.strip() for c in row.strip().strip("|").split("|")] for row in table_lines]
            parsed = [r for r in parsed if not all(re.fullmatch(r":?-{3,}:?", c or "-") for c in r)]
            if parsed:
                out.append("<table>")
                for ri, row in enumerate(parsed):
                    tag = "th" if ri == 0 else "td"
                    out.append("<tr>" + "".join(f"<{tag}>{inline_html(c)}</{tag}>" for c in row) + "</tr>")
                out.append("</table>")
            continue
        m = re.match(r"^(#{1,3})\s+(.*)$", line)
        if m:
            level = len(m.group(1)); out.append(f"<h{level}>{inline_html(m.group(2))}</h{level}>")
        elif line.startswith("> "):
            out.append(f"<blockquote>{inline_html(line[2:])}</blockquote>")
        elif line.startswith("- "):
            out.append("<ul>")
            while i < len(lines) and lines[i].startswith("- "):
                out.append(f"<li>{inline_html(lines[i][2:])}</li>"); i += 1
            out.append("</ul>"); continue
        else:
            out.append(f"<p>{inline_html(line)}</p>")
        i += 1
    out.append("</body></html>")
    return "\n".join(out)


def inline_html(s: str):
    s = html.escape(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    return s


def build_report(root: Path):
    summary_path = root / "summary.json"
    comparison_path = root / "comparison.csv"
    if not summary_path.is_file() or not comparison_path.is_file():
        raise FileNotFoundError("--run 目录必须同时含 summary.json 和 comparison.csv")
    data = read_json(summary_path)
    comparisons = read_csv(comparison_path)
    results = data.get("results", [])
    cases = sorted({str(x.get("case", "")) for x in results
                    if x.get("case") and not x.get("stress") and not is_random(str(x.get("case")))})
    cases = [c for c in cases if c not in {"baseline", "cash"}]
    periods = sorted({str(x.get("period")) for x in results if x.get("period")})
    baseline_name = "baseline" if "baseline" in {str(x.get("case")) for x in results} else None
    candidates = [c for c in cases if c != baseline_name]
    by_key = {(str(r.get("period")), str(r.get("case"))): r for r in comparisons}
    periods = [p for p in ("2022-2025", "2026") if p in periods] + [p for p in periods if p not in {"2022-2025", "2026"}]
    display_path = root / 'display-version.json'
    display = read_json(display_path) if display_path.is_file() else {}
    title = display.get('display_name', '排序模型历史研究')
    lines = [f"# {title}", "", "当前已采用最新版：**V0.8**；V1.0保留给正式可用版本。", "", f"运行目录（历史编号，供复现）：`{root}`", "",
             "> 本报告仅整理已运行实验的账户回放结果。涉及的历史年份均已查看，不构成独立样本外验证，也不代表未来收益。", ""]

    lines += ["## 正常与双倍滑点结果", "",
              "收益为扣费后账户累计收益；最大回撤以负值表示。通过条件是候选在每个时期均优于基线收益、回撤不恶化，且双倍滑点收益仍高于基线。", "",
              "| 时期 | 版本 | 正常收益 | 正常最大回撤 | 双倍滑点收益 | 双倍滑点最大回撤 | 判定 |",
              "|---|---|---:|---:|---:|---:|---|"]
    verdicts = data.get("verdicts", {})
    verdict_by_case = {k: v for k, v in verdicts.items() if isinstance(v, dict)}
    display_cases = ([baseline_name] if baseline_name else []) + candidates
    for period in periods:
        for case in display_cases:
            normal = by_key.get((period, case), {})
            stress = by_key.get((period, f"stress_{case}"), {})
            if not normal:
                s = case_summary(root, period, case) or {}
                normal = {"total_return": s.get("total_return"), "max_drawdown": s.get("max_drawdown")}
            if not stress:
                s = case_summary(root, period, f"stress_{case}") or {}
                stress = {"total_return": s.get("total_return"), "max_drawdown": s.get("max_drawdown")}
            status = "基线" if case == baseline_name else "待综合判定"
            if case != baseline_name:
                v = verdict_by_case.get(case, {})
                check = next((x for x in v.get("checks", []) if x.get("period") == period), None)
                if check:
                    failed = [label for key, label in (("higher_net_return", "正常收益未高于基线"),
                                                        ("drawdown_not_worse", "回撤恶化"),
                                                        ("higher_stressed_return", "双倍滑点收益未高于基线"))
                              if not check.get(key)]
                    status = "通过本期条件" if not failed else "未通过：" + "、".join(failed)
            lines.append(f"| {period} | {case} | {pct(number(normal.get('total_return')))} | {pct_abs(number(normal.get('max_drawdown')))} | {pct(number(stress.get('total_return')))} | {pct_abs(number(stress.get('max_drawdown')))} | {status} |")
    lines.append("")

    lines += ["## 候选判定", ""]
    any_pass = False
    for case in candidates:
        v = verdict_by_case.get(case, {})
        checks = v.get("checks", [])
        passed = bool(v.get("passed"))
        any_pass = any_pass or passed
        failures = []
        for check in checks:
            failed = []
            if not check.get("higher_net_return"): failed.append("正常收益未高于基线")
            if not check.get("drawdown_not_worse"): failed.append("最大回撤恶化")
            if not check.get("higher_stressed_return"): failed.append("双倍滑点收益未高于基线")
            if failed: failures.append(f"{check.get('period')}: " + "、".join(failed))
        if passed:
            lines.append(f"- **{case}：达到本轮预设历史筛选条件。** 这不等于独立验证或未来实盘验证；仍需按原计划继续观察。")
        else:
            lines.append(f"- **{case}：未通过。** " + ("；".join(failures) if failures else "预设判定条件未全部满足。"))
    if not candidates:
        lines.append("本次汇总中未识别到可评估的新模型候选。")
    if any_pass:
        lines += ["", "本轮至少有一个候选满足已查看历史上的预设条件，但这不是独立验证，不应据此声称稳定盈利。"]
    else:
        lines += ["", "**所有候选均未通过；本轮未达到超过 V0.8 的最低升级要求。**"]
    lines.append("")

    lines += ["## 正常模型逐年收益", "", "按账户每日净值计算年度收益；每年起点取该年前一条净值，首年使用初始资金。", "",
              "| 年份 | 版本 | 年度收益 |", "|---:|---|---:|"]
    year_cases = ([baseline_name] if baseline_name else []) + candidates
    for case in year_cases:
        for p in periods:
            base_sum = case_summary(root, p, case) or {}
            initial = number(base_sum.get("initial_capital"), 30000.0) or 30000.0
            for year, ret in sorted(annual_returns(root, p, case, initial).items()):
                lines.append(f"| {year} | {case} | {pct(ret)} |")
    lines.append("")

    lines += ["## 前三笔盈利交易的静态贡献扣除", "",
              "按单笔净利润排序，展示前三笔盈利交易合计及其相对账户净利润的占比，并从最终账户利润中直接扣除，折算静态调整收益。**这是静态贡献分析，不是删除交易后重跑账户**；资金、后续订单和持仓路径不会重算。", "",
              "| 时期 | 版本 | 前三笔盈利合计 | 占账户净利润 | 静态扣除后收益 |", "|---|---|---:|---:|---:|"]
    for period in periods:
        for case in display_cases:
            if case == baseline_name or case in candidates:
                s = case_summary(root, period, case)
                if not s: continue
                concentration = trade_concentration(root, period, case, s)
                if not concentration: continue
                lines.append(f"| {period} | {case} | {fmt(concentration['top3'])} 元 | {pct_abs(concentration['share'])} | {pct(concentration['adjusted_return'])} |")
    lines.append("")

    lines += ["## 随机对照（描述性）", "",
              "仅将随机账户中存在完整 summary、账本审计通过且质量警告为零的样本计入有效数；结果只作描述性比较，不是显著性检验。", ""]
    random_controls = data.get("random_controls", [])
    for period in periods:
        for case in candidates:
            model = by_key.get((period, case), {})
            if not model:
                continue
            valid = []
            for row in random_controls:
                if str(row.get("period")) != period:
                    continue
                audit = row.get("audit", {})
                if not isinstance(audit, dict) or audit.get("passed") is not True or number(row.get("quality_warning_count"), 0) != 0:
                    continue
                ret = number(row.get("total_return"))
                if ret is not None: valid.append(ret)
            if valid:
                model_return = number(model.get("total_return"))
                beat = sum(1 for x in valid if model_return is not None and model_return > x)
                lines.append(f"- {period} / {case}：高于 {beat}/{len(valid)} 个有效随机样本（共 {len(valid)} 个有效；未通过核验或缺失的样本不计入）。")
            else:
                lines.append(f"- {period} / {case}：没有满足完整账本审计与零质量警告条件的随机样本，未计算比较。")
    if not candidates:
        lines.append("没有候选模型可与随机样本作描述性比较。")
    lines.append("")

    lines += ["## 研究边界与未来观察", "",
              "- 所有报告时期均属于已查看历史，不能充当独立盲测。若有候选满足筛选条件，也只代表本轮历史筛选通过。",
              "- 随机对照有效数按审计通过且质量警告为零的实际样本计算；失败或缺失样本不会冒充成功，也不以“19/20全通过”等说法概括。",
              "- 本报告不修改模型、门槛、执行规则或未来观察记录；当前 V0.8 未来观察未被新模型替换。",
              f"- 汇总中的 `adopted_into_forward_session` 为 `{data.get('adopted_into_forward_session', '未提供')}`。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="从已完成的排序实验目录生成中文 Markdown 与 HTML 研究报告")
    parser.add_argument("--run", required=True, type=Path, help="含 summary.json 与 comparison.csv 的实验目录")
    args = parser.parse_args()
    root = args.run.resolve()
    markdown = build_report(root)
    (root / "research-report.md").write_text(markdown, encoding="utf-8")
    (root / "report.html").write_text(html_from_markdown(markdown), encoding="utf-8")


if __name__ == "__main__":
    main()
