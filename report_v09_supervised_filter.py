"""Build a human-readable Chinese report for the V0.9 supervised-filter study.

The report intentionally reads every result from the run's summary and companion
artifacts. It does not promote a research candidate or rewrite the V0.8 record.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any


PERIODS = [
    ("2022_train", "2022训练期"),
    ("2023", "2023"),
    ("2024", "2024"),
    ("2025", "2025"),
    ("2023-2025", "2023—2025连续期"),
    ("2026", "2026截至10月13日"),
]
POLICIES = ["baseline", "fixed_breadth", "supervised", "rl_vote"]
POLICY_NAMES = {
    "baseline": "V0.8基线",
    "fixed_breadth": "固定广度过滤",
    "supervised": "监督过滤",
    "rl_vote": "RL三种子投票",
}
ARTIFACTS = [
    "summary.json", "training.json", "training-labels.csv", "models/logit.json",
    "model-freeze.json", "comparison.csv", "protocol.md",
    "independent-export-audit.json", "supervised-diagnostics.json",
]
CASE_FILES = [
    ("summary.json", "账户摘要"), ("equity.csv", "每日净值"),
    ("fills.csv", "成交与费用"), ("trades.csv", "平仓记录"),
    ("gate-decisions.csv", "原始过滤决策"),
]


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def finite(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def fmt_num(value: Any, digits: int = 2) -> str:
    n = finite(value)
    return "—" if n is None else f"{n:,.{digits}f}"


def fmt_pct(value: Any, digits: int = 2) -> str:
    n = finite(value)
    return "—" if n is None else f"{100 * n:.{digits}f}%"


def fmt_money(value: Any, digits: int = 0) -> str:
    n = finite(value)
    return "—" if n is None else f"¥{n:,.{digits}f}"


def truth(value: Any) -> bool:
    return value is True or (isinstance(value, str) and value.lower() == "true")


def yesno(value: Any) -> str:
    return "是" if value is True else "否" if value is False else "缺失"


def metric(result: dict[str, Any] | None, name: str, default: Any = None) -> Any:
    if not result:
        return default
    if name in result:
        return result[name]
    return (result.get("summary") or {}).get(name, default)


def coverage(result: dict[str, Any] | None, name: str) -> Any:
    if not result:
        return None
    return (result.get("coverage") or {}).get(name)


def is_stress(result: dict[str, Any]) -> bool:
    case = str(result.get("case", ""))
    return case.startswith("stress_")


def get_result(results: list[dict[str, Any]], period: str, policy: str,
               stress: bool = False, family: str = "primary",
               seed: Any = None) -> dict[str, Any] | None:
    for result in results:
        result_policy = result.get("policy")
        result_family = result.get("family", "primary")
        if (result.get("period") == period and result_policy == policy
                and result_family == family and is_stress(result) == stress):
            if family != "seed" or result.get("seed") == seed:
                return result
    return None


def table(headers: list[str], rows: list[list[str]], cls: str = "") -> str:
    head = "<thead><tr>" + "".join(f"<th>{esc(x)}</th>" for x in headers) + "</tr></thead>"
    body = "<tbody>" + "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    ) + "</tbody>"
    return f"<div class='scroll'><table class='{esc(cls)}'>{head}{body}</table></div>"


def result_cell(result: dict[str, Any] | None) -> str:
    if not result:
        return "<span class='missing'>结果缺失</span>"
    return (
        f"收益 {esc(fmt_pct(metric(result, 'total_return')))}<br>"
        f"回撤 {esc(fmt_pct(metric(result, 'max_drawdown')))}<br>"
        f"暴露 {esc(fmt_pct(metric(result, 'average_exposure')))}<br>"
        f"平仓 {esc(fmt_num(metric(result, 'closed_trades'), 0))}<br>"
        f"接受率 {esc(fmt_pct(coverage(result, 'acceptance_rate')))}"
    )


def primary_tables(results: list[dict[str, Any]]) -> tuple[str, str, list[list[str]], list[list[str]]]:
    focus_rows: list[list[str]] = []
    md_focus: list[list[str]] = []
    for period in ("2026", "2023-2025"):
        label = dict(PERIODS).get(period, period)
        for metric_name, title in [
            ("total_return", "净收益"), ("max_drawdown", "最大回撤"),
            ("average_exposure", "平均暴露"), ("closed_trades", "平仓数"),
            ("acceptance_rate", "批次接受率"),
        ]:
            row = [esc(label), esc(title)]
            md_row = [label, title]
            for policy in POLICIES:
                r = get_result(results, period, policy)
                value = (fmt_pct(coverage(r, metric_name)) if metric_name == "acceptance_rate"
                         else fmt_pct(metric(r, metric_name)) if metric_name != "closed_trades"
                         else fmt_num(metric(r, metric_name), 0))
                row.append(esc(value) if r else "<span class='missing'>结果缺失</span>")
                md_row.append(value if r else "结果缺失")
            focus_rows.append(row)
            md_focus.append(md_row)
    focus = table(["时期", "指标"] + [POLICY_NAMES[p] for p in POLICIES], focus_rows, "focus")

    all_rows: list[list[str]] = []
    md_all: list[list[str]] = []
    for period, label in PERIODS:
        for stress in (False, True):
            scenario = "双倍滑点" if stress else "正常成本"
            row = [esc(label), esc(scenario)]
            md_row = [label, scenario]
            for policy in POLICIES:
                r = get_result(results, period, policy, stress)
                value = (f"收益 {fmt_pct(metric(r, 'total_return'))}；回撤 {fmt_pct(metric(r, 'max_drawdown'))}；"
                         f"暴露 {fmt_pct(metric(r, 'average_exposure'))}；平仓 {fmt_num(metric(r, 'closed_trades'), 0)}；"
                         f"接受率 {fmt_pct(coverage(r, 'acceptance_rate'))}；未见状态比例 {fmt_pct(coverage(r, 'novel_state_rate'))}") if r else "结果缺失"
                row.append(esc(value) if r else "<span class='missing'>结果缺失</span>")
                md_row.append(value)
            all_rows.append(row)
            md_all.append(md_row)
    all_table = table(["时期", "情景"] + [POLICY_NAMES[p] for p in POLICIES], all_rows, "wide")
    return focus, all_table, md_focus, md_all


def training_summary(run: Path, summary: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
    training = summary.get("training")
    training = training if isinstance(training, dict) else {}
    training_file = run / "training.json"
    if not training and training_file.is_file():
        try:
            candidate = json.loads(training_file.read_text(encoding="utf-8-sig"))
            if isinstance(candidate, dict):
                training = candidate
        except (OSError, json.JSONDecodeError):
            pass
    rows = training.get("rows")
    opportunities = training.get("total_opportunities")
    status_counts = training.get("status_counts") or {}
    class_counts = training.get("class_counts") or {}
    state_count = training.get("unique_states")
    last_label_end = training.get("last_label_end")
    model_file = training.get("model_file")
    fit_available = training.get("fit_available")
    details = [
        ("进入拟合的成熟批次", fmt_num(rows, 0)),
        ("有待买单批次（含no_execution）", fmt_num(opportunities, 0)),
        ("不同状态数", fmt_num(state_count, 0)),
        ("最后标签结束日期", str(last_label_end) if last_label_end not in (None, "") else "缺失"),
        ("拟合状态", yesno(fit_available)),
        ("模型文件记录", str(model_file) if model_file else "缺失"),
    ]
    status_rows = [[esc(str(k)), esc(fmt_num(v, 0))] for k, v in sorted(status_counts.items())]
    class_rows = [[esc(str(k)), esc(fmt_num(v, 0))] for k, v in sorted(class_counts.items(), key=lambda kv: str(kv[0]))]
    status_html = table(["标签状态", "批次数"], status_rows) if status_rows else "<p class='missing'>标签状态计数缺失。</p>"
    class_html = table(["类别标签", "批次数"], class_rows) if class_rows else "<p class='missing'>类别计数缺失。</p>"
    return training, table(["训练概况", "记录"], [[esc(k), esc(v)] for k, v in details]), status_html + class_html


def seed_table(results: list[dict[str, Any]]) -> tuple[str, str, int]:
    seeds = [r for r in results if r.get("family") == "seed"]
    seeds.sort(key=lambda r: (str(r.get("seed")), str(r.get("period")), is_stress(r), str(r.get("policy"))))
    html_rows: list[list[str]] = []
    md_lines = ["| 种子 | 时期 | 情景 | 策略 | 收益 | 最大回撤 | 期末净值 | 平仓 | 胜率 | 平均暴露 | 费用 | 接受率 | 未见状态比例 | 接受批次 | 跳过批次 | 买入成交单 | 新状态批次 | 质量警告 |",
                "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in seeds:
        cov = r.get("coverage") or {}
        values = [
            str(r.get("seed", "缺失")), str(r.get("period", "缺失")),
            "双倍滑点" if is_stress(r) else "正常成本",
            POLICY_NAMES.get(r.get("policy"), str(r.get("policy", "缺失"))),
            fmt_pct(metric(r, "total_return")), fmt_pct(metric(r, "max_drawdown")),
            fmt_money(metric(r, "final_equity")), fmt_num(metric(r, "closed_trades"), 0),
            fmt_pct(metric(r, "win_rate")), fmt_pct(metric(r, "average_exposure")),
            fmt_money(metric(r, "execution_costs")), fmt_pct(cov.get("acceptance_rate")),
            fmt_pct(cov.get("novel_state_rate")),
            fmt_num(cov.get("accepted_batches"), 0), fmt_num(cov.get("skipped_batches"), 0),
            fmt_num(cov.get("filled_buy_orders"), 0), fmt_num(cov.get("novel_state_batches"), 0),
            fmt_num(metric(r, "quality_warning_count"), 0),
        ]
        html_rows.append([esc(x) for x in values])
        md_lines.append("| " + " | ".join(values) + " |")
    return (table(["种子", "时期", "情景", "策略", "收益", "最大回撤", "期末净值", "平仓", "胜率", "平均暴露", "费用", "接受率", "未见状态比例", "接受批次", "跳过批次", "买入成交单", "新状态批次", "质量警告"], html_rows, "seed"),
            "\n".join(md_lines), len(seeds))


def artifact_links(run: Path) -> tuple[str, list[str]]:
    html_items, md_items = [], []
    for relative in ARTIFACTS:
        path = run / relative
        if path.is_file():
            html_items.append(f"<li><a href='{esc(relative)}'>{esc(relative)}</a></li>")
            md_items.append(f"- [{relative}]({relative})")
        else:
            html_items.append(f"<li>{esc(relative)}：文件缺失</li>")
            md_items.append(f"- {relative}：文件缺失")
    return "<ul>" + "".join(html_items) + "</ul>", md_items


def case_links(run: Path, results: list[dict[str, Any]]) -> tuple[str, list[str]]:
    cases = sorted({(str(r.get("period", "缺失")), str(r.get("case", "缺失"))) for r in results})
    html_items, md_items = [], []
    for period, case in cases:
        folder = f"{period}_{case}"
        found_html, found_md = [], []
        for filename, label in CASE_FILES:
            target = run / folder / filename
            rel = f"{folder}/{filename}"
            if target.is_file():
                found_html.append(f"<a href='{esc(rel)}'>{esc(label)}</a>")
                found_md.append(f"[{label}]({rel})")
            else:
                found_html.append(f"{esc(label)}：缺失")
                found_md.append(f"{label}：缺失")
        html_items.append(f"<li><strong>{esc(folder)}</strong>：" + " · ".join(found_html) + "</li>")
        md_items.append(f"- **{folder}**：" + " · ".join(found_md))
    return "<ul>" + "".join(html_items) + "</ul>" if html_items else "<p>没有账户结果，无法生成分账户链接。</p>", md_items


def records_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error, UnicodeError):
        return []


def opportunity_score_range(run: Path, results: list[dict[str, Any]]) -> tuple[str, int]:
    scores: list[float] = []
    for result in results:
        if result.get("policy") != "supervised":
            continue
        folder = f"{result.get('period', '缺失')}_{result.get('case', '缺失')}"
        decision_path = run / folder / "gate-decisions.csv"
        for row in read_csv_rows(decision_path):
            pending = finite(row.get("pending_before"))
            score = finite(row.get("score"))
            if pending is not None and pending > 0 and score is not None:
                scores.append(score)
    if not scores:
        return "机会批次分数缺失", 0
    return (f"{min(scores):.4f}—{max(scores):.4f}（累计{len(scores)}条账户级机会记录，"
            "含重叠时期和两种滑点账户，不代表独立样本数）"), len(scores)


def fixed_2026_trade_summary(run: Path, results: list[dict[str, Any]]) -> tuple[int | None, int | None, float | None]:
    result = get_result(results, "2026", "fixed_breadth")
    if result is None:
        return None, None, None
    trade_path = run / "2026_fixed_breadth" / "trades.csv"
    rows = read_csv_rows(trade_path)
    if rows:
        count = len(rows)
        wins = sum(1 for row in rows if (finite(row.get("net_pnl")) or 0.0) > 0)
        return count, wins, (wins / count if count else None)
    count = finite(metric(result, "closed_trades"))
    rate = finite(metric(result, "win_rate"))
    if count is None:
        return None, None, None
    count_int = int(round(count))
    if rate is None or count_int <= 0:
        return count_int, None, rate
    wins = int(round(rate * count_int))
    return count_int, wins, wins / count_int


def read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {"read_error": "文件存在但无法解析"}


def model_threshold(run: Path, summary: dict[str, Any]) -> float | None:
    training = summary.get("training") or {}
    model_rel = training.get("model_file") or "models/logit.json"
    model = read_json(run / str(model_rel))
    return finite(model.get("threshold")) if isinstance(model, dict) else None


def in_sample_accuracy(diagnostics: Any) -> tuple[int | None, int | None, float | None]:
    if not isinstance(diagnostics, dict):
        return None, None, None
    matrix = diagnostics.get("training_only_confusion") or []
    total = sum(int(finite(row.get("count")) or 0) for row in matrix if isinstance(row, dict))
    correct = sum(int(finite(row.get("count")) or 0) for row in matrix
                  if isinstance(row, dict) and row.get("actual") == row.get("predicted"))
    if not total:
        return None, None, None
    return correct, total, correct / total


def direct_conclusions(run: Path, summary: dict[str, Any], results: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    verdicts = summary.get("verdicts") or {}
    lines: list[str] = []
    diagnostics = read_json(run / "supervised-diagnostics.json")
    threshold = model_threshold(run, summary)

    fixed_verdict = verdicts.get("fixed_breadth") or {}
    if "passed" in fixed_verdict:
        lines.append("固定广度：" + ("通过本轮已查看历史的预设验收标准。" if truth(fixed_verdict.get("passed"))
                                      else "未通过本轮已查看历史的预设验收标准。"))
    else:
        lines.append("固定广度：验收记录缺失，暂不能判断是否通过。")

    sl_rows = [r for r in results if r.get("policy") == "supervised" and r.get("family", "primary") == "primary"]
    sl_zero = bool(sl_rows) and all((finite(metric(r, "closed_trades")) or 0.0) == 0 for r in sl_rows)
    if sl_zero:
        lines.append(f"监督过滤（SL）：摘要中的{len(sl_rows)}个主账户记录均为零平仓；零交易带来的零回撤不能称为成功。")
    elif sl_rows:
        nonzero = sum(1 for r in sl_rows if (finite(metric(r, "closed_trades")) or 0.0) > 0)
        lines.append(f"监督过滤（SL）：主账户记录中有{nonzero}个出现平仓；需结合各时期收益、风险及覆盖情况判断。")
    else:
        lines.append("监督过滤（SL）：主账户结果缺失，不能判断交易与回撤情况。")

    rl_checks = (verdicts.get("rl_vote") or {}).get("checks") or []
    old_dd_failed = any(c.get("period") == "2023-2025" and c.get("drawdown_not_worse") is False for c in rl_checks)
    old_check = next((c for c in rl_checks if c.get("period") == "2023-2025"), None)
    if old_check is None:
        lines.append("RL三种子投票：2023—2025回撤验收记录缺失。")
    elif old_dd_failed:
        lines.append("RL三种子投票：2023—2025旧期回撤条件未通过。")
    else:
        lines.append("RL三种子投票：2023—2025旧期回撤条件按记录通过。")

    n, wins, rate = fixed_2026_trade_summary(run, results)
    if n is None:
        lines.append("2026固定广度平仓胜负明细缺失。")
    elif n == 0:
        lines.append("2026固定广度：没有可用于计算胜率的平仓；不能据此证明稳定胜率高于60%。")
    elif wins is None or rate is None:
        lines.append(f"2026固定广度：平仓数为{n}，胜负明细不足；不能据此证明稳定胜率高于60%。")
    else:
        lines.append(f"2026固定广度：{n}笔平仓中{wins}笔盈利，样本胜率{fmt_pct(rate)}；这个样本量不能证明稳定胜率高于60%。")

    score_range, score_count = opportunity_score_range(run, results)
    threshold_text = fmt_num(threshold, 3) if threshold is not None else "阈值缺失"
    lines.append(f"SL有待买单批次分数范围：{score_range}；冻结阈值为{threshold_text}。这些分数未经校准，不能解释为可信概率。")
    if isinstance(diagnostics, dict) and diagnostics.get("evaluation"):
        evaluations = [x for x in diagnostics.get("evaluation", []) if isinstance(x, dict)]
        evaluated_max = max((finite(x.get("max_score")) for x in evaluations if finite(x.get("max_score")) is not None), default=None)
        all_empty = bool(evaluations) and all(truth(x.get("all_opportunities_empty_account")) for x in evaluations)
        if all_empty and evaluated_max is not None and threshold is not None and evaluated_max < threshold:
            lines.append(f"空账户反馈形成退化闭环：诊断中所有有待买单机会都来自空账户，最高分{fmt_num(evaluated_max, 6)}仍低于阈值{threshold_text}，所以持续拒绝买单；账户没有交易后，仓位、回撤和持仓年龄等状态继续留在空账户区域，后续分数也难以越过阈值。")
        else:
            lines.append("空账户反馈说明：诊断文件没有证实所有机会都来自空账户，或无法与冻结阈值比较；不把无交易结果解释成低风险成功。")
        correct, total, accuracy = in_sample_accuracy(diagnostics)
        if correct is not None:
            if truth(diagnostics.get("training_in_sample_is_not_validation")):
                lines.append(f"训练内批次分类为{correct}/{total}正确（{fmt_pct(accuracy)}），但这是训练内结果，不能验证部署效果。")
            else:
                lines.append(f"诊断记录训练内分类为{correct}/{total}正确（{fmt_pct(accuracy)}）；是否属于独立验证需以审计材料为准。")
    else:
        lines.append("空账户反馈机制诊断文件缺失；仅凭零平仓和零回撤不把监督过滤判为成功。")

    audit_path = run / "independent-export-audit.json"
    audit_data: Any = None
    if audit_path.is_file():
        try:
            audit_data = json.loads(audit_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            audit_data = {"read_error": "文件存在但无法解析"}
        if isinstance(audit_data, dict) and "passed" in audit_data:
            audit_line = "独立导出审计：文件已生成，记录状态为" + ("通过。" if truth(audit_data.get("passed")) else "未通过。")
        else:
            audit_line = "独立导出审计：文件已生成，但没有明确的通过字段；审计仍在开发中，不预先宣称通过。"
    else:
        audit_line = "独立导出审计：文件尚未生成；审计仍在开发中，不预先宣称通过。"
    lines.append(audit_line)
    metadata = {"score_range": score_range, "score_count": score_count, "threshold": threshold,
                "diagnostics": diagnostics, "audit": audit_data,
                "fixed_2026_trades": n, "fixed_2026_wins": wins, "fixed_2026_win_rate": rate}
    return lines, metadata


def generate(run: Path) -> None:
    summary_path = run / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"缺少实验摘要：{summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    results = summary.get("results") or []
    failures = summary.get("failures") or []
    checks = summary.get("checks") or {}
    verdicts = summary.get("verdicts") or {}
    conclusions, direct_meta = direct_conclusions(run, summary, results)
    training, training_html, training_breakdown_html = training_summary(run, summary)
    focus_html, all_html, md_focus, md_all = primary_tables(results)
    seed_html, seed_md, seed_count = seed_table(results)
    links_html, links_md = artifact_links(run)
    cases_html, cases_md = case_links(run, results)

    verdict_rows = []
    for policy in POLICIES:
        if policy == "baseline":
            continue
        entry = verdicts.get(policy) or {}
        for check in entry.get("checks", []) if isinstance(entry, dict) else []:
            verdict_rows.append([
                esc(POLICY_NAMES[policy]), esc(check.get("period", "缺失")),
                esc(check.get("higher_net_return", "缺失")),
                esc(check.get("drawdown_not_worse", "缺失")),
                esc(check.get("higher_stressed_return", "缺失")),
                esc(check.get("passed", "缺失")),
            ])
    verdict_html = table(["策略", "时期", "净收益更高", "回撤不更差", "压力收益更高", "通过"], verdict_rows) if verdict_rows else "<p>没有验收判定记录。</p>"
    failure_html = "<ul>" + "".join(f"<li>{esc(json.dumps(f, ensure_ascii=False, default=str))}</li>" for f in failures) + "</ul>" if failures else "<p>摘要中没有失败记录。</p>"
    checks_html = esc(records_json(checks))
    verdicts_json = esc(records_json(verdicts))
    failures_json = esc(records_json(failures))
    training_json = esc(records_json(training))
    conclusions_html = "<ul>" + "".join(f"<li>{esc(line)}</li>" for line in conclusions) + "</ul>"
    audit_data = direct_meta.get("audit")
    audit_raw = records_json(audit_data) if audit_data is not None else "独立导出审计文件尚未生成。"
    diagnostics_raw = records_json(direct_meta.get("diagnostics")) if direct_meta.get("diagnostics") is not None else "监督过滤诊断文件缺失。"

    display_version = summary.get("display_version") or "缺失"
    adopted = summary.get("current_adopted") or "缺失"
    complete = yesno(summary.get("complete"))
    results_count = len(results)
    period_missing = [p for p, _ in PERIODS if not any(r.get("period") == p for r in results)]
    status_counts = training.get("status_counts") or {}
    class_counts = training.get("class_counts") or {}
    status_text = "、".join(f"{k} {fmt_num(v, 0)}批" for k, v in sorted(status_counts.items())) or "状态计数缺失"
    class_text = "、".join(f"{k} {fmt_num(v, 0)}批" for k, v in sorted(class_counts.items(), key=lambda kv: str(kv[0]))) or "类别计数缺失"
    verdict_status = []
    for policy in POLICIES:
        if policy == "baseline":
            continue
        verdict = verdicts.get(policy)
        if not isinstance(verdict, dict) or "passed" not in verdict:
            state = "缺失"
        else:
            state = "通过" if truth(verdict.get("passed")) else "未通过"
        verdict_status.append(f"{POLICY_NAMES[policy]}{state}")
    acceptance_status = "；".join(verdict_status) if verdict_status else "预设验收状态缺失"
    summary_intro = (
        f"本轮可用训练数据记录为 {fmt_num(training.get('rows'), 0)} 个拟合批次，"
        f"类别计数：{class_text}；标签状态：{status_text}。"
        f"本轮验收状态：运行完整={complete}；监督拟合可用={yesno(training.get('fit_available'))}；"
        f"预设比较验收：{acceptance_status}；结果记录 {results_count} 条。"
    )
    limitations = (
        "标签取自V0.8实际成交批次最终完成后的净盈亏，不是对接受与跳过动作分别构造的反事实效用。"
        "训练使用2022年，状态沿用RL实验的7个离散状态；模型为标准化逻辑回归，C=1，阈值0.5。"
        "RL对照使用冻结的原2022年三种子投票；退出规则沿用V0.8。标签样本少、状态分布可能变化，"
        "因此结果只说明这次已查看历史中的对照，不能据此称监督过滤整体优于RL，也不能宣称稳定胜率达到60%。"
        "2022—2026历史均已查看，不是盲测或独立未来验证；V0.8仍为当前采用版本，本研究不修改正式V0.8。"
    )
    state_support_note = (
        "状态支持范围的判定口径不同：监督过滤（SL）按是否精确匹配2022年成熟标签训练状态集合；"
        "RL三种子投票只把三个Q表状态集合的交集视作已支持状态；单种子诊断则按该种子自身Q表状态集合判定。"
        "这些只是粗粒度的状态支持范围诊断，不能解释为模型输出概率可信，也不能证明未见状态上的表现。"
    )

    md = [
        f"# {display_version}：监督过滤对照研究", "",
        f"- 显示版本：{display_version}", f"- 当前采用版本：{adopted}",
        f"- 运行完整：{complete}", f"- 结果记录：{results_count}",
        f"- 拟合批次：{fmt_num(training.get('rows'), 0)}；有待买单批次（含no_execution）：{fmt_num(training.get('total_opportunities'), 0)}；不同状态：{fmt_num(training.get('unique_states'), 0)}",
        f"- 类别计数：{class_text}", f"- 标签状态：{status_text}",
        f"- 拟合可用：{yesno(training.get('fit_available'))}；标签最后结束日：{training.get('last_label_end') or '缺失'}",
        "", summary_intro, "", "## 直接结论", "",
        *[f"- {line}" for line in conclusions], "", "## 主要时期四组对照", "",
        "以下为正常成本主结果；数值直接取本次运行摘要。接受率按候选批次记录。", "",
        "| 时期 | 指标 |" + " | ".join(POLICY_NAMES[p] for p in POLICIES) + " |",
        "|---|---|" + "|".join(["---:"] * len(POLICIES)) + "|",
    ]
    md += ["| " + " | ".join(row) + " |" for row in md_focus]
    md += ["", "## 六个时期：正常成本与双倍滑点", "",
           "每格同时列出净收益、最大回撤、平均暴露、平仓数、批次接受率和未见状态比例；缺少结果明确标为缺失。", "",
           "| 时期 | 情景 |" + " | ".join(POLICY_NAMES[p] for p in POLICIES) + " |",
           "|---|---|" + "|".join(["---"] * len(POLICIES)) + "|"]
    md += ["| " + " | ".join(row) + " |" for row in md_all]
    md += ["", "## 拟合批次标签状态与类别", "",
           f"标签最后结束日：{training.get('last_label_end') or '缺失'}；模型文件：{training.get('model_file') or '缺失'}。", "",
           "### 标签状态", "", "| 状态 | 批次数 |", "|---|---:|"]
    md += [f"| {k} | {fmt_num(v, 0)} |" for k, v in sorted(status_counts.items())] or ["| 缺失 | — |"]
    md += ["", "### 类别计数", "", "| 类别 | 批次数 |", "|---|---:|"]
    md += [f"| {k} | {fmt_num(v, 0)} |" for k, v in sorted(class_counts.items(), key=lambda kv: str(kv[0]))] or ["| 缺失 | — |"]
    md += ["", "## 预设验收判定", "", "| 策略 | 时期 | 净收益更高 | 回撤不更差 | 压力收益更高 | 通过 |", "|---|---|---|---|---|---|"]
    for policy in POLICIES:
        if policy == "baseline":
            continue
        entry = verdicts.get(policy) or {}
        for check in entry.get("checks", []) if isinstance(entry, dict) else []:
            md.append("| " + " | ".join(str(v) for v in [
                POLICY_NAMES[policy], check.get("period", "缺失"), check.get("higher_net_return", "缺失"),
                check.get("drawdown_not_worse", "缺失"), check.get("higher_stressed_return", "缺失"),
                check.get("passed", "缺失")]) + " |")
    if not verdict_rows:
        md.append("| 缺失 | — | — | — | — | — |")
    md += ["", "## 状态支持范围说明", "", state_support_note, "",
           "## 研究口径与限制", "", limitations, "",
           "本轮没有改变正式版本，也不把历史对照当作未来表现保证。", "",
           "## 独立导出审计摘要", "", "```json", audit_raw, "```",
           "", "## 监督过滤诊断摘要", "", "```json", diagnostics_raw, "```",
           "", "## 完整性检查、失败和判定原文", "", "### checks（原样）", "", "```json", records_json(checks), "```",
           "", "### verdicts（原样）", "", "```json", records_json(verdicts), "```",
           "", "### failures（原样）", "", "```json", records_json(failures), "```",
           "", "## 训练概况原文", "", "```json", records_json(training), "```",
           "", "## 种子诊断：全部披露", "", f"全部三种子的诊断项均披露、不挑选；摘要中记录 {seed_count} 条 seed 家族结果。", "",
           seed_md if seed_md else "没有种子诊断记录。", "",
           "## 运行产物链接", "", *links_md, "", "## 各时期与情景账户文件", "", *cases_md]
    if period_missing:
        md += ["", "摘要中完全没有这些时期的结果：" + "、".join(period_missing) + "。"]
    (run / "research-report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    html_doc = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(display_version)}</title><style>
body{{font:15px/1.7 system-ui,'Microsoft YaHei',sans-serif;max-width:1320px;margin:28px auto;padding:0 20px;color:#1d2937;background:#f7f9fc}}h1,h2,h3{{line-height:1.3}}h2{{margin-top:2rem;border-bottom:2px solid #d8e1ec;padding-bottom:.3rem}}.box{{background:#fff6e5;border-left:5px solid #d18c26;padding:14px 18px;margin:16px 0}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;min-width:850px;background:white;margin:.5rem 0 1.2rem}}th,td{{border:1px solid #d8e0e8;padding:8px 10px;text-align:right;vertical-align:top;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}.wide td{{white-space:normal;min-width:225px;text-align:left}}.focus td{{text-align:right}}.focus td:first-child,.focus td:nth-child(2){{text-align:left}}thead{{background:#eaf0f6}}tbody tr:nth-child(even){{background:#f5f8fb}}a{{color:#075e9b}}details{{margin:1rem 0;background:#fff;padding:8px 12px;border:1px solid #d8e0e8}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}.missing{{color:#a33}}.lead{{font-size:1.05rem}}.seed td{{font-size:.9rem}}</style></head><body>
<h1>{esc(display_version)}</h1><p class="lead"><strong>{esc(summary_intro)}</strong></p>
<div class="box">{esc(limitations)}</div>
<h2>主要时期四组对照</h2><p>以下为正常成本主结果。所有数值直接取运行摘要；接受率表示候选批次的接受比例。</p>{focus_html}
<p>固定规则：候选池中已知近5个交易日复权收益为正的股票占比达到58%才接受新买单，沿用原候选资格及至少100个已知收益的广度有效性检查。它沿用原股票池及信号，不是全A股的当日上涨比例。回撤保存为负值，越接近零越小；2023—2025为三年累计收益。</p>
<h2>结论与原因</h2>{conclusions_html}
<h2>六个时期：正常成本与双倍滑点</h2><p>压力情景仅将滑点从 0.0005 调整为 0.001；佣金与印花税不作倍增。每格列出净收益、最大回撤、平均暴露、平仓数、批次接受率和未见状态比例。缺失数据直接标明。</p>{all_html}
<h2>训练标签概况</h2>{training_html}<h3>状态分布与类别</h3>{training_breakdown_html}
<details><summary>展开训练概况记录</summary><pre>{training_json}</pre></details>
<h2>状态支持范围说明</h2><p>{esc(state_support_note)}</p>
<h2>预设验收判定</h2>{verdict_html}
<h2>审计与监督过滤诊断摘要</h2><details><summary>展开独立导出审计摘要</summary><pre>{esc(audit_raw)}</pre></details><details><summary>展开监督过滤诊断摘要</summary><pre>{esc(diagnostics_raw)}</pre></details>
<h2>完整性检查与失败记录</h2><p>失败记录：{failure_html}</p>
<details><summary>展开 checks 原文</summary><pre>{checks_html}</pre></details><details><summary>展开 verdicts 原文</summary><pre>{verdicts_json}</pre></details><details><summary>展开 failures 原文</summary><pre>{failures_json}</pre></details>
<h2>种子诊断：全部披露</h2><p>全部三种子的诊断项均披露、不挑选；摘要中记录 {seed_count} 条 seed 家族结果，覆盖其全部时期、策略和情景记录。</p><details><summary>展开全部种子结果</summary>{seed_html}</details>
<h2>运行产物</h2>{links_html}<h2>各时期与情景账户文件</h2><details><summary>展开账户摘要、净值、成交、平仓与原始决策链接</summary>{cases_html}</details>
<p>运行完整：{complete}；当前采用版本：{esc(adopted)}。本研究不更新正式 V0.8。</p></body></html>"""
    (run / "report.html").write_text(html_doc, encoding="utf-8")


def cli() -> None:
    parser = argparse.ArgumentParser(description="生成 V0.9 监督过滤对照研究报告")
    parser.add_argument("--run", type=Path, required=True, help="包含 summary.json 的运行目录")
    args = parser.parse_args()
    run = args.run.resolve()
    generate(run)
    print(f"HTML: {run / 'report.html'}")
    print(f"Markdown: {run / 'research-report.md'}")


if __name__ == "__main__":
    cli()
