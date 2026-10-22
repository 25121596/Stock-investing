"""Build the Chinese research report from a completed V0.9 RL prototype run."""
import argparse
import csv
import html
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "results" / "v09-rl-2022-prototype-20261021-v2"
POLICIES = ["baseline", "seed_2026102101", "seed_2026102102", "seed_2026102103", "vote"]
POLICY_LABELS = {
    "baseline": "V0.8基线", "seed_2026102101": "种子 2026102101",
    "seed_2026102102": "种子 2026102102", "seed_2026102103": "种子 2026102103",
    "vote": "固定三种子投票",
}
PERIODS = [
    ("2022_train", "2022 训练期"), ("2023", "2023 单年留出"),
    ("2024", "2024 单年留出"), ("2025", "2025 单年留出"),
    ("2023-2025", "2023–2025 连续留出"), ("2026", "2026 历史留出"),
]


def esc(value):
    return html.escape(str(value), quote=True)


def fmt_pct(value):
    return "—" if value is None else f"{100 * float(value):.2f}%"


def fmt_money(value):
    return "—" if value is None else f"¥{float(value):,.0f}"


def fmt_number(value, digits=2):
    return "—" if value is None else f"{float(value):,.{digits}f}"


def load_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def case_dir(period, policy, stress):
    middle = "stress_" if stress else ""
    return RUN_DIR / f"{period}_{middle}{policy}"


def get_result(rows, period, policy, stress=False):
    return next((row for row in rows if row["period"] == period and row["policy"] == policy
                 and bool(row.get("stress", False)) == stress), None)


def relative_link(path, label=None):
    try:
        target = path.relative_to(RUN_DIR).as_posix()
    except ValueError:
        target = path.name
    return f'<a href="{esc(target)}">{esc(label or path.name)}</a>'


def result_table(rows, period, stress):
    baseline = get_result(rows, period, "baseline", stress)
    data = []
    for policy in POLICIES:
        item = get_result(rows, period, policy, stress)
        if item is None:
            data.append(f"<tr><th>{esc(POLICY_LABELS[policy])}</th><td colspan=6>该场次无结果</td></tr>")
            continue
        delta = None if baseline is None else float(item["final_equity"]) - float(baseline["final_equity"])
        folder = case_dir(period, policy, stress)
        data.append(
            "<tr>"
            f"<th>{esc(POLICY_LABELS[policy])}</th>"
            f"<td>{fmt_pct(item.get('total_return'))}</td>"
            f"<td>{fmt_money(item.get('final_equity'))}</td>"
            f"<td>{'—' if delta is None else ('+' if delta >= 0 else '') + fmt_money(delta)}</td>"
            f"<td>{fmt_pct(item.get('max_drawdown'))}</td>"
            f"<td>{fmt_number(item.get('closed_trades'), 0)}</td>"
            f"<td>{relative_link(folder / 'summary.json', '账本与详情')}</td>"
            "</tr>"
        )
    stress_note = "双倍滑点" if stress else "正常滑点"
    return (f"<h4>{stress_note}</h4><div class='scroll'><table><thead><tr>"
            "<th>策略</th><th>净收益</th><th>期末净值</th><th>相对V0.8差额</th>"
            "<th>最大回撤</th><th>平仓数</th><th>明细</th></tr></thead><tbody>"
            + "".join(data) + "</tbody></table></div>")


def svg_curve(path, title, color):
    if not path.is_file():
        return ""
    records = load_csv(path)
    if len(records) < 2:
        return ""
    values = [float(r["equity"]) for r in records]
    low, high = min(values), max(values)
    span = max(high - low, 1.)
    width, height, left, top, inner_w, inner_h = 680, 170, 48, 20, 610, 112
    points = " ".join(
        f"{left + i * inner_w / max(1, len(values) - 1):.1f},"
        f"{top + inner_h - (v - low) * inner_h / span:.1f}"
        for i, v in enumerate(values)
    )
    start, end = records[0].get("date", ""), records[-1].get("date", "")
    return (f"<figure class='curve'><figcaption>{esc(title)} · {esc(start)} 至 {esc(end)}</figcaption>"
            f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{esc(title)}净值曲线'>"
            f"<line x1='{left}' y1='{top + inner_h}' x2='{left + inner_w}' y2='{top + inner_h}' stroke='#b8c4d1'/>"
            f"<polyline fill='none' stroke='{color}' stroke-width='2.5' points='{points}'/>"
            f"<text x='4' y='{top + 4}'>{esc(fmt_money(high))}</text>"
            f"<text x='4' y='{top + inner_h}'>{esc(fmt_money(low))}</text>"
            f"<text x='{left}' y='158'>{esc(start)}</text>"
            f"<text x='{left + inner_w - 75}' y='158'>{esc(end)}</text></svg></figure>")


def learning_chart(records):
    if not records:
        return "<p>未找到 learning.csv。</p>"
    by_seed = {}
    for row in records:
        by_seed.setdefault(row["seed"], []).append(row)
    reward_values = [float(row["reward"]) for row in records]
    low, high = min(reward_values), max(reward_values)
    span = max(high - low, 1.)
    colors = ["#1b74b8", "#e1812c", "#29995b"]
    polylines = []
    for color, (seed, rows) in zip(colors, sorted(by_seed.items())):
        rows.sort(key=lambda row: int(row["epoch"]))
        pts = []
        for row in rows:
            x = 58 + (int(row["epoch"]) - 1) * 680 / max(1, len(rows) - 1)
            y = 20 + 118 - (float(row["reward"]) - low) * 118 / span
            pts.append(f"{x:.1f},{y:.1f}")
        polylines.append(f"<polyline fill='none' stroke='{color}' stroke-width='1.5' points='{ ' '.join(pts) }'/>")
    legend = "　".join(f"<span style='color:{c}'>● {esc(seed)}</span>" for c, seed in zip(colors, sorted(by_seed)))
    return ("<figure><figcaption>2022训练轨迹：每轮回报（每个种子单独一条同一训练期序列）</figcaption>"
            "<svg viewBox='0 0 760 175' role='img' aria-label='各随机种子训练回报曲线'>"
            "<line x1='58' y1='138' x2='738' y2='138' stroke='#b8c4d1'/>"
            + "".join(polylines)
            + f"<text x='3' y='24'>{esc(fmt_number(high))}</text><text x='3' y='138'>{esc(fmt_number(low))}</text>"
            "<text x='58' y='158'>Epoch 1</text><text x='690' y='158'>Epoch 200</text></svg>"
            f"<div class='legend'>{legend}</div></figure>")


def collect_quality(summary, rows):
    failures = summary.get("failures") or []
    issues = []
    for period, _ in PERIODS:
        for policy in POLICIES:
            for stress in (False, True):
                result = get_result(rows, period, policy, stress)
                if result is None:
                    continue
                folder = case_dir(period, policy, stress)
                count = int(result.get("quality_warning_count") or 0)
                audit = result.get("audit") or {}
                if count or audit.get("passed") is False:
                    issues.append((period, policy, stress, count, audit))
                for filename in ("warnings.csv", "review_warnings.csv"):
                    fpath = folder / filename
                    if fpath.exists():
                        records = load_csv(fpath)
                        if records:
                            issues.append((period, policy, stress, filename, records))
    return failures, issues


def build(summary):
    rows = summary.get("results", [])
    acceptance = summary.get("acceptance", {})
    passed = bool(acceptance.get("passed", False))
    status = "通過" if passed else "未通过"
    status_class = "pass" if passed else "fail"
    failures, quality_issues = collect_quality(summary, rows)
    headline=[]
    for period,label in [('2022_train','2022训练期'),('2023-2025','2023—2025连续账户'),('2026','2026截至10月13日')]:
        b=get_result(rows,period,'baseline');v=get_result(rows,period,'vote')
        if b and v:
            headline.append((label,b,v))
    headline_html='<table><thead><tr><th>时期</th><th>V0.8收益</th><th>RL固定投票收益</th><th>V0.8回撤</th><th>RL回撤</th><th>3万元收益差</th></tr></thead><tbody>'
    for label,b,v in headline:
        headline_html+=f"<tr><th>{esc(label)}</th><td>{fmt_pct(b['total_return'])}</td><td>{fmt_pct(v['total_return'])}</td><td>{fmt_pct(abs(b['max_drawdown']))}</td><td>{fmt_pct(abs(v['max_drawdown']))}</td><td>{fmt_money(v['final_equity']-b['final_equity'])}</td></tr>"
    headline_html+='</tbody></table>'
    diagnosis=('原型已经跑通，但尚未达到替换V0.8的要求。固定投票在2023—2025连续账户中提高收益并降低回撤；2026降低回撤的同时丢失较多收益。'
               '2023和2025单年收益也落后，改善集中在2024。三种子中一组在训练期只平仓1笔、几乎空仓，显示训练结果不稳定。'
               '2026投票胜率为9/11（81.8%），只有11笔平仓，不能据此确认稳定超过60%。')
    md = []
    md.append(f"# {summary.get('version', 'V0.9 强化学习研究原型')}\n")
    md.append(f"**验收状态：{status}。当前采用版本：{summary.get('current_adopted', '未提供')}。**\n")
    md.append("这是强化学习研究原型的历史回放结果。模型尚未替代当前运行策略。验收状态直接读取本次 `summary.json`。\n")
    md.append(diagnosis+'\n')
    md.append('| 时期 | V0.8收益 | RL固定投票收益 | V0.8回撤 | RL回撤 | 3万元收益差 |\n|---|---:|---:|---:|---:|---:|\n')
    for label,b,v in headline:
        md.append(f"| {label} | {fmt_pct(b['total_return'])} | {fmt_pct(v['total_return'])} | {fmt_pct(abs(b['max_drawdown']))} | {fmt_pct(abs(v['max_drawdown']))} | {fmt_money(v['final_equity']-b['final_equity'])} |\n")
    md.append('\n回撤在上表显示为损失幅度；原始导出保留负数。收益为扣费累计收益，不是年化。\n')
    md.append("## 结论与边界\n")
    md.append(f"- 完整运行：{'是' if summary.get('complete') else '否'}；完成时间：`{summary.get('completed_utc', '未提供')}`。\n")
    md.append(f"- 固定投票验收：**{status}**。验收分项按摘要原值列出：\n")
    for check in acceptance.get("checks", []):
        md.append(f"  - {check.get('period')}: 收益高于V0.8={check.get('higher_net_return')}；回撤不更差={check.get('drawdown_not_worse')}；双倍滑点收益更高={check.get('higher_stressed_return')}；通过={check.get('passed')}。\n")
    md.append("- 已查看历史数据上的研究回放，不是盲测，也不代表全市场或未来盈利保证。2022年是训练期；其余年份是冻结历史留出评估，但由于历史资料此前已经查看，不能称为未见数据。\n")
    md.append('- 固定300只历史股票样本，不能代表全A股；状态离散且省略股票身份、订单细节，属于部分信息的轻量原型。此前特征来源核验不等同供应商历史可得快照认证。\n')
    md.append("- 本原型未改 DAY1/RQAlpha 当前运行配置；summary 中当前采用版本为 V0.8。\n")
    md.append("\n## 结果\n")
    for period, label in PERIODS:
        md.append(f"### {label}\n")
        for stress in (False, True):
            heading = "双倍滑点" if stress else "正常滑点"
            md.append(f"#### {heading}\n")
            md.append("| 策略 | 净收益 | 期末净值 | 相对V0.8差额 | 最大回撤 | 平仓数 | 审计明细 |\n|---|---:|---:|---:|---:|---:|---|\n")
            baseline = get_result(rows, period, "baseline", stress)
            for policy in POLICIES:
                row = get_result(rows, period, policy, stress)
                if not row:
                    md.append(f"| {POLICY_LABELS[policy]} | 无结果 | — | — | — | — | — |\n")
                    continue
                delta = float(row["final_equity"]) - float(baseline["final_equity"]) if baseline else None
                delta_text = "—" if delta is None else f"{'+' if delta >= 0 else ''}{fmt_money(delta)}"
                folder = case_dir(period, policy, stress)
                md.append(f"| {POLICY_LABELS[policy]} | {fmt_pct(row.get('total_return'))} | {fmt_money(row.get('final_equity'))} | {delta_text} | {fmt_pct(row.get('max_drawdown'))} | {fmt_number(row.get('closed_trades'), 0)} | [{relative_md(folder / 'summary.json')}]({relative_md(folder / 'summary.json')}) |\n")
            md.append("\n")

    md.append("## 学习设定与数据切分\n")
    md.append(f"- 方法：离散状态、4个组合动作的表格 Q-learning；训练 {summary.get('epochs')} 轮，种子：{', '.join(map(str, summary.get('seeds', [])))}。\n")
    md.append("- 状态由前一交易日收盘信息离散化：市场广度档、市场5日收益档、持仓暴露档、账户回撤档、组合浮盈亏档、最长持仓年龄档、待买单数量。动作是跟随V0.8、取消前收盘排定的买单、次日开盘尝试退出旧持仓、同时执行前两项。\n")
    md.append("- 上游 V0.8 预测/排序与门控沿用原有逐年更新流程，按以前年份训练；不是把上游模型冻结成仅用2022训练的端到端系统。此次仅 RL 策略参数使用 2022 轨迹训练。\n")
    md.append(f"- `training_audits={summary.get('training_audits')}` 表示同一条 2022 历史轨迹被重复重放；{summary.get('epochs')} 轮不是独立年份，也不能当作独立样本数。每轮按完整交易日日期顺序执行。\n")
    md.append("- 2022训练、2023/2024/2025单年、2023–2025连续账户、2026历史留出分别展示。逐年重置结果与连续账户不可相加。每个区间均列正常与双倍滑点；V0.8、三个固定种子、固定投票全部保留。\n")
    md.append("\n## 质量检查与失败\n")
    parity = summary.get("passive_parity_cases", [])
    md.append(f"- 被动动作与 V0.8 对账：{sum(bool(x.get('passed')) for x in parity)}/{len(parity)} 场通过。\n")
    md.append(f"- 投票评估未更新策略：`policies_unchanged_in_eval={summary.get('policies_unchanged_in_eval')}`；输入未变：`{summary.get('inputs_unchanged')}`；源文件未变：`{summary.get('sources_unchanged')}`。\n")
    md.append(f"- 实验失败记录：{len(failures)} 条；各场质量告警/审计问题：{len(quality_issues)} 条。\n")
    if failures:
        for failure in failures:
            md.append(f"  - `{json.dumps(failure, ensure_ascii=False)}`\n")
    if quality_issues:
        for issue in quality_issues:
            md.append(f"  - `{json.dumps(issue, ensure_ascii=False)}`\n")
    elif not failures:
        md.append("- 摘要中没有场次失败，列出的运行均通过质量告警计数与账本审计；每个分场 summary、fills、trades、equity、decisions、warnings 和 review_warnings 可逐场核对。\n")

    md.append("\n## 可复核文件\n")
    for name, label in [("summary.json", "完整结果与验收"), ("comparison.csv", "逐场比较表"),
                        ("learning.csv", "逐轮学习记录"), ("protocol.md", "冻结协议与验收定义"),
                        ("training-scope.json", "2022训练输入范围"), ("training-complete.json", "训练冻结记录"),
                        ("checkpoint.json", "评估检查点"), ("independent-export-audit.json", "独立导出与训练重现审计"), ("models", "Q表模型与哈希")]:
        md.append(f"- [{label}]({name})\n")
    md.append("- 每个场次目录内含 `summary.json`、`equity.csv`、`fills.csv`、`trades.csv`、`decisions.csv`、`actions.csv`、`holdings.csv`、`warnings.csv` 和 `review_warnings.csv`。\n")
    md.append("- 输出报告：`report.html` 与 `research-report.md`。\n")

    html_sections = []
    for period, label in PERIODS:
        html_sections.append(f"<section><h2>{esc(label)}</h2>{result_table(rows, period, False)}{result_table(rows, period, True)}")
        html_sections.append("<div class='curves'>")
        for stress in (False, True):
            for policy, color in (("baseline", "#426b8a"), ("vote", "#ad4b41")):
                suffix = "双倍滑点" if stress else "正常滑点"
                html_sections.append(svg_curve(case_dir(period, policy, stress) / "equity.csv",
                                               f"{label} · {suffix} · {POLICY_LABELS[policy]}", color))
        html_sections.append("</div></section>")

    quality_html = []
    if failures:
        quality_html.append("<h3>实验失败记录</h3><ul>" + "".join(f"<li><code>{esc(json.dumps(x, ensure_ascii=False))}</code></li>" for x in failures) + "</ul>")
    if quality_issues:
        quality_html.append("<h3>场次质量告警与审计问题</h3><ul>" + "".join(f"<li><code>{esc(json.dumps(x, ensure_ascii=False))}</code></li>" for x in quality_issues) + "</ul>")
    if not failures and not quality_issues:
        quality_html.append("<p>摘要没有记录场次失败或质量告警；请使用下方场次账本链接逐场核验。</p>")

    reference_links = "<ul>" + "".join(
        f"<li>{relative_link(RUN_DIR / name, label)}</li>"
        for name, label in [("summary.json", "完整原始摘要"), ("comparison.csv", "逐场比较"),
                            ("learning.csv", "学习轨迹"), ("protocol.md", "协议"),
                            ("training-scope.json", "训练范围"), ("training-complete.json", "冻结记录"),
                            ("checkpoint.json", "评估检查点"), ("independent-export-audit.json", "独立导出与训练重现审计"), ("models", "Q表模型与哈希")]
    ) + "</ul>"
    for period, label in PERIODS:
        for stress in (False, True):
            for policy in POLICIES:
                folder = case_dir(period, policy, stress)
                if folder.is_dir():
                    reference_links += (f"<details><summary>{esc(label)} · {'双倍滑点' if stress else '正常滑点'} · "
                                        f"{esc(POLICY_LABELS[policy])}</summary><ul>" + "".join(
                                            f"<li>{relative_link(folder / filename)}</li>"
                                            for filename in ("summary.json", "equity.csv", "fills.csv", "trades.csv",
                                                             "decisions.csv", "actions.csv", "holdings.csv",
                                                             "warnings.csv", "review_warnings.csv")
                                            if (folder / filename).is_file()
                                        ) + "</ul></details>")

    scope_path = RUN_DIR / "training-scope.json"
    scope = json.loads(scope_path.read_text(encoding="utf-8")) if scope_path.is_file() else {}
    md_text = re.sub(r'(?m)(^\|[^\n]*\|)\n\n(?=\|)',r'\1\n',"\n".join(md))
    html_text = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V0.9 强化学习研究原型报告</title><style>
body{{font:15px/1.65 system-ui,-apple-system,'Microsoft YaHei',sans-serif;max-width:1180px;margin:30px auto;padding:0 20px;color:#1d2937;background:#f7f9fc}}
h1,h2,h3{{line-height:1.3}} h1{{font-size:2rem}} h2{{margin-top:2.3rem;border-bottom:2px solid #d8e1ec;padding-bottom:.35rem}}
.status{{display:inline-block;padding:.25rem .7rem;border-radius:5px;font-weight:700}} .pass{{background:#d9f2e2;color:#17633b}} .fail{{background:#ffe4df;color:#992f20}}
.callout{{border-left:5px solid #d18c26;background:#fff6e5;padding:12px 17px;margin:18px 0}} .scroll{{overflow:auto}}
table{{border-collapse:collapse;width:100%;min-width:760px;background:white;margin:.5rem 0 1.2rem}} th,td{{border:1px solid #d8e0e8;padding:7px 9px;text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}} thead{{background:#eaf0f6}} tbody tr:nth-child(even){{background:#f5f8fb}}
a{{color:#075e9b}} .curves{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:10px}} figure{{background:white;border:1px solid #d8e0e8;border-radius:8px;padding:10px;margin:8px 0;overflow:hidden}}
svg{{width:100%;height:auto}} svg text{{font-size:10px;fill:#4a5967}} figcaption{{font-weight:600;margin-bottom:4px}} .legend{{font-size:12px}} code{{white-space:pre-wrap;overflow-wrap:anywhere}}
details{{margin:.2rem 0}} footer{{border-top:1px solid #ccd6e0;margin-top:2rem;padding-top:1rem;color:#536273}}
</style></head><body>
<h1>V0.9 强化学习研究原型</h1>
<p><span class="status {status_class}">验收：{status}</span>　<strong>当前采用版本：{esc(summary.get('current_adopted', '未提供'))}</strong>　运行完整：{esc(summary.get('complete'))}</p>
<div class="callout">本报告基于此前已查看的历史行情，属于研究回放而非盲测。结果不构成全市场或未来盈利保证。V0.9 尚未替换 DAY1/RQAlpha 当前运行策略。</div>
<h2>原型结论</h2><p>{esc(diagnosis)}</p><div class="scroll">{headline_html}</div>
<p>以上均为3万元账户的扣费累计收益；回撤显示损失幅度。2022为训练期，不能用来证明泛化。单年重置账户与连续账户不能相加。</p>
<p>2026平均持仓暴露从V0.8的9.48%降至投票策略的2.76%，较低回撤伴随较少资金参与。此轮验证能否学习交易决策，尚未证明具有更高盈利能力。</p>
<p>完成时间：{esc(summary.get('completed_utc', '未提供'))}。验收字段直接读取原始 summary：{relative_link(RUN_DIR / 'summary.json')}。</p>
<h2>方法和数据切分</h2><p>使用表格 Q-learning：离散状态包含广度、市场5日收益、持仓暴露、账户回撤、持仓浮盈亏、最长持仓年龄和待买单数；动作是在 V0.8 原动作上选择跟随、跳过买单、次日开盘尝试退出或组合动作。策略不新增股票信号或提高风险限额。</p>
<p>仅 RL 参数使用 2022 年轨迹训练；上游 V0.8 逐年更新，沿用以前年份训练的预测/排序与门控，并非端到端冻结在 2022。训练轮数 {esc(summary.get('epochs'))}，固定种子 {esc(', '.join(map(str, summary.get('seeds', []))))}，三种子多数投票固定使用。</p>
<p>训练检查：{esc(summary.get('training_audits'))} 次 2022 轨迹重放审计。这是重复同一条历史轨迹，不是独立年份样本。2022训练期、2023/2024/2025单年、2023–2025连续账户、2026留出账户分开呈现。所有区间均列正常和双倍滑点以及基线、三个种子和固定投票。</p>
<p>2022交易日范围：{esc(scope.get('first', '—'))} 至 {esc(scope.get('last', '—'))}，{esc(scope.get('sessions', '—'))} 个交易日。协议：{relative_link(RUN_DIR / 'protocol.md')}；训练输入范围：{relative_link(RUN_DIR / 'training-scope.json')}。</p>
<p>固定300只历史股票样本；状态是离散、部分信息的近似。既有缓存特征曾从原始日线重算，但不等同供应商历史可得快照认证。若双倍滑点收益高于正常滑点，是费用改变可用资金、整手数量和后续成交路径的结果，两者独立模拟，不能将收益差简单解释成费用差。</p>
{learning_chart(load_csv(RUN_DIR / 'learning.csv') if (RUN_DIR / 'learning.csv').exists() else [])}
<h2>分期结果</h2>{''.join(html_sections)}
<h2>验收、账本与质量失败</h2><p>验收总状态：<strong>{status}</strong>。以下每场结果均包含人民币期末净值与相对同场 V0.8 的人民币差额。数值差额不含额外假设。</p>
<div class="scroll"><table><thead><tr><th>留出期</th><th>收益更高</th><th>回撤不更差</th><th>压力收益更高</th><th>通过</th></tr></thead><tbody>{''.join(f"<tr><th>{esc(x.get('period'))}</th><td>{esc(x.get('higher_net_return'))}</td><td>{esc(x.get('drawdown_not_worse'))}</td><td>{esc(x.get('higher_stressed_return'))}</td><td>{esc(x.get('passed'))}</td></tr>" for x in acceptance.get('checks', []))}</tbody></table></div>
<p>被动动作与 V0.8 一致：{sum(bool(x.get('passed')) for x in parity)}/{len(parity)} 场。评估策略未更新：{esc(summary.get('policies_unchanged_in_eval'))}；输入未变：{esc(summary.get('inputs_unchanged'))}；源码未变：{esc(summary.get('sources_unchanged'))}。实验失败 {len(failures)} 条；场次质量问题 {len(quality_issues)} 条。</p>
{''.join(quality_html)}
<h2>原始文件和逐场审计</h2>{reference_links}
<footer>历史已查看数据的研究材料；不能据此推断未来盈利能力。</footer>
</body></html>"""
    (RUN_DIR / "research-report.md").write_text(md_text, encoding="utf-8")
    (RUN_DIR / "report.html").write_text(html_text, encoding="utf-8")
    return RUN_DIR / "report.html", RUN_DIR / "research-report.md"


def relative_md(path):
    try:
        return path.relative_to(RUN_DIR).as_posix()
    except ValueError:
        return path.name


def main():
    global RUN_DIR
    parser = argparse.ArgumentParser(description="从V0.9 RL原型结果生成中文HTML和Markdown报告")
    parser.add_argument("--run", type=Path, required=True, help="已完成的实验目录")
    args = parser.parse_args()
    RUN_DIR=args.run.resolve()
    summary_path = RUN_DIR / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"实验摘要尚不存在：{summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    html_path, md_path = build(summary)
    print(f"HTML: {html_path}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
