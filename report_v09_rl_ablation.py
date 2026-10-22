"""Generate a Chinese report for the fixed-weight V0.9 action ablation."""
import argparse
import html
import json
import math
from pathlib import Path


PERIODS = [
    ("2022_train", "2022 训练期"), ("2023", "2023"), ("2024", "2024"),
    ("2025", "2025"), ("2023-2025", "2023–2025 连续期"), ("2026", "2026截至10月13日"),
]
MODES = ["baseline", "filter_only", "exit_only", "full_rl"]
LABELS = {"baseline": "基线", "filter_only": "仅过滤", "exit_only": "仅退出", "full_rl": "过滤+退出"}
LINK_FILES = ["summary.json", "comparison.csv", "primary-comparison.csv", "effects.csv",
              "protocol.md", "started.json", "independent-export-audit.json", "model-freeze.json"]


def esc(x):
    return html.escape(str(x), quote=True)


def number(x, digits=2):
    if x is None or x == "":
        return "—"
    try:
        v = float(x)
        return "—" if not math.isfinite(v) else f"{v:,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def pct(x):
    if x is None or x == "":
        return "—"
    try:
        v = float(x)
        return "—" if not math.isfinite(v) else f"{100*v:.2f}%"
    except (TypeError, ValueError):
        return "—"


def money(x):
    v = number(x, 0)
    return "—" if v == "—" else f"¥{v}"


def truth(v):
    return v is True or (isinstance(v, str) and v.lower() == "true")


def yesno(v):return '是' if v is True else ('否' if v is False else '缺失')


def get_result(results, period, mode, stress=False, family="primary"):
    return next((r for r in results if r.get("period") == period and r.get("mode") == mode
                 and r.get("family", "primary") == family and truth(r.get("stress")) == stress), None)


def table(headers, rows):
    head = "<thead><tr>" + "".join(f"<th>{esc(h)}</th>" for h in headers) + "</tr></thead>"
    body = "<tbody>" + "".join("<tr>" + "".join(f"<td>{x}</td>" for x in row) + "</tr>" for row in rows) + "</tbody>"
    return "<div class='scroll'><table>" + head + body + "</table></div>"


def main_tables(summary):
    results = summary.get("results") or []
    primary = [r for r in results if r.get("family", "primary") == "primary"]
    ret_rows, risk_rows, stress_rows, md_ret, md_risk, md_stress = [], [], [], [], [], []
    for period, label in PERIODS:
        row = [esc(label)]
        mrow = [label]
        for mode in MODES:
            r = get_result(primary, period, mode)
            val = pct(r.get("total_return")) if r else "— 缺失"
            row.append(esc(val)); mrow.append(val)
        ret_rows.append(row); md_ret.append(mrow)
        if period in ("2023-2025", "2026"):
            row = [esc(label)]; mrow = [label]
            for mode in MODES:
                r = get_result(primary, period, mode)
                cell = "— 缺失" if not r else f"DD {pct(r.get('max_drawdown'))}；暴露 {pct(r.get('average_exposure'))}；费用 {money(r.get('execution_costs'))}"
                row.append(esc(cell)); mrow.append(cell)
            risk_rows.append(row); md_risk.append(mrow)
        for stress in (False, True):
            row = [esc(label), "压力" if stress else "正常"]
            mrow = [label, "压力" if stress else "正常"]
            for mode in MODES:
                r = get_result(primary, period, mode, stress)
                val = pct(r.get("total_return")) if r else "— 缺失"
                row.append(esc(val)); mrow.append(val)
            stress_rows.append(row); md_stress.append(mrow)
    return (table(["时期"]+[LABELS[m]+"收益" for m in MODES], ret_rows),
            table(["时期"]+[LABELS[m]+" 回撤 / 平均暴露 / 执行费用" for m in MODES], risk_rows),
            table(["时期", "情景"]+[LABELS[m]+"净收益" for m in MODES], stress_rows),
            md_ret, md_risk, md_stress)


def derive_headlines(summary):
    results = summary.get("results") or []
    verdicts = summary.get("verdicts") or {}
    lines, md = [], []
    for period, title in [("2026", "2026截至10月13日"), ("2023-2025", "2023–2025 连续期")]:
        baseline = get_result(results, period, "baseline")
        comps = [(mode, get_result(results, period, mode)) for mode in MODES[1:]]
        present = [(m, r) for m, r in comps if r]
        if not baseline or not present:
            text = f"{title}：主比较数据缺失，无法自动概括。"
        else:
            best = max(present, key=lambda item: float(item[1].get("total_return", -float("inf"))))
            delta = float(best[1].get("total_return", 0)) - float(baseline.get("total_return", 0))
            dd_delta = float(best[1].get("max_drawdown", 0)) - float(baseline.get("max_drawdown", 0))
            verdict_text = []
            for mode in MODES[1:]:
                v = verdicts.get(mode, {})
                checks = [c for c in v.get("checks", []) if c.get("period") == period]
                verdict_text.append(f"{LABELS[mode]}验收" + ("通过" if checks and all(truth(c.get("passed")) for c in checks) else "未通过/缺失"))
            text = (f"{title}：按本期主比较实际记录，收益最高的是{LABELS[best[0]]}（{pct(best[1].get('total_return'))}），"
                    f"相对基线收益率差 {number(100*delta)} 个百分点，回撤幅度{'减少' if dd_delta>=0 else '增加'} {number(abs(100*dd_delta))} 个百分点。" + "；".join(verdict_text) + "。")
        lines.append(f"<p>{esc(text)}</p>"); md.append(text)
    return "".join(lines), md


def diagnostic_table(results):
    seeds = [r for r in results if r.get("family") == "seed"]
    rows = []
    for r in sorted(seeds, key=lambda x: (str(x.get("seed")), str(x.get("period")), truth(x.get("stress")), str(x.get("mode")))):
        rows.append([esc(r.get("seed", "—")), esc(r.get("period", "—")), "压力" if truth(r.get("stress")) else "正常",
                     esc(LABELS.get(r.get("mode"), r.get("mode", "—"))), esc(pct(r.get("total_return"))),
                     esc(pct(r.get("max_drawdown"))), esc(money(r.get("final_equity"))),
                     esc(number(r.get("closed_trades"), 0)), esc(pct(r.get("average_exposure"))),
                     esc(money(r.get("execution_costs"))), esc(number(r.get("quality_warning_count"), 0))])
    return table(["种子", "时期", "情景", "模式", "收益", "最大回撤", "期末净值", "平仓数", "平均暴露", "执行费用", "质量警告"], rows) if rows else "<p>没有种子诊断记录。</p>"


def effect_table(effects):
    rows = [[esc(e.get("period", "—")), "压力" if truth(e.get("stress")) else "正常"] +
            [esc(money(e.get(k))) for k in ("filter_without_exit_rmb", "exit_without_filter_rmb", "filter_with_exit_rmb", "exit_with_filter_rmb", "interaction_rmb")]
            for e in effects]
    return table(["时期", "情景", "过滤−基线", "退出−基线", "完整−仅退出", "完整−仅过滤", "交互项"], rows)


def mechanism_notes(summary):
    rows=summary.get('results',[]);notes=[]
    f=get_result(rows,'2026','filter_only');a=get_result(rows,'2026','full_rl');b=get_result(rows,'2026','baseline')
    if f and a and b:
        delta=a['final_equity']-f['final_equity']
        notes.append(f"在已有买入过滤时，加入提前退出使2026期末净值变化{money(delta)}。这是同一组冻结权重在不同动作规则下完整重放的账户差额，包含后续资金和持仓路径变化。")
        notes.append(f"仅过滤的2026收益为{pct(f['total_return'])}，相对V0.8多获利{money(f['final_equity']-b['final_equity'])}；其平仓仅{int(f['closed_trades'])}笔，胜率{pct(f['win_rate'])}。")
    f=get_result(rows,'2023-2025','filter_only');b=get_result(rows,'2023-2025','baseline')
    if f and b:
        gap=100*(abs(f['max_drawdown'])-abs(b['max_drawdown']))
        notes.append(f"仅过滤在2023—2025连续账户收益{pct(f['total_return'])}，基线为{pct(b['total_return'])}；回撤幅度分别{pct(abs(f['max_drawdown']))}、{pct(abs(b['max_drawdown']))}，差{number(gap,4)}个百分点。验收严格使用未四舍五入数值，不按结果放宽回撤条件。")
    seed_rows=[r for r in rows if r.get('family')=='seed' and r.get('period')=='2023-2025' and r.get('mode')=='filter_only' and not r.get('stress')]
    if seed_rows:
        values=[r['total_return'] for r in seed_rows]
        notes.append(f"单种子仅过滤的2023—2025收益范围为{pct(min(values))}至{pct(max(values))}，仍有明显训练随机性；主结论始终使用原先固定的三种子投票。")
    return notes


def generate(run):
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    results = summary.get("results") or []
    ret, risk, stress, md_ret, md_risk, md_stress = main_tables(summary)
    headline_html, headline_md = derive_headlines(summary)
    failures = summary.get("failures") or []
    checks = summary.get("checks") or {}
    verdicts = summary.get("verdicts") or {}
    verdict_rows = []
    for mode in MODES[1:]:
        v = verdicts.get(mode, {})
        for c in v.get("checks", []):
            verdict_rows.append([esc(LABELS[mode]), esc(c.get("period", "—")), esc(c.get("higher_net_return", "—")),
                                 esc(c.get("drawdown_not_worse", "—")), esc(c.get("higher_stressed_return", "—")),
                                 esc(c.get("passed", "—"))])
    links = "<ul>" + "".join(f"<li><a href='{esc(name)}'>{esc(name)}</a></li>" for name in LINK_FILES if (run/name).is_file()) + "</ul>"
    fail_html = "<ul>" + "".join(f"<li>{esc(json.dumps(f, ensure_ascii=False))}</li>" for f in failures) + "</ul>" if failures else "<p>无记录失败。</p>"
    diag = diagnostic_table(results)
    seed_json = esc(json.dumps([r for r in results if r.get("family") == "seed"], ensure_ascii=False, indent=2))
    effects = effect_table(summary.get("effects") or [])
    notes=mechanism_notes(summary)
    passed=[LABELS[m] for m in MODES[1:] if truth(verdicts.get(m,{}).get('passed'))]
    status=('达到历史研究门槛：'+'、'.join(passed)) if passed else '三个RL方案均未通过原定历史研究门槛'
    case_links=[]
    for r in results:
        folder=f"{r['period']}_{r['case']}"
        links_for_case=' · '.join(f"<a href='{esc(folder+'/'+name)}'>{esc(label)}</a>" for name,label in
            [('summary.json','摘要'),('equity.csv','每日净值'),('fills.csv','成交费用'),('trades.csv','平仓'),('raw-decisions.csv','决策及原始投票')])
        case_links.append(f"<li>{esc(folder)}：{links_for_case}</li>")
    md = ["# V0.9 动作拆分消融研究", "", f"- 显示版本：{summary.get('display_version', '—')}",
          f"- 当前采用版本：{summary.get('current_adopted', '—')}（本报告不表示已替换）",
          f"- 运行完整：{yesno(summary.get('complete'))}",f"- 验收：{status}", "", *notes, "", "## 自动提炼的主要时期结论", "", *[f"- {x}" for x in headline_md],
          "", "## 主比较收益（按时期 × 四组）", "", "| 时期 |" + " | ".join(LABELS[m] for m in MODES) + " |",
          "|---|" + "|".join(["---:"]*4) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in md_ret]
    md += ["", "## 2023–2025 与 2026：回撤、暴露、费用", "", "| 时期 |" + " | ".join(LABELS[m] for m in MODES) + " |", "|---|"+"|".join(["---"]*4)+"|"]
    md += ["| " + " | ".join(r) + " |" for r in md_risk]
    md += ["", "## 正常与压力收益", "", "| 时期 | 情景 |"+" | ".join(LABELS[m] for m in MODES)+" |", "|---|---|"+"|".join(["---:"]*4)+"|"]
    md += ["| " + " | ".join(r) + " |" for r in md_stress]
    md += ["", "## 口径与限制", "", "权重固定，按各账户前收盘状态先完整投票，再应用动作 mask。这里是机制拆分，不是分别重训最优单动作。账户仓位、资金和名额仍会相互影响，风险并未匹配。2022 为训练期；2023 以后及 2026 均为已查看历史，不称盲测。V0.8 仍是当前采用版本。",
          "", "## 四组验收", "", "| 模式 | 时期 | 收益更高 | 回撤不更差 | 压力收益更高 | 通过 |", "|---|---|---|---|---|---|"]
    for mode in MODES[1:]:
        for c in verdicts.get(mode, {}).get("checks", []):
            md += ["| " + " | ".join([LABELS[mode], str(c.get("period", "—")), str(c.get("higher_net_return", "—")),
                                       str(c.get("drawdown_not_worse", "—")), str(c.get("higher_stressed_return", "—")),
                                       str(c.get("passed", "—"))]) + " |"]
    md += ["", "## 种子诊断（全部记录）", "", "表内包含全部种子与时期，不筛选种子。", "", "## 交互效应", ""]
    md += ["| 时期 | 情景 | 过滤−基线 | 退出−基线 | 完整−仅退出 | 完整−仅过滤 | 交互项 |", "|---|---|---:|---:|---:|---:|---:|"]
    md += ["| " + " | ".join([str(e.get("period", "—")), "压力" if truth(e.get("stress")) else "正常"] + [money(e.get(k)) for k in ("filter_without_exit_rmb", "exit_without_filter_rmb", "filter_with_exit_rmb", "exit_with_filter_rmb", "interaction_rmb")]) + " |" for e in summary.get("effects", [])]
    md += ["", "## 完整性、失败与来源", "", f"完整性 / 哈希检查：`{json.dumps(checks, ensure_ascii=False)}`", "", f"失败记录：`{json.dumps(failures, ensure_ascii=False)}`", ""]
    md += [f"- [{name}]({name})" for name in LINK_FILES if (run/name).is_file()]
    md += ["", "### 全部种子诊断数值", "", "种子 | 时期 | 情景 | 模式 | 收益 | 回撤 | 净值 | 平仓 | 暴露 | 费用 | 警告", "---|---|---|---|---:|---:|---:|---:|---:|---:|---:"]
    for r in sorted([x for x in results if x.get("family") == "seed"], key=lambda x: (str(x.get("seed")),str(x.get("period")),truth(x.get("stress")),str(x.get("mode")))):
        md += [" | ".join(map(str, [r.get("seed","—"),r.get("period","—"),"压力" if truth(r.get("stress")) else "正常",LABELS.get(r.get("mode"),r.get("mode","—")),pct(r.get("total_return")),pct(r.get("max_drawdown")),money(r.get("final_equity")),number(r.get("closed_trades"),0),pct(r.get("average_exposure")),money(r.get("execution_costs")),number(r.get("quality_warning_count"),0)]))]
    (run / "research-report.md").write_text("\n".join(md)+"\n", encoding="utf-8")
    verdict_html = table(["模式", "时期", "收益更高", "回撤不更差", "压力收益更高", "通过"], verdict_rows)
    html_doc = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>V0.9 动作拆分消融研究</title><style>
body{{font:15px/1.65 system-ui,'Microsoft YaHei',sans-serif;max-width:1200px;margin:28px auto;padding:0 20px;color:#1d2937;background:#f7f9fc}}h1,h2{{line-height:1.3}}h2{{margin-top:2rem;border-bottom:2px solid #d8e1ec;padding-bottom:.3rem}}.box{{background:#fff6e5;border-left:5px solid #d18c26;padding:12px 17px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;min-width:760px;background:white;margin:.5rem 0 1.2rem}}th,td{{border:1px solid #d8e0e8;padding:7px 9px;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}thead{{background:#eaf0f6}}tbody tr:nth-child(even){{background:#f5f8fb}}a{{color:#075e9b}}details{{margin:1rem 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style></head><body>
<h1>V0.9 动作拆分研究</h1><p><strong>{esc(status)}</strong>。当前采用：{esc(summary.get('current_adopted','—'))}；运行完整：{yesno(summary.get('complete'))}。</p>
<div class='box'>固定权重；各账户使用前收盘状态先完整投票，再应用 mask。该实验拆分动作机制，不是重训各单动作的最优策略。仓位、资金与名额仍有交互，风险并非匹配。2022 是训练期；2023+ 与 2026 都是已查看历史，不称盲测。V0.8 当前未替换。</div>
<h2>本轮发现</h2>{''.join('<p>'+esc(n)+'</p>' for n in notes)}<h2>主比较收益</h2>{ret}
<h2>主要时期结论</h2>{headline_html}<h2>2023–2025 与 2026：回撤、暴露、费用</h2>{risk}<h2>正常与压力结果</h2>{stress}<h2>验收判定</h2>{verdict_html}<h2>账户差额与交互</h2><p>列名表示两组期末净值相减；交互项=完整−仅过滤−仅退出+基线。均包含资金和名额变化的后续影响，不是等风险或逐笔纯因果贡献。</p>{effects}<h2>失败记录</h2>{fail_html}<details><summary>展开完整性与文件哈希检查</summary><pre>{esc(json.dumps(checks,ensure_ascii=False,indent=2))}</pre></details>
<h2>种子诊断（全部披露）</h2><p>共 {sum(1 for x in results if x.get('family')=='seed')} 条种子诊断记录；不挑选种子。</p><details><summary>展开所有种子 × 模式 × 时期 × 情景结果</summary>{diag}<details><summary>原始诊断记录（含动作计数与审计字段）</summary><pre>{seed_json}</pre></details></details><h2>来源文件</h2>{links}<details><summary>展开全部156个账户的账本与决策链接</summary><ul>{''.join(case_links)}</ul></details></body></html>"""
    (run / "report.html").write_text(html_doc, encoding="utf-8")


def cli():
    parser = argparse.ArgumentParser(description="生成 V0.9 固定权重动作消融报告")
    parser.add_argument("--run", type=Path, required=True, help="包含 summary.json 的运行目录")
    args = parser.parse_args()
    run = args.run.resolve()
    if not (run / "summary.json").is_file():
        raise FileNotFoundError(f"缺少实验摘要：{run / 'summary.json'}")
    generate(run)
    print(f"HTML: {run / 'report.html'}")
    print(f"Markdown: {run / 'research-report.md'}")


if __name__ == "__main__":
    cli()
