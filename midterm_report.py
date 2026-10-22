"""Render an offline Chinese report for the exploratory midterm-hold study."""
from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path
from statistics import median
from typing import Any


INITIAL_CAPITAL = 30_000
LEDGER_FILES = ("equity.csv", "trades.csv", "fills.csv", "actions.csv")
GROUPS = {
    "model": "主模型",
    "random": "随机基准",
    "simple": "简单对照",
    "matched_random": "配对随机",
    "stress": "压力情景",
    "cash": "现金基准",
}
METRICS = (
    ("累计净收益", "total_return", "pct"),
    ("最大回撤", "max_drawdown", "pct"),
    ("胜率", "win_rate", "pct"),
    ("盈利因子", "profit_factor", "num"),
    ("已平仓笔数", "closed_trades", "num"),
)


def _number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, digits: int = 2) -> str:
    number = _number(value)
    return "—" if number is None else f"{number:,.{digits}f}"


def _pct(value: Any) -> str:
    number = _number(value)
    return "—" if number is None else f"{number * 100:.2f}%"


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _json(value: Any) -> str:
    return html.escape(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _warning_count(item: dict[str, Any]) -> int:
    value = _number(item.get("quality_warning_count"))
    return max(0, int(value or 0))


def _safe_case_dir(output: Path, case: Any) -> Path | None:
    """Resolve a case folder only when it remains below output."""
    raw = str(case or "").replace("\\", "/")
    parts = raw.split("/")
    if not raw or raw.startswith("/") or any(part in ("", ".", "..") for part in parts):
        return None
    root = output.resolve()
    candidate = (output.joinpath(*parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _links(output: Path, case: Any) -> str:
    folder = _safe_case_dir(output, case)
    if folder is None:
        return "—"
    links = []
    rel = "/".join(str(case).replace("\\", "/").split("/"))
    for filename in LEDGER_FILES:
        if (folder / filename).is_file():
            links.append(f'<a href="{_esc(rel + "/" + filename)}">{filename}</a>')
    return " · ".join(links) or "—"


def _family(item: dict[str, Any]) -> str:
    text = " ".join(str(item.get(key, "")) for key in ("case", "label")).lower()
    if any(token in text for token in ("logistic", "逻辑回归")) or re.search(r"(^|[^a-z])lr([^a-z]|$)", text):
        return "LR"
    if any(token in text for token in ("lightgbm", "lgb", "gbm")):
        return "LGB"
    return "未标明模型"


def _table(headers: list[str], rows: list[str]) -> str:
    return ('<div class="table-wrap"><table><thead><tr>' +
            "".join(f"<th>{h}</th>" for h in headers) +
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def _result_rows(output: Path, results: list[dict[str, Any]]) -> list[str]:
    rows = []
    for item in results:
        warning = _warning_count(item)
        row_class = ' class="warning-row"' if warning else ""
        group = GROUPS.get(str(item.get("group")), item.get("group", "—"))
        actual_hold = _fmt(item.get("median_holding_days"))
        planned = _fmt(item.get("hold_days"), 0)
        rows.append(
            f"<tr{row_class}><td>{_esc(group)}</td><td>{_esc(item.get('case'))}</td>"
            f"<td>{_esc(item.get('label'))}</td><td>{_esc(item.get('period'))}</td>"
            f"<td>{planned}</td><td>{_pct(item.get('total_return'))}</td>"
            f"<td>{_pct(item.get('max_drawdown'))}</td><td>{_pct(item.get('win_rate'))}</td>"
            f"<td>{_fmt(item.get('profit_factor'))}</td><td>{_fmt(item.get('closed_trades'), 0)}</td>"
            f"<td>{_fmt(item.get('execution_costs'))}</td>"
            f"<td>{_pct((_number(item.get('execution_costs')) or 0) / INITIAL_CAPITAL)}</td>"
            f"<td>{_pct(item.get('avg_exposure'))}</td><td>{actual_hold} 天</td>"
            f"<td>{_fmt(item.get('mean_holding_days'))} 天</td>"
            f"<td>{_fmt(item.get('signal_dates'), 0)}</td><td>{_fmt(item.get('buy_fills'), 0)}</td>"
            f"<td>{_warning_count(item)}</td><td>{_fmt(item.get('open_positions'), 0)}</td>"
            f"<td class=\"links\">{_links(output, item.get('case'))}</td></tr>")
    return rows


def _stats(values: list[dict[str, Any]], key: str, kind: str) -> tuple[str, str, str]:
    numbers = [n for item in values if (n := _number(item.get(key))) is not None]
    if not numbers:
        return ("—", "—", "—")
    render = _pct if kind == "pct" else _fmt
    return render(median(numbers)), render(min(numbers)), render(max(numbers))


def _distribution(results: list[dict[str, Any]], group: str, family: str | None = None) -> str:
    chosen = [item for item in results if item.get("group") == group and
              (family is None or _family(item) == family)]
    keys = sorted({(str(item.get("period", "—")), _fmt(item.get("hold_days"), 0)) for item in chosen})
    if not keys:
        return "<p class=\"muted\">无此类运行。</p>"
    rows = []
    for period, hold in keys:
        subset = [item for item in chosen if str(item.get("period", "—")) == period and
                  _fmt(item.get("hold_days"), 0) == hold]
        clean = [item for item in subset if _warning_count(item) == 0]
        for label, group_rows in (("全体", subset), ("无警告", clean)):
            cells = [f"<td>{_esc(period)}</td><td>{hold} 天</td><td>{label}</td><td>{len(group_rows)}</td>"]
            for _, key, kind in METRICS:
                med, low, high = _stats(group_rows, key, kind)
                cells.append(f"<td>{med}</td><td>{low}</td><td>{high}</td>")
            rows.append("<tr>" + "".join(cells) + "</tr>")
    headers = ["区间", "计划持有", "警告口径", "运行数"]
    for label, _, _ in METRICS:
        headers.extend((label + "中位数", "最小", "最大"))
    return _table(headers, rows)


def _detail_list(items: Any) -> str:
    if not items:
        return "<p class=\"muted\">无记录。</p>"
    if isinstance(items, dict):
        items = [{"项目": key, "内容": value} for key, value in items.items()]
    if not isinstance(items, list):
        items = [items]
    return "<ul>" + "".join(f"<li><pre>{_json(item)}</pre></li>" for item in items) + "</ul>"


def _annual_models(results: list[dict[str, Any]]) -> str:
    rows = []
    for item in results:
        if item.get("group") != "model" or not item.get("yearly"):
            continue
        yearly = item.get("yearly")
        entries = yearly.items() if isinstance(yearly, dict) else ((None, entry) for entry in yearly)
        for year_key, entry in entries:
            if not isinstance(entry, dict):
                continue
            year = entry.get("year", year_key if year_key is not None else "—")
            ret = entry.get("total_return", entry.get("return"))
            rows.append("<tr>" + "".join(f"<td>{v}</td>" for v in (
                _esc(item.get("case")), _esc(item.get("label")), _esc(year), _pct(ret),
                _pct(entry.get("max_drawdown")), _pct(entry.get("win_rate")),
                _fmt(entry.get("closed_trades", entry.get("trades")), 0), _fmt(entry.get("execution_costs")))) + "</tr>")
    if not rows:
        return "<p class=\"muted\">payload 未提供主模型 yearly 年度账本汇总。</p>"
    return _table(["案例", "说明", "年份", "净收益", "最大回撤", "胜率", "已平仓", "费用"], rows)


def make_report(output: Path, payload: dict) -> Path:
    """Write ``midterm-report.html`` beneath output and return its path."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = [item for item in (payload.get("results") or []) if isinstance(item, dict)]
    protocol = payload.get("protocol") or {}
    metadata = payload.get("metadata") or {}
    hashes = payload.get("hashes") or {}
    periods = sorted({str(item.get("period", "—")) for item in results})
    holds = sorted({_fmt(item.get("hold_days"), 0) for item in results})
    model_items = [item for item in results if item.get("group") == "model"]
    model_cards = "".join(
        f"<li><strong>{_esc(item.get('label', item.get('case')))}</strong>："
        f"{_esc(item.get('period'))}、计划 { _fmt(item.get('hold_days'), 0)} 天，净收益 {_pct(item.get('total_return'))}，"
        f"最大回撤 {_pct(item.get('max_drawdown'))}，胜率 {_pct(item.get('win_rate'))}，"
        f"平仓 {_fmt(item.get('closed_trades'), 0)} 笔，警告 {_warning_count(item)} 条。</li>"
        for item in model_items)
    if not model_cards:
        model_cards = "<li>payload 未提供主模型结果。</li>"
    comparison = _table(
        ["类别", "案例", "说明", "区间", "计划持有", "累计净收益", "最大回撤", "胜率", "PF",
         "已平仓", "费用(元)", "费用/3万元", "平均暴露", "实际持有中位数", "实际持有均值",
         "信号日", "买入成交", "质量警告", "未平仓", "账本"], _result_rows(output, results))
    randoms = _distribution(results, "random")
    paired_lr = _distribution(results, "matched_random", "LR")
    paired_lgb = _distribution(results, "matched_random", "LGB")
    other_groups = []
    for group in ("simple", "stress", "cash"):
        subset = [item for item in results if item.get("group") == group]
        if subset:
            other_groups.append(f"<h3>{GROUPS[group]}</h3>{_table(['案例', '说明', '区间', '持有期', '累计净收益', '最大回撤', '胜率', '费用', '警告'], [
                '<tr>' + ''.join(f'<td>{v}</td>' for v in (
                    _esc(x.get('case')), _esc(x.get('label')), _esc(x.get('period')),
                    _fmt(x.get('hold_days'), 0), _pct(x.get('total_return')), _pct(x.get('max_drawdown')),
                    _pct(x.get('win_rate')), _fmt(x.get('execution_costs')), str(_warning_count(x)))) + '</tr>'
                for x in subset])}")
    audits = _detail_list(payload.get("label_audits"))
    training = _detail_list(payload.get("training"))
    style = """
:root{font-family:system-ui,"Microsoft YaHei",sans-serif;color:#1d2733;background:#f3f6f8;line-height:1.6}
body{max-width:1480px;margin:0 auto;padding:22px}h1{font-size:1.8rem;margin:.2em 0}h2{font-size:1.25rem;margin:28px 0 8px}h3{font-size:1.05rem}
.lead,.notice,.card{background:#fff;border:1px solid #dce2e8;border-radius:10px;padding:15px;margin:12px 0}.notice{border-left:5px solid #b7791f;background:#fffaf0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}.table-wrap{overflow-x:auto;max-width:100%;background:#fff;border:1px solid #dce2e8;border-radius:8px;margin:10px 0 18px;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:.88rem;white-space:nowrap}th,td{padding:8px 10px;border-bottom:1px solid #e5e9ed;text-align:right;vertical-align:top}th{background:#edf2f6;position:sticky;top:0}td:first-child,th:first-child,td:nth-child(2),th:nth-child(2),td:nth-child(3),th:nth-child(3){text-align:left}
.warning-row{background:#fff0ef}.links{min-width:230px}a{color:#145da0}.muted{color:#596675}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f3f5;padding:10px;border-radius:6px;font-size:.82rem}
@media(max-width:650px){body{padding:10px}h1{font-size:1.4rem}.table-wrap{width:100%}}
"""
    html_text = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>A股中短线持有期净收益探索研究</title><style>{style}</style></head><body>
<h1>A股中短线持有期净收益探索研究</h1>
<p class="muted">区间：{_esc("、".join(periods) or "—")} · 计划持有：{_esc("、".join(holds) or "—")} 天</p>
<div class="notice"><strong>研究边界：</strong>2022–2026 年数据均已查看。本报告属于探索性比较，不是新的独立验证；历史结果不代表未来收益或实盘表现，也不构成达到 60% 胜率的声明。所有案例逐项保留，质量警告单独标出。</div>
<section class="card"><h2>主模型结果</h2><ul>{model_cards}</ul><p>重点比较净收益、回撤、盈利因子、交易数、费用及质量警告。不能只按胜率判断。</p></section>
<div class="card"><strong>现金基准：</strong>不持仓、无交易费用，累计收益为 0.00%；用于对照资金闲置的结果。</div>
<h2>全部运行逐项对照</h2><p class="muted">浅红行存在质量警告。计划持有期与实际持有中位数、均值并列；实际持有时间会受提前退出、延迟成交和期末未平仓等影响。费用比例按 30,000 元初始本金计算。</p>
{comparison}
<h2>随机基准分布：区间 × 计划持有期</h2><p>每格报告中位数、最小值和最大值；“全体”包含所有运行，“无警告”仅纳入质量警告数为零的运行。样本数列帮助判断分布规模。</p>{randoms}
<h2>配对随机基准：按模型家族</h2><p>配对随机结果分别按 LR 与 LGB 案例标识归组，再按区间和计划持有期汇总；全体及无警告统计口径与随机基准相同。</p>
<h3>LR 配对随机</h3>{paired_lr}<h3>LGB 配对随机</h3>{paired_lgb}
{''.join(other_groups)}
<h2>主模型年度账户结果</h2><p>只展示 payload 中提供的 yearly 汇总；该年度结果仍属于已经查看过的历史区间。</p>{_annual_models(results)}
<h2>标签审计</h2>{audits}<h2>训练记录</h2>{training}
<h2>协议与元数据</h2><div class="grid"><div class="card"><strong>协议</strong><pre>{_json(protocol)}</pre></div>
<div class="card"><strong>元数据</strong><pre>{_json(metadata)}</pre></div><div class="card"><strong>文件哈希</strong><pre>{_json(hashes)}</pre></div></div>
<p class="muted">账本链接只在对应 case 目录内的 equity.csv、trades.csv、fills.csv、actions.csv 文件实际存在时显示。本报告离线生成，不加载外部资源。</p>
</body></html>'''
    report_path = output / "midterm-report.html"
    report_path.write_text(html_text, encoding="utf-8")
    return report_path
