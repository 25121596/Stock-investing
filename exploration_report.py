"""Render the offline HTML report for the other-mechanism exploration."""
from __future__ import annotations

import html
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


INITIAL_CAPITAL = 30_000
POLICIES = (
    "base_lgb10", "breadth_lgb10", "weekly_lowvol",
    "weekly_lowvol_trend", "weekly_risk_momentum",
)
PERIODS = ("2022-2025", "2026")
POLICY_LABELS = {
    "base_lgb10": "基础 LightGBM（10日）",
    "breadth_lgb10": "广度 LightGBM（10日）",
    "weekly_lowvol": "周频低波动",
    "weekly_lowvol_trend": "周频低波动＋趋势",
    "weekly_risk_momentum": "周频风险动量",
}
GROUP_LABELS = {"policy": "政策", "matched_random": "匹配随机", "stress": "压力情景", "cash": "现金"}
METRICS = (
    ("累计净收益", "total_return", "pct"),
    ("最大回撤", "max_drawdown", "pct"),
    ("胜率", "win_rate", "pct"),
    ("盈利因子", "profit_factor", "num"),
    ("已平仓笔数", "closed_trades", "num"),
    ("费用(元)", "execution_costs", "num"),
    ("平均仓位", "avg_exposure", "pct"),
    ("实际持有中位天数", "median_holding_days", "num"),
    ("实际持有均值天数", "mean_holding_days", "num"),
)
LEDGER_FILES = ("equity.csv", "trades.csv", "fills.csv", "actions.csv")


def _num(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, digits: int = 2) -> str:
    number = _num(value)
    return "—" if number is None else f"{number:,.{digits}f}"


def _pct(value: Any) -> str:
    number = _num(value)
    return "—" if number is None else f"{number * 100:.2f}%"


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _json(value: Any) -> str:
    return html.escape(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _warnings(item: dict[str, Any]) -> int:
    return max(0, int(_num(item.get("quality_warning_count")) or 0))


def _table(headers: list[str], rows: list[str]) -> str:
    return ('<div class="table-wrap"><table><thead><tr>' +
            "".join(f"<th>{h}</th>" for h in headers) +
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def _case_dir(output: Path, case: Any) -> Path | None:
    raw = str(case or "").replace("\\", "/")
    parts = raw.split("/")
    if not raw or raw.startswith("/") or any(p in ("", ".", "..") for p in parts):
        return None
    root = output.resolve()
    candidate = output.joinpath(*parts).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _links(output: Path, case: Any) -> str:
    folder = _case_dir(output, case)
    if folder is None:
        return "—"
    href = "/".join(str(case).replace("\\", "/").split("/"))
    links = [f'<a href="{_esc(href + "/" + name)}">{name}</a>'
             for name in LEDGER_FILES if (folder / name).is_file()]
    return " · ".join(links) or "—"


def _metric(value: Any, kind: str) -> str:
    return _pct(value) if kind == "pct" else _fmt(value)


def _random_summary(items: list[dict[str, Any]], policy: str, period: str, clean: bool) -> str:
    chosen = [x for x in items if x.get("group") == "matched_random"
              and x.get("policy") == policy and x.get("period") == period
              and (not clean or _warnings(x) == 0)]
    if not chosen:
        return "n=0（无警告口径同样为 n=0）" if clean else "n=0"
    values = []
    for label, key, kind in METRICS:
        nums = [v for x in chosen if (v := _num(x.get(key))) is not None]
        if nums:
            values.append(f"{label} { _metric(median(nums), kind)} / {_metric(min(nums), kind)} / {_metric(max(nums), kind)}")
    n_text = f"n={len(chosen)}" + (f"（{sum(_warnings(x) > 0 for x in chosen)} 项含警告）" if not clean else "（全为无警告）")
    return n_text + "；" + "；".join(values)


def _policy_comparisons(items: list[dict[str, Any]]) -> str:
    rows = []
    for policy in POLICIES:
        for period in PERIODS:
            primary = next((x for x in items if x.get("group") == "policy" and
                            x.get("policy") == policy and x.get("period") == period), None)
            if primary is None:
                continue
            # The three summaries are intentionally scoped to this exact policy and period.
            all_random = _random_summary(items, policy, period, clean=False)
            clean_random = _random_summary(items, policy, period, clean=True)
            row = [_esc(POLICY_LABELS.get(policy, policy)), _esc(period), _esc(primary.get("label", "主政策")),
                   _pct(primary.get("total_return")), _pct(primary.get("max_drawdown")),
                   _fmt(primary.get("closed_trades"), 0), _esc(all_random), _esc(clean_random)]
            rows.append("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>")
    return _table(["政策", "时期", "主政策标签", "主政策净收益", "主政策最大回撤", "主政策平仓笔数",
                   "匹配随机：全体 median / min / max", "匹配随机：无警告 median / min / max"], rows)


def _base_breadth(items: list[dict[str, Any]]) -> str:
    rows = []
    for period in PERIODS:
        base = next((x for x in items if x.get("group") == "policy" and
                     x.get("policy") == "base_lgb10" and x.get("period") == period), None)
        breadth = next((x for x in items if x.get("group") == "policy" and
                        x.get("policy") == "breadth_lgb10" and x.get("period") == period), None)
        if not base or not breadth:
            continue
        cells = [_esc(period)]
        for label, key, kind in METRICS:
            a, b = _num(base.get(key)), _num(breadth.get(key))
            if a is None or b is None:
                change = "—"
            elif kind == "pct":
                change = _pct(b - a) + "（百分点）"
            else:
                change = _fmt(b - a)
            cells.append(f"{_metric(a, kind)} → {_metric(b, kind)}")
            cells.append(change)
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    headers = ["时期"]
    for label, _, _ in METRICS:
        headers.extend((label + "（基础→广度）", "变化"))
    return _table(headers, rows)


def _all_cases(output: Path, items: list[dict[str, Any]]) -> str:
    rows = []
    for x in items:
        warning = _warnings(x)
        klass = ' class="warning-row"' if warning else ""
        rows.append(f"<tr{klass}>" + "".join((
            f"<td>{_esc(GROUP_LABELS.get(str(x.get('group')), x.get('group', '—')))}</td>",
            f"<td>{_esc(POLICY_LABELS.get(str(x.get('policy')), x.get('policy', '—')))}</td>",
            f"<td>{_esc(x.get('period'))}</td><td>{_esc(x.get('label'))}</td><td>{_esc(x.get('case'))}</td>",
            f"<td>{_esc(x.get('seed', '—'))}</td><td>{_fmt(x.get('hold_days'), 0)}</td>",
            f"<td>{_pct(x.get('total_return'))}</td><td>{_pct(x.get('max_drawdown'))}</td>",
            f"<td>{_pct(x.get('win_rate'))}</td><td>{_fmt(x.get('profit_factor'))}</td>",
            f"<td>{_fmt(x.get('closed_trades'), 0)}</td><td>{_fmt(x.get('execution_costs'))}</td>",
            f"<td>{_pct((_num(x.get('execution_costs')) or 0) / INITIAL_CAPITAL)}</td>",
            f"<td>{_pct(x.get('avg_exposure'))}</td><td>{_fmt(x.get('median_holding_days'))}</td>",
            f"<td>{_fmt(x.get('mean_holding_days'))}</td><td>{_fmt(x.get('signal_dates'), 0)}</td>",
            f"<td>{_fmt(x.get('signal_candidates'), 0)}</td><td>{warning}</td>",
            f"<td>{_fmt(x.get('open_positions'), 0)}</td><td class=\"links\">{_links(output, x.get('case'))}</td>",
        )) + "</tr>")
    return _table(["类别", "政策/匹配键", "时期", "标签", "案例目录", "种子", "计划持有天数", "净收益",
                   "最大回撤", "胜率", "PF", "已平仓笔数", "执行费用(元)", "费用/3万元", "平均仓位",
                   "实际持有中位天数", "实际持有均值天数", "信号日", "候选信号", "质量警告", "未平仓", "账本链接"], rows)


def _annual(items: list[dict[str, Any]]) -> str:
    rows = []
    for item in items:
        if item.get("group") != "policy" or not item.get("yearly"):
            continue
        yearly = item["yearly"]
        entries = yearly.items() if isinstance(yearly, dict) else ((None, x) for x in yearly)
        for year_key, entry in entries:
            if not isinstance(entry, dict):
                continue
            year = entry.get("year", year_key if year_key is not None else "—")
            vals = (item.get("policy"), item.get("period"), item.get("label"), year,
                    entry.get("total_return", entry.get("return")), entry.get("max_drawdown"),
                    entry.get("win_rate"), entry.get("profit_factor"),
                    entry.get("closed_trades", entry.get("trades")), entry.get("execution_costs"))
            rows.append("<tr>" + "".join(f"<td>{v}</td>" for v in (
                _esc(POLICY_LABELS.get(str(vals[0]), vals[0])), _esc(vals[1]), _esc(vals[2]), _esc(vals[3]),
                _pct(vals[4]), _pct(vals[5]), _pct(vals[6]), _fmt(vals[7]), _fmt(vals[8], 0), _fmt(vals[9]))) + "</tr>")
    if not rows:
        return '<p class="muted">payload 未提供主政策 fresh annual summaries。</p>'
    return _table(["政策", "时期", "主政策标签", "年份", "独立年度净收益", "最大回撤", "胜率", "PF", "已平仓", "费用(元)"], rows)


def _details(title: str, value: Any) -> str:
    if not value:
        return f'<details><summary>{_esc(title)}</summary><p class="muted">未提供。</p></details>'
    return f'<details><summary>{_esc(title)}</summary><pre>{_json(value)}</pre></details>'


def make_report(output: Path, payload: dict) -> Path:
    """Write ``exploration-report.html`` beneath output and return its path."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    items = [x for x in (payload.get("results") or []) if isinstance(x, dict)]
    period_counts = {p: sum(str(x.get("period")) == p for x in items) for p in PERIODS}
    policy_counts = {p: sum(x.get("group") == "policy" and x.get("policy") == p for x in items) for p in POLICIES}
    warnings = sum(_warnings(x) > 0 for x in items)
    protocol = payload.get("protocol") or {}
    metadata = payload.get("metadata") or {}
    hashes = payload.get("hashes") or {}
    replication = payload.get("replication") or {}
    html_doc = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>其他机制探索</title><style>
:root{{color-scheme:light;--ink:#172033;--muted:#5a6678;--line:#d9e0e9;--paper:#fff;--wash:#f4f7fb;--accent:#164e63;--warn:#fff3d5}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--wash);color:var(--ink);font:16px/1.6 system-ui,"Segoe UI",sans-serif}}
main{{max-width:1240px;margin:auto;padding:clamp(16px,4vw,42px)}}h1{{font-size:clamp(1.8rem,4vw,2.7rem);line-height:1.2;margin:.2em 0}}h2{{margin:1.8em 0 .5em;font-size:1.35rem}}p{{margin:.6em 0}}.lede,.panel{{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:18px;margin:14px 0}}
.notice{{border-left:4px solid var(--accent)}}.muted{{color:var(--muted)}}.table-wrap{{max-width:100%;overflow-x:auto;overscroll-behavior-inline:contain;border:1px solid var(--line);border-radius:9px;background:var(--paper);margin:12px 0}}
table{{border-collapse:collapse;width:max-content;min-width:100%;font-size:.9rem}}th,td{{padding:9px 11px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;white-space:nowrap}}th{{background:#edf3f7;position:sticky;top:0}}tr:last-child td{{border-bottom:0}}.warning-row{{background:var(--warn)}}.links{{white-space:normal;min-width:150px}}a{{color:#075985}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:12px;border-radius:8px;max-height:520px;overflow:auto}}details{{background:var(--paper);border:1px solid var(--line);border-radius:9px;padding:10px 14px;margin:10px 0}}summary{{cursor:pointer;font-weight:650}}
@media(max-width:600px){{main{{padding:14px}}.lede,.panel{{padding:14px}}}}
</style></head><body><main>
<h1>其他机制探索</h1>
<section class="lede notice"><p>这份报告比较五种政策、匹配随机对照、压力情景和现金基准。2022–2026 已全部用于探索与查看；这不是新的独立检验，也不构成前瞻有效性证据。</p>
<p>ETF 与基本面方向另见可行性材料，尚无收益证明。年度结果是每个年份单独重置账户的结果，不能连乘或拼成连续账户收益。</p></section>
<section class="panel"><strong>数据概况</strong><p>结果 {len(items)} 项；2022–2025 {period_counts['2022-2025']} 项；2026 {period_counts['2026']} 项；含质量警告 {warnings} 项。五政策计数：{_esc(', '.join(f'{POLICY_LABELS[p]} {policy_counts[p]}' for p in POLICIES))}。</p>
<p>费用另列金额及其占 30,000 元模拟本金比例；实际持有天数来自账本字段。匹配随机按同一 policy × period 分开统计，全体与无警告分别呈现。此匹配只代表报告的匹配键，不意味着随机组合复制了实际成交或仓位。</p></section>
<h2>主政策与各自匹配随机对照</h2><p class="muted">每个政策×时期的主政策与其匹配随机单独比较；随机指标依次给出中位数 / 最小值 / 最大值。无警告口径仅剔除有质量警告的随机运行，样本数单列。</p>
{_policy_comparisons(items)}
<h2>基础与广度 LightGBM 的相对变化</h2><p class="muted">仅比较 base_lgb10 与 breadth_lgb10 的同一时期政策结果。百分比指标变化以百分点表示；其它数值指标显示绝对差。</p>{_base_breadth(items)}
<h2>主政策年度独立账户汇总</h2><p class="muted">各年单独启动 30,000 元账户。年度净收益不可连乘，不代表 2022–2025 连续持仓的累计收益。</p>{_annual(items)}
<h2>全部案例账本</h2><p class="muted">52 个案例应全部保留，包括含质量警告的案例；有警告行以浅黄色标出。账本链接仅在相对案例目录下对应文件实际存在时显示。</p>{_all_cases(output, items)}
{_details('实验协议', protocol)}{_details('文件哈希', hashes)}{_details('生成元数据', metadata)}{_details('复现信息', replication)}
<footer class="muted"><p>本报告只描述这些已查看探索的账本结果，不代表收益保证或新的独立验证。解释时应同时考虑样本数、警告、成本、仓位暴露和实际持有期。</p></footer>
</main></body></html>'''
    path = output / "exploration-report.html"
    path.write_text(html_doc, encoding="utf-8")
    return path
