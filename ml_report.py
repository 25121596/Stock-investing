"""Render an offline Chinese research report for rolling A-share ML experiments."""
from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path, PurePosixPath
from statistics import mean, median
from typing import Any


CASE_FILES = ("trades.csv", "fills.csv", "equity.csv", "holdings.csv", "warnings.csv")
GROUP_LABELS = {
    "model": "模型",
    "random": "随机基准",
    "simple": "简单对照",
    "legacy": "原规则",
    "matched_random": "匹配随机基准",
}
MODEL_CASES = ("logistic_rank", "logistic_selective", "lgb_rank", "lgb_selective")
PERIODS = ("2022", "2023", "2024", "2025")


def _num(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
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


def _json_block(value: Any) -> str:
    return html.escape(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _case_href(case: Any, filename: str) -> str | None:
    """Return a relative link for known ledger files, rejecting traversal."""
    normalized = str(case).replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    if not normalized or normalized.startswith("/") or any(part in ("", ".", "..") for part in parts):
        return None
    return "/".join((*parts, filename))


def _case_links(case: Any) -> str:
    links = []
    for filename in CASE_FILES:
        href = _case_href(case, filename)
        if href:
            links.append(f'<a href="{_esc(href)}">{_esc(filename)}</a>')
    return " · ".join(links)


def _stat(values: list[Any], kind: str) -> dict[str, str]:
    numbers = [n for value in values if (n := _num(value)) is not None]
    if not numbers:
        return {key: "—" for key in ("mean", "median", "min", "max")}
    render = _pct if kind == "pct" else _fmt
    return {"mean": render(mean(numbers)), "median": render(median(numbers)),
            "min": render(min(numbers)), "max": render(max(numbers))}


def _comparison_rows(results: list[dict[str, Any]]) -> str:
    rows = []
    for item in results:
        case = _esc(item.get("case", ""))
        group = _esc(GROUP_LABELS.get(str(item.get("group", "")), item.get("group", "—")))
        warning_count = int(_num(item.get("quality_warning_count")) or 0)
        cls = " class=\"warning-row\"" if warning_count else ""
        rows.append(
            f"<tr{cls}><td>{group}</td><td>{case}</td><td>{_esc(item.get('label', ''))}</td>"
            f"<td>{_pct(item.get('total_return'))}</td><td>{_pct(item.get('max_drawdown'))}</td>"
            f"<td>{_pct(item.get('win_rate'))}</td><td>{_fmt(item.get('profit_factor'))}</td>"
            f"<td>{_fmt(item.get('closed_trades'), 0)}</td><td>{_fmt(item.get('execution_costs'))}</td>"
            f"<td>{_pct(item.get('avg_exposure'))}</td><td>{_fmt(item.get('invested_days'), 0)}</td>"
            f"<td>{_fmt(item.get('signal_dates'), 0)}</td><td>{_fmt(item.get('buy_fills'), 0)}</td>"
            f"<td>{warning_count}</td><td class=\"links\">{_case_links(item.get('case', ''))}</td></tr>"
        )
    return "".join(rows)


def _distribution_table(randoms: list[dict[str, Any]], clean_randoms: list[dict[str, Any]]) -> str:
    measures = (("累计收益", "total_return", "pct"), ("最大回撤", "max_drawdown", "pct"),
                ("胜率", "win_rate", "pct"), ("盈利因子", "profit_factor", "num"),
                ("平仓数", "closed_trades", "num"))
    rows = []
    for label, key, kind in measures:
        all_stats = _stat([item.get(key) for item in randoms], kind)
        clean_stats = _stat([item.get(key) for item in clean_randoms], kind)
        rows.append("<tr><th>" + label + "</th>" + "".join(
            f"<td>{all_stats[stat]}</td>" for stat in ("mean", "median", "min", "max")) + "".join(
            f"<td>{clean_stats[stat]}</td>" for stat in ("mean", "median", "min", "max")) + "</tr>")
    return "".join(rows)


def _annual_table(results: list[dict[str, Any]]) -> str:
    rows = []
    for item in results:
        years = item.get("yearly") or []
        by_year = {str(entry.get("year")): entry for entry in years if isinstance(entry, dict)}
        cells = []
        for year in range(2022, 2026):
            entry = by_year.get(str(year), {})
            cells.append(f"<td>{_pct(entry.get('return'))}</td><td>{_fmt(entry.get('trades'), 0)}</td>")
        rows.append(f"<tr><td>{_esc(GROUP_LABELS.get(str(item.get('group')), item.get('group', '—')))}</td>"
                    f"<td>{_esc(item.get('case', ''))}</td><td>{_esc(item.get('label', ''))}</td>"
                    + "".join(cells) + "</tr>")
    return "".join(rows)


def _period_table(results: list[dict[str, Any]]) -> str:
    rows = []
    for item in results:
        periods = item.get("periods") or {}
        cells = []
        for year in PERIODS:
            summary = periods.get(year) or {}
            cells.append(f"<td>{_pct(summary.get('total_return'))}</td>"
                         f"<td>{_pct(summary.get('max_drawdown'))}</td>"
                         f"<td>{_pct(summary.get('win_rate'))}</td>"
                         f"<td>{_fmt(summary.get('closed_trades'), 0)}</td>")
        rows.append(f"<tr><td>{_esc(GROUP_LABELS.get(str(item.get('group')), item.get('group', '—')))}</td>"
                    f"<td>{_esc(item.get('case', ''))}</td><td>{_esc(item.get('label', ''))}</td>"
                    + "".join(cells) + "</tr>")
    return "".join(rows)


def _training_table(folds: list[dict[str, Any]]) -> str:
    rows = []
    for fold in folds:
        rows.append("<tr>" + "".join(f"<td>{_esc(fold.get(key, '—'))}</td>" for key in (
            "year", "train_rows", "train_date_max", "train_label_end_max", "eval_rows",
            "logistic_auc", "logistic_brier", "lgb_mse", "lgb_rank_ic")) + "</tr>")
    return "".join(rows)


def _period_comparison_table(results: list[dict[str, Any]]) -> str:
    return _period_table(results)


def make_report(output: Path, payload: dict) -> Path:
    """Write comparison.csv and an offline HTML report; return the HTML path.

    `output` is the experiment result directory. Case ledger folders and the
    three optional metadata documents are expected beneath it.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = list(payload.get("results") or [])
    models = [item for item in results if item.get("group") == "model"]
    randoms = [item for item in results if item.get("group") == "random"]
    clean_randoms = [item for item in randoms if not int(_num(item.get("quality_warning_count")) or 0)]

    # A flat CSV preserves every case, yearly outcome, period summary, and key protocol metric.
    fieldnames = ["case", "label", "group", "total_return", "max_drawdown", "win_rate", "profit_factor",
                  "closed_trades", "execution_costs", "quality_warning_count", "open_positions",
                  "avg_exposure", "invested_days", "signal_dates", "buy_fills"]
    fieldnames.extend(f"{year}_{metric}" for year in range(2020, 2026) for metric in ("return", "trades"))
    fieldnames.extend(f"{year}_{metric}" for year in PERIODS for metric in
                      ("total_return", "max_drawdown", "win_rate", "closed_trades"))
    csv_path = output / "comparison.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for item in results:
            row = {key: item.get(key) for key in fieldnames if key in item}
            row.update({key: item.get(key) for key in
                        ("case", "label", "group", "total_return", "max_drawdown", "win_rate", "profit_factor",
                         "closed_trades", "execution_costs", "quality_warning_count", "open_positions",
                         "avg_exposure", "invested_days", "signal_dates", "buy_fills")})
            by_year = {str(entry.get("year")): entry for entry in (item.get("yearly") or [])
                       if isinstance(entry, dict)}
            for year in range(2020, 2026):
                entry = by_year.get(str(year), {})
                row[f"{year}_return"] = entry.get("return")
                row[f"{year}_trades"] = entry.get("trades")
            for year in PERIODS:
                summary = (item.get("periods") or {}).get(year) or {}
                for metric in ("total_return", "max_drawdown", "win_rate", "closed_trades"):
                    row[f"{year}_{metric}"] = summary.get(metric)
            writer.writerow(row)

    model_rows = []
    for case in MODEL_CASES:
        matches = [item for item in models if str(item.get("case", "")).startswith(case)]
        for item in matches:
            model_rows.append(f"<li><strong>{_esc(item.get('label', item.get('case')))}</strong>："
                              f"收益 {_pct(item.get('total_return'))}，最大回撤 {_pct(item.get('max_drawdown'))}，"
                              f"胜率 {_pct(item.get('win_rate'))}，PF {_fmt(item.get('profit_factor'))}，"
                              f"平仓 {_fmt(item.get('closed_trades'), 0)} 笔，质量警告 "
                              f"{int(_num(item.get('quality_warning_count')) or 0)} 条。</li>")
    if not model_rows:
        model_rows = [f"<li><strong>{_esc(item.get('case', ''))}</strong>：收益 {_pct(item.get('total_return'))}，"
                      f"最大回撤 {_pct(item.get('max_drawdown'))}，胜率 {_pct(item.get('win_rate'))}，"
                      f"PF {_fmt(item.get('profit_factor'))}，平仓 {_fmt(item.get('closed_trades'), 0)} 笔。</li>"
                      for item in models]

    random_details = []
    for item in randoms:
        warning_count = int(_num(item.get("quality_warning_count")) or 0)
        cls = " class=\"warning-row\"" if warning_count else ""
        random_details.append(
            f"<tr{cls}><td>{_esc(item.get('case', ''))}</td><td>{_esc(item.get('label', ''))}</td>"
            f"<td>{_pct(item.get('total_return'))}</td><td>{_pct(item.get('max_drawdown'))}</td>"
            f"<td>{_pct(item.get('win_rate'))}</td><td>{_fmt(item.get('profit_factor'))}</td>"
            f"<td>{_fmt(item.get('closed_trades'), 0)}</td><td>{warning_count}</td>"
            f"<td class=\"links\">{_case_links(item.get('case', ''))}</td></tr>")

    features = payload.get("features") or []
    if isinstance(features, dict):
        feature_items = [f"<li><code>{_esc(key)}</code>：{_esc(value)}</li>" for key, value in features.items()]
    else:
        feature_items = [f"<li><code>{_esc(value)}</code></li>" for value in features]

    doc_links = []
    for filename, label in (("summary.json", "summary.json"), ("protocol.md", "protocol.md"),
                            ("label_audit.json", "label_audit.json")):
        doc_links.append(f'<a href="{filename}">{label}</a>')
    hashes = payload.get("hashes", {})
    version = payload.get("version", "—")
    protocol = payload.get("protocol", {})
    panel_meta = payload.get("panel_meta", {})

    html_text = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>A股短周期滚动模型研究</title>
<style>
:root{{font-family:system-ui,"Microsoft YaHei",sans-serif;color:#1d2733;background:#f4f6f8;line-height:1.6}}
body{{max-width:1440px;margin:0 auto;padding:24px}}h1{{font-size:1.9rem;margin:0}}h2{{font-size:1.25rem;margin:28px 0 10px}}h3{{font-size:1.05rem;margin:20px 0 8px}}
.lead,.notice,.card{{background:#fff;border:1px solid #dce2e8;border-radius:10px;padding:16px;margin:14px 0}}
.notice{{border-left:5px solid #b7791f;background:#fffaf0}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}}
.table-wrap{{overflow-x:auto;background:#fff;border:1px solid #dce2e8;border-radius:8px;margin:10px 0 18px}}
table{{border-collapse:collapse;width:100%;font-size:.9rem;white-space:nowrap}}th,td{{padding:8px 10px;border-bottom:1px solid #e5e9ed;text-align:right;vertical-align:top}}th{{background:#edf2f6;position:sticky;top:0}}td:first-child,th:first-child,td:nth-child(2),th:nth-child(2),td:nth-child(3),th:nth-child(3){{text-align:left}}
.warning-row{{background:#fff1f0}}.links{{font-size:.82rem;min-width:400px}}a{{color:#145da0}}small,.muted{{color:#596675}}code,pre{{font-family:Consolas,monospace}}pre{{overflow:auto;background:#f1f3f5;padding:12px;border-radius:6px;font-size:.82rem}}
@media(max-width:650px){{body{{padding:12px}}h1{{font-size:1.45rem}}}}
</style></head><body>
<h1>A股短周期滚动预测研究</h1>
<p class="muted">结果目录：{_esc(output.name)} · 版本：{_esc(version)} · {" · ".join(doc_links)}</p>
<div class="notice"><strong>样本边界：</strong>本项目使用的2020–2025年已经查看过。按年份滚动训练和评估可减少时间泄漏，但这些年份不是全新未见时期；本报告属于探索性证据，不构成未来盈利或实盘可用的证明。</div>
<section class="card"><h2>四个模型版本</h2><ul>{''.join(model_rows)}</ul>
<p>四个版本按同一2022–2025滚动评估窗口并列；每年表现和随机分布完整保留。Logistic选择版采用55%阈值，回归选择版采用预测净收益0.5%阈值；两者按固定3个交易日持有。具体触发及执行约定见 protocol.md。</p></section>
<h2>全案例对照</h2>
<p class="muted">模型、10次随机、简单对照、原规则及匹配随机全部逐项列出。质量警告行以浅红色标记；警告案例仍保留在比较中，收益不能视为已确认有效。</p>
<div class="table-wrap"><table><thead><tr><th>类别</th><th>案例</th><th>说明</th><th>累计收益</th><th>最大回撤</th><th>胜率</th><th>PF</th><th>平仓笔数</th><th>费用</th><th>平均仓位暴露</th><th>持仓日</th><th>信号日</th><th>买入数</th><th>警告</th><th>逐笔数据</th></tr></thead><tbody>{_comparison_rows(results)}</tbody></table></div>
<h2>随机基准分布</h2>
<p>展示全部 {_fmt(len(randoms),0)} 次随机运行，同时单独给出无质量警告子集的描述统计。主表中的每一次随机结果（包含有警告案例）均保留在下表。</p>
<div class="table-wrap"><table><thead><tr><th rowspan="2">指标</th><th colspan="4">全部随机运行</th><th colspan="4">无警告子集</th></tr><tr><th>均值</th><th>中位数</th><th>最小</th><th>最大</th><th>均值</th><th>中位数</th><th>最小</th><th>最大</th></tr></thead><tbody>{_distribution_table(randoms,clean_randoms)}</tbody></table></div>
<div class="table-wrap"><table><thead><tr><th>案例</th><th>说明</th><th>累计收益</th><th>最大回撤</th><th>胜率</th><th>PF</th><th>平仓笔数</th><th>警告数</th><th>逐笔数据</th></tr></thead><tbody>{''.join(random_details)}</tbody></table></div>
<h2>连续账户的年度收益（2022–2025）</h2>
<p>2020–2021用于起始训练；连续账户从2022年起运行，逐年展示收益和已平仓笔数，包含所有对照。年份已被查看过，不能把表现较好的年份当作新验证。</p>
<div class="table-wrap"><table><thead><tr><th rowspan="2">类别</th><th rowspan="2">案例</th><th rowspan="2">说明</th>{''.join(f'<th colspan="2">{year}</th>' for year in range(2022,2026))}</tr><tr>{''.join('<th>收益</th><th>笔数</th>' for _ in range(4))}</tr></thead><tbody>{_annual_table(results)}</tbody></table></div>
<h2>相同2022–2025滚动段比较</h2>
<p>每年独立以3万元重新起账，列出累计收益、最大回撤、胜率和交易数；训练与评估边界见训练折表。各年度仍属于已查看历史期，结果与上方连续账户年度收益的口径不同。</p>
<div class="table-wrap"><table><thead><tr><th rowspan="2">类别</th><th rowspan="2">案例</th><th rowspan="2">说明</th>{''.join(f'<th colspan="4">{year}</th>' for year in PERIODS)}</tr><tr>{''.join('<th>收益</th><th>回撤</th><th>胜率</th><th>笔数</th>' for _ in PERIODS)}</tr></thead><tbody>{_period_comparison_table(results)}</tbody></table></div>
<h2>滚动训练折与时间边界</h2>
<p>核对训练特征/标签的结束时间是否早于评估起点。Logistic AUC、Brier分数、LightGBM MSE和排序相关系数作为模型诊断，不能替代账户净值评估。</p>
<div class="table-wrap"><table><thead><tr><th>评估年</th><th>训练行</th><th>训练特征最晚日期</th><th>训练标签最晚结束日</th><th>评估行</th><th>Logistic AUC</th><th>Logistic Brier</th><th>LightGBM MSE</th><th>LightGBM Rank IC</th></tr></thead><tbody>{_training_table(payload.get('training') or [])}</tbody></table></div>
<h2>执行与解释边界</h2>
<div class="grid"><div class="card"><strong>匹配随机基准</strong><p>按入场机会日期匹配；它不保证每次运行成交笔数相同。匹配随机案例用“匹配随机基准”分类逐个展示。</p></div>
<div class="card"><strong>盈利判读</strong><p>单看胜率不够。即使某模型净盈利，也需要新的独立时期验证，并同时检查盈利因子、回撤、交易数量、费用敏感性和质量警告。</p></div>
<div class="card"><strong>研究用途</strong><p>当前结果用于比较预测信号和固定执行规则的历史表现，不代表未来收益，不承诺60%胜率，也不构成实盘建议。</p></div></div>
<h2>特征清单（{len(features)}项）</h2><ul>{''.join(feature_items)}</ul>
<h2>面板元数据</h2><pre>{_json_block(panel_meta)}</pre>
<h2>协议、版本和哈希</h2><div class="grid"><div class="card"><strong>协议</strong><pre>{_json_block(protocol)}</pre></div>
<div class="card"><strong>哈希</strong><pre>{_json_block(hashes)}</pre></div></div>
<p class="muted">明细入口：各案例下的 trades.csv、fills.csv、equity.csv、holdings.csv、warnings.csv；全局材料：{" · ".join(doc_links)}。本报告只呈现payload与已有账本，不发起网络请求或模型训练。</p>
</body></html>'''

    report_path = output / "ml-report.html"
    report_path.write_text(html_text, encoding="utf-8")
    return report_path
