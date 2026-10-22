"""Correct a release label and add report guidance; no economic values change."""
from pathlib import Path
import hashlib
import json
import shutil

from midterm_report import make_report

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/midterm-access-v3'


def main():
    source = OUT / 'summary.json'
    raw = OUT / 'summary.raw.json'
    if raw.exists():
        raise SystemExit('Metadata revision already saved')
    shutil.copyfile(source, raw)
    payload = json.loads(raw.read_text(encoding='utf-8'))
    old_results = json.dumps(payload['results'], sort_keys=True)
    payload['version'] = 'midterm-access-v3'
    payload['metadata_revision'] = {
        'reason': 'Correct stale v2 label in v3 round-trip-precision release; no financial or model values changed',
        'raw_summary_sha256': hashlib.sha256(raw.read_bytes()).hexdigest(),
        'execution_sources': 'sources/ retains the actual code used during the run, including its original label'}
    assert old_results == json.dumps(payload['results'], sort_keys=True)
    source.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    report = make_report(OUT, payload)
    content = report.read_text(encoding='utf-8')
    note = ('<div class="notice"><strong>最终结果版本 v3：</strong>已修复退市整理期买入权限并统一浮点缓存精度。'
            '带质量警告的账户收益未确认，不用于确认性比较。'
            '10日选择性收益模型只作为未来模拟候选；2026为25笔、36%胜率，未达到60%。'
            '<a href="../../midterm-assessment.md">阅读简明结论与限制</a>。</div>')
    content = content.replace('<section class="card"><h2>主模型结果</h2>', note + '<section class="card"><h2>主模型结果</h2>', 1)
    report.write_text(content, encoding='utf-8')
    (OUT / 'presentation-revision.json').write_text(json.dumps({
        'metadata_revision': payload['metadata_revision'],
        'report_sha256': hashlib.sha256(report.read_bytes()).hexdigest(),
        'revision_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2), encoding='utf-8')
    print('Metadata label corrected; all economic results preserved')


if __name__ == '__main__':
    main()
