"""Save one research candidate after exploration; never claim an unseen test."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'results/midterm-access-v3'
OUTPUT = ROOT / 'results/midterm-paper-frozen-v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():
        raise SystemExit('Preserve the existing candidate freeze; choose a new revision explicitly')
    summary = json.loads((SOURCE / 'summary.json').read_text(encoding='utf-8'))
    training = next(t for t in summary['training'] if t['hold_days'] == 10 and t['year'] == 2026)
    model_case = next(r for r in summary['results'] if r['case'] == 'h10_2026_lgb_selective')
    assert training['train_label_end_max'] < '2026-01-01'
    assert model_case['quality_warning_count'] == 0
    OUTPUT.mkdir()
    model = SOURCE / 'models/h10/2026-lgb.txt'
    shutil.copyfile(model, OUTPUT / 'model.txt')
    shutil.copytree(SOURCE / 'sources', OUTPUT / 'sources')
    shutil.copyfile(ROOT / 'midterm-paper-plan.md', OUTPUT / 'observation-plan.md')
    payload = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'purpose': 'future paper observation candidate only',
        'selection_used_viewed_2022_2026_results': True, 'has_prospective_prediction_record': False,
        'model': 'LightGBM return regressor, 10-session target', 'threshold_predicted_net_return': .005,
        'model_sha256': digest(model), 'training': training, 'config': model_case['config'],
        'feature_names': summary['metadata']['features'], 'source_summary_sha256': digest(SOURCE / 'summary.json'),
        'source_hashes': summary['hashes'], 'no_live_orders': True, 'no_scheduled_job': True}
    (OUTPUT / 'freeze.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'output': str(OUTPUT), 'model_sha256': digest(model)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
