"""Freeze observation rules after a completed historical mechanism study."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'results/exploration-v2'
OUT = ROOT / 'results/exploration-paper-frozen-v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise SystemExit('Preserve the existing observation freeze')
    payload = json.loads((SOURCE / 'summary.json').read_text(encoding='utf-8'))
    OUT.mkdir()
    shutil.copytree(SOURCE / 'sources', OUT / 'sources')
    model = ROOT / 'results/midterm-access-v3/models/h10/2026-lgb.txt'
    shutil.copyfile(model, OUT / '2026-lgb.txt')
    shutil.copyfile(ROOT / 'results/midterm-access-v3/sources/ml_features.py', OUT / 'sources/ml_features-frozen.py')
    shutil.copyfile(ROOT / 'exploration-paper-plan.md', OUT / 'observation-plan.md')
    record = {'frozen_utc': datetime.now(timezone.utc).isoformat(), 'selected_after_viewing_2022_2026': True,
        'has_future_prediction_record': False, 'model_sha256': digest(model),
        'policies': {'base_lgb10': {'hold_days': 10, 'threshold_predicted_net_return': .005},
            'breadth_lgb10': {'hold_days': 10, 'threshold_predicted_net_return': .005, 'breadth5_min': .5, 'known_stock_min': 100},
            'weekly_lowvol': {'hold_days': 20, 'selection_weekday': 4, 'volatility_valid_bars': 60, 'standard_deviation_ddof': 0}},
        'source_summary_sha256': digest(SOURCE / 'summary.json'), 'source_hashes': payload['hashes'],
        'source_features_sha256': payload['metadata']['source_features_sha256'], 'no_live_orders': True,
        'no_automation': True, 'legacy_source_copy': 'sources/ml_features-frozen.py comes from original frozen feature snapshot'}
    (OUT / 'freeze.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(f'FREEZE {OUT / "freeze.json"}')


if __name__ == '__main__':
    main()
