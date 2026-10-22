"""Reproduce legacy training with transparent save-only instrumentation."""
from __future__ import annotations
import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def compare(original, recreated, score):
    old = pd.read_csv(original, dtype={'date': str, 'code': str}, float_precision='round_trip')
    new = pd.read_csv(recreated, dtype={'date': str, 'code': str}, float_precision='round_trip')
    if old.duplicated(['date', 'code']).any() or new.duplicated(['date', 'code']).any():
        raise ValueError('Duplicate prediction keys')
    joined = old[['date', 'code', score]].merge(new[['date', 'code', score]], on=['date', 'code'],
        suffixes=('_old', '_new'), how='outer', indicator=True, validate='one_to_one')
    covered = bool(joined._merge.eq('both').all())
    delta = (joined[score + '_old'] - joined[score + '_new']).abs()
    return {'old_rows': len(old), 'new_rows': len(new), 'keys_equal': covered,
        'max_abs_difference': float(delta.max()), 'mean_abs_difference': float(delta.mean()),
        'within_1e_10': bool(covered and delta.le(1e-10).all()),
        'exact_float_equal_rows': int(delta.eq(0).sum()),
        'correlation': float(joined[score + '_old'].corr(joined[score + '_new']))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(exist_ok=False)
    inputs = ['train_model_v2.py', 'train_model_v3.py', 'train_model_v08.py',
        'results/midterm-access-v3/panel-h10.csv', 'results/midterm-access-v3/features.csv',
        'results/exploration-v1/new-features.csv', 'results/midterm-access-v3/predictions-h10.csv',
        'results/model-v2/predictions-h10-v2.csv', 'results/model-v08/predictions-h10-v08.csv',
        'data/forward-2026-10-13/manifest.json']
    hashes = {name: digest(ROOT / name) for name in inputs}
    (output / 'sources').mkdir()
    for path in ROOT.glob('*.py'):
        shutil.copyfile(path, output / 'sources' / path.name)
    write(output / 'started.json', {'started_utc': datetime.now(timezone.utc).isoformat(),
        'inputs': hashes, 'python': sys.version, 'lightgbm': lgb.__version__,
        'numpy': np.__version__, 'pandas': pd.__version__,
        'scope': 'Save-only instrumentation, same model parameters and row ordering; no tuning'})
    fits, fills = [], []

    def capture(model, year, x, y, train, features, kind):
        folder = output / kind
        folder.mkdir(exist_ok=True)
        path = folder / f'{year}.txt'
        model.booster_.save_model(str(path))
        restored = lgb.Booster(model_file=str(path))
        sample = x[:min(1000, len(x))]
        err = float(np.max(np.abs(model.predict(sample) - restored.predict(sample, num_threads=1))))
        if err > 1e-12 or not train.label_end.lt(f'{year}-01-01').all():
            raise ValueError('Saved model parity or chronological training check failed')
        record = {'kind': kind, 'year': int(year), 'training_rows': len(train),
            'training_last_signal': train.date.max(), 'training_last_label_end': train.label_end.max(),
            'features': list(features), 'model_sha256': digest(path), 'saved_reload_max_error': err,
            'training_x_sha256': hashlib.sha256(np.asarray(x).tobytes()).hexdigest(),
            'training_y_sha256': hashlib.sha256(np.asarray(y).tobytes()).hexdigest(),
            'parameters': model.get_params(), 'note': 'No extra 5-session embargo in legacy code; labels end before test year'}
        fits.append(record)
        write(folder / f'{year}-training.json', record)
        print(f'SAVED {kind} {year}: {len(train)} train rows, reload error {err}', file=sys.__stdout__, flush=True)

    def record_fill(col, values):
        fills.append({'feature': col, 'all_period_missing_rows': int(values.isna().sum()),
                      'scope': 'Legacy all-period median fallback; retain evidence of whether actually used'})

    for name, kind in [('train_model_v2.py', 'v2'), ('train_model_v08.py', 'v08')]:
        source = (ROOT / name).read_text(encoding='utf-8')
        if kind == 'v2':
            needle = 'regressor.fit(X_train, y_train)'
            addition = needle + '\n        capture(regressor, year, X_train, y_train, train_df, FEATURES_V2, "v2")'
            source = source.replace(needle, addition)
            source = source.replace('out_dir = ROOT / "results/model-v2"', 'out_dir = AUDIT_OUTPUT / "v2"')
        else:
            needle = 'ranker.fit(X_train, y_train_rel, group=group_train)'
            addition = needle + '\n        capture(ranker, year, X_train, y_train_rel, train_df, FEATURES_V08, "v08")'
            source = source.replace(needle, addition)
            source = source.replace('panel[col] = panel[col].fillna(', 'record_fill(col, panel[col])\n        panel[col] = panel[col].fillna(')
        if addition not in source:
            raise ValueError('Unexpected original training source; refuse instrumentation')
        instrumented = output / 'sources' / ('instrumented-' + name)
        instrumented.write_text(source, encoding='utf-8')
        env = {'__name__': 'verified_training', '__file__': str(ROOT / name),
            'AUDIT_OUTPUT': output, 'capture': capture, 'record_fill': record_fill}
        exec(compile(source, str(instrumented), 'exec'), env)
        if kind == 'v08':
            env['OUT_DIR'] = output / kind
        with (output / f'{kind}-training.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
            env['main' if kind == 'v2' else 'train_model_v08']()
    verification = {
        'v2': compare(ROOT / 'results/model-v2/predictions-h10-v2.csv', output / 'v2/predictions-h10-v2.csv', 'pred_net_return_v2'),
        'v08': compare(ROOT / 'results/model-v08/predictions-h10-v08.csv', output / 'v08/predictions-h10-v08.csv', 'pred_score_v08')}
    if hashes != {name: digest(ROOT / name) for name in inputs}:
        raise ValueError('Original evidence changed during training')
    write(output / 'reproduction.json', {'finished_utc': datetime.now(timezone.utc).isoformat(),
        'comparison': verification, 'fits': fits, 'legacy_fill_usage': fills,
        'all_predictions_reproduced': all(v['within_1e_10'] for v in verification.values()),
        'intent_cannot_be_determined_from_code': True})
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == '__main__':
    main()
