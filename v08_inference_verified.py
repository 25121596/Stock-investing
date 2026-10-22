"""Label-free features and predictions from explicitly saved V2/V0.8 boosters."""
import importlib.util
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from accessibility import accessible_candidates
from engine import Config
from exploration_features import extend_features
from train_model_v2 import BASE_FEATURES, FEATURES_V2, enrich_features_v2
from train_model_v3 import ASHARE_FEATURES, extract_ashare_features
from train_model_v08 import FEATURES_V08, MICRO_FEATURES_V08, extract_microstructure_features


def inference_panel(frames, calendar, feature_source):
    spec = importlib.util.spec_from_file_location('v08_base_frozen_features', feature_source)
    feature = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(feature)
    feature._label = lambda *a, **k: (None, None, None, 'inference_only')
    panel = feature.build_panel(frames, calendar[-21:], [], Config(hold_days=10))
    panel = accessible_candidates(panel[panel.date.eq(calendar[-1])].copy())
    panel = enrich_features_v2(extend_features(panel, frames))
    panel = panel.merge(extract_ashare_features(frames), on=['date', 'code'], validate='one_to_one')
    panel = panel.merge(extract_microstructure_features(frames), on=['date', 'code'], validate='one_to_one')
    # The extractors already use fixed missing-value constants. A missing join
    # or nonfinite feature is an error, never fitted on future evaluation rows.
    if not np.isfinite(panel[FEATURES_V08].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite inference features; no global future-period imputation allowed')
    return panel


def predict(panel, v2_path, v08_path):
    result = panel.copy()
    v2 = lgb.Booster(model_file=str(v2_path))
    ranker = lgb.Booster(model_file=str(v08_path))
    if v2.feature_name() != FEATURES_V2 or ranker.num_feature() != len(FEATURES_V08):
        raise ValueError('Saved model feature order/size differs from frozen recipe')
    result['pred_net_return_v2'] = v2.predict(result[FEATURES_V2], num_threads=1) if len(result) else np.array([])
    result['pred_score_v08'] = ranker.predict(result[FEATURES_V08].to_numpy(dtype=float), num_threads=1) if len(result) else np.array([])
    if not np.isfinite(result[['pred_net_return_v2', 'pred_score_v08']].to_numpy()).all():
        raise ValueError('Nonfinite saved-model output')
    return result
