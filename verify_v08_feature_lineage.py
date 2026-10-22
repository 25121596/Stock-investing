"""Recompute the complete cached feature panel from raw bars without labels."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from accessibility import accessible_candidates
from engine import Config
from run import load
from exploration_features import extend_features
from train_model_v2 import enrich_features_v2
from train_model_v3 import extract_ashare_features
from train_model_v08 import extract_microstructure_features,FEATURES_V08

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/v08-codex-certified-20261020-v1/feature-lineage.json'


def main():
    if OUT.exists():raise ValueError('Preserve prior feature-lineage evidence')
    _,frames,calendar,_=load(ROOT/'data/forward-2026-10-13')
    spec=importlib.util.spec_from_file_location('feature_lineage_frozen',ROOT/'results/exploration-paper-frozen-v1/sources/ml_features-frozen.py')
    feature=importlib.util.module_from_spec(spec);spec.loader.exec_module(feature)
    feature._label=lambda *a,**k:(None,None,None,'inference_only')
    fresh=accessible_candidates(feature.build_panel(frames,calendar,[],Config(hold_days=10)))
    fresh=enrich_features_v2(extend_features(fresh,frames))
    ashare=extract_ashare_features(frames);micro=extract_microstructure_features(frames)
    fresh=fresh.merge(ashare,on=['date','code'],validate='one_to_one').merge(micro,on=['date','code'],validate='one_to_one')
    old=pd.read_csv(ROOT/'results/midterm-access-v3/panel-h10.csv',dtype={'date':str,'code':str})
    sixty=pd.read_csv(ROOT/'results/exploration-v1/new-features.csv',dtype={'date':str,'code':str})
    old=enrich_features_v2(old.merge(sixty[['date','code','adj_ret60','ret_vol60']],on=['date','code'],how='left',validate='one_to_one'))
    old=old.merge(ashare,on=['date','code'],how='left',validate='one_to_one').merge(micro,on=['date','code'],how='left',validate='one_to_one')
    joined=old[['date','code',*FEATURES_V08]].merge(fresh[['date','code',*FEATURES_V08]],on=['date','code'],suffixes=('_old','_fresh'),how='outer',indicator=True,validate='one_to_one')
    differences={}
    for col in FEATURES_V08:
        a,b=joined[col+'_old'].to_numpy(),joined[col+'_fresh'].to_numpy()
        equal=np.isclose(a,b,rtol=1e-8,atol=1e-9,equal_nan=True)
        finite=np.isfinite(a)&np.isfinite(b)
        differences[col]={'mismatch_rows':int((~equal).sum()),'max_abs_error_finite':float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else None}
    result={'cached_rows':len(old),'fresh_rows':len(fresh),'candidate_keys_equal':bool(joined._merge.eq('both').all()),
        'features':differences,'scope':'All cached candidate dates; new feature extractors recomputed from raw bars, no future label used',
        'all_within_tolerance':bool(joined._merge.eq('both').all() and all(v['mismatch_rows']==0 for v in differences.values()))}
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print({k:v for k,v in result.items() if k!='features'},flush=True)
    print({k:v for k,v in differences.items() if v['mismatch_rows']},flush=True)


if __name__=='__main__':main()
