"""Create a separate, fully pinned V0.8 manual forward experiment."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent

EXTENSION = r'''
# V0.8 adapter. The append-only journal implementation above is retained;
# only inference, execution rules, and complete strategy-state serialization
# are adapted. Historical account signals are never recomputed.
_base_load_runtime = load_runtime
_base_account_state = account_state
_SIM_PANEL = None


def load_runtime(folder):
    return _base_load_runtime(folder)


def infer(frames, calendar, cutoff, runtime, modules):
    global _SIM_PANEL
    from v08_inference_verified import inference_panel, predict
    from exploration_features import breadth_days
    panel = inference_panel(frames, calendar, runtime / 'sources/ml_features-frozen.py')
    panel = predict(panel, runtime / 'sources/2026-v2.txt', runtime / '2026-lgb.txt')
    old = [pd.read_csv(p, dtype={'date':str,'code':str},float_precision='round_trip')
           for p in sorted((runtime.parent / 'days').glob('*/scores-and-features.csv'))
           if not p.parent.name.startswith('.')]
    _SIM_PANEL = pd.concat(old + [panel], ignore_index=True)
    breadth = breadth_days(panel)
    allowed = bool(len(breadth) and breadth.pool_size.iloc[0] >= 100 and breadth.breadth5.iloc[0] >= .45)
    selected = panel[panel.pred_net_return_v2.ge(.005)] if allowed else panel.iloc[:0]
    sig = selected[['date','code','pred_score_v08']].rename(columns={'pred_score_v08':'score'})
    signals = {'v08_verified':sig, 'cash':pd.DataFrame(columns=['date','code','score'])}
    for seed in SEEDS:
        signals[f'v08_verified_random_{seed}'] = random_matched(panel,len(sig),seed)
    return panel,breadth,signals


def simulate(frames,bars,calendar,signals,events,hold,modules):
    from v08_engine_verified import VerifiedV08Replay
    from exploration_features import breadth_days
    engine = modules[0]
    b = breadth_days(_SIM_PANEL)
    bm=b.set_index('date').breadth5.to_dict()
    pm=b.set_index('date').pool_size.to_dict()
    limits={d:0 if pm[d]<100 or v<.45 else (1 if v<.55 else 2) for d,v in bm.items()}
    vol=_SIM_PANEL.set_index(['date','code']).ret_vol20.to_dict()
    cfg=engine.Config(capital=30000,order_budget=7500,max_positions=5,max_exposure=.90,hold_days=10)
    rep=VerifiedV08Replay({},calendar,signals,events,cfg,vol_map=vol,daily_limits=limits,
        breadth_lookup=bm,enable_cash_overlay=False)
    rep.frames,rep.bars=frames,bars
    return rep.run()


def account_state(rep):
    result=_base_account_state(rep)
    result['peak_prices']=normal(rep.peak_prices)
    result['profit_lock_flags']={code:{'breakeven_active':bool(getattr(p,'breakeven_active',False)),
        'moonbag_active':bool(getattr(p,'moonbag_active',False))} for code,p in rep.positions.items()}
    return result


def v08_sell_due(rep,position,days):
    horizon=12 if rep.daily_limits.get(days[-1],1)>=2 else 8
    return position.stop_pending or len(days)-position.entry_index>=horizon

'''


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--version',choices=['v1','v2','v3'],default='v1')
    args=parser.parse_args()
    version=args.version
    target=ROOT/('v08_paper_forward.py' if version=='v1' else f'v08_paper_forward_{version}.py')
    freeze=ROOT/f'results/v08-codex-frozen-{version}'
    bundle=ROOT/'results/v08-codex-rebuild-20261020-v1'
    certification=ROOT/'results/v08-codex-certified-20261020-v1'
    if target.exists() or freeze.exists():raise ValueError('Preserve existing V0.8 version')
    reproduction=json.loads((bundle/'reproduction.json').read_text())
    if not reproduction['all_predictions_reproduced']:raise ValueError('Original predictions did not reproduce')
    parity=json.loads((certification/'inference-audit.json').read_text())
    if not parity['coverage_equal'] or max(parity['maximum_errors'].values())>1e-10:raise ValueError('Inference not verified')
    source=(ROOT/'paper_forward.py').read_text(encoding='utf-8')
    substitutions={
      "FREEZE = ROOT / 'results/exploration-paper-frozen-v1'":f"FREEZE = ROOT / 'results/v08-codex-frozen-{version}'",
      "SESSION = ROOT / 'results/paper-forward-v1'":f"SESSION = ROOT / 'results/v08-paper-forward-{version}'",
      "MAIN = {'base_lgb10': 10, 'breadth_lgb10': 10, 'weekly_lowvol': 20}":"MAIN = {'v08_verified': 10}",
      "'historical_data_last_day': '2026-10-13'":"'historical_data_last_day': '2026-10-19'",
      "if p.stop_pending or len(days) - p.entry_index >= spec['hold_days']":"if v08_sell_due(replay, p, days)",
      "names = {'base_lgb10': '原10日模型', 'breadth_lgb10': '模型＋广度过滤', 'weekly_lowvol': '周频低波动', 'cash': '现金对照'}":"names = {'v08_verified':'V0.8核验版（无现金利息）','cash':'现金对照'}",
      "三个主策略、各三个固定随机对照、一个现金对照。":"一个V0.8主策略、三个固定随机对照和现金对照。",
      "3 policies + 9 random controls + cash":"1 V0.8 policy + 3 random controls + cash",
      "'version': 'paper-forward-v1'":f"'version': 'v08-paper-forward-{version}'"}
    for old,new in substitutions.items():
        if old not in source:raise ValueError(f'Generic recorder changed: {old}')
        source=source.replace(old,new)
    source=source.replace("if __name__ == '__main__':",EXTENSION+"\nif __name__ == '__main__':")
    compile(source,str(target),'exec')
    target.write_text(source,encoding='utf-8')
    freeze.mkdir()
    sources=freeze/'sources';sources.mkdir()
    for path in ROOT.glob('*.py'):shutil.copyfile(path,sources/path.name)
    shutil.copyfile(ROOT/'accessibility_sources.json',sources/'accessibility_sources.json')
    shutil.copyfile(ROOT/'results/exploration-paper-frozen-v1/sources/ml_features-frozen.py',sources/'ml_features-frozen.py')
    shutil.copyfile(bundle/'v08/2026.txt',freeze/'2026-lgb.txt')
    shutil.copyfile(bundle/'v2/2026.txt',sources/'2026-v2.txt')
    for kind in ['v2','v08']:shutil.copyfile(bundle/kind/'2026-training.json',freeze/f'{kind}-training.json')
    hashes={p.name:digest(p) for p in sources.iterdir() if p.is_file()}
    record={'frozen_utc':datetime.now(timezone.utc).isoformat(),'model_sha256':digest(freeze/'2026-lgb.txt'),
       'source_hashes':hashes,'source_features_sha256':digest(sources/'ml_features-frozen.py'),
       'rank_model_features':reproduction['fits'][-1]['features'],
       'policies':{'v08_verified':{'capital':30000,'cash_interest':False,'threshold_v2':.005,
          'breadth_min':.45,'two_buys_breadth':.55,'high_budget_breadth':.58,
          'high_budget':7500,'neutral_budget':5500,'high_max_positions':5,'neutral_max_positions':4,
          'max_exposure_on_buy':.9,'horizon_strong':12,'horizon_neutral':8}},
       'first_signal_date_not_before':'2026-10-20','model_year':2026,
       'all_history_through_2026_10_13_previously_viewed':True,'extra_embargo_sessions':0,
       'global_test_median_imputation_allowed':False,'no_automation':True,'no_live_orders':True,
       'base_recorder_sha256':digest(ROOT/'paper_forward.py'),'adapted_recorder_sha256':digest(target),
       'reproduction_sha256':digest(bundle/'reproduction.json'),
       'certification_sha256':digest(certification/'certification.json')}
    (freeze/'freeze.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FROZEN',freeze,'; RECORDER',target)


if __name__=='__main__':main()
