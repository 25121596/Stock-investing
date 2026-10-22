"""Small immutable source package for the two data-consistent equity ETFs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import time

import baostock as bs
import pandas as pd

from corporate_events import canonicalize_events

ROOT = Path(__file__).resolve().parent
PROBE = ROOT / 'data/etf-probe-2019-2026'
OUT = ROOT / 'data/etf-cash-two-v1'
CODES = ('sh.510300', 'sh.510050')
FIELDS = 'date,code,open,high,low,close,preclose,volume,amount,tradestatus,isST,pctChg,adjustflag'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise SystemExit('Keep previous source package; choose a new revision explicitly')
    OUT.mkdir()
    calendar_source = ROOT / 'data/forward-2026-10-13/calendar.csv'
    calendar = pd.read_csv(calendar_source, dtype=str)
    calendar.to_csv(OUT / 'calendar.csv', index=False)
    events = pd.read_csv(PROBE / 'dividend_operate_events_deduplicated_2019-2026.csv', dtype=str).fillna('')
    selected, conversions = [], []
    for event in events[events.code.isin(CODES)].to_dict('records'):
        before = float(event.get('dividCashPsBeforeTax') or 0)
        after = float(event.get('dividCashPsAfterTax') or 0)
        if before == 0 and after > 0:
            conversions.append({'code': event['code'], 'ex_date': event['dividOperateDate'], 'cash_per_unit': after,
                'reason': 'ETF before-tax field zero; after-tax cash retained under issuer stated zero withholding scenario'})
            event['dividCashPsBeforeTax'] = str(after)
        selected.append(event)
    selected, normalization = canonicalize_events(selected)
    socket.setdefaulttimeout(35)
    login = bs.login()
    if login.error_code != '0':
        raise RuntimeError(login.error_msg)
    inventory, audits = [], []
    for code in CODES:
        data = None
        for attempt in range(3):
            try:
                result = bs.query_history_k_data_plus(code, FIELDS, start_date='2018-10-01', end_date='2026-10-13', frequency='d', adjustflag='3')
                rows = []
                while result.error_code == '0' and result.next():
                    rows.append(result.get_row_data())
                if result.error_code != '0' or not rows:
                    raise ValueError(result.error_msg)
                data = pd.DataFrame(rows, columns=result.fields)
                if set(data.code) != {code} or set(data.adjustflag) != {'3'} or data.date.duplicated().any():
                    raise ValueError('Bad source identity/flags/dates')
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1)
                bs.login()
        for col in data.columns.difference(['date', 'code']):
            data[col] = pd.to_numeric(data[col], errors='raise')
        adj_path = PROBE / (code.replace('.', '_') + '_daily_adjustflag1_2018-10-01_2026-10-13.csv')
        adj = pd.read_csv(adj_path, dtype={'date': str, 'code': str}, float_precision='round_trip')
        if set(adj.adjustflag) != {1} or set(adj.date) != set(data.date):
            raise ValueError('Adjusted source is not uniformly flagged/complete')
        data = data.merge(adj[['date', 'close']].rename(columns={'close': 'adj_close'}), on='date', validate='one_to_one').sort_values('date')
        for col in ('open', 'high', 'low'):
            data[f'adj_{col}'] = data[col] * data.adj_close / data.close
        expected_days = set(calendar.calendar_date)
        if not expected_days.issubset(set(data.date)):
            raise ValueError('Missing market sessions')
        by_ex = {event['dividOperateDate']: event for event in selected if event['code'] == code}
        gaps, matched = [], []
        previous_close = None
        for bar in data.to_dict('records'):
            if previous_close is not None:
                event = by_ex.get(bar['date'])
                expected = previous_close - float(event['dividCashPsBeforeTax']) if event else previous_close
                if abs(float(bar['preclose']) - expected) > .0015:
                    gaps.append({'date': bar['date'], 'preclose': bar['preclose'], 'expected': expected})
                if event:
                    matched.append(bar['date'])
            previous_close = float(bar['close'])
        if gaps or set(by_ex) - set(matched):
            raise ValueError(f'Unexplained or missing ex-price events: {code}, {gaps}, {set(by_ex)-set(matched)}')
        own_events = [event for event in selected if event['code'] == code]
        price_path, event_path = OUT / f'{code}.csv', OUT / f'{code}.dividends.csv'
        data.to_csv(price_path, index=False, encoding='utf-8-sig')
        pd.DataFrame(own_events).to_csv(event_path, index=False, encoding='utf-8-sig')
        inventory.append({'code': code, 'sha256': digest(price_path), 'dividends_sha256': digest(event_path),
            'rows': len(data), 'adjusted_source_sha256': digest(adj_path)})
        audits.append({'code': code, 'rows': len(data), 'ex_price_events_checked': matched, 'unexplained_preclose_changes': gaps})
        print(f'SOURCE {code}: {len(data)} bars, {len(matched)} cash events checked', flush=True)
    manifest = {'provider': 'BaoStock, separate raw execution and backward indicator prices', 'start': '2020-01-02',
        'end': '2026-10-13', 'inventory': inventory, 'failures': [], 'downloaded_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'two ETFs selected for consistent source flags/events before strategy evaluation; not full ETF universe',
        'source_events_sha256': digest(PROBE / 'dividend_operate_events_deduplicated_2019-2026.csv'),
        'event_cash_conversions': conversions, 'normalization': normalization, 'audits': audits,
        'calendar_sha256': digest(OUT / 'calendar.csv')}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(f'MANIFEST {OUT / "manifest.json"}')


if __name__ == '__main__':
    main()
