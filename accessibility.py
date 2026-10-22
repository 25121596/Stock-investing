"""Point-in-time delisting-period buy restrictions for a small novice account."""
import json
from pathlib import Path

from engine_2026 import Replay2026

SOURCE_PATH = Path(__file__).with_name('accessibility_sources.json')
SOURCES = json.loads(SOURCE_PATH.read_text(encoding='utf-8'))
RETIREMENT_START = {}
for entry in SOURCES['entries']:
    if entry['status'] == 'confirmed' and entry.get('startdate'):
        if entry['announcement_date'] > entry['startdate']:
            raise ValueError('Restriction source must be known before its effective date')
        RETIREMENT_START[entry['code']] = entry['startdate']


def allowed_buy(code, day):
    return code not in RETIREMENT_START or day < RETIREMENT_START[code]


def accessible_candidates(panel):
    # Earlier history is retained even for stocks that later leave the exchange.
    mask = [allowed_buy(str(c), str(d)) for c, d in zip(panel.code, panel.date)]
    return panel.loc[mask].copy().reset_index(drop=True)


class AccessibleReplay(Replay2026):
    def buy(self, order, day, index):
        if not allowed_buy(order['code'], day):
            self.reject('retirement_buy_permission')
            return
        return super().buy(order, day, index)

    def tradable(self, bar, side):
        if (side == 'buy' and bar is not None and
                not allowed_buy(str(bar.get('code', '')), self.execution_day)):
            return False, 'retirement_buy_permission'
        return super().tradable(bar, side)
