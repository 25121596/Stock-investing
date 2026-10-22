"""Main-board equity ETF execution with mill price-limit ticks and cash distributions."""
from decimal import Decimal, ROUND_HALF_UP

from engine import Replay, number, tick


def price_tick(value):
    return float(Decimal(str(value)).quantize(Decimal('.001'), rounding=ROUND_HALF_UP))


class ETFReplay(Replay):
    def tradable(self, bar, side):
        if bar is None or number(bar.get('tradestatus')) != 1 or number(bar.get('volume')) <= 0:
            return False, 'suspended_or_missing'
        price, reference = number(bar.get('open')), number(bar.get('preclose'))
        if price <= 0 or reference <= 0:
            return False, 'invalid_open'
        if side == 'buy' and price >= price_tick(reference * 1.10) - .0000001:
            return False, 'limit_up_open'
        if side == 'sell' and price <= price_tick(reference * .90) + .0000001:
            return False, 'limit_down_open'
        return True, ''

    def run(self):
        # Same accounting/execution order as the stock ledger. The ETF order
        # budget additionally respects remaining exposure at the signal close.
        self.order_plans = []
        for index, day in enumerate(self.calendar):
            self.corporate_actions(day)
            for code, position in list(self.positions.items()):
                bar = self.bars[code].get(day)
                if bar is not None and code in self.marks:
                    reference = number(bar.get('preclose'))
                    if reference > 0 and abs(reference - self.marks[code]) > .0015:
                        known = [e for e in self.actions_by_ex.get(day, []) if e.get('code') == code]
                        if not known:
                            self.warnings.append({'date': day, 'code': code, 'issue': 'unexplained_price_adjustment'})
                        else:
                            cash = sum(number(e.get('dividCashPsBeforeTax')) for e in known)
                            shares = sum(number(e.get('dividStocksPs')) + number(e.get('dividReserveToStockPs')) for e in known)
                            if abs(reference - (self.marks[code] - cash) / (1 + shares)) > .0015:
                                self.warnings.append({'date': day, 'code': code, 'issue': 'corporate_action_reference_mismatch'})
                elif bar is None:
                    self.warnings.append({'date': day, 'code': code, 'issue': 'missing_held_stock_quote'})
                if index - position.entry_index >= self.config.hold_days or position.stop_pending:
                    self.sell(position, day, index)
            for code in self.positions:
                bar = self.bars[code].get(day)
                if bar is not None and number(bar.get('open')) > 0:
                    self.marks[code] = float(bar['open'])
            for order in self.pending_buys:
                self.buy(order, day, index)
            self.pending_buys = []
            for code, position in self.positions.items():
                bar = self.bars[code].get(day)
                if bar is not None:
                    self.marks[code] = float(bar['close'])
                value = position.shares * self.marks[code] + position.proceeds + position.dividends
                if self.config.stop_loss and value < position.entry_notional * (1 - self.config.stop_loss):
                    position.stop_pending = True
            for event in self.actions_by_record.get(day, []):
                position = self.positions.get(event.get('code'))
                if position and position.shares:
                    position.pending_actions += 1
                    self.entitlements.append({'event': event, 'position': position, 'shares': position.shares, 'posted': False})
            nav = self.nav()
            self.equity.append({'date': day, 'equity': nav, 'cash': self.cash,
                'receivables': sum(item[1] for item in self.receivables), 'positions': len(self.positions)})
            if self.cash < -1e-7 or any(p.shares < 0 for p in self.positions.values()):
                raise AssertionError('Negative cash or ETF shares')
            for candidate in sorted(self.candidates.get(day, []), key=lambda row: (-row['score'], row['code'])):
                code = candidate['code']
                if code in self.positions:
                    continue
                bar = self.bars[code].get(day)
                if bar is None:
                    continue
                limit = tick(float(bar['close']) * (1 + self.config.max_buy_gap))
                invested = sum(p.shares * self.marks[p.code] for p in self.positions.values())
                room = max(0., nav * self.config.max_exposure - invested)
                budget = min(self.config.order_budget, self.cash, room)
                quantity = int(max(0., budget - self.config.min_commission) /
                    (limit * (1 + self.config.commission + self.config.transfer_fee + self.config.slippage))) // 100 * 100
                if quantity < 100:
                    self.reject('lot_size_or_exposure_budget')
                    continue
                order = {'code': code, 'signal_date': day, 'shares': quantity, 'limit': limit}
                self.pending_buys.append(order)
                self.order_plans.append({**order, 'equity_at_signal': nav, 'remaining_exposure_at_signal': room, 'budget': budget})
                break
        return self
