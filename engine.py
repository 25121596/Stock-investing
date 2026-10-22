"""Small auditable daily A-share ledger; no live orders or brokerage access."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from decimal import Decimal, ROUND_HALF_UP
import math

import pandas as pd


def number(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def tick(value):
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


@dataclass
class Config:
    capital: float = 30000.0
    order_budget: float = 5000.0
    max_positions: int = 3
    max_exposure: float = 0.5
    hold_days: int = 3
    stop_loss: float = 0.03
    max_buy_gap: float = 0.03
    commission: float = 0.00025
    min_commission: float = 5.0
    transfer_fee: float = 0.00001
    stamp_tax: float = 0.0005
    slippage: float = 0.0005
    dividend_tax: float = 0.20

    def __post_init__(self):
        if self.hold_days < 1 or self.capital <= 0 or self.order_budget <= 0:
            raise ValueError("Require positive capital, budget, and holding >= 1 trading day")
        if self.max_positions < 1 or not 0 < self.max_exposure <= 1:
            raise ValueError("Invalid position/exposure limits")
        if any(getattr(self, key) < 0 for key in
               ("stop_loss", "max_buy_gap", "commission", "min_commission", "transfer_fee", "stamp_tax", "slippage")):
            raise ValueError("Negative execution parameters")
        if not 0 <= self.dividend_tax <= 1:
            raise ValueError("Invalid dividend tax")

    def fees(self, notional, sell=False):
        return {"commission": tick(max(self.min_commission, notional * self.commission)),
                "transfer": tick(notional * self.transfer_fee),
                "stamp": tick(notional * self.stamp_tax) if sell else 0.0,
                "slippage": tick(notional * self.slippage)}


@dataclass
class Position:
    trade_id: int
    code: str
    signal_date: str
    entry_date: str
    entry_index: int
    shares: int
    initial_shares: int
    entry_notional: float
    entry_fees: float
    proceeds: float = 0.0
    exit_fees: float = 0.0
    dividends: float = 0.0
    locked: list = field(default_factory=list)
    pending_actions: int = 0
    stop_pending: bool = False


class Replay:
    def __init__(self, frames, calendar, signals, dividends=None, config=None):
        self.config = config or Config()
        self.frames = frames
        self.calendar = list(calendar)
        self.bars = {code: frame.set_index("date").to_dict("index") for code, frame in frames.items()}
        if not self.calendar or sorted(set(self.calendar)) != self.calendar:
            raise ValueError("Calendar must be nonempty, increasing and unique")
        self.candidates = defaultdict(list)
        for row in signals.to_dict("records"):
            self.candidates[row["date"]].append(row)
        self.actions_by_record = defaultdict(list)
        self.actions_by_ex = defaultdict(list)
        for event in dividends or []:
            self.actions_by_record[event.get("dividRegistDate", "")].append(event)
            self.actions_by_ex[event.get("dividOperateDate", "")].append(event)
        self.positions = {}
        self.cash = self.config.capital
        self.marks = {}
        self.receivables = []
        self.entitlements = []
        self.pending_buys = []
        self.fills, self.trades, self.equity, self.actions, self.warnings = [], [], [], [], []
        self.rejected = Counter()
        self.next_id = 1

    def nav(self):
        return self.cash + sum(p.shares * self.marks.get(p.code, 0) for p in self.positions.values()) + sum(
            item[1] for item in self.receivables)

    def reject(self, reason):
        self.rejected[reason] += 1

    def tradable(self, bar, side):
        if bar is None or number(bar.get("tradestatus")) != 1 or number(bar.get("volume")) <= 0:
            return False, "suspended_or_missing"
        if side == "buy" and number(bar.get("isST")) != 0:
            return False, "ST_entry"
        price, reference = number(bar.get("open")), number(bar.get("preclose"))
        if price <= 0 or reference <= 0:
            return False, "invalid_open"
        # Dataset is restricted to mature main-board stocks in 2020-2025.
        # Conservatively reject at-limit opening fills; no intraday queue model.
        rate = 0.05 if number(bar.get("isST")) == 1 else 0.10
        if side == "buy" and price >= tick(reference * (1 + rate)) - 0.0001:
            return False, "limit_up_open"
        if side == "sell" and price <= tick(reference * (1 - rate)) + 0.0001:
            return False, "limit_down_open"
        return True, ""

    def finish(self, position, day, reason):
        if position.shares != 0 or position.pending_actions:
            return
        pnl = position.proceeds + position.dividends - position.entry_notional - position.entry_fees
        self.trades.append({"trade_id": position.trade_id, "code": position.code,
                            "signal_date": position.signal_date, "entry_date": position.entry_date,
                            "exit_date": day, "initial_shares": position.initial_shares,
                            "entry_notional": position.entry_notional, "entry_fees": position.entry_fees,
                            "exit_fees": position.exit_fees, "net_dividends": position.dividends,
                            "net_pnl": pnl, "net_return": pnl / (position.entry_notional + position.entry_fees),
                            "exit_reason": reason})
        self.positions.pop(position.code, None)

    def corporate_actions(self, day):
        # Receivable cash is in NAV, but cannot finance an order before payment.
        waiting = []
        for pay_date, cash in self.receivables:
            if pay_date <= day:
                self.cash += cash
            else:
                waiting.append((pay_date, cash))
        self.receivables = waiting
        for claim in self.entitlements:
            event = claim["event"]
            if claim["posted"] or event.get("dividOperateDate") != day:
                continue
            claim["posted"] = True
            position, quantity = claim["position"], claim["shares"]
            per_cash = number(event.get("dividCashPsBeforeTax"))
            cash = quantity * per_cash * (1 - self.config.dividend_tax)
            if cash:
                pay = event.get("dividPayDate") or day
                if not event.get("dividPayDate"):
                    self.warnings.append({"date": day, "code": position.code, "issue": "missing_dividend_payment_date"})
                position.dividends += cash
                if pay <= day:
                    self.cash += cash
                else:
                    self.receivables.append((pay, cash))
            ratio = number(event.get("dividStocksPs")) + number(event.get("dividReserveToStockPs"))
            bonus_exact = quantity * ratio
            bonus = int(round(bonus_exact))
            if abs(bonus_exact - bonus) > 0.001:
                self.warnings.append({"date": day, "code": position.code, "issue": "fractional_bonus_share"})
            if bonus:
                listing = event.get("dividStockMarketDate") or "9999-12-31"
                if listing == "9999-12-31":
                    self.warnings.append({"date": day, "code": position.code, "issue": "missing_bonus_listing_date"})
                position.shares += bonus
                position.locked.append((listing, bonus))
            position.pending_actions -= 1
            self.actions.append({"date": day, "trade_id": position.trade_id, "code": position.code,
                                 "entitled_shares": quantity, "net_cash": cash, "bonus_shares": bonus})
            self.finish(position, day, "corporate_action_complete")

    def buy(self, order, day, index):
        code = order["code"]
        if code in self.positions or len(self.positions) >= self.config.max_positions:
            self.reject("position_limit")
            return
        bar = self.bars[code].get(day)
        allowed, reason = self.tradable(bar, "buy")
        if not allowed:
            self.reject(reason)
            return
        price = float(bar["open"])
        if price > order["limit"] + 0.0001:
            self.reject("buy_gap_limit")
            return
        quantity = order["shares"]
        notional = quantity * price
        fees = self.config.fees(notional)
        cost = notional + sum(fees.values())
        if cost > self.cash + 1e-8:
            self.reject("cash_limit")
            return
        invested = sum(p.shares * self.marks[p.code] for p in self.positions.values())
        if invested + notional > self.nav() * self.config.max_exposure + 1e-8:
            self.reject("exposure_limit")
            return
        self.cash -= cost
        position = Position(self.next_id, code, order["signal_date"], day, index, quantity,
                            quantity, notional, sum(fees.values()))
        self.next_id += 1
        self.positions[code] = position
        self.marks[code] = price
        self.fills.append({"date": day, "trade_id": position.trade_id, "code": code, "side": "buy",
                           "shares": quantity, "price": price, "notional": notional, **fees})

    def sell(self, position, day, index):
        if index <= position.entry_index:
            self.reject("T_plus_1")
            return
        bar = self.bars[position.code].get(day)
        allowed, reason = self.tradable(bar, "sell")
        if not allowed:
            self.reject(reason)
            return
        locked = sum(quantity for listing, quantity in position.locked if listing > day)
        available = position.shares - locked
        if available < position.shares:
            available = available // 100 * 100
        if available <= 0:
            self.reject("locked_bonus_shares")
            return
        price = float(bar["open"])
        notional = available * price
        fees = self.config.fees(notional, sell=True)
        proceeds = notional - sum(fees.values())
        self.cash += proceeds
        position.proceeds += proceeds
        position.exit_fees += sum(fees.values())
        position.shares -= available
        self.fills.append({"date": day, "trade_id": position.trade_id, "code": position.code,
                           "side": "sell", "shares": available, "price": price, "notional": notional, **fees})
        self.finish(position, day, "stop_close_next_open" if position.stop_pending else "fixed_horizon")

    def run(self):
        for index, day in enumerate(self.calendar):
            self.corporate_actions(day)
            # Unknown corporate actions / stale quotes cannot silently become
            # a claimed validated return. Keep the ledger and flag the run.
            for code, position in list(self.positions.items()):
                bar = self.bars[code].get(day)
                if bar is not None and code in self.marks:
                    reference = number(bar.get("preclose"))
                    if reference > 0 and abs(reference - self.marks[code]) > 0.025:
                        known = [e for e in self.actions_by_ex.get(day, []) if e.get("code") == code]
                        if not known:
                            self.warnings.append({"date": day, "code": code, "issue": "unexplained_price_adjustment"})
                        else:
                            cash_ps = sum(number(e.get("dividCashPsBeforeTax")) for e in known)
                            share_ps = sum(number(e.get("dividStocksPs")) + number(e.get("dividReserveToStockPs")) for e in known)
                            expected = (self.marks[code] - cash_ps) / (1 + share_ps)
                            if abs(reference - expected) > 0.025:
                                self.warnings.append({"date": day, "code": code, "issue": "corporate_action_reference_mismatch"})
                elif bar is None:
                    self.warnings.append({"date": day, "code": code, "issue": "missing_held_stock_quote"})
                if index - position.entry_index >= self.config.hold_days or position.stop_pending:
                    self.sell(position, day, index)
            for code in self.positions:
                bar = self.bars[code].get(day)
                if bar is not None and number(bar.get("open")) > 0:
                    self.marks[code] = float(bar["open"])
            for order in self.pending_buys:
                self.buy(order, day, index)
            self.pending_buys = []

            for code, position in self.positions.items():
                bar = self.bars[code].get(day)
                if bar is not None:
                    self.marks[code] = float(bar["close"])
                value = position.shares * self.marks[code] + position.proceeds + position.dividends
                if self.config.stop_loss and value < position.entry_notional * (1 - self.config.stop_loss):
                    position.stop_pending = True
            for event in self.actions_by_record.get(day, []):
                position = self.positions.get(event.get("code"))
                if position and position.shares:
                    position.pending_actions += 1
                    self.entitlements.append({"event": event, "position": position,
                                              "shares": position.shares, "posted": False})
            nav = self.nav()
            self.equity.append({"date": day, "equity": nav, "cash": self.cash,
                                "receivables": sum(item[1] for item in self.receivables),
                                "positions": len(self.positions)})
            if self.cash < -1e-7 or any(p.shares < 0 for p in self.positions.values()):
                raise AssertionError("Negative cash or shares")
            for candidate in sorted(self.candidates.get(day, []), key=lambda row: (-row["score"], row["code"])):
                code = candidate["code"]
                if code in self.positions:
                    continue
                bar = self.bars[code].get(day)
                if bar is None:
                    continue
                limit = tick(float(bar["close"]) * (1 + self.config.max_buy_gap))
                budget = min(self.config.order_budget, self.cash)
                quantity = int(max(0, budget - self.config.min_commission) /
                               (limit * (1 + self.config.commission + self.config.transfer_fee + self.config.slippage))) // 100 * 100
                if quantity < 100:
                    self.reject("lot_size_or_budget")
                    continue
                self.pending_buys.append({"code": code, "signal_date": day, "shares": quantity, "limit": limit})
                break
        return self

    def summary(self):
        equity = pd.DataFrame(self.equity)
        curve = pd.concat([pd.Series([self.config.capital]), equity.equity], ignore_index=True)
        pnl = [trade["net_pnl"] for trade in self.trades]
        wins = sum(value > 0 for value in pnl)
        positive = sum(value for value in pnl if value > 0)
        negative = -sum(value for value in pnl if value < 0)
        final = float(curve.iloc[-1])
        duration = (pd.Timestamp(self.calendar[-1]) - pd.Timestamp(self.calendar[0])).days
        yearly = []
        previous = self.config.capital
        for year, group in equity.groupby(equity.date.str[:4]):
            ending = float(group.equity.iloc[-1])
            yearly.append({"year": year, "return": ending / previous - 1,
                           "trades": sum(trade["exit_date"].startswith(year) for trade in self.trades)})
            previous = ending
        open_pnl = sum(item["marked_pnl"] for item in self.holdings())
        reconciliation = final - self.config.capital - sum(pnl) - open_pnl
        if abs(reconciliation) > 1e-6:
            raise AssertionError(f"NAV does not reconcile to closed/open P&L: {reconciliation}")
        return {"initial_capital": self.config.capital, "final_equity": final,
                "total_return": final / self.config.capital - 1,
                "annualized_return": (final / self.config.capital) ** (365.25 / duration) - 1 if duration else None,
                "max_drawdown": float((curve / curve.cummax() - 1).min()),
                "closed_trades": len(pnl), "win_rate": wins / len(pnl) if pnl else None,
                "profit_factor": positive / negative if negative else None,
                "mean_trade_pnl": sum(pnl) / len(pnl) if pnl else None,
                "execution_costs": sum(sum(fill[key] for key in ("commission", "transfer", "stamp", "slippage")) for fill in self.fills),
                "open_positions": len(self.positions), "receivables": sum(item[1] for item in self.receivables),
                "quality_warning_count": len(self.warnings), "rejected": dict(self.rejected),
                "pnl_reconciliation_error": reconciliation,
                "config": asdict(self.config), "yearly": yearly}

    def holdings(self):
        return [{"trade_id": p.trade_id, "code": p.code, "entry_date": p.entry_date,
                 "shares": p.shares, "last_mark": self.marks[p.code],
                 "market_value": p.shares * self.marks[p.code],
                 "entry_notional": p.entry_notional, "entry_fees": p.entry_fees,
                 "net_partial_proceeds": p.proceeds, "net_dividends": p.dividends,
                 "marked_pnl": p.shares * self.marks[p.code] + p.proceeds + p.dividends
                               - p.entry_notional - p.entry_fees,
                 "pending_actions": p.pending_actions,
                 "locked_shares": sum(qty for day, qty in p.locked if day > self.calendar[-1])}
                for p in self.positions.values()]
