import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from engine import Config, Replay
from signals import evaluate, generate


CODE = "sh.600000"


def market(count=8, prices=None):
    dates = pd.bdate_range("2024-01-02", periods=count).strftime("%Y-%m-%d").tolist()
    prices = list(prices) if prices is not None else [10.0] * count
    frame = pd.DataFrame({"date": dates, "code": CODE, "open": prices, "close": prices,
                          "high": np.array(prices) * 1.01, "low": np.array(prices) * .99,
                          "preclose": [prices[0]] + prices[:-1], "volume": 1_000_000.0,
                          "amount": np.array(prices) * 1_000_000, "tradestatus": 1, "isST": 0})
    for key in ("open", "high", "low", "close"):
        frame[f"adj_{key}"] = frame[key] * 2
    signals = pd.DataFrame([{"date": dates[0], "code": CODE, "score": 1.0, "strategy": "test"}])
    return {CODE: frame}, dates, signals


def config(**kwargs):
    return Config(capital=3000, order_budget=2000, hold_days=1, max_exposure=1, stop_loss=0, **kwargs)


class AccountTests(unittest.TestCase):
    def test_simple_execution_matches_backtrader_independent_engine(self):
        import backtrader as bt
        frames, dates, signals = market(prices=[10, 10.2, 10.4, 10.4, 10.4, 10.4, 10.4, 10.4])
        rules = config()
        sim = Replay(frames, dates, signals, config=rules).run()

        class Costs(bt.CommInfoBase):
            params = (("stocklike", True), ("commtype", bt.CommInfoBase.COMM_PERC), ("percabs", True))

            def _getcommission(self, size, price, pseudoexec):
                return sum(rules.fees(abs(size) * price, sell=size < 0).values()) if size else 0

        class FixedTrade(bt.Strategy):
            def next(self):
                if len(self) == 1:
                    self.buy(size=100)
                elif len(self) == 2:
                    self.close()

        feed = frames[CODE].copy()
        feed.index = pd.to_datetime(feed.date)
        cerebro = bt.Cerebro(stdstats=False)
        cerebro.adddata(bt.feeds.PandasData(dataname=feed))
        cerebro.addstrategy(FixedTrade)
        cerebro.broker.setcash(rules.capital)
        cerebro.broker.addcommissioninfo(Costs())
        cerebro.run()
        self.assertAlmostEqual(cerebro.broker.getvalue(), sim.nav(), places=7)

    def test_next_day_entry_and_net_loss_at_flat_prices(self):
        frames, dates, signals = market()
        sim = Replay(frames, dates, signals, config=config()).run()
        self.assertEqual(sim.fills[0]["date"], dates[1])
        self.assertEqual(sim.trades[0]["exit_date"], dates[2])
        self.assertEqual(sim.trades[0]["initial_shares"] % 100, 0)
        self.assertLess(sim.trades[0]["net_pnl"], 0)
        self.assertEqual(sim.summary()["win_rate"], 0)
        self.assertAlmostEqual(sim.nav(), 3000 + sim.trades[0]["net_pnl"])
        self.assertEqual(sim.fills[0]["commission"], 5)
        self.assertGreater(sim.fills[1]["stamp"], 0)
        self.assertEqual(sim.fills[0]["stamp"], 0)

    def test_t_plus_one_rejects_same_day_sell(self):
        frames, dates, signals = market()
        sim = Replay(frames, dates, signals, config=config())
        sim.buy({"code": CODE, "signal_date": dates[0], "shares": 100, "limit": 10.3}, dates[1], 1)
        sim.sell(sim.positions[CODE], dates[1], 1)
        self.assertEqual(sim.positions[CODE].shares, 100)
        self.assertEqual(sim.rejected["T_plus_1"], 1)

    def test_limit_down_delays_exit(self):
        frames, dates, signals = market(prices=[10, 10, 9, 9.2, 9.2, 9.2, 9.2, 9.2])
        sim = Replay(frames, dates, signals, config=config()).run()
        self.assertEqual(sim.rejected["limit_down_open"], 1)
        self.assertEqual(sim.trades[0]["exit_date"], dates[3])

    def test_st_entry_and_limit_up_entry_rejected(self):
        frames, dates, signals = market()
        frames[CODE].loc[1, "isST"] = 1
        sim = Replay(frames, dates, signals, config=config()).run()
        self.assertFalse(sim.fills)
        self.assertEqual(sim.rejected["ST_entry"], 1)
        frames[CODE].loc[1, "isST"] = 0
        frames[CODE].loc[1, "open"] = 11
        sim = Replay(frames, dates, signals, config=config()).run()
        self.assertFalse(sim.fills)
        self.assertEqual(sim.rejected["limit_up_open"], 1)

    def test_suspension_delays_exit(self):
        frames, dates, signals = market()
        frames[CODE].loc[2, ["tradestatus", "volume"]] = 0
        sim = Replay(frames, dates, signals, config=config()).run()
        self.assertEqual(sim.trades[0]["exit_date"], dates[3])
        self.assertEqual(sim.rejected["suspended_or_missing"], 1)

    def test_end_position_marked_and_not_counted_as_closed(self):
        frames, dates, signals = market(count=2)
        sim = Replay(frames, dates, signals, config=config()).run()
        self.assertEqual(sim.summary()["open_positions"], 1)
        self.assertEqual(sim.summary()["closed_trades"], 0)
        self.assertIsNone(sim.summary()["win_rate"])
        self.assertLess(sim.nav(), 3000)

    def test_cash_dividend_entitlement_and_payment_delay(self):
        frames, dates, signals = market(prices=[10, 10, 9.8, 9.8, 9.8, 9.8, 9.8, 9.8])
        frames[CODE].loc[2, "preclose"] = 9.8
        event = {"code": CODE, "dividRegistDate": dates[1], "dividOperateDate": dates[2],
                 "dividPayDate": dates[3], "dividCashPsBeforeTax": "0.2"}
        sim = Replay(frames, dates, signals, [event], config()).run()
        self.assertAlmostEqual(sim.trades[0]["net_dividends"], 16)
        self.assertAlmostEqual(sim.equity[2]["receivables"], 16)
        self.assertEqual(sim.equity[3]["receivables"], 0)
        self.assertAlmostEqual(sim.cash, 3000 + sim.trades[0]["net_pnl"])

    def test_buyer_after_record_date_gets_no_dividend(self):
        frames, dates, signals = market(prices=[10, 9.8, 9.8, 9.8, 9.8, 9.8, 9.8, 9.8])
        frames[CODE].loc[1, "preclose"] = 9.8
        event = {"code": CODE, "dividRegistDate": dates[0], "dividOperateDate": dates[1],
                 "dividPayDate": dates[2], "dividCashPsBeforeTax": "0.2"}
        sim = Replay(frames, dates, signals, [event], config()).run()
        self.assertEqual(sim.trades[0]["net_dividends"], 0)

    def test_bonus_shares_locked_and_odd_lot_final_sale(self):
        frames, dates, signals = market(prices=[10, 10, 9.09, 9.09, 9.09, 9.09, 9.09, 9.09])
        frames[CODE].loc[2, "preclose"] = 9.09
        event = {"code": CODE, "dividRegistDate": dates[1], "dividOperateDate": dates[2],
                 "dividStockMarketDate": dates[4], "dividStocksPs": "0.1"}
        sim = Replay(frames, dates, signals, [event], config()).run()
        sells = [fill for fill in sim.fills if fill["side"] == "sell"]
        self.assertEqual([fill["shares"] for fill in sells], [100, 10])
        self.assertEqual(sells[-1]["date"], dates[4])
        self.assertEqual(sim.trades[0]["exit_date"], dates[4])
        self.assertAlmostEqual(sim.cash, 3000 + sim.trades[0]["net_pnl"])

    def test_unexplained_adjustment_is_flagged(self):
        frames, dates, signals = market()
        frames[CODE].loc[2, "preclose"] = 9
        sim = Replay(frames, dates, signals, config=config()).run()
        self.assertTrue(any(warning["issue"] == "unexplained_price_adjustment" for warning in sim.warnings))

    def test_zero_day_holding_invalid(self):
        with self.assertRaises(ValueError):
            Config(hold_days=0)


class CausalityTests(unittest.TestCase):
    def test_future_adjustment_scale_cannot_change_current_signal(self):
        frames, _, _ = market(count=300, prices=np.linspace(8, 12, 300))
        frame = frames[CODE]
        frame.loc[299, "close"] = frame.loc[298, "close"] * 1.03
        frame.loc[299, "open"] = frame.loc[299, "close"] * .99
        frame.loc[299, "volume"] = 30_000_000
        for key in ("open", "high", "low", "close"):
            frame[f"adj_{key}"] = frame[key] * 2
        baseline = evaluate(frame)
        self.assertTrue(baseline["volume"])
        altered = frame.copy()
        for key in ("open", "high", "low", "close"):
            altered[f"adj_{key}"] *= 1000
        self.assertEqual(baseline, evaluate(altered))

    def test_appending_extreme_future_does_not_change_past_signals(self):
        frames, dates, _ = market(count=270, prices=np.linspace(8, 12, 270))
        cutoff = dates[259]
        altered = frames[CODE].copy()
        altered.loc[260:, ["open", "high", "low", "close", "adj_open", "adj_high", "adj_low", "adj_close"]] = 1_000_000
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / "left", Path(directory) / "right"
            left.mkdir()
            right.mkdir()
            for folder in (left, right):
                (folder / "manifest.json").write_text("{}")
            first = generate(left, {CODE: frames[CODE].iloc[:260].copy()}, dates[0], cutoff)
            second = generate(right, {CODE: altered}, dates[0], cutoff)
            pd.testing.assert_frame_equal(first, second)


if __name__ == "__main__":
    unittest.main()
