"""Unit test for trader_bridge.py"""
import tempfile
import unittest
from pathlib import Path

from trader_bridge import TraderBridge, ExecutionConfig, Position


class TestTraderBridge(unittest.TestCase):
    def test_trader_bridge_paper_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage_dir = Path(tmp_dir) / "trading_test"
            cfg = ExecutionConfig(
                mode="paper",
                capital=30000.0,
                commission_rate=0.00025,
                min_commission=5.0,
                stamp_duty_rate=0.0005,
                transfer_fee_rate=0.00001,
                slippage=0.0015,
                storage_dir=storage_dir,
            )
            bridge = TraderBridge(cfg)

            # 1. Initial State
            snap = bridge.sync_account()
            self.assertEqual(snap.cash, 30000.0)
            self.assertEqual(snap.total_asset, 30000.0)
            self.assertEqual(len(snap.positions), 0)

            # 2. Buy 000001.SZ with budget 7500 @ price 10.0
            # Price with 0.15% slippage = 10.015
            # Max shares for ~7500 = 700 shares
            buy_res = bridge.order_buy(code="000001.SZ", price=10.0, budget=7500.0, stop_loss=9.5, date_str="2026-10-21")
            self.assertIsNotNone(buy_res)
            self.assertEqual(buy_res["volume"], 700)
            self.assertEqual(bridge.positions["000001.SZ"].volume, 700)
            self.assertEqual(bridge.positions["000001.SZ"].can_use_volume, 0)  # Locked due to T+1

            # 3. Try to sell same day (T+1 violation check)
            sell_fail = bridge.order_sell(code="000001.SZ", price=10.5, volume=700, date_str="2026-10-21")
            self.assertIsNone(sell_fail)  # Blocked!
            self.assertEqual(bridge.positions["000001.SZ"].volume, 700)

            # 4. EOD Settlement (Day ends, marks to market and unlocks T+1)
            bridge.settle_daily_t1({"000001.SZ": 10.50})
            self.assertEqual(bridge.positions["000001.SZ"].can_use_volume, 700)
            self.assertEqual(bridge.positions["000001.SZ"].current_price, 10.50)

            # 5. Sell on Day 2
            sell_res = bridge.order_sell(code="000001.SZ", price=10.50, volume=700, reason="TAKE_PROFIT", date_str="2026-10-22")
            self.assertIsNotNone(sell_res)
            self.assertEqual(sell_res["volume"], 700)
            self.assertNotIn("000001.SZ", bridge.positions)
            self.assertGreater(bridge.cash, 30000.0)  # Profitable trade

            # 6. Check files created
            self.assertTrue((storage_dir / "positions.json").exists())
            self.assertTrue((storage_dir / "ledger.csv").exists())

    def test_trader_bridge_insufficient_cash(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage_dir = Path(tmp_dir) / "trading_test_low_cash"
            cfg = ExecutionConfig(
                mode="paper",
                capital=500.0,  # only 500 RMB
                storage_dir=storage_dir,
            )
            bridge = TraderBridge(cfg)
            # Price is 100 RMB, 100 shares = 10,000 RMB, budget 7500 > cash
            buy_res = bridge.order_buy(code="600519.SH", price=100.0, budget=7500.0)
            self.assertIsNone(buy_res)
            self.assertEqual(bridge.cash, 500.0)


if __name__ == "__main__":
    unittest.main()
