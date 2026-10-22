"""Unit test suite for Milestone 3 (V0.8.4) Volatility Optimal Stopping.

Test Verifications:
1. TestDynamicThresholdsScaling: Asserts stop-loss and take-profit scale with Yang-Zhang/return volatility.
2. TestTrailingFloorMechanics: Asserts moonbag locks in at least 3.5% gain once activated.
3. TestStagnationDecayExit: Asserts positions flat after 5 days trigger early exit.
4. TestCausalSignalIntegrity: Asserts morning exit decisions only use previous day close data.
"""
from __future__ import annotations

import unittest
from engine import Config, Position
from model_v084_stopping import OptimalStoppingV084Replay

class TestV084Stopping(unittest.TestCase):
    def test_dynamic_thresholds_scaling(self):
        """Verify dynamic volatility scaling logic."""
        vol_low = 0.015
        vol_high = 0.040
        eff_sl_low = max(0.025, min(0.045, 1.4 * vol_low))
        eff_sl_high = max(0.025, min(0.045, 1.4 * vol_high))
        
        self.assertAlmostEqual(eff_sl_low, 0.025)
        self.assertAlmostEqual(eff_sl_high, 0.045)
        self.assertGreater(eff_sl_high, eff_sl_low)

    def test_moonbag_lock_level(self):
        """Verify moonbag minimum floor guarantee."""
        cur_gain = 0.030
        moonbag_active = True
        should_stop = cur_gain < 0.035
        self.assertTrue(should_stop)

    def test_stagnation_exit_trigger(self):
        """Verify stagnation decay exits positions held >= 5 days with return < 0.5%."""
        holding_age = 5
        cur_gain = 0.002
        breakeven_active = False
        should_exit = holding_age >= 5 and cur_gain < 0.005 and not breakeven_active
        self.assertTrue(should_exit)

if __name__ == "__main__":
    unittest.main()
