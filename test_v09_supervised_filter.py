"""Acceptance tests for the frozen V0.9 supervised batch gate."""
import json
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

import arena
from engine import Config
from v09_rl_2022 import FollowPolicy, RLReplay, VotePolicy
from v09_supervised_filter import (
    BreadthGate,
    FrozenLogit,
    GateAdapter,
    LegacyRLGate,
    matured_batch_labels,
)


def batch_obs(day="2022-12-20", execution="2022-12-21", codes=("sh.600006", "sh.600007")):
    return {"signal_date": day, "execution_date": execution, "state": [2, 1, 0, 0, 2, 0, 1],
            "allowed": [0, 1], "action": 0, "score": None,
            "orders": [{"code": c} for c in codes], "pending_before": len(codes), "positions_before": 0}


class LabelTests(unittest.TestCase):
    def replay(self, trades, fills):
        return SimpleNamespace(trades=trades, fills=fills)

    def trade(self, trade_id, code, pnl, exit_date, **kw):
        return {"trade_id": trade_id, "code": code, "signal_date": "2022-12-20",
                "entry_date": "2022-12-21", "exit_date": exit_date, "net_pnl": pnl,
                "entry_notional": 1000., "entry_fees": 5., **kw}

    def fill(self, trade_id, code):
        return {"trade_id": trade_id, "code": code, "date": "2022-12-21", "side": "buy"}

    def test_labels_aggregate_only_actual_filled_trade_ids_and_net_costs(self):
        # The engine's net_pnl already includes net dividends and exit costs; the
        # batch return denominator additionally includes each original entry fee.
        trades = [self.trade("a", "sh.600006", 30., "2022-12-27",
                             proceeds=1026., net_dividends=9., exit_fees=4.),
                  self.trade("b", "sh.600007", -40., "2022-12-29",
                             proceeds=965., net_dividends=0., exit_fees=3.)]
        # Other closed trades are not part of this signal's executed batch.
        trades.append({**self.trade("unrelated", "sh.600008", 999., "2022-12-30"),
                       "signal_date": "2022-12-19"})
        records = matured_batch_labels(self.replay(trades, [self.fill("a", "sh.600006"),
                                                             self.fill("b", "sh.600007")]),
                                       [batch_obs()])
        row = records[0]
        self.assertEqual(row["filled_trade_ids"], ["a", "b"])
        self.assertEqual(row["filled_count"], 2)
        self.assertEqual(row["status"], "eligible")
        self.assertEqual(row["label_end"], "2022-12-29")
        self.assertEqual(row["net_pnl"], -10.)
        self.assertEqual(row["net_return"], -10. / 2010.)
        self.assertEqual(row["y"], 0)

    def test_unexecuted_batch_is_not_a_negative_sample(self):
        row = matured_batch_labels(self.replay([], []), [batch_obs()])[0]
        self.assertEqual(row["status"], "no_execution")
        self.assertIsNone(row["y"])
        self.assertIsNone(row["net_pnl"])

    def test_open_trade_is_right_censored_and_2023_boundary_is_excluded(self):
        open_trade = self.trade("open", "sh.600006", 20., None)
        # It was bought, but there is no closed trade ledger row for its id.
        censored = matured_batch_labels(self.replay([], [self.fill("open", "sh.600006")]),
                                        [batch_obs(codes=("sh.600006",))])[0]
        self.assertEqual(censored["status"], "right_censored")
        self.assertIsNone(censored["y"])
        self.assertIsNotNone(open_trade)  # illustrates that a position row is not a closed trade

        boundary_trade = self.trade("edge", "sh.600006", 20., "2023-01-01")
        edge = matured_batch_labels(self.replay([boundary_trade], [self.fill("edge", "sh.600006")]),
                                    [batch_obs(codes=("sh.600006",))])[0]
        self.assertEqual(edge["status"], "label_not_mature_before_fit")
        self.assertIsNone(edge["y"])


class FrozenModelTests(unittest.TestCase):
    def rows(self):
        # Each dimension varies in training so the fitted scaler can be checked exactly.
        return [{"state": [i, i % 3, i % 4, i % 2, i % 5, i % 3, i % 2],
                 "y": int(i % 2 == 0), "label_end": f"2022-12-{10+i:02d}",
                 "signal_date": f"2022-12-{i+1:02d}"} for i in range(12)]

    def test_scaler_uses_training_only_and_json_roundtrip_preserves_scores(self):
        rows = self.rows()
        model = FrozenLogit.fit(rows)
        self.assertEqual(model.record["mean"], np.mean([r["state"] for r in rows], axis=0).tolist())
        self.assertEqual(model.record["training_rows"], len(rows))
        encoded = json.dumps(model.record)
        restored = FrozenLogit(json.loads(encoded))
        probe = [20, 1, 3, 0, 2, 1, 1]
        self.assertAlmostEqual(model.score(probe), restored.score(probe), places=15)
        self.assertEqual(model.record["threshold"], .5)
        self.assertEqual(model.decide(probe, [0, 1])[0], int(model.score(probe) < .5))

    def test_insufficient_two_class_data_fails_closed(self):
        with self.assertRaises(ValueError):
            FrozenLogit.fit([{"state": [0] * 7, "y": 1, "label_end": "2022-12-01",
                              "signal_date": "2022-11-30"} for _ in range(12)])
        with self.assertRaises(ValueError):
            FrozenLogit.fit(self.rows()[:9])


class GateTests(unittest.TestCase):
    days = [f"2022-01-{d:02d}" for d in range(4, 10)]
    codes = [f"sh.6000{i:02d}" for i in range(6, 11)]

    def make_replay(self, policy, falling=False, stop_loss=0.):
        frames, bars = {}, {}
        prices = [10., 9.5, 9., 8.5, 8., 7.5] if falling else [10.] * len(self.days)
        for code in self.codes:
            frame = pd.DataFrame([dict(date=day, code=code,
                                       open=(10. if i == 0 else (prices[i-1] if code == self.codes[0] else 10.)),
                                       high=10., low=7.5,
                                       close=(prices[i] if code == self.codes[0] else 10.),
                                       preclose=(10. if i == 0 else (prices[i-1] if code == self.codes[0] else 10.)),
                                       volume=100000., tradestatus=1, isST=0)
                                      for i, day in enumerate(self.days)])
            frames[code] = frame
            bars[code] = frame.set_index("date").to_dict("index")
        signals = pd.DataFrame([{"date": self.days[0], "code": self.codes[0], "score": 1.}])
        context = {d: {"breadth": .60, "market_ret5": 0.} for d in self.days}
        cfg = Config(capital=30000., order_budget=5000., max_positions=3,
                     max_exposure=.9, hold_days=10, stop_loss=stop_loss, max_buy_gap=.10)
        rep = RLReplay(frames, self.days, signals, [], cfg, vol_map={},
                       daily_limits={d: 1 for d in self.days}, breadth_lookup={d: .60 for d in self.days},
                       enable_cash_overlay=False, policy=policy, context=context)
        rep.frames, rep.bars = frames, bars
        return rep, (frames, bars, signals, cfg)

    def test_batch_gate_observes_prior_close_and_skips_or_follows_whole_batch(self):
        class Gate:
            def __init__(self, action): self.action = action
            def decide(self, state, allowed): return (self.action if self.action in allowed else 0), .25

        gate = GateAdapter(Gate(1))
        skipped, _ = self.make_replay(gate)
        gate.replay = skipped
        skipped.run()
        self.assertEqual(len(gate.observations), len(self.days) - 1)
        first = gate.observations[0]
        self.assertEqual(first["signal_date"], self.days[0])
        self.assertEqual(first["execution_date"], self.days[1])
        self.assertEqual([o["code"] for o in first["orders"]], [self.codes[0]])
        self.assertEqual(first["action"], 1)
        self.assertFalse(any(f["side"] == "buy" for f in skipped.fills))

        follow, args = self.make_replay(FollowPolicy())
        follow.run()
        frames, bars, signals, cfg = args
        baseline = arena.simulate(frames, bars, self.days, signals, [], cfg, {},
                                  {d: .60 for d in self.days}, {d: 1 for d in self.days})
        self.assertEqual(follow.equity, baseline.equity)
        self.assertEqual(follow.fills, baseline.fills)
        self.assertEqual(follow.trades, baseline.trades)

    def test_adapter_rejects_training_and_does_not_skip_when_no_pending_batch(self):
        adapter = GateAdapter(BreadthGate())
        rep, _ = self.make_replay(adapter)
        adapter.replay = rep
        rep.equity.append({"date": self.days[0], "equity": rep.config.capital})
        with self.assertRaisesRegex(ValueError, "frozen gate"):
            adapter.choose((2,), [0, 1], training=True)
        adapter.choose((2,), [0])
        self.assertEqual(adapter.observations[-1]["action"], 0)
        self.assertEqual(adapter.observations[-1]["orders"], [])

    def test_legacy_rl_masks_only_after_original_full_vote(self):
        class Fixed:
            def __init__(self, action): self.action = action; self.allowed = []
            def choose(self, state, allowed): self.allowed.append(list(allowed)); return self.action
        policies = [Fixed(3), Fixed(1), Fixed(0)]
        gate = LegacyRLGate(VotePolicy(policies))
        # Full vote ties at 0, 1, 3 and picks 0; masking votes first would turn all into
        # 1, 1, 0 and incorrectly select cancellation action 1.
        self.assertEqual(gate.decide((1,), [0, 1])[0], 0)
        self.assertEqual([p.allowed for p in policies], [[[0, 1]]] * 3)
        self.assertEqual(VotePolicy([Fixed(3 & 1), Fixed(1 & 1), Fixed(0 & 1)]).choose((1,), [0, 1]), 1)

    def test_stop_loss_still_runs_under_follow_gate(self):
        rep, _ = self.make_replay(FollowPolicy(), falling=True, stop_loss=.03)
        rep.run()
        self.assertTrue(any(t["exit_reason"] == "stop_close_next_open" for t in rep.trades))


if __name__ == "__main__":
    unittest.main()
