import unittest

import pandas as pd

import arena
from engine import Config
from v09_rl_2022 import (
    FollowPolicy,
    QPolicy,
    RLReplay,
    VotePolicy,
    available_actions,
    reward,
)


class FixedPolicy:
    def __init__(self, action=0):
        self.action = action

    def choose(self, state, allowed, **kwargs):
        return self.action if self.action in allowed else 0


class V09RLTests(unittest.TestCase):
    days = ["2022-01-04", "2022-01-05", "2022-01-06", "2022-01-07"]
    code = "sh.600006"

    def make_data(self, closes=None, missing_day=None, limit_down_day=None):
        closes = closes or [10., 10., 10., 10.]
        rows = []
        for i, (day, close) in enumerate(zip(self.days, closes)):
            open_price = 9. if day == limit_down_day else 10.
            preclose = 10. if i == 0 else closes[i - 1]
            rows.append(dict(date=day, code=self.code, open=open_price, high=10.,
                             low=open_price, close=close, preclose=preclose,
                             volume=100000., tradestatus=1, isST=0))
        frame = pd.DataFrame(rows)
        if missing_day:
            frame = frame[frame.date != missing_day]
        frames = {self.code: frame}
        bars = {self.code: frame.set_index("date").to_dict("index")}
        signals = pd.DataFrame([dict(date=self.days[0], code=self.code, score=1.)])
        context = {d: {"breadth": .60, "market_ret5": 0.} for d in self.days}
        cfg = Config(capital=30000., order_budget=5000., max_positions=3,
                     max_exposure=.5, hold_days=3, stop_loss=0., max_buy_gap=.10)
        return frames, bars, signals, context, cfg

    def replay(self, policy=None, **data_kwargs):
        frames, bars, signals, context, cfg = self.make_data(**data_kwargs)
        rep = RLReplay(frames, self.days, signals, [], cfg, vol_map={}, daily_limits={d: 1 for d in self.days},
                       breadth_lookup={d: .60 for d in self.days}, enable_cash_overlay=False,
                       policy=policy or FollowPolicy(), context=context)
        return rep, (frames, bars, signals, context, cfg)

    def test_action_masking(self):
        self.assertEqual(available_actions(False, False), [0])
        self.assertEqual(available_actions(True, False), [0, 1])
        self.assertEqual(available_actions(False, True), [0, 2])
        self.assertEqual(available_actions(True, True), [0, 1, 2, 3])

    def test_q_learning_bootstrap_and_terminal_target(self):
        q = QPolicy(1, alpha=1., gamma=.5)
        q.values[("next",)] = __import__("numpy").array([0., 8., 4., 0.])
        q.update(("s",), 0, 2., ("next",), [0, 2])
        self.assertEqual(q.values[("s",)][0], 4.)
        q.update(("s",), 1, 2., ("next",), [0, 2], terminal=True)
        self.assertEqual(q.values[("s",)][1], 2.)
        self.assertEqual(q.update_count, 2)

    def test_reward_uses_nav_and_penalizes_drawdown_increase(self):
        self.assertAlmostEqual(reward(100., .10, 110., .10), 10000 * __import__("math").log(1.1))
        self.assertAlmostEqual(reward(100., .10, 100., .20), -250.)
        with self.assertRaisesRegex(ValueError, "Nonpositive"):
            reward(100., 0., 0., 0.)

    def test_dump_restore_and_eval_do_not_update(self):
        q = QPolicy(7)
        q.values[(1,)] = __import__("numpy").array([1., 2., 3., 4.])
        q.visits[(1,)] = __import__("numpy").array([10, 11, 12, 13])
        q.update_count = 9
        clone = QPolicy.restore(q.dump())
        self.assertEqual(clone.dump(), q.dump())
        before = clone.dump()
        self.assertEqual(clone.choose((1,), [0, 1, 2, 3]), 3)
        self.assertEqual(clone.update_count, 9)
        self.assertEqual(clone.dump(), before)
        self.assertEqual(VotePolicy([clone, clone]).choose((1,), [0, 1, 2, 3]), 3)
        self.assertEqual(clone.dump(), before)

    def test_follow_policy_matches_arena_simulate_on_multiday_synthetic_market(self):
        rep, (frames, bars, signals, _context, cfg) = self.replay()
        rep.run()
        baseline = arena.simulate(frames, bars, self.days, signals, [], cfg, {},
                                  {d: .60 for d in self.days}, {d: 1 for d in self.days})
        self.assertEqual([r["date"] for r in rep.equity], self.days)
        self.assertEqual(rep.equity, baseline.equity)
        self.assertEqual(rep.fills, baseline.fills)
        self.assertEqual(rep.trades, baseline.trades)

    def test_current_close_is_not_used_for_same_day_preopen_decision(self):
        a, _ = self.replay(closes=[10., 10., 10., 10.])
        b, _ = self.replay(closes=[10., 12., 10., 10.])
        a.run()
        b.run()
        # Changing Jan 5 close cannot change the Jan 5 opening decision, sourced from Jan 4.
        self.assertEqual(a.decisions[0]["execution_date"], self.days[1])
        self.assertEqual(a.decisions[0]["signal_date"], self.days[0])
        self.assertEqual(a.decisions[0]["state"], b.decisions[0]["state"])
        self.assertEqual(a.decisions[0]["action"], b.decisions[0]["action"])

    def test_exit_is_next_open_t_plus_one_and_limit_down_failure_waits(self):
        rep, _ = self.replay(policy=FixedPolicy(2), limit_down_day=self.days[2])
        rep.run()
        sells = [f for f in rep.fills if f["side"] == "sell"]
        self.assertEqual(len(sells), 1)
        self.assertEqual(sells[0]["date"], self.days[3])
        self.assertEqual(rep.rejected["limit_down_open"], 1)
        self.assertFalse(rep.positions)
        self.assertEqual(sells[0]["trade_id"], next(f["trade_id"] for f in rep.fills if f["side"] == "buy"))

    def test_missing_held_quote_is_blocked_and_final_day_is_not_force_closed(self):
        missing, _ = self.replay(missing_day=self.days[2])
        with self.assertRaisesRegex(ValueError, "Blocked quality issue") as caught:
            missing.run()
        self.assertTrue(any(w["issue"] == "missing_held_stock_quote" for w in missing.review_warnings))

        final, _ = self.replay()
        final.run()
        self.assertTrue(final.positions)
        self.assertFalse(any(f["side"] == "sell" and f["date"] == self.days[-1] for f in final.fills))


if __name__ == "__main__":
    unittest.main()
