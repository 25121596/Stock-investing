import unittest

import pandas as pd

import arena
from engine import Config
from v09_rl_2022 import FollowPolicy, RLReplay, VotePolicy
from v09_rl_ablation import MaskedPolicy, compare_criteria, factorial_effects


class FixedPolicy:
    def __init__(self, action):
        self.action = action
        self.calls = []

    def choose(self, state, allowed, **kwargs):
        self.calls.append((tuple(state), list(allowed)))
        return self.action if self.action in allowed else 0


class AblationReplayTests(unittest.TestCase):
    days = [f"2022-01-{d:02d}" for d in range(4, 10)]
    codes = ["sh.600006", "sh.600007", "sh.600008", "sh.600009", "sh.600010"]

    def make_market(self, falling=False, stop_loss=0.):
        frames, bars = {}, {}
        prices = [10., 9.5, 9., 8.5, 8., 7.5] if falling else [10.] * len(self.days)
        for code in self.codes:
            rows = []
            for i, day in enumerate(self.days):
                close = prices[i] if code == self.codes[0] else 10.
                prev = 10. if i == 0 else (prices[i - 1] if code == self.codes[0] else 10.)
                rows.append(dict(date=day, code=code, open=prev, high=max(prev, close),
                                 low=min(prev, close), close=close, preclose=prev,
                                 volume=100000., tradestatus=1, isST=0))
            frame = pd.DataFrame(rows)
            frames[code] = frame
            bars[code] = frame.set_index("date").to_dict("index")
        signals = pd.DataFrame([
            {"date": day, "code": self.codes[i], "score": 1.}
            for i, day in enumerate(self.days[:-1])
        ])
        context = {d: {"breadth": .60, "market_ret5": 0.} for d in self.days}
        cfg = Config(capital=30000., order_budget=5000., max_positions=3,
                     max_exposure=.9, hold_days=10, stop_loss=stop_loss,
                     max_buy_gap=.10)
        return frames, bars, signals, context, cfg

    def replay(self, policy, mask=None, falling=False, stop_loss=0.):
        frames, bars, signals, context, cfg = self.make_market(falling, stop_loss)
        if mask is not None:
            policy = MaskedPolicy(policy, mask)
        rep = RLReplay(frames, self.days, signals, [], cfg, vol_map={},
                       daily_limits={d: 1 for d in self.days},
                       breadth_lookup={d: .60 for d in self.days},
                       enable_cash_overlay=False, policy=policy, context=context)
        rep.frames, rep.bars = frames, bars
        return rep, (frames, bars, signals, cfg)

    def test_vote_uses_original_allowed_set_then_masks(self):
        allowed = [0, 1, 2, 3]
        policies = [FixedPolicy(3), FixedPolicy(1), FixedPolicy(0)]
        vote = VotePolicy(policies)
        masked = MaskedPolicy(vote, 1)
        # Full vote ties at actions 0, 1, 3, so the original tie-break gives 0.
        self.assertEqual(masked.choose(("s",), allowed), 0)
        self.assertEqual([call[1] for p in policies for call in p.calls], [allowed] * 3)
        self.assertEqual(masked.trace[0]["raw_action"], 0)
        # The incorrect per-seed masking route collapses 3 and 1 to 1, changing the vote.
        wrong_votes = [p.choose(("s",), allowed) & 1 for p in policies]
        self.assertEqual(VotePolicy([FixedPolicy(a) for a in wrong_votes]).choose(("s",), allowed), 1)

    def test_masked_policy_rejects_training_or_epsilon(self):
        policy = MaskedPolicy(FixedPolicy(0), 0)
        with self.assertRaises(ValueError):
            policy.choose((0,), [0], training=True)
        with self.assertRaises(ValueError):
            policy.choose((0,), [0], epsilon=.01)

    def test_baseline_mask_matches_follow_policy_synthetic_ledger(self):
        masked, args = self.replay(FixedPolicy(0), mask=0)
        follow, _ = self.replay(FollowPolicy())
        masked.run(); follow.run()
        for field in ("equity", "fills", "trades", "actions", "warnings",
                      "review_warnings", "pending_buys", "rejected"):
            self.assertEqual(getattr(masked, field), getattr(follow, field), field)
        self.assertEqual(masked.holdings(), follow.holdings())
        frames, bars, signals, cfg = args
        baseline = arena.simulate(frames, bars, self.days, signals, [], cfg, {},
                                  {d: .60 for d in self.days}, {d: 1 for d in self.days})
        self.assertEqual(masked.equity, baseline.equity)
        self.assertEqual(masked.fills, baseline.fills)
        self.assertEqual(masked.trades, baseline.trades)

    def test_full_mask_matches_unmasked_vote_ledger(self):
        def vote():
            return VotePolicy([FixedPolicy(0), FixedPolicy(1), FixedPolicy(3)])
        masked, _ = self.replay(vote(), mask=3)
        raw, _ = self.replay(vote())
        masked.run(); raw.run()
        for field in ("equity", "fills", "trades", "actions", "warnings",
                      "review_warnings", "pending_buys", "rejected"):
            self.assertEqual(getattr(masked, field), getattr(raw, field), field)
        self.assertEqual(masked.decisions, raw.decisions)

    def test_filter_cancels_next_open_order_and_exit_sells_next_open(self):
        filtered, _ = self.replay(FixedPolicy(1), mask=1)
        exited, _ = self.replay(FixedPolicy(2), mask=2)
        filtered.run(); exited.run()
        self.assertTrue(filtered.policy.trace)
        self.assertTrue(any(t["raw_action"] == 1 for t in filtered.policy.trace))
        # The pending order is cancelled before the parent engine's next-open buy loop.
        self.assertFalse(any(f["code"] == self.codes[1] and f["side"] == "buy"
                             for f in filtered.fills))
        self.assertTrue(any(t["raw_action"] == 2 for t in exited.policy.trace))
        self.assertTrue(any(f["side"] == "sell" and f["date"] == self.days[2]
                            for f in exited.fills))
        self.assertTrue(any(t["exit_reason"] == "stop_close_next_open" for t in exited.trades))

    def test_portfolio_paths_recompute_distinct_states_and_keep_stop_loss(self):
        filtered, _ = self.replay(FixedPolicy(1), mask=1)
        exited, _ = self.replay(FixedPolicy(2), mask=2)
        filtered.run(); exited.run()
        # Each account has its own replay; after divergent actions the later state reflects
        # its own positions/pending orders and age rather than a copied full-RL path.
        fstates = {r["signal_date"]: r["state"] for r in filtered.decisions}
        estates = {r["signal_date"]: r["state"] for r in exited.decisions}
        common = sorted(set(fstates) & set(estates))
        self.assertTrue(any(fstates[d] != estates[d] for d in common[1:]))

        baseline, _ = self.replay(FixedPolicy(0), mask=0, falling=True, stop_loss=.03)
        baseline.run()
        self.assertTrue(any(t["exit_reason"] == "stop_close_next_open" for t in baseline.trades))

    def test_compare_criteria_keeps_normal_and_stress_rows_paired(self):
        rows = [
            {"family": "primary", "period": "p", "case": "baseline", "total_return": .10, "max_drawdown": -.20},
            {"family": "primary", "period": "p", "case": "stress_baseline", "total_return": .03, "max_drawdown": -.25},
            {"family": "primary", "period": "p", "case": "filter_only", "total_return": .11, "max_drawdown": -.19},
            {"family": "primary", "period": "p", "case": "stress_filter_only", "total_return": .04, "max_drawdown": -.24},
        ]
        self.assertTrue(compare_criteria(rows, "filter_only", "p")["passed"])
        rows[3]["total_return"] = .02
        self.assertFalse(compare_criteria(rows, "filter_only", "p")["higher_stressed_return"])
        self.assertTrue(compare_criteria(rows, "exit_only", "p")["missing_or_failed"])

    def test_factorial_effects_mapping_interaction_and_missing_arm(self):
        rows = {"baseline": {"final_equity": 100.},
                "filter_only": {"final_equity": 110.},
                "exit_only": {"final_equity": 120.},
                "full_rl": {"final_equity": 140.}}
        effects = factorial_effects(rows)
        self.assertEqual(effects["filter_without_exit_rmb"], 10.)
        self.assertEqual(effects["exit_without_filter_rmb"], 20.)
        self.assertEqual(effects["filter_with_exit_rmb"], 20.)
        self.assertEqual(effects["exit_with_filter_rmb"], 30.)
        self.assertEqual(effects["interaction_rmb"], 10.)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            factorial_effects({k: v for k, v in rows.items() if k != "exit_only"})


if __name__ == "__main__":
    unittest.main()
