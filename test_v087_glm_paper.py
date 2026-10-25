"""Unit tests for the V0.8.7-GLM forward parallel paper batch (built by GLM checkpoint)."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import v087_glm_paper as paper
from engine import Config
from engine_core import BaselineRegimeStopping
from model_v087_glm import BreadthGatekeeperGLM, TouchModeStoppingGLM


EXPECTED_ACCOUNTS = [
    'v087_glm',
    'v08_verified_ref',
    'v087_glm_random_2026101401',
    'v087_glm_random_2026101402',
    'v087_glm_random_2026101403',
    'cash',
]


class TestAccountsAndProtocol(unittest.TestCase):
    def test_six_expected_accounts(self):
        acc = paper.accounts()
        self.assertEqual(list(acc), EXPECTED_ACCOUNTS)

    def test_candidate_spec(self):
        spec = paper.accounts()['v087_glm']
        self.assertEqual(spec['policy'], 'v087_glm')
        self.assertAlmostEqual(spec['breadth_floor'], 0.58)
        self.assertEqual(spec['intraday_mode'], 'sl')
        self.assertFalse(spec['turnover_damping'])
        self.assertAlmostEqual(spec['damping_hurdle'], 0.0)
        self.assertIsNone(spec['seed'])

    def test_baseline_reference_spec(self):
        spec = paper.accounts()['v08_verified_ref']
        self.assertEqual(spec['model_type'], 'v08_verified')
        self.assertEqual(spec['intraday_mode'], 'baseline')

    def test_random_controls_use_candidate_engine(self):
        for seed in (2026101401, 2026101402, 2026101403):
            spec = paper.accounts()[f'v087_glm_random_{seed}']
            self.assertEqual(spec['policy'], 'v087_glm')
            self.assertEqual(spec['seed'], seed)
            self.assertAlmostEqual(spec['breadth_floor'], 0.58)
            self.assertEqual(spec['intraday_mode'], 'sl')

    def test_protocol_declares_glm_identity_and_rules(self):
        text = paper.protocol_text()
        for fragment in ['0.58', 'GLM', 'REJECT_DAMPING', 'pred_score_v08', '6个独立账户', '15:30']:
            self.assertIn(fragment, text)

    def test_session_path_carries_glm_marker(self):
        self.assertIn('-glm-', paper.SESSION.name)


class TestBuildEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Config(capital=30000, order_budget=7500, max_positions=5, max_exposure=0.90, hold_days=10)
        cls.common = dict(
            frames={},
            calendar=['2026-01-05', '2026-01-06'],
            signals=pd.DataFrame(columns=['date', 'code', 'score']),
            events=[],
            vol={},
            yz_vol={},
            limits={'2026-01-05': 1},
            bm={'2026-01-05': 0.6},
        )

    def test_candidate_engine_assembly(self):
        rep = paper.build_engine(paper.accounts()['v087_glm'], cfg=self.cfg, **self.common)
        self.assertIsInstance(rep.gatekeeper, BreadthGatekeeperGLM)
        self.assertAlmostEqual(rep.gatekeeper.BREADTH_FLOOR, 0.58)
        self.assertIsInstance(rep.stopping, TouchModeStoppingGLM)
        self.assertEqual(rep.stopping.intraday_mode, 'sl')
        self.assertTrue(rep.enable_intraday_touch)
        self.assertFalse(rep.enable_turnover_damping)
        self.assertAlmostEqual(rep.turnover_hurdle, 0.0)

    def test_baseline_reference_engine_assembly(self):
        rep = paper.build_engine(paper.accounts()['v08_verified_ref'], cfg=self.cfg, **self.common)
        self.assertIsInstance(rep.stopping, BaselineRegimeStopping)
        self.assertFalse(rep.enable_intraday_touch)

    def test_cash_engine_assembly(self):
        rep = paper.build_engine(paper.accounts()['cash'], cfg=self.cfg, **self.common)
        self.assertIsInstance(rep.stopping, BaselineRegimeStopping)
        self.assertFalse(rep.enable_intraday_touch)


class TestInitializeGuard(unittest.TestCase):
    def test_initialize_refuses_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'batch'
            target.mkdir()
            with self.assertRaises(ValueError) as ctx:
                paper.initialize(target)
            self.assertIn('Batch exists', str(ctx.exception))


class TestFrozenSourceManifest(unittest.TestCase):
    def test_glm_sources_cover_candidate_model_and_engines(self):
        import json
        session = paper.SESSION
        manifest_path = session / 'runtime-manifest.json'
        if not manifest_path.exists():
            self.skipTest('GLM batch not initialized yet')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        for name in ['model_v087_glm.py', 'engine_core.py', 'engine_intraday.py', 'run_v087_glm_day.py']:
            self.assertTrue(
                name in manifest.get('glm_sources', {}) or name in manifest.get('operational_sources', {}),
                f'{name} must be digest-frozen by the GLM batch',
            )


if __name__ == '__main__':
    unittest.main()
