"""model_v086_synthesized.py - Grand Synthesized Model V0.8.6.

Unifies:
1. ModularReplayEngine (from engine_core.py)
2. AdaptiveTopRanker with Turnover Damping (1.2% belief excess threshold, friction penalty, open slots exemption)
3. GatekeeperV082 selective classification (from model_v082_gatekeeper.py)
4. IntradayTouchStopping with winning Plan A dynamic 50% floating floor (from engine_intraday.py)

Evaluates performance across 2022-2025 continuous period and 2026 out-of-sample period under
normal (5 bps) and double slippage (10 bps) stress.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import arena
from engine import Config, Position, tick
from engine_core import (
    ExitDecision,
    GatekeeperProtocol,
    MarketContext,
    ModularReplayEngine,
    OptimalStoppingProtocol,
    PositionState,
    RankerProtocol,
    audit_ledger,
)
from engine_intraday import (
    IntradayTouchStateMachine,
    IntradayTouchStopping,
    compute_intraday_stop_levels,
)
from model_v082_gatekeeper import (
    MetaGatekeeperV082,
    train_walk_forward_gatekeeper,
)
from model_v083_ranker import compute_adaptive_scores
from train_model_v08 import extract_microstructure_features

BUNDLE = ROOT / "results/v08-codex-rebuild-20261020-v1"
OUT_DIR = ROOT / "results/v086-synthesized"


# =============================================================================
# 1. Turnover Damping Evaluator & Protocol Conformance
# =============================================================================

class TurnoverDampingEvaluator:
    """Turnover Damping & 1.2% Belief Excess Threshold Evaluator.

    Rules:
    - Belief excess threshold: candidate Delta E[R] >= 0.0120 (1.2%) over weakest holding to approve swap.
    - Exact boundary: 0.01199 rejected vs 0.01201 accepted; 0.0000 rejected.
    - Empty portfolio / open slots exemption: approved without damping penalty when slots are open.
    """
    HURDLE: float = 0.0120  # 1.2% (120 bps)

    @classmethod
    def evaluate_swap(
        cls,
        candidate_pred_er: float,
        holdings_pred_ers: List[float],
        open_slots: int = 0,
        hurdle: Optional[float] = None,
    ) -> Tuple[bool, float, str]:
        """Evaluates whether candidate passes the conviction excess hurdle over weakest holding."""
        eff_hurdle = cls.HURDLE if hurdle is None else hurdle

        if open_slots > 0:
            return True, 0.0, "approved_open_slot"

        if not holdings_pred_ers:
            return True, 0.0, "approved_empty_portfolio"

        weakest_er = min(holdings_pred_ers)
        delta_er = float(candidate_pred_er - weakest_er)

        if delta_er >= eff_hurdle:
            return True, delta_er, "approved_conviction_excess"
        else:
            return False, delta_er, "rejected_damping_insufficient_excess"


# Alias for backwards compatibility with tests importing ReferenceTurnoverDamping
ReferenceTurnoverDamping = TurnoverDampingEvaluator


class AdaptiveTopRanker:
    """Adaptive Top-Loss Ranker with Turnover Damping conforming to RankerProtocol."""

    def __init__(
        self,
        hurdle: float = 0.0120,
        ranking_friction_penalty: float = 0.0010,
        max_positions: int = 5,
    ):
        self.hurdle = hurdle
        self.ranking_friction_penalty = ranking_friction_penalty
        self.max_positions = max_positions

    def score_and_rank(
        self,
        candidates: pd.DataFrame,
        date: str,
        breadth: float,
        context: MarketContext,
        turnover_damping: bool = False,
    ) -> pd.DataFrame:
        """Score and sort daily candidates. Conforms to RankerProtocol."""
        return self.rank_candidates(date, candidates, context, positions=None, turnover_damping=turnover_damping)

    def rank_candidates(
        self,
        day: str,
        candidates: pd.DataFrame,
        context: MarketContext,
        positions: Optional[Dict[str, PositionState]] = None,
        turnover_damping: bool = False,
    ) -> pd.DataFrame:
        """Rank candidates for evening queueing with ranking friction penalty and turnover damping."""
        if candidates.empty:
            return candidates

        df = candidates.copy()
        score_col = "score" if "score" in df.columns else ("pred_score_v083" if "pred_score_v083" in df.columns else "pred_score_v08")
        er_col = "pred_net_return" if "pred_net_return" in df.columns else "pred_net_return_v2"

        # 1. Ranking friction penalty for non-held candidates
        held_codes = set(positions.keys()) if positions else set()
        if self.ranking_friction_penalty > 0 and len(held_codes) > 0:
            is_non_held = ~df["code"].isin(held_codes)
            df["effective_score"] = df[score_col] - np.where(is_non_held, self.ranking_friction_penalty, 0.0)
        else:
            df["effective_score"] = df[score_col]

        # 2. Turnover Damping: Belief Excess Hurdle
        if turnover_damping and positions and len(positions) > 0:
            open_slots = max(0, context.max_positions - len(positions))
            # Open slots exemption: when open slots exist, no damping penalty
            if open_slots == 0:
                holdings_ers = [p.current_gain for p in positions.values()]
                weakest_er = min(holdings_ers) if holdings_ers else 0.0
                if er_col in df.columns:
                    passes_hurdle = (df["code"].isin(held_codes)) | ((df[er_col] - weakest_er) >= self.hurdle)
                    df["damping_penalty"] = np.where(passes_hurdle, 0.0, -999.0)
                    df["effective_score"] = df["effective_score"] + df["damping_penalty"]
                    df = df[df["effective_score"] > -500.0]
                    if df.empty:
                        return df

        # Sort descending by effective_score, tie-break by code ascending
        df = df.sort_values(by=["effective_score", "code"], ascending=[False, True]).reset_index(drop=True)
        return df

    def evaluate_turnover_damping(
        self,
        new_candidate: Dict[str, Any],
        held_positions: Any,
        hurdle: float = 0.0120,
    ) -> bool:
        """Conforms to RankerProtocol."""
        candidate_er = new_candidate.get("pred_net_return", new_candidate.get("pred_net_return_v2", 0.0))
        if isinstance(held_positions, dict):
            holdings_ers = [getattr(p, "current_gain", 0.0) for p in held_positions.values()]
        elif isinstance(held_positions, (list, tuple)):
            holdings_ers = [p if isinstance(p, (int, float)) else getattr(p, "current_gain", 0.0) for p in held_positions]
        else:
            holdings_ers = []
        approved, _, _ = TurnoverDampingEvaluator.evaluate_swap(
            candidate_pred_er=candidate_er,
            holdings_pred_ers=holdings_ers,
            open_slots=0,
            hurdle=hurdle,
        )
        return approved


# =============================================================================
# 2. GatekeeperV082 conforming to GatekeeperProtocol
# =============================================================================

class GatekeeperV082:
    """Selective Classification Meta-Gatekeeper conforming to GatekeeperProtocol."""

    def __init__(
        self,
        prob_map: Optional[Dict[Tuple[str, str], float]] = None,
        threshold: float = 0.45,
    ):
        self.prob_map = prob_map or {}
        self.threshold = threshold

    def evaluate_candidate(
        self,
        code: str,
        date: str,
        features: Optional[Dict[str, Any]],
        context: MarketContext,
    ) -> Tuple[bool, float, str]:
        """Conforms to GatekeeperProtocol."""
        prob = float(self.prob_map.get((date, code), 0.50))
        accepted = prob >= self.threshold
        reason = "gatekeeper_pass" if accepted else "gatekeeper_abstain"
        return accepted, prob, reason

    def evaluate_order(
        self,
        candidate_row: Any,
        context: MarketContext,
    ) -> Tuple[bool, float, str]:
        """Conforms to GatekeeperProtocol."""
        if isinstance(candidate_row, dict):
            code = candidate_row.get("code", "")
        else:
            code = getattr(candidate_row, "code", "")
        return self.evaluate_candidate(code, context.date, candidate_row, context)


print("model_v086_synthesized module loaded successfully.")
