"""model_v087_glm.py - V0.8.7 Research Candidate (built by GLM checkpoint).

Pre-registered experiment, fixed before any result was viewed (2026-10-25).

Diagnosis being tested (from results/v086-synthesized/comparison.csv, 2026-10-23):
  - Intraday touch execution (V0.8.6 plan_a) costs ~19.7pp in 2022-2025
    (v085 close-stopping +43.24% vs v086_no_damping +23.53%) while helping 2026
    risk (MaxDD -4.51% -> -2.70%, Calmar 8.41). The suspected damage source is the
    INTRADAY take-profit / plan_a floor legs, which truncate long winners that the
    2022-2025 period depends on; the intraday STOP-LOSS leg is the protective part.
  - Turnover damping (1.2% hurdle + 0.65% conviction + 10bps penalty) reduced
    returns in BOTH periods (23.53->21.46 and 17.02->12.79) with marginal DD benefit.

Pre-registered hypotheses (no re-tuning after viewing results):
  H1 (attribution): in 2022-2025 normal cost, tp_touch_only <= close_only_ref
      and sl_touch_only >= touch_full_ref.
  H2 (candidate): v087_glm (breadth gate >= 0.58 + SL-touch-only intraday +
      close-based profit exits) passes ALL hurdles:
        2022-2025 net return > +45% and MaxDD >= -8.32%;
        2026 net return > +15% and MaxDD >= -3.99%;
        stressed (10bps) candidate return > stressed v08_baseline in both periods.
  H3 (damping recheck, single pre-registered setting): v087_glm_damping
      (hurdle 0.005, friction 5bps, conviction 0.0065) is kept ONLY IF
      2022-2025 return >= 90% of v087_glm AND 2026 return >= v08_baseline;
      otherwise damping is rejected and stays OFF in the candidate.

Engine parity self-checks inside this run (not acceptance criteria):
  close_only_ref should approximate v085 synthesis (+43.24% in 2022-2025) and
  touch_full_ref should approximate v086_no_damping (+23.53%); both use the same
  signals/ranker/gatekeeper as the v086 evaluation.

Breadth gate semantics replicate the pre-registered V0.9 fixed rule
(v09_rl_2022.py observe_close: breadth_bin = searchsorted([.45,.58], breadth),
BreadthGate accepts the next-day buy batch only when bin == 2, i.e. breadth >= 0.58).
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import time
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import arena
from engine import Config
from engine_core import (
    BaselineRanker,
    ExitDecision,
    MarketContext,
    ModularReplayEngine,
    audit_ledger,
)
from engine_intraday import IntradayTouchStopping
from model_v082_gatekeeper import train_walk_forward_gatekeeper
from model_v083_ranker import compute_adaptive_scores
from model_v086_synthesized import BUNDLE, AdaptiveTopRanker, GatekeeperV082
from train_model_v2 import enrich_features_v2
from train_model_v3 import extract_ashare_features
from train_model_v08 import extract_microstructure_features

OUT_DIR = ROOT / "results/v087-glm-v1"
PERIODS = [
    ("2022-2025", "2022-01-01", "2025-12-31"),
    ("2026", "2026-01-01", "2026-10-13"),
]

# Pre-registered acceptance hurdles (Tier-4 semantics, now evaluated on real replay rows)
HURDLES = {
    "p1_return_gt": 0.4500,
    "p1_maxdd_ge": -0.0832,
    "p2_return_gt": 0.1500,
    "p2_maxdd_ge": -0.0399,
}
CANDIDATE_CASE = "v087_glm"
RANDOM_SEEDS = [2026101401, 2026101402, 2026101403]

INTRADAY_REASON_PREFIX = "intraday_"


# =============================================================================
# 1. Breadth Gatekeeper (V0.9 pre-validated fixed rule) - GLM
# =============================================================================

class BreadthGatekeeperGLM:
    """Fixed breadth floor gate (V0.9 pre-registered rule), conforming to GatekeeperProtocol.

    Accepts next-morning buy candidates only when signal-day market breadth
    (fraction of pool with positive trailing 5-day adjusted returns) >= 0.58.
    """

    BREADTH_FLOOR: float = 0.58

    def __init__(self, breadth_floor: Optional[float] = None):
        if breadth_floor is not None:
            self.BREADTH_FLOOR = float(breadth_floor)

    def evaluate_candidate(
        self,
        code: str,
        date: str,
        features: Optional[Dict[str, Any]],
        context: MarketContext,
    ) -> Tuple[bool, float, str]:
        breadth = float(context.breadth5)
        accepted = breadth >= self.BREADTH_FLOOR
        reason = "breadth_floor_pass" if accepted else "breadth_floor_block"
        return accepted, breadth, reason

    def evaluate_order(
        self,
        candidate_row: Any,
        context: MarketContext,
    ) -> Tuple[bool, float, str]:
        if isinstance(candidate_row, dict):
            code = candidate_row.get("code", "")
        else:
            code = getattr(candidate_row, "code", "")
        return self.evaluate_candidate(code, context.date, candidate_row, context)


# =============================================================================
# 2. Mode-Restricted Intraday Touch Stopping - GLM
# =============================================================================

class TouchModeStoppingGLM(IntradayTouchStopping):
    """Intraday touch stopping with a configurable intraday rule set.

    intraday_mode:
      'sl'  -> ONLY the stop-loss touch leg executes intraday (10 bps penalty kept).
               Take-profit and plan_a floor legs stay close-based (exit next morning
               at open through the inherited check_close_stopping).
      'tp'  -> ONLY take-profit / plan_a floor legs execute intraday (attribution
               ablation); stop-loss stays close-based. Note: on a dual-touch bar the
               underlying state machine already returns the stop-loss decision first,
               so any profit-side decision returned here implies the stop-loss line
               was NOT touched - the anti-hindsight priority is preserved.
      'all' -> full V0.8.6 plan_a behaviour (replication reference).

    Intraday exit reasons are relabelled with an 'intraday_' prefix so trades.csv
    distinguishes intraday executions from close-detected, next-morning exits.
    Labelling only; execution prices and fees are identical to the parent class.
    """

    _SL_REASONS = ("stop_loss", "stop_loss_intraday_priority")

    def __init__(self, plan_mode: str = "plan_a", intraday_mode: str = "sl"):
        super().__init__(plan_mode=plan_mode)
        if intraday_mode not in ("sl", "tp", "all"):
            raise ValueError(f"Unknown intraday_mode: {intraday_mode!r}. Must be 'sl', 'tp' or 'all'.")
        self.intraday_mode = intraday_mode

    def check_intraday_touch(self, position, bar, context) -> ExitDecision:
        decision = self.state_machine.evaluate_bar(position, bar, context)
        if decision is None or not decision.should_exit:
            return ExitDecision(should_exit=False)

        reason = decision.exit_reason
        is_sl = reason in self._SL_REASONS
        if self.intraday_mode == "all":
            keep = True
        elif self.intraday_mode == "sl":
            keep = is_sl
        else:  # 'tp'
            keep = not is_sl
        if not keep:
            return ExitDecision(should_exit=False)

        decision.exit_reason = INTRADAY_REASON_PREFIX + reason
        return decision


# =============================================================================
# 3. Acceptance verdict computed from real replay rows
# =============================================================================

def compute_acceptance(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluates the pre-registered hurdles against real result rows.

    Returns a verdict dict with one boolean per criterion plus an overall pass.
    """

    def row(period: str, case: str, stress: bool) -> Optional[Dict[str, Any]]:
        # Stress rows carry a 'stress_' prefix baked into their case name by the
        # evaluation loop; non-stress rows use the bare case name.
        eff_case = f"stress_{case}" if stress else case
        for r in rows:
            if r.get("period") == period and r.get("case") == eff_case:
                return r
        return None

    cand_p1 = row("2022-2025", CANDIDATE_CASE, False)
    cand_p2 = row("2026", CANDIDATE_CASE, False)
    base_p1s = row("2022-2025", "v08_baseline", True)
    base_p2s = row("2026", "v08_baseline", True)
    cand_p1s = row("2022-2025", CANDIDATE_CASE, True)
    cand_p2s = row("2026", CANDIDATE_CASE, True)

    criteria: Dict[str, Dict[str, Any]] = {}
    for name, rr, key, hurdle, ok in [
        ("p1_return_gt_45pct", cand_p1, "total_return", HURDLES["p1_return_gt"], None),
        ("p1_maxdd_not_worse_than_-8.32pct", cand_p1, "max_drawdown", HURDLES["p1_maxdd_ge"], None),
        ("p2_return_gt_15pct", cand_p2, "total_return", HURDLES["p2_return_gt"], None),
        ("p2_maxdd_not_worse_than_-3.99pct", cand_p2, "max_drawdown", HURDLES["p2_maxdd_ge"], None),
    ]:
        if rr is None:
            criteria[name] = {"passed": False, "actual": None, "hurdle": hurdle, "missing_row": True}
            continue
        actual = float(rr[key])
        passed = actual > hurdle if key == "total_return" else actual >= hurdle
        criteria[name] = {"passed": bool(passed), "actual": actual, "hurdle": hurdle}

    for name, cand, base in [
        ("stress_p1_beats_stressed_baseline", cand_p1s, base_p1s),
        ("stress_p2_beats_stressed_baseline", cand_p2s, base_p2s),
    ]:
        if cand is None or base is None:
            criteria[name] = {"passed": False, "actual": None, "hurdle": "> stressed baseline", "missing_row": True}
            continue
        actual = float(cand["total_return"])
        passed = actual > 0.0 and actual > float(base["total_return"])
        criteria[name] = {
            "passed": bool(passed),
            "actual": actual,
            "hurdle": f"> 0 and > stressed baseline {float(base['total_return']):.4f}",
        }

    overall = all(c["passed"] for c in criteria.values())
    return {
        "candidate_case": CANDIDATE_CASE,
        "cost_basis": "normal (5 bps slippage) unless criterion name says stressed",
        "criteria": criteria,
        "overall_pass": bool(overall),
        "pre_registered": True,
        "registered_at": "2026-10-25 (before any V0.8.7 result was viewed)",
        "made_by_checkpoint": "GLM",
    }


def compute_damping_verdict(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pre-registered H3: keep damping only if it keeps >=90% of candidate 2022-2025
    return AND does not undercut the v08 baseline 2026 return."""

    def row(period: str, case: str) -> Optional[Dict[str, Any]]:
        for r in rows:
            if r.get("period") == period and r.get("case") == case and not bool(r.get("stress")):
                return r
        return None

    cand = row("2022-2025", CANDIDATE_CASE)
    damped = row("2022-2025", "v087_glm_damping")
    cand_26 = row("2026", CANDIDATE_CASE)
    damped_26 = row("2026", "v087_glm_damping")
    base_26 = row("2026", "v08_baseline")
    if None in (cand, damped, cand_26, damped_26, base_26):
        return {"verdict": "INCONCLUSIVE", "reason": "missing rows", "made_by_checkpoint": "GLM"}

    keep_p1 = float(damped["total_return"]) >= 0.9 * float(cand["total_return"])
    keep_p2 = float(damped_26["total_return"]) >= float(base_26["total_return"])
    verdict = "KEEP_DAMPING" if (keep_p1 and keep_p2) else "REJECT_DAMPING"
    return {
        "verdict": verdict,
        "keep_p1_90pct_of_candidate": bool(keep_p1),
        "keep_p2_not_below_v08_baseline": bool(keep_p2),
        "damped_p1_return": float(damped["total_return"]),
        "candidate_p1_return": float(cand["total_return"]),
        "damped_p2_return": float(damped_26["total_return"]),
        "baseline_p2_return": float(base_26["total_return"]),
        "settings": {"hurdle": 0.005, "ranking_friction_penalty": 0.0005, "conviction_hurdle": 0.0065},
        "pre_registered": True,
        "made_by_checkpoint": "GLM",
    }


def attribution_breakdown(out_dir: Path, cases: List[str]) -> Dict[str, Any]:
    """Aggregates per-exit-reason trade stats from the saved case directories."""
    breakdown: Dict[str, Any] = {}
    for period, _, _ in PERIODS:
        for case in cases:
            folder = out_dir / f"{period}_{case}"
            trades_file = folder / "trades.csv"
            if not trades_file.exists():
                continue
            trades = pd.read_csv(trades_file)
            if trades.empty:
                breakdown[f"{period}|{case}"] = {"trades": 0}
                continue
            grp = (
                trades.groupby("exit_reason")
                .agg(count=("net_pnl", "size"), total_net_pnl=("net_pnl", "sum"), avg_net_return=("net_return", "mean"))
                .round(6)
            )
            breakdown[f"{period}|{case}"] = {
                "trades": int(len(trades)),
                "by_exit_reason": grp.to_dict("index"),
                "intraday_exit_trades": int(trades.exit_reason.str.startswith(INTRADAY_REASON_PREFIX).sum()),
            }
    return {"made_by_checkpoint": "GLM", "breakdown": breakdown}


# =============================================================================
# 4. Evaluation pipeline
# =============================================================================

def run_v087_glm_evaluation(out_dir: Path = OUT_DIR) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 108)
    print("RUNNING V0.8.7-GLM RESEARCH CANDIDATE EVALUATION (breadth gate + SL-touch-only intraday)")
    print("=" * 108)

    _, frames, calendar, raw = arena.load(arena.DATASET)
    events, _ = arena.canonicalize_events(raw)
    bars = arena.prepare_bars(frames)

    panel = pd.read_csv(
        ROOT / "results/midterm-access-v3/panel-h10.csv", dtype={"date": str, "code": str, "label_end": str}
    )
    sixty = pd.read_csv(ROOT / "results/exploration-v1/new-features.csv", dtype={"date": str, "code": str})
    panel = enrich_features_v2(
        panel.merge(sixty[["date", "code", "adj_ret60", "ret_vol60"]], on=["date", "code"], how="left", validate="one_to_one")
    )
    panel = panel.merge(extract_ashare_features(frames), on=["date", "code"], how="left", validate="one_to_one").merge(
        extract_microstructure_features(frames), on=["date", "code"], how="left", validate="one_to_one"
    )

    b = arena.breadth_days(panel)
    bm = b.set_index("date").breadth5.to_dict()
    pm = b.set_index("date").pool_size.to_dict()
    limits = {d: 0 if pm[d] < 100 or x < 0.45 else (1 if x < 0.55 else 2) for d, x in bm.items()}
    vol = panel.set_index(["date", "code"]).ret_vol20.to_dict()
    yz_vol_map = panel.set_index(["date", "code"]).yz_vol20.to_dict()

    old2 = pd.read_csv(
        BUNDLE / "v2/predictions-h10-v2.csv", dtype={"date": str, "code": str}, float_precision="round_trip"
    )
    old8 = pd.read_csv(
        BUNDLE / "v08/predictions-h10-v08.csv", dtype={"date": str, "code": str}, float_precision="round_trip"
    )
    preds_v083 = pd.read_csv(
        ROOT / "results/v083-adaptive-ranker/predictions-v083.csv", dtype={"date": str, "code": str}
    )
    cand_df = pd.read_csv(ROOT / "results/v082-meta-gatekeeper/candidates.csv", dtype={"date": str, "code": str})
    gatekeeper_model, _ = train_walk_forward_gatekeeper(cand_df)
    gatekeeper = GatekeeperV082(prob_map=gatekeeper_model.lookup_probs, threshold=0.45)

    joined = old2[["date", "code", "pred_net_return_v2"]].merge(
        old8[["date", "code", "pred_score_v08"]], on=["date", "code"], validate="one_to_one"
    ).merge(
        preds_v083[["date", "code", "pred_score_v083"]], on=["date", "code"], validate="one_to_one"
    )
    joined = compute_adaptive_scores(joined, bm, pm)

    allow_base = old8.date.map(limits).fillna(0).gt(0) & old2.pred_net_return_v2.ge(0.005)
    sig_v08 = old8[allow_base][["date", "code", "pred_score_v08"]].rename(columns={"pred_score_v08": "score"})
    allow_v085 = joined.date.map(limits).fillna(0).gt(0) & joined.pred_net_return_v2.ge(0.005)
    sig_v087 = joined.loc[allow_v085, ["date", "code", "adaptive_score", "pred_net_return_v2"]].rename(
        columns={"adaptive_score": "score"}
    )

    cfg = Config(capital=30000, order_budget=7500, max_positions=5, max_exposure=0.9, hold_days=10)
    results: List[Dict[str, Any]] = []

    print("\n" + "=" * 118)
    print(f"{'Period':<12} {'Case':<26} {'Return':>9} {'MaxDD':>9} {'Sharpe':>8} {'Trades':>8} {'WinRate':>9} {'Audit':>7}")
    print("-" * 118)

    attribution_cases = ["close_only_ref", "sl_touch_only", "tp_touch_only", "touch_full_ref"]
    for period, start, end in PERIODS:
        days = [d for d in calendar if start <= d <= end]
        s_v08 = sig_v08[sig_v08.date.between(start, end)]
        s_v087 = sig_v087[sig_v087.date.between(start, end)]

        for stress in [False, True]:
            slip = 0.0010 if stress else 0.0005
            cfg_run = replace(cfg, slippage=slip)
            prefix = "stress_" if stress else ""

            # Attribution-chain ranker/gatekeeper replicate model_v086_synthesized's
            # v086_no_damping setup exactly (friction 10 bps active, damping block off),
            # so touch_full_ref should reproduce +23.53% (2022-2025) as a parity check.
            attr_ranker = AdaptiveTopRanker(hurdle=0.0120, ranking_friction_penalty=0.0010, conviction_hurdle=0.0)

            case_defs = [
                # (case, signals, ranker, gatekeeper, stopping, enable_intraday_touch, enable_turnover_damping)
                (f"{prefix}v08_baseline", s_v08, None, None, None, False, False),
                (f"{prefix}close_only_ref", s_v087, attr_ranker, gatekeeper, TouchModeStoppingGLM(intraday_mode="all"), False, False),
                (f"{prefix}sl_touch_only", s_v087, attr_ranker, gatekeeper, TouchModeStoppingGLM(intraday_mode="sl"), True, False),
                (f"{prefix}tp_touch_only", s_v087, attr_ranker, gatekeeper, TouchModeStoppingGLM(intraday_mode="tp"), True, False),
                (f"{prefix}touch_full_ref", s_v087, attr_ranker, gatekeeper, TouchModeStoppingGLM(intraday_mode="all"), True, False),
                (f"{prefix}{CANDIDATE_CASE}", s_v087, BaselineRanker(), BreadthGatekeeperGLM(), TouchModeStoppingGLM(intraday_mode="sl"), True, False),
                (
                    f"{prefix}v087_glm_damping",
                    s_v087,
                    AdaptiveTopRanker(hurdle=0.005, ranking_friction_penalty=0.0005, conviction_hurdle=0.0065),
                    BreadthGatekeeperGLM(),
                    TouchModeStoppingGLM(intraday_mode="sl"),
                    True,
                    True,
                ),
            ]

            for case, signals, ranker, gate, stopping, touch, damping in case_defs:
                rep = ModularReplayEngine(
                    frames=frames,
                    calendar=days,
                    signals=signals,
                    dividends=events,
                    config=cfg_run,
                    vol_map=vol,
                    daily_limits=limits,
                    breadth_lookup=bm,
                    yz_vol_map=yz_vol_map,
                    ranker=ranker,
                    gatekeeper=gate,
                    stopping=stopping,
                    enable_cash_overlay=False,
                    enable_intraday_touch=touch,
                    enable_turnover_damping=damping,
                )
                rep.bars = bars
                rep.run()
                summary = rep.summary()
                audit = audit_ledger(rep, bars, events)
                summary.update(period=period, case=case, stress=stress, audit=audit)
                results.append(summary)
                arena.save_case(out_dir / f"{period}_{case}", rep, summary, signals)
                print(
                    f"{period:<12} {case:<26} {summary['total_return']:>8.2%} {summary['max_drawdown']:>8.2%} "
                    f"{summary['sharpe']:>8.2f} {summary['closed_trades']:>8} {summary['win_rate']:>8.2%} {str(audit['passed']):>7}"
                )

    # Multi-seed random controls for the candidate (normal cost)
    random_controls = []
    print("\n" + "=" * 80)
    print("RUNNING V0.8.7-GLM CANDIDATE MULTI-SEED RANDOM CONTROLS (normal cost)")
    print("=" * 80)
    for period, start, end in PERIODS:
        days = [d for d in calendar if start <= d <= end]
        pool = joined[joined.date.between(start, end)]
        s_period = sig_v087[sig_v087.date.between(start, end)]
        for seed in RANDOM_SEEDS:
            s_rnd = arena.matched_candidates(s_period, arena.stable_random(pool, seed))
            rep_rnd = ModularReplayEngine(
                frames=frames,
                calendar=days,
                signals=s_rnd,
                dividends=events,
                config=cfg,
                vol_map=vol,
                daily_limits=limits,
                breadth_lookup=bm,
                yz_vol_map=yz_vol_map,
                ranker=BaselineRanker(),
                gatekeeper=BreadthGatekeeperGLM(),
                stopping=TouchModeStoppingGLM(intraday_mode="sl"),
                enable_cash_overlay=False,
                enable_intraday_touch=True,
                enable_turnover_damping=False,
            )
            rep_rnd.bars = bars
            rep_rnd.run()
            r_sum = rep_rnd.summary()
            r_aud = audit_ledger(rep_rnd, bars, events)
            r_sum.update(period=period, seed=seed, audit=r_aud)
            random_controls.append(r_sum)
            print(
                f"Random Control {period} Seed {seed:<10}: Return={r_sum['total_return']:>7.2%}, "
                f"MaxDD={r_sum['max_drawdown']:>7.2%}, Trades={r_sum['closed_trades']:>3}, Audit={str(r_aud['passed'])}"
            )

    acceptance = compute_acceptance(results)
    damping_verdict = compute_damping_verdict(results)
    attribution = attribution_breakdown(out_dir, attribution_cases)

    payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "display_version": "V0.8.7研究候选（验收结论见 acceptance.overall_pass）",
        "made_by_checkpoint": "GLM",
        "architecture": {
            "gatekeeper": "BreadthGatekeeperGLM (signal-day breadth5 >= 0.58, V0.9 pre-validated fixed rule)",
            "intraday_execution": "stop-loss touch only (+10 bps penalty); take-profit & plan_a floor stay close-based",
            "ranker": "BaselineRanker on adaptive_score signals (same signal set as V0.8.5/V0.8.6 evaluation)",
            "damping": "OFF in candidate; pre-registered recheck variant v087_glm_damping (0.5%/5bps/0.65%)",
        },
        "pre_registered_hypotheses": ["H1 attribution", "H2 candidate hurdles", "H3 damping recheck"],
        "results": results,
        "random_controls": random_controls,
        "acceptance": acceptance,
        "damping_verdict": damping_verdict,
        "attribution": attribution,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out_dir / "acceptance.json").write_text(json.dumps(acceptance, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out_dir / "attribution.json").write_text(json.dumps(attribution, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out_dir / "display-version.json").write_text(
        json.dumps(
            {
                "display_version": payload["display_version"],
                "made_by_checkpoint": "GLM",
                "acceptance_pass": acceptance["overall_pass"],
                "note": "V0.8.7研究候选由 GLM 检查点构建；验收结论以 acceptance.json 真实回放数字为准。",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    comp_df = pd.DataFrame(results)[
        ["period", "case", "stress", "total_return", "max_drawdown", "sharpe", "closed_trades", "win_rate", "cagr", "calmar"]
    ]
    comp_df.to_csv(out_dir / "comparison.csv", index=False)

    print("\n" + "=" * 80)
    print(f"ACCEPTANCE VERDICT (real replay rows, pre-registered hurdles): {'PASS' if acceptance['overall_pass'] else 'FAIL'}")
    for name, c in acceptance["criteria"].items():
        actual = "missing" if c["actual"] is None else f"{c['actual']:.4f}"
        print(f"  {'[PASS]' if c['passed'] else '[FAIL]'} {name:<42} actual={actual} hurdle={c['hurdle']}")
    print(f"DAMPING VERDICT (H3): {damping_verdict.get('verdict')}")
    print(f"\nSaved artifacts to {out_dir}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run V0.8.7-GLM research candidate evaluation")
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()
    run_v087_glm_evaluation(out_dir=Path(args.out_dir))
