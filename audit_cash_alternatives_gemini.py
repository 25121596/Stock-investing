"""audit_cash_alternatives_gemini.py - Comprehensive Parallel Benchmarking of Cash Management Alternatives.

Benchmarked Alternatives:
  1. pure_zero_yield: 0.00% annual yield (Pure stock alpha baseline, no overlay).
  2. current_deposit: 0.25% annual yield (Standard bank/brokerage demand deposit rate).
  3. lingqiantong_yu_ebao: 1.60% annual yield (Retail public money market funds).
  4. broker_cash_sweep: 1.85% annual yield (Automated night cash sweep: 天天发/现金管家/天天利).
  5. etf_cash_sweep: 1.95% annual yield (On-exchange MMF ETF: 511880/511990 T+0).
  6. treasury_repo_gc001: 2.20% annual yield (Treasury reverse repo GC001 benchmark - V0.8.9 original).
  7. stock_util_slots5_zero: 0.00% annual yield (Internal stock capacity expansion: 5 slots, 0% overlay).
  8. stock_util_slots5_sweep: 1.85% annual yield (Combined: 5 slots + broker automated sweep).

Outputs:
  - Full tabular comparison across 2022-2025 and 2026.
  - Yearly returns breakdown (2022, 2023, 2024, 2025, 2026).
  - Pure stock trading profit vs. Cash overlay yield breakdown.
  - Double slippage (10 bps) stress resilience.
  - Ledger balance sheet conservation audit for every single run.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import arena
from engine import Config
from engine_core import BaselineRanker, audit_ledger
from model_v083_ranker import compute_adaptive_scores
from model_v089_gemini import (
    BreadthGatekeeperGemini,
    ReplayEngineV089,
    TouchModeStoppingGemini,
    build_dynamic_daily_limits,
)
from train_model_v08 import extract_microstructure_features
from train_model_v2 import enrich_features_v2
from train_model_v3 import extract_ashare_features

BUNDLE = ROOT / "results/v08-codex-rebuild-20261020-v1"
OUT_DIR = ROOT / "results/v089-cash-alternatives-gemini"


CASH_SCHEMES = [
    {
        "id": "pure_zero_yield",
        "name": "零收益纯股票基线 (Pure Stock Alpha)",
        "rate": 0.00,
        "enable_overlay": False,
        "slots": 4,
        "description": "闲置资金不理财，收益率0.00%。绝对纯净的股票交易收益基线。",
        "practical_ops": "完全无需任何操作，钱躺在账户。",
        "convenience_score": 10.0,
    },
    {
        "id": "current_deposit",
        "name": "券商/银行活期结息 (Current Deposit)",
        "rate": 0.0025,
        "enable_overlay": True,
        "slots": 4,
        "description": "法定活期存款年化0.25%。证券账户默认享有，按季结息。",
        "practical_ops": "完全无需任何操作，自动到账。",
        "convenience_score": 10.0,
    },
    {
        "id": "lingqiantong_yu_ebao",
        "name": "微信零钱通 / 支付宝余额宝 (Retail MMF)",
        "rate": 0.0160,
        "enable_overlay": True,
        "slots": 4,
        "description": "公募货币基金年化1.60%。散户最普遍的零钱理财方案。",
        "practical_ops": "需将未占用资金划转至零钱通/余额宝，买入股票时需T+0转回。",
        "convenience_score": 7.5,
    },
    {
        "id": "broker_cash_sweep",
        "name": "券商保证金夜间自动理财 (Auto Cash Sweep)",
        "rate": 0.0185,
        "enable_overlay": True,
        "slots": 4,
        "description": "券商天天发/天天利/现金管家，年化1.85%。15:00自动归集，次日9:15自动解冻。",
        "practical_ops": "一次性签约开通，全自动免维护，完全不影响盘中任何挂单和交易。",
        "convenience_score": 9.8,
    },
    {
        "id": "etf_cash_sweep",
        "name": "场内货币ETF (511880/511990 On-Exchange MMF)",
        "rate": 0.0195,
        "enable_overlay": True,
        "slots": 4,
        "description": "场内货基ETF年化1.95%(扣费)。支持T+0回转，场内免印花税。",
        "practical_ops": "白天未买股票时买入货基ETF，需要买股票时T+0实时卖出变现。",
        "convenience_score": 8.0,
    },
    {
        "id": "treasury_repo_gc001",
        "name": "国债逆回购对照组 (GC001 Benchmark / 原V0.8.9)",
        "rate": 0.0220,
        "enable_overlay": True,
        "slots": 4,
        "description": "上交所1天期逆回购年化2.20%。原V0.8.9对照基准。",
        "practical_ops": "每日15:00-15:30需手动或预约挂单卖出GC001/R-001。",
        "convenience_score": 6.0,
    },
    {
        "id": "stock_util_slots5_zero",
        "name": "纯内生股票扩容 (5 Slots Reinvestment, 0% Overlay)",
        "rate": 0.00,
        "enable_overlay": False,
        "slots": 5,
        "description": "不靠外部利息(0%)，将槽位从4扩到5，靠股票策略自身吞吐闲置资金。",
        "practical_ops": "完全靠量化交易算法调度，无需任何理财操作。",
        "convenience_score": 9.5,
    },
    {
        "id": "stock_util_slots5_sweep",
        "name": "扩容组合拳 (5 Slots + 券商夜间自动理财 1.85%)",
        "rate": 0.0185,
        "enable_overlay": True,
        "slots": 5,
        "description": "5槽位股票端积极吞吐 + 盘后券商自动理财(1.85%)双轮驱动。",
        "practical_ops": "一次性签约券商自动理财，策略全自动运行。",
        "convenience_score": 9.5,
    },
]


def load_inputs():
    print("Loading datasets and panels...")
    _, frames, calendar, raw = arena.load(arena.DATASET)
    events, _ = arena.canonicalize_events(raw)
    bars = arena.prepare_bars(frames)

    panel = pd.read_csv(ROOT / "results/midterm-access-v3/panel-h10.csv", dtype={"date": str, "code": str, "label_end": str})
    sixty = pd.read_csv(ROOT / "results/exploration-v1/new-features.csv", dtype={"date": str, "code": str})
    panel = enrich_features_v2(panel.merge(sixty[["date", "code", "adj_ret60", "ret_vol60"]], on=["date", "code"], how="left", validate="one_to_one"))
    panel = panel.merge(extract_ashare_features(frames), on=["date", "code"], how="left", validate="one_to_one").merge(
        extract_microstructure_features(frames), on=["date", "code"], how="left", validate="one_to_one"
    )

    b = arena.breadth_days(panel)
    bm = b.set_index("date").breadth5.to_dict()
    pm = b.set_index("date").pool_size.to_dict()
    vol = panel.set_index(["date", "code"]).ret_vol20.to_dict()
    yz_vol_map = panel.set_index(["date", "code"]).yz_vol20.to_dict()

    old2 = pd.read_csv(BUNDLE / "v2/predictions-h10-v2.csv", dtype={"date": str, "code": str})
    old8 = pd.read_csv(BUNDLE / "v08/predictions-h10-v08.csv", dtype={"date": str, "code": str})
    preds_v083 = pd.read_csv(ROOT / "results/v083-adaptive-ranker/predictions-v083.csv", dtype={"date": str, "code": str})

    joined = old2[["date", "code", "pred_net_return_v2"]].merge(
        old8[["date", "code", "pred_score_v08"]], on=["date", "code"], validate="one_to_one"
    ).merge(
        preds_v083[["date", "code", "pred_score_v083"]], on=["date", "code"], validate="one_to_one"
    )
    joined = compute_adaptive_scores(joined, bm, pm)

    limits_boost = build_dynamic_daily_limits(bm, pm, explosive_boost=True)
    allow = joined.date.map(limits_boost).fillna(0).gt(0) & joined.pred_net_return_v2.ge(0.0030)
    signals = joined.loc[allow, ["date", "code", "adaptive_score", "pred_net_return_v2"]].rename(
        columns={"adaptive_score": "score"}
    )
    return frames, calendar, events, bars, bm, vol, yz_vol_map, limits_boost, signals


def run_single_simulation(
    scheme: Dict[str, Any],
    period: str,
    stress: bool,
    frames,
    calendar,
    events,
    bars,
    bm,
    vol,
    yz_vol_map,
    limits_boost,
    signals,
    cfg_base,
):
    start = "2022-01-01" if period == "2022-2025" else "2026-01-01"
    end = "2025-12-31" if period == "2022-2025" else "2026-10-13"
    days = [d for d in calendar if start <= d <= end]
    s_period = signals[signals.date.between(start, end)]

    slip = 0.0010 if stress else 0.0005
    cfg = replace(cfg_base, slippage=slip)

    rep = ReplayEngineV089(
        frames=frames,
        calendar=days,
        signals=s_period,
        dividends=events,
        config=cfg,
        vol_map=vol,
        daily_limits=limits_boost,
        breadth_lookup=bm,
        yz_vol_map=yz_vol_map,
        ranker=BaselineRanker(),
        gatekeeper=BreadthGatekeeperGemini(0.58),
        stopping=TouchModeStoppingGemini(plan_mode="plan_a", intraday_mode="sl"),
        enable_cash_overlay=scheme["enable_overlay"],
        cash_annual_rate=scheme["rate"],
        enable_intraday_touch=True,
        enable_turnover_damping=False,
        enable_nav_compounding=True,
        compounding_mode="floor_capped",
        target_slots=scheme["slots"],
        hold_strong=11,
        hold_normal=7,
        recovery_max_slots=0,
    )
    rep.bars = bars
    rep.run()
    summary = rep.summary()
    audit = audit_ledger(rep, bars, events)

    # Compute detailed yearly breakdowns
    eq_df = pd.DataFrame(rep.equity)
    eq_df["year"] = eq_df["date"].str[:4]
    yearly_stats = {}
    for yr, y_df in eq_df.groupby("year"):
        y_start_eq = y_df["equity"].iloc[0]
        y_end_eq = y_df["equity"].iloc[-1]
        y_ret = (y_end_eq / y_start_eq) - 1.0
        # Peak drawdown within year
        roll_peak = y_df["equity"].cummax()
        dd = (y_df["equity"] - roll_peak) / roll_peak
        y_maxdd = float(dd.min())
        yearly_stats[yr] = {"return": y_ret, "maxdd": y_maxdd}

    total_net_profit = summary["final_equity"] - summary["initial_capital"]
    cash_yield_rmb = rep.accumulated_cash_yield
    stock_net_profit_rmb = total_net_profit - cash_yield_rmb
    stock_return_pct = stock_net_profit_rmb / cfg.capital
    cash_yield_pct = cash_yield_rmb / cfg.capital

    calmar = summary["total_return"] / abs(summary["max_drawdown"]) if summary["max_drawdown"] != 0 else 0.0

    return {
        "scheme_id": scheme["id"],
        "scheme_name": scheme["name"],
        "rate": scheme["rate"],
        "period": period,
        "stress": stress,
        "total_return": summary["total_return"],
        "net_profit": total_net_profit,
        "cash_yield_rmb": cash_yield_rmb,
        "cash_yield_pct": cash_yield_pct,
        "stock_net_profit_rmb": stock_net_profit_rmb,
        "stock_return_pct": stock_return_pct,
        "max_drawdown": summary["max_drawdown"],
        "sharpe": summary["sharpe"],
        "calmar": calmar,
        "trades": summary["closed_trades"],
        "win_rate": summary["win_rate"],
        "profit_factor": summary["profit_factor"],
        "final_nav": summary["final_equity"],
        "audit_pass": audit["passed"],
        "yearly_stats": yearly_stats,
    }


def main():
    start_time = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames, calendar, events, bars, bm, vol, yz_vol_map, limits_boost, signals = load_inputs()
    cfg_base = Config(capital=30000, order_budget=7500, max_positions=5, max_exposure=0.9, hold_days=10)

    print("\n" + "=" * 135)
    print("RUNNING PARALLEL CASH ALTERNATIVES MATRIX (8 SCHEMES x 2 PERIODS x 2 STRESS MODES)")
    print("=" * 135)

    all_results = []
    for scheme in CASH_SCHEMES:
        print(f"\nEvaluating Scheme: {scheme['name']} (Rate: {scheme['rate']:.2%}, Slots: {scheme['slots']})")
        # 1. Period 1: 2022-2025 Normal
        r_p1 = run_single_simulation(scheme, "2022-2025", False, frames, calendar, events, bars, bm, vol, yz_vol_map, limits_boost, signals, cfg_base)
        # 2. Period 1: 2022-2025 Stressed
        r_p1_stress = run_single_simulation(scheme, "2022-2025", True, frames, calendar, events, bars, bm, vol, yz_vol_map, limits_boost, signals, cfg_base)
        # 3. Period 2: 2026 Normal
        r_p2 = run_single_simulation(scheme, "2026", False, frames, calendar, events, bars, bm, vol, yz_vol_map, limits_boost, signals, cfg_base)
        # 4. Period 2: 2026 Stressed
        r_p2_stress = run_single_simulation(scheme, "2026", True, frames, calendar, events, bars, bm, vol, yz_vol_map, limits_boost, signals, cfg_base)

        combined_record = {
            "scheme": scheme,
            "p1": r_p1,
            "p1_stress": r_p1_stress,
            "p2": r_p2,
            "p2_stress": r_p2_stress,
        }
        all_results.append(combined_record)

        print(
            f"  -> 2022-2025: Return {r_p1['total_return']:>7.2%} (Stock {r_p1['stock_return_pct']:>6.2%} + Cash {r_p1['cash_yield_pct']:>5.2%}) | "
            f"MaxDD {r_p1['max_drawdown']:>6.2%} | Sharpe {r_p1['sharpe']:>4.2f} | Trades {r_p1['trades']} | Audit: {r_p1['audit_pass']}"
        )
        print(
            f"  -> 2026 OOS : Return {r_p2['total_return']:>7.2%} (Stock {r_p2['stock_return_pct']:>6.2%} + Cash {r_p2['cash_yield_pct']:>5.2%}) | "
            f"MaxDD {r_p2['max_drawdown']:>6.2%} | Sharpe {r_p2['sharpe']:>4.2f} | Trades {r_p2['trades']} | Audit: {r_p2['audit_pass']}"
        )
        print(
            f"  -> Stress 10bps: P1 {r_p1_stress['total_return']:>7.2%} (MaxDD {r_p1_stress['max_drawdown']:>6.2%}) | "
            f"P2 {r_p2_stress['total_return']:>6.2%} (MaxDD {r_p2_stress['max_drawdown']:>6.2%})"
        )

    # Save complete JSON
    out_json = OUT_DIR / "cash_alternatives_benchmark.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Results successfully saved to {out_json}")
    print(f"Elapsed: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
