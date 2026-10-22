"""Independent verification of V0.9 exported accounts and frozen training artifacts."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import arena
from engine import Config
from v09_rl_2022 import EPOCHS, SEEDS, QPolicy, RLReplay
import run_v09_rl_2022 as runner


ROOT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def objhash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path: Path):
    try:
        return pd.read_csv(path, float_precision="round_trip").to_dict("records")
    except pd.errors.EmptyDataError:
        return []


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def verify_account(record, folder: Path, bars, events):
    cfg = Config(**record["config"])
    cash = cfg.capital
    shares = defaultdict(int)
    waiting = []
    entries = {}
    fills_by_day, actions_by_day = defaultdict(list), defaultdict(list)
    fills = csv_rows(folder / "fills.csv")
    actions = csv_rows(folder / "actions.csv")
    equity = csv_rows(folder / "equity.csv")
    trades = csv_rows(folder / "trades.csv")
    holdings = csv_rows(folder / "holdings.csv")
    decisions = csv_rows(folder / "decisions.csv")
    for fill in fills:
        fills_by_day[str(fill["date"])].append(fill)
    for action in actions:
        actions_by_day[str(action["date"])].append(action)
    peak = cfg.capital
    max_dd = 0.0
    total_cost = 0.0
    for row in equity:
        day = str(row["date"])
        cash += sum(amount for pay, amount in waiting if pay <= day)
        waiting = [(pay, amount) for pay, amount in waiting if pay > day]
        for action in actions_by_day[day]:
            code = str(action["code"])
            bonus = int(action["bonus_shares"])
            shares[code] += bonus
            amount = float(action["net_cash"])
            if amount:
                matched = [e for e in events if e.get("code") == code and e.get("dividOperateDate") == day]
                require(len(matched) == 1, f"{folder.name}: ambiguous dividend action {day} {code}")
                pay = matched[0].get("dividPayDate") or day
                if pay <= day:
                    cash += amount
                else:
                    waiting.append((pay, amount))
        for fill in fills_by_day[day]:
            code = str(fill["code"])
            side = str(fill["side"])
            qty = int(fill["shares"])
            price = float(fill["price"])
            notional = float(fill["notional"])
            require(code in bars and day in bars[code], f"{folder.name}: fill has no bar {day} {code}")
            require(abs(price - float(bars[code][day]["open"])) < 1e-9,
                    f"{folder.name}: fill is not next-open {day} {code}")
            require(abs(notional - qty * price) < 1e-6, f"{folder.name}: fill notional mismatch")
            fees = cfg.fees(notional, sell=side == "sell")
            require(all(abs(float(fill[k]) - v) < 1e-9 for k, v in fees.items()),
                    f"{folder.name}: fee mismatch {day} {code}")
            total_cost += sum(fees.values())
            trade_id = int(fill["trade_id"])
            if side == "buy":
                require(qty % 100 == 0, f"{folder.name}: non-lot buy")
                entries[trade_id] = day
            else:
                require(side == "sell", f"{folder.name}: unknown fill side {side}")
                require(trade_id in entries and entries[trade_id] < day,
                        f"{folder.name}: T+1 violation {day} trade {trade_id}")
            sign = 1 if side == "buy" else -1
            shares[code] += sign * qty
            cash -= sign * notional + sum(fees.values())
            require(shares[code] >= 0 and cash >= -1e-7, f"{folder.name}: negative shares/cash {day}")
        receivable = sum(amount for _, amount in waiting)
        nav = cash + receivable + sum(qty * float(bars[code][day]["close"])
                                      for code, qty in shares.items() if qty)
        require(max(abs(cash - float(row["cash"])), abs(receivable - float(row["receivables"])),
                    abs(nav - float(row["equity"]))) < 1e-6,
                f"{folder.name}: independent ledger/NAV mismatch on {day}")
        peak = max(peak, nav)
        max_dd = min(max_dd, nav / peak - 1)
    require(equity, f"{folder.name}: empty equity export")
    final_nav = float(equity[-1]["equity"])
    expected_return = final_nav / cfg.capital - 1
    require(abs(final_nav - float(record["final_equity"])) < 1e-6, f"{folder.name}: final equity mismatch")
    require(abs(expected_return - float(record["total_return"])) < 1e-10, f"{folder.name}: return mismatch")
    require(abs(max_dd - float(record["max_drawdown"])) < 1e-10, f"{folder.name}: max drawdown mismatch")
    curve = pd.Series([cfg.capital] + [float(row["equity"]) for row in equity], dtype=float)
    returns = curve.pct_change().dropna()
    duration = (pd.Timestamp(str(equity[-1]["date"])) - pd.Timestamp(str(equity[0]["date"]))).days
    cagr = (final_nav / cfg.capital) ** (365.25 / duration) - 1 if duration else None
    sharpe_std = returns.std()
    sharpe = float((242 ** .5) * returns.mean() / sharpe_std) if len(returns) > 1 and sharpe_std > 1e-12 else None
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std()
    sortino = (float((242 ** .5) * returns.mean() / downside_std)
               if len(downside_returns) > 1 and downside_std > 1e-12 else None)
    pnl = [float(t["net_pnl"]) for t in trades]
    positive = sum(x for x in pnl if x > 0)
    negative = -sum(x for x in pnl if x < 0)
    profit_factor = positive / negative if negative else None
    mean_trade = sum(pnl) / len(pnl) if pnl else None
    calmar = cagr / abs(max_dd) if cagr is not None and max_dd else None
    for metric, rebuilt in (("cagr", cagr), ("calmar", calmar), ("sharpe", sharpe),
                            ("sortino", sortino), ("profit_factor", profit_factor)):
        stored = record.get(metric)
        if rebuilt is None:
            require(stored is None, f"{folder.name}: {metric} should be undefined")
        else:
            require(stored is not None and abs(float(rebuilt) - float(stored)) < 1e-9,
                    f"{folder.name}: {metric} mismatch ({rebuilt} != {stored})")
    require(abs(total_cost - float(record["execution_costs"])) < 1e-6, f"{folder.name}: execution costs mismatch")
    require(len(trades) == int(record["closed_trades"]), f"{folder.name}: closed trades mismatch")
    winning = sum(float(t["net_pnl"]) > 0 for t in trades)
    win_rate = winning / len(trades) if trades else None
    if win_rate is None:
        require(record["win_rate"] is None, f"{folder.name}: empty-trade win rate mismatch")
    else:
        require(abs(win_rate - float(record["win_rate"])) < 1e-12,
                f"{folder.name}: win rate mismatch")
    require(len(holdings) == int(record["open_positions"]), f"{folder.name}: open position count mismatch")
    warning_count = len(csv_rows(folder / "warnings.csv")) + len(csv_rows(folder / "review_warnings.csv"))
    require(warning_count == int(record["quality_warning_count"]), f"{folder.name}: warning count mismatch")
    require(abs(sum(amount for _, amount in waiting) - float(record["receivables"])) < 1e-6,
            f"{folder.name}: ending receivable mismatch")
    held_codes = {str(h["code"]): int(h["shares"]) for h in holdings}
    ledger_codes = {code: qty for code, qty in shares.items() if qty}
    require(held_codes == ledger_codes, f"{folder.name}: ending share ledger/holdings mismatch")
    if decisions:
        action_counts = {str(a): sum(int(d["action"]) == a for d in decisions) for a in range(4)}
        require(action_counts == {str(a): int(record.get("action_counts", {}).get(str(a), 0)) for a in range(4)},
                f"{folder.name}: action count mismatch")
        for d in decisions:
            require(str(d["signal_date"]) < str(d["execution_date"]), f"{folder.name}: noncausal decision")
            action = int(d["action"])
            require(0 <= action <= 3, f"{folder.name}: invalid action")
            require(not (action & 1) or int(d["pending_before"]) > 0,
                    f"{folder.name}: skipped absent pending buys")
            require(not (action & 2) or int(d["positions_before"]) > 0,
                    f"{folder.name}: exited absent positions")
    return {"case": record["case"], "period": record["period"], "stress": record["stress"],
            "fills": len(fills), "sessions": len(equity), "closed_trades": len(trades),
            "winning_trades": winning, "win_rate": win_rate, "mean_trade_pnl": mean_trade,
            "profit_factor": profit_factor,
            "cagr": cagr, "max_drawdown": max_dd, "sharpe": sharpe, "sortino": sortino,
            "passed": True}


def reproduce_training(out: Path, days, events):
    material = out / "training-inputs"
    frames = {}
    prices = pd.read_csv(material / "prices-2022.csv", dtype={"date": str, "code": str},
                         float_precision="round_trip")
    for code, group in prices.groupby("code", sort=False):
        frames[str(code)] = group.reset_index(drop=True)
    panel = pd.read_csv(material / "context-vol-2022.csv", dtype={"date": str, "code": str},
                        float_precision="round_trip")
    signals = pd.read_csv(material / "signals-2022.csv", dtype={"date": str, "code": str},
                          float_precision="round_trip")
    bars = arena.prepare_bars(frames)
    # The materialized signal list is already the fully gated V0.8 list. Reconstitute
    # equal-key synthetic prediction columns so runner.prepare rebuilds the context,
    # breadth, volatility maps, limits and signal table using the original code path.
    two = signals[["date", "code"]].copy()
    two["pred_net_return_v2"] = 0.005
    eight = signals[["date", "code"]].copy()
    eight["pred_score_v08"] = signals["score"].to_numpy()
    inputs = runner.prepare(panel, two, eight, days)
    rebuilt = inputs[0].sort_values(["date", "code"]).reset_index(drop=True)
    wanted = signals.sort_values(["date", "code"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(rebuilt, wanted, check_dtype=False, check_exact=False, atol=1e-12, rtol=1e-12)
    rebuilt_models = []
    cfg = Config(capital=30000, order_budget=7500, max_positions=5, max_exposure=.9, hold_days=10)
    for seed in SEEDS:
        policy = QPolicy(seed)
        for epoch in range(EPOCHS):
            epsilon = .30 - .25 * epoch / (EPOCHS - 1)
            rep = runner.replay(frames, bars, days, rebuilt, events, Config(**vars(cfg)), inputs,
                                policy, training=True, epsilon=epsilon)
            require(len(rep.equity) == len(days), f"seed {seed} epoch {epoch}: incomplete replay")
        saved = load_json(out / "models" / f"q-{seed}.json")
        require(objhash(policy.dump()) == objhash(saved), f"seed {seed}: reproduced Q dump differs")
        rebuilt_models.append({"seed": seed, "updates": policy.update_count, "states": len(policy.values),
                               "q_dump_sha256": objhash(policy.dump()), "matches_frozen_q": True})
    return rebuilt_models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    out = args.run.resolve()
    summary = load_json(out / "summary.json")
    started = load_json(out / "started.json")
    scope = load_json(out / "training-scope.json")
    frozen = load_json(out / "training-complete.json")
    require(summary["complete"] and len(summary["results"]) == 60 and not summary["failures"],
            "Run is incomplete or has failed cases")

    source_results = []
    for name, digest in started["source_hashes"].items():
        snapshot = out / "sources" / name
        require(sha(snapshot) == digest, f"source snapshot mismatch: {name}")
        source_results.append(name)
    input_checks = {}
    for relative, digest in started["input_hashes"].items():
        path = ROOT / relative
        require(sha(path) == digest, f"live input hash mismatch: {relative}")
        input_checks[relative] = digest
    for absolute, digest in started["protected_hashes"].items():
        require(sha(Path(absolute)) == digest, f"protected hash mismatch: {absolute}")

    _, frames, calendar, raw = arena.load(arena.DATASET)
    events, event_audit = arena.canonicalize_events(raw)
    require(not event_audit["unresolved_incomplete_groups"], "Unresolved source corporate actions")
    calendar_2022 = [d for d in calendar if "2022-01-01" <= d <= "2022-12-31"]
    material_days = load_json(out / "training-inputs" / "calendar-2022.json")
    require(calendar_2022 == material_days, "2022 calendar differs from source calendar")
    price_dates = [str(frame.date.max()) for frame in frames.values() if len(frame)]
    source_inventory = load_json(arena.DATASET / "manifest.json")["inventory"]
    require(len(source_inventory) == len(frames), "Manifest inventory/frame count mismatch")
    for item in source_inventory:
        code = item["code"]
        require(sha(arena.DATASET / f"{code}.csv") == item["sha256"], f"price source hash mismatch: {code}")
        require(sha(arena.DATASET / f"{code}.dividends.csv") == item["dividends_sha256"],
                f"dividend source hash mismatch: {code}")

    material = out / "training-inputs"
    expected_material = scope["materialized_input_hashes"]
    material_hashes = {p.name: sha(p) for p in material.iterdir() if p.is_file()}
    require(material_hashes == expected_material, "Materialized training input hash mismatch")
    train_prices = pd.read_csv(material / "prices-2022.csv", dtype={"date": str, "code": str})
    train_context = pd.read_csv(material / "context-vol-2022.csv", dtype={"date": str, "code": str})
    train_signals = pd.read_csv(material / "signals-2022.csv", dtype={"date": str, "code": str})
    require(train_prices.date.max() <= "2022-12-31", "Training price input exceeds 2022")
    require(train_context.date.max() <= "2022-12-31", "Training context input exceeds 2022")
    require(train_signals.date.max() <= "2022-12-31", "Training signal input exceeds 2022")
    require(train_prices.date.min() == scope["first"] and train_prices.date.max() == scope["last"],
            "Training price date scope mismatch")
    require(train_context.date.max() == scope["max_context_date"] and
            train_signals.date.max() == scope["max_signal_date"], "Training scope dates mismatch")
    require(len(material_days) == scope["sessions"] and material_days[0] == scope["first"] and
            material_days[-1] == scope["last"], "Training calendar scope mismatch")
    require(scope["max_price_date"] <= "2022-12-31", "Recorded scope price date exceeds cutoff")

    for name, digest in frozen["model_hashes"].items():
        require(sha(out / "models" / name) == digest, f"Frozen Q hash mismatch: {name}")
    summary_time = datetime.fromisoformat(summary["completed_utc"])
    freeze_time = datetime.fromisoformat(frozen["utc"])
    require(freeze_time < summary_time, "Q freeze timestamp is not before completed summary")
    require(frozen["training_cutoff"] == "2022-12-31" and frozen["epochs_per_seed"] == EPOCHS and
            tuple(frozen["seeds"]) == tuple(SEEDS), "Frozen training protocol differs")

    bars = arena.prepare_bars(frames)
    audits = [verify_account(record, out / f"{record['period']}_{record['case']}", bars, events)
              for record in summary["results"]]
    rows = {(r["period"], r["case"]): r for r in summary["results"]}
    independent_acceptance = []
    for period in ("2023-2025", "2026"):
        baseline, vote = rows[(period, "baseline")], rows[(period, "vote")]
        stress_base, stress_vote = rows[(period, "stress_baseline")], rows[(period, "stress_vote")]
        checks = {
            "higher_net_return": float(vote["total_return"]) > float(baseline["total_return"]) + 1e-12,
            "drawdown_not_worse": float(vote["max_drawdown"]) >= float(baseline["max_drawdown"]) - 1e-12,
            "higher_stressed_return": float(stress_vote["total_return"]) > float(stress_base["total_return"]) + 1e-12,
        }
        checks["passed"] = all(checks.values())
        independent_acceptance.append({"period": period, **checks})
    require(independent_acceptance == summary["acceptance"]["checks"] and
            summary["acceptance"]["passed"] == all(x["passed"] for x in independent_acceptance),
            "Acceptance decision does not match independently rebuilt account metrics")
    q_reproduction = reproduce_training(out, material_days,
                                        load_json(material / "events-2022.json"))

    prior = ROOT / "results" / "v09-rl-2022-prototype-20261021-v1"
    prior_comparison = {"available": False}
    if prior.exists():
        prior_summary = load_json(prior / "summary.json")
        prior_model_hashes = {name: sha(prior / "models" / name) for name in frozen["model_hashes"]}
        current_model_hashes = {name: sha(out / "models" / name) for name in frozen["model_hashes"]}
        require(prior_model_hashes == current_model_hashes, "V1 and V2 trained Q model hashes differ")
        previous_rows = {(r["period"], r["case"], bool(r["stress"])): r for r in prior_summary["results"]}
        current_rows = {(r["period"], r["case"], bool(r["stress"])): r for r in summary["results"]}
        require(previous_rows.keys() == current_rows.keys(), "V1/V2 evaluation case sets differ")
        compared = ["total_return", "max_drawdown", "final_equity", "closed_trades", "execution_costs"]
        for key, current in current_rows.items():
            previous = previous_rows[key]
            for metric in compared:
                require(abs(float(previous[metric]) - float(current[metric])) < 1e-12,
                        f"V1/V2 result changed: {key} {metric}")
        prior_comparison = {"available": True, "q_model_hashes_equal": True,
                            "results_equal_for": compared, "cases_compared": len(current_rows)}

    result = {
        "audit_version": 1,
        "created_utc": datetime.now().astimezone().isoformat(),
        "all_60_exported_accounts_rebuilt": True,
        "accounts_checked": len(audits),
        "total_fills_checked": sum(item["fills"] for item in audits),
        "checks": {
            "independent_cash_share_receivable_fee_next_open_t1_nav": True,
            "summary_return_drawdown_costs_trades_holdings": True,
            "trades_profit_count_and_win_rate": True,
            "source_snapshots_match_started_hashes": len(source_results),
            "input_hashes_match": input_checks,
            "protected_hashes_match": len(started["protected_hashes"]),
            "all_price_dividend_hashes_match_manifest": len(source_inventory),
            "source_calendar_2022_matches_materialized_calendar": len(material_days),
            "materialized_training_hashes_match_scope": material_hashes,
            "training_price_context_signal_cutoff": "2022-12-31",
            "q_hashes_match_training_complete": frozen["model_hashes"],
            "q_freeze_precedes_completed_summary": True,
            "reproduced_seed_epoch_q_dumps": q_reproduction,
            "independently_recomputed_acceptance": independent_acceptance,
            "v1_vs_v2_same_training_and_evaluation": prior_comparison,
            "feature_causality": "Existing 411295x37 raw-feature recomputation evidence was not rerun in this export audit.",
        },
        "event_source_audit": event_audit,
    }
    audit_path = out / "independent-export-audit.json"
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"PASSED: {len(audits)} exports, {result['total_fills_checked']} fills, "
          f"{len(q_reproduction)} reproduced Q tables; wrote {audit_path}")


if __name__ == "__main__":
    main()
