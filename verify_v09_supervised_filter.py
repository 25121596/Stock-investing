"""Independent audit for the frozen supervised-filter comparison export."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import arena
import run_v09_supervised_filter as supervised_runner
from v09_rl_2022 import QPolicy, SEEDS, VotePolicy
import run_v09_rl_2022 as rl_runner
import verify_v09_rl_2022 as rl_verifier


ROOT = Path(__file__).resolve().parent
SOURCE_RUN = ROOT / "results" / "v09-rl-ablation-20261021-v2"
Q_SOURCE = ROOT / "results" / "v09-rl-2022-prototype-20261021-v2"
BOUNDARY = "2023-01-01"
PRIMARY_PERIODS = ("2023-2025", "2026")


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def close(a, b, tol=1e-10):
    return abs(float(a) - float(b)) <= tol


def cell(value, default=None):
    """Decode pandas CSV cells containing lists/dicts while tolerating empty cells."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s:
        return default
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return value


def csv_rows(path: Path):
    try:
        return pd.read_csv(path, dtype={"date": str, "signal_date": str, "execution_date": str,
                                        "entry_date": str, "exit_date": str},
                           float_precision="round_trip").to_dict("records")
    except pd.errors.EmptyDataError:
        return []


def read_gate_rows(path: Path):
    rows = csv_rows(path)
    for row in rows:
        for key in ("state", "allowed", "orders"):
            row[key] = cell(row.get(key), [] if key in ("state", "allowed", "orders") else None)
        row["state"] = [int(x) for x in row["state"]]
        row["allowed"] = [int(x) for x in row["allowed"]]
        row["orders"] = row["orders"] or []
        row["action"] = int(row["action"])
        row["pending_before"] = int(row["pending_before"])
        row["positions_before"] = int(row["positions_before"])
        if row.get("score") is not None and not (isinstance(row["score"], float) and math.isnan(row["score"])):
            row["score"] = float(row["score"])
        else:
            row["score"] = None
    return rows


def independent_labels(folder: Path, boundary=BOUNDARY):
    """Rebuild outcome labels solely from exported baseline observations and ledger."""
    observations = read_gate_rows(folder / "gate-decisions.csv")
    fills = csv_rows(folder / "fills.csv")
    trades = csv_rows(folder / "trades.csv")
    fills_by_day = {}
    for fill in fills:
        fills_by_day.setdefault(str(fill["date"]), []).append(fill)
    closed = {}
    for trade in trades:
        tid = int(trade["trade_id"])
        require(tid not in closed, f"Duplicate trade ID in training ledger: {tid}")
        closed[tid] = trade

    rebuilt = []
    seen_trade_ids = set()
    for obs in observations:
        if not obs["orders"]:
            continue
        signal_day = str(obs["signal_date"])
        execution_day = str(obs["execution_date"])
        orders = obs["orders"]
        order_codes = [str(o["code"]) for o in orders]
        require(len(order_codes) == len(set(order_codes)), f"Duplicate code in captured batch {signal_day}")
        require(obs["pending_before"] == len(orders), f"Pending count differs from captured orders {signal_day}")
        same_day_buys = [f for f in fills_by_day.get(execution_day, []) if str(f["side"]) == "buy"]
        batch_buys = [f for f in same_day_buys if str(f["code"]) in set(order_codes)]
        require(len(batch_buys) == len(same_day_buys), f"Unmatched same-day buy fill in training batch {signal_day}")
        for fill in batch_buys:
            tid = int(fill["trade_id"])
            require(tid not in seen_trade_ids, f"Trade assigned to multiple training batches: {tid}")
            seen_trade_ids.add(tid)
            require(str(fill["code"]) in order_codes, f"Fill code absent from batch {signal_day}")

        ids = [int(f["trade_id"]) for f in batch_buys]
        row = {**obs, "filled_trade_ids": ids, "filled_count": len(ids), "status": "eligible",
               "label_end": None, "net_pnl": None, "net_return": None, "y": None}
        if not ids:
            row["status"] = "no_execution"
        elif any(tid not in closed for tid in ids):
            row["status"] = "right_censored"
        else:
            batch_trades = [closed[tid] for tid in ids]
            for trade in batch_trades:
                require(str(trade["signal_date"]) == signal_day and
                        str(trade["entry_date"]) == execution_day,
                        f"Trade does not map to captured batch {signal_day}")
                require(str(trade["code"]) in order_codes, f"Trade code absent from captured batch {signal_day}")
            end = max(str(t["exit_date"]) for t in batch_trades)
            row["label_end"] = end
            if end >= boundary:
                row["status"] = "label_not_mature_before_fit"
            else:
                pnl = sum(float(t["net_pnl"]) for t in batch_trades)
                cost = sum(float(t["entry_notional"]) + float(t["entry_fees"]) for t in batch_trades)
                require(cost > 0, f"Nonpositive training batch cost: {signal_day}")
                row.update(net_pnl=pnl, net_return=pnl / cost, y=int(pnl > 0))
        rebuilt.append(row)
    return rebuilt


def compare_label_exports(rebuilt, saved):
    require(len(rebuilt) == len(saved), "Training label row count differs from independent reconstruction")
    for i, (a, b) in enumerate(zip(rebuilt, saved)):
        for key in ("signal_date", "execution_date", "state", "allowed", "action", "pending_before",
                    "positions_before", "filled_trade_ids", "filled_count", "status", "label_end", "y"):
            require(a.get(key) == b.get(key), f"Training label row {i} field {key} mismatch")
        for key in ("net_pnl", "net_return"):
            if a.get(key) is None:
                require(b.get(key) is None, f"Training label row {i} field {key} should be empty")
            else:
                require(b.get(key) is not None and close(a[key], b[key], 1e-10),
                        f"Training label row {i} field {key} mismatch")


def refit_logit(eligible):
    require(len(eligible) >= 10, "Frozen protocol minimum of 10 mature batches not met")
    counts = Counter(int(r["y"]) for r in eligible)
    require(counts[0] >= 2 and counts[1] >= 2, "Frozen protocol minimum class counts not met")
    x = np.asarray([r["state"] for r in eligible], dtype=float)
    y = np.asarray([int(r["y"]) for r in eligible], dtype=int)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=1.0, class_weight=None, solver="lbfgs", max_iter=2000)
    model.fit(scaler.transform(x), y)
    record = {"mean": scaler.mean_, "scale": scaler.scale_, "coef": model.coef_[0],
              "intercept": float(model.intercept_[0]), "class_counts": counts}
    return record, model, scaler


def saved_score(model_record, state):
    x = (np.asarray(state, dtype=float) - np.asarray(model_record["mean"])) / np.asarray(model_record["scale"])
    z = float(x @ np.asarray(model_record["coef"]) + float(model_record["intercept"]))
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


def q_choose(policy: QPolicy, state, allowed):
    values = policy.values.get(tuple(state), np.zeros(4))
    visits = policy.visits.get(tuple(state), np.zeros(4, dtype=int))
    eligible = [a for a in allowed if visits[a] >= 10]
    if not eligible:
        return 0
    return max(eligible, key=lambda a: (float(values[a]), -a))


def expected_gate(policy_name, state, allowed, model_record, qpolicies):
    if policy_name == "baseline":
        return 0, None
    if policy_name == "fixed_breadth":
        return (int(state[0] < 2) if 1 in allowed else 0), None
    if policy_name == "supervised":
        score = saved_score(model_record, state)
        return (int(score < 0.5) if 1 in allowed else 0), score
    if policy_name == "rl_vote":
        votes = Counter(q_choose(q, state, allowed) for q in qpolicies)
        action = max(sorted(votes), key=lambda a: (votes[a], -a))
        return int(action) & 1, None
    if policy_name.startswith("seed_"):
        seed = int(policy_name.removeprefix("seed_"))
        policy = next(q for q in qpolicies if q.seed == seed)
        return q_choose(policy, state, allowed) & 1, None
    raise AssertionError(f"Unknown policy in exported account: {policy_name}")


def verify_gate_account(record, folder: Path, model_record, qpolicies, sl_support, vote_support,
                        context, bars):
    gate_rows = read_gate_rows(folder / "gate-decisions.csv")
    raw_decisions = csv_rows(folder / "decisions.csv")
    equity_rows = csv_rows(folder / "equity.csv")
    account_calendar = [str(r["date"]) for r in equity_rows]
    calendar_index = {day: i for i, day in enumerate(account_calendar)}
    record_by_day = {str(r["date"]): r for r in equity_rows}
    fills_by_day, actions_by_day = {}, {}
    for fill in csv_rows(folder / "fills.csv"):
        fills_by_day.setdefault(str(fill["date"]), []).append(fill)
    for action in csv_rows(folder / "actions.csv"):
        actions_by_day.setdefault(str(action["date"]), []).append(action)
    # Rebuild holdings and active-position entry bases in the same event order as the ledger verifier.
    active = {}
    shares_by_day = {}
    entry_basis_by_day = {}
    active_by_day = {}
    for day in account_calendar:
        for action in actions_by_day.get(day, []):
            tid = int(action["trade_id"])
            if tid in active:
                active[tid]["shares"] += int(action["bonus_shares"])
        for fill in fills_by_day.get(day, []):
            tid = int(fill["trade_id"])
            qty = int(fill["shares"])
            if str(fill["side"]) == "buy":
                require(tid not in active, f"{folder.name}: duplicate active buy trade {tid}")
                active[tid] = {"code": str(fill["code"]), "shares": qty,
                               "entry_notional": float(fill["notional"]),
                               "entry_index": calendar_index[day]}
            else:
                require(tid in active and active[tid]["shares"] >= qty,
                        f"{folder.name}: sell without active shares for trade {tid}")
                active[tid]["shares"] -= qty
                if active[tid]["shares"] == 0:
                    del active[tid]
        shares_by_day[day] = {str(code): sum(p["shares"] for p in active.values() if p["code"] == code)
                              for code in {p["code"] for p in active.values()}}
        entry_basis_by_day[day] = sum(p["entry_notional"] for p in active.values())
        active_by_day[day] = {tid: dict(position) for tid, position in active.items()}

    peak = float(record["config"]["capital"])
    rebuilt_states = 0
    require(len(gate_rows) == len(raw_decisions), f"{folder.name}: gate/raw decision count mismatch")
    for i, (gate, decision) in enumerate(zip(gate_rows, raw_decisions)):
        require(str(gate["signal_date"]) == str(decision["signal_date"]) and
                str(gate["execution_date"]) == str(decision["execution_date"]),
                f"{folder.name}: decision date mismatch at row {i}")
        signal_day, execution_day = str(gate["signal_date"]), str(gate["execution_date"])
        require(signal_day in calendar_index and calendar_index[signal_day] + 1 < len(account_calendar),
                f"{folder.name}: signal date is not followed by an execution session at row {i}")
        require(account_calendar[calendar_index[signal_day] + 1] == execution_day,
                f"{folder.name}: execution date is not the next account session at row {i}")
        require(gate["state"] == [int(x) for x in cell(decision["state"], [])],
                f"{folder.name}: state mismatch at row {i}")
        require(gate["action"] == int(decision["action"]), f"{folder.name}: action trace mismatch at row {i}")
        pending, positions = gate["pending_before"], gate["positions_before"]
        expected_allowed = [a for a in range(4)
                            if (not (a & 1) or pending > 0) and (not (a & 2) or positions > 0)]
        require(gate["allowed"] == expected_allowed,
                f"{folder.name}: allowed actions mismatch at row {i}")
        require(pending == len(gate["orders"]), f"{folder.name}: pending/order count mismatch at row {i}")
        require(gate["state"][6] == min(2, pending), f"{folder.name}: pending state bin mismatch at row {i}")
        require(int(decision["pending_before"]) == pending and int(decision["positions_before"]) == positions,
                f"{folder.name}: raw decision counts mismatch at row {i}")

        # Recompute every state field from causal context and the preceding close ledger.
        state = gate["state"]
        require(signal_day in context, f"{folder.name}: missing independent context for {signal_day}")
        c = context[signal_day]
        nav = float(record_by_day[signal_day]["equity"])
        peak = max(peak, nav)
        dd = 1.0 - nav / peak
        invested = 0.0
        max_age = 0
        entry_basis = entry_basis_by_day[signal_day]
        require(len(active_by_day[signal_day]) == positions,
                f"{folder.name}: positions_before differs from reconstructed holdings at {signal_day}")
        for code, qty in shares_by_day[signal_day].items():
            require(code in bars and signal_day in bars[code],
                    f"{folder.name}: state holdings lack close quote {signal_day} {code}")
            invested += qty * float(bars[code][signal_day]["close"])
        float_gain = invested / entry_basis - 1.0 if entry_basis else 0.0
        # The RL state uses session ordinals within this independent account period.
        for tid, position in active_by_day[signal_day].items():
            max_age = max(max_age, calendar_index[signal_day] - position["entry_index"])
        expected_state = [
            int(np.searchsorted([.45, .58], float(c["breadth"]), side="right")),
            int(np.searchsorted([-.02, .02], float(c["market_ret5"]), side="right")),
            int(np.searchsorted([.05, .35, .65], invested / nav, side="right")),
            int(np.searchsorted([.02, .05], dd, side="right")),
            int(np.searchsorted([-.03, 0., .03], float_gain, side="right")),
            int(np.searchsorted([3, 8], max_age, side="right")),
            min(2, pending),
        ]
        require(state == expected_state,
                f"{folder.name}: independently rebuilt state mismatch at {signal_day}: {state} != {expected_state}")
        rebuilt_states += 1

    policy = record["policy"]
    scores_checked = 0
    rebuilt_scores = []
    for i, gate in enumerate(gate_rows):
        expected, score = expected_gate(policy, gate["state"], gate["allowed"], model_record, qpolicies)
        require(expected == gate["action"], f"{folder.name}: gate decision mismatch at row {i}")
        if policy == "supervised":
            require(gate["score"] is not None and close(score, gate["score"], 1e-11),
                    f"{folder.name}: saved supervised score mismatch at row {i}")
            scores_checked += 1
            if gate["pending_before"] > 0:
                rebuilt_scores.append(float(score))
        else:
            require(gate["score"] is None, f"{folder.name}: unexpected score for {policy} at row {i}")

    # Rebuild coverage using only actionable pending-buy batches and the frozen training support.
    opportunities = [r for r in gate_rows if r["pending_before"] > 0]
    accepted = [r for r in opportunities if r["action"] == 0]
    novel_support = (sl_support if policy == "supervised" else
                     vote_support if policy == "rl_vote" else
                     set(next(q for q in qpolicies if q.seed == int(policy.removeprefix("seed_"))).values)
                     if policy.startswith("seed_") else None)
    novel = sum(tuple(r["state"]) not in novel_support for r in opportunities) if novel_support is not None else None
    fills = csv_rows(folder / "fills.csv")
    rebuilt_coverage = {
        "opportunity_batches": len(opportunities),
        "accepted_batches": len(accepted),
        "skipped_batches": len(opportunities) - len(accepted),
        "acceptance_rate": len(accepted) / len(opportunities) if opportunities else None,
        "proposed_orders": sum(r["pending_before"] for r in opportunities),
        "accepted_orders": sum(r["pending_before"] for r in accepted),
        "filled_buy_orders": sum(str(f["side"]) == "buy" for f in fills),
        "novel_state_batches": novel,
        "novel_state_rate": novel / len(opportunities) if novel is not None and opportunities else None,
    }
    stored = record["coverage"]
    for key, value in rebuilt_coverage.items():
        if value is None:
            require(stored.get(key) is None, f"{folder.name}: coverage {key} should be null")
        elif isinstance(value, float):
            require(stored.get(key) is not None and close(value, stored[key], 1e-12),
                    f"{folder.name}: coverage {key} mismatch")
        else:
            require(value == stored.get(key), f"{folder.name}: coverage {key} mismatch")
    return {"period": record["period"], "case": record["case"], "gate_decisions": len(gate_rows),
            "fully_rebuilt_state_rows": rebuilt_states,
            "opportunity_batches": len(opportunities), "scores_rebuilt": scores_checked,
            "score_min": min(rebuilt_scores) if rebuilt_scores else None,
            "score_max": max(rebuilt_scores) if rebuilt_scores else None,
            "coverage": rebuilt_coverage, "passed": True}


def verify_reference_parities(parities, out: Path):
    checked = []
    for item in parities:
        require(item.get("passed") is True, "Stored reference parity is not marked passed")
        folder = out / item["case"] if not Path(item["case"]).is_absolute() else Path(item["case"])
        reference = Path(item["reference"])
        require(reference.is_absolute(), "Reference parity path must be absolute")
        for name in ("equity", "fills", "trades", "actions", "holdings", "warnings",
                     "review_warnings", "decisions", "signals"):
            try:
                a = pd.read_csv(folder / f"{name}.csv", float_precision="round_trip")
            except pd.errors.EmptyDataError:
                a = pd.DataFrame()
            try:
                b = pd.read_csv(reference / f"{name}.csv", float_precision="round_trip")
            except pd.errors.EmptyDataError:
                b = pd.DataFrame()
            pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=True,
                                          obj=f"{folder.name}/{name} vs {reference.name}/{name}")
        a, b = load_json(folder / "summary.json"), load_json(reference / "summary.json")
        for key in ("rejected", "config", "total_return", "max_drawdown", "final_equity",
                    "execution_costs", "closed_trades", "win_rate"):
            require(a[key] == b[key], f"Reference summary mismatch {folder.name}: {key}")
        checked.append({"case": item["case"], "reference": str(reference),
                        "fields": item["fields"], "passed": True})
    return checked


def independently_verdict(results, failures):
    verdicts = {}
    for policy in ("fixed_breadth", "supervised", "rl_vote"):
        checks = []
        for period in PRIMARY_PERIODS:
            rows = {(r["policy"], bool(r["stress"])): r for r in results
                    if r.get("family") == "primary" and r["period"] == period}
            names = (("baseline", False), ("baseline", True), (policy, False), (policy, True))
            if not all(k in rows for k in names):
                checks.append({"period": period, "passed": False, "missing_or_failed": True})
                continue
            b, sb, c, sc = [rows[k] for k in names]
            item = {"period": period,
                    "higher_net_return": float(c["total_return"]) > float(b["total_return"]) + 1e-12,
                    "drawdown_not_worse": float(c["max_drawdown"]) >= float(b["max_drawdown"]) - 1e-12,
                    "higher_stressed_return": float(sc["total_return"]) > float(sb["total_return"]) + 1e-12}
            item["passed"] = all(item[k] for k in ("higher_net_return", "drawdown_not_worse", "higher_stressed_return"))
            checks.append(item)
        verdicts[policy] = {"checks": checks, "passed": all(c["passed"] for c in checks) and not failures}
    return verdicts


def verify_common_inputs_and_configs(out: Path, results):
    """Ensure policy groups share candidates, portfolio limits, and intended slippage."""
    by_period = {}
    checked = []
    expected_common = {"capital": 30000.0, "order_budget": 7500.0, "max_positions": 5,
                       "max_exposure": 0.9, "hold_days": 10}
    for record in results:
        period = record["period"]
        folder = out / f"{period}_{record['case']}"
        cfg = record["config"]
        for key, expected in expected_common.items():
            require(close(cfg[key], expected, 1e-12), f"{folder.name}: shared config {key} mismatch")
        wanted_slippage = 0.001 if bool(record["stress"]) else 0.0005
        require(close(cfg["slippage"], wanted_slippage, 1e-12),
                f"{folder.name}: slippage does not match stress pairing")
        signals = pd.read_csv(folder / "signals.csv", dtype={"date": str, "code": str},
                              float_precision="round_trip")
        normalized = signals.sort_values(list(signals.columns)).reset_index(drop=True)
        if period in by_period:
            reference, reference_case = by_period[period]
            pd.testing.assert_frame_equal(normalized, reference, check_dtype=False, check_exact=True,
                                          obj=f"{folder.name} signals vs {reference_case}")
        else:
            by_period[period] = (normalized, folder.name)
        checked.append(folder.name)
    return {"account_configs_and_slippage_checked": len(checked),
            "period_signal_sets_shared_across_groups": len(by_period),
            "common_config": expected_common}


def verify_files(out, started, training, frozen, summary):
    source_checks = {}
    for name, digest in started["source_hashes"].items():
        require(sha(out / "sources" / name) == digest, f"Source snapshot hash mismatch: {name}")
        require(sha(ROOT / name) == digest, f"Live source hash mismatch: {name}")
        source_checks[name] = digest
    input_checks = {}
    for path, digest in started["input_hashes"].items():
        require(sha(Path(path)) == digest, f"Live input hash mismatch: {path}")
        input_checks[path] = digest
    reference_checks = {}
    for path, digest in started["reference_hashes"].items():
        require(sha(Path(path)) == digest, f"Frozen reference hash mismatch: {path}")
        reference_checks[path] = digest
    protected_checks = {}
    for path, digest in started["protected_hashes"].items():
        require(sha(Path(path)) == digest, f"Protected file hash mismatch: {path}")
        protected_checks[path] = digest
    protocol_digest = started["protocol_sha256"]
    require(sha(out / "protocol.md") == protocol_digest, "Protocol snapshot hash mismatch")
    require(sha(ROOT / "docs" / "v09-supervised-filter-protocol.md") == protocol_digest,
            "Live protocol hash mismatch")
    model_checks = {}
    for name, digest in frozen["model_hashes"].items():
        require(sha(out / "models" / name) == digest, f"Frozen supervised model hash mismatch: {name}")
        model_checks[name] = digest
    require(frozen["fit_cutoff"] == BOUNDARY, "Model fit cutoff differs from preregistration")
    require(frozen["threshold"] == 0.5, "Frozen model threshold mismatch")
    require(frozen["freeze_before_later_data_load"] is True, "Model freeze ordering was not recorded")
    require(training["last_label_end"] is None or training["last_label_end"] < BOUNDARY,
            "Training metadata contains an immature label")
    require(training["upstream"]["v2"]["last_label_end"] < "2022-01-01" and
            training["upstream"]["v08"]["last_label_end"] < "2022-01-01",
            "Upstream 2022 models cross their training cutoff")
    qfreeze = load_json(Q_SOURCE / "training-complete.json")
    original_started = load_json(Q_SOURCE / "started.json")
    require(frozen["q_source_freeze_sha256"] == sha(Q_SOURCE / "training-complete.json"),
            "Referenced original Q freeze hash mismatch")
    for seed in SEEDS:
        name = f"q-{seed}.json"
        require(sha(Q_SOURCE / "models" / name) == qfreeze["model_hashes"][name],
                f"Original Q model hash mismatch: {name}")
        require(name in original_started["source_hashes"] or name in qfreeze["model_hashes"],
                f"Q model absent from frozen manifest: {name}")
    require(summary["source_run"] == started["reference_source"] and
            summary["q_source"] == started["q_source"], "Summary source paths differ from start record")
    return {"sources": source_checks, "inputs": input_checks, "references": reference_checks,
            "protected": protected_checks, "models": model_checks, "protocol_sha256": protocol_digest}


def rebuild_context(out: Path, calendar):
    """Rebuild breadth and market-return bins from materialized training and frozen source inputs."""
    material = out / "training-inputs"
    train_panel = pd.read_csv(material / "context-2022.csv", dtype={"date": str, "code": str},
                              float_precision="round_trip")
    train_signals = pd.read_csv(material / "signals-2022.csv", dtype={"date": str, "code": str},
                                float_precision="round_trip")
    train_two = train_signals[["date", "code"]].copy()
    train_two["pred_net_return_v2"] = .005
    train_eight = train_signals[["date", "code"]].copy()
    train_eight["pred_score_v08"] = train_signals["score"].to_numpy()
    train_days = load_json(material / "calendar-2022.json")
    train_context = rl_runner.prepare(train_panel, train_two, train_eight, train_days)[4]

    panel = pd.read_csv(rl_runner.FEATURES, dtype={"date": str, "code": str},
                        usecols=["date", "code", "adj_ret5", "market_ret5", "ret_vol20"],
                        float_precision="round_trip")
    two = pd.read_csv(rl_runner.PRED2, dtype={"date": str, "code": str},
                      usecols=["date", "code", "pred_net_return_v2"], float_precision="round_trip")
    eight = pd.read_csv(rl_runner.PRED8, dtype={"date": str, "code": str},
                        usecols=["date", "code", "pred_score_v08"], float_precision="round_trip")
    all_days = [d for d in calendar if "2022-01-01" <= d <= "2026-10-13"]
    full_context = rl_runner.prepare(panel, two, eight, all_days)[4]
    for day, values in train_context.items():
        require(day in full_context and values == full_context[day],
                f"Materialized 2022 context differs from rebuilt cached source on {day}")
    return full_context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    out = args.run.resolve()
    audit_path = out / "independent-export-audit.json"
    require(not audit_path.exists(), f"Refusing to overwrite existing audit: {audit_path}")
    started = load_json(out / "started.json")
    summary = load_json(out / "summary.json")
    training = load_json(out / "training.json")
    frozen = load_json(out / "model-freeze.json")
    require(summary["complete"] and not summary["failures"] and len(summary["results"]) == 84,
            "Run must contain 84 successful accounts and no failures")
    require(started["expected_accounts"] == 84, "Started manifest does not declare 84 accounts")
    require(len({(r["period"], r["case"]) for r in summary["results"]}) == 84,
            "Duplicate period/account keys")

    file_checks = verify_files(out, started, training, frozen, summary)
    # Verify every materialized training input against the hashes recorded before fitting.
    material = out / "training-inputs"
    material_hashes = {p.name: sha(p) for p in material.iterdir() if p.is_file()}
    require(material_hashes == training["materialized_input_hashes"], "Training materialized input hash mismatch")
    for name in ("prices-2022.csv", "context-2022.csv", "signals-2022.csv"):
        frame = pd.read_csv(material / name, dtype={"date": str, "code": str}, float_precision="round_trip")
        require(not frame.empty and frame["date"].max() <= "2022-12-31", f"{name} exceeds training date scope")
    calendar = load_json(material / "calendar-2022.json")
    require(calendar and calendar[0] >= "2022-01-01" and calendar[-1] <= "2022-12-31",
            "Training calendar crosses 2022 boundary")

    # Independently derive every batch label from the exported baseline decision and execution ledger.
    rebuilt_labels = independent_labels(out / "training-baseline")
    saved_labels = load_json(out / "training-labels.json")
    compare_label_exports(rebuilt_labels, saved_labels)
    eligible = [r for r in rebuilt_labels if r["status"] == "eligible"]
    require(all(r["label_end"] < BOUNDARY for r in eligible), "Ineligible label entered supervised fit")
    require(all(r["y"] in (0, 1) for r in eligible), "Eligible label is not binary")
    status_counts = dict(Counter(r["status"] for r in rebuilt_labels))
    classes = {str(k): sum(int(r["y"]) == k for r in eligible) for k in (0, 1)}
    require(len(eligible) == training["rows"] and status_counts == training["status_counts"] and
            classes == training["class_counts"], "Training summary counts differ from rebuilt labels")
    require(training["total_opportunities"] == len(rebuilt_labels), "Training opportunity count mismatch")
    require(training["last_label_end"] == max((r["label_end"] for r in eligible), default=None),
            "Training last label end mismatch")

    logit_path = out / "models" / "logit.json"
    model_record = load_json(logit_path) if training["fit_available"] else None
    if training["fit_available"]:
        require(model_record is not None, "Training says model fit available but logit model missing")
        require(model_record["training_labels_sha256"] == sha(out / "training-labels.json"),
                "Model references a different label export")
        refit, model, scaler = refit_logit(eligible)
        for field in ("mean", "scale", "coef"):
            stored = np.asarray(model_record[field], dtype=float)
            require(stored.shape == np.asarray(refit[field]).shape and
                    np.allclose(stored, refit[field], atol=1e-10, rtol=1e-10),
                    f"Independently refit model {field} mismatch")
        require(close(model_record["intercept"], refit["intercept"], 1e-10), "Logistic intercept mismatch")
        require(model_record["features"] == ["breadth_bin", "market_ret5_bin", "exposure_bin", "drawdown_bin",
                                              "floating_gain_bin", "holding_age_bin", "pending_count_bin"],
                "Feature list differs from preregistered 7-state input")
        require(model_record["C"] == 1.0 and model_record["class_weight"] is None and
                model_record["solver"] == "lbfgs" and model_record["max_iter"] == 2000 and
                model_record["threshold"] == 0.5, "Supervised model hyperparameters differ")
        require(model_record["training_rows"] == len(eligible) and
                model_record["last_label_end"] == training["last_label_end"], "Model training scope mismatch")
        train_probs = model.predict_proba(scaler.transform(np.asarray([r["state"] for r in eligible], dtype=float)))[:, 1]
        max_train_score_error = max(abs(saved_score(model_record, row["state"]) - float(prob))
                                    for row, prob in zip(eligible, train_probs))
        require(max_train_score_error <= 1e-12, "Saved model inference does not reproduce sklearn training scores")
    else:
        require(model_record is None, "Unfit run unexpectedly contains a logistic model")
        refit = model = scaler = None
        max_train_score_error = None

    # Restore frozen Q policies; do not update them during decision verification.
    qfreeze = load_json(Q_SOURCE / "training-complete.json")
    qpolicies = []
    for seed in SEEDS:
        qpolicies.append(QPolicy.restore(load_json(Q_SOURCE / "models" / f"q-{seed}.json")))
    q_before = [rl_runner.object_hash(q.dump()) for q in qpolicies]
    sl_support = {tuple(r["state"]) for r in eligible}
    vote_support = set.intersection(*(set(q.values) for q in qpolicies))

    # Full execution-ledger rebuild for all 84 account exports using the established RL verifier.
    _, all_frames, source_calendar, raw_events = arena.load(arena.DATASET)
    events, event_audit = arena.canonicalize_events(raw_events)
    require(not event_audit["unresolved_incomplete_groups"], "Unresolved corporate-action source groups")
    bars = arena.prepare_bars(all_frames)
    context = rebuild_context(out, source_calendar)
    account_checks, gate_checks = [], []
    for record in summary["results"]:
        folder = out / f"{record['period']}_{record['case']}"
        account_checks.append(rl_verifier.verify_account(record, folder, bars, events))
        gate_checks.append(verify_gate_account(record, folder, model_record, qpolicies, sl_support,
                                               vote_support, context, bars))
    require([rl_runner.object_hash(q.dump()) for q in qpolicies] == q_before,
            "Frozen Q policy object changed during independent verification")
    common_input_checks = verify_common_inputs_and_configs(out, summary["results"])

    # Compare every declared baseline/RL parity folder by its actual case and absolute reference path.
    parity_records = summary["checks"]["reference_parity"]
    parity_checks = verify_reference_parities(parity_records, out)
    require(len(parity_checks) == 60, f"Expected 60 reference parity records, got {len(parity_checks)}")
    independent_verdicts = independently_verdict(summary["results"], summary["failures"])
    require(independent_verdicts == summary["verdicts"], "Stored verdicts differ from independent account metrics")

    audit = {
        "audit_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "all_84_exported_accounts_rebuilt": True,
        "accounts_checked": len(account_checks),
        "total_fills_checked": sum(x["fills"] for x in account_checks),
        "training": {
            "baseline_opportunities_rebuilt": len(rebuilt_labels),
            "eligible_mature_batches": len(eligible),
            "status_counts": status_counts,
            "class_counts": classes,
            "last_label_end": training["last_label_end"],
            "no_execution_excluded_from_classes": True,
            "matured_strictly_before_cutoff": True,
            "independent_logit_refit": bool(training["fit_available"]),
            "max_training_score_error": max_train_score_error,
            "materialized_input_hashes": material_hashes,
        },
        "gate_rebuild": {
            "accounts": len(gate_checks),
            "decision_rows": sum(x["gate_decisions"] for x in gate_checks),
            "independently_rebuilt_state_rows": sum(x["fully_rebuilt_state_rows"] for x in gate_checks),
            "supervised_scores_rebuilt": sum(x["scores_rebuilt"] for x in gate_checks),
            "coverage_rebuilt": True,
            "rl_vote_used_full_allowed_before_buy_bit_mask": True,
            "supervised_state_support_size": len(sl_support),
            "rl_vote_state_intersection_size": len(vote_support),
            "supervised_opportunity_score_ranges": {
                period: {"min": min(x["score_min"] for x in gate_checks
                                     if x["period"] == period and
                                     x["case"] in ("supervised", "stress_supervised")),
                         "max": max(x["score_max"] for x in gate_checks
                                     if x["period"] == period and
                                     x["case"] in ("supervised", "stress_supervised"))}
                for period in sorted({r["period"] for r in summary["results"]})
            },
            "common_signals_and_execution_config": common_input_checks,
        },
        "reference_parities": parity_checks,
        "independent_verdicts": independent_verdicts,
        "file_hashes": file_checks,
        "event_source_audit": event_audit,
        "checks": {
            "account_cash_share_receivable_fee_nav_rebuild": True,
            "training_labels_rebuilt_from_baseline_fills_and_trades": True,
            "scaler_and_logistic_coefficients_refit_from_eligible_states": bool(training["fit_available"]),
            "all_gate_actions_and_coverage_recomputed": True,
            "allowed_actions_pending_positions_next_session_verified": True,
            "all_seven_state_dimensions_independently_rebuilt": True,
            "all_60_reference_parities_rebuilt": len(parity_checks),
            "all_verdicts_recomputed": True,
            "source_input_reference_model_protected_hashes_verified": True,
            "training_and_label_date_cuts_verified": True,
        },
    }
    with audit_path.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"PASSED: {len(account_checks)} accounts, {audit['total_fills_checked']} fills, "
          f"{len(eligible)} mature labels, {len(parity_checks)} reference parities; wrote {audit_path}")


if __name__ == "__main__":
    main()
