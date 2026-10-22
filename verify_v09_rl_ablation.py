"""Independent audit for frozen V0.9 action-ablation exports.

This verifies exported accounts from source prices, then independently checks
the unmasked policy choice and applies the preregistered action mask.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import arena
from v09_rl_2022 import FollowPolicy, QPolicy, SEEDS, VotePolicy, available_actions
from verify_v09_rl_2022 import verify_account

ROOT = Path(__file__).resolve().parent
PERIODS = ("2022_train", "2023", "2024", "2025", "2023-2025", "2026")
MODES = {"baseline": 0, "filter_only": 1, "exit_only": 2, "full_rl": 3}
MASKS = {"baseline": 0, "filter_only": 1, "exit_only": 2, "full_rl": 3}
METRICS = ("total_return", "max_drawdown", "final_equity", "closed_trades",
           "win_rate", "execution_costs", "open_positions", "quality_warning_count")


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def objhash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def close(a, b, tol=1e-10):
    if a is None or b is None:
        return a is None and b is None
    return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a)-float(b)) <= tol


def read_export_csv(path: Path):
    try:
        return pd.read_csv(path, float_precision="round_trip")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def expected_effect(rows):
    b, f, e, a = (float(rows[m]["final_equity"]) for m in ("baseline", "filter_only", "exit_only", "full_rl"))
    return {"filter_without_exit_rmb": f-b, "exit_without_filter_rmb": e-b,
            "filter_with_exit_rmb": a-e, "exit_with_filter_rmb": a-f,
            "interaction_rmb": a-f-e+b}


def criteria(rows, mode, period):
    b, sb = rows[(period, "baseline", False)], rows[(period, "baseline", True)]
    c, sc = rows[(period, mode, False)], rows[(period, mode, True)]
    checks = {"period": period,
              "higher_net_return": float(c["total_return"]) > float(b["total_return"])+1e-12,
              "drawdown_not_worse": float(c["max_drawdown"]) >= float(b["max_drawdown"])-1e-12,
              "higher_stressed_return": float(sc["total_return"]) > float(sb["total_return"])+1e-12}
    checks["passed"] = all(checks[k] for k in ("higher_net_return", "drawdown_not_worse", "higher_stressed_return"))
    return checks


def policy_for(row, policies):
    if row["family"] == "primary":
        if row["mode"] == "baseline":
            return FollowPolicy()
        return VotePolicy(list(policies.values()))
    return policies[int(row["seed"])]


def verify_raw(folder: Path, record, policy):
    raw_path = folder / "raw-decisions.csv"
    require(raw_path.is_file(), f"{folder.name}: raw-decisions.csv missing")
    raw_rows = pd.read_csv(raw_path, dtype={"signal_date": str, "execution_date": str},
                           float_precision="round_trip").to_dict("records")
    decisions = pd.read_csv(folder / "decisions.csv", dtype={"signal_date": str, "execution_date": str},
                           float_precision="round_trip").to_dict("records")
    require(len(raw_rows) == len(decisions), f"{folder.name}: raw/decision row count mismatch")
    mask = int(record["mask"])
    require(mask == MASKS[record["mode"]], f"{folder.name}: wrong action mask")
    for i, (raw, dec) in enumerate(zip(raw_rows, decisions)):
        for key in ("signal_date", "execution_date"):
            require(str(raw[key]) == str(dec[key]), f"{folder.name}: decision date mismatch at {i}")
        state = ast.literal_eval(str(raw["state"]))
        allowed_saved = ast.literal_eval(str(raw["allowed"]))
        require(isinstance(state, (list, tuple)) and isinstance(allowed_saved, list),
                f"{folder.name}: malformed state/allowed at {i}")
        pending, positions = int(raw["pending_before"]), int(raw["positions_before"])
        allowed = available_actions(pending > 0, positions > 0)
        require(allowed_saved == allowed, f"{folder.name}: allowed-action set mismatch at {i}")
        require(int(raw["mask"]) == mask, f"{folder.name}: row mask mismatch at {i}")
        if record["family"] == "primary" and record["mode"] == "baseline":
            raw_action = int(FollowPolicy().choose(tuple(state), allowed))
        else:
            raw_action = int(policy.choose(tuple(state), allowed))
        require(int(raw["raw_action"]) == raw_action,
                f"{folder.name}: recomputed raw action mismatch at {i}")
        action = raw_action & mask
        require(action in allowed, f"{folder.name}: masked action unavailable at {i}")
        require(int(raw["action"]) == action == int(dec["action"]),
                f"{folder.name}: executed action/mask mismatch at {i}")
        require(str(raw["state"]) == str(dec["state"]), f"{folder.name}: state trace mismatch at {i}")
        require(pending == int(dec["pending_before"]) and positions == int(dec["positions_before"]),
                f"{folder.name}: pre-action account state mismatch at {i}")
        require(str(raw["signal_date"]) < str(raw["execution_date"]),
                f"{folder.name}: noncausal raw decision at {i}")
        require((not (action & 1) or pending > 0) and (not (action & 2) or positions > 0),
                f"{folder.name}: disabled/invalid action occurred at {i}")
    return len(raw_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    out = args.run.resolve()
    summary, started, freeze = load(out/"summary.json"), load(out/"started.json"), load(out/"model-freeze.json")
    require(summary.get("complete") is True and len(summary.get("results", [])) == 156 and not summary.get("failures"),
            "Run incomplete, wrong result count, or failures present")
    rows = summary["results"]
    keys = [(r["period"], r["case"]) for r in rows]
    require(len(set(keys)) == len(keys), "Duplicate period/case records")
    require(len(rows) == 156, "Expected 156 result rows")
    primary = [r for r in rows if r["family"] == "primary"]
    seeds = [r for r in rows if r["family"] == "seed"]
    require(len(primary) == 48 and len(seeds) == 108, "Primary/seed family counts mismatch")

    for name, digest in started.get("source_hashes", {}).items():
        require(sha(out/"sources"/name) == digest, f"Source snapshot hash mismatch: {name}")
    for path, digest in started.get("input_hashes", {}).items():
        p = Path(path)
        require(p.is_absolute(), f"Input hash path is not absolute: {path}")
        require(sha(p) == digest, f"Input hash mismatch: {path}")
    for path, digest in started.get("protected_hashes", {}).items():
        require(sha(Path(path)) == digest, f"Protected hash mismatch: {path}")
    for path, digest in started.get("reference_hashes", {}).items():
        require(sha(Path(path)) == digest, f"Reference hash mismatch: {path}")
    protocol = out/"protocol.md"
    require(protocol.is_file() and sha(protocol) == started["protocol_sha256"], "Protocol snapshot hash mismatch")
    require(freeze.get("source_freeze_sha256"), "Frozen source freeze hash missing")
    source_run = Path(freeze["source_run"])
    source_freeze = source_run/"training-complete.json"
    require(source_freeze.is_file() and sha(source_freeze) == freeze["source_freeze_sha256"],
            "Source run training freeze changed")
    policies, model_hashes = {}, {}
    for seed in SEEDS:
        name = f"q-{seed}.json"
        path = out/"models"/name
        digest = sha(path)
        require(freeze["model_hashes"].get(name) == digest, f"Frozen Q hash mismatch: {name}")
        require(sha(source_run/"models"/name) == digest, f"Original V0.9 Q hash changed: {name}")
        policies[seed] = QPolicy.restore(load(path))
        model_hashes[name] = digest
    require(freeze.get("seeds") == list(SEEDS) or tuple(freeze.get("seeds", [])) == tuple(SEEDS), "Frozen seed list mismatch")

    _, frames, _, raw_events = arena.load(arena.DATASET)
    events, event_audit = arena.canonicalize_events(raw_events)
    require(not event_audit["unresolved_incomplete_groups"], "Unresolved corporate actions")
    bars = arena.prepare_bars(frames)
    account_audits, raw_count = [], 0
    result_by_key = {}
    for r in rows:
        require(r["period"] in PERIODS, f"Unknown period {r['period']}")
        require(r["mode"] in MODES and int(r["mask"]) == MASKS[r["mode"]], "Mode/mask mismatch")
        if r["family"] == "seed":
            require(r["seed"] in SEEDS and r["mode"] in ("filter_only", "exit_only", "full_rl"), "Bad seed diagnostic")
            require(r["policy"] == f"seed_{r['seed']}_{r['mode']}", "Seed policy name mismatch")
        else:
            require(r["seed"] is None and r["policy"] == r["mode"], "Primary policy identity mismatch")
        folder = out/f"{r['period']}_{r['case']}"
        require(folder.is_dir(), f"Missing case folder: {folder.name}")
        result_by_key[(r["period"], r["case"])] = r
        account_audits.append(verify_account(r, folder, bars, events))
        raw_count += verify_raw(folder, r, policy_for(r, policies))
        exp = float(np.mean([(float(e["equity"])-float(e["cash"])-float(e["receivables"]))/float(e["equity"])
                             for e in pd.read_csv(folder/"equity.csv", float_precision="round_trip").to_dict("records")]))
        require(close(exp, r["average_exposure"], 1e-12), f"{folder.name}: average exposure mismatch")

    expected_cases = set()
    for period in PERIODS:
        for stress in (False, True):
            pfx = "stress_" if stress else ""
            for mode in MODES:
                expected_cases.add((period, pfx+mode))
            for seed in SEEDS:
                for mode in ("filter_only", "exit_only", "full_rl"):
                    expected_cases.add((period, pfx+f"seed_{seed}_{mode}"))
    require(set(keys) == expected_cases, "Case set is incomplete or contains extras")

    # Compare all 60 diagnostic parity records against the exact referenced exports.
    parity = summary.get("checks", {}).get("reference_parity")
    require(isinstance(parity, list) and len(parity) == 60, "Expected 60 reference parity records")
    checked_parity = []
    parity_cases = set()
    parity_files = ("equity", "fills", "trades", "actions", "holdings", "warnings",
                    "review_warnings", "decisions", "signals")
    for p in parity:
        case = p["case"]
        require(p.get("passed") is True, f"Reference parity record failed: {case}")
        require(case in {f"{r['period']}_{r['case']}" for r in rows} and case not in parity_cases,
                f"Unexpected/duplicate parity case: {case}")
        parity_cases.add(case)
        current_folder = out/case
        reference = Path(p["reference"]).resolve()
        require(reference.is_dir() and source_run.resolve() in reference.parents,
                f"Parity reference is missing or outside frozen source run: {reference}")
        require(p.get("fields") == list(parity_files), f"Parity field list mismatch: {case}")
        for name in parity_files:
            current_csv = read_export_csv(current_folder/f"{name}.csv")
            reference_csv = read_export_csv(reference/f"{name}.csv")
            pd.testing.assert_frame_equal(current_csv, reference_csv, check_dtype=False, check_exact=True)
        current_summary, reference_summary = load(current_folder/"summary.json"), load(reference/"summary.json")
        for metric in ("rejected", "config", *METRICS):
            require(current_summary.get(metric) == reference_summary.get(metric),
                    f"Reference parity mismatch {case}/{metric}")
        checked_parity.append({"case": case, "reference": str(reference), "fields": list(parity_files), "passed": True})
    expected_parity_cases = {
        f"{period}_{stress}{name}"
        for period in PERIODS for stress in ("", "stress_")
        for name in ("baseline", "full_rl", *(f"seed_{seed}_full_rl" for seed in SEEDS))
    }
    require(parity_cases == expected_parity_cases, "Reference parity case set mismatch")

    # Recompute group effects for each period and stress arm.
    computed_effects = []
    for period in PERIODS:
        for stress in (False, True):
            arms = {}
            for mode in MODES:
                case = ("stress_" if stress else "")+mode
                arms[mode] = result_by_key[(period, case)]
            computed_effects.append({"period": period, "stress": stress, **expected_effect(arms)})
    stored_effects = summary.get("effects", [])
    require(len(stored_effects) == len(computed_effects), "Effects row count mismatch")
    effect_fields = tuple(expected_effect({m: {"final_equity": 0} for m in MODES}))
    for expected, stored in zip(computed_effects, stored_effects):
        require(expected["period"] == stored["period"] and bool(expected["stress"]) == bool(stored["stress"]), "Effects key mismatch")
        for field in effect_fields:
            require(close(expected[field], stored[field], 1e-9), f"Effects mismatch: {expected['period']} {field}")

    verdicts = {}
    lookup = {}
    for r in rows:
        if r["family"] == "primary":
            lookup[(r["period"], r["mode"], bool(r["stress"]))] = r
    for mode in ("filter_only", "exit_only", "full_rl"):
        checks = [criteria(lookup, mode, period) for period in ("2023-2025", "2026")]
        verdicts[mode] = {"checks": checks, "passed": all(c["passed"] for c in checks)}
    require(verdicts == summary.get("verdicts"), "Verdicts differ from independently recomputed compare criteria")
    for check in ("q_objects_unchanged", "model_files_unchanged", "original_models_unchanged",
                  "sources_unchanged", "inputs_unchanged", "references_unchanged",
                  "protected_unchanged", "protocol_unchanged"):
        require(summary.get("checks", {}).get(check) is True, f"Summary integrity check failed/missing: {check}")

    prior_run = out.parent/"v09-rl-ablation-20261021-v1"
    version_comparison = None
    if out.name == "v09-rl-ablation-20261021-v2":
        require(prior_run.is_dir(), "Required v1 comparison run is missing")
        prior_summary = load(prior_run/"summary.json")
        prior_freeze = load(prior_run/"model-freeze.json")
        require(prior_summary.get("complete") is True and len(prior_summary.get("results", [])) == 156
                and not prior_summary.get("failures"), "v1 comparison run is incomplete")
        prior_results = {(r["period"], r["case"]): r for r in prior_summary["results"]}
        current_results = {(r["period"], r["case"]): r for r in rows}
        require(prior_results.keys() == current_results.keys(), "v1/v2 case sets differ")
        changed = []
        for key in sorted(current_results):
            if prior_results[key] != current_results[key]:
                changed.append(f"{key[0]}/{key[1]}")
        require(not changed, f"v1/v2 result records changed: {changed[:5]}")
        require(prior_freeze["model_hashes"] == freeze["model_hashes"], "v1/v2 model SHA-256 hashes differ")
        version_comparison = {"compared_run": str(prior_run), "result_records_compared": 156,
                              "all_result_fields_identical": True,
                              "model_hashes_identical": dict(model_hashes)}

    audit = {"audit_version": 1, "created_utc": datetime.now().astimezone().isoformat(),
             "passed": True, "accounts_checked": len(account_audits), "raw_decisions_checked": raw_count,
             "total_fills_checked": sum(x["fills"] for x in account_audits),
             "daily_nav_rows_checked": sum(x["sessions"] for x in account_audits),
             "actions_checked": raw_count,
             "all_156_exports_rebuilt": True, "all_60_reference_parities_rechecked": True,
             "original_q_hashes_unchanged": model_hashes,
             "checks": {"source_input_protected_reference_protocol_hashes": True,
                        "independent_account_ledger_metrics": True, "raw_policy_votes_and_masks": True,
                        "disabled_actions_absent": True, "average_exposure": True,
                        "effects_recomputed": computed_effects, "verdicts_recomputed": verdicts,
                        "reference_parity": checked_parity,
                        "v1_v2_comparison": version_comparison}}
    destination = out/"independent-export-audit.json"
    require(not destination.exists(), "Refusing to overwrite existing independent-export-audit.json")
    destination.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"PASSED: {len(account_audits)} accounts, {raw_count} raw decisions; wrote {destination}")


if __name__ == "__main__":
    main()
