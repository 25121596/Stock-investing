"""Independent reliability audit for Super-V5 MaxProfit.

Read-only: loads frozen predictions and data, never retrains and never
overwrites existing result directories. All outputs go to
``results/reliability-audit-v1``.

Tests
-----
R0  Reproduce the frozen headline number.
R1  Return decomposition: cash overlay, breadth gate, regime horizon,
    profit escalator, peak trailing, volatility-adaptive stops.
R2  Look-ahead probe: the hold-period regime is read from the same session's
    close but applied at that session's open; re-run with a one-session lag.
R3  Execution delay: trade signals 1 and 2 sessions late.
R4  Cost stress: slippage 0.05%..0.30%, doubled commission, 10 RMB minimum.
R5  Parameter perturbation around the frozen specification.
R6  Statistics: Newey-West t-stat, block bootstrap, deflated Sharpe,
    win-rate binomial test, trade concentration.
R7  Out-of-sample: 2026 slice alone, plus 2022-2026 continuous.
R8  Benchmarks: equal-weight pool index and exposure-matched comparison.
R9  Random-signal placebo, 100 draws.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from experiment_super_v5_maxprofit import SuperV5MaxProfitReplay, DATASET, DONOR, MODEL_V2_DIR
from engine import Config
from exploration_features import breadth_days
from run import load
from validate_expanded import prepare_bars

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/reliability-audit-v1"
SEED = 20261019
TRADING_DAYS = 242.0


# --------------------------------------------------------------------------
# variant engine
# --------------------------------------------------------------------------
class AuditReplay(SuperV5MaxProfitReplay):
    """Super-V5 with individually switchable components."""

    def __init__(self, *args, fixed_hold=None, fixed_sl=None, fixed_tp=None,
                 escalator=True, trailing=True, regime_lag=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_hold = fixed_hold
        self.fixed_sl = fixed_sl
        self.fixed_tp = fixed_tp
        self.escalator = escalator
        self.trailing = trailing
        self.regime_lag = regime_lag
        self._order = list(self.calendar)

    def _regime_limit(self, index):
        if self.regime_lag <= 0:
            return self.daily_limits.get(self.calendar[index], 1)
        j = max(0, index - self.regime_lag)
        return self.daily_limits.get(self.calendar[j], 1)

    def run(self):
        for index, day in enumerate(self.calendar):
            self.corporate_actions(day)

            hold_limit = self._regime_limit(index)
            max_hold = self.fixed_hold if self.fixed_hold else (12 if hold_limit >= 2 else 8)

            for code, position in list(self.positions.items()):
                if index - position.entry_index >= max_hold or position.stop_pending:
                    self.sell(position, day, index)

            for code in self.positions:
                bar = self.bars[code].get(day)
                if bar is not None and float(bar.get("open", 0.0) or 0.0) > 0:
                    self.marks[code] = float(bar["open"])

            for order in self.pending_buys:
                self.buy(order, day, index)
            self.pending_buys = []

            for code, position in self.positions.items():
                bar = self.bars[code].get(day)
                if bar is None:
                    continue
                close_px = float(bar["close"])
                self.marks[code] = close_px

                entry_px = position.entry_notional / position.shares
                current_peak = self.peak_prices.get(position.trade_id, entry_px)
                if close_px > current_peak:
                    self.peak_prices[position.trade_id] = close_px
                    current_peak = close_px

                cur_gain = (close_px - entry_px) / entry_px
                peak_gain = (current_peak - entry_px) / entry_px
                ret_vol = self.vol_map.get((position.entry_date, code), 0.025)

                eff_sl = self.fixed_sl if self.fixed_sl else max(0.025, min(0.050, 1.5 * ret_vol))
                eff_tp = self.fixed_tp if self.fixed_tp else max(0.090, min(0.150, 4.0 * ret_vol))

                if self.escalator:
                    if cur_gain >= max(0.030, 1.2 * ret_vol):
                        position.breakeven_active = True
                    if cur_gain >= max(0.080, 2.5 * ret_vol):
                        position.moonbag_active = True
                    if getattr(position, "moonbag_active", False):
                        if cur_gain < 0.035:
                            position.stop_pending = True
                    elif getattr(position, "breakeven_active", False):
                        if cur_gain < 0.005:
                            position.stop_pending = True
                    else:
                        value = position.shares * close_px + position.proceeds + position.dividends
                        if eff_sl and value < position.entry_notional * (1 - eff_sl):
                            position.stop_pending = True
                else:
                    value = position.shares * close_px + position.proceeds + position.dividends
                    if eff_sl and value < position.entry_notional * (1 - eff_sl):
                        position.stop_pending = True

                if self.trailing and peak_gain >= 0.100 and (current_peak - close_px) / current_peak >= 0.020:
                    position.stop_pending = True

                if cur_gain >= eff_tp:
                    position.stop_pending = True

            for event in self.actions_by_record.get(day, []):
                position = self.positions.get(event.get("code"))
                if position and position.shares:
                    position.pending_actions += 1
                    self.entitlements.append({"event": event, "position": position,
                                              "shares": position.shares, "posted": False})

            if self.enable_cash_overlay and self.cash > 0:
                interest = self.cash * self.daily_cash_yield
                self.cash += interest
                self.accumulated_cash_yield += interest

            nav = self.nav()
            self.equity.append({"date": day, "equity": nav, "cash": self.cash,
                                "receivables": sum(item[1] for item in self.receivables),
                                "positions": len(self.positions)})
            if self.cash < -1e-7 or any(p.shares < 0 for p in self.positions.values()):
                raise AssertionError("Negative cash or shares")

            candidates = sorted(self.candidates.get(day, []), key=lambda row: (-row["score"], row["code"]))
            queued = 0
            daily_max_buy = self.daily_limits.get(day, 1)
            available_slots = max(0, self.config.max_positions - len(self.positions))
            limit_buys = min(daily_max_buy, available_slots)
            for candidate in candidates:
                if queued >= limit_buys:
                    break
                code = candidate["code"]
                if code in self.positions or any(o["code"] == code for o in self.pending_buys):
                    continue
                bar = self.bars[code].get(day)
                if bar is None:
                    continue
                budget = min(self.config.order_budget, self.cash)
                limit = tick_price(float(bar["close"]) * (1 + self.config.max_buy_gap))
                unit_cost = limit * (1 + self.config.commission + self.config.transfer_fee + self.config.slippage)
                quantity = int(max(0, budget - self.config.min_commission) / unit_cost) // 100 * 100
                if quantity < 100:
                    self.reject("lot_size_or_budget")
                    continue
                self.pending_buys.append({"code": code, "signal_date": day, "shares": quantity, "limit": limit})
                queued += 1
        return self


def tick_price(value):
    from engine import tick
    return tick(value)


def make_config(**over):
    base = dict(capital=30000.0, order_budget=5000.0, max_positions=4, max_exposure=0.65,
                hold_days=10, stop_loss=0.03)
    base.update(over)
    return Config(**base)


# --------------------------------------------------------------------------
# statistics helpers
# --------------------------------------------------------------------------
def newey_west_t(returns: pd.Series, lags: int = 5) -> float:
    r = returns.dropna().to_numpy(dtype=float)
    n = len(r)
    if n < 10:
        return float("nan")
    mu = r.mean()
    e = r - mu
    gamma0 = (e @ e) / n
    var = gamma0
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        cov = (e[k:] @ e[:-k]) / n
        var += 2 * w * cov
    se = math.sqrt(max(var, 0) / n)
    return mu / se if se > 0 else float("nan")


def block_bootstrap(returns: np.ndarray, n_boot=2000, block=20, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(returns)
    n_blocks = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block, size=n_blocks)
        sample = np.concatenate([returns[s:s + block] for s in starts])[:n]
        out[b] = np.prod(1 + sample) - 1
    return out


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1).min())


def deflated_sharpe(sr_daily: float, n_obs: int, sr_std_daily: float,
                    n_trials: int, skew: float, kurt: float) -> tuple[float, float]:
    """Bailey & Lopez de Prado deflated Sharpe ratio and expected max SR."""
    g = 0.5772156649
    if n_trials < 2 or sr_std_daily <= 0:
        return float("nan"), float("nan")
    z1 = norm.ppf(1 - 1.0 / n_trials)
    z2 = norm.ppf(1 - 1.0 / (n_trials * math.e))
    sr0 = sr_std_daily * ((1 - g) * z1 + g * z2)
    denom = math.sqrt(max(1e-12, 1 - skew * sr_daily + (kurt - 1) / 4 * sr_daily ** 2))
    dsr = norm.cdf((sr_daily - sr0) * math.sqrt(n_obs - 1) / denom)
    return dsr, sr0


def summarize(name, rep, extra=None):
    s = rep.summary()
    eq = pd.DataFrame(rep.equity)
    daily = eq.equity.pct_change().dropna()
    exposure = float((eq.equity - eq.cash).clip(lower=0).div(eq.equity).mean())
    durations = [(pd.Timestamp(t["exit_date"]) - pd.Timestamp(t["entry_date"])).days for t in rep.trades]
    rec = {
        "variant": name,
        "total_return": s["total_return"],
        "max_drawdown": s["max_drawdown"],
        "trades": s["closed_trades"],
        "win_rate": s["win_rate"],
        "profit_factor": s["profit_factor"],
        "mean_trade_return": float(np.mean([t["net_return"] for t in rep.trades])) if rep.trades else None,
        "median_hold_calendar_days": float(np.median(durations)) if durations else None,
        "cash_yield_rmb": rep.accumulated_cash_yield,
        "costs_rmb": s["execution_costs"],
        "avg_exposure": exposure,
        "daily_mean": float(daily.mean()),
        "daily_std": float(daily.std()),
        "sharpe": float(math.sqrt(TRADING_DAYS) * daily.mean() / daily.std()) if daily.std() > 0 else None,
        "nw_t": newey_west_t(daily),
        "n_days": len(daily),
        "open_positions": len(rep.positions),
        "warnings": len(rep.warnings),
    }
    if extra:
        rec.update(extra)
    return rec, daily, rep


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading data ...")
    manifest, frames, calendar, dividends = load(DATASET)
    bars = prepare_bars(frames)
    panel = pd.read_csv(DONOR / "features.csv", dtype={"date": str, "code": str})
    pred = pd.read_csv(MODEL_V2_DIR / "predictions-h10-v2.csv", dtype={"date": str, "code": str})
    vol_map = panel.set_index(["date", "code"])["ret_vol20"].to_dict()

    breadth = breadth_days(panel, return_col="adj_ret5")
    b_lookup = breadth.set_index("date")["breadth5"].to_dict()
    p_lookup = breadth.set_index("date")["pool_size"].to_dict()

    def gate(lo=0.45, hi=0.55):
        return {d: (0 if p_lookup.get(d, 0) < 100 or b < lo else (1 if b < hi else 2))
                for d, b in b_lookup.items()}

    limits = gate()
    allowed = {d for d, v in limits.items() if v > 0}
    sig_all = pred[pred.pred_net_return_v2 >= 0.005][["date", "code", "pred_net_return_v2"]]
    sig_all = sig_all.rename(columns={"pred_net_return_v2": "score"})

    days = [d for d in calendar if "2022-01-01" <= d <= "2025-12-31"]
    days_2026 = [d for d in calendar if "2026-01-01" <= d <= "2026-10-13"]

    def run(name, sig=None, calendar_slice=None, limits_override=None, cfg=None, **kw):
        sig_use = sig_all if sig is None else sig
        lim = limits_override if limits_override is not None else limits
        sig_use = sig_use[sig_use.date.isin({d for d, v in lim.items() if v > 0})]
        cal = days if calendar_slice is None else calendar_slice
        rep = AuditReplay(frames, cal, sig_use, dividends, cfg or make_config(),
                          vol_map=vol_map, daily_limits=lim, **kw)
        rep.bars = bars
        rep.run()
        return summarize(name, rep)

    results, dailies = {}, {}

    # ---------------- R0 reproduce ----------------
    print("R0 reproduce")
    base, base_daily, base_rep = run("R0 frozen Super-V5 MaxProfit")
    results["R0"] = [base]
    dailies["R0"] = base_daily
    print(f"   {base['total_return']:.4%}  trades={base['trades']}  DD={base['max_drawdown']:.2%}")

    trades_df = pd.DataFrame(base_rep.trades)
    trades_df.to_csv(OUT / "trades-frozen.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(base_rep.equity).to_csv(OUT / "equity-frozen.csv", index=False)

    # ---------------- R1 decomposition ----------------
    print("R1 decomposition")
    r1 = []
    r1.append(base)
    r1.append(run("R1a cash overlay off", enable_cash_overlay=False)[0])
    r1.append(run("R1b breadth gate off (always 2 buys)", limits_override={d: 2 for d in calendar})[0])
    r1.append(run("R1c fixed 10-day horizon, no regime", fixed_hold=10)[0])
    r1.append(run("R1d no profit escalator", escalator=False)[0])
    r1.append(run("R1e no peak trailing", trailing=False)[0])
    r1.append(run("R1f fixed 3%/10% stops (no vol adaptation)", fixed_sl=0.03, fixed_tp=0.10)[0])
    r1.append(run("R1g all overlays off (plain core)", limits_override={d: 2 for d in calendar},
                  fixed_hold=10, fixed_sl=0.03, fixed_tp=0.10, escalator=False,
                  trailing=False, enable_cash_overlay=False)[0])
    results["R1"] = r1

    # ---------------- R2 look-ahead probe ----------------
    print("R2 regime look-ahead")
    r2 = [base]
    for lag in (1, 2):
        r2.append(run(f"R2 regime read {lag} session(s) late", regime_lag=lag)[0])
    results["R2"] = r2

    # ---------------- R3 execution delay ----------------
    print("R3 execution delay")
    r3 = [base]
    cal_index = {d: i for i, d in enumerate(calendar)}
    for k in (1, 2):
        shifted = sig_all.copy()
        shifted["date"] = shifted["date"].map(
            lambda d: calendar[min(len(calendar) - 1, cal_index[d] + k)] if d in cal_index else d)
        r3.append(run(f"R3 signal delayed {k} session(s)", sig=shifted)[0])
    results["R3"] = r3

    # ---------------- R4 cost stress ----------------
    print("R4 cost stress")
    r4 = [base]
    for slip in (0.0010, 0.0020, 0.0030):
        r4.append(run(f"R4 slippage {slip:.2%}/side", cfg=make_config(slippage=slip))[0])
    r4.append(run("R4 commission x2 + 10 RMB minimum",
                  cfg=make_config(commission=0.0005, min_commission=10.0))[0])
    r4.append(run("R4 no cash overlay + 0.20% slippage",
                  cfg=make_config(slippage=0.0020), enable_cash_overlay=False)[0])
    results["R4"] = r4

    # ---------------- R5 parameter perturbation ----------------
    print("R5 parameter perturbation")
    r5 = []
    for lo, hi in ((0.40, 0.50), (0.42, 0.52), (0.45, 0.55), (0.48, 0.58), (0.50, 0.60), (0.55, 0.65)):
        r5.append(run(f"R5 gate {lo:.2f}/{hi:.2f}", limits_override=gate(lo, hi))[0])
    for hold in (6, 8, 10, 12, 15):
        r5.append(run(f"R5 fixed hold {hold}d", fixed_hold=hold)[0])
    for tp in (0.07, 0.09, 0.12, 0.15, 0.20):
        r5.append(run(f"R5 fixed take-profit {tp:.0%}", fixed_tp=tp)[0])
    for sl in (0.02, 0.025, 0.03, 0.04, 0.05):
        r5.append(run(f"R5 fixed stop-loss {sl:.1%}", fixed_sl=sl)[0])
    results["R5"] = r5

    # ---------------- R6 statistics ----------------
    print("R6 statistics")
    eq = pd.DataFrame(base_rep.equity)
    eq["date"] = pd.to_datetime(eq.date)
    d = eq.set_index("date").equity.pct_change().dropna()
    sr_d = d.mean() / d.std()
    boot = block_bootstrap(d.to_numpy(), n_boot=2000, block=20)
    # annualised Sharpe dispersion across the variant family actually tested here
    fam = [x["sharpe"] for x in results["R1"] + results["R2"] + results["R4"] + results["R5"]
           if x.get("sharpe")]
    sr_std_daily = float(np.std(fam) / math.sqrt(TRADING_DAYS)) if len(fam) > 2 else 0.0
    n_trials = 50  # conservative count of evaluated specifications on this window
    dsr, sr0 = deflated_sharpe(sr_d, len(d), sr_std_daily, n_trials,
                               float(d.skew()), float(d.kurtosis() + 3))
    pnl = trades_df.net_pnl.to_numpy()
    tr_ret = trades_df.net_return.to_numpy()
    wins = int((pnl > 0).sum())
    binom_p = float(sum(math.comb(len(pnl), k) for k in range(wins, len(pnl) + 1)) / 2 ** len(pnl))
    top5 = float(trades_df.nlargest(5, "net_pnl").net_pnl.sum())
    gross_win = float(pnl[pnl > 0].sum())
    stats = {
        "daily_obs": int(len(d)),
        "annualized_sharpe": float(math.sqrt(TRADING_DAYS) * sr_d),
        "newey_west_t_lag5": newey_west_t(d),
        "bootstrap_total_return_p05": float(np.quantile(boot, 0.05)),
        "bootstrap_total_return_p50": float(np.quantile(boot, 0.50)),
        "bootstrap_total_return_p95": float(np.quantile(boot, 0.95)),
        "bootstrap_prob_positive": float((boot > 0).mean()),
        "skew_daily": float(d.skew()),
        "excess_kurtosis": float(d.kurtosis()),
        "trial_count_assumed": n_trials,
        "sr_dispersion_daily": sr_std_daily,
        "expected_max_sharpe_annualized_under_null": float(sr0 * math.sqrt(TRADING_DAYS)),
        "deflated_sharpe_prob": dsr,
        "trades": int(len(pnl)),
        "win_rate": float(wins / len(pnl)),
        "win_rate_binomial_p_vs_50pct": binom_p,
        "mean_trade_return": float(tr_ret.mean()),
        "median_trade_return": float(np.median(tr_ret)),
        "trade_return_t_stat": float(tr_ret.mean() / (tr_ret.std(ddof=1) / math.sqrt(len(tr_ret)))),
        "best_trade": float(tr_ret.max()),
        "worst_trade": float(tr_ret.min()),
        "top5_winners_share_of_gross_profit": top5 / gross_win if gross_win else None,
        "return_excluding_top5_winners": float((eq.equity.iloc[-1] - top5) / 30000.0 - 1),
        "cash_yield_share_of_total_pnl": float(base_rep.accumulated_cash_yield /
                                               (eq.equity.iloc[-1] - 30000.0)),
    }
    # per-year and split-half
    yr = {}
    for y in ("2022", "2023", "2024", "2025"):
        sub = [x for x in calendar if x.startswith(y)]
        yr[y] = run(f"year {y}", calendar_slice=sub)[0]
    stats["yearly"] = yr
    stats["split_half_2022_2023"] = run("2022-2023", calendar_slice=[x for x in days if x < "2024-01-01"])[0]
    stats["split_half_2024_2025"] = run("2024-2025", calendar_slice=[x for x in days if x >= "2024-01-01"])[0]
    results["R6"] = stats

    # ---------------- R7 out of sample ----------------
    print("R7 2026 out-of-sample")
    r7 = [run("R7 2026 Jan-Oct", calendar_slice=days_2026)[0],
          run("R7 2022-2026 continuous", calendar_slice=[x for x in calendar if "2022-01-01" <= x <= "2026-10-13"])[0]]
    results["R7"] = {
        "variants": r7,
        "note": "2026 is the only slice where the strategy layer was not tuned; 13 closed trades cannot establish significance",
        "forward_2026_annualized": (1 + r7[0]["total_return"]) ** (365.25 / 285) - 1 if r7[0]["total_return"] > -1 else None,
    }

    # ---------------- R8 benchmarks ----------------
    print("R8 benchmarks")
    ewm = panel[panel.date.between("2022-01-01", "2025-12-31")].groupby("date").adj_ret1.mean()
    ewm = ewm.reindex([x for x in calendar if "2022-01-01" <= x <= "2025-12-31"]).fillna(0.0)
    ew_curve = (1 + ewm).cumprod()
    bench = {
        "equal_weight_pool_total_return": float(ew_curve.iloc[-1] - 1),
        "equal_weight_pool_max_dd": max_drawdown(ew_curve.to_numpy()),
        "strategy_total_return": base["total_return"],
        "strategy_avg_exposure": base["avg_exposure"],
        "strategy_return_per_unit_exposure": base["total_return"] / base["avg_exposure"],
        "equal_weight_per_unit_exposure": float(ew_curve.iloc[-1] - 1),
    }
    results["R8"] = bench

    # ---------------- R9 placebo ----------------
    print("R9 placebo, 100 draws")
    all_codes = list(frames.keys())
    rng = np.random.default_rng(SEED)
    sig_dates = sorted(sig_all.date.unique())
    sig_dates = [x for x in sig_dates if x in allowed and x in set(days)]
    placebo = []
    for i in range(100):
        rows = []
        for dd in sig_dates:
            picked = rng.choice(all_codes, size=10, replace=False)
            for rank, c in enumerate(picked):
                rows.append({"date": dd, "code": c, "score": float(10 - rank)})
        rs = pd.DataFrame(rows)
        rep = AuditReplay(frames, days, rs, dividends, make_config(),
                          vol_map=vol_map, daily_limits=limits, enable_cash_overlay=False)
        rep.bars = bars
        rep.run()
        placebo.append(rep.summary()["total_return"])
    placebo = np.array(placebo)
    results["R9"] = {
        "draws": len(placebo),
        "mean": float(placebo.mean()),
        "std": float(placebo.std()),
        "min": float(placebo.min()),
        "max": float(placebo.max()),
        "p_value_vs_strategy_no_cash": float((placebo >= results["R1"][1]["total_return"]).mean()),
        "resolution_note": f"with {len(placebo)} draws the smallest non-zero p-value is 1/{len(placebo) + 1}",
    }

    payload = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "dataset": str(DATASET.relative_to(ROOT)),
        "prediction_file": str((MODEL_V2_DIR / 'predictions-h10-v2.csv').relative_to(ROOT)),
        "period": "2022-01-02..2025-12-31 unless stated",
        "headline": {
            "total_return": base["total_return"],
            "cagr": (1 + base["total_return"]) ** 0.25 - 1,
            "max_drawdown": base["max_drawdown"],
            "trades": base["trades"],
        },
        "R0": results["R0"],
        "R1_decomposition": results["R1"],
        "R2_lookahead_probe": results["R2"],
        "R3_execution_delay": results["R3"],
        "R4_cost_stress": results["R4"],
        "R5_parameter_perturbation": results["R5"],
        "R6_statistics": results["R6"],
        "R7_out_of_sample": results["R7"],
        "R8_benchmarks": results["R8"],
        "R9_placebo": results["R9"],
    }
    with open(OUT / "audit.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=float)
    print("saved", OUT / "audit.json")
    return payload


if __name__ == "__main__":
    main()
