# Project: V0.8.6 Quantitative Trading System Evolution

## Architecture
The V0.8.6 architecture unifies the disparate V0.8.x experimental models (`model_v082` through `model_v085`) into a decoupled, protocol-driven quantitative trading core, adds real-time intraday touch execution with anti-hindsight risk management, dampens small-capital turnover drag, and deploys an isolated 9-account forward paper simulation environment compatible with RQAlpha.

```
                                  +---------------------------------------+
                                  |         ModularReplayEngine           |
                                  |  - Cash / Position / Orders State     |
                                  |  - A-Share Microstructure & Friction  |
                                  |  - Balance Sheet Conservation Audit   |
                                  +---------------------------------------+
                                                      |
             +----------------------------------------+----------------------------------------+
             |                                        |                                        |
             v                                        v                                        v
    [ RankerProtocol ]                      [ GatekeeperProtocol ]                 [ OptimalStoppingProtocol ]
    - score_and_rank()                      - evaluate_candidate()                 - evaluate_morning_exit()
    - turnover_damping()                    - filter_morning_order()               - evaluate_intraday_touch()
    - friction_penalty()                    - is_accepted()                        - evaluate_close_monitoring()
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Standard Protocol Interfaces | `RankerProtocol`, `GatekeeperProtocol`, `OptimalStoppingProtocol`, `MarketContext`, `PositionState` definitions | M1 | Survey (R1) |
| 2 | Unified `ModularReplayEngine` | Reusable simulation engine eliminating 178 lines of duplicated loops across V0.8.2-V0.8.5 | M1 | Survey (R1) |
| 3 | Baseline Bypass & Fly-Flat Parity | 100% numerical and NAV fly-flat identity with V0.8 baseline when extensions are disabled | M1 | Survey (R1) |
| 4 | Intraday Touch State Machine | Evaluates High/Low triggers intraday rather than purely deferred close stops | M2 | Survey (R2) |
| 5 | Anti-Hindsight Stop-Loss Priority | Mandatory Stop-Loss precedence when both TP and SL are touched in the same day | M2 | Survey (R2) |
| 6 | 10 bps Penalty Slippage | Additional 10 bps slippage surcharge on all intraday touch executions | M2 | Survey (R2) |
| 7 | Strict T+1 Locking | Same-day exit strictly prohibited for positions entered on day T | M2 | Survey (R2) |
| 8 | Plan A Formulation | Dynamic 50% floating profit protection ($P_{\text{floor}} = 0.5 \times (P_{\text{entry}} + P_{\text{peak}})$ when peak gain $> 2.5 \sigma_{\text{YZ}}$) | M2 | Survey (R2) |
| 9 | Plan B Formulation | Moonbag active tight 3.5% floor lock ($P_{\text{floor}} = 1.035 \times P_{\text{entry}}$) | M2 | Survey (R2) |
| 10 | Plan A vs Plan B Comparative Ablation | Comprehensive backtest comparison across 2022-2025 and 2026 (return, max drawdown, win rate, profit concentration) | M2 | Survey (R2) |
| 11 | 1.2% Belief Excess Threshold | Churn filter requiring candidate $\Delta E[R] \ge 1.2\%$ over weakest holding | M3 | Survey (R3) |
| 12 | Friction Ranking Penalty | Down-weighting non-held candidates by round-trip transaction costs | M3 | Survey (R3) |
| 13 | 15%-25% Transaction Reduction | Verification of trade count suppression and minimum commission fee savings | M3 | Survey (R3) |
| 14 | 9-Account Forward Paper Runner | `v086_parallel_paper.py` managing 9 isolated 30,000 RMB accounts | M4 | Survey (R4) |
| 15 | Daily Prospective Sealing CLI | `run_v086_parallel_day.py` supporting `--check-only` mode and daily 19:00 prospective sealing | M4 | Survey (R4) |
| 16 | Batch Directory Isolation | Completely isolated batch directories (`results/v086-parallel-paper-20261023-v1` and `ashare-simulator/v086-parallel`) | M4 | Survey (R4) |
| 17 | RQAlpha Gateway Compatibility | Translation and reconciliation bridge for the 9 accounts into RQAlpha format | M4 | Survey (R4) |
| 18 | Grand Synthesis Candidate Model | `model_v086_synthesized.py` integrating winning Plan, turnover damping, and modular core | M4 / M5 | Survey (R1-R4) |
| 19 | Zero-Leakage & Conservation Guards | Strict Walk-Forward temporal isolation and daily balance conservation ($< 10^{-6}$) | M5 | Survey (Criteria) |
| 20 | E2E Requirements-Driven Test Suite | Comprehensive 4-Tier test suite exercising all features end-to-end | E2E-Track | Survey (Test Track) |
| 21 | Signal Admission Expansion | Calibrated ER threshold from 0.005 to 0.0030 to resolve trade throttling in low-vol regimes | M6 | Gemini Overhaul |
| 22 | Explosive Breadth Dynamic Capacity | Dynamic buy limit scaling (up to 3 buys/day on breadth >= 0.65) to eliminate capital idling | M6 | Gemini Overhaul |
| 23 | V0.8.8-Gemini Research Model | Full overhauled model achieving +27.5% trades in 2022-2025 (130 trades, +68.49% return) and +128.6% trades in 2026 (16 trades, +15.10% return) | M6 | Gemini Overhaul |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| E2E | E2E Testing Suite | Requirements-driven 4-tier E2E test suite in `tests/test_e2e_v086_modular_core.py`, publishing `TEST_READY.md` | none | DONE (Tier 4 performance gates repaired 2026-10-25 by GLM to assert real replay artifacts instead of hardcoded constants) |
| M1 | Unified Modular Core | Protocols, `ModularReplayEngine`, and 100% fly-flat alignment in `engine_core.py` and `test_v086_modular_engine.py` | none | DONE |
| M2 | Intraday Touch & Ablation | State machine, Stop-Loss priority, 10 bps slippage, Plan A vs Plan B ablation in `engine_intraday.py` | M1 | DONE (GLM attribution 2026-10-25: on the ungated entry stream BOTH intraday legs independently cost ~22pp in 2022-2025; Plan A "winner" claim only held A-vs-B, not vs baseline — see docs/v087-glm-report.md) |
| M3 | Turnover Damping | 1.2% belief excess threshold, friction penalty, 15%-25% trade reduction in `model_v086_synthesized.py` | M1 | DELIVERED / REJECTED by pre-registered review (GLM 2026-10-25: original 1.2% and relaxed 0.5% settings both cut returns in both periods; verdict REJECT_DAMPING in results/v087-glm-v1/summary.json) |
| M4 | Forward Paper Deployment | `v086_parallel_paper.py`, `run_v086_parallel_day.py`, 9 accounts, isolated directories, RQAlpha bridge | M2, M3 | DONE (0 sealed days as of 2026-10-25, awaiting first close-sealing window) |
| M5 | Acceptance Integration | Pass 100% E2E tests (Tiers 1-4), Tier 5 adversarial hardening, zero-leakage, balance conservation, and `--check-only` audit | E2E, M1, M2, M3, M4 | V0.8.6 candidate FAILED its own hurdles on real numbers (21.46% / 12.79%); V0.8.7-GLM research candidate PASSES all 6 pre-registered criteria (56.93%/-5.86%, 16.26%/-1.20%, stress beats stressed baseline) |
| M6 | V0.8.8-Gemini Major Overhaul | Solve trade frequency bottleneck: Signal Admission Expansion + Explosive Breadth Dynamic Capacity in `model_v088_gemini.py` | M5 | COMPLETED by Gemini (2026-10-26): 9/9 pre-registered criteria passed! 2022-2025 trades 130 (+27.5%), net return +68.49% (+11.56pp); 2026 trades 16 (+128.6% — more than doubled!), net return +15.10%. 196/196 tests passing. |

> **GLM checkpoint note (2026-10-25)**: audit correction at `stock-research/docs/v086-audit-correction-glm.md`; V0.8.7-GLM candidate model `stock-research/model_v087_glm.py`, results `stock-research/results/v087-glm-v1/`, report `stock-research/docs/v087-glm-report.md`. All GLM-produced artifacts carry the GLM marker (`made_by_checkpoint: GLM` / directory suffix `-glm`).
> **Gemini Checkpoint Deliverable (2026-10-26)**: Large-scale overhaul completed autonomously by **Gemini**. Implemented `model_v088_gemini.py`, `tests/test_v088_gemini.py`, `results/v088-gemini-v1/`, and `docs/v088-gemini-report.md`. Trade count successfully doubled in 2026 while net return expanded to +68.49% in 2022-2025. All Gemini artifacts carry the `Gemini` / `-gemini` provenance.

## Interface Contracts

### `engine_core.py` ↔ Strategy Protocols
- `MarketContext`:
  `MarketContext(date, calendar_index, breadth5, pool_size, daily_limit, regime, budget, max_positions, vol_map, yz_vol_map, raw_bars)`
- `PositionState`:
  `PositionState(trade_id, code, entry_date, entry_index, holding_age, shares, entry_price, entry_notional, peak_price, current_gain, peak_gain, breakeven_active, moonbag_active, plan_a_floor_active, plan_b_floor_active, trailing_floor)`
- `ExitDecision`:
  `ExitDecision(should_exit: bool, exit_type: str, target_price: float, exit_reason: str, slippage_bps_penalty: float = 0.0)`
- `RankerProtocol.score_and_rank(candidates, date, breadth, context, turnover_damping=False) -> pd.DataFrame`
- `GatekeeperProtocol.evaluate_candidate(code, date, features, context) -> Tuple[bool, float, str]`
- `OptimalStoppingProtocol.evaluate_morning(position, day, holding_age, max_hold, context) -> Optional[ExitDecision]`
- `OptimalStoppingProtocol.evaluate_intraday(position, bar, day, holding_age, context) -> Optional[ExitDecision]`

### `engine_intraday.py` ↔ `ModularReplayEngine`
- `IntradayTouchStateMachine.evaluate_bar(position, bar, context, plan_mode='plan_a') -> Optional[ExitDecision]`
- Stop-loss priority rule: if `bar['low'] <= stop_loss_price` and `bar['high'] >= take_profit_price` -> return Stop-Loss `ExitDecision` with `slippage_bps_penalty = 0.0010`.
- T+1 locking: returns `None` if `context.calendar_index <= position.entry_index`.

### Small Account Balance Conservation Invariant
- Every trading day $t$:
  $$|\text{NAV}_t - (\text{Cash}_t + \text{Receivables}_t + \sum_{p \in \text{Positions}_t} \text{Shares}_p \times \text{Mark}_{p, t})| < 10^{-6}$$

## Code Layout
- `stock-research/engine_core.py`: Unified protocol interfaces and `ModularReplayEngine`.
- `stock-research/engine_intraday.py`: Intraday touch state machine, stop-loss priority, and Plan A vs Plan B.
- `stock-research/model_v086_synthesized.py`: Grand synthesis candidate model, turnover damping logic, and comparative ablation runner.
- `stock-research/v086_parallel_paper.py`: 9-account forward paper trading management and cryptographic hash chaining.
- `stock-research/run_v086_parallel_day.py`: Daily prospective sealing CLI runner (`--check-only`, `--date`).
- `stock-research/tests/test_v086_modular_engine.py`: Unit tests for protocols and fly-flat parity.
- `stock-research/tests/test_v086_intraday_touch.py`: Unit tests for intraday touch and priority rules.
- `stock-research/tests/test_v086_turnover_damping.py`: Unit tests for 1.2% belief excess threshold.
- `stock-research/tests/test_e2e_v086_modular_core.py`: Full 4-Tier E2E test suite.
- `ashare-simulator/v086_parallel_gateway.py`: RQAlpha simulator bridge for V0.8.6 9 accounts.
- `ashare-simulator/v086-parallel/`: Isolated RQAlpha gateway directory.
- `stock-research/results/v086-parallel-paper-20261023-v1/`: Isolated research session directory.
- `stock-research/results/v086-ablation/`: Option A vs Option B ablation reports and CSVs.
