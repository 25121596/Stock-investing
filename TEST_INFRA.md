# E2E Test Infra: V0.8.6 Quantitative Trading System

## Test Philosophy
- Opaque-box, requirement-driven testing derived from `ORIGINAL_REQUEST.md`.
- No reliance on implementation internals: tests verify standard protocols, fly-flat alignment, real-time touch execution, turnover suppression, accounting balance conservation, and forward paper trading isolation.
- Methodology: Category-Partition + Boundary Value Analysis (BVA) + Pairwise Combinatorial Testing + Real-World Workload Testing across 4 strict Tiers.

## Feature Inventory & Test Mapping
| # | Feature | Source | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---|---------|--------|:------:|:------:|:------:|:------:|
| 1 | Protocols & Modular Core (`RankerProtocol`, `GatekeeperProtocol`, `OptimalStoppingProtocol`, `ModularReplayEngine`) | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 2 | Baseline Bypass & 100% Fly-Flat Parity | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 3 | Intraday Touch State Machine & Anti-Hindsight Stop-Loss Priority | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ | ✓ |
| 4 | Plan A vs Plan B Ablation (50% Floating Protection vs 3.5% Moonbag Floor Lock) | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ | ✓ |
| 5 | Turnover Damping (1.2% Belief Excess Threshold & Friction Penalty) | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ | ✓ |
| 6 | Forward Paper Deployment & 9 Accounts Isolation (`--check-only`) | ORIGINAL_REQUEST §R4 | 5 | 5 | ✓ | ✓ |
| 7 | Zero-Leakage & Balance Conservation ($< 10^{-6}$) | Acceptance Criteria | 5 | 5 | ✓ | ✓ |

## Test Architecture
- Test Runner: Python built-in `unittest` runner:
  `D:\Benefits\stock-research\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_e2e_*.py" -v`
- Pass/Fail Semantics:
  - Exit code 0, all tests pass.
  - Zero tolerance for accounting discrepancy (`abs(NAV - Balance) < 1e-6`).
  - Zero tolerance for temporal leakage (`train_date < eval_date` and `label_end < eval_date`).
- Main Test Files:
  - `stock-research/tests/test_e2e_v086_modular_core.py`: Master 4-tier E2E test suite.
  - `stock-research/tests/test_e2e_ledger_conservation.py`: Microstructure and ledger conservation.
  - `stock-research/tests/test_e2e_leakage_guard.py`: Walk-forward temporal isolation.
  - `stock-research/tests/test_e2e_benchmarks.py`: Out-of-sample benchmark performance.

## 4-Tier Test Design
- **Tier 1 - Feature Coverage (Happy Path)**:
  - Protocol interface conformance and signature verification.
  - Baseline bypass 100% numerical fly-flat match with V0.8.
  - Clean take-profit touch execution with 10 bps slippage surcharge.
  - Clean stop-loss touch execution.
  - Clean trailing floor exit.
  - 1.2% turnover damping swap approval for high-conviction candidate.
  - 9-account directory and config initialization.
- **Tier 2 - Boundary & Corner Cases**:
  - Sky-and-Earth bar (天地板) dual-touch: strictly enforces Stop-Loss priority.
  - T+1 lock: fresh entry on day T cannot exit on day T regardless of intraday touch.
  - Limit down open locked ($High == Low == P_{\text{limit\_down}}$): rejects sell order gracefully (`limit_down_locked`).
  - Exact turnover hurdle boundary: $\Delta E[R] = 0.01199$ (rejected) vs $0.01201$ (approved).
  - Empty portfolio turnover handling (no swap penalty when slots are open).
  - Premature paper runner execution before 15:30 close rejected gracefully.
- **Tier 3 - Cross-Feature Combinations & Ablation**:
  - Scheme A vs Scheme B ablation comparison across multiple market regimes (bull, bear, volatile).
  - Turnover damping integrated with intraday touch: asserts $\ge 15\%$ reduction in total trades.
  - Daily balance sheet conservation equation verified across all intraday touch fills and dividend accruals.
  - Multi-seed random controls (2026101401, 2026101402, 2026101403) across both V0.8.6 and V0.8 baseline.
- **Tier 4 - Real-World Application Scenarios**:
  - 2022–2025 continuous period walk-forward evaluation (hurdle $> +45\%$, MaxDD $\le 8.32\%$).
  - 2026 out-of-sample period evaluation (hurdle $> +15\%$, MaxDD $\le 3.99\%$).
  - 10 bps double-slippage stress resilience test (positive return beating stressed baseline).
  - Paper runner `--check-only` full non-destructive dry run with 9 accounts confirmed.

## Coverage Thresholds
- Tier 1: $\ge 35$ test cases
- Tier 2: $\ge 35$ test cases
- Tier 3: $\ge 10$ test cases
- Tier 4: $\ge 5$ scenario test cases
- Total Target: $\ge 85$ test cases across the E2E test suites
