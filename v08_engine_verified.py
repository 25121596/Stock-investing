"""V0.8 execution rules with mandatory data checks and actual elapsed annualization."""
import pandas as pd
import math
from dataclasses import asdict
from audit_gemini_v08_codex import ObservedReplay


class VerifiedV08Replay(ObservedReplay):
    def __init__(self, *args, **kwargs):
        self.review_warnings = []
        super().__init__(*args, **kwargs)

    def corporate_actions(self, day):
        super().corporate_actions(day)
        if self.review_warnings or self.warnings:
            raise ValueError(f'Blocked quality issue on {day}: {self.review_warnings + self.warnings}')

    def summary(self):
        result = super().summary()
        elapsed = (pd.Timestamp(self.calendar[-1]) - pd.Timestamp(self.calendar[0])).days
        result['cagr'] = (result['final_equity'] / self.config.capital) ** (365.25 / elapsed) - 1 if elapsed else None
        result['calmar'] = result['cagr'] / abs(result['max_drawdown']) if elapsed and result['max_drawdown'] else None
        result['quality_warning_count'] = len(self.review_warnings + self.warnings)
        result['open_positions'] = len(self.positions)
        result['receivables'] = sum(amount for _, amount in self.receivables)
        result['rejected'] = dict(self.rejected)
        result['config'] = asdict(self.config)
        curve = pd.Series([self.config.capital] + [r['equity'] for r in self.equity], dtype=float)
        returns = curve.pct_change().dropna()
        deviation = returns.std()
        result['sharpe'] = float((242 ** .5) * returns.mean() / deviation) if len(returns)>1 and deviation>1e-12 else None
        downside = returns[returns<0].std()
        result['sortino'] = float((242 ** .5) * returns.mean() / downside) if returns.lt(0).sum()>1 and downside>1e-12 else None
        result['sharpe_risk_free_annual_assumption'] = 0.0
        for key,value in list(result.items()):
            if isinstance(value,float) and not math.isfinite(value):
                result[key] = None
        result['validation_scope'] = 'Previously viewed historical replay, not an independent blind test'
        return result
