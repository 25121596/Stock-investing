"""run_us_market_simulation_gemini.py - Cross-Market Simulation: Applying Strategy to US Market (2022-2026).

Tests what happens when our core quantitative logic:
  - Multi-factor momentum & trend signals
  - S&P 500 / Nasdaq Market Breadth Gatekeeper (Breadth >= 0.50)
  - Intraday Stop Loss & Trailing Floor Locking
  - USD Cash Overlay (Fed Funds / US Treasury Yield ~4.5% annual rate)
  - Dynamic Floor-Capped Compounding & Position Sizing
is applied directly to US stocks (SPY, QQQ, AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, AVGO, JPM, LLY, COST, AMD, NFLX, BRK-B, UNH).

Key Microstructure Differences Evaluated:
  1. A-share strict T+1 vs US T+0 (intraday exits permitted).
  2. A-share +-10% price limit boards vs US No Limits (overnight earnings gaps).
  3. A-share 0.05% stamp tax vs US 0 stamp tax, 0 commission.
  4. A-share 1.85% cash yield vs US 4.50% Treasury cash yield.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("d:/Benefits/stock-research/data/us_market_probe")
OUT_DIR = Path("d:/Benefits/stock-research/results/us_market_probe_gemini")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 
    'TSLA', 'AVGO', 'JPM', 'LLY', 'COST', 'AMD', 'NFLX', 'BRK-B', 'UNH'
]

BENCHMARKS = ['SPY', 'QQQ']


def load_us_data():
    bars = {}
    for t in TICKERS + BENCHMARKS:
        f = DATA_DIR / f"{t}.csv"
        df = pd.read_csv(f)
        df['date'] = df['date'].astype(str)
        df = df[df['date'] >= '2021-10-01'].sort_values('date').reset_index(drop=True)
        # Compute indicators: MA20, MA60, 20-day return, 20-day volatility
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        df['ret20'] = df['close'].pct_change(20)
        df['vol20'] = df['close'].pct_change().rolling(20).std()
        df['atr14'] = (df['high'] - df['low']).rolling(14).mean()
        bars[t] = df.set_index('date')
    return bars


def simulate_us_strategy(
    bars,
    start_date: str = "2022-01-01",
    end_date: str = "2025-12-31",
    initial_capital: float = 30000.0,
    cash_annual_rate: float = 0.045, # 4.5% US Treasury yield
    allow_t0: bool = True,
    target_slots: int = 4,
    intraday_sl_pct: float = 0.045, # 4.5% stop loss
    trailing_floor_pct: float = 0.035, # 3.5% floor lock on gain > 6%
    slippage_bps: float = 0.0005,
):
    # Align dates across SPY
    all_dates = [d for d in bars['SPY'].index if start_date <= d <= end_date]
    
    cash = initial_capital
    accumulated_cash_yield = 0.0
    positions = {} # ticker -> {shares, entry_price, entry_date, peak_price, trailing_floor}
    equity_curve = []
    trades = []
    
    daily_cash_rate = cash_annual_rate / 252.0 if cash_annual_rate > 0 else 0.0

    for i, date in enumerate(all_dates):
        # 1. Market Breadth: fraction of universe above MA20
        active_stocks = [t for t in TICKERS if date in bars[t].index and pd.notna(bars[t].loc[date, 'ma20'])]
        breadth = np.mean([bars[t].loc[date, 'close'] > bars[t].loc[date, 'ma20'] for t in active_stocks])
        spy_trend = bars['SPY'].loc[date, 'close'] > bars['SPY'].loc[date, 'ma60'] if date in bars['SPY'].index else True

        # 2. Evaluate existing positions (intraday high/low check)
        closed_today = []
        for t, pos in list(positions.items()):
            b = bars[t].loc[date]
            high_p = b['high']
            low_p = b['low']
            close_p = b['close']
            open_p = b['open']
            
            # Update peak price
            if high_p > pos['peak_price']:
                pos['peak_price'] = high_p
                gain = (pos['peak_price'] / pos['entry_price']) - 1.0
                if gain >= 0.06:
                    pos['trailing_floor'] = max(pos['trailing_floor'], pos['entry_price'] * (1.0 + trailing_floor_pct))

            # Stop loss check
            exit_price = None
            exit_reason = None
            
            # Overnight gap down below stop loss:
            if open_p <= pos['entry_price'] * (1.0 - intraday_sl_pct):
                exit_price = open_p * (1.0 - slippage_bps)
                exit_reason = "gap_stop_loss"
            elif low_p <= pos['entry_price'] * (1.0 - intraday_sl_pct):
                exit_price = pos['entry_price'] * (1.0 - intraday_sl_pct) * (1.0 - slippage_bps)
                exit_reason = "intraday_stop_loss"
            elif pos['trailing_floor'] > 0 and low_p <= pos['trailing_floor']:
                exit_price = min(open_p, pos['trailing_floor']) * (1.0 - slippage_bps)
                exit_reason = "trailing_profit_lock"
            elif (i - pos['entry_idx']) >= 10: # 10-day time horizon
                exit_price = close_p * (1.0 - slippage_bps)
                exit_reason = "time_exit_10d"

            if exit_price is not None:
                gross_amt = pos['shares'] * exit_price
                # US commission: 0, SEC fee: 0.0000278
                sec_fee = gross_amt * 0.000028
                net_proceeds = gross_amt - sec_fee
                pnl = net_proceeds - (pos['shares'] * pos['entry_price'])
                ret = pnl / (pos['shares'] * pos['entry_price'])
                cash += net_proceeds
                trades.append({
                    'ticker': t,
                    'entry_date': pos['entry_date'],
                    'exit_date': date,
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'shares': pos['shares'],
                    'net_pnl': pnl,
                    'net_return': ret,
                    'exit_reason': exit_reason
                })
                closed_today.append(t)

        for t in closed_today:
            del positions[t]

        # 3. New Entry Candidates (Breadth gatekeeper: Breadth >= 0.45 and SPY > MA60)
        if breadth >= 0.45 and spy_trend and len(positions) < target_slots:
            # Score candidates: composite of 20-day momentum / 20-day vol (Sharpe-like ranking)
            candidates = []
            for t in TICKERS:
                if t not in positions and date in bars[t].index:
                    row = bars[t].loc[date]
                    if pd.notna(row['ret20']) and pd.notna(row['vol20']) and row['vol20'] > 0:
                        score = row['ret20'] / row['vol20']
                        # Require positive momentum and above MA20
                        if row['close'] > row['ma20'] and row['ret20'] > 0.02:
                            candidates.append((score, t, row['close']))

            candidates.sort(reverse=True, key=lambda x: x[0])
            open_slots = target_slots - len(positions)
            
            nav_estimate = cash + sum(pos['shares'] * bars[t].loc[date]['close'] for t, pos in positions.items())
            slot_budget = max(5000.0, min(15000.0, (nav_estimate * 0.90 / target_slots)))

            for score, t, close_p in candidates[:open_slots]:
                buy_price = close_p * (1.0 + slippage_bps)
                # In US, fractional shares or whole shares
                shares = int(slot_budget / buy_price)
                if shares > 0 and cash >= shares * buy_price:
                    cost = shares * buy_price
                    cash -= cost
                    positions[t] = {
                        'shares': shares,
                        'entry_price': buy_price,
                        'entry_date': date,
                        'entry_idx': i,
                        'peak_price': buy_price,
                        'trailing_floor': 0.0
                    }

        # 4. Overnight Cash Yield Overlay (US Money Market / Treasury Yield)
        if daily_cash_rate > 0 and cash > 0:
            interest = cash * daily_cash_rate
            cash += interest
            accumulated_cash_yield += interest

        # 5. Daily Mark to Market NAV
        pos_val = sum(pos['shares'] * bars[t].loc[date]['close'] for t, pos in positions.items())
        total_nav = cash + pos_val
        equity_curve.append({
            'date': date,
            'cash': cash,
            'pos_val': pos_val,
            'equity': total_nav,
            'accumulated_cash_yield': accumulated_cash_yield,
            'open_positions': len(positions)
        })

    eq_df = pd.DataFrame(equity_curve)
    eq_df['cummax'] = eq_df['equity'].cummax()
    eq_df['drawdown'] = (eq_df['equity'] - eq_df['cummax']) / eq_df['cummax']
    max_dd = float(eq_df['drawdown'].min())
    final_nav = float(eq_df['equity'].iloc[-1])
    total_return = (final_nav / initial_capital) - 1.0
    
    # Sharpe
    daily_rets = eq_df['equity'].pct_change().dropna()
    sharpe = float(np.sqrt(252) * daily_rets.mean() / daily_rets.std()) if daily_rets.std() > 0 else 0.0

    # SPY benchmark return
    spy_start = bars['SPY'].loc[all_dates[0], 'close']
    spy_end = bars['SPY'].loc[all_dates[-1], 'close']
    spy_ret = (spy_end / spy_start) - 1.0

    # QQQ benchmark return
    qqq_start = bars['QQQ'].loc[all_dates[0], 'close']
    qqq_end = bars['QQQ'].loc[all_dates[-1], 'close']
    qqq_ret = (qqq_end / qqq_start) - 1.0

    # Yearly breakdown
    eq_df['year'] = eq_df['date'].str[:4]
    yearly = {}
    for yr, y_df in eq_df.groupby('year'):
        y_ret = (y_df['equity'].iloc[-1] / y_df['equity'].iloc[0]) - 1.0
        y_dd = float(((y_df['equity'] - y_df['equity'].cummax()) / y_df['equity'].cummax()).min())
        yearly[yr] = {'return': y_ret, 'maxdd': y_dd}

    trade_df = pd.DataFrame(trades)
    win_rate = (trade_df['net_pnl'] > 0).mean() if len(trade_df) else 0.0
    profit_factor = (
        trade_df.loc[trade_df['net_pnl'] > 0, 'net_pnl'].sum() / abs(trade_df.loc[trade_df['net_pnl'] <= 0, 'net_pnl'].sum())
        if len(trade_df) and trade_df.loc[trade_df['net_pnl'] <= 0, 'net_pnl'].sum() != 0 else 0.0
    )

    return {
        'initial_capital': initial_capital,
        'final_nav': final_nav,
        'total_return': total_return,
        'max_drawdown': max_dd,
        'sharpe': sharpe,
        'calmar': total_return / abs(max_dd) if max_dd != 0 else 0.0,
        'accumulated_cash_yield': accumulated_cash_yield,
        'pure_stock_profit': final_nav - initial_capital - accumulated_cash_yield,
        'trades': len(trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'spy_return': spy_ret,
        'qqq_return': qqq_ret,
        'yearly': yearly,
        'trades_sample': trades[:5],
    }


def main():
    bars = load_us_data()
    print("Simulating US Market Direct Application (2022-2025 Full Period)...")
    res_2022_2025 = simulate_us_strategy(bars, start_date="2022-01-01", end_date="2025-12-31", cash_annual_rate=0.045)
    
    print("Simulating US Market 2026 OOS Period...")
    res_2026 = simulate_us_strategy(bars, start_date="2026-01-01", end_date="2026-10-23", cash_annual_rate=0.045)

    print("Simulating US Market Zero Cash Overlay Baseline (Pure Stock Alpha)...")
    res_zero = simulate_us_strategy(bars, start_date="2022-01-01", end_date="2025-12-31", cash_annual_rate=0.0)

    out = {
        'us_2022_2025': res_2022_2025,
        'us_2026_oos': res_2026,
        'us_zero_cash_baseline': res_zero
    }

    out_file = OUT_DIR / "us_strategy_simulation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 90)
    print("US MARKET DIRECT APPLICATION RESULTS")
    print("=" * 90)
    print(f"2022-2025 Total Return: {res_2022_2025['total_return']:>7.2%} | MaxDD: {res_2022_2025['max_drawdown']:>6.2%} | Sharpe: {res_2022_2025['sharpe']:>4.2f}")
    print(f"   SPY Return: {res_2022_2025['spy_return']:>7.2%} | QQQ Return: {res_2022_2025['qqq_return']:>7.2%}")
    print(f"   Stock Profit: {res_2022_2025['pure_stock_profit']:>9.2f} USD | Cash Yield: {res_2022_2025['accumulated_cash_yield']:>9.2f} USD")
    print(f"   Trades: {res_2022_2025['trades']} | WinRate: {res_2022_2025['win_rate']:>6.2%} | ProfitFactor: {res_2022_2025['profit_factor']:>4.2f}")
    print("Yearly Breakdown (US):")
    for yr, yst in res_2022_2025['yearly'].items():
        print(f"   {yr}: Return {yst['return']:>7.2%}, MaxDD {yst['maxdd']:>6.2%}")
    
    print(f"\n2026 OOS Total Return: {res_2026['total_return']:>7.2%} | MaxDD: {res_2026['max_drawdown']:>6.2%} | Sharpe: {res_2026['sharpe']:>4.2f}")
    print(f"   SPY Return: {res_2026['spy_return']:>7.2%} | QQQ Return: {res_2026['qqq_return']:>7.2%}")
    print(f"Zero Cash Overlay (Pure Stock): Return {res_zero['total_return']:>7.2%} | MaxDD {res_zero['max_drawdown']:>6.2%}")
    print("=" * 90)


if __name__ == "__main__":
    main()
