"""Execution Bridge for Super-V0.8 / V7 Quants.

Unified Trading Adapter supporting:
1. Paper Sandbox Mode: Independent zero-cost local paper broker with daily A-share settlement.
2. QMT / xtquant Mode: Direct broker connection to official simulated or live trading via miniQMT.

Usage:
    from trader_bridge import TraderBridge, ExecutionConfig
    bridge = TraderBridge(config=ExecutionConfig(mode="paper", capital=30000.0))
    bridge.sync_account()
    bridge.execute_target_orders(target_orders)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("TraderBridge")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")


@dataclass
class Position:
    code: str
    volume: int = 0
    can_use_volume: int = 0  # T+1 available for selling
    avg_price: float = 0.0
    current_price: float = 0.0
    entry_date: str = ""
    stop_loss_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.volume * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return self.volume * (self.current_price - self.avg_price)

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.avg_price <= 0:
            return 0.0
        return (self.current_price - self.avg_price) / self.avg_price


@dataclass
class AccountSnapshot:
    date: str
    total_asset: float
    cash: float
    market_value: float
    positions: Dict[str, Position] = field(default_factory=dict)


@dataclass
class ExecutionConfig:
    mode: str = "paper"  # "paper" or "qmt"
    capital: float = 30000.0
    qmt_path: str = r"D:\qmt\userdata_mini"
    qmt_account_id: str = "your_simulate_account_id"
    commission_rate: float = 0.00025  # 佣金 0.025%
    min_commission: float = 5.0       # 最低 5 元
    stamp_duty_rate: float = 0.0005   # 印花税 0.05% (仅卖出)
    transfer_fee_rate: float = 0.00001 # 过户费 0.001%
    slippage: float = 0.0015          # 滑点 0.15%
    max_position_weight: float = 0.25 # 单票最大仓位 25%
    storage_dir: Path = Path("results/live_trading")


class TraderBridge:
    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        self.config.storage_dir.mkdir(parents=True, exist_ok=True)
        self.positions_file = self.config.storage_dir / "positions.json"
        self.ledger_file = self.config.storage_dir / "ledger.csv"
        
        # State
        self.cash = self.config.capital
        self.positions: Dict[str, Position] = {}
        self.xt_trader = None
        self.xt_account = None

        if self.config.mode == "qmt":
            self._init_qmt()
        else:
            self._init_paper()

    def _init_paper(self):
        """Initialize local paper trading ledger."""
        logger.info(f"Initializing Local Paper Sandbox Broker with capital: {self.config.capital:,.2f} RMB")
        if self.positions_file.exists():
            try:
                data = json.loads(self.positions_file.read_text(encoding="utf-8"))
                self.cash = data.get("cash", self.config.capital)
                for code, p in data.get("positions", {}).items():
                    self.positions[code] = Position(**p)
                logger.info(f"Loaded existing paper portfolio: Cash={self.cash:,.2f}, Positions={len(self.positions)}")
            except Exception as e:
                logger.warning(f"Failed to load existing portfolio, starting fresh: {e}")

    def _init_qmt(self):
        """Initialize miniQMT connection via xtquant."""
        try:
            from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
            from xtquant.xttype import StockAccount
            from xtquant import xtconstant
            self.xtconstant = xtconstant

            session_id = int(datetime.now().strftime("%H%M%S"))
            self.xt_trader = XtQuantTrader(self.config.qmt_path, session_id)
            self.xt_account = StockAccount(self.config.qmt_account_id, "STOCK")
            
            self.xt_trader.start()
            res = self.xt_trader.connect()
            if res == 0:
                logger.info("Successfully connected to QMT (miniQMT) Gateway!")
                self.xt_trader.subscribe(self.xt_account)
            else:
                logger.error(f"Failed to connect to miniQMT (Code: {res}). Please ensure miniQMT is running.")
                raise ConnectionError("miniQMT connection failed")
        except ImportError:
            logger.error("`xtquant` is not installed! Copy `xtquant` from miniQMT directory or use paper mode.")
            raise

    def sync_account(self) -> AccountSnapshot:
        """Sync balance and positions."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.config.mode == "qmt" and self.xt_trader:
            asset = self.xt_trader.query_stock_asset(self.xt_account)
            if asset:
                self.cash = asset.cash
                total_asset = asset.total_asset
            else:
                total_asset = self.cash

            qmt_positions = self.xt_trader.query_stock_positions(self.xt_account) or []
            self.positions = {}
            mv = 0.0
            for p in qmt_positions:
                pos = Position(
                    code=p.stock_code,
                    volume=p.volume,
                    can_use_volume=p.can_use_volume,
                    avg_price=p.open_price,
                    current_price=p.market_value / p.volume if p.volume > 0 else 0.0,
                )
                self.positions[p.stock_code] = pos
                mv += p.market_value
            return AccountSnapshot(date=now_str, total_asset=total_asset, cash=self.cash, market_value=mv, positions=self.positions)
        else:
            # Paper mode
            mv = sum(p.market_value for p in self.positions.values())
            total = self.cash + mv
            return AccountSnapshot(date=now_str, total_asset=total, cash=self.cash, market_value=mv, positions=self.positions)

    def order_buy(self, code: str, price: float, budget: float, stop_loss: float = 0.0, date_str: str = "") -> Optional[dict]:
        """Execute Buy Order with 100-share rounding and cost calculation."""
        if price <= 0:
            logger.warning(f"Invalid buy price for {code}: {price}")
            return None

        # Effective unit cost including slippage and fees
        effective_price = price * (1 + self.config.slippage)
        max_shares = int(budget / (effective_price * (1 + self.config.commission_rate + self.config.transfer_fee_rate)))
        volume = (max_shares // 100) * 100

        if volume < 100:
            logger.info(f"Budget {budget:.2f} insufficient to buy 100 shares of {code} @ {price:.2f}")
            return None

        gross_amount = volume * effective_price
        comm = max(self.config.min_commission, gross_amount * self.config.commission_rate)
        transfer = gross_amount * self.config.transfer_fee_rate
        total_cost = gross_amount + comm + transfer

        if total_cost > self.cash:
            # Scale down
            volume -= 100
            if volume < 100:
                logger.warning(f"Insufficient cash for {code}: needs {total_cost:.2f}, available {self.cash:.2f}")
                return None
            gross_amount = volume * effective_price
            comm = max(self.config.min_commission, gross_amount * self.config.commission_rate)
            transfer = gross_amount * self.config.transfer_fee_rate
            total_cost = gross_amount + comm + transfer

        if self.config.mode == "qmt" and self.xt_trader:
            order_id = self.xt_trader.order_stock(
                account=self.xt_account,
                stock_code=code,
                order_type=self.xtconstant.STOCK_BUY,
                order_volume=volume,
                price_type=self.xtconstant.FIX_PRICE,
                price=round(effective_price, 2),
                strategy_name="Super-V0.8",
                order_remark="Arena_Bridge_Buy"
            )
            logger.info(f"[QMT BUY SENT] {code} | Vol: {volume} | Price: {effective_price:.2f} | OrderID: {order_id}")
            return {"action": "BUY", "code": code, "volume": volume, "price": effective_price, "order_id": order_id}
        else:
            # Paper execution
            self.cash -= total_cost
            if code in self.positions:
                old = self.positions[code]
                new_vol = old.volume + volume
                new_avg = (old.volume * old.avg_price + gross_amount) / new_vol
                self.positions[code].volume = new_vol
                self.positions[code].avg_price = new_avg
                self.positions[code].current_price = price
            else:
                self.positions[code] = Position(
                    code=code,
                    volume=volume,
                    can_use_volume=0, # T+1 restriction
                    avg_price=effective_price,
                    current_price=price,
                    entry_date=date_str or datetime.now().strftime("%Y-%m-%d"),
                    stop_loss_price=stop_loss,
                )
            self._save_paper_state()
            self._append_ledger("BUY", code, volume, effective_price, comm + transfer, total_cost, date_str)
            logger.info(f"[PAPER BUY FILLED] {code} | Vol: {volume} | Price: {effective_price:.2f} | TotalCost: {total_cost:.2f} | Cash: {self.cash:.2f}")
            return {"action": "BUY", "code": code, "volume": volume, "price": effective_price, "cost": total_cost}

    def order_sell(self, code: str, price: float, volume: Optional[int] = None, reason: str = "SIGNAL", date_str: str = "") -> Optional[dict]:
        """Execute Sell Order with T+1 check and cost calculation."""
        if code not in self.positions:
            logger.warning(f"No position in {code} to sell.")
            return None

        pos = self.positions[code]
        max_sellable = pos.can_use_volume
        sell_vol = min(volume, max_sellable) if volume is not None else max_sellable
        if sell_vol <= 0:
            logger.warning(f"Cannot sell {code}: T+1 restriction or zero available (can_use: {pos.can_use_volume}, total: {pos.volume})")
            return None

        effective_price = price * (1 - self.config.slippage)
        gross_amount = sell_vol * effective_price
        comm = max(self.config.min_commission, gross_amount * self.config.commission_rate)
        stamp = gross_amount * self.config.stamp_duty_rate
        transfer = gross_amount * self.config.transfer_fee_rate
        net_proceeds = gross_amount - comm - stamp - transfer

        if self.config.mode == "qmt" and self.xt_trader:
            order_id = self.xt_trader.order_stock(
                account=self.xt_account,
                stock_code=code,
                order_type=self.xtconstant.STOCK_SELL,
                order_volume=sell_vol,
                price_type=self.xtconstant.FIX_PRICE,
                price=round(effective_price, 2),
                strategy_name="Super-V0.8",
                order_remark=f"Arena_Bridge_Sell_{reason}"
            )
            logger.info(f"[QMT SELL SENT] {code} | Vol: {sell_vol} | Price: {effective_price:.2f} | Reason: {reason}")
            return {"action": "SELL", "code": code, "volume": sell_vol, "price": effective_price, "order_id": order_id}
        else:
            # Paper execution
            self.cash += net_proceeds
            pos.volume -= sell_vol
            pos.can_use_volume -= sell_vol
            pnl = net_proceeds - (sell_vol * pos.avg_price)
            pnl_pct = pnl / (sell_vol * pos.avg_price) if pos.avg_price > 0 else 0.0

            if pos.volume <= 0:
                del self.positions[code]
            self._save_paper_state()
            self._append_ledger("SELL", code, sell_vol, effective_price, comm + stamp + transfer, net_proceeds, date_str, pnl, pnl_pct, reason)
            logger.info(f"[PAPER SELL FILLED] {code} | Vol: {sell_vol} | Price: {effective_price:.2f} | PnL: {pnl:+,.2f} ({pnl_pct:+.2%}) | Reason: {reason}")
            return {"action": "SELL", "code": code, "volume": sell_vol, "price": effective_price, "pnl": pnl, "pnl_pct": pnl_pct}

    def settle_daily_t1(self, current_prices: Dict[str, float]):
        """A-Share EOD settlement: unlock T+1 shares and mark-to-market."""
        for code, pos in self.positions.items():
            pos.can_use_volume = pos.volume  # Unlock T+1 for tomorrow
            if code in current_prices:
                pos.current_price = current_prices[code]
        self._save_paper_state()
        logger.info(f"EOD Settlement complete. Total positions marked-to-market: {len(self.positions)}")

    def _save_paper_state(self):
        data = {
            "cash": self.cash,
            "updated_at": datetime.now().isoformat(),
            "positions": {
                code: {
                    "code": p.code,
                    "volume": p.volume,
                    "can_use_volume": p.can_use_volume,
                    "avg_price": round(p.avg_price, 4),
                    "current_price": round(p.current_price, 4),
                    "entry_date": p.entry_date,
                    "stop_loss_price": round(p.stop_loss_price, 4),
                }
                for code, p in self.positions.items()
            }
        }
        self.positions_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _append_ledger(self, action, code, volume, price, fees, amount, date_str="", pnl=0.0, pnl_pct=0.0, reason=""):
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "trade_date": date_str or datetime.now().strftime("%Y-%m-%d"),
            "action": action,
            "code": code,
            "volume": volume,
            "price": round(price, 4),
            "fees": round(fees, 4),
            "amount": round(amount, 4),
            "pnl": round(pnl, 4),
            "pnl_pct": round(pnl_pct, 6),
            "reason": reason,
            "balance_cash": round(self.cash, 2),
        }
        df = pd.DataFrame([row])
        if self.ledger_file.exists():
            df.to_csv(self.ledger_file, mode="a", header=False, index=False, encoding="utf-8-sig")
        else:
            df.to_csv(self.ledger_file, mode="w", header=True, index=False, encoding="utf-8-sig")
