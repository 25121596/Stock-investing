"""Calendar-aware ST limit adaptation; historical engine.py stays unchanged."""
from engine import Replay, number, tick


class Replay2026(Replay):
    execution_day = ""

    def buy(self, order, day, index):
        self.execution_day = day
        return super().buy(order, day, index)

    def sell(self, position, day, index):
        self.execution_day = day
        return super().sell(position, day, index)

    def tradable(self, bar, side):
        if (side == "sell" and bar is not None and number(bar.get("isST")) == 1
                and self.execution_day >= "2026-07-06"):
            if number(bar.get("tradestatus")) != 1 or number(bar.get("volume")) <= 0:
                return False, "suspended_or_missing"
            price, reference = number(bar.get("open")), number(bar.get("preclose"))
            if price <= 0 or reference <= 0:
                return False, "invalid_open"
            if price <= tick(reference * .90) + .0001:
                return False, "limit_down_open"
            return True, ""
        return super().tradable(bar, side)
