import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

from bybit_client import BybitAPIClient


@dataclass
class OrderResult:
    ok: bool
    order_id: str = ""
    error: str = ""


class TradingService:
    def __init__(self, client: BybitAPIClient, broker_id: str, category: str):
        self.client = client
        self.broker_id = broker_id
        self.category = category

    def _best_bid_ask(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        data = self.client.fetch_orderbook(symbol, limit=1)
        if not data:
            return None, None
        result = data.get("result") or {}
        bids = result.get("b") or []
        asks = result.get("a") or []
        bid = float(bids[0][0]) if bids else None
        ask = float(asks[0][0]) if asks else None
        return bid, ask

    def _create_order(self, payload: Dict[str, Any]) -> OrderResult:
        data = self.client.create_order(payload, referral=self.broker_id)
        if not data:
            return OrderResult(ok=False, error=self.client.last_error or "Order create failed")
        result = data.get("result") or {}
        order_id = result.get("orderId") or ""
        return OrderResult(ok=True, order_id=order_id)

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str,
        price: Optional[float] = None,
        reduce_only: bool = False,
        post_only: bool = False,
    ) -> OrderResult:
        payload: Dict[str, Any] = {
            "category": self.category,
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(qty),
        }
        if order_type == "Limit":
            if price is None:
                return OrderResult(ok=False, error="Limit price required")
            payload["price"] = str(price)
            if post_only:
                payload["timeInForce"] = "PostOnly"
        if reduce_only:
            payload["reduceOnly"] = True
        return self._create_order(payload)

    def place_best_bid_offer(
        self,
        symbol: str,
        side: str,
        qty: float,
        reduce_only: bool = False,
    ) -> OrderResult:
        bid, ask = self._best_bid_ask(symbol)
        if bid is None or ask is None:
            return OrderResult(ok=False, error="Orderbook unavailable")
        price = bid if side.lower().startswith("b") else ask
        return self.place_order(
            symbol=symbol,
            side=side,
            qty=qty,
            order_type="Limit",
            price=price,
            reduce_only=reduce_only,
            post_only=True,
        )

    def chase_limit(
        self,
        symbol: str,
        side: str,
        qty: float,
        reduce_only: bool = False,
        attempts: int = 4,
        delay: float = 0.5,
    ) -> OrderResult:
        last_order_id = ""
        last_error = ""
        filled_total = 0.0
        for _ in range(attempts):
            if filled_total >= qty:
                return OrderResult(ok=True, order_id=last_order_id)
            bid, ask = self._best_bid_ask(symbol)
            if bid is None or ask is None:
                last_error = "Orderbook unavailable"
                time.sleep(delay)
                continue
            price = bid if side.lower().startswith("b") else ask
            if last_order_id:
                filled_total = self._update_filled_from_active(symbol, last_order_id, filled_total)
                if filled_total >= qty:
                    return OrderResult(ok=True, order_id=last_order_id)
                self.client.cancel_order(order_id=last_order_id, symbol=symbol)
                if not self._wait_for_cancel(symbol, last_order_id):
                    last_error = "Cancel not confirmed"
                    time.sleep(delay)
                    continue
            remaining = max(qty - filled_total, 0.0)
            if remaining <= 0:
                return OrderResult(ok=True, order_id=last_order_id)
            result = self.place_order(
                symbol=symbol,
                side=side,
                qty=remaining,
                order_type="Limit",
                price=price,
                reduce_only=reduce_only,
                post_only=True,
            )
            if result.ok:
                last_order_id = result.order_id
                last_error = ""
            else:
                last_error = result.error
            time.sleep(delay)
        if last_order_id:
            return OrderResult(ok=True, order_id=last_order_id)
        return OrderResult(ok=False, error=last_error or "Chase limit failed")

    def _wait_for_cancel(self, symbol: str, order_id: str, timeout: float = 2.0) -> bool:
        if not order_id:
            return True
        start = time.time()
        while time.time() - start < timeout:
            orders = self.client.fetch_active_orders(symbol=symbol)
            active = False
            for order in orders:
                oid = order.get("orderId") or order.get("orderID")
                if oid == order_id:
                    active = True
                    break
            if not active:
                return True
            time.sleep(0.2)
        return False

    def _update_filled_from_active(self, symbol: str, order_id: str, filled_total: float) -> float:
        orders = self.client.fetch_active_orders(symbol=symbol)
        for order in orders:
            oid = order.get("orderId") or order.get("orderID")
            if oid != order_id:
                continue
            try:
                cum = float(order.get("cumExecQty") or order.get("cumExecQty") or 0)
            except (TypeError, ValueError):
                cum = 0.0
            if cum > filled_total:
                return cum
        return filled_total

    def close_position_market(self, symbol: str, side: str, qty: float) -> OrderResult:
        close_side = "Sell" if side.lower().startswith("b") else "Buy"
        return self.place_order(
            symbol=symbol,
            side=close_side,
            qty=qty,
            order_type="Market",
            reduce_only=True,
        )

    def set_tpsl(self, symbol: str, take_profit: Optional[float], stop_loss: Optional[float]) -> OrderResult:
        data = self.client.set_trading_stop(symbol, take_profit=take_profit, stop_loss=stop_loss)
        if not data:
            return OrderResult(ok=False, error=self.client.last_error or "TP/SL update failed")
        return OrderResult(ok=True)

    def smart_tp(self, symbol: str, side: str, qty: float) -> OrderResult:
        close_side = "Sell" if side.lower().startswith("b") else "Buy"
        return self.chase_limit(symbol, close_side, qty, reduce_only=True)

    def smart_sl(self, symbol: str, side: str, qty: float, break_even: float, mark_price: float) -> OrderResult:
        if qty <= 0:
            return OrderResult(ok=False, error="Position size invalid")
        close_side = "Sell" if side.lower().startswith("b") else "Buy"
        first_qty = qty / 2.0
        second_qty = qty - first_qty
        new_break_even = (2 * break_even) - mark_price
        first = self.chase_limit(symbol, close_side, first_qty, reduce_only=True)
        if not first.ok:
            return first
        second = self.place_order(
            symbol=symbol,
            side=close_side,
            qty=second_qty,
            order_type="Limit",
            price=new_break_even,
            reduce_only=True,
            post_only=True,
        )
        if not second.ok:
            return second
        return OrderResult(ok=True)
