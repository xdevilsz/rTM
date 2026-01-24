"""
Bybit API Client for Trade Optimizer
Handles REST API calls and WebSocket connections for sync and realtime modes.
"""
import asyncio
import hashlib
import hmac
import json
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from typing import Dict, List, Optional, Callable, Any
import websockets
import orjson


class BybitAPIClient:
    """Bybit REST API client"""
    
    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://api.bybit.com", category: str = "linear"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self.category = category
        self.last_error: Optional[str] = None
    
    def _signed_request(self, path: str, params: Dict[str, str]) -> Optional[Dict]:
        """Make a signed GET request to Bybit API"""
        query = urllib.parse.urlencode(sorted(params.items()))
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        payload = f"{timestamp}{self.api_key}{recv_window}{query}"
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        
        url = f"{self.base_url}{path}?{query}"
        req = urllib.request.Request(url, method="GET")
        req.add_header("X-BAPI-API-KEY", self.api_key)
        req.add_header("X-BAPI-SIGN", signature)
        req.add_header("X-BAPI-TIMESTAMP", timestamp)
        req.add_header("X-BAPI-RECV-WINDOW", recv_window)
        
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if str(data.get("retCode")) != "0":
                self.last_error = f"Bybit API error {data.get('retCode')}: {data.get('retMsg')}"
                return None
            self.last_error = None
            return data
        except Exception as e:
            self.last_error = f"Bybit API request failed: {e}"
            return None

    def _signed_post(
        self,
        path: str,
        payload_params: Dict[str, Any],
        extra_headers: Optional[Dict[str, str]] = None
    ) -> Optional[Dict]:
        """Make a signed POST request to Bybit API"""
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        body = json.dumps(payload_params, separators=(",", ":"))
        payload = f"{timestamp}{self.api_key}{recv_window}{body}"
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url, method="POST", data=body.encode("utf-8"))
        req.add_header("Content-Type", "application/json")
        req.add_header("X-BAPI-API-KEY", self.api_key)
        req.add_header("X-BAPI-SIGN", signature)
        req.add_header("X-BAPI-TIMESTAMP", timestamp)
        req.add_header("X-BAPI-RECV-WINDOW", recv_window)
        for key, value in (extra_headers or {}).items():
            req.add_header(key, value)

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if str(data.get("retCode")) != "0":
                self.last_error = f"Bybit API error {data.get('retCode')}: {data.get('retMsg')}"
                return None
            self.last_error = None
            return data
        except Exception as e:
            self.last_error = f"Bybit API request failed: {e}"
            return None

    def _public_request(self, path: str, params: Dict[str, str]) -> Optional[Dict]:
        query = urllib.parse.urlencode(sorted(params.items()))
        url = f"{self.base_url}{path}?{query}"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if str(data.get("retCode")) != "0":
                self.last_error = f"Bybit API error {data.get('retCode')}: {data.get('retMsg')}"
                return None
            self.last_error = None
            return data
        except Exception as e:
            self.last_error = f"Bybit API request failed: {e}"
            return None
    
    def fetch_executions(self, limit: int = 200, symbol: Optional[str] = None, start_time: Optional[int] = None, end_time: Optional[int] = None) -> List[Dict]:
        """Fetch trade executions (fills)"""
        params = {
            "category": self.category,
            "limit": str(limit),
        }
        if symbol:
            params["symbol"] = symbol
        if start_time:
            params["startTime"] = str(start_time)
        if end_time:
            params["endTime"] = str(end_time)
        
        data = self._signed_request("/v5/execution/list", params)
        if not data:
            return []
        
        result = data.get("result") or {}
        rows = result.get("list") or []
        
        fills = []
        for r in rows:
            exec_type = (r.get("execType") or "").lower()
            ts_ms = int(r.get("execTime", 0))
            qty = float(r.get("execQty", 0))
            price = float(r.get("execPrice", 0))
            notional = float(r.get("execValue", 0)) or (qty * price if qty and price else 0)
            
            fills.append({
                "exec_type": exec_type,
                "ts": ts_ms / 1000 if ts_ms else 0,
                "order_id": r.get("orderId") or "",
                "side": r.get("side") or "",
                "qty": qty,
                "price": price,
                "notional": notional,
                "fee": float(r.get("execFee", 0)),
                "fee_ccy": r.get("feeCurrency") or "",
                "symbol": r.get("symbol") or "",
                "exec_pnl": float(r.get("execPnl", 0)),
                "is_maker": bool(r.get("isMaker")) if r.get("isMaker") is not None else None,
            })
        
        return fills

    def fetch_executions_page(
        self,
        limit: int = 200,
        symbol: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        cursor: Optional[str] = None
    ) -> tuple[List[Dict], Optional[str]]:
        """Fetch a single page of executions and return (fills, next_cursor)."""
        params = {
            "category": self.category,
            "limit": str(limit),
        }
        if symbol:
            params["symbol"] = symbol
        if start_time:
            params["startTime"] = str(start_time)
        if end_time:
            params["endTime"] = str(end_time)
        if cursor:
            params["cursor"] = cursor

        data = self._signed_request("/v5/execution/list", params)
        if not data:
            return [], None

        result = data.get("result") or {}
        rows = result.get("list") or []
        next_cursor = result.get("nextPageCursor") or None

        fills = []
        for r in rows:
            exec_type = (r.get("execType") or "").lower()
            ts_ms = int(r.get("execTime", 0))
            qty = float(r.get("execQty", 0))
            price = float(r.get("execPrice", 0))
            notional = float(r.get("execValue", 0)) or (qty * price if qty and price else 0)

            fills.append({
                "exec_type": exec_type,
                "ts": ts_ms / 1000 if ts_ms else 0,
                "order_id": r.get("orderId") or "",
                "side": r.get("side") or "",
                "qty": qty,
                "price": price,
                "notional": notional,
                "fee": float(r.get("execFee", 0)),
                "fee_ccy": r.get("feeCurrency") or "",
                "symbol": r.get("symbol") or "",
                "exec_pnl": float(r.get("execPnl", 0)),
                "is_maker": bool(r.get("isMaker")) if r.get("isMaker") is not None else None,
            })

        return fills, next_cursor

    def fetch_closed_pnl_page(
        self,
        limit: int = 200,
        symbol: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        cursor: Optional[str] = None
    ) -> tuple[List[Dict], Optional[str]]:
        """Fetch a single page of closed PnL and return (rows, next_cursor)."""
        params = {
            "category": self.category,
            "limit": str(limit),
        }
        if symbol:
            params["symbol"] = symbol
        if start_time:
            params["startTime"] = str(start_time)
        if end_time:
            params["endTime"] = str(end_time)
        if cursor:
            params["cursor"] = cursor

        data = self._signed_request("/v5/position/closed-pnl", params)
        if not data:
            return [], None

        result = data.get("result") or {}
        rows = result.get("list") or []
        next_cursor = result.get("nextPageCursor") or None

        normalized = []
        for r in rows:
            normalized.append({
                "order_id": r.get("orderId") or "",
                "symbol": r.get("symbol") or "",
                "side": r.get("side") or "",
                "qty": float(r.get("qty", 0) or 0),
                "closed_pnl": float(r.get("closedPnl", 0) or 0),
                "created_time": int(r.get("createdTime", 0) or 0),
                "updated_time": int(r.get("updatedTime", 0) or 0),
            })

        return normalized, next_cursor

    def fetch_order_history_page(
        self,
        limit: int = 200,
        symbol: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        cursor: Optional[str] = None
    ) -> tuple[List[Dict], Optional[str]]:
        """Fetch a single page of order history and return (rows, next_cursor)."""
        params = {
            "category": self.category,
            "limit": str(limit),
        }
        if symbol:
            params["symbol"] = symbol
        if start_time:
            params["startTime"] = str(start_time)
        if end_time:
            params["endTime"] = str(end_time)
        if cursor:
            params["cursor"] = cursor

        data = self._signed_request("/v5/order/history", params)
        if not data:
            return [], None

        result = data.get("result") or {}
        rows = result.get("list") or []
        next_cursor = result.get("nextPageCursor") or None
        return rows, next_cursor
    
    def fetch_positions(self, settle_coin: Optional[str] = None, symbol: Optional[str] = None) -> List[Dict]:
        """Fetch current positions"""
        params = {"category": self.category}
        if settle_coin:
            params["settleCoin"] = settle_coin
        if symbol:
            params["symbol"] = symbol
        data = self._signed_request("/v5/position/list", params)
        if not data:
            return []
        
        result = data.get("result") or {}
        return result.get("list") or []

    def fetch_positions_option(self, settle_coin: str = "USDT") -> List[Dict]:
        """Fetch option positions"""
        params = {"category": "option", "settleCoin": settle_coin}
        data = self._signed_request("/v5/position/list", params)
        if not data:
            return []
        result = data.get("result") or {}
        return result.get("list") or []

    def fetch_active_orders(
        self,
        symbol: Optional[str] = None,
        settle_coin: Optional[str] = None,
        base_coin: Optional[str] = None,
    ) -> List[Dict]:
        """Fetch active orders"""
        params: Dict[str, str] = {"category": self.category}
        if symbol:
            params["symbol"] = symbol
        if settle_coin:
            params["settleCoin"] = settle_coin
        if base_coin:
            params["baseCoin"] = base_coin
        data = self._signed_request("/v5/order/realtime", params)
        if not data:
            return []
        result = data.get("result") or {}
        return result.get("list") or []

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an active order by order ID"""
        payload = {
            "category": self.category,
            "symbol": symbol,
            "orderId": order_id,
        }
        data = self._signed_post("/v5/order/cancel", payload)
        return bool(data)

    def create_order(self, payload: Dict[str, Any], referral: Optional[str] = None) -> Optional[Dict]:
        extra_headers = {"referral": referral} if referral else None
        return self._signed_post("/v5/order/create", payload, extra_headers=extra_headers)

    def set_trading_stop(
        self,
        symbol: str,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None
    ) -> Optional[Dict]:
        payload: Dict[str, Any] = {
            "category": self.category,
            "symbol": symbol,
            "tpslMode": "Full",
        }
        if take_profit is not None:
            payload["takeProfit"] = str(take_profit)
        if stop_loss is not None:
            payload["stopLoss"] = str(stop_loss)
        return self._signed_post("/v5/position/trading-stop", payload)

    def fetch_orderbook(self, symbol: str, limit: int = 1) -> Optional[Dict]:
        params = {"category": self.category, "symbol": symbol, "limit": str(limit)}
        return self._public_request("/v5/market/orderbook", params)
    
    def fetch_wallet_balance(self) -> Optional[Dict]:
        """Fetch wallet balance"""
        params = {"accountType": "UNIFIED"}
        data = self._signed_request("/v5/account/wallet-balance", params)
        return data.get("result") if data else None


class BybitWebSocketClient:
    """Bybit WebSocket client for real-time data"""
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        category: str = "linear",
        testnet: bool = False
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.category = category
        self.ws_url = "wss://stream-testnet.bybit.com/v5/private" if testnet else "wss://stream.bybit.com/v5/private"
        
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
    
    def _generate_signature(self, expires: int) -> str:
        """Generate WebSocket authentication signature"""
        payload = f"GET/realtime{expires}"
        return hmac.new(
            self.api_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
    
    def on_execution(self, callback: Callable[[Dict], None]):
        """Register callback for execution (fill) events"""
        self._callbacks["execution"].append(callback)
    
    def on_order(self, callback: Callable[[Dict], None]):
        """Register callback for order events"""
        self._callbacks["order"].append(callback)
    
    def on_position(self, callback: Callable[[Dict], None]):
        """Register callback for position events"""
        self._callbacks["position"].append(callback)
    
    async def connect(self):
        """Connect to WebSocket and authenticate"""
        try:
            self._ws = await websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20)
            
            # Authenticate
            expires = int(time.time() * 1000) + 10000
            signature = self._generate_signature(expires)
            
            auth_msg = {
                "op": "auth",
                "args": [self.api_key, expires, signature]
            }
            await self._ws.send(orjson.dumps(auth_msg).decode())
            
            # Wait for auth response
            auth_resp_raw = await asyncio.wait_for(self._ws.recv(), timeout=5.0)
            auth_resp = orjson.loads(auth_resp_raw)
            
            if auth_resp.get("success") is not True and auth_resp.get("retCode") != 0:
                raise RuntimeError(f"WebSocket auth failed: {auth_resp}")
            
            # Subscribe to topics (execution.fast for lower latency, execution for completeness)
            subscribe_msg = {
                "op": "subscribe",
                "args": [
                    "execution",
                    "execution.fast",
                    "order",
                    "position",
                ]
            }
            await self._ws.send(orjson.dumps(subscribe_msg).decode())
            
            self._running = True
            
        except Exception as e:
            self._running = False
            raise
    
    async def listen(self):
        """Listen for incoming messages"""
        if not self._ws:
            raise RuntimeError("Not connected. Call connect() first.")
        
        try:
            async for raw in self._ws:
                try:
                    msg = orjson.loads(raw)
                except Exception:
                    continue
                
                topic = msg.get("topic", "")
                data = msg.get("data", [])
                
                if not topic or not data:
                    continue
                
                # Handle execution (fills)
                if "execution" in topic:
                    for cb in self._callbacks["execution"]:
                        try:
                            for item in data:
                                cb(item)
                        except Exception as e:
                            print(f"Error in execution callback: {e}")
                
                # Handle orders
                elif "order" in topic:
                    for cb in self._callbacks["order"]:
                        try:
                            for item in data:
                                cb(item)
                        except Exception as e:
                            print(f"Error in order callback: {e}")
                
                # Handle positions
                elif "position" in topic:
                    for cb in self._callbacks["position"]:
                        try:
                            for item in data:
                                cb(item)
                        except Exception as e:
                            print(f"Error in position callback: {e}")
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"WebSocket listen error: {e}")
            raise
    
    async def run(self):
        """Run the WebSocket client (connect + listen)"""
        backoff = 1.0
        while True:
            try:
                if not self._running:
                    await self.connect()
                    backoff = 1.0
                
                await self.listen()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"WebSocket error: {e}. Reconnecting in {backoff:.1f}s...")
                self._running = False
                if self._ws:
                    try:
                        await self._ws.close()
                    except:
                        pass
                    self._ws = None
                
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.7, 30.0)
    
    async def close(self):
        """Close the WebSocket connection"""
        self._running = False
        if self._ws:
            try:
                await self._ws.close()
            except:
                pass
            self._ws = None
