import math
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

import urllib.parse
import urllib.request


def parse_bybit_option_symbol(sym: str):
    """
    Parse Bybit option symbol, e.g. BTC-26JAN24-40000-C or BTC-26JAN24-40000-P
    Returns dict with underlying, expiry (datetime), strike (float), cp (C/P)
    """
    if not sym or "-" not in sym:
        return None
    parts = sym.split("-")
    if len(parts) < 4:
        return None
    und = parts[0].upper()
    expiry_raw = parts[1]
    strike_raw = parts[2]
    cp = parts[3].upper()
    try:
        expiry = datetime.strptime(expiry_raw, "%d%b%y").replace(tzinfo=timezone.utc)
    except Exception:
        return None
    try:
        strike = float(strike_raw)
    except Exception:
        return None
    return {"underlying": und, "expiry": expiry, "strike": strike, "cp": cp}


def _norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price_greeks(S: float, K: float, T: float, sigma: float, r: float, cp: str):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {"price": 0.0, "delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if cp == "C":
        price = S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
        delta = _norm_cdf(d1)
    else:
        price = K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
        delta = _norm_cdf(d1) - 1
    gamma = _norm_pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * _norm_pdf(d1) * math.sqrt(T)
    theta = - (S * _norm_pdf(d1) * sigma) / (2 * math.sqrt(T))
    return {"price": price, "delta": delta, "gamma": gamma, "theta": theta, "vega": vega}


def _public_get(base_url: str, path: str, params: dict) -> dict | None:
    query = urllib.parse.urlencode(sorted(params.items()))
    url = f"{base_url}{path}?{query}"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    if str(data.get("retCode")) != "0":
        return None
    return data


def _fetch_spot_price(base_url: str, symbol: str) -> Optional[float]:
    data = _public_get(base_url, "/v5/market/tickers", {"category": "linear", "symbol": symbol})
    if not data:
        return None
    rows = (data.get("result") or {}).get("list") or []
    if not rows:
        return None
    try:
        return float(rows[0].get("lastPrice") or rows[0].get("markPrice") or 0)
    except Exception:
        return None


def build_options_report(bybit_client, base_url: str, settle_coin: str, iv_btc: float, iv_eth: float, iv_shock: float) -> Dict:
    positions = bybit_client.fetch_positions_option(settle_coin=settle_coin)
    btc0 = _fetch_spot_price(base_url, "BTCUSDT")
    eth0 = _fetch_spot_price(base_url, "ETHUSDT")

    rows = []
    total_delta = 0.0
    total_gamma = 0.0
    total_theta = 0.0
    total_vega = 0.0
    total_pnl = 0.0

    for pos in positions:
        sym = pos.get("symbol") or ""
        info = parse_bybit_option_symbol(sym)
        if not info:
            continue
        size = float(pos.get("size") or 0)
        side = (pos.get("side") or "").lower()
        sign = 1.0 if side == "buy" else -1.0
        mark = float(pos.get("markPrice") or 0)
        if size == 0 or mark == 0:
            continue
        und = info["underlying"]
        S0 = btc0 if und == "BTC" else eth0 if und == "ETH" else None
        if not S0:
            continue
        sigma0 = iv_btc if und == "BTC" else iv_eth
        sigma1 = max(1e-6, sigma0 + iv_shock)
        T0 = max((info["expiry"] - datetime.now(timezone.utc)).total_seconds() / (365 * 24 * 3600), 1e-8)

        g0 = bs_price_greeks(S0, info["strike"], T0, sigma0, 0.0, info["cp"])
        g1 = bs_price_greeks(S0, info["strike"], T0, sigma1, 0.0, info["cp"])
        pnl = sign * size * (g1["price"] - g0["price"])
        total_pnl += pnl

        total_delta += sign * size * g0["delta"]
        total_gamma += sign * size * g0["gamma"]
        total_theta += sign * size * g0["theta"]
        total_vega += sign * size * g0["vega"]

        rows.append({
            "symbol": sym,
            "side": side,
            "size": size,
            "markPrice": mark,
            "delta": g0["delta"] * sign * size,
            "gamma": g0["gamma"] * sign * size,
            "theta": g0["theta"] * sign * size,
            "vega": g0["vega"] * sign * size,
            "iv_base": sigma0,
            "iv_shock": sigma1,
            "pnl_reprice": pnl,
        })

    return {
        "settle_coin": settle_coin,
        "spot": {"BTC": btc0, "ETH": eth0},
        "total": {
            "delta": total_delta,
            "gamma": total_gamma,
            "theta": total_theta,
            "vega": total_vega,
            "pnl_reprice": total_pnl,
        },
        "rows": rows,
    }
