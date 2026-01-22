"""
Comprehensive Trade Analyzer
Provides performance metrics, risk analysis, and statistical insights for trading data.
"""
import math
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np


class TradeAnalyzer:
    """Comprehensive trade analysis engine"""
    
    def __init__(self, fills: List[Dict]):
        """
        Initialize analyzer with trade fills.
        
        Args:
            fills: List of trade fill dictionaries with keys: ts, side, qty, price, notional, fee, exec_pnl, symbol
        """
        self.fills = sorted(fills, key=lambda x: x.get("ts", 0))
        self.df = self._build_dataframe()
        
    def _build_dataframe(self) -> pd.DataFrame:
        """Convert fills list to pandas DataFrame for analysis"""
        if not self.fills:
            return pd.DataFrame()
            
        data = []
        for f in self.fills:
            data.append({
                "timestamp": datetime.fromtimestamp(f.get("ts", 0)),
                "ts": f.get("ts", 0),
                "order_id": f.get("order_id", ""),
                "symbol": f.get("symbol", ""),
                "side": f.get("side", ""),
                "qty": float(f.get("qty", 0)),
                "price": float(f.get("price", 0)),
                "notional": float(f.get("notional", 0) or (f.get("qty", 0) * f.get("price", 0))),
                "fee": float(f.get("fee", 0)),
                "fee_ccy": f.get("fee_ccy", ""),
                "exec_pnl": float(f.get("exec_pnl", 0)),
                "is_maker": f.get("is_maker", None),
            })
        
        df = pd.DataFrame(data)
        if df.empty:
            return df
            
        df = df.sort_values("timestamp")
        df["cumulative_pnl"] = df["exec_pnl"].cumsum()
        df["cumulative_fee"] = df["fee"].cumsum()
        df["net_pnl"] = df["exec_pnl"] - df["fee"]
        df["cumulative_net_pnl"] = df["net_pnl"].cumsum()
        
        # Calculate drawdown
        df["running_max"] = df["cumulative_net_pnl"].expanding().max()
        df["drawdown"] = df["cumulative_net_pnl"] - df["running_max"]
        df["drawdown_pct"] = (df["drawdown"] / df["running_max"].replace(0, 1)) * 100
        
        return df
    
    def get_basic_statistics(self) -> Dict[str, Any]:
        """Get basic trading statistics"""
        if self.df.empty:
            return {}
        
        buy_trades = self.df[self.df["side"].str.contains("Buy|buy", case=False, na=False)]
        sell_trades = self.df[self.df["side"].str.contains("Sell|sell", case=False, na=False)]
        
        total_volume = self.df["notional"].sum()
        total_qty = self.df["qty"].sum()
        total_fee = self.df["fee"].sum()
        net_pnl = self.df["cumulative_net_pnl"].iloc[-1] if len(self.df) > 0 else 0
        realized_pnl = self.df["exec_pnl"].sum()
        
        time_span = (self.df["timestamp"].max() - self.df["timestamp"].min()).total_seconds() / 3600
        trade_frequency = len(self.df) / time_span if time_span > 0 else 0
        
        return {
            "total_trades": len(self.df),
            "buy_count": len(buy_trades),
            "sell_count": len(sell_trades),
            "buy_sell_ratio": len(buy_trades) / len(sell_trades) if len(sell_trades) > 0 else float("inf"),
            "total_volume": total_volume,
            "total_qty": total_qty,
            "total_fee": total_fee,
            "realized_pnl": realized_pnl,
            "net_pnl": net_pnl,
            "fee_rate": total_fee / total_volume if total_volume > 0 else 0,
            "profit_rate": net_pnl / total_volume if total_volume > 0 else 0,
            "trade_frequency_per_hour": trade_frequency,
            "avg_buy_price": float(buy_trades["price"].mean()) if len(buy_trades) > 0 else None,
            "avg_sell_price": float(sell_trades["price"].mean()) if len(sell_trades) > 0 else None,
            "symbols": list(self.df["symbol"].unique()),
        }
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate advanced performance metrics"""
        if self.df.empty or len(self.df) < 2:
            return {}
        
        returns = self.df["net_pnl"].values
        cumulative_returns = self.df["cumulative_net_pnl"].values
        
        # Win/loss stats
        win_trades = returns[returns > 0]
        loss_trades = returns[returns < 0]
        total_trades = len(returns)
        win_rate = len(win_trades) / total_trades if total_trades > 0 else 0
        avg_win = float(win_trades.mean()) if len(win_trades) > 0 else 0
        avg_loss = float(loss_trades.mean()) if len(loss_trades) > 0 else 0
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
        
        # Drawdown
        max_drawdown = float(self.df["drawdown"].min()) if not self.df["drawdown"].empty else 0
        max_drawdown_pct = float(self.df["drawdown_pct"].min()) if not self.df["drawdown_pct"].empty else 0
        
        # Sharpe Ratio (annualised, assuming daily frequency)
        if len(returns) > 1 and returns.std() != 0:
            # Approximate annualisation: sqrt(number of periods per year)
            periods_per_year = 365 * 24  # hourly trades
            sharpe = (returns.mean() / returns.std()) * math.sqrt(periods_per_year)
        else:
            sharpe = 0
        
        # Calmar Ratio
        total_return = cumulative_returns[-1] if len(cumulative_returns) > 0 else 0
        calmar = total_return / abs(max_drawdown) if max_drawdown != 0 else float("inf")
        
        # Sortino Ratio (only downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1 and downside_returns.std() != 0:
            sortino = (returns.mean() / downside_returns.std()) * math.sqrt(periods_per_year)
        else:
            sortino = 0
        
        return {
            "win_rate": win_rate,
            "profit_loss_ratio": profit_loss_ratio,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown_pct,
            "sharpe_ratio": sharpe,
            "calmar_ratio": calmar,
            "sortino_ratio": sortino,
            "total_return": total_return,
        }
    
    def calculate_hourly_stats(self) -> pd.DataFrame:
        """Calculate statistics grouped by hour"""
        if self.df.empty:
            return pd.DataFrame()
        
        self.df["hour"] = self.df["timestamp"].dt.floor("h")  # Use 'h' instead of deprecated 'H'
        hourly = self.df.groupby("hour").agg({
            "net_pnl": ["sum", "count", "mean"],
            "fee": "sum",
            "notional": "sum",
            "qty": "sum",
        }).reset_index()
        
        hourly.columns = ["hour", "hourly_pnl", "trade_count", "avg_pnl", "hourly_fee", "hourly_volume", "hourly_qty"]
        hourly = hourly.sort_values("hour")
        hourly["cumulative_pnl"] = hourly["hourly_pnl"].cumsum()
        # Convert hour column to string for JSON serialization
        hourly["hour"] = hourly["hour"].astype(str)
        
        return hourly
    
    def calculate_daily_stats(self) -> pd.DataFrame:
        """Calculate statistics grouped by day"""
        if self.df.empty:
            return pd.DataFrame()
        
        self.df["date"] = self.df["timestamp"].dt.date
        daily = self.df.groupby("date").agg({
            "net_pnl": ["sum", "count", "mean"],
            "fee": "sum",
            "notional": "sum",
            "qty": "sum",
        }).reset_index()
        
        daily.columns = ["date", "daily_pnl", "trade_count", "avg_pnl", "daily_fee", "daily_volume", "daily_qty"]
        daily = daily.sort_values("date")
        daily["cumulative_pnl"] = daily["daily_pnl"].cumsum()
        # Convert date column to string for JSON serialization
        daily["date"] = daily["date"].astype(str)
        
        return daily
    
    def calculate_symbol_stats(self) -> pd.DataFrame:
        """Calculate statistics grouped by symbol"""
        if self.df.empty:
            return pd.DataFrame()
        
        symbol_stats = self.df.groupby("symbol").agg({
            "net_pnl": ["sum", "count", "mean"],
            "fee": "sum",
            "notional": "sum",
            "qty": "sum",
        }).reset_index()
        
        symbol_stats.columns = ["symbol", "total_pnl", "trade_count", "avg_pnl", "total_fee", "total_volume", "total_qty"]
        symbol_stats = symbol_stats.sort_values("total_pnl", ascending=False)
        
        return symbol_stats
    
    def calculate_risk_metrics(self, confidence_level: float = 0.05) -> Dict[str, Any]:
        """Calculate risk metrics including VaR and CVaR"""
        if self.df.empty:
            return {}
        
        returns = self.df["net_pnl"].values
        
        if len(returns) < 10:
            return {}
        
        # Historical VaR and CVaR
        var_percentile = confidence_level * 100
        var_value = float(np.percentile(returns, var_percentile))
        
        # CVaR (Conditional VaR / Expected Shortfall)
        cvar_value = float(returns[returns <= var_value].mean()) if len(returns[returns <= var_value]) > 0 else 0
        
        # Volatility (annualised)
        volatility = float(np.std(returns)) * math.sqrt(365 * 24) if len(returns) > 1 else 0
        
        # Maximum daily loss
        daily_stats = self.calculate_daily_stats()
        max_daily_loss = float(daily_stats["daily_pnl"].min()) if not daily_stats.empty else 0
        
        return {
            f"var_{int((1-confidence_level)*100)}": var_value,
            f"cvar_{int((1-confidence_level)*100)}": cvar_value,
            "volatility_annualised": volatility,
            "max_daily_loss": max_daily_loss,
        }
    
    def analyze_maker_taker(self) -> Dict[str, Any]:
        """Analyze maker vs taker performance"""
        if self.df.empty:
            return {}
        
        maker_trades = self.df[self.df["is_maker"] == True]
        taker_trades = self.df[self.df["is_maker"] == False]
        
        return {
            "maker_count": len(maker_trades),
            "taker_count": len(taker_trades),
            "maker_pnl": float(maker_trades["net_pnl"].sum()) if not maker_trades.empty else 0,
            "taker_pnl": float(taker_trades["net_pnl"].sum()) if not taker_trades.empty else 0,
            "maker_fee": float(maker_trades["fee"].sum()) if not maker_trades.empty else 0,
            "taker_fee": float(taker_trades["fee"].sum()) if not taker_trades.empty else 0,
            "maker_volume": float(maker_trades["notional"].sum()) if not maker_trades.empty else 0,
            "taker_volume": float(taker_trades["notional"].sum()) if not taker_trades.empty else 0,
        }
    
    def get_consistency_metrics(self) -> Dict[str, Any]:
        """Calculate trade consistency metrics (from original implementation)"""
        if len(self.df) < 2:
            return {
                "score": 0,
                "avg_interval": 0,
                "std_interval": 0,
                "buy_count": 0,
                "sell_count": 0,
                "trades_per_hour": 0,
            }
        
        # Calculate intervals between trades
        timestamps = sorted(self.df["ts"].values)
        intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps)) if timestamps[i] > timestamps[i-1]]
        
        if not intervals:
            return {"score": 0, "avg_interval": 0, "std_interval": 0}
        
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        cv = std_interval / avg_interval if avg_interval > 0 else 1
        
        time_score = max(0, 100 - min(100, cv * 100))
        
        buy_count = len(self.df[self.df["side"].str.contains("Buy|buy", case=False, na=False)])
        sell_count = len(self.df[self.df["side"].str.contains("Sell|sell", case=False, na=False)])
        total = len(self.df)
        balance = abs(buy_count - sell_count) / total if total > 0 else 1
        balance_score = max(0, 100 - min(100, balance * 100))
        
        score = int(time_score * 0.7 + balance_score * 0.3)
        
        time_span = (self.df["timestamp"].max() - self.df["timestamp"].min()).total_seconds() / 3600
        trades_per_hour = total / time_span if time_span > 0 else total
        
        return {
            "score": score,
            "avg_interval": float(avg_interval),
            "std_interval": float(std_interval),
            "buy_count": buy_count,
            "sell_count": sell_count,
            "trades_per_hour": float(trades_per_hour),
        }
    
    def get_comprehensive_analysis(self) -> Dict[str, Any]:
        """Get all analysis results in one dictionary"""
        return {
            "basic": self.get_basic_statistics(),
            "performance": self.calculate_performance_metrics(),
            "risk": self.calculate_risk_metrics(),
            "maker_taker": self.analyze_maker_taker(),
            "consistency": self.get_consistency_metrics(),
            "hourly_stats": self.calculate_hourly_stats().to_dict("records") if not self.df.empty else [],
            "daily_stats": self.calculate_daily_stats().to_dict("records") if not self.df.empty else [],
            "symbol_stats": self.calculate_symbol_stats().to_dict("records") if not self.df.empty else [],
        }