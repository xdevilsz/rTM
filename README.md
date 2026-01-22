# Resilient Trade Manager v1

Resilient Trade Manager is a real-time Bybit monitoring and analytics dashboard with charting, advanced analytics, and optional trading controls. It runs locally and supports realtime streams, multi-symbol analysis, and premium chart overlays.

## Highlights

- Realtime + sync modes (REST + WebSocket)
- Advanced analytics (performance, risk, maker/taker, consistency)
- Multi-symbol history with filtering and export
- Positions + active orders with PnL tooltips
- Chart overlays (breakeven, liquidation, active orders, trades)
- Tick + 5s streaming charts (SSE from realtime/public trades)
- Trading controls (chart buy/sell, position controls)
- Smart TP/SL (chase limit + split stop logic)
- Bilingual UI (EN / 中文)
- About & donation section with Resilient Lab 505 links

## Quick Setup

1. Install dependencies:
   pip install -r requirements.txt

2. Add Bybit keys to .env (optional for API features):
   BYBIT_API_KEY=your_key
   BYBIT_API_SECRET=your_secret
   BYBIT_CATEGORY=linear

3. Run the server:
   python trade_optimizer_server.py

4. Open in browser:
   http://127.0.0.1:8010

## Notes

- Trading features require API keys with execution permissions.
- The app runs locally by default; no data is sent externally.
- Tick/5s charts require realtime fills or public trade stream.

---

Trade Optimizer Dashboard
=========================

Comprehensive trade analysis and monitoring dashboard with sync and realtime modes.
Provides advanced analytics, performance metrics, risk analysis, and live position tracking.

Features
--------
- **Sync Mode**: Static historical analysis of past trades via Bybit REST API
- **Realtime Mode**: Live monitoring via WebSocket for trades, positions, and orders
- **Advanced Analytics**: Performance metrics (Sharpe, Calmar, Sortino), risk metrics (VaR, CVaR), maker/taker breakdown
- **Position Tracking**: Real-time position monitoring and analysis
- **Trade Analysis**: Comprehensive statistics, hourly/daily breakdowns, symbol-level analysis

Quick start
-----------
1) Install dependencies:
   pip install -r requirements.txt

2) Run the server:
   python trade_optimizer_server.py --data-root "/path/to/Resilient Maker"

3) Open:
   http://127.0.0.1:8010/

Modes
-----
1. **File Mode** (default): Reads from local metrics files
   - Reads: `<data-root>/runtime/metrics.json`
   - For use with Resilient Maker or other products that write metrics files

2. **Sync Mode**: Historical data analysis
   - Fetches past trades from Bybit API
   - Performs comprehensive historical analysis
   - Usage: `TRADE_OPTIMIZER_MODE=sync python trade_optimizer_server.py`

3. **Realtime Mode**: Live WebSocket monitoring
   - Continuous connection to Bybit WebSocket
   - Real-time trade, position, and order updates
   - Live analytics and monitoring
   - Usage: `TRADE_OPTIMIZER_MODE=realtime python trade_optimizer_server.py`

4. **Bybit API Mode** (legacy): Simple API-based metrics
   - Basic REST API fetching
   - Usage: `TRADE_OPTIMIZER_MODE=bybit_api python trade_optimizer_server.py`

Environment Variables
---------------------
Required for sync/realtime/bybit_api modes:
- `BYBIT_API_KEY`: Your Bybit API key
- `BYBIT_API_SECRET`: Your Bybit API secret

Optional:
- `BYBIT_CATEGORY`: `linear` (default) | `spot` | `inverse`
- `BYBIT_API_URL`: `https://api.bybit.com` (default) or testnet URL
- `TRADE_OPTIMIZER_MODE`: `file` | `sync` | `realtime` | `bybit_api`
- `TRADE_OPTIMIZER_DATA_ROOT`: Path to data directory (alternative to --data-root)

Examples
--------

**Sync Mode** (Historical Analysis):
```bash
export BYBIT_API_KEY="your_key"
export BYBIT_API_SECRET="your_secret"
export TRADE_OPTIMIZER_MODE=sync
python trade_optimizer_server.py
```
Then access: http://127.0.0.1:8010/
- Use "Advanced Analysis" tab for comprehensive metrics
- Adjust time range via API: `/api/analysis?days=30&limit=1000`

**Realtime Mode** (Live Monitoring):
```bash
export BYBIT_API_KEY="your_key"
export BYBIT_API_SECRET="your_secret"
export TRADE_OPTIMIZER_MODE=realtime
python trade_optimizer_server.py
```
- WebSocket connects automatically
- Real-time updates in dashboard
- Live position tracking
- Continuous analytics

**File Mode** (Local Metrics):
```bash
python trade_optimizer_server.py --data-root "/path/to/Resilient Maker"
```
- Works with products that write `runtime/metrics.json`

API Endpoints
-------------
- `/api/status`: Server status and mode information
- `/api/metrics`: Trade metrics and fills (mode-dependent)
- `/api/analysis`: Comprehensive analysis (sync/realtime only)
  - Query params: `days` (default: 30), `limit` (default: 1000), `symbol` (optional)
- `/api/positions`: Current positions (API modes only)
- `/api/skew`: Skew metrics (file mode only)

Dashboard Features
------------------
1. **Overview Tab**: Trade consistency, history, filtering
2. **Advanced Analysis Tab**: 
   - Performance metrics (Win rate, Sharpe, Calmar, Sortino, Max Drawdown)
   - Risk metrics (VaR, CVaR, Volatility)
   - Maker/Taker breakdown
   - Symbol-level statistics
   - Hourly/Daily statistics
3. **Positions Tab**: Current position tracking
4. **Resilient Maker Tab**: Product-specific metrics (if available)

Advanced Features
-----------------
- **Performance Analysis**: Sharpe ratio, Calmar ratio, Sortino ratio, win rate, profit/loss ratio
- **Risk Analysis**: Value at Risk (VaR), Conditional VaR (CVaR), volatility metrics
- **Trade Breakdown**: Maker vs taker analysis, symbol-level stats, hourly/daily aggregations
- **Real-time Monitoring**: Live WebSocket connection with automatic reconnection
- **Historical Sync**: Batch historical data retrieval with configurable time ranges

Dependencies
------------
- pandas >= 1.5.0
- numpy >= 1.23.0
- websockets >= 11.0
- orjson >= 3.8.0

Notes
-----
- Sync mode fetches historical data via REST API (rate limits apply)
- Realtime mode maintains persistent WebSocket connection
- Analysis is performed on client-side from fetched data
- Dashboard stores trade history in browser localStorage
