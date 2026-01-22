# Repository Guidelines

## Project Structure & Module Organization
- `trade_optimizer_server.py`: Main server entry point and API routes.
- `analyzer.py`: Analytics and metrics calculations.
- `bybit_client.py`: Bybit REST/WebSocket client helpers.
- `dashboard/index.html`: Single-page dashboard UI served by the server.
- `requirements.txt`: Python dependencies.

## Build, Test, and Development Commands
- Install deps: `pip install -r requirements.txt`
- Run (file mode): `python trade_optimizer_server.py --data-root "/path/to/Resilient Maker"`
- Run (sync mode): `TRADE_OPTIMIZER_MODE=sync python trade_optimizer_server.py`
- Run (realtime mode): `TRADE_OPTIMIZER_MODE=realtime python trade_optimizer_server.py`
- Open UI: `http://127.0.0.1:8010/`

## Coding Style & Naming Conventions
- Python: 4-space indentation, PEP 8 naming (`snake_case` for functions/vars, `CamelCase` for classes).
- Keep I/O boundaries clear: API handlers in `trade_optimizer_server.py`, analytics in `analyzer.py`, Bybit API logic in `bybit_client.py`.
- HTML/JS/CSS for the UI should stay in `dashboard/index.html` unless splitting becomes necessary.

## Testing Guidelines
- No automated test suite is present. If adding tests, prefer `pytest` and locate them under `tests/` (e.g., `tests/test_analyzer.py`).
- Name tests as `test_<unit>.py` and functions `test_<behavior>()`.

## Commit & Pull Request Guidelines
- No commit message convention is available in this repo. Use clear, scoped messages (e.g., `feat: add skew endpoint` or `fix: handle empty metrics`).
- PRs should include: purpose, relevant mode(s) tested (file/sync/realtime), and screenshots if the dashboard UI changes.
- Note any required environment variables (`BYBIT_API_KEY`, `BYBIT_API_SECRET`, `TRADE_OPTIMIZER_MODE`).

## Configuration & Secrets
- API credentials are required for sync/realtime/bybit_api modes. Use environment variables only; do not hardcode secrets.
- Common env vars: `BYBIT_API_KEY`, `BYBIT_API_SECRET`, `BYBIT_API_URL`, `BYBIT_CATEGORY`, `TRADE_OPTIMIZER_DATA_ROOT`.
