# Perpetuals Simulator

Endpoints for the read-only perps copilot, exposed under `/perps`.

## POST `/perps/simulate`

Simulate a prospective perpetual futures position.

- **Body schema** (`application/json`):
  ```json
  {
    "symbol": "ETH-PERP",
    "side": "LONG",
    "notional_usd": 10000,
    "max_leverage": 3,
    "time_horizon_days": 3,
    "take_profit": 4100,
    "stop_loss": 3600,
    "session_id": "conv-123",
    "user_id": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
  }
  ```
  - Provide either `notional_usd` or `quantity`.
  - `side` accepts `LONG` or `SHORT`.
  - Optional fields such as `max_leverage`, `take_profit`, `stop_loss`, `entry_price` override defaults.
- **Success response** (`200 OK`):
  ```json
  {
    "entry_price": 3890.42,
    "est_funding_apr": 0.12,
    "liq_price": 3450.10,
    "fee_estimate": 18.23,
    "var_95": 420.55,
    "es_95": 580.12,
    "max_drawdown_est": 580.12,
    "position_size_suggestion": 7500.0,
    "policy_ok": true,
    "policy_violations": [],
    "notes": ["Simulation only – funding impact included"],
    "explainability": {...}
  }
  ```
- **Errors**:
  - `400` if validation fails (missing notional/quantity, invalid leverage, etc.).
  - `500` for unexpected simulator failures.

The simulator respects risk guardrails defined in `PolicyManager` and defaults to deterministic provider data when `FEATURE_FLAG_FAKE_PERPS` is enabled.
