# Swap Quotes

A lightweight swap estimator that returns a mocked route and pricing context.

## POST `/swap/quote`

- **Body schema** (`application/json`):
  ```json
  {
    "token_in": "ETH",
    "token_out": "USDC",
    "amount_in": 1,
    "chain": "ethereum",
    "slippage_bps": 50,
    "wallet_address": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
  }
  ```
  - `token_in` / `token_out` accept symbols or addresses.
  - `amount_in` must be greater than zero.
  - `slippage_bps` defaults to `50` (0.5%).
  - `wallet_address` is optional but required if you plan to hand the quote off to Relay.
- **Success response** (`200 OK`):
  ```json
  {
    "success": true,
    "from_token": "ETH",
    "to_token": "USDC",
    "amount_in": 1.0,
    "amount_out_est": 3450.23,
    "price_in_usd": 3450.23,
    "price_out_usd": 1.00,
    "fee_est": 10.35,
    "slippage_bps": 50,
    "route": {...},
    "sources": ["mock"],
    "warnings": []
  }
  ```
- **Errors**:
  - `400` for validation problems (e.g., negative amount).
  - `500` when the underlying aggregator helper raises unexpectedly.

> The current implementation is a deterministic placeholder. Replace `quote_swap_simple` with a real aggregator integration for production use.
