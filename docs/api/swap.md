# Swap Quotes

Swap quoting via Relay aggregator for EVM chains. Returns live quotes with transaction-ready payloads.

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
  - `token_in` / `token_out` accept symbols (e.g., "ETH", "USDC") or contract addresses.
  - `amount_in` must be greater than zero.
  - `slippage_bps` defaults to `50` (0.5%).
  - `wallet_address` is **required** for Relay quotes.
  - `chain` accepts chain names ("ethereum", "base", "arbitrum", "optimism", "polygon") or chain IDs.
- **Success response** (`200 OK`):
  ```json
  {
    "success": true,
    "from": "ETH",
    "to": "USDC",
    "amount_in": 1.0,
    "amount_out_est": 3450.23,
    "price_in_usd": 3450.23,
    "price_out_usd": 1.00,
    "fee_est": 10.35,
    "slippage_bps": 50,
    "route": {
      "kind": "relay",
      "request_id": "abc123...",
      "steps": [...],
      "tx_ready": true,
      "primary_tx": {...}
    },
    "sources": [{"name": "Relay", "url": "https://relay.link"}],
    "warnings": [],
    "wallet": {"address": "0x..."},
    "chain_id": 1,
    "quote_type": "swap"
  }
  ```
- **Errors**:
  - `400` for validation problems (e.g., negative amount, unsupported token/chain).
  - `500` when the Relay API is unavailable or returns an error.

## Supported Chains

| Chain | ID | Aliases |
|-------|-----|---------|
| Ethereum | 1 | `ethereum`, `eth`, `mainnet` |
| Base | 8453 | `base` |
| Arbitrum | 42161 | `arbitrum`, `arb` |
| Optimism | 10 | `optimism`, `op` |
| Polygon | 137 | `polygon`, `matic` |

> **Note**: Solana swaps are not currently supported via Relay. Use Jupiter directly for Solana tokens.

## Token Resolution

Tokens are resolved via the `TOKEN_REGISTRY` which includes common tokens per chain. Contract addresses are also accepted for tokens not in the registry.
