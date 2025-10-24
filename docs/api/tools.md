# Tools API

Utility endpoints under the `/tools` prefix supply portfolio data, DeFi metrics, and market signals.

## GET `/tools/portfolio`

Return enriched portfolio data for a wallet.

- **Query parameters**:
  - `address` (required): Checksummed wallet address (must be `0x` prefixed and 42 chars).
  - `chain` (optional, default `ethereum`): Supported chain identifier.
- **Success response** (`200 OK`):
  ```json
  {
    "success": true,
    "portfolio": {
      "address": "0xd8d...",
      "tokens": [
        {"symbol": "ETH", "balance": "12.34", "value_usd": 42000}
      ],
      "totals": {"value_usd": 42000}
    },
    "sources": [{"provider": "alchemy"}, {"provider": "coingecko"}]
  }
  ```
- **Errors**:
  - `400` for malformed addresses.
  - `500` if portfolio aggregation fails.

## GET `/tools/defillama/tvl`

Return a time series of Total Value Locked (TVL) for a protocol.

- **Query parameters**:
  - `protocol` (optional, default `uniswap`): Protocol slug on DefiLlama.
  - `range` (optional, default `7d`): Supported values `7d` or `30d`.
- **Success response** (`200 OK`): `{ "timestamps": [...], "tvl": [...], "source": "defillama" }`

## GET `/tools/defillama/current`

Return the latest TVL point for a protocol.

- **Query parameters**: `protocol` (default `uniswap`).
- **Success response** (`200 OK`): `{ "timestamp": 1717161600, "tvl": 123456789, "source": "defillama" }`

## GET `/tools/polymarket/markets`

Search or fetch trending Polymarket markets.

- **Query parameters**:
  - `query` (optional): Search string; empty string returns trending.
  - `limit` (optional, default `5`, min `1`, max `20`).
- **Success response** (`200 OK`): `{ "markets": [{ "id": "123", "question": "Will ETH>4k?", ... }] }`

## GET `/tools/prices/top`

Return top market-cap coins from CoinGecko.

- **Query parameters**:
  - `limit` (optional, default `5`, max `10`).
  - `exclude_stable` (optional, default `true`): Filter out stablecoins.
- **Success response** (`200 OK`): `{ "success": true, "coins": [...] }`
- **Errors**: `503` when the price provider is unavailable; `500` for unexpected failures.

## GET `/tools/prices/trending`

Return trending EVM-compatible tokens.

- **Query parameters**: `limit` (optional, default `10`, max `25`).
- **Success response** (`200 OK`): `{ "success": true, "tokens": [...] }`
- **Errors**: `500` with detail message when trending fetch fails.

> All tools endpoints are GET-only and return JSON. Ensure you handle rate limits from upstream providers in production deployments.
