# Relay Bridge API

Helpers for interacting with Relay's cross-chain bridge quoting API. All routes are prefixed with `/tools/relay`.

## POST `/tools/relay/quote`

Request a bridge quote.

- **Body schema** (`application/json`):
  ```json
  {
    "user": "0x50ac5CFcc81BB0872e85255D7079F8a529345D16",
    "originChainId": 1,
    "destinationChainId": 8453,
    "originCurrency": "0x0000000000000000000000000000000000000000",
    "destinationCurrency": "0x0000000000000000000000000000000000000000",
    "recipient": "0x50ac5CFcc81BB0872e85255D7079F8a529345D16",
    "tradeType": "EXACT_INPUT",
    "amount": "1000000000000000",
    "referrer": "sherpa.chat",
    "useExternalLiquidity": false,
    "useDepositAddress": false,
    "topupGas": false
  }
  ```
  Optional fields: `usePermit`, `slippageTolerance` (string).
- **Success response** (`200 OK`): `{ "success": true, "quote": { ...raw Relay response... } }`
- **Errors**:
  - Relay HTTP errors are forwarded with the provider's status code and message.
  - Other failures yield `500` with `{ "detail": "Failed to fetch Relay quote: ..." }`.

## GET `/tools/relay/requests/{request_id}/signature`

Fetch a signature for a previously created Relay request.

- **Path parameters**: `request_id` (string) from the quote response.
- **Success response** (`200 OK`): `{ "success": true, "signature": { ...relay payload... } }`
- **Errors**: Same pattern as `/quote` (provider status code or `500`).

> Relay responses are passed through largely untouched. Consult Relay's documentation for field-level semantics.
