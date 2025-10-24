# Entitlement API

Evaluate whether a wallet currently has access to Pro features.

## GET `/entitlement`

- **Query parameters**:
  - `address` (required): Wallet address to evaluate.
  - `chain` (optional): Override chain (defaults to configured chain when omitted).
- **Success response** (`200 OK`):
  ```json
  {
    "address": "0xd8d...",
    "chain": "ethereum",
    "pro": true,
    "gating": "token",
    "standard": "erc20",
    "token_address": "0x123...",
    "token_id": null,
    "reason": null,
    "checked_at": "2024-06-01T12:34:56.789Z",
    "cached": false,
    "metadata": {}
  }
  ```
- **Errors**:
  - `400` when entitlement evaluation fails (e.g., invalid address or provider error). The `detail` field contains the reason.

Use this endpoint from the frontend to determine whether gated UI functionality should be unlocked.
