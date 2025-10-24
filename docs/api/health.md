# Health Endpoints

## GET `/healthz`

Returns a quick heartbeat of upstream data providers and overall service status.

- **Query parameters**: none
- **Success response** (`200 OK`):
  ```json
  {
    "status": "healthy",
    "providers": {
      "alchemy": {"status": "healthy", "latency_ms": 120},
      "coingecko": {"status": "degraded", "error": "..."}
    },
    "available_providers": 1,
    "total_providers": 2
  }
  ```
- **Failure modes**:
  - Network or provider errors still return `200` with `status: "degraded"` so long as at least one provider responds.
  - If both providers fail, the API returns `200` with `status: "degraded"` and `available_providers: 0`.

Use this endpoint for readiness/liveness probes in deployment environments.
