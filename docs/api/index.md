# Agentic Wallet API Reference

Welcome to the dedicated API reference for the Agentic Wallet backend. The service is built with FastAPI and exposes a mix of conversational, portfolio, tooling, and simulation endpoints.

- **Base URL (local development)**: `http://localhost:8000`
- **Documentation UIs**: Swagger UI is available at `/docs` and ReDoc at `/redoc` when the server is running.
- **Authentication**: The current MVP surface does not require API keys. Protect your deployment behind your own auth or gateway if exposed publicly.

## Available Endpoint Guides

| Area | Description |
| --- | --- |
| [Health](health.md) | Provider-level heartbeat information. |
| [Chat](chat.md) | Conversational analysis endpoints (standard and streaming). |
| [Conversations](conversations.md) | Conversation lifecycle management helpers. |
| [Tools](tools.md) | Portfolio, TVL, market, and price utilities under `/tools`. |
| [Relay](relay.md) | Relay bridge quoting helpers under `/tools/relay`. |
| [Swap](swap.md) | Simple swap quote estimator under `/swap`. |
| [Perps](perps.md) | Read-only perpetuals simulator under `/perps`. |
| [Entitlement](entitlement.md) | Wallet entitlement checks. |
| [LLM](llm.md) | LLM provider catalogue surfaced to the frontend. |

Each page documents request parameters, example payloads, and response structures pulled directly from the FastAPI routers.

## Error Handling

- Unless otherwise specified, validation issues return `400 Bad Request` with a descriptive `detail` message.
- Downstream provider failures return either the provider's status code or `503/500` with context in the `detail` field.
- All responses are JSON encoded.

## Versioning

The current API is tagged as `v0.1.0` in `app/main.py`. Breaking changes will bump the version and will be reflected in this reference.
