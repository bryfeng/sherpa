# Chat API

Conversational endpoints that drive the agentic wallet assistant.

## POST `/chat`

Initiates or continues a conversation and returns a fully synthesized response.

- **Body schema** (`application/json`):
  ```json
  {
    "messages": [
      {"role": "user", "content": "Analyze wallet 0xd8d..."}
    ],
    "address": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
    "chain": "ethereum",
    "conversation_id": "optional-existing-id",
    "llm_provider": "optional-provider-alias",
    "llm_model": "optional-model-id"
  }
  ```
  - `messages` must include at least the latest user prompt. Prior assistant messages help preserve context.
  - `conversation_id` is optional; omit it to let the backend continue the most recent session for the wallet address.
  - `llm_provider` / `llm_model` override the defaults configured in `app.config.Settings`.
- **Success response** (`200 OK`):
  ```json
  {
    "reply": "🤖 Narrative reply with insights...",
    "panels": {"portfolio": {"total_value_usd": 12345.67}},
    "sources": [{"provider": "alchemy", "type": "blockchain_data"}],
    "conversation_id": "ethereum-abc12345",
    "llm_provider": "anthropic",
    "llm_model": "claude-sonnet-4-20250514"
  }
  ```
- **Failure responses**:
  - `500` when the chat pipeline fails; payload contains `{ "detail": "Chat processing failed: ..." }`.

## POST `/chat/stream`

Streams Server-Sent Events (SSE) for the same request schema as `/chat`.

- **Body schema**: identical to `/chat`.
- **Response**: `text/event-stream` where each event contains serialized JSON fragments. The stream terminates with `data: [DONE]`.
- **Failure responses**: `500` with `{ "detail": "Chat streaming failed: ..." }`.

Use the streaming endpoint when you want incremental tokens; otherwise `/chat` returns the full message in a single payload.
