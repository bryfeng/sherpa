# Conversation Management

Helper endpoints to inspect and maintain server-side conversation state.

## GET `/conversations`

List the recent conversations associated with a wallet address.

- **Query parameters**:
  - `address` (required): Wallet address used when initiating chats.
- **Success response** (`200 OK`): array of items shaped like:
  ```json
  {
    "conversation_id": "ethereum-abc12345",
    "title": "Portfolio check-in",
    "last_activity": "2024-06-01T12:34:56.789Z",
    "message_count": 8,
    "archived": false
  }
  ```

## POST `/conversations`

Create a new conversation ID for a wallet.

- **Body**:
  ```json
  {
    "address": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
    "title": "Optional display title"
  }
  ```
- **Response** (`200 OK`):
  ```json
  {
    "conversation_id": "ethereum-xyz6789",
    "title": "Optional display title"
  }
  ```
- **Errors**: `500` if the context manager is disabled.

## PATCH `/conversations/{conversation_id}`

Update metadata on an existing conversation.

- **Body** (all fields optional):
  ```json
  {
    "title": "Updated title",
    "archived": true
  }
  ```
- **Success response** (`200 OK`): full `ConversationSummary` record after the update.
- **Errors**:
  - `404` if the conversation ID cannot be found.
  - `500` if the context manager is disabled.

## GET `/conversations/{conversation_id}`

Return a full conversation transcript. Intended for debugging and local tooling.

- **Response** (`200 OK`):
  ```json
  {
    "conversation_id": "ethereum-abc12345",
    "owner_address": "0xd8d...",
    "title": "Portfolio check-in",
    "archived": false,
    "created_at": "2024-05-31T09:12:45.000Z",
    "last_activity": "2024-06-01T12:34:56.789Z",
    "total_tokens": 2345,
    "message_count": 8,
    "messages": [
      {
        "id": "msg_123",
        "role": "user",
        "content": "What's my ETH exposure?",
        "timestamp": "2024-06-01T12:30:00Z"
      }
    ]
  }
  ```
- **Errors**: `404` if the conversation is unknown.

> ⚠️ The transcripts include model metadata and should not be exposed publicly without additional access control.
