# LLM Provider Catalogue

Surface metadata about configured Large Language Model providers. All routes are prefixed with `/llm`.

## GET `/llm/providers`

- **Description**: Returns the available providers, their models, and which provider/model pair should be considered the default.
- **Response** (`200 OK`):
  ```json
  {
    "providers": [
      {
        "id": "anthropic",
        "display_name": "Anthropic Claude",
        "status": "available",
        "default_model": "claude-sonnet-4-20250514",
        "description": "Anthropic Claude API",
        "reason": null,
        "models": [
          {"id": "claude-sonnet-4-20250514", "label": "Claude Sonnet 4", "default": true}
        ]
      },
      {
        "id": "zai",
        "display_name": "Zeta AI",
        "status": "available",
        "default_model": "glm-4.6",
        "description": "Z AI Chat Completions API",
        "models": [{"id": "glm-4.6", "label": "Zeta GLM 4.6"}]
      }
    ],
    "default_provider": "anthropic",
    "default_model": "claude-sonnet-4-20250514",
    "fetched_at": "2024-06-01T12:34:56.789Z"
  }
  ```

This endpoint powers provider pickers in the client. It does not require authentication but should be cached client-side to reduce load.
