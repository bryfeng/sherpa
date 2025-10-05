"""Async client for Relay's public bridge API."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx

from ..config import settings


class RelayProvider:
    """Thin wrapper around https://api.relay.link endpoints."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        timeout_s: int = 20,
    ) -> None:
        configured = (
            base_url
            or getattr(settings, "relay_base_url", "")
            or os.environ.get("RELAY_BASE_URL", "")
        )
        if configured:
            self.base_urls: List[str] = [configured.rstrip("/")]
        else:
            self.base_urls = ["https://api.relay.link"]
        self.timeout_s = timeout_s

    def _headers(self) -> Dict[str, str]:
        return {
            "accept": "application/json, text/plain, */*",
            "content-type": "application/json",
            "user-agent": "SherpaRelayClient/2025-10",
            "origin": "https://relay.link",
        }

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> httpx.Response:
        merged_headers = {**self._headers(), **(headers or {})}
        last_error: Optional[Exception] = None

        for index, base_url in enumerate(self.base_urls):
            try:
                async with httpx.AsyncClient(base_url=base_url, timeout=self.timeout_s) as client:
                    response = await client.request(method, path, json=json, headers=merged_headers, **kwargs)
                    response.raise_for_status()
                    return response
            except httpx.HTTPStatusError as exc:
                # Relay returns JSON error bodies with useful context; stop early unless we have another base URL to try.
                if exc.response.status_code in (404, 405) and index < len(self.base_urls) - 1:
                    last_error = exc
                    continue
                raise
            except httpx.RequestError as exc:
                last_error = exc
                continue

        if last_error is not None:
            raise last_error
        raise RuntimeError("All Relay hosts failed without providing an error response")

    async def quote(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Request a bridge quote from Relay.

        `payload` should follow the schema documented at
        https://docs.relay.link/ (e.g. originChainId, destinationChainId, amount, etc.).
        """

        resp = await self._request("POST", "/quote", json=payload)
        return resp.json()

    async def get_request_signature(self, request_id: str) -> Dict[str, Any]:
        """Fetch the attestation/signature for a quote if needed."""

        path = f"/requests/{request_id}/signature/v2"
        resp = await self._request("GET", path)
        return resp.json()
