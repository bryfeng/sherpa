"""Async client for Relay's public bridge API."""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

import httpx

from ..config import settings

logger = logging.getLogger(__name__)


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
        json_payload: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> httpx.Response:
        merged_headers = {**self._headers(), **(headers or {})}
        last_error: Optional[Exception] = None
        request_id = str(uuid.uuid4())[:8]  # Short ID for log correlation

        # Log outgoing request
        logger.info(
            "[Relay:%s] %s %s",
            request_id,
            method,
            path,
        )
        if json_payload:
            logger.debug(
                "[Relay:%s] Request payload: %s",
                request_id,
                json.dumps(json_payload, indent=2),
            )

        for index, base_url in enumerate(self.base_urls):
            try:
                async with httpx.AsyncClient(base_url=base_url, timeout=self.timeout_s) as client:
                    response = await client.request(
                        method, path, json=json_payload, headers=merged_headers, **kwargs
                    )

                    # Log response
                    logger.debug(
                        "[Relay:%s] Response %d: %s",
                        request_id,
                        response.status_code,
                        response.text[:500] if response.text else "(empty)",
                    )

                    response.raise_for_status()
                    return response
            except httpx.HTTPStatusError as exc:
                # Log the error with full context
                error_body = exc.response.text or exc.response.reason_phrase
                logger.error(
                    "[Relay:%s] HTTP %d error on %s %s: %s | Request: %s",
                    request_id,
                    exc.response.status_code,
                    method,
                    path,
                    error_body[:500],
                    json.dumps(json_payload) if json_payload else "(none)",
                )
                # Relay returns JSON error bodies with useful context; stop early unless we have another base URL to try.
                if exc.response.status_code in (404, 405) and index < len(self.base_urls) - 1:
                    last_error = exc
                    continue
                raise
            except httpx.RequestError as exc:
                logger.error(
                    "[Relay:%s] Request error on %s %s: %s",
                    request_id,
                    method,
                    path,
                    str(exc),
                )
                last_error = exc
                continue

        if last_error is not None:
            raise last_error
        raise RuntimeError("All Relay hosts failed without providing an error response")

    async def quote(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Request a bridge quote from Relay.

        `payload` should follow the schema documented at
        https://docs.relay.link/references/api/get-quote-v2

        Uses the official v2 API endpoint: https://api.relay.link/quote/v2
        """
        resp = await self._request("POST", "/quote/v2", json_payload=payload)
        return resp.json()

    async def get_request_signature(self, request_id: str) -> Dict[str, Any]:
        """Fetch the attestation/signature for a quote if needed."""

        path = f"/requests/{request_id}/signature/v2"
        resp = await self._request("GET", path)
        return resp.json()

    async def get_chains(self) -> List[Dict[str, Any]]:
        """Fetch all supported chains from Relay.

        Returns a list of chain objects with metadata including:
        - id: Chain ID (e.g., 1 for Ethereum, 57073 for Ink)
        - name: Internal name
        - displayName: Human-readable name
        - currency: Native token info (symbol, decimals, address)
        - httpRpcUrl, explorerUrl: Network endpoints
        - depositEnabled, disabled: Availability status
        """
        resp = await self._request("GET", "/chains")
        data = resp.json()
        return data.get("chains", [])
