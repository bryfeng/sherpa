import os
from typing import Any, Dict, List, Optional

import httpx

from ..config import settings


class BungeeProvider:
    """Thin client for the Bungee (Socket) public API surface."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, timeout_s: int = 20):
        # Try explicit args → settings → environment → defaults
        self.api_key = (
            api_key
            or getattr(settings, 'bungee_api_key', '')
            or os.environ.get('BUNGEE_API_KEY', '')
        )

        configured = (
            base_url
            or getattr(settings, 'bungee_base_url', '')
            or os.environ.get('BUNGEE_BASE_URL', '')
        )
        if configured:
            self.base_urls: List[str] = [configured.rstrip('/')]
        else:
            self.base_urls = [
                'https://public-backend.bungee.exchange',
                'https://api.socket.tech',
            ]

        self.timeout_s = timeout_s

    def _headers(self) -> Dict[str, str]:
        headers = {
            'Accept': 'application/json',
        }
        if self.api_key:
            headers['API-KEY'] = self.api_key
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> httpx.Response:
        merged_headers = {**self._headers(), **(headers or {})}
        last_error: Optional[Exception] = None

        for index, base_url in enumerate(self.base_urls):
            try:
                async with httpx.AsyncClient(base_url=base_url, timeout=self.timeout_s) as client:
                    resp = await client.request(method, path, headers=merged_headers, **kwargs)
                    resp.raise_for_status()
                    return resp
            except httpx.HTTPStatusError as exc:
                # Some public hosts omit certain routes. Fall back when we hit 404/405.
                if exc.response.status_code in (404, 405) and index < len(self.base_urls) - 1:
                    last_error = exc
                    continue
                raise
            except httpx.RequestError as exc:
                last_error = exc
                continue

        if last_error:
            raise last_error
        raise RuntimeError('All Bungee hosts failed without a specific error')

    async def quote(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch a bridge/auto-route quote via the public v1 API."""

        cleaned_params: Dict[str, Any] = {k: v for k, v in params.items() if v is not None}
        resp = await self._request('GET', '/api/v1/bungee/quote', params=cleaned_params)
        return resp.json()

    async def build_tx(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Build a manual transaction for a previously fetched quote (public v1 API)."""

        cleaned_params: Dict[str, Any] = {k: v for k, v in params.items() if v is not None}
        resp = await self._request('GET', '/api/v1/bungee/build-tx', params=cleaned_params)
        return resp.json()
