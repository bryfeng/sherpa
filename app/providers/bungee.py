import httpx
from typing import Any, Dict, Optional
from ..config import settings


class BungeeProvider:
    """Thin client for Bungee (Socket) public API.

    Notes:
    - API base and key are optional; defaults to Socket public base.
    - For best results, set BUNGEE_API_KEY in env (if required by tier).
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, timeout_s: int = 20):
        self.api_key = api_key or getattr(settings, 'bungee_api_key', '')
        self.base_url = base_url or getattr(settings, 'bungee_base_url', 'https://api.socket.tech')
        self.timeout_s = timeout_s

    async def quote(self, params: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            'Accept': 'application/json',
        }
        if self.api_key:
            headers['API-KEY'] = self.api_key

        # Expected params (subset): fromChainId, toChainId, fromTokenAddress, toTokenAddress, amount, userAddress, slippage
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/v2/quote",
                headers=headers,
                params=params,
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            return resp.json()

