from typing import List, Dict
import os
import httpx


async def fetch_markets(query: str = "", limit: int = 5) -> List[Dict]:
    """
    Fetch trending or searched Polymarket markets.
    Uses a configurable base URL if provided via POLYMARKET_BASE_URL; otherwise returns a small mock set
    to keep the endpoint functional in offline/dev environments.
    """
    base = os.getenv("POLYMARKET_BASE_URL", "")
    if base:
        url = f"{base.rstrip('/')}/markets"
        params = {"query": query, "limit": limit}
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
            return _normalize_markets(data)

    # Fallback mock
    sample = [
        {"id": "m1", "question": "ETH above $4k by EOY?", "yesPrice": 0.42, "noPrice": 0.58, "url": "https://polymarket.com/"},
        {"id": "m2", "question": "BTC reaches new ATH this year?", "yesPrice": 0.55, "noPrice": 0.45, "url": "https://polymarket.com/"},
    ]
    return sample[:limit]


def _normalize_markets(data) -> List[Dict]:
    markets: List[Dict] = []
    if isinstance(data, dict) and isinstance(data.get("markets"), list):
        items = data["markets"]
    elif isinstance(data, list):
        items = data
    else:
        items = []

    for m in items:
        if not isinstance(m, dict):
            continue
        markets.append({
            "id": str(m.get("id") or m.get("slug") or m.get("question") or "market"),
            "question": str(m.get("question") or m.get("title") or "Untitled"),
            "yesPrice": float(m.get("yesPrice") or m.get("yes") or 0.0),
            "noPrice": float(m.get("noPrice") or m.get("no") or 0.0),
            "url": str(m.get("url") or m.get("link") or "https://polymarket.com"),
        })
    return markets

