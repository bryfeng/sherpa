from typing import List, Tuple
import httpx

BASE = "https://api.llama.fi"


async def get_tvl_series(protocol: str = "uniswap", window: str = "7d") -> Tuple[List[int], List[float]]:
    """
    Fetch protocol info from DefiLlama and return a recent TVL time series.
    Uses /protocol/{protocol} which includes a 'tvl' array of {date, totalLiquidityUSD}.
    window: '7d' or '30d'
    """
    url = f"{BASE}/protocol/{protocol}"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()

    tvl_arr = data.get("tvl", [])
    # Normalize
    points = [
        (
            int(item.get("date", 0)) * 1000,  # ms
            float(item.get("totalLiquidityUSD", 0.0)),
        )
        for item in tvl_arr
        if isinstance(item, dict)
    ]
    if not points:
        return [], []

    # Sort by timestamp ascending
    points.sort(key=lambda x: x[0])

    # Window slicing
    n = 7 if window == "7d" else 30
    recent = points[-n:]

    timestamps = [p[0] for p in recent]
    values = [p[1] for p in recent]
    return timestamps, values


async def get_tvl_current(protocol: str = "uniswap") -> Tuple[int, float]:
    """
    Fetch the latest TVL point for a protocol from DefiLlama.
    Returns (timestamp_ms, tvl_usd).
    """
    url = f"{BASE}/protocol/{protocol}"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()

    tvl_arr = data.get("tvl", [])
    if not tvl_arr or not isinstance(tvl_arr, list):
        return 0, 0.0
    last = tvl_arr[-1] if isinstance(tvl_arr[-1], dict) else None
    if not last:
        return 0, 0.0
    ts_ms = int(last.get("date", 0)) * 1000
    tvl = float(last.get("totalLiquidityUSD", 0.0))
    return ts_ms, tvl
