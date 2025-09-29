"""Smoke test for the public Bungee quote + build endpoints.

Usage:
    python -m sherpa.tests.test_bungee_public

The script fetches a USDC bridge quote from Optimism -> Arbitrum and then
attempts to build a manual transaction using the returned quoteId.
The public endpoints do not require an API key, but they may respond with
HTTP 400 if the route expires quickly or manual build is unavailable for the
selected pair. The script prints both the quote summary and the build result
(or error message) so we can quickly verify the integration end-to-end.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict

import httpx


# Allow running via ``python -m sherpa.tests.test_bungee_public`` (repo root)
# or ``python -m tests.test_bungee_public`` (from the sherpa/ package).
if 'sherpa' not in sys.modules:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from sherpa.app.providers.bungee import BungeeProvider


QUOTE_PARAMS: Dict[str, str] = {
    "originChainId": "10",  # Optimism
    "destinationChainId": "42161",  # Arbitrum
    "userAddress": "0x50ac5cfcc81bb0872e85255d7079f8a529345d16",
    "receiverAddress": "0x50ac5cfcc81bb0872e85255d7079f8a529345d16",
    "inputToken": "0x0b2c639c533813f4aa9d7837caf62653d097ff85",  # USDC (OP)
    "outputToken": "0xaf88d065e77c8cc2239327c5edb3a432268e5831",  # USDC (ARB)
    "inputAmount": "1000000",  # 1 USDC (6 decimals)
    "enableManual": "true",
}


async def main() -> None:
    provider = BungeeProvider()

    print("👉 Requesting quote via /api/v1/bungee/quote...")
    quote = await provider.quote(QUOTE_PARAMS)
    success = quote.get("success")
    status_code = quote.get("statusCode")
    result = quote.get("result", {})
    print(f"   success={success}, statusCode={status_code}")

    manual_routes = result.get("manualRoutes") or []
    default_route = result.get("autoRoute") or {}
    route_details = manual_routes[0] if manual_routes else default_route

    quote_id = route_details.get("quoteId") or result.get("quoteId")
    route_hash = route_details.get("requestHash") or route_details.get("routeId")
    output = route_details.get("output", {})
    output_amount = output.get("amount")
    output_usd = output.get("valueInUsd")

    print("   quoteId=", quote_id)
    print("   routeRequestHash=", route_hash)
    print("   estimated output=", output_amount, "(valueUsd=", output_usd, ")")

    if not quote_id:
        print("❌ Quote missing quoteId; cannot build manual transaction.")
        return

    build_params = {"quoteId": quote_id}

    print("👉 Requesting manual build via /api/v1/bungee/build-tx...")
    try:
        build = await provider.build_tx(build_params)
        print("   build response statusCode=", build.get("statusCode"))
        print("   keys=", list(build.keys()))
    except httpx.HTTPStatusError as exc:
        print("❌ build-tx HTTP error:", exc.response.status_code, exc.response.text)
    except Exception as exc:  # pragma: no cover - smoke test logging only
        print("❌ build-tx unexpected error:", exc)


if __name__ == "__main__":
    asyncio.run(main())
