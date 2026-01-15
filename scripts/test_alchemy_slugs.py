#!/usr/bin/env python3
"""
Test Alchemy slugs for all chains in the Convex database.

This script:
1. Fetches all chains from Convex
2. Tests each chain's Alchemy endpoint
3. Updates the alchemyVerified field based on results

Usage:
    python scripts/test_alchemy_slugs.py [--update] [--chain CHAIN]

Options:
    --update    Actually update Convex with verification results (default: dry run)
    --chain     Test only a specific chain (by name or chain ID)
    --verbose   Show detailed output
"""

import argparse
import asyncio
import os
import sys
from typing import Any, Dict, List, Optional

import httpx

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.db.convex_client import ConvexClient, ConvexQueryError


class AlchemySlugTester:
    """Tests Alchemy API endpoints for various chains."""

    def __init__(self, api_key: str, verbose: bool = False):
        self.api_key = api_key
        self.verbose = verbose
        self.timeout = 10.0

    async def test_slug(self, slug: str) -> Dict[str, Any]:
        """
        Test if an Alchemy slug works by calling eth_chainId.

        Returns:
            {
                "success": bool,
                "chain_id": int | None,
                "latency_ms": int | None,
                "error": str | None
            }
        """
        url = f"https://{slug}.g.alchemy.com/v2/{self.api_key}"

        payload = {
            "jsonrpc": "2.0",
            "method": "eth_chainId",
            "params": [],
            "id": 1
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout
                )

                latency_ms = int(response.elapsed.total_seconds() * 1000)

                if response.status_code == 401:
                    return {
                        "success": False,
                        "chain_id": None,
                        "latency_ms": latency_ms,
                        "error": "Authentication failed - check API key"
                    }

                if response.status_code == 404:
                    return {
                        "success": False,
                        "chain_id": None,
                        "latency_ms": latency_ms,
                        "error": f"Endpoint not found - slug '{slug}' may be invalid"
                    }

                response.raise_for_status()
                data = response.json()

                if "error" in data:
                    return {
                        "success": False,
                        "chain_id": None,
                        "latency_ms": latency_ms,
                        "error": f"RPC error: {data['error']}"
                    }

                # Parse chain ID from hex
                chain_id_hex = data.get("result")
                chain_id = int(chain_id_hex, 16) if chain_id_hex else None

                return {
                    "success": True,
                    "chain_id": chain_id,
                    "latency_ms": latency_ms,
                    "error": None
                }

        except httpx.TimeoutException:
            return {
                "success": False,
                "chain_id": None,
                "latency_ms": None,
                "error": "Request timed out"
            }
        except httpx.HTTPStatusError as e:
            return {
                "success": False,
                "chain_id": None,
                "latency_ms": None,
                "error": f"HTTP {e.response.status_code}: {e.response.text[:100]}"
            }
        except Exception as e:
            return {
                "success": False,
                "chain_id": None,
                "latency_ms": None,
                "error": str(e)
            }


async def fetch_chains(convex: ConvexClient) -> List[Dict[str, Any]]:
    """Fetch all chains from Convex."""
    try:
        chains = await convex.query("chains:listAll", {})
        return chains or []
    except ConvexQueryError as e:
        print(f"Error fetching chains: {e}")
        return []


async def update_chain_verification(
    convex: ConvexClient,
    chain_id: int,
    alchemy_slug: Optional[str],
    verified: bool
) -> bool:
    """Update the alchemyVerified field in Convex."""
    try:
        await convex.mutation("chains:updateAlchemyStatus", {
            "chainId": chain_id,
            "alchemySlug": alchemy_slug,
            "alchemyVerified": verified
        })
        return True
    except Exception as e:
        print(f"  Error updating chain {chain_id}: {e}")
        return False


async def main():
    parser = argparse.ArgumentParser(description="Test Alchemy slugs for all chains")
    parser.add_argument("--update", action="store_true", help="Update Convex with results")
    parser.add_argument("--chain", type=str, help="Test only a specific chain")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Check for API key
    api_key = settings.alchemy_api_key
    if not api_key:
        print("Error: ALCHEMY_API_KEY not set in environment")
        sys.exit(1)

    # Initialize clients
    convex = ConvexClient()
    tester = AlchemySlugTester(api_key, verbose=args.verbose)

    print("=" * 60)
    print("Alchemy Slug Verification Tool")
    print("=" * 60)
    print(f"Mode: {'UPDATE' if args.update else 'DRY RUN (use --update to save results)'}")
    print()

    # Fetch chains
    print("Fetching chains from Convex...")
    chains = await fetch_chains(convex)

    if not chains:
        print("No chains found in database")
        sys.exit(1)

    print(f"Found {len(chains)} chains")
    print()

    # Filter if specific chain requested
    if args.chain:
        chain_filter = args.chain.lower()
        chains = [
            c for c in chains
            if (
                c["name"].lower() == chain_filter or
                str(c["chainId"]) == chain_filter or
                chain_filter in [a.lower() for a in c.get("aliases", [])]
            )
        ]
        if not chains:
            print(f"No chain found matching '{args.chain}'")
            sys.exit(1)

    # Test results
    results = {
        "verified": [],
        "failed": [],
        "no_slug": [],
        "already_verified": [],
        "chain_id_mismatch": []
    }

    # Test each chain
    for chain in chains:
        chain_id = chain["chainId"]
        name = chain["name"]
        slug = chain.get("alchemySlug")
        already_verified = chain.get("alchemyVerified", False)

        print(f"[{chain_id}] {name}")

        if not slug:
            print(f"  ⏭️  No Alchemy slug configured")
            results["no_slug"].append(chain)
            continue

        if already_verified and not args.chain:
            print(f"  ✅ Already verified: {slug}")
            results["already_verified"].append(chain)
            continue

        print(f"  Testing: {slug}...", end=" ", flush=True)

        result = await tester.test_slug(slug)

        if result["success"]:
            # Verify chain ID matches
            returned_chain_id = result["chain_id"]
            if returned_chain_id and returned_chain_id != chain_id:
                print(f"⚠️  Chain ID mismatch!")
                print(f"     Expected: {chain_id}, Got: {returned_chain_id}")
                results["chain_id_mismatch"].append({
                    **chain,
                    "returned_chain_id": returned_chain_id
                })

                if args.update:
                    await update_chain_verification(convex, chain_id, slug, False)
            else:
                print(f"✅ Working ({result['latency_ms']}ms)")
                results["verified"].append(chain)

                if args.update:
                    success = await update_chain_verification(convex, chain_id, slug, True)
                    if success:
                        print(f"     → Updated alchemyVerified=true")
        else:
            print(f"❌ Failed")
            print(f"     Error: {result['error']}")
            results["failed"].append({**chain, "error": result["error"]})

            if args.update:
                await update_chain_verification(convex, chain_id, slug, False)

        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ Verified:          {len(results['verified'])}")
    print(f"✅ Already verified:  {len(results['already_verified'])}")
    print(f"❌ Failed:            {len(results['failed'])}")
    print(f"⚠️  Chain ID mismatch: {len(results['chain_id_mismatch'])}")
    print(f"⏭️  No slug:           {len(results['no_slug'])}")
    print()

    if results["verified"]:
        print("Newly verified chains:")
        for c in results["verified"]:
            print(f"  • {c['name']} ({c['chainId']}): {c['alchemySlug']}")
        print()

    if results["failed"]:
        print("Failed chains:")
        for c in results["failed"]:
            print(f"  • {c['name']} ({c['chainId']}): {c['alchemySlug']}")
            print(f"    Error: {c['error']}")
        print()

    if results["chain_id_mismatch"]:
        print("Chain ID mismatches (slug points to wrong chain):")
        for c in results["chain_id_mismatch"]:
            print(f"  • {c['name']}: expected {c['chainId']}, got {c['returned_chain_id']}")
        print()

    if not args.update and (results["verified"] or results["failed"]):
        print("Run with --update to save verification results to Convex")

    # Close client
    await convex.close()


if __name__ == "__main__":
    asyncio.run(main())
