#!/usr/bin/env python3
"""
Quick test script to verify Convex connection is working.

Usage:
    cd sherpa
    python scripts/test_convex.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.db.convex_client import ConvexClient, ConvexError


async def test_convex_connection():
    print("=" * 50)
    print("Convex Connection Test")
    print("=" * 50)

    # Check if Convex is configured
    print(f"\n1. Checking configuration...")
    print(f"   CONVEX_URL: {settings.convex_url or '(not set)'}")
    print(f"   CONVEX_DEPLOY_KEY: {'****' + settings.convex_deploy_key[-8:] if settings.convex_deploy_key else '(not set)'}")

    if not settings.convex_url:
        print("\n❌ CONVEX_URL is not set. Please add it to your .env file.")
        print("   Run 'cd frontend && npm run convex:dev' to get your deployment URL.")
        return False

    # Try to connect
    print(f"\n2. Connecting to Convex...")
    try:
        client = ConvexClient()

        # Test health check query
        print(f"\n3. Running health check query...")
        result = await client.query("health:check", {})

        print(f"\n✅ Connection successful!")
        print(f"   Status: {result.get('status')}")
        print(f"   Database: {result.get('database')}")
        print(f"   User count: {result.get('userCount')}")
        print(f"   Timestamp: {result.get('timestamp')}")

        await client.close()
        return True

    except ConvexError as e:
        print(f"\n❌ Convex error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False


async def test_create_user():
    """Optional: Test creating a user"""
    print(f"\n4. Testing user creation...")

    try:
        client = ConvexClient()

        # Create a test user
        result = await client.get_or_create_user(
            address="0xTEST000000000000000000000000000000000",
            chain="ethereum"
        )

        print(f"   User ID: {result.get('user', {}).get('_id')}")
        print(f"   Wallet ID: {result.get('wallet', {}).get('_id')}")
        print(f"   Is new: {result.get('isNew')}")

        await client.close()
        return True

    except Exception as e:
        print(f"   ⚠️  Could not create test user: {e}")
        return False


async def main():
    success = await test_convex_connection()

    if success:
        # Optionally test user creation
        await test_create_user()

    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed! Convex is working.")
    else:
        print("❌ Tests failed. Check your configuration.")
    print("=" * 50)

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
