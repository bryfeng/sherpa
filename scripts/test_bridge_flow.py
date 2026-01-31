#!/usr/bin/env python3
"""
QA Test Script: Bridge Flow from Ink to Ethereum

Tests the complete flow:
1. Chain registry lookup (ink → chain ID)
2. Portfolio/balance lookup for USDC.e on Ink
3. Bridge quote from Ink USDC.e → Ethereum USDC
4. Tool call simulation through the agent

Usage:
    cd backend
    python scripts/test_bridge_flow.py
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from decimal import Decimal

# Add the backend directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("bridge_test")

# Test parameters
TEST_WALLET = "0x50ac5CFcc81BB0872e85255D7079F8a529345D16"
FROM_CHAIN = "ink"
TO_CHAIN = "ethereum"
TOKEN = "USDC.e"
AMOUNT = 5.0


async def test_chain_registry():
    """Test 1: Verify chain registry can resolve 'ink' to a chain ID."""
    logger.info("=" * 60)
    logger.info("TEST 1: Chain Registry Lookup")
    logger.info("=" * 60)

    try:
        from app.core.bridge.chain_registry import get_chain_registry

        registry = await get_chain_registry()

        # Test ink chain lookup
        ink_chain_id = registry.get_chain_id("ink")
        eth_chain_id = registry.get_chain_id("ethereum")

        logger.info(f"  Ink chain ID: {ink_chain_id}")
        logger.info(f"  Ethereum chain ID: {eth_chain_id}")
        logger.info(f"  Total chains loaded: {registry.chain_count}")

        if ink_chain_id is None:
            logger.error("  FAILED: Could not resolve 'ink' chain")
            return False

        if eth_chain_id is None:
            logger.error("  FAILED: Could not resolve 'ethereum' chain")
            return False

        # Get chain names for confirmation
        ink_name = registry.get_chain_name(ink_chain_id)
        eth_name = registry.get_chain_name(eth_chain_id)
        logger.info(f"  Ink display name: {ink_name}")
        logger.info(f"  Ethereum display name: {eth_name}")

        logger.info("  PASSED: Chain registry lookup successful")
        return True

    except Exception as e:
        logger.error(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_portfolio_lookup():
    """Test 2: Fetch portfolio and check for USDC.e on Ink."""
    logger.info("=" * 60)
    logger.info("TEST 2: Portfolio/Balance Lookup")
    logger.info("=" * 60)

    try:
        from app.tools.portfolio import get_portfolio

        # Fetch portfolio for Ink chain
        logger.info(f"  Fetching portfolio for {TEST_WALLET} on {FROM_CHAIN}...")
        result = await get_portfolio(TEST_WALLET, FROM_CHAIN)

        if result.data:
            logger.info(f"  Total value: ${result.data.total_value_usd:.2f}")
            logger.info(f"  Token count: {result.data.token_count}")

            # Look for USDC.e
            usdc_found = False
            for token in result.data.tokens or []:
                symbol = token.symbol or token.name or "Unknown"
                balance = token.balance_formatted or "0"
                value = token.value_usd or 0

                logger.info(f"    - {symbol}: {balance} (${value:.2f})")

                if symbol.lower() in ["usdc.e", "usdce", "usdc"]:
                    usdc_found = True
                    logger.info(f"  Found USDC.e: {balance}")

            if not usdc_found:
                logger.warning("  WARNING: No USDC.e found in portfolio")
                logger.info("  (This is OK if the wallet doesn't have USDC.e on Ink)")

            logger.info("  PASSED: Portfolio lookup successful")
            return True
        else:
            logger.warning(f"  No portfolio data returned. Warnings: {result.warnings}")
            return True  # Not a failure, wallet might just be empty on this chain

    except Exception as e:
        logger.error(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_bridge_quote():
    """Test 3: Get a bridge quote from Ink USDC.e to Ethereum USDC."""
    logger.info("=" * 60)
    logger.info("TEST 3: Bridge Quote (Relay API)")
    logger.info("=" * 60)

    try:
        from app.core.agent.tools import ToolRegistry

        # Initialize tool registry
        registry = ToolRegistry(logger=logger)

        # Get the bridge quote tool
        bridge_tool = registry.get_tool("get_bridge_quote")
        if not bridge_tool:
            logger.error("  FAILED: get_bridge_quote tool not registered")
            return False

        logger.info(f"  Tool registered: {bridge_tool.definition.name}")
        logger.info(f"  Description: {bridge_tool.definition.description[:100]}...")

        # Call the handler directly
        logger.info(f"  Calling bridge quote handler...")
        logger.info(f"    wallet_address: {TEST_WALLET}")
        logger.info(f"    from_chain: {FROM_CHAIN}")
        logger.info(f"    to_chain: {TO_CHAIN}")
        logger.info(f"    token: {TOKEN}")
        logger.info(f"    amount: {AMOUNT}")

        result = await bridge_tool.handler(
            wallet_address=TEST_WALLET,
            from_chain=FROM_CHAIN,
            to_chain=TO_CHAIN,
            token=TOKEN,
            amount=AMOUNT,
        )

        logger.info(f"  Result: {json.dumps(result, indent=2, default=str)[:1000]}")

        if result.get("success"):
            logger.info("  PASSED: Bridge quote retrieved successfully")

            # Log key details
            if "output_amount" in result:
                logger.info(f"    Output amount: {result['output_amount']}")
            if "fees" in result:
                logger.info(f"    Fees: {result['fees']}")
            if "estimated_time" in result:
                logger.info(f"    Estimated time: {result['estimated_time']}")

            return True
        else:
            error = result.get("error", "Unknown error")
            hint = result.get("hint", "")
            logger.error(f"  FAILED: {error}")
            if hint:
                logger.info(f"  Hint: {hint}")
            return False

    except Exception as e:
        logger.error(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_agent_tool_calling():
    """Test 4: Simulate agent processing a bridge request."""
    logger.info("=" * 60)
    logger.info("TEST 4: Agent Tool Calling Simulation")
    logger.info("=" * 60)

    try:
        from app.core.agent import Agent, PersonaManager, ContextManager
        from app.providers.llm import get_llm_provider
        from app.types.requests import ChatRequest, ChatMessage

        # Initialize the agent
        logger.info("  Initializing agent...")
        llm_provider = get_llm_provider()
        persona_manager = PersonaManager()
        context_manager = ContextManager(llm_provider=llm_provider)

        agent = Agent(
            llm_provider=llm_provider,
            persona_manager=persona_manager,
            context_manager=context_manager,
            logger=logger,
        )

        # Check if provider supports tools
        logger.info(f"  LLM provider: {llm_provider.__class__.__name__}")
        logger.info(f"  Supports tools: {llm_provider.supports_tools}")

        # List registered tools
        tools = agent.tool_registry.get_definitions()
        tool_names = [t.name for t in tools]
        logger.info(f"  Registered tools ({len(tools)}): {tool_names}")

        # Verify bridge tool is registered
        if "get_bridge_quote" not in tool_names:
            logger.error("  FAILED: get_bridge_quote not in registered tools")
            return False

        logger.info("  PASSED: Agent initialized with tools")

        # Create a test request
        test_message = f"Bridge {AMOUNT} {TOKEN} from {FROM_CHAIN} to {TO_CHAIN}"
        logger.info(f"  Test message: '{test_message}'")

        request = ChatRequest(
            messages=[ChatMessage(role="user", content=test_message)],
            address=TEST_WALLET,
            chain=FROM_CHAIN,
        )

        # Process the message
        logger.info("  Processing message through agent...")
        response = await agent.process_message(
            request=request,
            conversation_id="test-bridge-flow",
            persona_name="friendly",
        )

        logger.info(f"  Response reply: {response.reply[:500]}...")
        logger.info(f"  Panels: {list(response.panels.keys())}")
        logger.info(f"  Sources: {len(response.sources)}")
        logger.info(f"  Tool data keys: {response.agent_metadata.get('tool_data_keys', [])}")

        # Check if bridge tool was called
        tool_data_keys = response.agent_metadata.get('tool_data_keys', [])
        if 'get_bridge_quote' in tool_data_keys:
            logger.info("  PASSED: Agent called get_bridge_quote tool")
            return True
        else:
            logger.warning("  WARNING: Agent did not call get_bridge_quote tool")
            logger.info("  This may indicate the LLM is not using tools properly")
            return False

    except Exception as e:
        logger.error(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("BRIDGE FLOW QA TEST")
    logger.info(f"Wallet: {TEST_WALLET}")
    logger.info(f"Route: {AMOUNT} {TOKEN} on {FROM_CHAIN} → {TO_CHAIN}")
    logger.info("=" * 60)
    logger.info("")

    results = {}

    # Run tests
    results["chain_registry"] = await test_chain_registry()
    results["portfolio_lookup"] = await test_portfolio_lookup()
    results["bridge_quote"] = await test_bridge_quote()
    results["agent_tool_calling"] = await test_agent_tool_calling()

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        icon = "✓" if passed else "✗"
        logger.info(f"  {icon} {test_name}: {status}")
        if not passed:
            all_passed = False

    logger.info("")
    if all_passed:
        logger.info("All tests passed!")
        return 0
    else:
        logger.info("Some tests failed. Check the logs above for details.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
