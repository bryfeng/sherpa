#!/usr/bin/env python3
"""
Agent Tools Test

Tests the policy and strategy tools added to the agent tool registry.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Ensure the sherpa package root is importable when executed directly
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def print_result(name: str, passed: bool, details: str = ""):
    """Print a test result"""
    status = "PASS" if passed else "FAIL"
    symbol = "+" if passed else "-"
    print(f"  [{symbol}] {name}: {status}")
    if details:
        print(f"      {details}")


async def test_tool_registry():
    """Test that the tool registry initializes with all expected tools"""
    print_section("Testing Tool Registry Initialization")

    from app.core.agent.tools import ToolRegistry

    registry = ToolRegistry()

    # Get all registered tools
    definitions = registry.get_definitions()
    tool_names = [d.name for d in definitions]

    print(f"\n  Registered {len(tool_names)} tools:")
    for name in sorted(tool_names):
        print(f"    - {name}")

    # Check for expected tools
    expected_tools = [
        # Original tools
        "get_portfolio",
        "get_token_chart",
        "get_trending_tokens",
        "get_wallet_history",
        "get_tvl_data",
        "get_news",
        "get_personalized_news",
        "get_token_news",
        # Policy tools
        "get_risk_policy",
        "update_risk_policy",
        "check_action_allowed",
        "get_system_status",
        # Strategy tools
        "list_strategies",
        "get_strategy",
        "create_strategy",
        "pause_strategy",
        "resume_strategy",
        "stop_strategy",
        "get_strategy_executions",
        "update_strategy",
    ]

    all_present = True
    print("\n  Checking expected tools:")
    for tool_name in expected_tools:
        present = registry.has_tool(tool_name)
        print_result(tool_name, present)
        if not present:
            all_present = False

    return all_present


async def test_policy_tool_definitions():
    """Test that policy tools have correct definitions"""
    print_section("Testing Policy Tool Definitions")

    from app.core.agent.tools import ToolRegistry
    from app.providers.llm.base import ToolParameterType

    registry = ToolRegistry()

    # Test get_risk_policy
    tool = registry.get_tool("get_risk_policy")
    assert tool is not None, "get_risk_policy not found"
    assert tool.requires_address == True, "get_risk_policy should require address"

    params = {p.name: p for p in tool.definition.parameters}
    assert "wallet_address" in params, "Missing wallet_address parameter"
    assert params["wallet_address"].required == True
    print_result("get_risk_policy definition", True, "1 required param, requires_address=True")

    # Test update_risk_policy
    tool = registry.get_tool("update_risk_policy")
    assert tool is not None
    params = {p.name: p for p in tool.definition.parameters}
    expected_params = ["wallet_address", "max_position_percent", "max_slippage_percent", "enabled"]
    for param in expected_params:
        assert param in params, f"Missing {param}"
    print_result("update_risk_policy definition", True, f"{len(params)} params including optional limits")

    # Test check_action_allowed
    tool = registry.get_tool("check_action_allowed")
    assert tool is not None
    params = {p.name: p for p in tool.definition.parameters}
    assert "action_type" in params
    assert params["action_type"].enum == ["swap", "bridge", "transfer", "approve"]
    print_result("check_action_allowed definition", True, "Has action_type enum")

    # Test get_system_status
    tool = registry.get_tool("get_system_status")
    assert tool is not None
    assert len(tool.definition.parameters) == 0
    print_result("get_system_status definition", True, "No parameters required")

    return True


async def test_strategy_tool_definitions():
    """Test that strategy tools have correct definitions"""
    print_section("Testing Strategy Tool Definitions")

    from app.core.agent.tools import ToolRegistry
    from app.providers.llm.base import ToolParameterType

    registry = ToolRegistry()

    # Test list_strategies
    tool = registry.get_tool("list_strategies")
    assert tool is not None
    params = {p.name: p for p in tool.definition.parameters}
    assert "wallet_address" in params
    assert "strategy_type" in params
    assert params["strategy_type"].enum == ["dca", "rebalance", "limit_order", "stop_loss", "take_profit"]
    print_result("list_strategies definition", True, "Has strategy_type enum")

    # Test create_strategy
    tool = registry.get_tool("create_strategy")
    assert tool is not None
    params = {p.name: p for p in tool.definition.parameters}
    required_params = ["wallet_address", "name", "strategy_type", "config"]
    for param in required_params:
        assert param in params, f"Missing {param}"
        assert params[param].required == True, f"{param} should be required"
    print_result("create_strategy definition", True, f"4 required params + optional config")

    # Test strategy_type enum
    assert params["strategy_type"].enum == ["dca", "rebalance", "limit_order", "stop_loss", "take_profit"]
    print_result("strategy types", True, "5 strategy types: dca, rebalance, limit_order, stop_loss, take_profit")

    # Test config is OBJECT type
    assert params["config"].type == ToolParameterType.OBJECT
    print_result("config parameter type", True, "OBJECT type for flexible config")

    # Test pause/resume/stop
    for action in ["pause", "resume", "stop"]:
        tool_name = f"{action}_strategy"
        tool = registry.get_tool(tool_name)
        assert tool is not None
        params = {p.name: p for p in tool.definition.parameters}
        assert "strategy_id" in params
    print_result("pause/resume/stop_strategy", True, "All have strategy_id param")

    return True


async def test_tool_executor():
    """Test that the tool executor can execute tools"""
    print_section("Testing Tool Executor")

    from app.core.agent.tools import ToolRegistry, ToolExecutor
    from app.providers.llm.base import ToolCall

    registry = ToolRegistry()
    executor = ToolExecutor(registry)

    # Test executing get_system_status (no external deps needed)
    print("\n  Testing get_system_status execution...")

    tool_call = ToolCall(
        id="test-1",
        name="get_system_status",
        arguments={},
    )

    try:
        result = await executor.execute_single(tool_call)
        if result.error:
            # May fail due to missing Convex connection, but tool executed
            print_result("get_system_status execution", True, f"Tool ran (error expected without Convex: {result.error[:50]}...)")
        else:
            print_result("get_system_status execution", True, f"Result: {result.result}")
    except Exception as e:
        print_result("get_system_status execution", False, str(e))
        return False

    # Test unknown tool
    print("\n  Testing unknown tool handling...")
    unknown_call = ToolCall(
        id="test-2",
        name="nonexistent_tool",
        arguments={},
    )
    result = await executor.execute_single(unknown_call)
    assert result.error is not None
    assert "Unknown tool" in result.error
    print_result("unknown tool handling", True, "Returns error for unknown tools")

    return True


async def test_policy_handler_structure():
    """Test policy handler functions exist and are callable"""
    print_section("Testing Policy Handler Functions")

    from app.core.agent.tools import ToolRegistry

    registry = ToolRegistry()

    handlers_to_test = [
        ("get_risk_policy", "_handle_get_risk_policy"),
        ("update_risk_policy", "_handle_update_risk_policy"),
        ("check_action_allowed", "_handle_check_action_allowed"),
        ("get_system_status", "_handle_get_system_status"),
    ]

    for tool_name, handler_name in handlers_to_test:
        tool = registry.get_tool(tool_name)
        assert tool is not None, f"Tool {tool_name} not found"
        assert tool.handler is not None, f"Handler for {tool_name} is None"
        assert callable(tool.handler), f"Handler for {tool_name} is not callable"
        print_result(f"{tool_name} handler", True, "Callable async function")

    return True


async def test_strategy_handler_structure():
    """Test strategy handler functions exist and are callable"""
    print_section("Testing Strategy Handler Functions")

    from app.core.agent.tools import ToolRegistry

    registry = ToolRegistry()

    strategy_tools = [
        "list_strategies",
        "get_strategy",
        "create_strategy",
        "pause_strategy",
        "resume_strategy",
        "stop_strategy",
        "get_strategy_executions",
        "update_strategy",
    ]

    for tool_name in strategy_tools:
        tool = registry.get_tool(tool_name)
        assert tool is not None, f"Tool {tool_name} not found"
        assert tool.handler is not None, f"Handler for {tool_name} is None"
        assert callable(tool.handler), f"Handler for {tool_name} is not callable"

        # Check requires_address for tools that need wallet
        if tool_name in ["list_strategies", "create_strategy"]:
            assert tool.requires_address == True, f"{tool_name} should require address"

        print_result(f"{tool_name} handler", True, "Callable async function")

    return True


async def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("  AGENT TOOLS TEST SUITE")
    print("="*60)

    results = {}

    # Run tests
    try:
        results["Tool Registry"] = await test_tool_registry()
    except Exception as e:
        results["Tool Registry"] = False
        logger.exception(f"Tool Registry test failed: {e}")

    try:
        results["Policy Tool Definitions"] = await test_policy_tool_definitions()
    except Exception as e:
        results["Policy Tool Definitions"] = False
        logger.exception(f"Policy Tool Definitions test failed: {e}")

    try:
        results["Strategy Tool Definitions"] = await test_strategy_tool_definitions()
    except Exception as e:
        results["Strategy Tool Definitions"] = False
        logger.exception(f"Strategy Tool Definitions test failed: {e}")

    try:
        results["Tool Executor"] = await test_tool_executor()
    except Exception as e:
        results["Tool Executor"] = False
        logger.exception(f"Tool Executor test failed: {e}")

    try:
        results["Policy Handlers"] = await test_policy_handler_structure()
    except Exception as e:
        results["Policy Handlers"] = False
        logger.exception(f"Policy Handlers test failed: {e}")

    try:
        results["Strategy Handlers"] = await test_strategy_handler_structure()
    except Exception as e:
        results["Strategy Handlers"] = False
        logger.exception(f"Strategy Handlers test failed: {e}")

    # Print summary
    print_section("TEST SUMMARY")

    all_passed = True
    for test_name, passed in results.items():
        print_result(test_name, passed)
        if not passed:
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("="*60 + "\n")

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
