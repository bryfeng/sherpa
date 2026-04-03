#!/usr/bin/env python3
"""
Interactive Chat Test Harness

Tests natural language understanding and tool calling through the real API or in-process.
Supports both interactive REPL mode and automated test suites.

Usage:
    # Interactive mode - type messages, see what tools fire
    python scripts/test_chat.py --interactive

    # Run test suite with natural language variations
    python scripts/test_chat.py --suite bridge

    # Single message test
    python scripts/test_chat.py --message "move my usdc.e from ink to ethereum"

    # Use real API instead of in-process
    python scripts/test_chat.py --api http://localhost:8000 --interactive

    # With specific provider/model
    python scripts/test_chat.py --provider zai --model glm-4-plus --interactive
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import httpx

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ToolCallResult:
    """Captured tool call from agent."""
    name: str
    arguments: Dict[str, Any]
    result: Any = None
    error: str = None


@dataclass
class TestResult:
    """Result of a single test case."""
    message: str
    expected_tool: Optional[str]
    actual_tools: List[str]
    tool_calls: List[ToolCallResult]
    response: str
    passed: bool
    notes: str = ""


# =============================================================================
# Test Suites - Natural Language Variations
# =============================================================================

BRIDGE_TEST_SUITE = [
    # Direct bridge requests
    {"message": "bridge 1 USDC.e from ink to ethereum", "expect_tool": "get_bridge_quote"},
    {"message": "I want to bridge my USDC.e to mainnet", "expect_tool": "get_bridge_quote"},
    {"message": "move 5 usdc.e from ink chain to eth", "expect_tool": "get_bridge_quote"},

    # Swap phrasing (cross-chain)
    {"message": "swap 1 USDC.e on ink to USDC on ethereum", "expect_tool": "get_bridge_quote"},
    {"message": "swap my stablecoins from ink to mainnet", "expect_tool": "get_bridge_quote"},
    {"message": "I want to swap usdc.e on ink for usdc on ethereum", "expect_tool": "get_bridge_quote"},

    # Casual/conversational
    {"message": "can you help me move my usdc.e to ethereum?", "expect_tool": "get_bridge_quote"},
    {"message": "how do I get my USDC.e from ink over to mainnet?", "expect_tool": "get_bridge_quote"},
    {"message": "transfer usdc.e ink → ethereum", "expect_tool": "get_bridge_quote"},

    # Ambiguous - should ask for clarification or infer
    {"message": "bridge my usdc.e", "expect_tool": None, "notes": "Missing destination - should ask"},
    {"message": "send usdc to ethereum", "expect_tool": "get_bridge_quote", "notes": "Should infer source from context"},
]

SWAP_TEST_SUITE = [
    # Same-chain swaps
    {"message": "swap 1 ETH for USDC on ethereum", "expect_tool": "get_swap_quote"},
    {"message": "I want to trade my ETH for USDC on mainnet", "expect_tool": "get_swap_quote"},
    {"message": "exchange 0.5 eth to usdc", "expect_tool": "get_swap_quote"},

    # Should NOT trigger bridge
    {"message": "swap eth to usdc on the same chain", "expect_tool": "get_swap_quote"},
]

PORTFOLIO_TEST_SUITE = [
    {"message": "what's in my wallet?", "expect_tool": "get_portfolio"},
    {"message": "show me my holdings", "expect_tool": "get_portfolio"},
    {"message": "check my balance on ink", "expect_tool": "get_portfolio"},
    {"message": "how much usdc.e do I have?", "expect_tool": "get_portfolio"},
]

TEST_SUITES = {
    "bridge": BRIDGE_TEST_SUITE,
    "swap": SWAP_TEST_SUITE,
    "portfolio": PORTFOLIO_TEST_SUITE,
    "all": BRIDGE_TEST_SUITE + SWAP_TEST_SUITE + PORTFOLIO_TEST_SUITE,
}


# =============================================================================
# Test Harness
# =============================================================================

class ChatTestHarness:
    """Flexible test harness for chat/agent testing."""

    def __init__(
        self,
        api_url: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        wallet: str = "0x50ac5CFcc81BB0872e85255D7079F8a529345D16",
        chain: str = "ink",
        verbose: bool = False,
    ):
        self.api_url = api_url
        self.provider = provider
        self.model = model
        self.wallet = wallet
        self.chain = chain
        self.verbose = verbose

        self._agent = None
        self._http_client = None

    async def setup(self):
        """Initialize agent or HTTP client."""
        if self.api_url:
            self._http_client = httpx.AsyncClient(base_url=self.api_url, timeout=60.0)
            print(f"Using API: {self.api_url}")
        else:
            # In-process agent
            from app.core.agent import Agent, PersonaManager, ContextManager
            from app.providers.llm import get_llm_provider

            llm = get_llm_provider(provider_name=self.provider, model=self.model)
            print(f"Using in-process agent: {llm.__class__.__name__} / {llm.model}")

            self._agent = Agent(
                llm_provider=llm,
                persona_manager=PersonaManager(),
                context_manager=ContextManager(llm_provider=llm),
            )

    async def cleanup(self):
        """Cleanup resources."""
        if self._http_client:
            await self._http_client.aclose()

    async def send_message(self, message: str) -> Dict[str, Any]:
        """Send a message and capture tool calls."""
        if self.api_url:
            return await self._send_via_api(message)
        else:
            return await self._send_in_process(message)

    async def _send_via_api(self, message: str) -> Dict[str, Any]:
        """Send message through HTTP API."""
        payload = {
            "messages": [{"role": "user", "content": message}],
            "address": self.wallet,
            "chain": self.chain,
        }

        try:
            # Try streaming endpoint first
            response = await self._http_client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

            return {
                "response": data.get("reply", ""),
                "tool_calls": self._extract_tool_calls_from_api(data),
                "raw": data,
            }
        except Exception as e:
            return {"error": str(e), "response": "", "tool_calls": []}

    async def _send_in_process(self, message: str) -> Dict[str, Any]:
        """Send message directly to agent."""
        from app.types.requests import ChatRequest, ChatMessage

        request = ChatRequest(
            messages=[ChatMessage(role="user", content=message)],
            address=self.wallet,
            chain=self.chain,
        )

        try:
            response = await self._agent.process_message(
                request=request,
                conversation_id="test-harness",
                persona_name="friendly",
            )

            # Extract tool calls from metadata
            tool_data_keys = response.agent_metadata.get("tool_data_keys", [])
            tool_calls = []

            # Get actual tool call details from agent's last run
            for key in tool_data_keys:
                tool_calls.append(ToolCallResult(
                    name=key,
                    arguments={},  # Would need to capture from agent internals
                ))

            return {
                "response": response.reply,
                "tool_calls": tool_calls,
                "tool_names": tool_data_keys,
                "panels": list(response.panels.keys()),
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e), "response": "", "tool_calls": []}

    def _extract_tool_calls_from_api(self, data: Dict) -> List[ToolCallResult]:
        """Extract tool calls from API response."""
        tool_calls = []
        # API response format may vary - adapt as needed
        if "agent_metadata" in data:
            for key in data["agent_metadata"].get("tool_data_keys", []):
                tool_calls.append(ToolCallResult(name=key, arguments={}))
        return tool_calls

    async def run_test(self, test_case: Dict) -> TestResult:
        """Run a single test case."""
        message = test_case["message"]
        expected = test_case.get("expect_tool")
        notes = test_case.get("notes", "")

        result = await self.send_message(message)

        tool_names = result.get("tool_names", [tc.name for tc in result.get("tool_calls", [])])

        # Determine pass/fail
        if expected is None:
            # No specific tool expected - just checking it doesn't crash
            passed = "error" not in result
        else:
            passed = expected in tool_names

        return TestResult(
            message=message,
            expected_tool=expected,
            actual_tools=tool_names,
            tool_calls=result.get("tool_calls", []),
            response=result.get("response", "")[:500],
            passed=passed,
            notes=notes,
        )

    async def run_suite(self, suite_name: str) -> List[TestResult]:
        """Run a test suite."""
        if suite_name not in TEST_SUITES:
            print(f"Unknown suite: {suite_name}. Available: {list(TEST_SUITES.keys())}")
            return []

        suite = TEST_SUITES[suite_name]
        results = []

        print(f"\n{'='*60}")
        print(f"Running test suite: {suite_name} ({len(suite)} tests)")
        print(f"{'='*60}\n")

        for i, test_case in enumerate(suite, 1):
            print(f"[{i}/{len(suite)}] Testing: \"{test_case['message']}\"")

            result = await self.run_test(test_case)
            results.append(result)

            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"  {status} | Expected: {result.expected_tool} | Got: {result.actual_tools}")

            if self.verbose and result.response:
                print(f"  Response: {result.response[:200]}...")

            if result.notes:
                print(f"  Note: {result.notes}")
            print()

        # Summary
        passed = sum(1 for r in results if r.passed)
        print(f"{'='*60}")
        print(f"Results: {passed}/{len(results)} passed")
        print(f"{'='*60}")

        return results

    async def interactive_mode(self):
        """Interactive REPL for testing messages."""
        print("\n" + "="*60)
        print("Interactive Chat Test Mode")
        print("="*60)
        print("Type messages to test. Commands:")
        print("  /quit - Exit")
        print("  /suite <name> - Run test suite (bridge, swap, portfolio, all)")
        print("  /verbose - Toggle verbose mode")
        print("="*60 + "\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break

            if not user_input:
                continue

            # Commands
            if user_input.startswith("/"):
                cmd = user_input[1:].split()
                if cmd[0] == "quit":
                    break
                elif cmd[0] == "suite" and len(cmd) > 1:
                    await self.run_suite(cmd[1])
                    continue
                elif cmd[0] == "verbose":
                    self.verbose = not self.verbose
                    print(f"Verbose mode: {self.verbose}")
                    continue
                else:
                    print(f"Unknown command: {cmd[0]}")
                    continue

            # Regular message
            print("Processing...")
            result = await self.send_message(user_input)

            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"\nTools called: {result.get('tool_names', [])}")
                if result.get("panels"):
                    print(f"Panels: {result['panels']}")
                print(f"\nResponse:\n{result['response'][:1000]}")
                if len(result['response']) > 1000:
                    print("... (truncated)")
            print()


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive Chat Test Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive REPL mode")
    parser.add_argument("--suite", "-s", type=str,
                        help="Run test suite (bridge, swap, portfolio, all)")
    parser.add_argument("--message", "-m", type=str,
                        help="Test a single message")

    parser.add_argument("--api", type=str,
                        help="API URL (e.g., http://localhost:8000). If not set, runs in-process")
    parser.add_argument("--provider", "-p", type=str,
                        help="LLM provider (anthropic, zai, openai)")
    parser.add_argument("--model", type=str,
                        help="Model name")

    parser.add_argument("--wallet", "-w", type=str,
                        default="0x50ac5CFcc81BB0872e85255D7079F8a529345D16",
                        help="Wallet address")
    parser.add_argument("--chain", type=str, default="ink",
                        help="Default chain context")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    return parser.parse_args()


async def main():
    args = parse_args()

    harness = ChatTestHarness(
        api_url=args.api,
        provider=args.provider,
        model=args.model,
        wallet=args.wallet,
        chain=args.chain,
        verbose=args.verbose,
    )

    await harness.setup()

    try:
        if args.interactive:
            await harness.interactive_mode()
        elif args.suite:
            await harness.run_suite(args.suite)
        elif args.message:
            result = await harness.send_message(args.message)
            print(f"Tools called: {result.get('tool_names', [])}")
            print(f"Response: {result.get('response', '')}")
        else:
            # Default: run bridge suite
            await harness.run_suite("bridge")
    finally:
        await harness.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
