#!/usr/bin/env python3
"""Simple CLI for testing the Agentic Wallet locally"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional

import httpx

from app.tools.portfolio import get_portfolio
from app.core.chat import run_chat
from app.types import ChatRequest, ChatMessage
from app.services.address import normalize_chain


def print_portfolio(portfolio_data, sources, cached=False):
    """Pretty print portfolio data"""
    if not portfolio_data:
        print("❌ No portfolio data available")
        return
    
    portfolio = portfolio_data
    cache_indicator = "💾" if cached else "🔄"
    
    print(f"\n{cache_indicator} Portfolio Analysis")
    print("=" * 50)
    print(f"Address: {portfolio.address}")
    print(f"Chain: {portfolio.chain.title()}")
    print(f"Total Value: ${portfolio.total_value_usd:,.2f} USD")
    print(f"Token Count: {portfolio.token_count}")
    
    if portfolio.tokens:
        print("\nTokens:")
        print("-" * 50)
        
        # Sort by value descending
        sorted_tokens = sorted(portfolio.tokens, key=lambda t: t.value_usd or Decimal("0"), reverse=True)
        
        for i, token in enumerate(sorted_tokens, 1):
            value_str = f"${token.value_usd:,.2f}" if token.value_usd else "No price"
            price_str = f"@ ${token.price_usd:,.4f}" if token.price_usd else ""
            
            print(f"{i:2d}. {token.balance_formatted:>12} {token.symbol:<8} {value_str:>12} {price_str}")
            if token.name != token.symbol:
                print(f"    {token.name}")
    
    if sources:
        print(f"\nData Sources: {', '.join(s.name for s in sources)}")


async def cli_portfolio(address: str):
    """CLI command to get portfolio"""
    print(f"🔍 Fetching portfolio for {address}...")
    
    try:
        result = await get_portfolio(address)
        print_portfolio(result.data, result.sources, result.cached)
        
        if result.warnings:
            print(f"\n⚠️  Warnings: {'; '.join(result.warnings)}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


async def cli_chat():
    """Interactive chat mode"""
    print("🤖 Agentic Wallet Chat")
    print("Type 'exit' to quit, 'help' for commands")
    print("-" * 40)
    
    messages = []
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Goodbye! 👋")
                break
                
            elif user_input.lower() in ['help', 'h']:
                print("\nCommands:")
                print("  help - Show this help")
                print("  exit - Quit the chat")
                print("  What's in wallet 0x... - Analyze a wallet")
                print("  clear - Clear chat history")
                continue
                
            elif user_input.lower() == 'clear':
                messages = []
                print("Chat history cleared.")
                continue
                
            elif not user_input:
                continue
            
            # Add user message
            messages.append(ChatMessage(role="user", content=user_input))
            
            # Create chat request
            request = ChatRequest(messages=messages)
            
            print("🤖 Assistant: ", end="")
            
            # Get response
            response = await run_chat(request)
            print(response.reply)
            
            # Add assistant response to history
            messages.append(ChatMessage(role="assistant", content=response.reply))
            
            # Show structured data if available
            if response.panels.get("portfolio"):
                portfolio_panel = response.panels["portfolio"]
                print(f"\n📊 Portfolio Summary:")
                print(f"   Total Value: ${float(portfolio_panel['total_value_usd']):,.2f}")
                print(f"   Token Count: {portfolio_panel['token_count']}")
            
            if response.sources:
                sources_str = ", ".join(s["name"] for s in response.sources)
                print(f"   Sources: {sources_str}")
                
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


async def cli_wallet_history(address: str, chain: str = "ethereum", window_days: int = 30, limit: Optional[int] = None):
    """Call the wallet history API and print a concise summary."""

    base_url = "http://localhost:8000"
    normalized_chain = normalize_chain(chain)
    if limit is not None:
        params = {"chain": normalized_chain, "limit": limit}
        banner = f"latest {limit} transfers"
    else:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=window_days)
        params = {
            "chain": normalized_chain,
            "start": start.isoformat(),
            "end": end.isoformat(),
        }
        banner = f"{window_days}d window"

    print(f"📜 Fetching history summary for {address} ({normalized_chain}, {banner})...")
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{base_url}/wallets/{address}/history-summary", params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

    totals = data.get("totals", {})
    print(f"\nTimeframe: {data.get('timeWindow', {}).get('start')} → {data.get('timeWindow', {}).get('end')}")
    print(f"Bucket size: {data.get('bucketSize')}")
    print(f"Inflows:  {totals.get('inflowUsd', 0):,.2f} USD")
    print(f"Outflows: {totals.get('outflowUsd', 0):,.2f} USD")
    print(f"Fees:     {totals.get('feeUsd', 0):,.2f} USD")
    if data.get("notableEvents"):
        print("\nHighlights:")
        for event in data["notableEvents"]:
            print(f" - [{event.get('severity','info').upper()}] {event.get('summary')}")
    if data.get("exportRefs"):
        print("\nExports:")
        for ref in data["exportRefs"]:
            print(f" - {ref.get('format').upper()} ({ref.get('status')}): {ref.get('downloadUrl')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agentic Wallet CLI")
    subparsers = parser.add_subparsers(dest="command")

    portfolio_parser = subparsers.add_parser("portfolio", help="Get portfolio snapshot")
    portfolio_parser.add_argument("address", help="Wallet address")

    subparsers.add_parser("chat", help="Interactive chat mode")

    history_parser = subparsers.add_parser("wallet-history", help="Fetch wallet history summary")
    history_parser.add_argument("address", help="Wallet address")
    history_parser.add_argument("chain", nargs="?", default="ethereum", help="Chain (default: ethereum)")
    history_parser.add_argument("window_days", nargs="?", type=int, default=30, help="Window in days if limit not provided")
    history_parser.add_argument("--limit", type=int, help="Fetch the latest N transactions instead of a time window")

    return parser


async def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return
    
    command = args.command.lower()
    
    if command == "portfolio":
        await cli_portfolio(args.address)
        
    elif command == "chat":
        await cli_chat()

    elif command == "wallet-history":
        if args.limit is not None and args.limit <= 0:
            raise ValueError("Limit must be positive")
        await cli_wallet_history(args.address, args.chain, args.window_days, args.limit)

    elif command in ["help", "-h", "--help"]:
        parser.print_help()

    else:
        print(f"❌ Unknown command: {command}")
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
