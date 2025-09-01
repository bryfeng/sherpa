#!/usr/bin/env python3
"""Simple CLI for testing the Agentic Wallet locally"""

import asyncio
import sys
from datetime import datetime
from decimal import Decimal
from app.tools.portfolio import get_portfolio
from app.core.chat import run_chat
from app.types import ChatRequest, ChatMessage


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


def show_help():
    """Show CLI help"""
    print("Agentic Wallet CLI")
    print("=" * 40)
    print("Commands:")
    print("  portfolio <address>  - Get portfolio for wallet address")
    print("  chat                - Start interactive chat mode")
    print("  help                - Show this help")
    print("")
    print("Examples:")
    print("  python cli.py portfolio 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045")
    print("  python cli.py chat")


async def main():
    """Main CLI entry point"""
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == "portfolio":
        if len(sys.argv) < 3:
            print("❌ Please provide a wallet address")
            print("Usage: python cli.py portfolio <address>")
            return
        address = sys.argv[2]
        await cli_portfolio(address)
        
    elif command == "chat":
        await cli_chat()
        
    elif command in ["help", "-h", "--help"]:
        show_help()
        
    else:
        print(f"❌ Unknown command: {command}")
        show_help()


if __name__ == "__main__":
    asyncio.run(main())
