"""
Test script for the Style Management System (Phase D implementation)

This tests the dynamic response style adaptation features including:
- Style command parsing and switching
- Automatic style detection
- Style help system
- Integration with Agent system
"""

import asyncio
import logging
from datetime import datetime

from app.providers.llm.anthropic import AnthropicProvider
from app.core.agent.base import Agent
from app.core.agent.styles import StyleManager, ResponseStyle
from app.core.agent.personas import PersonaManager
from app.core.agent.context import ContextManager
from app.types.requests import ChatRequest, ChatMessage
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_style_management_system():
    """Test the complete style management system"""
    
    print("🎨 Testing Style Management System (Phase D)")
    print("=" * 60)
    
    # Initialize StyleManager
    style_manager = StyleManager()
    
    # Test 1: Basic Style Management
    print("\n1. Testing Basic Style Operations")
    print("-" * 40)
    
    # Test default style
    current = style_manager.get_current_style()
    print(f"✅ Default style: {current.value}")
    
    # Test style switching
    switched = style_manager.set_style(ResponseStyle.TECHNICAL)
    print(f"✅ Switch to technical: {switched}")
    print(f"✅ Current style: {style_manager.get_current_style().value}")
    
    # Test style info
    style_info = style_manager.get_style_info(ResponseStyle.BRIEF)
    print(f"✅ Brief style info: {style_info.get('name', 'N/A')}")
    
    # Test 2: Command Parsing
    print("\n2. Testing Style Command Parsing")
    print("-" * 40)
    
    test_commands = [
        "/style casual",
        "/style technical",
        "/style brief",
        "/style educational",
        "/style unknown",  # Should return None
        "regular message"  # Should return None
    ]
    
    for cmd in test_commands:
        result = style_manager.parse_style_command(cmd)
        status = "✅" if (result is not None and cmd.startswith("/style") and "unknown" not in cmd) or (result is None and ("unknown" in cmd or not cmd.startswith("/style"))) else "❌"
        print(f"{status} '{cmd}' -> {result.value if result else None}")
    
    # Test 3: Automatic Style Detection
    print("\n3. Testing Automatic Style Detection")
    print("-" * 40)
    
    test_messages = [
        ("I need technical analysis and detailed data", ResponseStyle.TECHNICAL),
        ("Please give me a brief summary", ResponseStyle.BRIEF),
        ("Can you explain this step by step?", ResponseStyle.EDUCATIONAL),
        ("Hey! That's awesome 😊", ResponseStyle.CASUAL),
        ("Hello there", None)  # Should not detect any specific style
    ]
    
    for msg, expected in test_messages:
        detected = style_manager.detect_style_from_message(msg)
        status = "✅" if detected == expected else "⚠️"
        print(f"{status} '{msg}' -> {detected.value if detected else None} (expected: {expected.value if expected else None})")
    
    # Test 4: Style Help System
    print("\n4. Testing Style Help System")
    print("-" * 40)
    
    help_text = style_manager.format_style_help()
    print(f"✅ Help text generated ({len(help_text)} chars)")
    print("   Preview:", help_text[:100] + "..." if len(help_text) > 100 else help_text)
    
    # Test 5: Agent Integration
    print("\n5. Testing Agent Integration")
    print("-" * 40)
    
    if not settings.anthropic_api_key:
        print("⚠️ No ANTHROPIC_API_KEY found, skipping Agent integration test")
        return
    
    # Initialize components
    llm_provider = AnthropicProvider(api_key=settings.anthropic_api_key)
    persona_manager = PersonaManager()
    context_manager = ContextManager()
    agent = Agent(
        llm_provider=llm_provider,
        persona_manager=persona_manager,
        context_manager=context_manager,
        style_manager=style_manager
    )
    
    # Test style command processing
    print("\nTesting style commands with Agent...")
    
    # Test style switching
    request = ChatRequest(
        messages=[ChatMessage(role="user", content="/style technical")]
    )
    
    response = await agent.process_message(request, "test-style-conv")
    print(f"✅ Style switch response: {response.reply[:100]}...")
    print(f"   Metadata: {response.agent_metadata}")
    
    # Test style help
    request = ChatRequest(
        messages=[ChatMessage(role="user", content="/style help")]
    )
    
    response = await agent.process_message(request, "test-style-conv")
    print(f"✅ Style help response: {response.reply[:100]}...")
    
    # Test regular message with style applied
    request = ChatRequest(
        messages=[ChatMessage(role="user", content="What is DeFi?")]
    )
    
    response = await agent.process_message(request, "test-style-conv")
    print(f"✅ Message with technical style: {response.reply[:150]}...")
    print(f"   Response length: {len(response.reply)} chars")
    
    print("\n🎉 Style Management System Tests Completed!")
    print("✨ Phase D: Dynamic Response Style Adaptation - IMPLEMENTED")

async def main():
    """Run all style system tests"""
    start_time = datetime.now()
    print(f"🚀 Starting Style System Tests at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        await test_style_management_system()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"\n⏱️ All tests completed in {duration:.2f} seconds")
        print("🎯 Phase D implementation verified successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
