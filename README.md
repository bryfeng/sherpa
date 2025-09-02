# Agentic Wallet Python Backend - AI-Powered Chat System

An intelligent cryptocurrency portfolio analysis system powered by Large Language Models (LLM). Features natural language conversations, multiple AI personas, and comprehensive portfolio insights for Ethereum addresses.

## 🤖 What's New: AI-Powered Chat System

This system has been completely transformed from basic hardcoded responses to a sophisticated AI agent that can:
- **Understand natural language** questions about crypto portfolios
- **Switch between AI personas** (Friendly Guide, Technical Analyst, Professional Advisor, Educational Teacher)  
- **Maintain conversation context** and remember previous interactions
- **Integrate real portfolio data** with intelligent LLM analysis
- **Provide graceful fallbacks** when AI services are unavailable

## ✨ Key Features

### 🧠 **Intelligent Agent System**
- **Multi-Persona AI**: 4 distinct personalities for different conversation styles
- **Context Awareness**: Remembers conversation history and user preferences  
- **Smart Tool Integration**: Seamlessly combines portfolio data with LLM insights
- **Dynamic Response Styling**: Adapts communication based on user needs

### 💼 **Portfolio Analysis**
- **Complete Token Analysis**: Balances, USD values, token metadata for any Ethereum address
- **AI-Generated Insights**: Intelligent analysis and recommendations from LLMs
- **Real-time Data**: Alchemy blockchain data + CoinGecko price feeds
- **Structured + Conversational**: Both API data and natural language explanations

### 🔧 **Developer-Friendly**
- **Multi-Provider LLM Support**: Currently Anthropic Claude, designed for OpenAI/others
- **FastAPI Backend**: REST API with automatic OpenAPI documentation
- **Comprehensive Testing**: Agent system, API integration, and provider testing
- **CLI Tools**: Command-line interface for development and testing

## 🚀 Quick Start

### 1. Setup Environment
```bash
cd agentic_wallet_py
./setup.sh
```

### 2. Configure API Keys
Edit `.env` file with your API credentials:
```env
# Required: Blockchain data
ALCHEMY_API_KEY=your_alchemy_key_here

# Required: AI/LLM provider  
ANTHROPIC_API_KEY=your_anthropic_key_here

# Optional: Enhanced price data
COINGECKO_API_KEY=your_coingecko_key_here

# Optional: LLM Configuration
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-sonnet-20241022
MAX_TOKENS=4000
TEMPERATURE=0.7
```

### 3. Test the AI System
```bash
# Test AI Agent system
python test_agent_system.py

# Test via CLI chat
python cli.py chat

# Test portfolio analysis  
python cli.py portfolio 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
```

### 4. Start API Server
```bash
python main.py
# Visit http://localhost:8000/docs for API documentation
```

## 🎭 AI Persona System

The system includes 4 distinct AI personalities that you can switch between, and **all personas are fully customizable** through user-friendly YAML configuration files:

### 🤗 **Friendly Crypto Guide** (Default)
- **Style**: Casual, conversational, encouraging
- **Best For**: Beginners, general portfolio questions
- **Switch**: Automatic default or `/persona friendly`
- **Example**: *"Hey there! 👋 I'd love to help you understand your crypto portfolio! Think of it like a digital wallet..."*

### 🔬 **Technical DeFi Analyst**  
- **Style**: Detailed technical analysis, protocol-focused
- **Best For**: Advanced users, yield farming, protocol analysis
- **Switch**: `/persona technical`
- **Example**: *"Analyzing your portfolio's DeFi exposure: Your LP tokens show impermanent loss risk of..."*

### 💼 **Professional Portfolio Advisor**
- **Style**: Formal financial guidance, risk-aware
- **Best For**: Investment strategy, portfolio optimization
- **Switch**: `/persona professional`  
- **Example**: *"Based on modern portfolio theory, your current allocation presents concentration risk..."*

### 🎓 **Educational Crypto Teacher**
- **Style**: Patient explanations, learning-focused
- **Best For**: Understanding concepts, crypto education
- **Switch**: `/persona educational`
- **Example**: *"Let me explain what each token in your portfolio does, starting with the basics..."*

### 🎨 **Dynamic Response Styles**
Combine personas with response styles for ultimate customization:
- **`/style casual`** - Relaxed, friendly communication
- **`/style technical`** - Precise, data-focused responses  
- **`/style brief`** - Concise, to-the-point answers
- **`/style educational`** - Detailed explanations with examples

**Example**: `/persona technical` + `/style brief` = *"Portfolio TVL: $1.2M. ETH dominance: 65%. DeFi exposure: 12% via UNI/AAVE. Risk score: Medium."*

## 🎨 Persona Customization

**🔥 New Feature**: All personas are now **100% customizable** through user-friendly YAML files! No coding required.

### Quick Persona Customization
```bash
# 1. Navigate to personas directory
cd personas/

# 2. Edit any persona (they're just text files!)
# Examples:
nano friendly.yaml      # Make the friendly guide more enthusiastic
nano technical.yaml     # Add new technical keywords
nano professional.yaml  # Adjust formality level
nano educational.yaml   # Modify teaching style

# 3. Changes take effect immediately - no restart needed!
```

### YAML Configuration Structure
Each persona is defined in a simple, commented YAML file:
```yaml
# personas/friendly.yaml
name: friendly
display_name: "Friendly Crypto Guide"
description: "Approachable and encouraging crypto assistant"

# Communication Style
tone: "warm and conversational"
formality: "casual"
technical_depth: "medium - explains concepts clearly"
use_emojis: true
response_length: "medium"

# Areas of Expertise
specializations:
  - portfolio_analysis
  - crypto_education
  - user_encouragement

# System Prompt - The AI's core behavior instructions
system_prompt: |
  You are a friendly and knowledgeable crypto portfolio assistant.
  Your personality is warm, approachable, and encouraging...
  [Full customizable prompt]

# Auto-detection keywords (when to use this persona automatically)
auto_detection_keywords:
  - friendly
  - help
  - explain
  - new to crypto
  - beginner
```

### Customization Examples

**Make a persona more enthusiastic:**
```yaml
# Edit personas/friendly.yaml
tone: "extremely enthusiastic and upbeat"
formality: "very casual and energetic"
```

**Add technical expertise:**
```yaml
# Edit personas/technical.yaml
auto_detection_keywords:
  - yield farming
  - liquidity pools
  - smart contracts  # New keywords
  - defi protocols
```

**Create conservative financial advice:**
```yaml
# Edit personas/professional.yaml
system_prompt: |
  Always emphasize risks before opportunities.
  Suggest diversification and conservative strategies.
  Warn about market volatility frequently.
```

📖 **Full Customization Guide**: See `docs/persona_customization.md` for complete documentation with examples, best practices, and troubleshooting.

## 💬 Example Conversations

### Basic Portfolio Analysis
```
🧑 User: "What's in wallet 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045?"

🤖 Friendly Guide: "Hey there! 👋 I just analyzed this wallet and wow, there's some interesting stuff here! 

This portfolio has a total value of $1,234,567 across 15 different tokens. The biggest holdings are:

• 123.45 ETH ($456,789) - The backbone of the portfolio! 
• 1,000 USDC ($1,000) - Nice stable foundation
• 50.5 UNI ($355) - Governance token for Uniswap

What's cool is this looks like someone who's really into DeFi - I can see some LP tokens and protocol tokens. Want me to break down what each of these does? 🚀"
```

### Technical Analysis Mode
```
🧑 User: "/persona technical"
🧑 User: "Analyze the yield farming potential"

🔬 Technical Analyst: "Portfolio yield optimization analysis:

**Current DeFi Exposure:**
- UNI governance token: Potential for 3.2% APR through staking
- LP positions detected: USDC/ETH pair showing 0.05% fee tier
- Idle USDC: $1,000 earning 0% - efficiency opportunity

**Recommended Strategies:**
1. **Uniswap V3**: Deploy USDC in concentrated liquidity (±5% range) → 8-15% APR
2. **Aave**: Collateralize ETH, borrow stablecoins → leveraged farming opportunities  
3. **Compound**: Basic lending of idle USDC → 3-5% APR

**Risk Assessment:** Current Sharpe ratio suggests portfolio could benefit from yield-bearing positions while maintaining risk profile."
```

## 🏗️ Architecture Overview

The system uses a sophisticated multi-layer architecture:

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   FastAPI Layer     │    │    Agent System      │    │   LLM Providers     │
│                     │    │                      │    │                     │
│  • /chat            │────│  • Agent Core        │────│  • Anthropic Claude │
│  • /tools/portfolio │    │  • 4 AI Personas     │    │  • OpenAI (ready)   │
│  • /healthz         │    │  • Context Manager   │    │  • Grok (ready)     │
│                     │    │  • Tool Integration  │    │  • Local (ready)    │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
           │                           │                           │
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   Data Providers    │    │  Response Builder    │    │   Configuration     │
│                     │    │                      │    │                     │
│  • Alchemy API      │────│  • Structure Format  │    │  • API Keys         │
│  • CoinGecko API    │    │  • Panel Generation  │    │  • Model Settings   │
│  • Portfolio Tools  │    │  • Source Attribution│    │  • Persona Configs  │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
```

### Key Components

#### 🤖 **Agent System** (`app/core/agent/`)
- **`base.py`**: Core Agent orchestrator managing the entire conversation flow
- **`personas.py`**: AI personality manager that loads from external YAML files
- **`styles.py`**: Dynamic response style system for customizing communication format
- **`context.py`**: Conversation memory, history management, and context compression
- **Smart tool integration**: Seamlessly incorporates portfolio data into LLM context

#### 🎭 **Persona Configuration** (`personas/`)
- **`friendly.yaml`**: Friendly Crypto Guide configuration
- **`technical.yaml`**: Technical DeFi Analyst configuration
- **`professional.yaml`**: Professional Portfolio Advisor configuration
- **`educational.yaml`**: Educational Crypto Teacher configuration
- **User-editable**: All personas can be customized without coding

#### 🔌 **LLM Provider Layer** (`app/providers/llm/`)  
- **`base.py`**: Abstract provider interface for multi-LLM support
- **`anthropic.py`**: Anthropic Claude implementation with streaming support
- **`__init__.py`**: Provider factory and dynamic selection system
- **Built-in error handling**: Graceful fallbacks when AI APIs are unavailable

#### 💬 **Enhanced Chat System** (`app/core/chat.py`)
- **Replaced hardcoded responses** with intelligent LLM-generated answers
- **Maintains full API compatibility** with existing frontend systems
- **Structured response format**: `{reply, panels, sources}` for rich UIs

## 📡 API Endpoints

### Core Endpoints
- **`GET /healthz`** - System and provider health status
- **`GET /tools/portfolio`** - Raw portfolio data (JSON)
- **`POST /chat`** - AI-powered conversational analysis ⭐

### Chat API Example
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Analyze wallet 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
      }
    ],
    "address": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
    "chain": "ethereum"
  }'
```

**Response Format:**
```json
{
  "reply": "🤖 AI-generated natural language response with insights...",
  "panels": {
    "portfolio": {
      "total_value_usd": 1234567.89,
      "tokens": [...],
      "metadata": {...}
    }
  },
  "sources": [
    {"provider": "alchemy", "type": "blockchain_data"},
    {"provider": "coingecko", "type": "price_data"}
  ]
}
```

## 🧪 Testing & Development

### Run Test Suite
```bash
# Test AI agent system
python test_agent_system.py

# Test response style system
python test_style_system.py

# Test API integration  
python -m pytest tests/test_api.py -v

# Test LLM providers
python test_llm_provider.py
```

### CLI Development Tools
```bash
# Interactive chat testing
python cli.py chat

# Portfolio analysis testing
python cli.py portfolio 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045

# Health check all providers
python cli.py health
```

## ⚙️ Configuration

### Environment Variables
```bash
# Blockchain Data (Required)
ALCHEMY_API_KEY=your_key

# AI/LLM Provider (Required)  
ANTHROPIC_API_KEY=your_key
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-sonnet-20241022

# Optional Settings
COINGECKO_API_KEY=optional_key
MAX_TOKENS=4000
TEMPERATURE=0.7
CONTEXT_WINDOW_SIZE=8000
CACHE_TTL_SECONDS=300
MAX_CONCURRENT_REQUESTS=10
```

### LLM Model Options
```python
# Anthropic Claude Models (Recommended)
claude-3-5-sonnet-20241022    # Latest, most capable
claude-3-5-haiku-20241022     # Faster, more economical  
claude-3-opus-20240229        # Most powerful, expensive

# OpenAI Models (Coming Soon)
gpt-4-turbo-preview          # GPT-4 Turbo
gpt-3.5-turbo               # Faster, cheaper
```

## 🔍 System Requirements

- **Python 3.11+** 
- **Required APIs**: Alchemy + Anthropic Claude
- **Optional APIs**: CoinGecko (for enhanced price data)
- **Memory**: ~100MB base + LLM context (varies by usage)
- **Network**: Internet connection for API calls

## 📚 Testing Sample Addresses

Try these addresses to see the AI system in action:

- **Vitalik Buterin**: `0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045` (Large, diverse portfolio)
- **ENS Treasury**: `0xFe89cc7aBB2C4183683ab71653C4cdc9B02D44b7` (DAO treasury)  
- **Empty Wallet**: `0x0000000000000000000000000000000000000001` (Test error handling)

## 🚧 What's Next

### Recently Completed ✅
- **✅ Dynamic Response Styles**: `/style casual`, `/style technical`, `/style brief`, `/style educational` - **IMPLEMENTED!**
- **✅ YAML Persona Customization**: Complete external persona configuration system - **IMPLEMENTED!**
- **✅ Enhanced Token Metadata**: Improved token symbol resolution with fallbacks - **IMPLEMENTED!**

### Planned Enhancements (Phase D - In Progress)
- **Cross-session Memory**: Remember user preferences between conversations
- **Advanced Crypto Knowledge Integration**: Enhanced crypto protocol understanding
- **Conversation Learning**: Adaptive responses based on user interaction patterns
- **DeFi Protocol Integration**: Deep Uniswap, Aave, Compound analysis
- **Market Sentiment Analysis**: Real-time market insights and news integration

### Future Roadmap
- **Multi-chain Support**: Polygon, Base, Arbitrum, Optimism
- **Advanced Analytics**: Risk metrics, correlation analysis, performance tracking
- **Transaction Simulation**: Preview swap/bridge operations before execution  
- **Real-time Updates**: WebSocket integration for live portfolio updates

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Add tests**: Ensure new features have test coverage
4. **Submit a pull request**: With clear description of changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ using FastAPI, Anthropic Claude, and modern Python async patterns**

*Want to see the AI in action? Try: `python cli.py chat` and ask about any Ethereum wallet!* 🚀
