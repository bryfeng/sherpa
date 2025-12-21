# Agentic Wallet Python Backend - AI-Powered Chat System

An intelligent cryptocurrency portfolio analysis system powered by Large Language Models (LLM). Features natural language conversations, multiple AI personas, and comprehensive portfolio insights for Ethereum and Solana addresses.

## 🤖 What's New: AI-Powered Chat System

This system has been completely transformed from basic hardcoded responses to a sophisticated AI agent that can:
- **Understand natural language** questions about crypto portfolios
- **Switch between AI personas** (Friendly Guide, Technical Analyst, Professional Advisor, Educational Teacher)  
- **Maintain conversation context** and remember previous interactions
- **Integrate real portfolio data** with intelligent LLM analysis
- **Provide graceful fallbacks** when AI services are unavailable

## ✨ Key Features

- **LLM-first agent** that keeps conversation context, swaps personas on demand, and blends tool outputs with natural language answers.
- **Multi-chain aware** portfolio tooling with first-class support for Ethereum and Solana (Helius-backed) wallets.
- **Portfolio intelligence** with on-chain balances, price feeds, and AI commentary in the same response payload.
- **Market signals + perps simulator** delivering trending tokens, bridge-ready quotes, and risk-checked strategies without live trading.
- **Developer-friendly tooling**: FastAPI endpoints, pluggable LLM providers, deterministic mocks, and a lean CLI/testing workflow.

## 🚀 Quick Start

### 1. Setup Environment
```bash
cd sherpa
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

# Optional: Solana balances (Helius)
SOLANA_HELIUS_API_KEY=your_helius_key_here

# Optional: History exports + caching
AWS_S3_EXPORT_BUCKET=sherpa-history-exports
REDIS_URL=redis://localhost:6379/0

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

# Summarize wallet history (30-day window or latest 200 tx with --limit)
python cli.py wallet-history 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045 ethereum 30
python cli.py wallet-history 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045 ethereum --limit 200
```

### 4. Start API Server
```bash
python main.py
# Visit http://localhost:8000/docs for API documentation
```

### 5. Explore Market Trends (Optional)
```bash
# Returns Relay-ready trending tokens filtered to EVM chains
curl "http://localhost:8000/tools/prices/trending?limit=10"
```

## 🎭 AI Persona System

- Four built-in personas (`friendly`, `technical`, `professional`, `educational`) switchable via `/persona ...`.
- Response styles like `/style brief`, `/style casual`, or `/style technical` layer tone control on top.
- Persona definitions live in `personas/*.yaml`; edit and save to customize without restarts. Full docs: `docs/persona_customization.md`.

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
│  • Relay Aggregator  │    │  • Source Attribution│    │  • Persona Configs  │
│  • Portfolio Tools  │    │                      │    │                     │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
```

### Key Components

#### 🤖 **Agent System** (`app/core/agent/`)
- **`base.py`**: Core Agent orchestrator managing the entire conversation flow
- **`graph.py`**: LangGraph-powered pipeline coordinating style, tooling, and LLM steps
- **`personas.py`**: AI personality manager that loads from external YAML files
- **`styles.py`**: Dynamic response style system for customizing communication format
- **`context.py`**: Conversation memory, history management, and context compression
- **Smart tool integration**: Seamlessly incorporates portfolio data into LLM context

#### 🌉 **Bridge Subsystem** (`app/core/bridge/`)
- **`manager.py`**: Dedicated bridge orchestrator handling intent parsing and Relay quotes/transactions
- **`constants.py`**: Chain metadata, keyword aliases, and API defaults shared across bridge flows
- **`models.py`**: Typed state/result containers used by the bridge manager

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

#### 📋 **Activity Planning System** (`app/core/planning/`)
- **Hybrid architecture**: YAML schemas + Python strategies + JSON-serializable config
- **`protocol.py`**: `BaseStrategy` Protocol, `TradeIntent`, `AmountSpec` for strategy abstraction
- **`models.py`**: `Plan`, `Action`, `PolicyConstraints` for execution management
- **`config.py`**: `AgentConfig`, `DCAStrategyParams` - ERC-7208 ready for onchain storage
- **`service.py`**: `PlanningService` orchestrates plan creation from chat or autonomous strategies
- **`registry.py`**: `ActivityRegistry` loads YAML definitions from `activities/`
- **`strategies/dca.py`**: Dollar Cost Averaging implementation
- **Full documentation**: See `docs/PLANNING_SYSTEM.md` for architecture details and extension guide

#### 🗂️ **YAML Activity Definitions** (`activities/`)
- **`swap.yaml`**: Swap activity schema with guardrails
- **`bridge.yaml`**: Cross-chain bridge activity
- **`strategies/dca.yaml`**: DCA strategy template with schedule and allocation config
- **Extensible**: Add new activities by creating YAML files - automatically loaded at startup

## 📡 API Endpoints

### Core Endpoints
- **`GET /healthz`** - System and provider health status
- **`GET /tools/portfolio`** - Raw portfolio data (JSON)
- **`POST /chat`** - AI-powered conversational analysis ⭐
- **`GET /conversations?address=0x…`** — List recent conversations for a wallet
- **`POST /conversations`** — Create a new conversation `{ address, title? }`
- **`PATCH /conversations/{id}`** — Update `title` or `archived`

### Tools Endpoints
- **`GET /tools/defillama/tvl`** — DefiLlama TVL timeseries for a protocol
  - Query params: `protocol=uniswap` `range=7d|30d`
  - Response: `{ timestamps: number[], tvl: number[], source: 'defillama' }`
  - Example:
    ```bash
    curl "http://localhost:8000/tools/defillama/tvl?protocol=uniswap&range=7d"
    ```

- **`GET /tools/defillama/current`** — Latest TVL point for a protocol
  - Query params: `protocol=uniswap`
  - Response: `{ timestamp: number, tvl: number, source: 'defillama' }`

- **`POST /tools/relay/quote`** — Bridge quote via Relay
  - Body fields (required unless noted):
    - `user` (address of the initiating wallet, also used as the recipient unless overridden)
    - `originChainId` / `destinationChainId`
    - `originCurrency` / `destinationCurrency` (token addresses, use `0x000…0000` for native assets)
    - `amount` (string, smallest units)
    - Optional flags: `referrer`, `useExternalLiquidity`, `useDepositAddress`, `topupGas`
  - Response: `{ success: boolean, quote: {...raw relay response...} }`
  - Example (0.001 ETH mainnet → Base):
    ```bash
    curl -X POST "http://localhost:8000/tools/relay/quote" \
      -H "Content-Type: application/json" \
      -d '{
        "user": "0x50ac5CFcc81BB0872e85255D7079F8a529345D16",
        "originChainId": 1,
        "destinationChainId": 8453,
        "originCurrency": "0x0000000000000000000000000000000000000000",
        "destinationCurrency": "0x0000000000000000000000000000000000000000",
        "recipient": "0x50ac5CFcc81BB0872e85255D7079F8a529345D16",
        "tradeType": "EXACT_INPUT",
        "amount": "1000000000000000",
        "referrer": "sherpa.chat",
        "useExternalLiquidity": false,
        "useDepositAddress": false,
        "topupGas": false
      }'
    ```
  - The chat agent persists bridge context (chains, amount, wallet) so follow-up prompts like “refresh bridge quote” reuse the pending request automatically.

- **`GET /tools/polymarket/markets`** — Trending/search markets (MVP)
  - Query params: `query=` `limit=5`
  - Response: `{ markets: Array<{ id, question, yesPrice, noPrice, url }> }`
  - Config: Set `POLYMARKET_BASE_URL` to use a real API; falls back to a small mock when unset.
  - Example:
    ```bash
    curl "http://localhost:8000/tools/polymarket/markets?query=ETH&limit=5"
    ```

### Swap (MVP)
- **`POST /swap/quote`** — Simple swap quote estimator (stub)
  - Body: `{ token_in: 'ETH', token_out: 'USDC', amount_in: 1, slippage_bps?: 50 }`
  - Response: `{ success, from_token, to_token, amount_in, amount_out_est, price_in_usd, price_out_usd, fee_est, slippage_bps, route, sources, warnings }`
  - Notes: Uses a static price table and 0.3% fee + slippage reserve. Intended to unblock UI while aggregator integration (0x/1inch) is pending.
  - Example:
    ```bash
    curl -X POST http://localhost:8000/swap/quote \
      -H "Content-Type: application/json" \
      -d '{
        "token_in": "ETH",
        "token_out": "USDC",
        "amount_in": 1,
        "slippage_bps": 50
      }'
    ```

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

### Conversation IDs & Wallet-Scoped Sessions
- Backend now associates conversations to a wallet address (if provided) and reuses the most recent active conversation for that address when no `conversation_id` is sent.
- ID format: `{lowercased_address}-{shortid}` (or `guest-{shortid}` when no wallet).
- Always echo `conversation_id` in every `/chat` response.
- In-memory only for MVP; old conversations auto-clean after 7+ days of inactivity.

Frontend localStorage keys:
- Per-address storage: `sherpa.conversation_id:{address}`; guest sessions use `sherpa.conversation_id:guest`.
- On wallet switch, the frontend loads the stored ID for that address and sends it with the next message; if unknown/expired, the backend returns a fresh ID which the frontend saves.

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

### Run All Tests (Sequential)
Use the consolidated runner to execute unit tests, start the API, run live API tests, then stop the server:
```bash
cd sherpa
python run_all_tests.py            # defaults to 127.0.0.1:8000
# or specify host/port
python run_all_tests.py --host 0.0.0.0 --port 8000
```
This runner:
- Runs `test_agent_system.py`, `test_style_system.py`
- Starts `uvicorn app.main:app`
- Runs `tests/test_api.py`, `tests/test_conversations_api.py`
- Terminates the server and prints a PASS/FAIL summary

### CLI Development Tools
```bash
# Interactive chat testing
python cli.py chat

# Portfolio analysis testing
python cli.py portfolio 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045

# Health check all providers
python cli.py health

# Smoke test the public Relay integration
python -m sherpa.tests.test_relay_public
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
RELAY_BASE_URL=https://api.relay.link    # Override the default relay API host
MAX_TOKENS=4000
TEMPERATURE=0.7
CONTEXT_WINDOW_SIZE=8000
CACHE_TTL_SECONDS=300
MAX_CONCURRENT_REQUESTS=10

# Optional: Tools
POLYMARKET_BASE_URL=https://api.your-polymarket-proxy.example   # If unset, markets endpoint uses a small mock
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
- **Optional APIs**: CoinGecko (price data), Relay (bridge quotes)
- **Memory**: ~100MB base + LLM context (varies by usage)
- **Network**: Internet connection for API calls

## 📚 Testing Sample Addresses

Try these addresses to see the AI system in action:

- **Vitalik Buterin**: `0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045` (Large, diverse portfolio)
- **ENS Treasury**: `0xFe89cc7aBB2C4183683ab71653C4cdc9B02D44b7` (DAO treasury)  
- **Empty Wallet**: `0x0000000000000000000000000000000000000001` (Test error handling)

## 🚧 What's Next

- Ship cross-session memory so the agent remembers wallet context between chats.
- Broaden protocol/domain knowledge and surface richer market sentiment signals.
- Bring multi-chain coverage and deeper analytics once the above foundations are stable.

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
