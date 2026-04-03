# Sherpa - AI-Powered DeFi Portfolio Assistant

An intelligent cryptocurrency portfolio management system powered by Large Language Models. Features natural language conversations, automated trading strategies, multi-chain support, and comprehensive portfolio insights.

## Overview

Sherpa is a full-stack DeFi assistant that combines:
- **AI Agent**: Simple tool-calling loop with multi-provider LLM support (no framework overhead)
- **Strategy Engine**: Automated DCA, copy trading, and extensible strategy types
- **Multi-chain Support**: Ethereum, Solana, and L2 networks (Arbitrum, Base, Optimism, Polygon)
- **Real-time Data**: Live portfolio tracking, price feeds, and market intelligence
- **Non-custodial Execution**: ERC-4337 smart sessions, Rhinestone intents, Jupiter (Solana)
- **3-Layer Policy Engine**: System → Session → Risk policy enforcement on every action

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Frontend (React)                                │
│  • Workspace widgets    • Chat interface    • Strategy management            │
│  • Pending approvals    • Portfolio views   • Session key management         │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
┌─────────────────────────────┐     ┌─────────────────────────────┐
│   Backend (FastAPI/Python)  │     │   Database (Convex)         │
│                             │     │                             │
│  • AI Agent (ReAct loop)    │◄───►│  • Users & Wallets          │
│  • LLM Adapter layer        │     │  • Conversations & Messages │
│  • Tool registry (20+)      │     │  • Strategies & Executions  │
│  • Policy engine             │     │  • Session Keys & Policies  │
│  • Selective tool loading   │     │  • Real-time subscriptions  │
└──────────────┬──────────────┘     └─────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           External Services                                   │
│  • Alchemy (EVM data)      • Helius (Solana)       • CoinGecko (prices)      │
│  • Relay (bridging)        • Anthropic Claude      • News APIs               │
│  • Jupiter (Solana swaps)  • Rhinestone (intents)  • Polymarket              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Agent System

The agent uses a simple ReAct tool-calling loop — no framework (LangGraph was removed). The LLM decides which tools to call, results are fed back, and it loops until it has a final response.

### LLM Provider Support

The adapter pattern (`app/providers/llm/`) supports multiple providers:

| Provider | Adapter | Features |
|----------|---------|----------|
| Anthropic Claude | `anthropic.py` | Native tool calling, prompt caching (`cache_control`), extended thinking |
| Z AI / GLM-4 | `zai.py` | OpenAI-compatible format |

Adding a new provider = one file implementing the `LLMProvider` base class.

### Selective Tool Loading

Tools are organized into groups and loaded based on query relevance — reducing input tokens by 50-80%:

| Group | Tools | Trigger |
|-------|-------|---------|
| Portfolio | `get_portfolio`, `get_wallet_history`, `get_token_chart` | wallet, balance, portfolio |
| Market | `get_trending_tokens`, `get_tvl_data` | trending, tvl, defi |
| Trading | `get_swap_quote`, `get_bridge_quote` | swap, bridge, trade |
| Strategy | `create_strategy`, `pause_strategy`, etc. | dca, strategy, automate |
| News | `get_news`, `get_personalized_news` | news, latest, update |
| Policy | `get_risk_policy`, `check_action_allowed` | risk, policy, limit |

### AI Personas

| Persona | Style | Use Case |
|---------|-------|----------|
| `friendly` | Casual, conversational | General users |
| `technical` | Data-driven, analytical | DeFi power users |
| `professional` | Formal, comprehensive | Institutional |
| `educational` | Teaching, explanatory | Beginners |

## Features

### Strategy Types

| Type | Description | Status |
|------|-------------|--------|
| DCA | Dollar-cost averaging with configurable frequency | ✅ Ready |
| Copy Trading | Mirror leader wallets with sizing controls | ✅ Ready |
| Rebalance | Maintain target portfolio allocation | 🔲 Planned |
| Limit Order | Execute when price conditions are met | 🔲 Planned |
| Stop Loss | Automatic sell below threshold | 🔲 Planned |

### Execution Pipeline

**EVM**: Relay API → unsigned tx → user signs → TransactionExecutor → monitor confirmation → log to Convex
**Solana**: Jupiter quote → Solana executor → confirmation
**Smart Sessions**: Rhinestone intent submission for autonomous DCA execution

### Policy Engine (3 layers)

| Layer | Checks |
|-------|--------|
| **System** | Emergency stop, maintenance mode, blocked contracts/tokens/chains |
| **Session** | Time bounds, value limits (per-tx, daily, total), action permissions |
| **Risk** | Max position %, max slippage, max daily tx USD, max single tx USD |

## Quick Start

### 1. Setup Environment

```bash
cd sherpa/backend
./setup.sh

# Or manually:
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create `.env` file (see `.env.example`):

```env
# Required
ALCHEMY_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
CONVEX_URL=https://your-deployment.convex.cloud
CONVEX_DEPLOY_KEY=prod:your-deploy-key

# Optional
COINGECKO_API_KEY=your_key
SOLANA_HELIUS_API_KEY=your_key
CRYPTOPANIC_API_KEY=your_key
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-20250514
```

### 3. Start the Server

```bash
python main.py
# API at http://localhost:8000
# Docs at http://localhost:8000/docs
```

## API Endpoints

### Chat & Conversations

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | AI-powered conversational interface |
| `/chat/stream` | POST | SSE streaming chat |
| `/conversations` | GET | List conversations for a wallet |
| `/conversations` | POST | Create new conversation |
| `/conversations/{id}` | PATCH | Update conversation title/archive |

### Portfolio & Tools

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tools/portfolio` | GET | Portfolio data for an address |
| `/tools/prices/top` | GET | Top token prices |
| `/tools/prices/trending` | GET | Trending tokens |
| `/tools/prices/token/chart` | GET | Price chart data |
| `/tools/defillama/tvl` | GET | Protocol TVL data |

### Trading

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/swap/quote` | POST | Swap quote |
| `/tools/relay/quote` | POST | Bridge quote |
| `/perps/simulate` | POST | Perps simulation |

### Strategies

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/dca` | POST | Create DCA strategy |
| `/dca/{id}/activate` | POST | Activate with session key |
| `/dca/{id}/pause` | POST | Pause |
| `/dca/{id}/resume` | POST | Resume |
| `/dca/{id}/stop` | POST | Stop/complete |
| `/copy-trading/relationships` | POST | Create copy relationship |
| `/copy-trading/leaderboard` | GET | Top leaders |

### Smart Accounts & Permissions

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/smart-accounts/{addr}/balances` | GET | Unified multi-chain balances |
| `/smart-accounts/gas/estimate` | GET | USDC gas estimation |
| `/permissions/request` | POST | Request permission grant |
| `/permissions/confirm` | POST | Confirm on-chain grant |

### Authentication

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/auth/nonce` | POST | Get wallet sign-in nonce |
| `/auth/verify` | POST | Verify signature (EVM SIWE / Solana) |
| `/auth/refresh` | POST | Refresh JWT token |
| `/auth/logout` | POST | Logout |

## Project Structure

```
backend/
├── app/
│   ├── api/                  # 20 route files
│   ├── core/
│   │   ├── agent/            # AI agent system
│   │   │   ├── base.py       # Agent orchestrator + ReAct loop
│   │   │   ├── context.py    # Conversation memory
│   │   │   ├── events.py     # Agent event system
│   │   │   ├── panels.py     # Response panels
│   │   │   ├── personas.py   # YAML-driven personas
│   │   │   ├── styles.py     # Response formatting
│   │   │   └── tools/        # Tool registry (9 domain modules)
│   │   │       ├── registry.py
│   │   │       ├── groups.py     # Selective tool loading
│   │   │       ├── trading.py
│   │   │       ├── strategy.py
│   │   │       ├── portfolio.py
│   │   │       ├── market_data.py
│   │   │       ├── news.py
│   │   │       ├── policy.py
│   │   │       ├── polymarket.py
│   │   │       └── copy_trading.py
│   │   ├── execution/        # TX execution (EVM, Solana, ERC-4337)
│   │   ├── swap/             # Swap orchestration
│   │   ├── bridge/           # Cross-chain bridging
│   │   ├── strategies/dca/   # DCA engine
│   │   ├── copy_trading/     # Copy trading system
│   │   ├── policy/           # 3-layer policy engine
│   │   ├── recovery/         # Error recovery + circuit breaker
│   │   ├── strategy/         # Strategy state machine (10 states)
│   │   ├── planning/         # Activity planning
│   │   └── wallet/           # Smart sessions + Swig
│   ├── providers/
│   │   ├── llm/              # LLM adapters (Anthropic, Z AI)
│   │   │   ├── base.py       # Abstract LLMProvider
│   │   │   ├── anthropic.py  # Claude with cache_control
│   │   │   ├── zai.py        # OpenAI-compatible
│   │   │   └── model_catalog.py  # Convex-backed model registry
│   │   ├── alchemy.py        # EVM indexing
│   │   ├── coingecko.py      # Prices + charts
│   │   ├── jupiter.py        # Solana swaps
│   │   ├── relay.py          # Cross-chain bridging
│   │   ├── rhinestone.py     # Smart session intents
│   │   └── polymarket/       # Prediction markets
│   ├── services/             # Business services
│   │   ├── token_resolution.py   # 10-source token lookup
│   │   ├── unified_balance.py    # Multi-chain aggregation
│   │   ├── news_fetcher/         # News aggregation
│   │   ├── token_catalog/        # Token taxonomy
│   │   └── events/               # Webhook processing
│   ├── auth/                 # Wallet auth (SIWE + Solana)
│   ├── db/                   # Convex client
│   ├── middleware/            # Rate limiting
│   ├── workers/              # Background tasks
│   └── types/                # Pydantic models
├── personas/                 # YAML persona configs
├── activities/               # YAML activity definitions
├── tests/                    # 558+ tests
├── main.py                   # FastAPI app (lifespan-based)
└── cli.py                    # CLI tools
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -x -q

# Skip execution tests (require eth-account)
python -m pytest tests/ -x -q --ignore=tests/core/execution

# Specific modules
python -m pytest tests/core/policy/ -v
python -m pytest tests/core/strategies/ -v

# Interactive testing
python cli.py chat
```

## Configuration Reference

### Required Environment Variables

| Variable | Description |
|----------|-------------|
| `ALCHEMY_API_KEY` | Alchemy API key for EVM data |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude |
| `CONVEX_URL` | Convex deployment URL |
| `CONVEX_DEPLOY_KEY` | Convex deploy key |

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `anthropic` | LLM provider |
| `LLM_MODEL` | `claude-sonnet-4-20250514` | Model to use |
| `MAX_TOKENS` | `4000` | Max response tokens |
| `TEMPERATURE` | `0.7` | LLM temperature |
| `COINGECKO_API_KEY` | - | CoinGecko API for prices |
| `SOLANA_HELIUS_API_KEY` | - | Helius API for Solana |
| `CRYPTOPANIC_API_KEY` | - | CryptoPanic for news |
| `REDIS_URL` | - | Redis for shared state (optional) |
| `RELAY_BASE_URL` | `https://api.relay.link` | Relay API URL |

## Security

- **Non-custodial**: Backend never holds private keys
- **Session keys**: Delegated permissions with spending limits and expiry
- **Policy engine**: 3-layer validation on every action
- **JWT auth**: SIWE (EVM) + Solana sign-in, HS256 tokens

## Roadmap

- [x] AI chat with portfolio analysis
- [x] Strategy creation and management (DCA, copy trading)
- [x] Convex persistence
- [x] Session key framework
- [x] Drop LangGraph — simple ReAct loop with LLM adapters
- [x] Selective tool loading + prompt caching
- [x] Split tool monolith into domain modules
- [ ] Phase 2: Real swap/bridge execution
- [ ] Redis for shared state + horizontal scaling
- [ ] Strategy execution worker process
- [ ] Advanced strategy types (rebalance, limit orders)
- [ ] Multi-chain strategy coordination

---

**Built with FastAPI, Anthropic Claude, and Convex**
