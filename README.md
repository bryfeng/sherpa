# Sherpa - AI-Powered DeFi Portfolio Assistant

An intelligent cryptocurrency portfolio management system powered by Large Language Models (LLM). Features natural language conversations, automated trading strategies, multi-chain support, and comprehensive portfolio insights.

## Overview

Sherpa is a full-stack DeFi assistant that combines:
- **AI Agent**: Natural language interface for portfolio analysis, strategy creation, and execution
- **Strategy Engine**: Automated DCA, rebalancing, and custom trading strategies
- **Multi-chain Support**: Ethereum, Solana, and L2 networks (Arbitrum, Base, Optimism, Polygon)
- **Real-time Data**: Live portfolio tracking, price feeds, and market intelligence
- **Non-custodial Execution**: Phase 1 manual approval flow for strategy execution

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
│  • AI Agent (LangGraph)     │◄───►│  • Users & Wallets          │
│  • Portfolio analysis       │     │  • Conversations & Messages │
│  • Strategy execution       │     │  • Strategies & Executions  │
│  • Tool orchestration       │     │  • Session Keys & Policies  │
│  • LLM providers            │     │  • Real-time subscriptions  │
└──────────────┬──────────────┘     └─────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           External Services                                   │
│  • Alchemy (EVM data)      • Helius (Solana)       • CoinGecko (prices)      │
│  • Relay (bridging)        • Anthropic Claude      • News APIs               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Features

### AI Agent Capabilities

The Sherpa agent can:
- **Analyze portfolios**: "What's in my wallet?" "Show my top holdings"
- **Create strategies**: "Set up a DCA to buy ETH weekly with $100 USDC"
- **Check prices**: "What's the price of ETH?" "Show me trending tokens"
- **Manage strategies**: "Pause my DCA strategy" "Show my active strategies"
- **Get news**: "What's the latest news about Ethereum?"
- **Bridge tokens**: "Bridge 0.1 ETH from mainnet to Base"

### Strategy Types

| Type | Description | Status |
|------|-------------|--------|
| DCA | Dollar-cost averaging with configurable frequency | ✅ Ready |
| Rebalance | Maintain target portfolio allocation | 🔲 Planned |
| Limit Order | Execute when price conditions are met | 🔲 Planned |
| Stop Loss | Automatic sell below threshold | 🔲 Planned |
| Take Profit | Automatic sell above threshold | 🔲 Planned |
| Custom | User-defined strategy logic | 🔲 Planned |

### Execution Flow (Phase 1 - Manual Approval)

```
Strategy Created (draft)
        │
        ▼
Strategy Activated ──► pending_session (no session key)
        │                     │
        │ (with session key)  │ (user creates session key)
        ▼                     ▼
    ┌───────────────────────────────┐
    │          active               │
    │  Scheduler checks due times   │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │    awaiting_approval          │
    │  (Pending Approvals Widget)   │
    └───────────────┬───────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
    [Approve]               [Skip]
        │                       │
        ▼                       ▼
   executing               cancelled
        │
        ▼
   completed / failed
```

## Quick Start

### 1. Setup Environment

```bash
cd sherpa
./setup.sh

# Or manually:
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create `.env` file:

```env
# Required: Blockchain data
ALCHEMY_API_KEY=your_alchemy_key

# Required: AI/LLM provider
ANTHROPIC_API_KEY=your_anthropic_key

# Required: Convex database
CONVEX_URL=https://your-deployment.convex.cloud
CONVEX_DEPLOY_KEY=prod:your-deploy-key

# Optional: Enhanced features
COINGECKO_API_KEY=your_coingecko_key
SOLANA_HELIUS_API_KEY=your_helius_key
CRYPTOPANIC_API_KEY=your_cryptopanic_key

# Optional: LLM configuration
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-sonnet-20241022
MAX_TOKENS=4000
TEMPERATURE=0.7
```

### 3. Start the Server

```bash
python main.py
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 4. Test the System

```bash
# Interactive chat
python cli.py chat

# Portfolio analysis
python cli.py portfolio 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045

# Health check
python cli.py health
```

## API Endpoints

### Chat & Conversations

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | AI-powered conversational interface |
| `/conversations` | GET | List conversations for a wallet |
| `/conversations` | POST | Create new conversation |
| `/conversations/{id}` | PATCH | Update conversation title/archive |

### Portfolio & Tools

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tools/portfolio` | GET | Get portfolio data for an address |
| `/tools/prices/top` | GET | Get top token prices |
| `/tools/prices/trending` | GET | Get trending tokens |
| `/tools/prices/token/chart` | GET | Get price chart data |
| `/tools/defillama/tvl` | GET | Get protocol TVL data |
| `/tools/defillama/current` | GET | Get current TVL |

### Trading

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/swap/quote` | POST | Get swap quote |
| `/tools/relay/quote` | POST | Get bridge quote |

### News

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tools/news` | GET | Get general crypto news |
| `/tools/news/personalized` | GET | Get personalized news |
| `/tools/news/token/{symbol}` | GET | Get token-specific news |

### Authentication

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/auth/nonce` | GET | Get SIWE nonce |
| `/auth/verify` | POST | Verify SIWE signature |
| `/auth/refresh` | POST | Refresh JWT token |
| `/auth/logout` | POST | Logout user |

### Health

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/healthz` | GET | System health check |
| `/healthz/providers` | GET | Provider status |

## Agent Tools

The AI agent has access to these tools when processing requests:

### Portfolio Tools
- `get_portfolio` - Fetch wallet balances and positions
- `get_wallet_history` - Get transaction history
- `get_token_chart` - Get price chart for a token

### Market Tools
- `get_trending_tokens` - Trending tokens from CoinGecko
- `get_tvl_data` - Protocol TVL from DefiLlama
- `get_news` - General crypto news
- `get_personalized_news` - News based on portfolio holdings
- `get_token_news` - News for specific token

### Strategy Tools
- `list_strategies` - List user's strategies
- `get_strategy` - Get strategy details
- `create_strategy` - Create new strategy (DCA, etc.)
- `pause_strategy` - Pause active strategy
- `resume_strategy` - Resume paused strategy
- `stop_strategy` - Stop strategy permanently
- `update_strategy` - Update strategy configuration
- `get_strategy_executions` - Get execution history

### Policy Tools
- `get_risk_policy` - Get user's risk settings
- `update_risk_policy` - Update risk limits
- `check_action_allowed` - Validate action against policies
- `get_system_status` - Get system operational status

## Project Structure

```
sherpa/
├── app/
│   ├── api/                  # API endpoints
│   │   ├── chat.py           # Chat endpoint
│   │   ├── conversations.py  # Conversation management
│   │   ├── tools.py          # Portfolio/price tools
│   │   ├── swap.py           # Swap quotes
│   │   ├── relay.py          # Bridge quotes
│   │   ├── news.py           # News endpoints
│   │   └── auth.py           # SIWE authentication
│   ├── core/
│   │   ├── agent/            # AI agent system
│   │   │   ├── base.py       # Agent orchestrator
│   │   │   ├── graph.py      # LangGraph pipeline
│   │   │   ├── tools.py      # Tool implementations
│   │   │   ├── context.py    # Conversation memory
│   │   │   ├── personas.py   # AI personalities
│   │   │   └── styles.py     # Response styles
│   │   ├── policy/           # Policy engine
│   │   ├── strategy/         # Strategy state machine
│   │   ├── planning/         # Activity planning
│   │   ├── bridge/           # Bridge orchestration
│   │   └── chat.py           # Chat integration
│   ├── providers/
│   │   ├── llm/              # LLM providers (Anthropic, OpenAI)
│   │   └── ...
│   ├── services/             # Business services
│   ├── db/
│   │   └── convex_client.py  # Convex database client
│   └── types/                # Data models
├── personas/                 # YAML persona configs
├── activities/               # YAML activity definitions
├── tests/                    # Test suite
├── config.py                 # Configuration
├── main.py                   # FastAPI app
└── cli.py                    # CLI tools
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
| `LLM_PROVIDER` | `anthropic` | LLM provider (anthropic/openai) |
| `LLM_MODEL` | `claude-3-5-sonnet-20241022` | Model to use |
| `MAX_TOKENS` | `4000` | Max response tokens |
| `TEMPERATURE` | `0.7` | LLM temperature |
| `COINGECKO_API_KEY` | - | CoinGecko API for prices |
| `SOLANA_HELIUS_API_KEY` | - | Helius API for Solana |
| `CRYPTOPANIC_API_KEY` | - | CryptoPanic for news |
| `RELAY_BASE_URL` | `https://api.relay.link` | Relay API URL |
| `CONTEXT_WINDOW_SIZE` | `8000` | Conversation context size |
| `CACHE_TTL_SECONDS` | `300` | Cache TTL |

## Testing

```bash
# Run all tests
python run_all_tests.py

# Unit tests
.venv/bin/python -m pytest tests/ -v

# Specific test modules
.venv/bin/python -m pytest tests/core/agent/ -v
.venv/bin/python -m pytest tests/core/policy/ -v

# Interactive testing
python cli.py chat
```

## AI Personas

Sherpa supports multiple AI personalities:

| Persona | Style | Use Case |
|---------|-------|----------|
| `friendly` | Casual, emoji-friendly | General users |
| `technical` | Data-driven, analytical | DeFi power users |
| `professional` | Formal, comprehensive | Institutional |
| `educational` | Teaching, explanatory | Beginners |

Switch personas in chat: `/persona technical`

## Convex Integration

The backend persists data to Convex for:
- **Conversations**: Chat history with messages
- **Strategies**: Trading strategy configurations
- **Executions**: Strategy execution history
- **Session Keys**: Delegated signing permissions
- **Policies**: User risk preferences

See `frontend/convex/schema.ts` for the complete data model.

## Security Considerations

- **Non-custodial**: Backend never holds private keys
- **Session keys**: Delegated permissions with limits
- **Policy engine**: Risk limits and action validation
- **Manual approval**: Phase 1 requires user signature for all transactions

## Roadmap

- [x] AI chat with portfolio analysis
- [x] Strategy creation and management
- [x] Convex persistence for conversations
- [x] Session key framework
- [x] Phase 1 manual approval flow
- [ ] Phase 2: Smart wallet integration (ERC-4337)
- [ ] Real swap/bridge execution
- [ ] Advanced strategy types
- [ ] Multi-chain strategy coordination

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**Built with FastAPI, LangGraph, Anthropic Claude, and Convex**
