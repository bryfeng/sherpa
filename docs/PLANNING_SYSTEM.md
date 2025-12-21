# Activity Planning System Architecture

> **Knowledge Base Document** - Reference for building autonomous agent capabilities

## Overview

The Activity Planning System enables agents to create and execute multi-step plans for trades and onchain activities. It uses a **hybrid architecture** that combines:

- **YAML definitions** for activity schemas and guardrails (declarative, storable)
- **Python Protocol** for strategy logic (expressive, auditable)
- **JSON-serializable config** for agent state (ERC-7208 compatible)

This design supports both interactive (chat-driven) and autonomous (scheduled) execution while maintaining a clear path to fully onchain autonomous agents.

---

## Architecture Diagram

```
                         User Request
                              │
            ┌─────────────────┴─────────────────┐
            │                                   │
     [Interactive Mode]                 [Autonomous Mode]
     (LangGraph Chat)                   (AgentRuntime)
            │                                   │
            v                                   v
    ┌───────────────┐                  ┌───────────────┐
    │  LangGraph    │                  │  AgentRuntime │
    │  handle_plan  │                  │   Strategy    │
    └───────────────┘                  └───────────────┘
            │                                   │
            └─────────────────┬─────────────────┘
                              v
                    ┌───────────────────┐
                    │  PlanningService  │
                    │  ┌─────────────┐  │
                    │  │TokenResolver│  │
                    │  └─────────────┘  │
                    └───────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            v                 v                 v
     ┌──────────┐      ┌──────────┐      ┌──────────┐
     │SwapMgr   │      │BridgeMgr │      │ Relay/   │
     └──────────┘      └──────────┘      │ Jupiter  │
                                         └──────────┘
```

---

## Key Design Decisions

### Why Hybrid? (YAML + Python + JSON)

| Concern | Pure YAML | Pure Python | Hybrid (Our Choice) |
|---------|-----------|-------------|---------------------|
| Complex logic | ❌ Limited | ✅ Expressive | ✅ Python strategies |
| Onchain storage | ✅ Serializable | ❌ Can't store code | ✅ Config as JSON/CBOR |
| Policy validation | ✅ Declarative | ❌ Arbitrary code | ✅ PolicyConstraints onchain |
| Audit trail | ✅ Easy | ❌ Code changes | ✅ DecisionLog onchain |
| Extensibility | ❌ New syntax | ✅ Import any lib | ✅ New strategy classes |

### What Lives Where

```
ONCHAIN (Future ERC-7208 Data Container)
├── AgentConfig (JSON)          # "Who am I, what wallets, what schedule"
├── PolicyConstraints (JSON)    # "What are my guardrails"
├── DecisionLog[] (JSON)        # "What did I decide and why"
└── State (JSON)                # "Current positions, daily spend"

OFFCHAIN (Python Package)
├── BaseStrategy (Protocol)     # "How do strategies work"
├── DCAStrategy (Class)         # "DCA logic"
├── MomentumStrategy (Class)    # "Momentum logic" (future)
└── FeatureEngine               # "Market data analysis" (future)

YAML (Git-versioned, loaded at startup)
├── activities/swap.yaml        # "What is a swap, what constraints"
├── activities/bridge.yaml      # "What is a bridge"
└── activities/strategies/      # "Default parameters, schemas"
```

---

## File Structure

```
sherpa/
├── activities/                          # YAML Activity Definitions
│   ├── swap.yaml                        # Swap activity schema
│   ├── bridge.yaml                      # Bridge activity schema
│   └── strategies/
│       └── dca.yaml                     # DCA strategy template
│
├── app/core/planning/                   # Core Planning Module
│   ├── __init__.py                      # Package exports
│   ├── protocol.py                      # BaseStrategy, TradeIntent, AmountSpec
│   ├── models.py                        # Plan, Action, PolicyConstraints
│   ├── config.py                        # AgentConfig, DCAStrategyParams
│   ├── registry.py                      # ActivityRegistry (loads YAML)
│   ├── service.py                       # PlanningService
│   └── strategies/
│       ├── __init__.py                  # Strategy registry
│       └── dca.py                       # DCAStrategy implementation
│
└── tests/core/planning/
    └── test_planning.py                 # 41 unit tests
```

---

## Core Components

### 1. BaseStrategy Protocol (`protocol.py`)

The foundation for all strategy implementations:

```python
class BaseStrategy(Protocol):
    id: str
    version: str

    def evaluate(self, ctx: StrategyContext) -> List[TradeIntent]:
        """Produce trade intents - NEVER execute directly."""
        ...

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Return validation errors."""
        ...
```

**Key Principle**: Strategies produce `TradeIntent` objects, not executions. This enables:
- Policy validation before execution
- Audit trails for all decisions
- Future onchain validation

### 2. TradeIntent (`protocol.py`)

Immutable object representing what a strategy wants to do:

```python
@dataclass(frozen=True)
class TradeIntent:
    action_type: ActionType      # SWAP, BRIDGE, STAKE
    chain_id: ChainId            # 1, 8453, "solana"
    token_in: TokenReference
    token_out: TokenReference
    amount: AmountSpec           # value + unit (USD, TOKEN, PERCENT)
    confidence: float            # 0.0 to 1.0
    reasoning: str               # Human-readable for audit
    metadata: Dict[str, Any]     # Strategy-specific data
```

### 3. PolicyConstraints (`models.py`)

Guardrails that can be stored/validated onchain:

```python
@dataclass
class PolicyConstraints:
    max_slippage_bps: int = 100
    per_trade_usd_cap: Decimal = Decimal("300")
    daily_usd_cap: Decimal = Decimal("5000")
    allowed_chains: List[ChainId]
    blocked_tokens: List[str]
    auto_approve_threshold_usd: Decimal = Decimal("100")

    def validate_intent(self, intent: TradeIntent, estimated_usd: Decimal) -> List[str]:
        """Returns policy violations."""

    def get_approval_level(self, estimated_usd: Decimal) -> ApprovalLevel:
        """NONE (auto), CONFIRMATION, or EXPLICIT_APPROVAL."""
```

### 4. AgentConfig (`config.py`)

Agent configuration designed for onchain storage:

```python
class AgentConfig(BaseModel):
    agent_id: str
    type: AgentType              # DCA, MOMENTUM, COPY, YIELD
    wallets: List[WalletConfig]
    policy: PolicyConfig
    strategy_params: Dict[str, Any]
    schedule: ScheduleConfig
    status: AgentStatus

    def to_cbor(self) -> bytes:
        """Serialize for ERC-7208 storage."""
```

### 5. PlanningService (`service.py`)

Main orchestrator for plan creation:

```python
class PlanningService:
    # Interactive mode (from chat)
    async def create_plan_from_intent(
        self,
        intent_text: str,           # "swap 100 USDC to ETH"
        context: PlanningContext,
        policy: Optional[PolicyConstraints] = None,
    ) -> Tuple[Plan, List[str]]:
        ...

    # Autonomous mode (from strategy)
    async def create_strategy_plan(
        self,
        agent_config: AgentConfig,
        context: PlanningContext,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Plan, DecisionLog]:
        ...

    # Lifecycle
    def approve_plan(self, conversation_id: str) -> Optional[Plan]
    def cancel_plan(self, conversation_id: str) -> Optional[Plan]
```

### 6. ActivityRegistry (`registry.py`)

Loads YAML definitions at startup:

```python
registry = get_activity_registry()

# Get definitions
swap_def = registry.get_activity("swap")
dca_def = registry.get_strategy("dca")

# Detect from text
activity = registry.detect_activity("I want to swap ETH")  # Returns "swap"

# Get guardrails (with strategy overrides)
guardrails = registry.get_guardrails("swap", "dca")
```

---

## Adding a New Strategy

### Step 1: Create YAML Template

```yaml
# activities/strategies/momentum.yaml
name: momentum
display_name: "Momentum Trading"
extends: swap

strategy_config:
  schedule:
    type: interval
    interval_seconds: 3600
  parameters:
    lookback_days: { type: int, default: 14 }
    threshold: { type: float, default: 0.1 }
    tokens: { type: token_list, required: true }

guardrails:
  inherit: true
  overrides:
    max_slippage_bps: 75
```

### Step 2: Implement Strategy Class

```python
# app/core/planning/strategies/momentum.py
class MomentumStrategy:
    id = "momentum"
    version = "1.0"

    def evaluate(self, ctx: StrategyContext) -> List[TradeIntent]:
        intents = []
        params = ctx.agent_config.strategy_params

        for token in params["tokens"]:
            momentum = ctx.get_feature(f"momentum_{token}")
            if momentum and momentum > params["threshold"]:
                intents.append(TradeIntent(
                    action_type=ActionType.SWAP,
                    chain_id=ctx.agent_config.get_primary_wallet().chain_id,
                    token_in=self._resolve_source(ctx),
                    token_out=self._resolve_target(token, ctx),
                    amount=AmountSpec.from_usd(params["amount_per_trade"]),
                    confidence=min(momentum, 1.0),
                    reasoning=f"Momentum signal: {momentum:.2f}",
                ))
        return intents

    def validate_config(self, config: Dict) -> List[str]:
        errors = []
        if "tokens" not in config:
            errors.append("tokens is required")
        return errors
```

### Step 3: Register Strategy

```python
# app/core/planning/strategies/__init__.py
from .momentum import MomentumStrategy

_STRATEGY_REGISTRY = {
    "dca": DCAStrategy,
    "momentum": MomentumStrategy,  # Add here
}
```

---

## Usage Examples

### Interactive Mode (Chat)

```python
from app.core.planning import PlanningService, PlanningContext

service = PlanningService(token_resolver=resolver)
context = PlanningContext(
    wallet_address="0x123...",
    chain_id=1,
    conversation_id="conv-abc",
)

# Create plan from natural language
plan, warnings = await service.create_plan_from_intent(
    "swap 100 USDC to ETH",
    context,
)

# User reviews, then approves
if user_approved:
    plan = service.approve_plan("conv-abc")
    # Execute via SwapManager...
```

### Autonomous Mode (AgentRuntime)

```python
from app.core.planning import (
    AgentConfig, AgentType, DCAStrategyParams,
    PlanningService, WalletConfig,
)

# Create agent config
config = AgentConfig(
    agent_id="my-dca-agent",
    type=AgentType.DCA,
    wallets=[WalletConfig(chain_id=1, address="0x123...")],
    strategy_params=DCAStrategyParams(
        source_token="USDC",
        target_tokens=["ETH", "BTC"],
        amount_per_execution=Decimal("100"),
        allocation={"ETH": 0.6, "BTC": 0.4},
    ).model_dump(),
)

# Create plan from strategy
plan, decision_log = await service.create_strategy_plan(
    config, context, market_data
)

# If auto-approved by policy, execute
if plan.status == PlanStatus.APPROVED:
    # Execute...
```

---

## Testing

Run planning module tests:

```bash
cd sherpa
source .venv/bin/activate
python -m pytest tests/core/planning/ -v
```

Current coverage: **41 tests** covering:
- Protocol objects (AmountSpec, TradeIntent)
- Models (TokenReference, PolicyConstraints, Plan)
- Config (AgentConfig, DCAStrategyParams)
- ActivityRegistry (YAML loading, detection)
- DCAStrategy (evaluation, validation)
- PlanningService (plan creation, lifecycle)

---

## Autonomy Progression Roadmap

### Stage 1: Interactive Planning (Current MVP)
```
User ─chat─> Agent ─plan─> User reviews ─approve─> Execute
```
- In-memory plan state
- User approves every plan
- Single-step activities (swap, bridge)

### Stage 2: Policy-Bounded Autonomy (Next)
```
User ─configure─> AgentConfig + Policy ─> Agent auto-executes within guardrails
```
- Redis persistence
- DCA runs autonomously
- Auto-execute if `estimated_usd < policy.auto_approve_threshold`

### Stage 3: Data-Driven Decisions
```
Data feeds ─features─> Decision Engine ─score─> Best candidates ─plan─> Execute
```
- Feature Engine (momentum, liquidity scores)
- Momentum and Yield strategies
- DecisionLog in database

### Stage 4: Multi-Agent Coordination
```
Multiple agents ─coordinate─> Shared budget ─orchestrate─> Parallel execution
```
- Portfolio-level risk management
- Copy-trading

### Stage 5: Onchain State (Future)
```
Agent state ─ERC-7208─> Onchain container ─verify─> Trustless execution
```
- AgentConfig stored onchain
- DecisionLog recorded onchain
- Smart contract validates intents against PolicyConstraints

---

## Pending Implementation

### LangGraph Integration
Add `handle_planning` node to `app/core/agent/graph.py`:

```python
async def handle_planning(state: AgentProcessState) -> AgentProcessState:
    tool_data = dict(state.get('tool_data', {}))

    if not planning_service.is_planning_request(latest_message):
        return {'tool_data': tool_data}

    context = PlanningContext(
        conversation_id=state['conversation_id'],
        wallet_address=tool_data.get('_address'),
        chain_id=state['request'].chain or 1,
    )

    plan, warnings = await planning_service.create_plan_from_intent(
        latest_message, context
    )

    tool_data['planning'] = {
        'plan': plan.to_dict(),
        'warnings': warnings,
    }
    return {'tool_data': tool_data}
```

### API Endpoints
Create `app/api/routes/planning.py`:

```python
router = APIRouter(prefix="/planning")

@router.post("/plans")
async def create_plan(request: CreatePlanRequest) -> PlanSummary

@router.post("/plans/{plan_id}/approve")
async def approve_plan(plan_id: str)

@router.post("/agents")
async def create_agent(config: AgentConfig)

@router.get("/agents/{agent_id}/status")
async def get_agent_status(agent_id: str)
```

---

## Related Documentation

- [TokenResolutionService](./TOKEN_RESOLUTION.md) - Multi-chain token lookup
- [SwapManager](../app/core/swap/) - Swap execution
- [BridgeManager](../app/core/bridge/) - Bridge execution
- [AgentRuntime](../app/agent_runtime/) - Background strategy execution

---

## Changelog

### 2024-12-21 - Initial Implementation
- Created hybrid architecture (YAML + Python + JSON)
- Implemented BaseStrategy Protocol and TradeIntent
- Created PolicyConstraints with validation
- Built AgentConfig for ERC-7208 compatibility
- Implemented DCAStrategy
- Created PlanningService and ActivityRegistry
- Added 41 unit tests
