"""Planning module for autonomous agent activities.

This module provides the core infrastructure for:
- Activity planning and execution
- Strategy protocol and implementations
- Policy-based guardrails
- Audit trail for autonomous decisions

Key Components:
    - BaseStrategy: Protocol for implementing strategies
    - TradeIntent: Immutable intent objects from strategies
    - Plan: Collection of validated actions
    - AgentConfig: Agent configuration (ERC-7208 ready)
    - PolicyConstraints: Guardrails for autonomous execution
    - PlanningService: Orchestrates plan creation and execution

Example Usage:
    from app.core.planning import (
        AgentConfig,
        DCAStrategyParams,
        PlanningService,
        PolicyConstraints,
    )

    # Create agent config
    config = AgentConfig(
        agent_id="my-dca-agent",
        type=AgentType.DCA,
        wallets=[WalletConfig(chain_id=1, address="0x...")],
        strategy_params=DCAStrategyParams(
            target_tokens=["ETH", "BTC"],
            amount_per_execution=Decimal("100"),
            allocation={"ETH": 0.6, "BTC": 0.4},
        ).model_dump(),
    )

    # Create and execute plan
    service = PlanningService(token_resolver, activity_registry)
    plan, warnings = await service.create_strategy_plan(config, context)
"""

from .config import (
    AgentConfig,
    AgentStatus,
    AgentType,
    DCAStrategyParams,
    PlanningContext,
    PolicyConfig,
    ScheduleConfig,
    WalletConfig,
)
from .models import (
    Action,
    ApprovalLevel,
    ChainId,
    DecisionLog,
    Plan,
    PlanStatus,
    PolicyConstraints,
    TokenReference,
)
from .protocol import (
    ActionType,
    AmountSpec,
    AmountUnit,
    BaseStrategy,
    PortfolioSnapshot,
    StrategyContext,
    TradeIntent,
)
from .registry import ActivityRegistry, get_activity_registry
from .service import PlanningService, get_planning_service

__all__ = [
    # Protocol
    "ActionType",
    "AmountSpec",
    "AmountUnit",
    "BaseStrategy",
    "PortfolioSnapshot",
    "StrategyContext",
    "TradeIntent",
    # Models
    "Action",
    "ApprovalLevel",
    "ChainId",
    "DecisionLog",
    "Plan",
    "PlanStatus",
    "PolicyConstraints",
    "TokenReference",
    # Config
    "AgentConfig",
    "AgentStatus",
    "AgentType",
    "DCAStrategyParams",
    "PlanningContext",
    "PolicyConfig",
    "ScheduleConfig",
    "WalletConfig",
    # Registry
    "ActivityRegistry",
    "get_activity_registry",
    # Service
    "PlanningService",
    "get_planning_service",
]
