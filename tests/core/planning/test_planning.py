"""Tests for the planning module.

Tests cover:
- Protocol objects (TradeIntent, AmountSpec)
- Models (TokenReference, PolicyConstraints, Plan)
- Config (AgentConfig, DCAStrategyParams)
- ActivityRegistry (YAML loading)
- PlanningService (plan creation)
- DCAStrategy (strategy evaluation)
"""

from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest

from app.core.planning import (
    Action,
    ActionType,
    AgentConfig,
    AgentStatus,
    AgentType,
    AmountSpec,
    AmountUnit,
    ApprovalLevel,
    DCAStrategyParams,
    Plan,
    PlanningContext,
    PlanningService,
    PlanStatus,
    PolicyConfig,
    PolicyConstraints,
    PortfolioSnapshot,
    ScheduleConfig,
    StrategyContext,
    TokenReference,
    TradeIntent,
    WalletConfig,
    get_activity_registry,
)
from app.core.planning.strategies import DCAStrategy, get_strategy


# =============================================================================
# Protocol Tests
# =============================================================================


class TestAmountSpec:
    """Tests for AmountSpec."""

    def test_from_tokens(self):
        amount = AmountSpec.from_tokens("100.5")
        assert amount.value == Decimal("100.5")
        assert amount.unit == AmountUnit.TOKEN

    def test_from_usd(self):
        amount = AmountSpec.from_usd(50)
        assert amount.value == Decimal("50")
        assert amount.unit == AmountUnit.USD

    def test_from_percent(self):
        amount = AmountSpec.from_percent("25.5")
        assert amount.value == Decimal("25.5")
        assert amount.unit == AmountUnit.PERCENT

    def test_positive_validation(self):
        with pytest.raises(ValueError, match="positive"):
            AmountSpec(value=Decimal("0"), unit=AmountUnit.TOKEN)

    def test_to_dict(self):
        amount = AmountSpec.from_usd("100")
        d = amount.to_dict()
        assert d["value"] == "100"
        assert d["unit"] == "usd"


class TestTradeIntent:
    """Tests for TradeIntent."""

    def test_creation(self):
        token_in = TokenReference(
            chain_id=1,
            address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            symbol="USDC",
            decimals=6,
        )
        token_out = TokenReference(
            chain_id=1,
            address="0x0000000000000000000000000000000000000000",
            symbol="ETH",
            decimals=18,
        )
        intent = TradeIntent(
            action_type=ActionType.SWAP,
            chain_id=1,
            token_in=token_in,
            token_out=token_out,
            amount=AmountSpec.from_usd("100"),
            confidence=0.9,
            reasoning="DCA buy",
        )
        assert intent.action_type == ActionType.SWAP
        assert intent.confidence == 0.9

    def test_confidence_validation(self):
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            TradeIntent(
                action_type=ActionType.SWAP,
                chain_id=1,
                token_in=TokenReference(chain_id=1, address="", symbol="USDC", decimals=6),
                token_out=TokenReference(chain_id=1, address="", symbol="ETH", decimals=18),
                amount=AmountSpec.from_usd("100"),
                confidence=1.5,  # Invalid
            )


# =============================================================================
# Models Tests
# =============================================================================


class TestTokenReference:
    """Tests for TokenReference."""

    def test_creation(self):
        ref = TokenReference(
            chain_id=1,
            address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            symbol="USDC",
            decimals=6,
            name="USD Coin",
            confidence=0.95,
        )
        assert ref.symbol == "USDC"
        assert ref.confidence == 0.95

    def test_to_dict(self):
        ref = TokenReference(
            chain_id=1,
            address="0x123",
            symbol="TEST",
            decimals=18,
        )
        d = ref.to_dict()
        assert d["chain_id"] == 1
        assert d["symbol"] == "TEST"

    def test_from_dict(self):
        data = {
            "chain_id": 8453,
            "address": "0x456",
            "symbol": "BASE",
            "decimals": 18,
        }
        ref = TokenReference.from_dict(data)
        assert ref.chain_id == 8453
        assert ref.symbol == "BASE"


class TestPolicyConstraints:
    """Tests for PolicyConstraints."""

    def test_defaults(self):
        policy = PolicyConstraints()
        assert policy.max_slippage_bps == 100
        assert policy.per_trade_usd_cap == Decimal("300")
        assert 1 in policy.allowed_chains

    def test_validate_intent_passes(self):
        policy = PolicyConstraints(allowed_chains=[1, 8453])
        intent = TradeIntent(
            action_type=ActionType.SWAP,
            chain_id=1,
            token_in=TokenReference(chain_id=1, address="0x1", symbol="USDC", decimals=6),
            token_out=TokenReference(chain_id=1, address="0x2", symbol="ETH", decimals=18),
            amount=AmountSpec.from_usd("50"),
        )
        violations = policy.validate_intent(intent, Decimal("50"))
        assert len(violations) == 0

    def test_validate_intent_chain_violation(self):
        policy = PolicyConstraints(allowed_chains=[1])
        intent = TradeIntent(
            action_type=ActionType.SWAP,
            chain_id=137,  # Not in allowed list
            token_in=TokenReference(chain_id=137, address="0x1", symbol="USDC", decimals=6),
            token_out=TokenReference(chain_id=137, address="0x2", symbol="MATIC", decimals=18),
            amount=AmountSpec.from_usd("50"),
        )
        violations = policy.validate_intent(intent, Decimal("50"))
        assert len(violations) == 1
        assert "not in allowed chains" in violations[0]

    def test_validate_intent_cap_violation(self):
        policy = PolicyConstraints(per_trade_usd_cap=Decimal("100"))
        intent = TradeIntent(
            action_type=ActionType.SWAP,
            chain_id=1,
            token_in=TokenReference(chain_id=1, address="0x1", symbol="USDC", decimals=6),
            token_out=TokenReference(chain_id=1, address="0x2", symbol="ETH", decimals=18),
            amount=AmountSpec.from_usd("200"),
        )
        violations = policy.validate_intent(intent, Decimal("200"))
        assert len(violations) == 1
        assert "exceeds per-trade cap" in violations[0]

    def test_get_approval_level(self):
        policy = PolicyConstraints(
            auto_approve_threshold_usd=Decimal("50"),
            per_trade_usd_cap=Decimal("500"),
        )
        assert policy.get_approval_level(Decimal("25")) == ApprovalLevel.NONE
        assert policy.get_approval_level(Decimal("100")) == ApprovalLevel.CONFIRMATION
        assert policy.get_approval_level(Decimal("1000")) == ApprovalLevel.EXPLICIT_APPROVAL

    def test_to_json_from_json(self):
        policy = PolicyConstraints(
            max_slippage_bps=50,
            per_trade_usd_cap=Decimal("500"),
        )
        json_data = policy.to_json()
        restored = PolicyConstraints.from_json(json_data)
        assert restored.max_slippage_bps == 50
        assert restored.per_trade_usd_cap == Decimal("500")


class TestPlan:
    """Tests for Plan."""

    def test_creation(self):
        plan = Plan(
            plan_id="test-plan-1",
            intents=[],
            actions=[],
            policy=PolicyConstraints(),
            status=PlanStatus.DRAFT,
        )
        assert plan.plan_id == "test-plan-1"
        assert plan.status == PlanStatus.DRAFT

    def test_total_estimated_usd(self):
        action1 = Action(
            action_id="a1",
            action_type=ActionType.SWAP,
            chain_id=1,
            token_in=TokenReference(chain_id=1, address="", symbol="USDC", decimals=6),
            token_out=TokenReference(chain_id=1, address="", symbol="ETH", decimals=18),
            amount=AmountSpec.from_usd("100"),
            estimated_usd=Decimal("100"),
        )
        action2 = Action(
            action_id="a2",
            action_type=ActionType.SWAP,
            chain_id=1,
            token_in=TokenReference(chain_id=1, address="", symbol="USDC", decimals=6),
            token_out=TokenReference(chain_id=1, address="", symbol="BTC", decimals=8),
            amount=AmountSpec.from_usd("50"),
            estimated_usd=Decimal("50"),
        )
        plan = Plan(
            plan_id="test",
            intents=[],
            actions=[action1, action2],
            policy=PolicyConstraints(),
        )
        assert plan.total_estimated_usd() == Decimal("150")

    def test_is_executable(self):
        action = Action(
            action_id="a1",
            action_type=ActionType.SWAP,
            chain_id=1,
            token_in=TokenReference(chain_id=1, address="", symbol="USDC", decimals=6),
            token_out=TokenReference(chain_id=1, address="", symbol="ETH", decimals=18),
            amount=AmountSpec.from_usd("100"),
            estimated_usd=Decimal("100"),
        )
        plan = Plan(
            plan_id="test",
            intents=[],
            actions=[action],
            policy=PolicyConstraints(),
            status=PlanStatus.DRAFT,
        )
        assert not plan.is_executable()

        plan.status = PlanStatus.APPROVED
        assert plan.is_executable()


# =============================================================================
# Config Tests
# =============================================================================


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_creation(self):
        config = AgentConfig(
            agent_id="test-agent-1",
            type=AgentType.DCA,
            wallets=[WalletConfig(chain_id=1, address="0x123")],
        )
        assert config.agent_id == "test-agent-1"
        assert config.type == AgentType.DCA

    def test_get_primary_wallet(self):
        config = AgentConfig(
            agent_id="test",
            type=AgentType.DCA,
            wallets=[
                WalletConfig(chain_id=1, address="0xeth", is_primary=False),
                WalletConfig(chain_id=8453, address="0xbase", is_primary=True),
            ],
        )
        primary = config.get_primary_wallet()
        assert primary.address == "0xbase"

        eth_wallet = config.get_primary_wallet(chain_id=1)
        assert eth_wallet.address == "0xeth"


class TestDCAStrategyParams:
    """Tests for DCAStrategyParams."""

    def test_creation(self):
        params = DCAStrategyParams(
            source_token="USDC",
            target_tokens=["ETH", "BTC"],
            amount_per_execution=Decimal("100"),
            allocation={"ETH": 0.6, "BTC": 0.4},
        )
        assert params.source_token == "USDC"
        assert len(params.target_tokens) == 2

    def test_validate_allocation_valid(self):
        params = DCAStrategyParams(
            source_token="USDC",
            target_tokens=["ETH", "BTC"],
            amount_per_execution=Decimal("100"),
            allocation={"ETH": 0.6, "BTC": 0.4},
        )
        errors = params.validate_allocation()
        assert len(errors) == 0

    def test_validate_allocation_sum_error(self):
        params = DCAStrategyParams(
            source_token="USDC",
            target_tokens=["ETH", "BTC"],
            amount_per_execution=Decimal("100"),
            allocation={"ETH": 0.5, "BTC": 0.3},  # Sum is 0.8
        )
        errors = params.validate_allocation()
        assert len(errors) == 1
        assert "sum to 1.0" in errors[0]


# =============================================================================
# Registry Tests
# =============================================================================


class TestActivityRegistry:
    """Tests for ActivityRegistry."""

    def test_loads_activities(self):
        registry = get_activity_registry()
        activities = registry.list_activities()
        assert "swap" in activities
        assert "bridge" in activities

    def test_loads_strategies(self):
        registry = get_activity_registry()
        strategies = registry.list_strategies()
        assert "dca" in strategies

    def test_get_activity(self):
        registry = get_activity_registry()
        swap = registry.get_activity("swap")
        assert swap is not None
        assert swap["name"] == "swap"
        assert "guardrails" in swap

    def test_get_strategy(self):
        registry = get_activity_registry()
        dca = registry.get_strategy("dca")
        assert dca is not None
        assert dca["name"] == "dca"
        assert "strategy_config" in dca

    def test_detect_activity_swap(self):
        registry = get_activity_registry()
        assert registry.detect_activity("I want to swap ETH for USDC") == "swap"
        assert registry.detect_activity("trade my tokens") == "swap"
        assert registry.detect_activity("buy some ETH") == "swap"

    def test_detect_activity_bridge(self):
        registry = get_activity_registry()
        assert registry.detect_activity("bridge to Base") == "bridge"
        assert registry.detect_activity("move tokens to Arbitrum") == "bridge"

    def test_get_guardrails(self):
        registry = get_activity_registry()
        guardrails = registry.get_guardrails("swap")
        assert "max_slippage_bps" in guardrails

        dca_guardrails = registry.get_guardrails("swap", "dca")
        # DCA overrides slippage
        assert dca_guardrails["max_slippage_bps"] == 50


# =============================================================================
# Strategy Tests
# =============================================================================


class TestDCAStrategy:
    """Tests for DCAStrategy."""

    def test_get_strategy(self):
        strategy = get_strategy("dca")
        assert strategy is not None
        assert strategy.id == "dca"

    def test_validate_config_valid(self):
        strategy = DCAStrategy()
        config = {
            "target_tokens": ["ETH", "BTC"],
            "amount_per_execution": "100",
            "allocation": {"ETH": 0.6, "BTC": 0.4},
        }
        errors = strategy.validate_config(config)
        assert len(errors) == 0

    def test_validate_config_missing_target_tokens(self):
        strategy = DCAStrategy()
        config = {
            "amount_per_execution": "100",
            "allocation": {"ETH": 1.0},
        }
        errors = strategy.validate_config(config)
        assert any("target_tokens" in e for e in errors)

    def test_validate_config_invalid_allocation(self):
        strategy = DCAStrategy()
        config = {
            "target_tokens": ["ETH", "BTC"],
            "amount_per_execution": "100",
            "allocation": {"ETH": 0.3, "BTC": 0.3},  # Sum is 0.6
        }
        errors = strategy.validate_config(config)
        assert any("sum to 1.0" in e for e in errors)

    def test_evaluate_with_params(self):
        params = DCAStrategyParams(
            source_token="USDC",
            target_tokens=["ETH", "BTC"],
            amount_per_execution=Decimal("100"),
            allocation={"ETH": 0.6, "BTC": 0.4},
            default_chain=1,
        )
        strategy = DCAStrategy(params=params)

        # Create context with sufficient balance
        portfolio = PortfolioSnapshot(
            wallet_address="0x123",
            chain_id=1,
            balances={"USDC": Decimal("1000")},
            usd_values={"USDC": Decimal("1000")},
            total_usd=Decimal("1000"),
            timestamp=datetime.utcnow(),
        )

        # We need an AgentConfig for the context
        agent_config = AgentConfig(
            agent_id="test",
            type=AgentType.DCA,
            wallets=[WalletConfig(chain_id=1, address="0x123")],
            strategy_params=params.model_dump(),
        )

        ctx = StrategyContext(
            agent_config=agent_config,
            portfolio=portfolio,
        )

        intents = strategy.evaluate(ctx)
        assert len(intents) == 2  # ETH and BTC

        # Check ETH intent
        eth_intent = next(i for i in intents if i.token_out.symbol == "ETH")
        assert eth_intent.amount.value == Decimal("60")  # 100 * 0.6
        assert eth_intent.amount.unit == AmountUnit.USD

        # Check BTC intent
        btc_intent = next(i for i in intents if i.token_out.symbol == "BTC")
        assert btc_intent.amount.value == Decimal("40")  # 100 * 0.4

    def test_evaluate_insufficient_balance(self):
        params = DCAStrategyParams(
            source_token="USDC",
            target_tokens=["ETH"],
            amount_per_execution=Decimal("100"),
            allocation={"ETH": 1.0},
            default_chain=1,
        )
        strategy = DCAStrategy(params=params)

        # Create context with insufficient balance
        portfolio = PortfolioSnapshot(
            wallet_address="0x123",
            chain_id=1,
            balances={"USDC": Decimal("50")},
            usd_values={"USDC": Decimal("50")},  # Less than 100
            total_usd=Decimal("50"),
            timestamp=datetime.utcnow(),
        )

        agent_config = AgentConfig(
            agent_id="test",
            type=AgentType.DCA,
            wallets=[WalletConfig(chain_id=1, address="0x123")],
            strategy_params=params.model_dump(),
        )

        ctx = StrategyContext(
            agent_config=agent_config,
            portfolio=portfolio,
        )

        intents = strategy.evaluate(ctx)
        assert len(intents) == 0  # Should skip due to insufficient balance


# =============================================================================
# PlanningService Tests
# =============================================================================


class TestPlanningService:
    """Tests for PlanningService."""

    def test_is_planning_request(self):
        service = PlanningService()
        assert service.is_planning_request("swap ETH to USDC")
        assert service.is_planning_request("I want to trade my tokens")
        assert service.is_planning_request("set up a DCA for me")
        assert service.is_planning_request("schedule recurring buys")
        assert not service.is_planning_request("what is the weather")

    @pytest.mark.asyncio
    async def test_create_plan_from_intent(self):
        service = PlanningService()
        context = PlanningContext(
            wallet_address="0x123",
            chain_id=1,
            conversation_id="test-conv-1",
        )

        plan, warnings = await service.create_plan_from_intent(
            "swap 100 USDC to ETH",
            context,
        )

        assert plan is not None
        assert plan.status == PlanStatus.PENDING_APPROVAL
        assert len(plan.intents) == 1
        assert plan.intents[0].token_in.symbol == "USDC"
        assert plan.intents[0].token_out.symbol == "ETH"

    def test_approve_plan(self):
        service = PlanningService()
        plan = Plan(
            plan_id="test-plan",
            intents=[],
            actions=[],
            policy=PolicyConstraints(),
            status=PlanStatus.PENDING_APPROVAL,
            conversation_id="test-conv",
        )
        service._pending_plans["test-conv"] = plan

        approved = service.approve_plan("test-conv")
        assert approved is not None
        assert approved.status == PlanStatus.APPROVED

    def test_cancel_plan(self):
        service = PlanningService()
        plan = Plan(
            plan_id="test-plan",
            intents=[],
            actions=[],
            policy=PolicyConstraints(),
            status=PlanStatus.PENDING_APPROVAL,
            conversation_id="test-conv",
        )
        service._pending_plans["test-conv"] = plan

        cancelled = service.cancel_plan("test-conv")
        assert cancelled is not None
        assert cancelled.status == PlanStatus.CANCELLED
        assert "test-conv" not in service._pending_plans
