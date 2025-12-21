"""PlanningService - Orchestrates plan creation and management.

This module provides:
- PlanningService: Main service for creating and managing plans
- Intent parsing from natural language
- Token resolution integration
- Policy validation
- Plan lifecycle management

The PlanningService bridges:
- Interactive mode (LangGraph chat)
- Autonomous mode (AgentRuntime strategies)
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from .config import AgentConfig, PlanningContext
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
from .protocol import ActionType, AmountSpec, AmountUnit, StrategyContext, TradeIntent
from .registry import ActivityRegistry, get_activity_registry

if TYPE_CHECKING:
    from ...services.token_resolution import TokenResolutionService

logger = logging.getLogger(__name__)


# Patterns for parsing amounts
AMOUNT_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*"
    r"(usd|dollars?|usdc|eth|sol|btc|%|percent)?",
    re.IGNORECASE,
)

# Patterns for parsing token references
TOKEN_PATTERN = re.compile(
    r"\b(swap|buy|sell|convert|trade)\b\s+"
    r"(?:(\d+(?:\.\d+)?)\s*)?"
    r"(\$?\w+)\s+"
    r"(?:to|for|into)\s+"
    r"(\$?\w+)",
    re.IGNORECASE,
)


class PlanningService:
    """Service for creating and managing execution plans.

    The PlanningService is the main entry point for plan creation:
    - Parses natural language intents
    - Resolves tokens
    - Validates against policy
    - Manages plan lifecycle

    Usage:
        service = PlanningService(token_resolver)

        # Interactive mode (from chat)
        plan, warnings = await service.create_plan_from_intent(
            "swap 100 USDC to ETH",
            context,
        )

        # Autonomous mode (from strategy)
        plan, decision_log = await service.create_strategy_plan(
            agent_config,
            context,
            market_data,
        )
    """

    def __init__(
        self,
        token_resolver: Optional[TokenResolutionService] = None,
        activity_registry: Optional[ActivityRegistry] = None,
    ):
        """Initialize the PlanningService.

        Args:
            token_resolver: Token resolution service
            activity_registry: Activity definition registry
        """
        self._token_resolver = token_resolver
        self._activity_registry = activity_registry or get_activity_registry()
        self._pending_plans: Dict[str, Plan] = {}  # conversation_id -> Plan

    def set_token_resolver(self, resolver: TokenResolutionService) -> None:
        """Set the token resolver (for lazy initialization)."""
        self._token_resolver = resolver

    # -------------------------------------------------------------------------
    # Plan Detection
    # -------------------------------------------------------------------------

    def is_planning_request(self, text: str) -> bool:
        """Check if text indicates a planning request.

        Args:
            text: User input text

        Returns:
            True if this looks like a planning request
        """
        # Check for activity keywords
        activity = self._activity_registry.detect_activity(text)
        if activity:
            return True

        # Check for explicit planning keywords
        planning_keywords = [
            "plan",
            "schedule",
            "automate",
            "dca",
            "dollar cost",
            "recurring",
        ]
        text_lower = text.lower()
        return any(kw in text_lower for kw in planning_keywords)

    # -------------------------------------------------------------------------
    # Interactive Mode - Plan from Intent
    # -------------------------------------------------------------------------

    async def create_plan_from_intent(
        self,
        intent_text: str,
        context: PlanningContext,
        policy: Optional[PolicyConstraints] = None,
    ) -> Tuple[Plan, List[str]]:
        """Create a plan from a natural language intent.

        This is the main entry point for interactive mode (chat).

        Args:
            intent_text: Natural language description of what to do
            context: Planning context with wallet, chain, etc.
            policy: Optional policy constraints (uses defaults if not provided)

        Returns:
            Tuple of (Plan, list of warnings)
        """
        warnings: List[str] = []
        policy = policy or context.get_default_policy()

        # Detect activity type
        activity_name = self._activity_registry.detect_activity(intent_text)
        if not activity_name:
            # Default to swap if we detect token patterns
            if TOKEN_PATTERN.search(intent_text):
                activity_name = "swap"
            else:
                raise ValueError(
                    "Could not detect activity type from intent. "
                    "Try: 'swap X to Y', 'bridge X to chain', etc."
                )

        # Parse intent
        parsed = self._parse_intent(intent_text, activity_name)

        # Resolve tokens
        intents = await self._build_intents(parsed, context, activity_name)
        if not intents:
            raise ValueError("Could not build any valid intents from the request")

        # Validate and build actions
        actions: List[Action] = []
        for intent in intents:
            violations = policy.validate_intent(intent, Decimal("0"))  # TODO: estimate USD
            if violations:
                warnings.extend(violations)
                continue

            action = await self._intent_to_action(intent, context)
            if action:
                actions.append(action)

        # Calculate total estimated USD
        total_usd = sum((a.estimated_usd for a in actions), Decimal("0"))

        # Determine approval level
        approval_level = policy.get_approval_level(total_usd)

        # Create plan
        plan = Plan(
            plan_id=str(uuid.uuid4()),
            intents=intents,
            actions=actions,
            policy=policy,
            status=PlanStatus.PENDING_APPROVAL,
            approval_level=approval_level,
            conversation_id=context.conversation_id,
            warnings=warnings,
        )

        # Store as pending
        if context.conversation_id:
            self._pending_plans[context.conversation_id] = plan

        return plan, warnings

    def _parse_intent(self, text: str, activity_name: str) -> Dict[str, Any]:
        """Parse natural language intent into structured data.

        Args:
            text: User input text
            activity_name: Detected activity type

        Returns:
            Parsed intent data
        """
        parsed: Dict[str, Any] = {"activity": activity_name, "raw_text": text}

        # Try to extract swap pattern
        match = TOKEN_PATTERN.search(text)
        if match:
            action_word, amount, token_in, token_out = match.groups()
            parsed["token_in"] = token_in.lstrip("$")
            parsed["token_out"] = token_out.lstrip("$")
            if amount:
                parsed["amount"] = amount

        # Try to extract amount with unit
        amount_match = AMOUNT_PATTERN.search(text)
        if amount_match and "amount" not in parsed:
            value, unit = amount_match.groups()
            parsed["amount"] = value
            if unit:
                unit_lower = unit.lower()
                if unit_lower in ("usd", "dollar", "dollars", "usdc"):
                    parsed["amount_unit"] = "usd"
                elif unit_lower in ("%", "percent"):
                    parsed["amount_unit"] = "percent"
                else:
                    parsed["amount_unit"] = "token"

        return parsed

    async def _build_intents(
        self,
        parsed: Dict[str, Any],
        context: PlanningContext,
        activity_name: str,
    ) -> List[TradeIntent]:
        """Build TradeIntent objects from parsed data.

        Args:
            parsed: Parsed intent data
            context: Planning context
            activity_name: Activity type

        Returns:
            List of TradeIntent objects
        """
        intents: List[TradeIntent] = []

        if activity_name == "swap":
            intent = await self._build_swap_intent(parsed, context)
            if intent:
                intents.append(intent)

        elif activity_name == "bridge":
            intent = await self._build_bridge_intent(parsed, context)
            if intent:
                intents.append(intent)

        return intents

    async def _build_swap_intent(
        self,
        parsed: Dict[str, Any],
        context: PlanningContext,
    ) -> Optional[TradeIntent]:
        """Build a swap intent from parsed data."""
        token_in_str = parsed.get("token_in")
        token_out_str = parsed.get("token_out")

        if not token_in_str or not token_out_str:
            return None

        # Resolve tokens
        token_in = await self._resolve_token(token_in_str, context.chain_id)
        token_out = await self._resolve_token(token_out_str, context.chain_id)

        if not token_in or not token_out:
            return None

        # Parse amount
        amount_value = parsed.get("amount", "0")
        amount_unit_str = parsed.get("amount_unit", "token")

        if amount_unit_str == "usd":
            amount = AmountSpec.from_usd(amount_value)
        elif amount_unit_str == "percent":
            amount = AmountSpec.from_percent(amount_value)
        else:
            amount = AmountSpec.from_tokens(amount_value)

        return TradeIntent(
            action_type=ActionType.SWAP,
            chain_id=context.chain_id,
            token_in=token_in,
            token_out=token_out,
            amount=amount,
            confidence=1.0,
            reasoning=f"Swap {amount_value} {token_in.symbol} to {token_out.symbol}",
        )

    async def _build_bridge_intent(
        self,
        parsed: Dict[str, Any],
        context: PlanningContext,
    ) -> Optional[TradeIntent]:
        """Build a bridge intent from parsed data."""
        # Bridge parsing is more complex - simplified for now
        return None

    async def _resolve_token(
        self,
        token_str: str,
        chain_id: ChainId,
    ) -> Optional[TokenReference]:
        """Resolve a token string to a TokenReference.

        Args:
            token_str: Token symbol or address
            chain_id: Chain to resolve on

        Returns:
            TokenReference or None if not found
        """
        if not self._token_resolver:
            # Fallback: create a basic reference
            return TokenReference(
                chain_id=chain_id,
                address="",
                symbol=token_str.upper(),
                decimals=18,
                confidence=0.5,
            )

        # Use token resolution service
        results = await self._token_resolver.resolve(
            query=token_str,
            chains=[chain_id] if isinstance(chain_id, int) else None,
            top_k=1,
        )

        if results:
            result = results[0]
            return TokenReference(
                chain_id=result.chain_id,
                address=result.address,
                symbol=result.symbol,
                decimals=result.decimals,
                name=result.name,
                confidence=result.confidence,
                logo_uri=result.logo_uri,
            )

        return None

    async def _intent_to_action(
        self,
        intent: TradeIntent,
        context: PlanningContext,
    ) -> Optional[Action]:
        """Convert a TradeIntent to an executable Action.

        This involves:
        1. Fetching a quote (for estimated USD value)
        2. Building the action with quote payload

        Args:
            intent: Trade intent
            context: Planning context

        Returns:
            Action or None if conversion fails
        """
        # For now, create action without quote
        # TODO: Integrate with SwapManager/BridgeManager for quotes

        return Action(
            action_id=str(uuid.uuid4()),
            action_type=intent.action_type,
            chain_id=intent.chain_id,
            token_in=intent.token_in,
            token_out=intent.token_out,
            amount=intent.amount,
            estimated_usd=Decimal("0"),  # TODO: Get from quote
            quote_payload={},
        )

    # -------------------------------------------------------------------------
    # Autonomous Mode - Plan from Strategy
    # -------------------------------------------------------------------------

    async def create_strategy_plan(
        self,
        agent_config: AgentConfig,
        context: PlanningContext,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Plan, DecisionLog]:
        """Create a plan from an autonomous strategy.

        This is the main entry point for autonomous mode.

        Args:
            agent_config: Agent configuration
            context: Planning context
            market_data: Market data for strategy evaluation

        Returns:
            Tuple of (Plan, DecisionLog)
        """
        from .strategies import get_strategy

        # Get strategy implementation
        strategy = get_strategy(agent_config.type.value)
        if not strategy:
            raise ValueError(f"Unknown strategy type: {agent_config.type}")

        # Build strategy context
        strategy_ctx = StrategyContext(
            agent_config=agent_config,
            portfolio=await self._get_portfolio_snapshot(context),
            market_data=market_data or {},
            timestamp=datetime.utcnow(),
        )

        # Evaluate strategy
        intents = strategy.evaluate(strategy_ctx)

        # Build decision log
        decision_log = DecisionLog(
            decision_id=str(uuid.uuid4()),
            agent_id=agent_config.agent_id,
            strategy_type=agent_config.type.value,
            timestamp=datetime.utcnow(),
            inputs={"market_data": market_data or {}},
            features={},
            policy=agent_config.policy.model_dump(),
            intents=[i.to_dict() for i in intents],
        )

        # Validate and build actions
        policy = agent_config.get_policy_constraints()
        actions: List[Action] = []
        warnings: List[str] = []

        for intent in intents:
            action = await self._intent_to_action(intent, context)
            if action:
                violations = policy.validate_intent(intent, action.estimated_usd)
                if violations:
                    warnings.extend(violations)
                    continue
                actions.append(action)

        decision_log.actions = [a.to_dict() for a in actions]

        # Calculate total and approval level
        total_usd = sum((a.estimated_usd for a in actions), Decimal("0"))
        approval_level = policy.get_approval_level(total_usd)

        # Create plan
        plan = Plan(
            plan_id=str(uuid.uuid4()),
            intents=intents,
            actions=actions,
            policy=policy,
            status=PlanStatus.PENDING_APPROVAL
            if approval_level != ApprovalLevel.NONE
            else PlanStatus.APPROVED,
            approval_level=approval_level,
            agent_id=agent_config.agent_id,
            decision_log=decision_log,
            warnings=warnings,
        )

        return plan, decision_log

    async def _get_portfolio_snapshot(
        self,
        context: PlanningContext,
    ) -> "PortfolioSnapshot":
        """Get a portfolio snapshot for strategy evaluation."""
        from .protocol import PortfolioSnapshot

        # TODO: Integrate with portfolio service
        return PortfolioSnapshot(
            wallet_address=context.wallet_address,
            chain_id=context.chain_id,
            balances={},
            usd_values={},
            total_usd=Decimal("0"),
            timestamp=datetime.utcnow(),
        )

    # -------------------------------------------------------------------------
    # Plan Lifecycle Management
    # -------------------------------------------------------------------------

    def get_pending_plan(self, conversation_id: str) -> Optional[Plan]:
        """Get the pending plan for a conversation."""
        return self._pending_plans.get(conversation_id)

    def approve_plan(self, conversation_id: str) -> Optional[Plan]:
        """Approve a pending plan.

        Args:
            conversation_id: Conversation ID

        Returns:
            Approved plan or None if not found
        """
        plan = self._pending_plans.get(conversation_id)
        if plan and plan.status == PlanStatus.PENDING_APPROVAL:
            plan.status = PlanStatus.APPROVED
            plan.updated_at = datetime.utcnow()
            return plan
        return None

    def cancel_plan(self, conversation_id: str) -> Optional[Plan]:
        """Cancel a pending plan.

        Args:
            conversation_id: Conversation ID

        Returns:
            Cancelled plan or None if not found
        """
        plan = self._pending_plans.pop(conversation_id, None)
        if plan:
            plan.status = PlanStatus.CANCELLED
            plan.updated_at = datetime.utcnow()
        return plan

    def clear_pending(self, conversation_id: str) -> None:
        """Clear pending plan for a conversation."""
        self._pending_plans.pop(conversation_id, None)


# Singleton instance
_service: Optional[PlanningService] = None


def get_planning_service() -> PlanningService:
    """Get the singleton PlanningService instance."""
    global _service
    if _service is None:
        _service = PlanningService()
    return _service
