"""
Gas Abstraction Service.

Provides USDC-first gas abstraction for Rhinestone intents and ERC-4337 operations.

For Rhinestone Intents:
- Gas is abstracted by the solver network
- Users pay execution fees in USDC from unified balance
- This service estimates fees and validates sufficient balance

For ERC-4337 UserOperations:
- Routes to PaymasterProvider for sponsored operations
- Supports USDC -> native token conversion for gas
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional

from ..config import settings
from ..providers.rhinestone import get_rhinestone_provider, RhinestoneError
from ..providers.paymaster import get_paymaster_provider, PaymasterError
from ..services.unified_balance import get_unified_balance_service

logger = logging.getLogger(__name__)


# Approximate gas costs in USD by operation type
DEFAULT_GAS_COSTS_USD: Dict[str, Decimal] = {
    "swap": Decimal("0.50"),
    "bridge": Decimal("1.00"),
    "transfer": Decimal("0.10"),
    "approve": Decimal("0.05"),
    "dca_execution": Decimal("0.50"),
    "copy_trade": Decimal("0.75"),
}

# Chain-specific multipliers (L2s are cheaper)
CHAIN_GAS_MULTIPLIERS: Dict[int, Decimal] = {
    1: Decimal("5.0"),      # Ethereum mainnet
    10: Decimal("0.2"),     # Optimism
    137: Decimal("0.3"),    # Polygon
    8453: Decimal("0.2"),   # Base
    42161: Decimal("0.2"),  # Arbitrum
    43114: Decimal("0.5"),  # Avalanche
    56: Decimal("0.3"),     # BSC
}


@dataclass
class GasEstimate:
    """Estimated gas cost for an operation."""
    operation_type: str
    estimated_usd: Decimal
    chain_id: int
    chain_name: str
    fee_token: str = "USDC"
    fee_amount: Decimal = Decimal("0")  # In fee token units
    confidence: str = "medium"  # low, medium, high
    breakdown: Optional[Dict[str, Any]] = None


@dataclass
class GasValidationResult:
    """Result of gas validation check."""
    valid: bool
    estimated_fee_usd: Decimal
    available_balance_usd: Decimal
    fee_token: str
    error: Optional[str] = None


class GasAbstractionService:
    """
    Service for gas estimation and abstraction.

    Handles:
    - Intent gas estimation (Rhinestone)
    - UserOp gas sponsorship (ERC-4337)
    - USDC balance validation for gas
    - Multi-chain gas cost comparison

    Usage:
        service = get_gas_abstraction_service()

        # Estimate gas for a swap intent
        estimate = await service.estimate_intent_gas(
            operation_type="swap",
            target_chain=8453,
        )

        # Validate user has sufficient USDC for gas
        validation = await service.validate_gas_balance(
            account_address="0x...",
            operation_type="swap",
            target_chain=8453,
        )
    """

    def __init__(self) -> None:
        self._rhinestone = get_rhinestone_provider()
        self._paymaster = get_paymaster_provider()
        self._balance_service = get_unified_balance_service()

    async def estimate_intent_gas(
        self,
        operation_type: str,
        target_chain: int,
        source_chains: Optional[List[int]] = None,
        cross_chain: bool = False,
    ) -> GasEstimate:
        """
        Estimate gas cost for a Rhinestone intent.

        Args:
            operation_type: Type of operation (swap, bridge, transfer, etc.)
            target_chain: Chain where the operation executes
            source_chains: Chains funds will be sourced from (for cross-chain)
            cross_chain: Whether this is a cross-chain intent

        Returns:
            GasEstimate with USD cost
        """
        # Get base cost for operation type
        base_cost = DEFAULT_GAS_COSTS_USD.get(operation_type, Decimal("0.50"))

        # Apply chain multiplier
        chain_multiplier = CHAIN_GAS_MULTIPLIERS.get(target_chain, Decimal("1.0"))
        estimated_usd = base_cost * chain_multiplier

        # Add cross-chain overhead if sourcing from multiple chains
        if cross_chain or (source_chains and len(source_chains) > 1):
            # Each additional source chain adds ~$0.30
            num_bridges = len(source_chains) - 1 if source_chains else 0
            estimated_usd += Decimal("0.30") * num_bridges

        # Get chain name
        from .unified_balance import SUPPORTED_CHAINS
        chain_name = SUPPORTED_CHAINS.get(target_chain, f"Chain {target_chain}")

        return GasEstimate(
            operation_type=operation_type,
            estimated_usd=estimated_usd.quantize(Decimal("0.01")),
            chain_id=target_chain,
            chain_name=chain_name,
            fee_token="USDC",
            fee_amount=estimated_usd.quantize(Decimal("0.000001")),  # USDC has 6 decimals
            confidence="medium",
            breakdown={
                "base_cost_usd": str(base_cost),
                "chain_multiplier": str(chain_multiplier),
                "cross_chain": cross_chain,
                "source_chains": source_chains or [target_chain],
            },
        )

    async def validate_gas_balance(
        self,
        account_address: str,
        operation_type: str,
        target_chain: int,
        source_chains: Optional[List[int]] = None,
    ) -> GasValidationResult:
        """
        Validate that account has sufficient USDC for gas.

        Args:
            account_address: Smart Account address
            operation_type: Type of operation
            target_chain: Target chain ID
            source_chains: Source chain IDs

        Returns:
            GasValidationResult with validation status
        """
        # Estimate gas
        estimate = await self.estimate_intent_gas(
            operation_type=operation_type,
            target_chain=target_chain,
            source_chains=source_chains,
            cross_chain=bool(source_chains and len(source_chains) > 1),
        )

        # Get USDC balance
        usdc_balance = await self._balance_service.get_token_balance(
            account_address=account_address,
            token_symbol="USDC",
        )

        available_usd = Decimal("0")
        if usdc_balance:
            available_usd = usdc_balance.total_usd_value

        # Validate
        if available_usd >= estimate.estimated_usd:
            return GasValidationResult(
                valid=True,
                estimated_fee_usd=estimate.estimated_usd,
                available_balance_usd=available_usd,
                fee_token="USDC",
            )
        else:
            return GasValidationResult(
                valid=False,
                estimated_fee_usd=estimate.estimated_usd,
                available_balance_usd=available_usd,
                fee_token="USDC",
                error=f"Insufficient USDC for gas. Need ${estimate.estimated_usd}, have ${available_usd}",
            )

    async def get_cheapest_chain(
        self,
        operation_type: str,
        chains: Optional[List[int]] = None,
    ) -> Optional[int]:
        """
        Find the cheapest chain for an operation type.

        Args:
            operation_type: Type of operation
            chains: Chains to consider (defaults to all supported)

        Returns:
            Chain ID with lowest gas cost
        """
        from .unified_balance import SUPPORTED_CHAINS

        chains = chains or list(SUPPORTED_CHAINS.keys())

        cheapest_chain = None
        cheapest_cost = Decimal("999999")

        for chain_id in chains:
            estimate = await self.estimate_intent_gas(
                operation_type=operation_type,
                target_chain=chain_id,
            )
            if estimate.estimated_usd < cheapest_cost:
                cheapest_cost = estimate.estimated_usd
                cheapest_chain = chain_id

        return cheapest_chain

    async def compare_chains(
        self,
        operation_type: str,
        chains: Optional[List[int]] = None,
    ) -> List[GasEstimate]:
        """
        Compare gas costs across chains for an operation.

        Args:
            operation_type: Type of operation
            chains: Chains to compare (defaults to all supported)

        Returns:
            List of GasEstimates sorted by cost (cheapest first)
        """
        from .unified_balance import SUPPORTED_CHAINS

        chains = chains or list(SUPPORTED_CHAINS.keys())

        estimates = []
        for chain_id in chains:
            estimate = await self.estimate_intent_gas(
                operation_type=operation_type,
                target_chain=chain_id,
            )
            estimates.append(estimate)

        # Sort by cost
        estimates.sort(key=lambda e: e.estimated_usd)

        return estimates

    async def estimate_batch_gas(
        self,
        operations: List[Dict[str, Any]],
        target_chain: int,
    ) -> GasEstimate:
        """
        Estimate gas for a batch of operations.

        Args:
            operations: List of {operation_type, ...} dicts
            target_chain: Chain where batch executes

        Returns:
            Combined GasEstimate
        """
        total_usd = Decimal("0")
        breakdown = []

        for op in operations:
            op_type = op.get("operation_type", "unknown")
            estimate = await self.estimate_intent_gas(
                operation_type=op_type,
                target_chain=target_chain,
            )
            total_usd += estimate.estimated_usd
            breakdown.append({
                "operation_type": op_type,
                "estimated_usd": str(estimate.estimated_usd),
            })

        # Batch discount (typically 10-20% cheaper)
        if len(operations) > 1:
            discount = Decimal("0.9")  # 10% discount
            total_usd = total_usd * discount

        from .unified_balance import SUPPORTED_CHAINS
        chain_name = SUPPORTED_CHAINS.get(target_chain, f"Chain {target_chain}")

        return GasEstimate(
            operation_type="batch",
            estimated_usd=total_usd.quantize(Decimal("0.01")),
            chain_id=target_chain,
            chain_name=chain_name,
            fee_token="USDC",
            fee_amount=total_usd.quantize(Decimal("0.000001")),
            confidence="medium",
            breakdown={
                "operations": breakdown,
                "batch_discount_applied": len(operations) > 1,
            },
        )


# Singleton instance
_gas_abstraction_service: Optional[GasAbstractionService] = None


def get_gas_abstraction_service() -> GasAbstractionService:
    """Get the singleton GasAbstraction service instance."""
    global _gas_abstraction_service
    if _gas_abstraction_service is None:
        _gas_abstraction_service = GasAbstractionService()
    return _gas_abstraction_service
