"""
Smart Sessions configuration helpers for Rhinestone ERC-7579 module.

Smart Sessions are on-chain permission grants that allow session keys to
execute transactions within defined constraints. Unlike off-chain session
keys (session_manager.py), Smart Sessions are enforced by the smart contract.

Docs: https://docs.rhinestone.wtf/module-sdk/modules/smart-sessions
ERC-7579: https://erc7579.com/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from app.config import settings

logger = logging.getLogger(__name__)


class SmartSessionError(Exception):
    """Smart Session configuration error."""
    pass


class ActionType(str, Enum):
    """Supported action types for Smart Sessions."""
    SWAP = "swap"
    BRIDGE = "bridge"
    TRANSFER = "transfer"
    APPROVE = "approve"
    STAKE = "stake"
    UNSTAKE = "unstake"
    CLAIM = "claim"
    DCA = "dca"
    COPY_TRADE = "copy_trade"


@dataclass
class SpendingLimit:
    """
    Spending limit configuration for Smart Sessions.

    Limits can be enforced per-transaction or over a time period.
    """
    max_per_tx_usd: Decimal = Decimal("500")
    max_daily_usd: Optional[Decimal] = None
    max_total_usd: Optional[Decimal] = None

    def to_module_config(self) -> Dict[str, Any]:
        """Convert to Rhinestone module configuration format."""
        config = {
            "maxPerTx": str(int(self.max_per_tx_usd * 10**6)),  # USDC decimals
        }
        if self.max_daily_usd:
            config["maxDaily"] = str(int(self.max_daily_usd * 10**6))
        if self.max_total_usd:
            config["maxTotal"] = str(int(self.max_total_usd * 10**6))
        return config


@dataclass
class TimeConstraint:
    """Time constraint for Smart Sessions."""
    valid_after: int = 0  # Unix timestamp, 0 = now
    valid_until: int = 0  # Unix timestamp, 0 = no expiry

    def to_module_config(self) -> Dict[str, Any]:
        """Convert to Rhinestone module configuration format."""
        return {
            "validAfter": self.valid_after,
            "validUntil": self.valid_until,
        }


@dataclass
class ContractPolicy:
    """
    Policy for allowed contract interactions.

    Can specify allowed functions and value limits per contract.
    """
    address: str
    allowed_functions: List[str] = field(default_factory=list)  # Function selectors
    max_value_per_call: Optional[Decimal] = None

    def to_module_config(self) -> Dict[str, Any]:
        """Convert to Rhinestone module configuration format."""
        config = {
            "target": self.address,
            "functions": self.allowed_functions or [],
        }
        if self.max_value_per_call:
            config["maxValue"] = str(int(self.max_value_per_call * 10**18))
        return config


@dataclass
class SmartSessionConfig:
    """
    Configuration for a Rhinestone Smart Session.

    This configuration is used to generate the on-chain permission grant
    that the user signs to enable autonomous execution.
    """
    session_key_address: str  # The address that will sign transactions
    owner_address: str  # The smart account owner

    # Permissions
    allowed_actions: List[ActionType] = field(default_factory=lambda: [ActionType.SWAP])
    allowed_chains: List[int] = field(default_factory=list)  # Empty = all chains
    allowed_tokens: List[str] = field(default_factory=list)  # Empty = all tokens

    # Limits
    spending_limit: SpendingLimit = field(default_factory=SpendingLimit)
    time_constraint: TimeConstraint = field(default_factory=TimeConstraint)

    # Contract policies (optional granular control)
    contract_policies: List[ContractPolicy] = field(default_factory=list)

    # Metadata
    label: Optional[str] = None

    def to_module_config(self) -> Dict[str, Any]:
        """
        Convert to Rhinestone Smart Sessions module configuration.

        This format is used by the frontend to create the permission grant
        transaction that the user signs.
        """
        return {
            "sessionKeyAddress": self.session_key_address,
            "permissions": {
                "allowedActions": [a.value for a in self.allowed_actions],
                "allowedChains": self.allowed_chains,
                "allowedTokens": self.allowed_tokens,
                "spendingLimit": self.spending_limit.to_module_config(),
                "timeConstraint": self.time_constraint.to_module_config(),
                "contractPolicies": [p.to_module_config() for p in self.contract_policies],
            },
            "label": self.label,
        }

    def validate(self) -> List[str]:
        """
        Validate the configuration.

        Returns list of validation errors, empty if valid.
        """
        errors = []

        if not self.session_key_address:
            errors.append("Session key address is required")

        if not self.owner_address:
            errors.append("Owner address is required")

        if not self.allowed_actions:
            errors.append("At least one action must be allowed")

        if self.spending_limit.max_per_tx_usd <= 0:
            errors.append("Max per transaction must be positive")

        if self.time_constraint.valid_until and self.time_constraint.valid_after:
            if self.time_constraint.valid_until <= self.time_constraint.valid_after:
                errors.append("Valid until must be after valid after")

        return errors


class SmartSessionsHelper:
    """
    Helper class for building Smart Session configurations.

    Provides preset configurations for common use cases like DCA,
    copy trading, and swaps.
    """

    # Known safe protocol contracts by chain
    PROTOCOL_CONTRACTS: Dict[int, Dict[str, str]] = {
        # Ethereum mainnet
        1: {
            "uniswap_v3_router": "0xE592427A0AEce92De3Edee1F18E0157C05861564",
            "uniswap_universal": "0x3fC91A3afd70395Cd496C647d5a6CC9D4B2b7FAD",
        },
        # Base
        8453: {
            "uniswap_v3_router": "0x2626664c2603336E57B271c5C0b26F421741e481",
            "uniswap_universal": "0x3fC91A3afd70395Cd496C647d5a6CC9D4B2b7FAD",
        },
        # Arbitrum
        42161: {
            "uniswap_v3_router": "0xE592427A0AEce92De3Edee1F18E0157C05861564",
            "uniswap_universal": "0x5E325eDA8064b456f4781070C0738d849c824258",
        },
        # Optimism
        10: {
            "uniswap_v3_router": "0xE592427A0AEce92De3Edee1F18E0157C05861564",
            "uniswap_universal": "0xCb1355ff08Ab38bBCE60111F1bb2B784bE25D7e8",
        },
    }

    # Common stablecoin addresses by chain
    STABLECOINS: Dict[int, List[str]] = {
        1: [
            "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
            "0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT
            "0x6B175474E89094C44Da98b954EesignatureHashdcFE92F63d",  # DAI
        ],
        8453: [
            "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",  # USDC
        ],
        42161: [
            "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",  # USDC
            "0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9",  # USDT
        ],
        10: [
            "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85",  # USDC
            "0x94b008aA00579c1307B0EF2c499aD98a8ce58e58",  # USDT
        ],
    }

    @classmethod
    def create_swap_session(
        cls,
        session_key_address: str,
        owner_address: str,
        max_per_tx_usd: Decimal = Decimal("500"),
        max_daily_usd: Optional[Decimal] = Decimal("2000"),
        valid_days: int = 7,
        chains: Optional[List[int]] = None,
    ) -> SmartSessionConfig:
        """
        Create a Smart Session config for swap operations.

        Args:
            session_key_address: Address of the session key
            owner_address: Smart account owner
            max_per_tx_usd: Maximum USD value per swap
            max_daily_usd: Maximum daily spending
            valid_days: Session validity in days
            chains: Allowed chains (None = all)

        Returns:
            SmartSessionConfig for swaps
        """
        import time

        return SmartSessionConfig(
            session_key_address=session_key_address,
            owner_address=owner_address,
            allowed_actions=[ActionType.SWAP, ActionType.APPROVE],
            allowed_chains=chains or [],
            spending_limit=SpendingLimit(
                max_per_tx_usd=max_per_tx_usd,
                max_daily_usd=max_daily_usd,
            ),
            time_constraint=TimeConstraint(
                valid_after=int(time.time()),
                valid_until=int(time.time()) + (valid_days * 24 * 60 * 60),
            ),
            label="Swap Session",
        )

    @classmethod
    def create_dca_session(
        cls,
        session_key_address: str,
        owner_address: str,
        max_per_execution_usd: Decimal = Decimal("100"),
        max_total_usd: Decimal = Decimal("5000"),
        valid_days: int = 30,
        chains: Optional[List[int]] = None,
    ) -> SmartSessionConfig:
        """
        Create a Smart Session config for DCA strategy execution.

        DCA sessions have lower per-execution limits but higher total limits
        to allow for many small recurring purchases.

        Args:
            session_key_address: Address of the session key
            owner_address: Smart account owner
            max_per_execution_usd: Maximum per DCA execution
            max_total_usd: Maximum total across all executions
            valid_days: Session validity in days
            chains: Allowed chains

        Returns:
            SmartSessionConfig for DCA
        """
        import time

        return SmartSessionConfig(
            session_key_address=session_key_address,
            owner_address=owner_address,
            allowed_actions=[ActionType.SWAP, ActionType.APPROVE, ActionType.DCA],
            allowed_chains=chains or [],
            spending_limit=SpendingLimit(
                max_per_tx_usd=max_per_execution_usd,
                max_total_usd=max_total_usd,
            ),
            time_constraint=TimeConstraint(
                valid_after=int(time.time()),
                valid_until=int(time.time()) + (valid_days * 24 * 60 * 60),
            ),
            label="DCA Session",
        )

    @classmethod
    def create_copy_trading_session(
        cls,
        session_key_address: str,
        owner_address: str,
        max_per_copy_usd: Decimal = Decimal("250"),
        max_daily_usd: Decimal = Decimal("1000"),
        valid_days: int = 14,
        chains: Optional[List[int]] = None,
    ) -> SmartSessionConfig:
        """
        Create a Smart Session config for copy trading.

        Copy trading sessions allow both swaps and bridges to follow
        leader trades across chains.

        Args:
            session_key_address: Address of the session key
            owner_address: Smart account owner
            max_per_copy_usd: Maximum per copied trade
            max_daily_usd: Maximum daily copy volume
            valid_days: Session validity in days
            chains: Allowed chains

        Returns:
            SmartSessionConfig for copy trading
        """
        import time

        return SmartSessionConfig(
            session_key_address=session_key_address,
            owner_address=owner_address,
            allowed_actions=[
                ActionType.SWAP,
                ActionType.BRIDGE,
                ActionType.APPROVE,
                ActionType.COPY_TRADE,
            ],
            allowed_chains=chains or [],
            spending_limit=SpendingLimit(
                max_per_tx_usd=max_per_copy_usd,
                max_daily_usd=max_daily_usd,
            ),
            time_constraint=TimeConstraint(
                valid_after=int(time.time()),
                valid_until=int(time.time()) + (valid_days * 24 * 60 * 60),
            ),
            label="Copy Trading Session",
        )

    @classmethod
    def get_protocol_contracts(cls, chain_id: int, protocol: str) -> Optional[str]:
        """Get known protocol contract address."""
        chain_contracts = cls.PROTOCOL_CONTRACTS.get(chain_id, {})
        return chain_contracts.get(protocol)

    @classmethod
    def get_stablecoins(cls, chain_id: int) -> List[str]:
        """Get stablecoin addresses for a chain."""
        return cls.STABLECOINS.get(chain_id, [])


def validate_session_config(config: SmartSessionConfig) -> Dict[str, Any]:
    """
    Validate a Smart Session configuration.

    Returns validation result with errors if invalid.
    """
    errors = config.validate()

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "config": config.to_module_config() if not errors else None,
    }


def build_permission_grant_data(
    config: SmartSessionConfig,
) -> Dict[str, Any]:
    """
    Build the data payload for a permission grant transaction.

    This is the data that the frontend will use to construct the
    transaction for the user to sign.

    Args:
        config: The Smart Session configuration

    Returns:
        Transaction data for permission grant
    """
    validation = validate_session_config(config)
    if not validation["valid"]:
        raise SmartSessionError(f"Invalid config: {validation['errors']}")

    return {
        "type": "smart_session_grant",
        "sessionKeyAddress": config.session_key_address,
        "permissions": config.to_module_config()["permissions"],
        "label": config.label,
    }
