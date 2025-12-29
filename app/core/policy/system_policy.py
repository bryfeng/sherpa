"""
System Policy

Evaluates actions against platform-wide system policies.
Handles emergency stops, blocked addresses, chain restrictions, and global limits.
"""

from decimal import Decimal
from typing import List, Optional

from .models import (
    ActionContext,
    PolicyType,
    PolicyViolation,
    SystemPolicyConfig,
    ViolationSeverity,
)


class SystemPolicy:
    """
    Evaluates actions against platform-wide system policies.

    System policies are controlled by the platform (not users) and include:
    - Emergency stop functionality
    - Maintenance mode
    - Blocked contracts and tokens (scams, exploits)
    - Protocol allowlist (whitelist mode)
    - Global transaction limits
    - Chain restrictions
    """

    def __init__(self, config: SystemPolicyConfig):
        self.config = config

    def evaluate(self, context: ActionContext) -> List[PolicyViolation]:
        """
        Evaluate an action against system policies.

        Returns a list of violations (system violations are always blocking).
        """
        violations: List[PolicyViolation] = []

        # Check emergency stop first
        emergency_violation = self._check_emergency_stop()
        if emergency_violation:
            violations.append(emergency_violation)
            return violations  # No point checking further

        # Check maintenance mode
        maintenance_violation = self._check_maintenance()
        if maintenance_violation:
            violations.append(maintenance_violation)
            return violations

        # Check global transaction limit
        tx_violation = self._check_global_tx_limit(context.value_usd)
        if tx_violation:
            violations.append(tx_violation)

        # Check chain restrictions
        chain_violation = self._check_chain_restrictions(context.chain_id)
        if chain_violation:
            violations.append(chain_violation)

        # Check blocked contracts
        if context.contract_address:
            contract_violation = self._check_blocked_contract(context.contract_address)
            if contract_violation:
                violations.append(contract_violation)

        # Check blocked tokens
        token_violations = self._check_blocked_tokens(
            context.token_in,
            context.token_out,
        )
        violations.extend(token_violations)

        return violations

    def is_operational(self) -> tuple[bool, Optional[str]]:
        """
        Check if the system is operational.

        Returns (is_operational, reason_if_not).
        """
        if self.config.emergency_stop:
            return False, self.config.emergency_stop_reason or "Emergency stop activated"

        if self.config.in_maintenance:
            return False, self.config.maintenance_message or "System is under maintenance"

        return True, None

    def _check_emergency_stop(self) -> Optional[PolicyViolation]:
        """Check if emergency stop is active."""
        if self.config.emergency_stop:
            return PolicyViolation(
                policy_type=PolicyType.SYSTEM,
                policy_name="emergency_stop",
                severity=ViolationSeverity.BLOCK,
                message="Trading is currently disabled",
                details={
                    "reason": self.config.emergency_stop_reason or "Emergency stop activated",
                },
                suggestion="Please wait for the platform to resume normal operations",
            )
        return None

    def _check_maintenance(self) -> Optional[PolicyViolation]:
        """Check if system is in maintenance mode."""
        if self.config.in_maintenance:
            return PolicyViolation(
                policy_type=PolicyType.SYSTEM,
                policy_name="maintenance",
                severity=ViolationSeverity.BLOCK,
                message=self.config.maintenance_message or "System is under maintenance",
                suggestion="Please try again later",
            )
        return None

    def _check_global_tx_limit(self, value_usd: Decimal) -> Optional[PolicyViolation]:
        """Check global transaction size limit."""
        if value_usd > self.config.max_single_tx_usd:
            return PolicyViolation(
                policy_type=PolicyType.SYSTEM,
                policy_name="global_tx_limit",
                severity=ViolationSeverity.BLOCK,
                message=f"Transaction ${value_usd} exceeds platform limit of ${self.config.max_single_tx_usd}",
                details={
                    "requestedValue": str(value_usd),
                    "platformMax": str(self.config.max_single_tx_usd),
                },
                suggestion=f"Split into transactions under ${self.config.max_single_tx_usd}",
            )
        return None

    def _check_chain_restrictions(self, chain_id: int) -> Optional[PolicyViolation]:
        """Check chain restrictions."""
        # Check if chain is explicitly blocked
        if chain_id in self.config.blocked_chains:
            return PolicyViolation(
                policy_type=PolicyType.SYSTEM,
                policy_name="blocked_chain",
                severity=ViolationSeverity.BLOCK,
                message=f"Chain {chain_id} is currently not supported",
                details={"chainId": chain_id},
                suggestion="Use a different chain for this transaction",
            )

        # Check if only specific chains are allowed (whitelist mode)
        if self.config.allowed_chains and chain_id not in self.config.allowed_chains:
            return PolicyViolation(
                policy_type=PolicyType.SYSTEM,
                policy_name="chain_allowlist",
                severity=ViolationSeverity.BLOCK,
                message=f"Chain {chain_id} is not in the allowed list",
                details={
                    "chainId": chain_id,
                    "allowedChains": self.config.allowed_chains,
                },
                suggestion=f"Use one of the supported chains: {self.config.allowed_chains}",
            )

        return None

    def _check_blocked_contract(self, contract_address: str) -> Optional[PolicyViolation]:
        """Check if contract is blocked."""
        normalized = contract_address.lower()

        for blocked in self.config.blocked_contracts:
            if blocked.lower() == normalized:
                return PolicyViolation(
                    policy_type=PolicyType.SYSTEM,
                    policy_name="blocked_contract",
                    severity=ViolationSeverity.BLOCK,
                    message="This contract has been flagged and is not allowed",
                    details={"contractAddress": contract_address},
                    suggestion="This contract may be associated with known exploits or scams",
                )

        return None

    def _check_blocked_tokens(
        self,
        token_in: Optional[str],
        token_out: Optional[str],
    ) -> List[PolicyViolation]:
        """Check if any tokens are blocked."""
        violations: List[PolicyViolation] = []
        blocked_set = {t.lower() for t in self.config.blocked_tokens}

        if token_in and token_in.lower() in blocked_set:
            violations.append(PolicyViolation(
                policy_type=PolicyType.SYSTEM,
                policy_name="blocked_token",
                severity=ViolationSeverity.BLOCK,
                message="This token has been flagged and cannot be traded",
                details={"token": token_in, "direction": "in"},
                suggestion="This token may be associated with known scams",
            ))

        if token_out and token_out.lower() in blocked_set:
            violations.append(PolicyViolation(
                policy_type=PolicyType.SYSTEM,
                policy_name="blocked_token",
                severity=ViolationSeverity.BLOCK,
                message="This token has been flagged and cannot be traded",
                details={"token": token_out, "direction": "out"},
                suggestion="This token may be associated with known scams",
            ))

        return violations
