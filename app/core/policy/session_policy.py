"""
Session Policy

Evaluates actions against session key constraints.
Wraps the existing session key validation logic.
"""

from decimal import Decimal
from typing import List, Optional

from ..wallet.models import SessionKey, Permission, SessionKeyStatus
from .models import (
    ActionContext,
    PolicyType,
    PolicyViolation,
    ViolationSeverity,
)


class SessionPolicy:
    """
    Evaluates actions against session key policies.

    Session keys define:
    - Allowed permissions (swap, bridge, transfer, etc.)
    - Value limits (per-tx and total)
    - Chain allowlist
    - Contract allowlist
    - Token allowlist
    - Time limits (expiry)
    """

    def __init__(self, session_key: SessionKey):
        self.session_key = session_key

    def evaluate(self, context: ActionContext) -> List[PolicyViolation]:
        """
        Evaluate an action against the session key policies.

        Returns a list of violations (empty if action is allowed).
        """
        violations: List[PolicyViolation] = []

        # Check session validity
        validity_violation = self._check_validity()
        if validity_violation:
            violations.append(validity_violation)
            return violations  # No point checking further if invalid

        # Check permission
        permission_violation = self._check_permission(context.action_type)
        if permission_violation:
            violations.append(permission_violation)

        # Check value limits
        value_violations = self._check_value_limits(context.value_usd)
        violations.extend(value_violations)

        # Check chain allowlist
        chain_violation = self._check_chain(context.chain_id)
        if chain_violation:
            violations.append(chain_violation)

        # Check contract allowlist
        if context.contract_address:
            contract_violation = self._check_contract(context.contract_address)
            if contract_violation:
                violations.append(contract_violation)

        # Check token allowlist
        token_violations = self._check_tokens(
            context.chain_id,
            context.token_in,
            context.token_out
        )
        violations.extend(token_violations)

        return violations

    def _check_validity(self) -> Optional[PolicyViolation]:
        """Check if the session key is valid."""
        if self.session_key.status == SessionKeyStatus.REVOKED:
            return PolicyViolation(
                policy_type=PolicyType.SESSION,
                policy_name="session_validity",
                severity=ViolationSeverity.BLOCK,
                message="Session key has been revoked",
                details={"reason": self.session_key.revoke_reason},
                suggestion="Create a new session key to continue",
            )

        if self.session_key.status == SessionKeyStatus.EXPIRED:
            return PolicyViolation(
                policy_type=PolicyType.SESSION,
                policy_name="session_validity",
                severity=ViolationSeverity.BLOCK,
                message="Session key has expired",
                details={"expiredAt": self.session_key.expires_at.isoformat()},
                suggestion="Create a new session key to continue",
            )

        if self.session_key.status == SessionKeyStatus.EXHAUSTED:
            return PolicyViolation(
                policy_type=PolicyType.SESSION,
                policy_name="session_validity",
                severity=ViolationSeverity.BLOCK,
                message="Session key limits have been exhausted",
                details={
                    "totalUsed": str(self.session_key.value_limits.total_value_used_usd),
                    "maxTotal": str(self.session_key.value_limits.max_total_value_usd),
                },
                suggestion="Create a new session key with higher limits",
            )

        if not self.session_key.is_valid:
            return PolicyViolation(
                policy_type=PolicyType.SESSION,
                policy_name="session_validity",
                severity=ViolationSeverity.BLOCK,
                message="Session key is not valid",
                suggestion="Create a new session key",
            )

        return None

    def _check_permission(self, action_type: str) -> Optional[PolicyViolation]:
        """Check if the action type is permitted."""
        try:
            permission = Permission(action_type.lower())
        except ValueError:
            return PolicyViolation(
                policy_type=PolicyType.SESSION,
                policy_name="permission",
                severity=ViolationSeverity.BLOCK,
                message=f"Unknown action type: {action_type}",
                suggestion="Use a valid action type (swap, bridge, transfer, etc.)",
            )

        if permission not in self.session_key.permissions:
            allowed = [p.value for p in self.session_key.permissions]
            return PolicyViolation(
                policy_type=PolicyType.SESSION,
                policy_name="permission",
                severity=ViolationSeverity.BLOCK,
                message=f"Action '{action_type}' is not permitted by this session",
                details={
                    "requestedAction": action_type,
                    "allowedActions": allowed,
                },
                suggestion=f"This session only allows: {', '.join(allowed)}",
            )

        return None

    def _check_value_limits(self, value_usd: Decimal) -> List[PolicyViolation]:
        """Check value limits."""
        violations: List[PolicyViolation] = []
        limits = self.session_key.value_limits

        # Check per-transaction limit
        if value_usd > limits.max_value_per_tx_usd:
            violations.append(PolicyViolation(
                policy_type=PolicyType.SESSION,
                policy_name="value_limit_per_tx",
                severity=ViolationSeverity.BLOCK,
                message=f"Transaction value ${value_usd} exceeds per-transaction limit of ${limits.max_value_per_tx_usd}",
                details={
                    "requestedValue": str(value_usd),
                    "maxPerTx": str(limits.max_value_per_tx_usd),
                },
                suggestion=f"Reduce transaction size to under ${limits.max_value_per_tx_usd}",
            ))

        # Check total limit
        remaining = limits.max_total_value_usd - limits.total_value_used_usd
        if value_usd > remaining:
            violations.append(PolicyViolation(
                policy_type=PolicyType.SESSION,
                policy_name="value_limit_total",
                severity=ViolationSeverity.BLOCK,
                message=f"Transaction would exceed session's total limit. Remaining: ${remaining}",
                details={
                    "requestedValue": str(value_usd),
                    "totalUsed": str(limits.total_value_used_usd),
                    "maxTotal": str(limits.max_total_value_usd),
                    "remaining": str(remaining),
                },
                suggestion=f"Reduce transaction to ${remaining} or create a new session",
            ))

        # Check transaction count
        if limits.max_transactions:
            if limits.transaction_count >= limits.max_transactions:
                violations.append(PolicyViolation(
                    policy_type=PolicyType.SESSION,
                    policy_name="transaction_count",
                    severity=ViolationSeverity.BLOCK,
                    message=f"Session has reached its transaction limit of {limits.max_transactions}",
                    details={
                        "transactionCount": limits.transaction_count,
                        "maxTransactions": limits.max_transactions,
                    },
                    suggestion="Create a new session key to continue",
                ))

        return violations

    def _check_chain(self, chain_id: int) -> Optional[PolicyViolation]:
        """Check if the chain is allowed."""
        if not self.session_key.chain_allowlist.is_allowed(chain_id):
            allowed = list(self.session_key.chain_allowlist.allowed_chain_ids)
            return PolicyViolation(
                policy_type=PolicyType.SESSION,
                policy_name="chain_allowlist",
                severity=ViolationSeverity.BLOCK,
                message=f"Chain {chain_id} is not allowed by this session",
                details={
                    "requestedChain": chain_id,
                    "allowedChains": allowed,
                },
                suggestion=f"Use one of the allowed chains: {allowed}",
            )
        return None

    def _check_contract(self, contract_address: str) -> Optional[PolicyViolation]:
        """Check if the contract is allowed."""
        if not self.session_key.contract_allowlist.is_allowed(contract_address):
            return PolicyViolation(
                policy_type=PolicyType.SESSION,
                policy_name="contract_allowlist",
                severity=ViolationSeverity.BLOCK,
                message="Contract is not in the allowlist for this session",
                details={
                    "contractAddress": contract_address,
                },
                suggestion="Add this contract to the session's allowlist or create a new session",
            )
        return None

    def _check_tokens(
        self,
        chain_id: int,
        token_in: Optional[str],
        token_out: Optional[str]
    ) -> List[PolicyViolation]:
        """Check if the tokens are allowed."""
        violations: List[PolicyViolation] = []
        token_allowlist = self.session_key.token_allowlist

        if token_in and not token_allowlist.is_allowed(chain_id, token_in):
            violations.append(PolicyViolation(
                policy_type=PolicyType.SESSION,
                policy_name="token_allowlist",
                severity=ViolationSeverity.BLOCK,
                message="Input token is not in the allowlist",
                details={"token": token_in, "direction": "in"},
                suggestion="Add this token to the session's allowlist",
            ))

        if token_out and not token_allowlist.is_allowed(chain_id, token_out):
            violations.append(PolicyViolation(
                policy_type=PolicyType.SESSION,
                policy_name="token_allowlist",
                severity=ViolationSeverity.BLOCK,
                message="Output token is not in the allowlist",
                details={"token": token_out, "direction": "out"},
                suggestion="Add this token to the session's allowlist",
            ))

        return violations
