"""
Fee Policy

Validates fee configuration for gas abstraction and paymaster routing.
"""

from typing import List

from app.core.chain_types import is_solana_chain

from .models import (
    ActionContext,
    FeePolicyConfig,
    PolicyType,
    PolicyViolation,
    ViolationSeverity,
)


class FeePolicy:
    """
    Evaluates actions against fee policy constraints.

    Fee policies define:
    - Stablecoin-first gas asset preference (USDC)
    - Optional native fallback order
    - Reimbursement mode (Solana)
    - Config enablement
    """

    def __init__(self, config: FeePolicyConfig):
        self.config = config

    def evaluate(self, context: ActionContext) -> List[PolicyViolation]:
        violations: List[PolicyViolation] = []

        if self.config.missing:
            violations.append(
                PolicyViolation(
                    policy_type=PolicyType.FEE,
                    policy_name="fee_policy_missing",
                    severity=ViolationSeverity.BLOCK,
                    message="Fee policy is missing for this chain",
                    details={"chainId": self.config.chain_id},
                    suggestion="Configure fee policy before enabling autonomy",
                )
            )
            return violations

        if not self.config.is_enabled:
            violations.append(
                PolicyViolation(
                    policy_type=PolicyType.FEE,
                    policy_name="fee_policy_disabled",
                    severity=ViolationSeverity.BLOCK,
                    message="Fee policy is disabled for this chain",
                    details={"chainId": self.config.chain_id},
                    suggestion="Enable fee policy before enabling autonomy",
                )
            )
            return violations

        if not self.config.stablecoin_address:
            violations.append(
                PolicyViolation(
                    policy_type=PolicyType.FEE,
                    policy_name="stablecoin_address_missing",
                    severity=ViolationSeverity.BLOCK,
                    message="Stablecoin address missing for fee policy",
                    details={
                        "chainId": self.config.chain_id,
                        "stablecoinSymbol": self.config.stablecoin_symbol,
                    },
                    suggestion="Set stablecoin address in fee policy config",
                )
            )

        if self.config.stablecoin_symbol.upper() != "USDC":
            violations.append(
                PolicyViolation(
                    policy_type=PolicyType.FEE,
                    policy_name="stablecoin_symbol",
                    severity=ViolationSeverity.BLOCK,
                    message="Fee policy must prioritize USDC for gas abstraction",
                    details={"stablecoinSymbol": self.config.stablecoin_symbol},
                    suggestion="Set stablecoinSymbol to USDC",
                )
            )

        fee_asset_order = list(self.config.fee_asset_order or [])
        if not fee_asset_order or fee_asset_order[0] != "stablecoin":
            violations.append(
                PolicyViolation(
                    policy_type=PolicyType.FEE,
                    policy_name="fee_asset_order",
                    severity=ViolationSeverity.BLOCK,
                    message="Fee asset order must start with stablecoin",
                    details={"feeAssetOrder": fee_asset_order},
                    suggestion="Use ['stablecoin', 'native'] or ['stablecoin']",
                )
            )
        elif "stablecoin" not in fee_asset_order:
            violations.append(
                PolicyViolation(
                    policy_type=PolicyType.FEE,
                    policy_name="fee_asset_order",
                    severity=ViolationSeverity.BLOCK,
                    message="Fee asset order must include stablecoin",
                    details={"feeAssetOrder": fee_asset_order},
                    suggestion="Add stablecoin to fee asset order",
                )
            )

        if self.config.allow_native_fallback and "native" not in fee_asset_order:
            violations.append(
                PolicyViolation(
                    policy_type=PolicyType.FEE,
                    policy_name="native_fallback_missing",
                    severity=ViolationSeverity.BLOCK,
                    message="Native fallback enabled but not present in fee asset order",
                    details={"feeAssetOrder": fee_asset_order},
                    suggestion="Add 'native' to fee asset order or disable fallback",
                )
            )
        if not self.config.allow_native_fallback and "native" in fee_asset_order:
            violations.append(
                PolicyViolation(
                    policy_type=PolicyType.FEE,
                    policy_name="native_fallback_disabled",
                    severity=ViolationSeverity.BLOCK,
                    message="Native fallback disabled but present in fee asset order",
                    details={"feeAssetOrder": fee_asset_order},
                    suggestion="Remove 'native' from fee asset order or enable fallback",
                )
            )

        if is_solana_chain(self.config.chain_id) and self.config.reimbursement_mode != "per_tx":
            violations.append(
                PolicyViolation(
                    policy_type=PolicyType.FEE,
                    policy_name="reimbursement_mode",
                    severity=ViolationSeverity.WARN,
                    message="Solana fee policy should use per-tx reimbursement",
                    details={"reimbursementMode": self.config.reimbursement_mode},
                    suggestion="Set reimbursementMode to per_tx for Solana",
                )
            )

        return violations
