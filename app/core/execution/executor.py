"""
Transaction executor for on-chain execution.

Handles the full lifecycle of transaction execution:
- Gas estimation
- Nonce management
- Transaction submission
- Confirmation monitoring
- Error handling and retries
"""

import asyncio
import logging
import re
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.config import settings
from app.db.convex_client import get_convex_client, ConvexClient
from app.core.policy import (
    ActionContext,
    FeePolicyConfig,
    PolicyEngine,
    RiskPolicyConfig,
    SystemPolicyConfig,
)
from app.core.recovery import RecoveryExecutor
from app.core.wallet.models import SessionKey, Permission

from .models import (
    PreparedTransaction,
    TransactionResult,
    TransactionStatus,
    TransactionType,
    GasEstimate,
    SwapQuote,
    BridgeQuote,
    ExecutionContext,
)
from .erc4337_executor import UserOpExecutor, UserOpExecutionError
from .userop import UserOperation
from .userop_builder import build_entrypoint_get_nonce_call, build_execute_call_data
from .nonce_manager import NonceManager, get_nonce_manager
from .tx_builder import TransactionBuilder


logger = logging.getLogger(__name__)

ERC20_ALLOWANCE_SELECTOR = "0xdd62ed3e"  # allowance(address owner, address spender)
NATIVE_TOKEN_SENTINELS = {
    "0x0000000000000000000000000000000000000000",
    "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
    "native",
}


class ExecutionError(Exception):
    """Base exception for execution errors."""
    pass


class GasEstimationError(ExecutionError):
    """Gas estimation failed."""
    pass


class TransactionSubmitError(ExecutionError):
    """Transaction submission failed."""
    pass


class TransactionRevertError(ExecutionError):
    """Transaction reverted on-chain."""

    def __init__(self, message: str, revert_reason: Optional[str] = None):
        super().__init__(message)
        self.revert_reason = revert_reason


class TransactionTimeoutError(ExecutionError):
    """Transaction confirmation timed out."""
    pass


class PolicyRequiredError(ExecutionError):
    """Risk policy required before autonomous execution."""
    pass


class SessionKeyRequiredError(ExecutionError):
    """Session key required before autonomous execution."""
    pass


class PolicyViolationError(ExecutionError):
    """Policy engine blocked the action."""

    def __init__(self, message: str, violations: Optional[list[dict]] = None):
        super().__init__(message)
        self.violations = violations or []


class ApprovalRequiredError(ExecutionError):
    """Action requires manual approval."""

    def __init__(self, message: str, approval_reason: Optional[str] = None):
        super().__init__(message)
        self.approval_reason = approval_reason


class SignatureRequiredError(ExecutionError):
    """Quote requires off-chain signatures before execution."""
    pass


class TransactionExecutor:
    """
    Executes transactions on EVM chains.

    Responsibilities:
    - Estimate gas costs
    - Manage nonces via NonceManager
    - Submit transactions to the network
    - Monitor for confirmation
    - Handle errors and retries
    - Log transactions to Convex
    """

    def __init__(
        self,
        nonce_manager: Optional[NonceManager] = None,
        convex: Optional[ConvexClient] = None,
        rpc_urls: Optional[Dict[int, str]] = None,
        session_manager: Optional[Any] = None,
    ):
        self.nonce_manager = nonce_manager or get_nonce_manager()
        self.convex = convex or get_convex_client()
        self._session_manager = session_manager
        self._client = httpx.AsyncClient(timeout=60.0)
        self._recovery = RecoveryExecutor()
        self._userop_executor = UserOpExecutor()

        # RPC URLs per chain
        self._rpc_urls = rpc_urls or {
            1: f"https://eth-mainnet.g.alchemy.com/v2/{settings.alchemy_api_key}",
            10: f"https://opt-mainnet.g.alchemy.com/v2/{settings.alchemy_api_key}",
            137: f"https://polygon-mainnet.g.alchemy.com/v2/{settings.alchemy_api_key}",
            42161: f"https://arb-mainnet.g.alchemy.com/v2/{settings.alchemy_api_key}",
            8453: f"https://base-mainnet.g.alchemy.com/v2/{settings.alchemy_api_key}",
        }

    async def _rpc_call_raw(
        self,
        chain_id: int,
        method: str,
        params: List[Any],
    ) -> Any:
        rpc_url = self._rpc_urls.get(chain_id)
        if not rpc_url:
            raise ValueError(f"No RPC URL configured for chain {chain_id}")

        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": 1,
        }

        response = await self._client.post(rpc_url, json=payload)
        response.raise_for_status()
        result = response.json()

        if "error" in result:
            raise RuntimeError(f"RPC error: {result['error']}")

        return result.get("result")

    async def _rpc_call(
        self,
        chain_id: int,
        method: str,
        params: List[Any],
    ) -> Any:
        """Make an RPC call to the chain with recovery."""
        operation_name = f"rpc:{method}"

        async def operation():
            return await self._rpc_call_raw(chain_id, method, params)

        recovery = await self._recovery.execute(
            operation,
            operation_name=operation_name,
            provider="alchemy",
            context={"chain_id": chain_id},
        )
        if not recovery.success:
            raise recovery.error or RuntimeError("RPC call failed")
        return recovery.result

    @staticmethod
    def _encode_uint256(value: int) -> str:
        return format(value, "064x")

    @staticmethod
    def _encode_address(address: str) -> str:
        addr = address.lower().replace("0x", "")
        return addr.zfill(64)

    async def _get_userop_nonce(self, chain_id: int, sender: str, key: int = 0) -> int:
        if not settings.erc4337_entrypoint_address:
            raise ExecutionError("EntryPoint address not configured for ERC-4337")

        data = build_entrypoint_get_nonce_call(sender, key)
        result = await self._rpc_call(
            chain_id,
            "eth_call",
            [
                {
                    "to": settings.erc4337_entrypoint_address,
                    "data": data,
                },
                "latest",
            ],
        )
        if not isinstance(result, str):
            raise ExecutionError("Invalid EntryPoint nonce response")
        return int(result, 16)

    async def _record_session_usage(
        self,
        *,
        context: ExecutionContext,
        action: Permission,
        value_usd: float,
        tx_hash: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not context.session_key_id:
            return

        if self._session_manager is None:
            from app.core.wallet import SessionKeyManager

            self._session_manager = SessionKeyManager(convex=self.convex)

        try:
            await self._session_manager.record_usage(
                session_id=context.session_key_id,
                action_type=action,
                value_usd=Decimal(str(value_usd)),
                tx_hash=tx_hash,
                metadata=metadata,
            )
        except Exception as exc:
            logger.warning(f"Failed to record session key usage: {exc}")

    def _is_native_token(self, token_address: Optional[str]) -> bool:
        if not token_address:
            return True
        return token_address.lower() in NATIVE_TOKEN_SENTINELS

    def _extract_revert_reason(self, error: Exception) -> Optional[str]:
        message = str(error)
        match = re.search(r"0x[0-9a-fA-F]{8,}", message)
        if not match:
            return None
        data = match.group(0)
        if not data.startswith("0x08c379a0"):
            return None
        try:
            # Strip selector and offset
            start = 2 + 8 + 64  # 0x + selector + offset
            length_hex = data[start:start + 64]
            length = int(length_hex, 16)
            reason_hex = data[start + 64:start + 64 + length * 2]
            return bytes.fromhex(reason_hex).decode("utf-8", errors="replace")
        except Exception:
            return None

    async def _simulate_transaction(
        self,
        tx: PreparedTransaction,
        block_tag: str = "latest",
    ) -> None:
        call_obj: Dict[str, Any] = {
            "from": tx.from_address,
            "to": tx.to_address,
            "data": tx.data,
        }
        if tx.value:
            call_obj["value"] = hex(tx.value)

        try:
            await self._rpc_call(
                tx.chain_id,
                "eth_call",
                [call_obj, block_tag],
            )
        except Exception as e:
            reason = self._extract_revert_reason(e)
            raise TransactionRevertError(
                f"Simulation reverted: {reason or str(e)}",
                revert_reason=reason,
            )

    async def _get_allowance(
        self,
        chain_id: int,
        token_address: str,
        owner_address: str,
        spender_address: str,
    ) -> int:
        calldata = (
            ERC20_ALLOWANCE_SELECTOR
            + self._encode_address(owner_address)
            + self._encode_address(spender_address)
        )
        call_obj = {
            "from": owner_address,
            "to": token_address,
            "data": calldata,
        }
        result = await self._rpc_call(chain_id, "eth_call", [call_obj, "latest"])
        return int(result, 16) if isinstance(result, str) else int(result or 0)

    async def _ensure_allowances(
        self,
        quote: SwapQuote | BridgeQuote,
        context: ExecutionContext,
    ) -> None:
        if not quote.approvals:
            return
        if self._is_native_token(quote.token_in_address):
            return

        chain_id = quote.chain_id if hasattr(quote, "chain_id") else quote.origin_chain_id
        for approval in quote.approvals:
            data = approval.get("data", {}) if isinstance(approval, dict) else {}
            spender = data.get("spender")
            amount = data.get("amount") or data.get("value")
            if not spender or amount is None:
                continue

            if isinstance(amount, str):
                amount = int(amount, 16) if amount.startswith("0x") else int(amount)

            allowance = await self._get_allowance(
                chain_id=chain_id,
                token_address=quote.token_in_address,
                owner_address=quote.wallet_address,
                spender_address=spender,
            )
            if allowance < amount:
                raise ExecutionError(
                    "Token approval required before execution (allowance insufficient)."
                )

    async def _get_risk_policy_config(self, wallet_address: str) -> RiskPolicyConfig:
        policy_data = await self.convex.query(
            "riskPolicies:getByWallet",
            {"walletAddress": wallet_address.lower()},
        )
        if not policy_data or not policy_data.get("config"):
            raise PolicyRequiredError(
                "Risk policy required before autonomous execution. Draft a policy to enable execution."
            )
        return RiskPolicyConfig.from_dict(policy_data["config"])

    async def _get_system_policy_config(self) -> SystemPolicyConfig:
        data = await self.convex.query("systemPolicy:get", {})
        if not data:
            return SystemPolicyConfig()
        return SystemPolicyConfig(
            emergency_stop=data.get("emergencyStop", False),
            emergency_stop_reason=data.get("emergencyStopReason"),
            in_maintenance=data.get("inMaintenance", False),
            maintenance_message=data.get("maintenanceMessage"),
            blocked_contracts=data.get("blockedContracts", []),
            blocked_tokens=data.get("blockedTokens", []),
            blocked_chains=data.get("blockedChains", []),
            allowed_chains=data.get("allowedChains", []),
            protocol_whitelist_enabled=data.get("protocolWhitelistEnabled", False),
            allowed_protocols=data.get("allowedProtocols", []),
            max_single_tx_usd=Decimal(str(data.get("maxSingleTxUsd", "100000"))),
        )

    async def _get_fee_policy_config(self, chain_id: int) -> FeePolicyConfig:
        data = await self.convex.query(
            "feeConfig:getByChainId",
            {"chainId": chain_id},
        )
        if not data:
            return FeePolicyConfig.missing_for_chain(chain_id)
        return FeePolicyConfig.from_dict(data)

    async def _get_session_key(self, session_key_id: str) -> SessionKey:
        session_data = await self.convex.query(
            "sessionKeys:get",
            {"sessionId": session_key_id},
        )
        if not session_data:
            raise SessionKeyRequiredError("Session key not found for autonomous execution.")
        return SessionKey.from_dict(session_data)

    async def _enforce_policy(
        self,
        *,
        context: ExecutionContext,
        action_type: str,
        value_usd: float,
        chain_id: Optional[int] = None,
        token_in: Optional[str] = None,
        token_out: Optional[str] = None,
        contract_address: Optional[str] = None,
        slippage_percent: Optional[float] = None,
        estimated_gas_usd: Optional[float] = None,
    ) -> None:
        if not context.require_policy and not context.require_session_key and not context.require_fee_policy:
            return

        risk_config: Optional[RiskPolicyConfig] = None
        if context.require_policy:
            risk_config = await self._get_risk_policy_config(context.wallet_address)

        system_config = await self._get_system_policy_config()
        fee_config: Optional[FeePolicyConfig] = None
        require_fee_policy = context.require_fee_policy or context.require_session_key
        if require_fee_policy:
            chain_id_value = chain_id or context.chain_id
            fee_config = await self._get_fee_policy_config(chain_id_value)

        session_key: Optional[SessionKey] = None
        if context.require_session_key:
            if not context.session_key_id:
                raise SessionKeyRequiredError("Session key required for autonomous execution.")
            session_key = await self._get_session_key(context.session_key_id)

        action_context = ActionContext(
            session_id=context.session_key_id or "autonomous",
            wallet_address=context.wallet_address.lower(),
            action_type=action_type,
            chain_id=chain_id or context.chain_id,
            value_usd=Decimal(str(value_usd)),
            contract_address=contract_address,
            token_in=token_in,
            token_out=token_out,
            slippage_percent=slippage_percent,
            estimated_gas_usd=Decimal(str(estimated_gas_usd)) if estimated_gas_usd is not None else None,
        )

        result = PolicyEngine(
            session_key=session_key,
            risk_config=risk_config,
            system_config=system_config,
            fee_config=fee_config,
        ).evaluate(action_context)

        if not result.approved:
            raise PolicyViolationError(
                f"Policy rejected action: {result.error_message or 'blocked'}",
                violations=[v.to_dict() for v in result.violations],
            )

        if result.requires_approval:
            raise ApprovalRequiredError(
                "Action requires manual approval.",
                approval_reason=result.approval_reason,
            )

    async def estimate_gas(
        self,
        tx: PreparedTransaction,
        context: Optional[ExecutionContext] = None,
    ) -> GasEstimate:
        """
        Estimate gas for a transaction.

        Args:
            tx: The prepared transaction
            context: Optional execution context with gas settings

        Returns:
            GasEstimate with limit and price
        """
        try:
            # Build call object
            call_obj = {
                "from": tx.from_address,
                "to": tx.to_address,
                "data": tx.data,
            }
            if tx.value > 0:
                call_obj["value"] = hex(tx.value)

            # Estimate gas limit
            gas_limit_hex = await self._rpc_call(
                tx.chain_id,
                "eth_estimateGas",
                [call_obj],
            )
            gas_limit = int(gas_limit_hex, 16)

            # Apply safety multiplier
            multiplier = context.gas_multiplier if context else 1.1
            gas_limit = int(gas_limit * multiplier)

            # Get gas price (EIP-1559 style)
            fee_history = await self._rpc_call(
                tx.chain_id,
                "eth_feeHistory",
                [1, "latest", [50]],
            )

            base_fee = int(fee_history["baseFeePerGas"][-1], 16)
            priority_fee = int(fee_history["reward"][0][0], 16) if fee_history.get("reward") else 1_000_000_000

            # Apply max caps if specified
            if context and context.max_priority_fee_gwei:
                priority_fee = min(priority_fee, int(context.max_priority_fee_gwei * 1e9))

            max_fee = base_fee * 2 + priority_fee
            if context and context.max_gas_price_gwei:
                max_fee = min(max_fee, int(context.max_gas_price_gwei * 1e9))

            # Estimate cost (rough)
            estimated_cost = gas_limit * max_fee

            return GasEstimate(
                gas_limit=gas_limit,
                gas_price_wei=max_fee,
                max_fee_per_gas=max_fee,
                max_priority_fee_per_gas=priority_fee,
                estimated_cost_wei=estimated_cost,
            )

        except Exception as e:
            logger.error(f"Gas estimation failed: {e}")
            raise GasEstimationError(f"Failed to estimate gas: {e}")

    async def execute_swap(
        self,
        quote: SwapQuote,
        context: ExecutionContext,
        signed_tx: Optional[str] = None,
    ) -> TransactionResult:
        """
        Execute a swap transaction.

        Args:
            quote: The swap quote
            context: Execution context
            signed_tx: Pre-signed transaction (if available)

        Returns:
            TransactionResult with status and details
        """
        await self._enforce_policy(
            context=context,
            action_type=TransactionType.SWAP.value,
            chain_id=quote.chain_id,
            value_usd=quote.value_in_usd,
            token_in=quote.token_in_address,
            token_out=quote.token_out_address,
            contract_address=quote.tx.get("to") if isinstance(quote.tx, dict) else None,
            slippage_percent=quote.slippage_bps / 100.0,
            estimated_gas_usd=quote.gas_fee_usd,
        )
        if quote.signatures:
            raise SignatureRequiredError(
                "Quote requires off-chain signatures before execution."
            )
        await self._ensure_allowances(quote, context)

        # Build transaction from quote
        tx = TransactionBuilder.build_from_swap_quote(quote)

        # Record in database
        db_tx_id = await self._create_db_transaction(
            tx=tx,
            context=context,
            input_data={
                "quote_id": quote.request_id,
                "token_in": quote.token_in_symbol,
                "token_out": quote.token_out_symbol,
                "amount_in": quote.amount_in,
                "amount_out_estimate": quote.amount_out_estimate,
                "value_usd": quote.value_in_usd,
            },
        )

        try:
            result = await self._execute_transaction(
                tx=tx,
                context=context,
                signed_tx=signed_tx,
            )

            # Update database
            await self._update_db_transaction(db_tx_id, result)
            if result.status in (TransactionStatus.SUBMITTED, TransactionStatus.CONFIRMED):
                await self._record_session_usage(
                    context=context,
                    action=Permission.SWAP,
                    value_usd=quote.value_in_usd,
                    tx_hash=result.tx_hash,
                    metadata={"quoteId": quote.request_id},
                )
            return result

        except Exception as e:
            # Update database with error
            await self._update_db_transaction(
                db_tx_id,
                TransactionResult(
                    tx_id=tx.tx_id,
                    status=TransactionStatus.FAILED,
                    chain_id=tx.chain_id,
                    error=str(e),
                ),
            )
            raise

    async def execute_bridge(
        self,
        quote: BridgeQuote,
        context: ExecutionContext,
        signed_tx: Optional[str] = None,
    ) -> TransactionResult:
        """
        Execute a bridge transaction.

        Args:
            quote: The bridge quote
            context: Execution context
            signed_tx: Pre-signed transaction (if available)

        Returns:
            TransactionResult with status and details
        """
        await self._enforce_policy(
            context=context,
            action_type=TransactionType.BRIDGE.value,
            chain_id=quote.origin_chain_id,
            value_usd=quote.value_in_usd,
            token_in=quote.token_in_address,
            token_out=quote.token_out_address,
            contract_address=quote.tx.get("to") if isinstance(quote.tx, dict) else None,
            slippage_percent=quote.slippage_bps / 100.0,
            estimated_gas_usd=quote.gas_fee_usd,
        )
        if quote.signatures:
            raise SignatureRequiredError(
                "Quote requires off-chain signatures before execution."
            )
        await self._ensure_allowances(quote, context)

        # Build transaction from quote
        tx = TransactionBuilder.build_from_bridge_quote(quote)

        # Record in database
        db_tx_id = await self._create_db_transaction(
            tx=tx,
            context=context,
            input_data={
                "quote_id": quote.request_id,
                "origin_chain": quote.origin_chain_id,
                "destination_chain": quote.destination_chain_id,
                "token_in": quote.token_in_symbol,
                "token_out": quote.token_out_symbol,
                "amount_in": quote.amount_in,
                "amount_out_estimate": quote.amount_out_estimate,
                "value_usd": quote.value_in_usd,
            },
        )

        try:
            result = await self._execute_transaction(
                tx=tx,
                context=context,
                signed_tx=signed_tx,
            )

            # Update database
            await self._update_db_transaction(db_tx_id, result)
            if result.status in (TransactionStatus.SUBMITTED, TransactionStatus.CONFIRMED):
                await self._record_session_usage(
                    context=context,
                    action=Permission.BRIDGE,
                    value_usd=quote.value_in_usd,
                    tx_hash=result.tx_hash,
                    metadata={"quoteId": quote.request_id},
                )
            return result

        except Exception as e:
            await self._update_db_transaction(
                db_tx_id,
                TransactionResult(
                    tx_id=tx.tx_id,
                    status=TransactionStatus.FAILED,
                    chain_id=tx.chain_id,
                    error=str(e),
                ),
            )
            raise

    async def execute_approval(
        self,
        chain_id: int,
        owner_address: str,
        token_address: str,
        spender_address: str,
        amount: int,
        context: ExecutionContext,
        signed_tx: Optional[str] = None,
    ) -> TransactionResult:
        """
        Execute an ERC20 approval transaction.

        Args:
            chain_id: The chain ID
            owner_address: Token owner
            token_address: ERC20 token contract
            spender_address: Address to approve
            amount: Amount to approve
            context: Execution context
            signed_tx: Pre-signed transaction

        Returns:
            TransactionResult with status and details
        """
        await self._enforce_policy(
            context=context,
            action_type=TransactionType.APPROVE.value,
            chain_id=chain_id,
            value_usd=0.0,
            token_in=token_address,
            contract_address=token_address,
            slippage_percent=None,
            estimated_gas_usd=None,
        )

        tx = TransactionBuilder.build_erc20_approve(
            chain_id=chain_id,
            owner_address=owner_address,
            token_address=token_address,
            spender_address=spender_address,
            amount=amount,
        )
        db_tx_id = await self._create_db_transaction(
            tx=tx,
            context=context,
            input_data={
                "owner": owner_address,
                "token": token_address,
                "spender": spender_address,
                "amount": amount,
            },
        )

        try:
            result = await self._execute_transaction(
                tx=tx,
                context=context,
                signed_tx=signed_tx,
            )
            await self._update_db_transaction(db_tx_id, result)
            if result.status in (TransactionStatus.SUBMITTED, TransactionStatus.CONFIRMED):
                await self._record_session_usage(
                    context=context,
                    action=Permission.APPROVE,
                    value_usd=0.0,
                    tx_hash=result.tx_hash,
                    metadata={
                        "token": token_address,
                        "spender": spender_address,
                        "amount": amount,
                    },
                )
            return result
        except Exception as e:
            await self._update_db_transaction(
                db_tx_id,
                TransactionResult(
                    tx_id=tx.tx_id,
                    status=TransactionStatus.FAILED,
                    chain_id=tx.chain_id,
                    error=str(e),
                ),
            )
            raise

    async def _execute_transaction(
        self,
        tx: PreparedTransaction,
        context: ExecutionContext,
        signed_tx: Optional[str] = None,
    ) -> TransactionResult:
        """
        Internal method to execute a prepared transaction.

        Note: This requires a signed transaction. In a real implementation,
        the signing would happen on the client side (wallet) or via
        a secure key management system.
        """
        if context.use_erc4337:
            return await self._execute_userop(tx=tx, context=context, signed_tx=signed_tx)

        result = TransactionResult(
            tx_id=tx.tx_id,
            chain_id=tx.chain_id,
            status=TransactionStatus.PENDING,
        )

        try:
            # Estimate gas if not already set
            if not tx.gas_estimate:
                tx.gas_estimate = await self.estimate_gas(tx, context)

            # Get nonce if not set
            if tx.nonce is None:
                if context.nonce_override is not None:
                    tx.nonce = context.nonce_override
                else:
                    tx.nonce = await self.nonce_manager.get_next_nonce(
                        address=tx.from_address,
                        chain_id=tx.chain_id,
                    )

            if context.simulate:
                await self._simulate_transaction(tx)

            # If we have a pre-signed transaction, submit it
            if signed_tx:
                tx_hash = await self._submit_raw_transaction(tx.chain_id, signed_tx)
                result.tx_hash = tx_hash
                result.status = TransactionStatus.SUBMITTED
                result.submitted_at = datetime.utcnow()

                # Monitor for confirmation
                result = await self._monitor_transaction(
                    chain_id=tx.chain_id,
                    tx_hash=tx_hash,
                    timeout=context.confirmation_timeout_seconds,
                    required_confirmations=context.required_confirmations,
                    initial_result=result,
                )

                # Confirm nonce if successful
                if result.status == TransactionStatus.CONFIRMED:
                    await self.nonce_manager.confirm_nonce(
                        address=tx.from_address,
                        chain_id=tx.chain_id,
                        nonce=tx.nonce,
                    )
                else:
                    # Release nonce on failure
                    await self.nonce_manager.release_nonce(
                        address=tx.from_address,
                        chain_id=tx.chain_id,
                        nonce=tx.nonce,
                    )
            else:
                # No signed transaction - return prepared state
                # Client must sign and call submit separately
                result.status = TransactionStatus.PENDING
                logger.info(
                    f"Transaction prepared: {tx.tx_id}, "
                    f"nonce={tx.nonce}, gas={tx.gas_estimate.gas_limit}"
                )

            return result

        except Exception as e:
            logger.error(f"Transaction execution failed: {e}")

            # Release nonce on error
            if tx.nonce is not None:
                await self.nonce_manager.release_nonce(
                    address=tx.from_address,
                    chain_id=tx.chain_id,
                    nonce=tx.nonce,
                )

            result.status = TransactionStatus.FAILED
            result.error = str(e)
            return result

    async def _execute_userop(
        self,
        tx: PreparedTransaction,
        context: ExecutionContext,
        signed_tx: Optional[str] = None,
    ) -> TransactionResult:
        if not settings.enable_erc4337:
            raise ExecutionError("ERC-4337 execution is disabled in settings")

        sender = context.smart_wallet_address or context.wallet_address or tx.from_address
        if not sender:
            raise ExecutionError("Smart wallet address is required for ERC-4337 execution")

        if not tx.gas_estimate:
            tx.gas_estimate = await self.estimate_gas(tx, context)

        max_fee_per_gas = (
            tx.gas_estimate.max_fee_per_gas
            if tx.gas_estimate.max_fee_per_gas is not None
            else tx.gas_estimate.gas_price_wei
        )
        max_priority_fee_per_gas = tx.gas_estimate.max_priority_fee_per_gas or 0

        nonce_key = context.user_op_nonce_key or 0
        nonce = await self._get_userop_nonce(tx.chain_id, sender, nonce_key)

        call_data = build_execute_call_data(
            to_address=tx.to_address,
            value_wei=tx.value,
            data=tx.data,
        )

        user_op = UserOperation(
            sender=sender,
            nonce=nonce,
            init_code="0x",
            call_data=call_data,
            call_gas_limit=0,
            verification_gas_limit=0,
            pre_verification_gas=0,
            max_fee_per_gas=max_fee_per_gas,
            max_priority_fee_per_gas=max_priority_fee_per_gas,
            paymaster_and_data="0x",
            signature="0x",
        )

        fee_config = await self._get_fee_policy_config(tx.chain_id)
        try:
            user_op = await self._userop_executor.sponsor_user_operation(user_op, fee_config)
        except UserOpExecutionError as exc:
            raise ExecutionError(f"Paymaster sponsorship failed: {exc}") from exc

        try:
            gas_estimate = await self._userop_executor.estimate_gas(user_op)
        except UserOpExecutionError as exc:
            raise ExecutionError(f"UserOp gas estimation failed: {exc}") from exc

        user_op.call_gas_limit = gas_estimate.call_gas_limit
        user_op.verification_gas_limit = gas_estimate.verification_gas_limit
        user_op.pre_verification_gas = gas_estimate.pre_verification_gas

        signature = context.user_op_signature or signed_tx
        if not signature:
            raise SignatureRequiredError("UserOperation signature required for ERC-4337 execution.")
        user_op.signature = signature

        result = TransactionResult(
            tx_id=tx.tx_id,
            chain_id=tx.chain_id,
            status=TransactionStatus.PENDING,
        )

        if context.simulate:
            return result

        try:
            send_result = await self._userop_executor.send_user_operation(user_op)
        except UserOpExecutionError as exc:
            result.status = TransactionStatus.FAILED
            result.error = str(exc)
            return result

        result.tx_hash = send_result.user_op_hash
        result.status = TransactionStatus.SUBMITTED
        result.submitted_at = datetime.utcnow()
        return result

    async def _submit_raw_transaction(
        self,
        chain_id: int,
        signed_tx: str,
    ) -> str:
        """Submit a signed transaction to the network."""
        tx_hash = await self._rpc_call(
            chain_id,
            "eth_sendRawTransaction",
            [signed_tx],
        )
        logger.info(f"Transaction submitted: {tx_hash}")
        return tx_hash

    async def monitor_transaction(
        self,
        chain_id: int,
        tx_hash: str,
        timeout_seconds: int = 300,
        required_confirmations: int = 1,
    ) -> TransactionResult:
        """
        Monitor a transaction until confirmation or timeout.

        Args:
            chain_id: The chain ID
            tx_hash: The transaction hash to monitor
            timeout_seconds: Maximum time to wait
            required_confirmations: Number of confirmations needed

        Returns:
            TransactionResult with final status
        """
        result = TransactionResult(
            tx_id=tx_hash,
            tx_hash=tx_hash,
            chain_id=chain_id,
            status=TransactionStatus.SUBMITTED,
        )

        return await self._monitor_transaction(
            chain_id=chain_id,
            tx_hash=tx_hash,
            timeout=timeout_seconds,
            required_confirmations=required_confirmations,
            initial_result=result,
        )

    async def _monitor_transaction(
        self,
        chain_id: int,
        tx_hash: str,
        timeout: int,
        required_confirmations: int,
        initial_result: TransactionResult,
    ) -> TransactionResult:
        """Internal monitoring implementation."""
        result = initial_result
        start_time = datetime.utcnow()
        poll_interval = 2  # seconds

        while True:
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            if elapsed > timeout:
                result.status = TransactionStatus.TIMEOUT
                result.error = f"Confirmation timeout after {timeout}s"
                return result

            try:
                receipt = await self._rpc_call(
                    chain_id,
                    "eth_getTransactionReceipt",
                    [tx_hash],
                )

                if receipt:
                    result.block_number = int(receipt["blockNumber"], 16)
                    result.block_hash = receipt["blockHash"]
                    result.gas_used = int(receipt["gasUsed"], 16)
                    result.effective_gas_price = int(receipt.get("effectiveGasPrice", "0x0"), 16)

                    # Check status (0x1 = success, 0x0 = revert)
                    status = int(receipt.get("status", "0x1"), 16)
                    if status == 0:
                        result.status = TransactionStatus.REVERTED
                        result.error = "Transaction reverted"
                        return result

                    # Check confirmations
                    current_block = await self._rpc_call(
                        chain_id,
                        "eth_blockNumber",
                        [],
                    )
                    current_block_num = int(current_block, 16)
                    confirmations = current_block_num - result.block_number + 1

                    if confirmations >= required_confirmations:
                        result.status = TransactionStatus.CONFIRMED
                        result.confirmed_at = datetime.utcnow()
                        logger.info(
                            f"Transaction confirmed: {tx_hash} "
                            f"(block {result.block_number}, {confirmations} confirmations)"
                        )
                        return result

                    result.status = TransactionStatus.CONFIRMING

            except Exception as e:
                logger.warning(f"Error checking transaction status: {e}")

            await asyncio.sleep(poll_interval)

    async def _create_db_transaction(
        self,
        tx: PreparedTransaction,
        context: ExecutionContext,
        input_data: Dict[str, Any],
    ) -> Optional[str]:
        """Create a transaction record in Convex."""
        try:
            if context.wallet_id:
                return await self.convex.create_transaction(
                    wallet_id=context.wallet_id,
                    chain=str(tx.chain_id),
                    tx_type=tx.tx_type.value,
                    input_data=input_data,
                    execution_id=context.execution_id,
                    value_usd=input_data.get("value_usd"),
                )
        except Exception as e:
            logger.warning(f"Failed to create DB transaction: {e}")
        return None

    async def _update_db_transaction(
        self,
        db_tx_id: Optional[str],
        result: TransactionResult,
    ) -> None:
        """Update a transaction record in Convex."""
        if not db_tx_id:
            return

        try:
            await self.convex.update_transaction(
                transaction_id=db_tx_id,
                status=result.status.value,
                tx_hash=result.tx_hash,
                output_data=result.output_data,
                gas_used=result.gas_used,
                gas_price=result.effective_gas_price,
            )
        except Exception as e:
            logger.warning(f"Failed to update DB transaction: {e}")

    async def close(self) -> None:
        """Close HTTP client."""
        await self._client.aclose()


# Singleton instance
_executor: Optional[TransactionExecutor] = None


def get_transaction_executor() -> TransactionExecutor:
    """Get the singleton transaction executor instance."""
    global _executor
    if _executor is None:
        _executor = TransactionExecutor()
    return _executor
