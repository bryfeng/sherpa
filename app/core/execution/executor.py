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
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.config import settings
from app.db.convex_client import get_convex_client, ConvexClient

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
from .nonce_manager import NonceManager, get_nonce_manager
from .tx_builder import TransactionBuilder


logger = logging.getLogger(__name__)


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
    ):
        self.nonce_manager = nonce_manager or get_nonce_manager()
        self.convex = convex or get_convex_client()
        self._client = httpx.AsyncClient(timeout=60.0)

        # RPC URLs per chain
        self._rpc_urls = rpc_urls or {
            1: f"https://eth-mainnet.g.alchemy.com/v2/{settings.alchemy_api_key}",
            10: f"https://opt-mainnet.g.alchemy.com/v2/{settings.alchemy_api_key}",
            137: f"https://polygon-mainnet.g.alchemy.com/v2/{settings.alchemy_api_key}",
            42161: f"https://arb-mainnet.g.alchemy.com/v2/{settings.alchemy_api_key}",
            8453: f"https://base-mainnet.g.alchemy.com/v2/{settings.alchemy_api_key}",
        }

    async def _rpc_call(
        self,
        chain_id: int,
        method: str,
        params: List[Any],
    ) -> Any:
        """Make an RPC call to the chain."""
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
        tx = TransactionBuilder.build_erc20_approve(
            chain_id=chain_id,
            owner_address=owner_address,
            token_address=token_address,
            spender_address=spender_address,
            amount=amount,
        )

        return await self._execute_transaction(
            tx=tx,
            context=context,
            signed_tx=signed_tx,
        )

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
