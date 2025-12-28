"""
Transaction Execution Layer

Provides the infrastructure for executing on-chain transactions:
- TransactionExecutor: Main class for executing swaps, bridges, approvals
- NonceManager: Manages nonces for concurrent transactions
- TransactionBuilder: Builds transactions from quotes and raw data

Usage:
    from app.core.execution import (
        TransactionExecutor,
        get_transaction_executor,
        TransactionBuilder,
        NonceManager,
    )

    # Get the singleton executor
    executor = get_transaction_executor()

    # Execute a swap from a quote
    result = await executor.execute_swap(quote, context)

    # Build custom transactions
    tx = TransactionBuilder.build_erc20_approve(
        chain_id=1,
        owner_address="0x...",
        token_address="0x...",
        spender_address="0x...",
    )
"""

from .models import (
    TransactionType,
    TransactionStatus,
    GasEstimate,
    PreparedTransaction,
    TransactionResult,
    SwapQuote,
    BridgeQuote,
    ExecutionContext,
)

from .nonce_manager import (
    NonceManager,
    NonceState,
    get_nonce_manager,
)

from .tx_builder import (
    TransactionBuilder,
)

from .executor import (
    TransactionExecutor,
    ExecutionError,
    GasEstimationError,
    TransactionSubmitError,
    TransactionRevertError,
    TransactionTimeoutError,
    get_transaction_executor,
)

from .solana_executor import (
    SolanaExecutor,
    SolanaRpcConfig,
    SolanaTransactionResult,
    SolanaTransactionStatus,
    SolanaExecutorError,
    get_solana_executor,
)

__all__ = [
    # Models
    "TransactionType",
    "TransactionStatus",
    "GasEstimate",
    "PreparedTransaction",
    "TransactionResult",
    "SwapQuote",
    "BridgeQuote",
    "ExecutionContext",
    # Nonce Manager
    "NonceManager",
    "NonceState",
    "get_nonce_manager",
    # Transaction Builder
    "TransactionBuilder",
    # Executor
    "TransactionExecutor",
    "ExecutionError",
    "GasEstimationError",
    "TransactionSubmitError",
    "TransactionRevertError",
    "TransactionTimeoutError",
    "get_transaction_executor",
    # Solana Executor
    "SolanaExecutor",
    "SolanaRpcConfig",
    "SolanaTransactionResult",
    "SolanaTransactionStatus",
    "SolanaExecutorError",
    "get_solana_executor",
]
