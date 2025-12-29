"""
Error Classification

Defines error types for the recovery system.
Errors are classified as recoverable (can retry) or unrecoverable (needs human).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ErrorCategory(str, Enum):
    """Categories of errors for recovery decisions."""

    NETWORK = "network"           # Network/connectivity issues
    RATE_LIMIT = "rate_limit"     # API rate limits
    TIMEOUT = "timeout"           # Operation timed out
    INSUFFICIENT_FUNDS = "insufficient_funds"  # Not enough balance
    TRANSACTION_REVERTED = "transaction_reverted"  # On-chain revert
    SLIPPAGE = "slippage"         # Slippage exceeded
    CONTRACT = "contract"         # Smart contract error
    PROVIDER = "provider"         # External provider error
    VALIDATION = "validation"     # Input validation error
    AUTHENTICATION = "authentication"  # Auth/permission error
    UNKNOWN = "unknown"           # Unclassified error


@dataclass
class ErrorContext:
    """Additional context about an error."""

    category: ErrorCategory = ErrorCategory.UNKNOWN
    recoverable: bool = True
    retry_after_seconds: Optional[float] = None
    suggested_action: Optional[str] = None
    provider: Optional[str] = None
    chain_id: Optional[int] = None
    tx_hash: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class RecoverableError(Exception):
    """
    Base class for errors that can be retried.

    These errors are typically transient:
    - Network issues
    - Rate limits
    - Timeouts
    - Temporary provider outages
    """

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        retry_after: Optional[float] = None,
        context: Optional[ErrorContext] = None,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.retry_after = retry_after
        self.context = context or ErrorContext(category=category, recoverable=True)


class UnrecoverableError(Exception):
    """
    Base class for errors that cannot be retried.

    These errors require human intervention:
    - Insufficient funds
    - Transaction reverts
    - Invalid contract calls
    - Permission denied
    """

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        context: Optional[ErrorContext] = None,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.context = context or ErrorContext(category=category, recoverable=False)


# Specific recoverable errors
class RateLimitError(RecoverableError):
    """API rate limit exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: float = 60.0,
        provider: Optional[str] = None,
    ):
        super().__init__(
            message,
            category=ErrorCategory.RATE_LIMIT,
            retry_after=retry_after,
            context=ErrorContext(
                category=ErrorCategory.RATE_LIMIT,
                recoverable=True,
                retry_after_seconds=retry_after,
                provider=provider,
                suggested_action=f"Wait {retry_after}s before retrying",
            ),
        )


class NetworkError(RecoverableError):
    """Network connectivity error."""

    def __init__(
        self,
        message: str = "Network error",
        provider: Optional[str] = None,
    ):
        super().__init__(
            message,
            category=ErrorCategory.NETWORK,
            retry_after=5.0,
            context=ErrorContext(
                category=ErrorCategory.NETWORK,
                recoverable=True,
                retry_after_seconds=5.0,
                provider=provider,
                suggested_action="Retry with exponential backoff",
            ),
        )


class TimeoutError(RecoverableError):
    """Operation timed out."""

    def __init__(
        self,
        message: str = "Operation timed out",
        operation: Optional[str] = None,
    ):
        super().__init__(
            message,
            category=ErrorCategory.TIMEOUT,
            retry_after=10.0,
            context=ErrorContext(
                category=ErrorCategory.TIMEOUT,
                recoverable=True,
                retry_after_seconds=10.0,
                suggested_action="Retry with longer timeout",
                details={"operation": operation} if operation else {},
            ),
        )


# Specific unrecoverable errors
class InsufficientFundsError(UnrecoverableError):
    """Wallet has insufficient funds."""

    def __init__(
        self,
        message: str = "Insufficient funds",
        required: Optional[str] = None,
        available: Optional[str] = None,
        token: Optional[str] = None,
    ):
        super().__init__(
            message,
            category=ErrorCategory.INSUFFICIENT_FUNDS,
            context=ErrorContext(
                category=ErrorCategory.INSUFFICIENT_FUNDS,
                recoverable=False,
                suggested_action="Add funds to wallet or reduce transaction amount",
                details={
                    "required": required,
                    "available": available,
                    "token": token,
                },
            ),
        )


class TransactionRevertedError(UnrecoverableError):
    """Transaction reverted on-chain."""

    def __init__(
        self,
        message: str = "Transaction reverted",
        tx_hash: Optional[str] = None,
        reason: Optional[str] = None,
        chain_id: Optional[int] = None,
    ):
        super().__init__(
            message,
            category=ErrorCategory.TRANSACTION_REVERTED,
            context=ErrorContext(
                category=ErrorCategory.TRANSACTION_REVERTED,
                recoverable=False,
                tx_hash=tx_hash,
                chain_id=chain_id,
                suggested_action="Review transaction parameters",
                details={"revert_reason": reason} if reason else {},
            ),
        )


class SlippageExceededError(RecoverableError):
    """
    Slippage exceeded tolerance.

    This is recoverable because the user can retry with higher slippage
    or wait for better market conditions.
    """

    def __init__(
        self,
        message: str = "Slippage exceeded",
        expected: Optional[str] = None,
        actual: Optional[str] = None,
        tolerance: Optional[float] = None,
    ):
        super().__init__(
            message,
            category=ErrorCategory.SLIPPAGE,
            retry_after=30.0,  # Wait for market conditions to stabilize
            context=ErrorContext(
                category=ErrorCategory.SLIPPAGE,
                recoverable=True,
                retry_after_seconds=30.0,
                suggested_action="Retry with higher slippage tolerance or wait",
                details={
                    "expected": expected,
                    "actual": actual,
                    "tolerance": tolerance,
                },
            ),
        )


class ContractError(UnrecoverableError):
    """Smart contract execution error."""

    def __init__(
        self,
        message: str = "Contract error",
        contract_address: Optional[str] = None,
        function_name: Optional[str] = None,
        error_data: Optional[str] = None,
    ):
        super().__init__(
            message,
            category=ErrorCategory.CONTRACT,
            context=ErrorContext(
                category=ErrorCategory.CONTRACT,
                recoverable=False,
                suggested_action="Review contract call parameters",
                details={
                    "contract": contract_address,
                    "function": function_name,
                    "error_data": error_data,
                },
            ),
        )


def classify_error(error: Exception) -> ErrorContext:
    """
    Classify an exception and return its error context.

    This function attempts to classify generic exceptions based on
    their message and type.
    """
    message = str(error).lower()

    # Check if it's already a classified error
    if isinstance(error, (RecoverableError, UnrecoverableError)):
        return error.context

    # Rate limit patterns
    rate_limit_patterns = [
        "rate limit",
        "too many requests",
        "429",
        "throttl",
        "quota exceeded",
    ]
    if any(p in message for p in rate_limit_patterns):
        return ErrorContext(
            category=ErrorCategory.RATE_LIMIT,
            recoverable=True,
            retry_after_seconds=60.0,
            suggested_action="Wait before retrying",
        )

    # Network patterns
    network_patterns = [
        "connection",
        "network",
        "unreachable",
        "refused",
        "dns",
        "socket",
        "ssl",
        "certificate",
    ]
    if any(p in message for p in network_patterns):
        return ErrorContext(
            category=ErrorCategory.NETWORK,
            recoverable=True,
            retry_after_seconds=5.0,
            suggested_action="Check network connectivity",
        )

    # Timeout patterns
    timeout_patterns = ["timeout", "timed out", "deadline"]
    if any(p in message for p in timeout_patterns):
        return ErrorContext(
            category=ErrorCategory.TIMEOUT,
            recoverable=True,
            retry_after_seconds=10.0,
            suggested_action="Retry with longer timeout",
        )

    # Insufficient funds patterns
    funds_patterns = [
        "insufficient",
        "not enough",
        "balance too low",
        "insufficient funds",
        "exceeds balance",
    ]
    if any(p in message for p in funds_patterns):
        return ErrorContext(
            category=ErrorCategory.INSUFFICIENT_FUNDS,
            recoverable=False,
            suggested_action="Add funds to wallet",
        )

    # Revert patterns
    revert_patterns = [
        "revert",
        "execution reverted",
        "transaction failed",
        "out of gas",
    ]
    if any(p in message for p in revert_patterns):
        return ErrorContext(
            category=ErrorCategory.TRANSACTION_REVERTED,
            recoverable=False,
            suggested_action="Review transaction parameters",
        )

    # Slippage patterns
    slippage_patterns = ["slippage", "price impact", "price changed"]
    if any(p in message for p in slippage_patterns):
        return ErrorContext(
            category=ErrorCategory.SLIPPAGE,
            recoverable=True,
            retry_after_seconds=30.0,
            suggested_action="Retry with higher slippage tolerance",
        )

    # Default to unknown but recoverable (safer to retry)
    return ErrorContext(
        category=ErrorCategory.UNKNOWN,
        recoverable=True,
        suggested_action="Retry operation",
    )
