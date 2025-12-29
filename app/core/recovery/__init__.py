"""
Error Recovery Module

Provides error classification, retry logic, and recovery strategies
for resilient execution of blockchain operations.
"""

from .errors import (
    RecoverableError,
    UnrecoverableError,
    RateLimitError,
    NetworkError,
    TimeoutError,
    InsufficientFundsError,
    TransactionRevertedError,
    SlippageExceededError,
    ContractError,
    classify_error,
)
from .executor import RecoveryExecutor, ExecutionResult, RecoveryConfig, execute_with_recovery
from .strategies import (
    RecoveryStrategy,
    RetryStrategy,
    ExponentialBackoffStrategy,
    CircuitBreakerStrategy,
    FallbackStrategy,
)

__all__ = [
    # Errors
    "RecoverableError",
    "UnrecoverableError",
    "RateLimitError",
    "NetworkError",
    "TimeoutError",
    "InsufficientFundsError",
    "TransactionRevertedError",
    "SlippageExceededError",
    "ContractError",
    "classify_error",
    # Executor
    "RecoveryExecutor",
    "RecoveryConfig",
    "ExecutionResult",
    "execute_with_recovery",
    # Strategies
    "RecoveryStrategy",
    "RetryStrategy",
    "ExponentialBackoffStrategy",
    "CircuitBreakerStrategy",
    "FallbackStrategy",
]
