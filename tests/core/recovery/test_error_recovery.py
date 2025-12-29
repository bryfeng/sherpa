"""
Tests for the Error Recovery System

Tests for error classification, retry strategies, and recovery executor.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.core.recovery import (
    # Errors
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
    # Executor
    RecoveryExecutor,
    RecoveryConfig,
    ExecutionResult,
    execute_with_recovery,
    # Strategies
    RetryStrategy,
    ExponentialBackoffStrategy,
    CircuitBreakerStrategy,
    FallbackStrategy,
)
from app.core.recovery.errors import ErrorCategory, ErrorContext
from app.core.recovery.strategies import RetryConfig, CircuitBreakerConfig, CircuitState


# =============================================================================
# Error Classification Tests
# =============================================================================

class TestErrorClassification:
    """Tests for error classification."""

    def test_recoverable_error_is_recoverable(self):
        """Test that RecoverableError is classified as recoverable."""
        error = RecoverableError("Test error")
        assert error.context.recoverable is True

    def test_unrecoverable_error_is_not_recoverable(self):
        """Test that UnrecoverableError is classified as not recoverable."""
        error = UnrecoverableError("Test error")
        assert error.context.recoverable is False

    def test_rate_limit_error(self):
        """Test RateLimitError properties."""
        error = RateLimitError(retry_after=30.0, provider="alchemy")

        assert error.category == ErrorCategory.RATE_LIMIT
        assert error.retry_after == 30.0
        assert error.context.provider == "alchemy"

    def test_network_error(self):
        """Test NetworkError properties."""
        error = NetworkError(provider="coingecko")

        assert error.category == ErrorCategory.NETWORK
        assert error.context.recoverable is True

    def test_insufficient_funds_error(self):
        """Test InsufficientFundsError properties."""
        error = InsufficientFundsError(
            required="100 ETH",
            available="10 ETH",
            token="ETH",
        )

        assert error.category == ErrorCategory.INSUFFICIENT_FUNDS
        assert error.context.recoverable is False
        assert error.context.details["required"] == "100 ETH"

    def test_transaction_reverted_error(self):
        """Test TransactionRevertedError properties."""
        error = TransactionRevertedError(
            tx_hash="0x123",
            reason="Slippage too high",
            chain_id=1,
        )

        assert error.category == ErrorCategory.TRANSACTION_REVERTED
        assert error.context.tx_hash == "0x123"

    def test_classify_rate_limit_error(self):
        """Test classification of rate limit errors."""
        error = Exception("429 Too Many Requests")
        context = classify_error(error)

        assert context.category == ErrorCategory.RATE_LIMIT
        assert context.recoverable is True

    def test_classify_network_error(self):
        """Test classification of network errors."""
        error = Exception("Connection refused")
        context = classify_error(error)

        assert context.category == ErrorCategory.NETWORK
        assert context.recoverable is True

    def test_classify_insufficient_funds(self):
        """Test classification of insufficient funds errors."""
        error = Exception("insufficient funds for transfer")
        context = classify_error(error)

        assert context.category == ErrorCategory.INSUFFICIENT_FUNDS
        assert context.recoverable is False

    def test_classify_revert_error(self):
        """Test classification of revert errors."""
        error = Exception("execution reverted: SLIPPAGE")
        context = classify_error(error)

        assert context.category == ErrorCategory.TRANSACTION_REVERTED
        assert context.recoverable is False

    def test_classify_unknown_error(self):
        """Test classification of unknown errors."""
        error = Exception("Something weird happened")
        context = classify_error(error)

        assert context.category == ErrorCategory.UNKNOWN
        # Unknown defaults to recoverable (safer to retry)
        assert context.recoverable is True


# =============================================================================
# Retry Strategy Tests
# =============================================================================

class TestRetryStrategy:
    """Tests for retry strategies."""

    @pytest.mark.asyncio
    async def test_successful_operation(self):
        """Test successful operation doesn't retry."""
        strategy = RetryStrategy(RetryConfig(max_attempts=3))

        call_count = 0
        async def operation():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await strategy.execute(operation)

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_recoverable_error(self):
        """Test retry on recoverable error."""
        strategy = RetryStrategy(RetryConfig(
            max_attempts=3,
            initial_delay_seconds=0.01,
        ))

        call_count = 0
        async def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkError("Temporary failure")
            return "success"

        result = await strategy.execute(operation)

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_unrecoverable(self):
        """Test no retry on unrecoverable error."""
        strategy = RetryStrategy(RetryConfig(max_attempts=3))

        call_count = 0
        async def operation():
            nonlocal call_count
            call_count += 1
            raise InsufficientFundsError("Not enough ETH")

        with pytest.raises(InsufficientFundsError):
            await strategy.execute(operation)

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        """Test error when all retries exhausted."""
        strategy = RetryStrategy(RetryConfig(
            max_attempts=3,
            initial_delay_seconds=0.01,
        ))

        async def operation():
            raise NetworkError("Always fails")

        with pytest.raises(NetworkError):
            await strategy.execute(operation)

    def test_should_retry_recoverable(self):
        """Test should_retry for recoverable errors."""
        strategy = RetryStrategy(RetryConfig(max_attempts=3))

        assert strategy.should_retry(NetworkError("test"), attempt=0) is True
        assert strategy.should_retry(NetworkError("test"), attempt=2) is False
        assert strategy.should_retry(InsufficientFundsError("test"), attempt=0) is False


# =============================================================================
# Exponential Backoff Tests
# =============================================================================

class TestExponentialBackoff:
    """Tests for exponential backoff strategy."""

    @pytest.mark.asyncio
    async def test_backoff_increases(self):
        """Test that backoff delay increases."""
        strategy = ExponentialBackoffStrategy(
            max_attempts=5,
            initial_delay=0.01,
            max_delay=1.0,
        )

        config = strategy.config
        delays = [config.get_delay(i) for i in range(5)]

        # Each delay should be roughly 2x the previous (with jitter)
        for i in range(1, 4):
            # Allow for jitter variation
            assert delays[i] >= delays[i-1] * 0.8

    def test_max_delay_enforced(self):
        """Test that max delay is enforced."""
        config = RetryConfig(
            initial_delay_seconds=1.0,
            max_delay_seconds=10.0,
            exponential_base=2.0,
        )

        delay = config.get_delay(100)  # Very high attempt number
        assert delay <= 10.0 * 1.1  # Allow for jitter


# =============================================================================
# Circuit Breaker Tests
# =============================================================================

class TestCircuitBreaker:
    """Tests for circuit breaker strategy."""

    @pytest.mark.asyncio
    async def test_starts_closed(self):
        """Test circuit breaker starts closed."""
        circuit = CircuitBreakerStrategy(
            "test",
            CircuitBreakerConfig(failure_threshold=3),
        )

        assert circuit.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_opens_after_failures(self):
        """Test circuit opens after threshold failures."""
        circuit = CircuitBreakerStrategy(
            "test",
            CircuitBreakerConfig(failure_threshold=3),
        )

        async def failing_op():
            raise NetworkError("fail")

        for _ in range(3):
            with pytest.raises(NetworkError):
                await circuit.execute(failing_op)

        assert circuit.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_open_rejects_requests(self):
        """Test open circuit rejects requests."""
        circuit = CircuitBreakerStrategy(
            "test",
            CircuitBreakerConfig(failure_threshold=1, timeout_seconds=10),
        )

        async def failing_op():
            raise NetworkError("fail")

        # Trip the circuit
        with pytest.raises(NetworkError):
            await circuit.execute(failing_op)

        # Now it should reject with RecoverableError
        with pytest.raises(RecoverableError) as exc_info:
            await circuit.execute(failing_op)

        assert "open" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_success_resets_failures(self):
        """Test success resets failure count."""
        circuit = CircuitBreakerStrategy(
            "test",
            CircuitBreakerConfig(failure_threshold=3),
        )

        async def failing_op():
            raise NetworkError("fail")

        async def success_op():
            return "ok"

        # Fail twice
        for _ in range(2):
            with pytest.raises(NetworkError):
                await circuit.execute(failing_op)

        # Succeed
        await circuit.execute(success_op)

        # Should be able to fail again without opening
        with pytest.raises(NetworkError):
            await circuit.execute(failing_op)

        assert circuit.state == CircuitState.CLOSED

    def test_manual_reset(self):
        """Test manual circuit reset."""
        circuit = CircuitBreakerStrategy(
            "test",
            CircuitBreakerConfig(failure_threshold=1),
        )

        circuit._state = CircuitState.OPEN
        circuit.reset()

        assert circuit.state == CircuitState.CLOSED


# =============================================================================
# Fallback Strategy Tests
# =============================================================================

class TestFallbackStrategy:
    """Tests for fallback strategy."""

    @pytest.mark.asyncio
    async def test_primary_succeeds(self):
        """Test primary success doesn't use fallbacks."""
        fallback_called = []

        async def primary():
            return "primary"

        async def fallback():
            fallback_called.append(True)
            return "fallback"

        strategy = FallbackStrategy(fallbacks=[fallback])
        result = await strategy.execute(primary)

        assert result == "primary"
        assert len(fallback_called) == 0

    @pytest.mark.asyncio
    async def test_uses_fallback(self):
        """Test fallback is used when primary fails."""
        async def primary():
            raise NetworkError("primary failed")

        async def fallback():
            return "fallback"

        strategy = FallbackStrategy(fallbacks=[fallback])
        result = await strategy.execute(primary)

        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_multiple_fallbacks(self):
        """Test multiple fallbacks tried in order."""
        calls = []

        async def primary():
            calls.append("primary")
            raise NetworkError("fail")

        async def fallback1():
            calls.append("fallback1")
            raise NetworkError("fail")

        async def fallback2():
            calls.append("fallback2")
            return "success"

        strategy = FallbackStrategy(fallbacks=[fallback1, fallback2])
        result = await strategy.execute(primary)

        assert result == "success"
        assert calls == ["primary", "fallback1", "fallback2"]

    @pytest.mark.asyncio
    async def test_all_fail(self):
        """Test error when all operations fail."""
        async def primary():
            raise NetworkError("primary fail")

        async def fallback():
            raise NetworkError("fallback fail")

        strategy = FallbackStrategy(fallbacks=[fallback])

        with pytest.raises(UnrecoverableError):
            await strategy.execute(primary)


# =============================================================================
# Recovery Executor Tests
# =============================================================================

class TestRecoveryExecutor:
    """Tests for the recovery executor."""

    @pytest.mark.asyncio
    async def test_successful_execution(self):
        """Test successful execution result."""
        executor = RecoveryExecutor()

        async def operation():
            return "success"

        result = await executor.execute(operation, "test_op")

        assert result.success is True
        assert result.result == "success"
        assert result.attempts == 1
        assert result.recovered is False

    @pytest.mark.asyncio
    async def test_retry_and_succeed(self):
        """Test retry and eventual success."""
        executor = RecoveryExecutor(RecoveryConfig(
            max_retries=3,
            initial_delay_seconds=0.01,
        ))

        call_count = 0
        async def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise NetworkError("temp fail")
            return "success"

        result = await executor.execute(operation, "test_op")

        assert result.success is True
        assert result.attempts == 2
        assert result.recovered is True

    @pytest.mark.asyncio
    async def test_unrecoverable_no_retry(self):
        """Test unrecoverable error doesn't retry."""
        executor = RecoveryExecutor(RecoveryConfig(max_retries=3))

        async def operation():
            raise InsufficientFundsError("not enough")

        result = await executor.execute(operation, "test_op")

        assert result.success is False
        assert result.attempts == 1
        assert isinstance(result.error, InsufficientFundsError)

    @pytest.mark.asyncio
    async def test_all_retries_fail(self):
        """Test result when all retries fail."""
        executor = RecoveryExecutor(RecoveryConfig(
            max_retries=3,
            initial_delay_seconds=0.01,
        ))

        async def operation():
            raise NetworkError("always fail")

        result = await executor.execute(operation, "test_op")

        assert result.success is False
        assert result.attempts == 3

    @pytest.mark.asyncio
    async def test_escalation_callback(self):
        """Test escalation callback is called."""
        escalated = []

        async def escalation(error, context, attempts):
            escalated.append((error, attempts))

        executor = RecoveryExecutor(
            RecoveryConfig(max_retries=1, escalate_on_failure=True),
            escalation_callback=escalation,
        )

        async def operation():
            raise NetworkError("fail")

        result = await executor.execute(operation, "test_op")

        assert result.success is False
        assert result.escalated is True
        assert len(escalated) == 1

    @pytest.mark.asyncio
    async def test_execute_with_fallbacks(self):
        """Test execution with fallbacks."""
        executor = RecoveryExecutor(RecoveryConfig(
            max_retries=1,
            initial_delay_seconds=0.01,
        ))

        async def primary():
            raise NetworkError("primary fail")

        async def fallback():
            return "fallback success"

        result = await executor.execute_with_fallbacks(
            primary,
            [fallback],
            "test_op",
        )

        assert result.success is True
        assert result.result == "fallback success"
        assert result.recovered is True
        assert "fallback" in result.recovery_strategy_used

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self):
        """Test circuit breaker integration."""
        executor = RecoveryExecutor(RecoveryConfig(
            enable_circuit_breaker=True,
            circuit_failure_threshold=2,
            max_retries=1,
        ))

        call_count = 0
        async def operation():
            nonlocal call_count
            call_count += 1
            raise NetworkError("fail")

        # First two calls should fail and trip the circuit
        await executor.execute(operation, "test", provider="test_provider")
        await executor.execute(operation, "test", provider="test_provider")

        state = executor.get_circuit_breaker_state("test_provider")
        assert state == "open"

    def test_execution_result_to_dict(self):
        """Test ExecutionResult serialization."""
        result = ExecutionResult(
            success=True,
            result="test",
            attempts=2,
            recovered=True,
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["attempts"] == 2
        assert data["recovered"] is True


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_execute_with_recovery(self):
        """Test execute_with_recovery convenience function."""
        async def operation():
            return "success"

        result = await execute_with_recovery(operation)

        assert result.success is True
        assert result.result == "success"
