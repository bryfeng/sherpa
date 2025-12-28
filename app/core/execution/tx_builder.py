"""
Transaction builder for constructing various transaction types.
"""

import secrets
from typing import Any, Dict, Optional

from .models import (
    PreparedTransaction,
    TransactionType,
    GasEstimate,
    SwapQuote,
    BridgeQuote,
)


# Common contract ABIs (minimal for encoding)
ERC20_APPROVE_SELECTOR = "0x095ea7b3"  # approve(address,uint256)
ERC20_TRANSFER_SELECTOR = "0xa9059cbb"  # transfer(address,uint256)

# Maximum uint256 for unlimited approval
MAX_UINT256 = 2**256 - 1


def _encode_uint256(value: int) -> str:
    """Encode a uint256 as a 32-byte hex string (without 0x prefix)."""
    return format(value, "064x")


def _encode_address(address: str) -> str:
    """Encode an address as a 32-byte hex string (without 0x prefix)."""
    # Remove 0x prefix and pad to 32 bytes
    addr = address.lower().replace("0x", "")
    return addr.zfill(64)


class TransactionBuilder:
    """
    Builds transactions for various protocols.

    Handles:
    - ERC20 approvals
    - Swaps from Relay quotes
    - Bridges from Relay quotes
    - Native token transfers
    """

    @staticmethod
    def generate_tx_id() -> str:
        """Generate a unique transaction ID."""
        return f"tx_{secrets.token_hex(16)}"

    @staticmethod
    def build_erc20_approve(
        chain_id: int,
        owner_address: str,
        token_address: str,
        spender_address: str,
        amount: int = MAX_UINT256,
        description: str = "",
    ) -> PreparedTransaction:
        """
        Build an ERC20 approval transaction.

        Args:
            chain_id: The chain ID
            owner_address: The token owner (sender)
            token_address: The ERC20 token contract
            spender_address: The address being approved to spend
            amount: The amount to approve (default: unlimited)
            description: Human-readable description

        Returns:
            PreparedTransaction ready to be signed
        """
        # Encode: approve(address spender, uint256 amount)
        calldata = (
            ERC20_APPROVE_SELECTOR +
            _encode_address(spender_address) +
            _encode_uint256(amount)
        )

        return PreparedTransaction(
            tx_id=TransactionBuilder.generate_tx_id(),
            tx_type=TransactionType.APPROVE,
            chain_id=chain_id,
            from_address=owner_address.lower(),
            to_address=token_address.lower(),
            data=calldata,
            value=0,
            description=description or f"Approve {spender_address[:10]}... to spend tokens",
        )

    @staticmethod
    def build_erc20_transfer(
        chain_id: int,
        from_address: str,
        token_address: str,
        to_address: str,
        amount: int,
        description: str = "",
    ) -> PreparedTransaction:
        """
        Build an ERC20 transfer transaction.

        Args:
            chain_id: The chain ID
            from_address: The sender address
            token_address: The ERC20 token contract
            to_address: The recipient address
            amount: The amount to transfer (in smallest units)
            description: Human-readable description

        Returns:
            PreparedTransaction ready to be signed
        """
        # Encode: transfer(address to, uint256 amount)
        calldata = (
            ERC20_TRANSFER_SELECTOR +
            _encode_address(to_address) +
            _encode_uint256(amount)
        )

        return PreparedTransaction(
            tx_id=TransactionBuilder.generate_tx_id(),
            tx_type=TransactionType.TRANSFER,
            chain_id=chain_id,
            from_address=from_address.lower(),
            to_address=token_address.lower(),
            data=calldata,
            value=0,
            description=description or f"Transfer tokens to {to_address[:10]}...",
        )

    @staticmethod
    def build_native_transfer(
        chain_id: int,
        from_address: str,
        to_address: str,
        amount_wei: int,
        description: str = "",
    ) -> PreparedTransaction:
        """
        Build a native token (ETH, MATIC, etc.) transfer.

        Args:
            chain_id: The chain ID
            from_address: The sender address
            to_address: The recipient address
            amount_wei: The amount in wei
            description: Human-readable description

        Returns:
            PreparedTransaction ready to be signed
        """
        return PreparedTransaction(
            tx_id=TransactionBuilder.generate_tx_id(),
            tx_type=TransactionType.TRANSFER,
            chain_id=chain_id,
            from_address=from_address.lower(),
            to_address=to_address.lower(),
            data="0x",
            value=amount_wei,
            description=description or f"Transfer native token to {to_address[:10]}...",
        )

    @staticmethod
    def build_from_swap_quote(
        quote: SwapQuote,
        nonce: Optional[int] = None,
        gas_estimate: Optional[GasEstimate] = None,
    ) -> PreparedTransaction:
        """
        Build a transaction from a Relay swap quote.

        Args:
            quote: The parsed swap quote
            nonce: Optional nonce override
            gas_estimate: Optional gas estimate

        Returns:
            PreparedTransaction ready to be signed
        """
        if not quote.tx:
            raise ValueError("Swap quote has no transaction data")

        tx_data = quote.tx
        tx_id = TransactionBuilder.generate_tx_id()

        return PreparedTransaction(
            tx_id=tx_id,
            tx_type=TransactionType.SWAP,
            chain_id=quote.chain_id,
            from_address=quote.wallet_address.lower(),
            to_address=tx_data.get("to", "").lower(),
            data=tx_data.get("data", "0x"),
            value=int(tx_data.get("value", "0"), 16) if isinstance(tx_data.get("value"), str) else tx_data.get("value", 0),
            gas_estimate=gas_estimate,
            nonce=nonce,
            description=f"Swap {quote.token_in_symbol} → {quote.token_out_symbol}",
            quote_id=quote.request_id,
        )

    @staticmethod
    def build_from_bridge_quote(
        quote: BridgeQuote,
        nonce: Optional[int] = None,
        gas_estimate: Optional[GasEstimate] = None,
    ) -> PreparedTransaction:
        """
        Build a transaction from a Relay bridge quote.

        Args:
            quote: The parsed bridge quote
            nonce: Optional nonce override
            gas_estimate: Optional gas estimate

        Returns:
            PreparedTransaction ready to be signed
        """
        if not quote.tx:
            raise ValueError("Bridge quote has no transaction data")

        tx_data = quote.tx
        tx_id = TransactionBuilder.generate_tx_id()

        return PreparedTransaction(
            tx_id=tx_id,
            tx_type=TransactionType.BRIDGE,
            chain_id=quote.origin_chain_id,
            from_address=quote.wallet_address.lower(),
            to_address=tx_data.get("to", "").lower(),
            data=tx_data.get("data", "0x"),
            value=int(tx_data.get("value", "0"), 16) if isinstance(tx_data.get("value"), str) else tx_data.get("value", 0),
            gas_estimate=gas_estimate,
            nonce=nonce,
            description=f"Bridge {quote.token_in_symbol} to chain {quote.destination_chain_id}",
            quote_id=quote.request_id,
        )

    @staticmethod
    def build_approvals_from_quote(
        quote: SwapQuote | BridgeQuote,
    ) -> list[PreparedTransaction]:
        """
        Build approval transactions from a quote's approval requirements.

        Args:
            quote: The parsed swap or bridge quote

        Returns:
            List of PreparedTransaction for approvals
        """
        approvals = []
        chain_id = quote.chain_id if hasattr(quote, 'chain_id') else quote.origin_chain_id

        for approval in quote.approvals:
            data = approval.get("data", {})
            spender = data.get("spender", "")
            amount = data.get("amount") or data.get("value") or MAX_UINT256

            if isinstance(amount, str):
                amount = int(amount, 16) if amount.startswith("0x") else int(amount)

            tx = TransactionBuilder.build_erc20_approve(
                chain_id=chain_id,
                owner_address=quote.wallet_address,
                token_address=quote.token_in_address,
                spender_address=spender,
                amount=amount,
                description=f"Approve {quote.token_in_symbol} for swap/bridge",
            )
            tx.quote_id = quote.request_id
            approvals.append(tx)

        return approvals

    @staticmethod
    def build_from_raw(
        chain_id: int,
        from_address: str,
        to_address: str,
        data: str,
        value: int = 0,
        tx_type: TransactionType = TransactionType.SWAP,
        description: str = "",
        quote_id: Optional[str] = None,
    ) -> PreparedTransaction:
        """
        Build a transaction from raw data.

        Args:
            chain_id: The chain ID
            from_address: The sender address
            to_address: The target contract
            data: The calldata (hex encoded)
            value: Wei to send
            tx_type: The transaction type
            description: Human-readable description
            quote_id: Optional quote reference

        Returns:
            PreparedTransaction ready to be signed
        """
        return PreparedTransaction(
            tx_id=TransactionBuilder.generate_tx_id(),
            tx_type=tx_type,
            chain_id=chain_id,
            from_address=from_address.lower(),
            to_address=to_address.lower(),
            data=data if data.startswith("0x") else f"0x{data}",
            value=value,
            description=description,
            quote_id=quote_id,
        )
