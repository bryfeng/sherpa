"""Trading tool handlers: swap quotes, bridge quotes, transfers."""

import logging
from typing import Any, Dict, List, Optional

from .base import tool_spec
from ....providers.llm.base import ToolParameter, ToolParameterType

logger = logging.getLogger(__name__)


# =========================================================================
# Solana Swap (Jupiter)
# =========================================================================


@tool_spec(
    name="get_solana_swap_quote",
    description=(
        "Get a swap quote for exchanging tokens WITHIN Solana using Jupiter DEX. "
        "This is for Solana-to-Solana token swaps only (e.g., SOL to USDC, BONK to JUP). "
        "Returns the expected output amount, price impact, and a transaction ready for signing. "
        "NOTE: For cross-chain transfers TO or FROM Solana, use get_bridge_quote instead."
    ),
    parameters=[
        ToolParameter(
            name="wallet_address",
            type=ToolParameterType.STRING,
            description="The Solana wallet address (public key) performing the swap",
            required=True,
        ),
        ToolParameter(
            name="input_token",
            type=ToolParameterType.STRING,
            description=(
                "The input token symbol or mint address "
                "(e.g., 'SOL', 'USDC', or full mint address)"
            ),
            required=True,
        ),
        ToolParameter(
            name="output_token",
            type=ToolParameterType.STRING,
            description=(
                "The output token symbol or mint address "
                "(e.g., 'SOL', 'USDC', or full mint address)"
            ),
            required=True,
        ),
        ToolParameter(
            name="amount",
            type=ToolParameterType.NUMBER,
            description="Amount of input token to swap (in human-readable units, not lamports)",
            required=True,
        ),
        ToolParameter(
            name="slippage_bps",
            type=ToolParameterType.INTEGER,
            description="Slippage tolerance in basis points (50 = 0.5%, default)",
            required=False,
            default=50,
        ),
    ],
    requires_address=True,
)
async def handle_get_solana_swap_quote(
    wallet_address: str,
    input_token: str,
    output_token: str,
    amount: float,
    slippage_bps: int = 50,
) -> Dict[str, Any]:
    """Handle Solana swap quote request via Jupiter."""
    from decimal import Decimal
    from ....providers.jupiter import (
        get_jupiter_swap_provider,
        JupiterQuoteError,
        JupiterSwapError,
        NATIVE_SOL_MINT,
        USDC_MINT,
        USDT_MINT,
    )
    from ...swap.constants import TOKEN_REGISTRY, SOLANA_CHAIN_ID

    try:
        jupiter = get_jupiter_swap_provider()

        # Resolve token symbols to mint addresses
        solana_tokens = TOKEN_REGISTRY.get(SOLANA_CHAIN_ID, {})

        def resolve_mint(token: str) -> str:
            """Resolve symbol to mint address or return as-is if already an address."""
            upper = token.upper()
            if upper in solana_tokens:
                return str(solana_tokens[upper]['address'])
            # Check if it looks like a Solana address (32-44 chars, base58)
            if len(token) >= 32 and len(token) <= 44:
                return token
            # Common aliases
            if upper in ('SOL', 'SOLANA'):
                return NATIVE_SOL_MINT
            if upper == 'USDC':
                return USDC_MINT
            if upper == 'USDT':
                return USDT_MINT
            # Return as-is, let Jupiter handle the error
            return token

        input_mint = resolve_mint(input_token)
        output_mint = resolve_mint(output_token)

        # Get input token decimals
        input_decimals = 9  # default SOL decimals
        for symbol, meta in solana_tokens.items():
            if str(meta.get('address')) == input_mint:
                input_decimals = int(meta.get('decimals', 9))
                break

        # Convert amount to smallest units
        amount_decimal = Decimal(str(amount))
        amount_lamports = int(amount_decimal * (Decimal(10) ** input_decimals))

        if amount_lamports <= 0:
            return {
                "success": False,
                "error": "Amount must be greater than 0",
            }

        # Get quote from Jupiter
        quote = await jupiter.get_swap_quote(
            input_mint=input_mint,
            output_mint=output_mint,
            amount=amount_lamports,
            slippage_bps=slippage_bps,
        )

        # Build swap transaction
        swap_result = await jupiter.build_swap_transaction(
            quote=quote,
            user_public_key=wallet_address,
        )

        # Get output token decimals
        output_decimals = 9
        for symbol, meta in solana_tokens.items():
            if str(meta.get('address')) == output_mint:
                output_decimals = int(meta.get('decimals', 9))
                break

        # Calculate human-readable amounts
        output_amount = Decimal(str(quote.out_amount)) / (Decimal(10) ** output_decimals)
        min_output_amount = Decimal(str(quote.other_amount_threshold)) / (Decimal(10) ** output_decimals)

        # Get token symbols
        input_symbol = input_token.upper()
        output_symbol = output_token.upper()
        if quote.input_token:
            input_symbol = quote.input_token.symbol
        if quote.output_token:
            output_symbol = quote.output_token.symbol

        return {
            "success": True,
            "chain": "solana",
            "provider": "jupiter",
            "input_token": {
                "symbol": input_symbol,
                "mint": input_mint,
                "amount": str(amount),
                "amount_lamports": amount_lamports,
            },
            "output_token": {
                "symbol": output_symbol,
                "mint": output_mint,
                "amount_estimate": str(output_amount),
                "min_amount": str(min_output_amount),
            },
            "price_impact_percent": quote.price_impact_pct,
            "slippage_bps": slippage_bps,
            "transaction": {
                "swap_transaction_base64": swap_result.swap_transaction,
                "last_valid_block_height": swap_result.last_valid_block_height,
                "priority_fee_lamports": swap_result.priority_fee_lamports,
                "compute_unit_limit": swap_result.compute_unit_limit,
            },
            "instructions": [
                f"Swap {amount} {input_symbol} for ~{output_amount:.6f} {output_symbol} on Solana",
                f"Minimum output: {min_output_amount:.6f} {output_symbol} (with {slippage_bps/100}% slippage)",
                "Sign the transaction in your Solana wallet to execute the swap.",
            ],
        }

    except JupiterQuoteError as e:
        logger.warning(f"Jupiter quote error: {e}")
        return {
            "success": False,
            "error": f"Could not get swap quote: {e}",
            "hint": "Check that the token symbols are valid and you have sufficient balance.",
        }
    except JupiterSwapError as e:
        logger.warning(f"Jupiter swap build error: {e}")
        return {
            "success": False,
            "error": f"Could not build swap transaction: {e}",
        }
    except Exception as e:
        logger.error(f"Error in Solana swap quote: {e}")
        return {"success": False, "error": str(e)}


# =========================================================================
# EVM Swap (Relay)
# =========================================================================


@tool_spec(
    name="get_swap_quote",
    description=(
        "Get a swap quote for exchanging tokens on a SINGLE EVM chain only. "
        "Use this ONLY when both input and output tokens are on THE SAME blockchain. "
        "Examples: 'swap ETH for USDC on Ethereum', 'exchange WBTC to ETH on Base'. "
        "DO NOT use this if user mentions two different chains - use get_bridge_quote instead. "
        "Supports ETH, USDC, USDT, WBTC, and many other EVM tokens. "
        "NOTE: For cross-chain (different source/destination chains), use get_bridge_quote."
    ),
    parameters=[
        ToolParameter(
            name="wallet_address",
            type=ToolParameterType.STRING,
            description="The EVM wallet address (0x...) performing the swap",
            required=True,
        ),
        ToolParameter(
            name="input_token",
            type=ToolParameterType.STRING,
            description=(
                "The input token symbol or contract address "
                "(e.g., 'ETH', 'USDC', 'WBTC', or '0x...')"
            ),
            required=True,
        ),
        ToolParameter(
            name="output_token",
            type=ToolParameterType.STRING,
            description=(
                "The output token symbol or contract address "
                "(e.g., 'ETH', 'USDC', 'WBTC', or '0x...')"
            ),
            required=True,
        ),
        ToolParameter(
            name="amount",
            type=ToolParameterType.NUMBER,
            description="Amount of input token to swap (in human-readable units)",
            required=True,
        ),
        ToolParameter(
            name="chain",
            type=ToolParameterType.STRING,
            description=(
                "The blockchain to swap on (e.g., 'ethereum', 'base', 'arbitrum', "
                "'optimism', 'polygon'). Defaults to 'ethereum'."
            ),
            required=False,
            default="ethereum",
        ),
        ToolParameter(
            name="slippage_percent",
            type=ToolParameterType.NUMBER,
            description="Slippage tolerance as a percentage (e.g., 0.5 for 0.5%). Defaults to 0.5%.",
            required=False,
            default=0.5,
        ),
    ],
    requires_address=True,
)
async def handle_get_swap_quote(
    wallet_address: str,
    input_token: str,
    output_token: str,
    amount: float,
    chain: str = "ethereum",
    slippage_percent: float = 0.5,
) -> Dict[str, Any]:
    """Handle EVM swap quote request via Relay."""
    from decimal import Decimal
    from ....providers.relay import RelayProvider
    from ...bridge.chain_registry import get_chain_registry
    from ...swap.constants import TOKEN_REGISTRY, TOKEN_ALIAS_MAP

    try:
        # Resolve chain name to chain ID
        registry = await get_chain_registry()
        chain_id = registry.get_chain_id(chain.lower())
        if chain_id is None:
            return {
                "success": False,
                "error": f"Unknown chain: {chain}",
                "hint": f"Supported chains include: ethereum, base, arbitrum, optimism, polygon",
            }

        # For Solana, redirect to the Solana swap tool
        if isinstance(chain_id, str) and chain_id.lower() == "solana":
            return {
                "success": False,
                "error": "For Solana swaps, use get_solana_swap_quote instead",
                "hint": "This tool is for EVM chains only. Use get_solana_swap_quote for Solana-to-Solana swaps.",
            }

        # Resolve token symbols to addresses
        def resolve_token(token: str, chain_id: int) -> Optional[Dict[str, Any]]:
            """Resolve token symbol to metadata."""
            chain_tokens = TOKEN_REGISTRY.get(chain_id, {})
            alias_map = TOKEN_ALIAS_MAP.get(chain_id, {})

            # Check if it's already an address
            if token.startswith("0x") and len(token) == 42:
                # Look up by address
                for sym, meta in chain_tokens.items():
                    if str(meta.get("address", "")).lower() == token.lower():
                        return {"symbol": sym, **meta}
                # Return as-is if not found in registry
                return {"symbol": token[:8], "address": token, "decimals": 18, "is_native": False}

            # Resolve alias to symbol
            symbol = alias_map.get(token.lower(), token.upper())
            if symbol in chain_tokens:
                return {"symbol": symbol, **chain_tokens[symbol]}

            # Common native token handling
            if token.lower() in ("eth", "ether", "native"):
                return {
                    "symbol": "ETH",
                    "address": "0x0000000000000000000000000000000000000000",
                    "decimals": 18,
                    "is_native": True,
                }

            return None

        input_meta = resolve_token(input_token, chain_id)
        output_meta = resolve_token(output_token, chain_id)

        if not input_meta:
            return {
                "success": False,
                "error": f"Unknown input token: {input_token}",
                "hint": f"Try using the full contract address or a common symbol like ETH, USDC, USDT",
            }

        if not output_meta:
            return {
                "success": False,
                "error": f"Unknown output token: {output_token}",
                "hint": f"Try using the full contract address or a common symbol like ETH, USDC, USDT",
            }

        # Convert amount to base units
        amount_decimal = Decimal(str(amount))
        decimals = int(input_meta.get("decimals", 18))
        amount_base_units = int(amount_decimal * (Decimal(10) ** decimals))

        if amount_base_units <= 0:
            return {
                "success": False,
                "error": "Amount must be greater than 0",
            }

        # Build Relay quote request
        relay = RelayProvider()
        relay_payload = {
            "user": wallet_address,
            "originChainId": chain_id,
            "destinationChainId": chain_id,  # Same chain for swap
            "originCurrency": input_meta["address"],
            "destinationCurrency": output_meta["address"],
            "recipient": wallet_address,
            "tradeType": "EXACT_INPUT",
            "amount": str(amount_base_units),
            "referrer": "sherpa.chat",
            "useExternalLiquidity": True,
        }

        quote = await relay.quote(relay_payload)

        # Parse quote response
        details = quote.get("details", {})
        output_decimals = int(output_meta.get("decimals", 18))

        # Get output amount
        currency_out = details.get("currencyOut", {})
        output_amount_raw = currency_out.get("amount", "0")
        output_amount = Decimal(output_amount_raw) / (Decimal(10) ** output_decimals)

        # Get fees
        total_fee_usd = Decimal(str(details.get("totalFeeUsd", "0")))

        # Get steps/transactions
        steps = quote.get("steps", [])

        chain_name = registry.get_chain_name(chain_id)

        return {
            "success": True,
            "provider": "relay",
            "chain": chain_name,
            "chain_id": chain_id,
            "input_token": {
                "symbol": input_meta["symbol"],
                "address": input_meta["address"],
                "amount": str(amount),
                "amount_base_units": str(amount_base_units),
            },
            "output_token": {
                "symbol": output_meta["symbol"],
                "address": output_meta["address"],
                "amount_estimate": str(output_amount),
            },
            "fees": {
                "total_usd": str(total_fee_usd),
                "slippage_percent": slippage_percent,
            },
            "steps_count": len(steps),
            "quote_data": quote,  # Full quote for execution
            "instructions": [
                f"Swap {amount} {input_meta['symbol']} for ~{output_amount:.6f} {output_meta['symbol']} on {chain_name}",
                f"Estimated fees: ${total_fee_usd:.2f}",
                "Review and sign the transaction in your wallet to execute the swap.",
            ],
        }

    except Exception as e:
        logger.error(f"Error in EVM swap quote: {e}")
        return {"success": False, "error": str(e)}


# =========================================================================
# Transfer
# =========================================================================


async def _check_action_allowed(
    wallet_address: str,
    action_type: str,
    value_usd: float,
    chain_id: int = 1,
    token_in: Optional[str] = None,
    token_out: Optional[str] = None,
    slippage_percent: Optional[float] = None,
    contract_address: Optional[str] = None,
) -> Dict[str, Any]:
    """Check if an action is allowed by policies (internal helper)."""
    from decimal import Decimal
    from ....db import get_convex_client
    from ...policy import PolicyEngine, ActionContext, RiskPolicyConfig, SystemPolicyConfig

    try:
        convex = get_convex_client()

        # Fetch risk policy for this wallet
        risk_policy_data = await convex.query(
            "riskPolicies:getByWallet",
            {"walletAddress": wallet_address.lower()},
        )

        if risk_policy_data and risk_policy_data.get("config"):
            risk_config = RiskPolicyConfig.from_dict(risk_policy_data["config"])
        else:
            return {
                "success": True,
                "approved": False,
                "policy_missing": True,
                "requires_approval": False,
                "violations": [
                    {
                        "policyType": "risk",
                        "policyName": "risk_policy_missing",
                        "severity": "block",
                        "message": "No risk policy configured. Draft a policy to enable autonomous execution.",
                    }
                ],
                "warnings": [],
            }

        # Fetch system policy
        system_policy_data = await convex.query("systemPolicy:get", {})
        system_config = SystemPolicyConfig()
        if system_policy_data:
            system_config = SystemPolicyConfig(
                emergency_stop=system_policy_data.get("emergencyStop", False),
                emergency_stop_reason=system_policy_data.get("emergencyStopReason"),
                in_maintenance=system_policy_data.get("inMaintenance", False),
                maintenance_message=system_policy_data.get("maintenanceMessage"),
                blocked_contracts=system_policy_data.get("blockedContracts", []),
                blocked_tokens=system_policy_data.get("blockedTokens", []),
                blocked_chains=system_policy_data.get("blockedChains", []),
                allowed_chains=system_policy_data.get("allowedChains", []),
                protocol_whitelist_enabled=system_policy_data.get("protocolWhitelistEnabled", False),
                allowed_protocols=system_policy_data.get("allowedProtocols", []),
                max_single_tx_usd=Decimal(str(system_policy_data.get("maxSingleTxUsd", 100000))),
            )

        # Build action context
        context = ActionContext(
            session_id="agent-check",
            wallet_address=wallet_address.lower(),
            action_type=action_type,
            chain_id=chain_id,
            value_usd=Decimal(str(value_usd)),
            contract_address=contract_address,
            token_in=token_in,
            token_out=token_out,
            slippage_percent=slippage_percent,
        )

        # Evaluate policies
        engine = PolicyEngine(
            risk_config=risk_config,
            system_config=system_config,
        )
        result = engine.evaluate(context)

        return {
            "success": True,
            "approved": result.approved,
            "risk_score": result.risk_score,
            "risk_level": result.risk_level.value,
            "requires_approval": result.requires_approval,
            "approval_reason": result.approval_reason,
            "violations": [v.to_dict() for v in result.violations],
            "warnings": [w.to_dict() for w in result.warnings],
            "action": {
                "type": action_type,
                "value_usd": value_usd,
                "chain_id": chain_id,
                "token_in": token_in,
                "token_out": token_out,
            },
        }

    except Exception as e:
        logger.error(f"Error checking action: {e}")
        return {"success": False, "error": str(e)}


@tool_spec(
    name="execute_transfer",
    description=(
        "Send tokens to another address. Builds a transfer transaction "
        "for the user to sign in their wallet. Supports native ETH and "
        "ERC20 token transfers on any EVM chain."
    ),
    parameters=[
        ToolParameter(
            name="wallet_address",
            type=ToolParameterType.STRING,
            description="The sender's wallet address (0x...)",
            required=True,
        ),
        ToolParameter(
            name="to_address",
            type=ToolParameterType.STRING,
            description="The recipient's wallet address (0x...)",
            required=True,
        ),
        ToolParameter(
            name="token",
            type=ToolParameterType.STRING,
            description=(
                "The token to send — symbol (ETH, USDC, WBTC) or "
                "contract address (0x...)"
            ),
            required=True,
        ),
        ToolParameter(
            name="amount",
            type=ToolParameterType.NUMBER,
            description="Amount to send in human-readable units (e.g., 0.5 for 0.5 ETH)",
            required=True,
        ),
        ToolParameter(
            name="chain",
            type=ToolParameterType.STRING,
            description=(
                "The blockchain to send on (e.g., 'ethereum', 'base', "
                "'arbitrum', 'optimism', 'polygon'). Defaults to 'ethereum'."
            ),
            required=False,
            default="ethereum",
        ),
    ],
    requires_address=True,
)
async def handle_execute_transfer(
    wallet_address: str,
    to_address: str,
    token: str,
    amount: float,
    chain: str = "ethereum",
) -> Dict[str, Any]:
    """Handle building a token transfer transaction."""
    from decimal import Decimal
    from ...bridge.chain_registry import get_chain_registry
    from ...swap.constants import TOKEN_REGISTRY, TOKEN_ALIAS_MAP

    try:
        # Validate addresses
        if not to_address.startswith("0x") or len(to_address) != 42:
            return {
                "success": False,
                "error": f"Invalid recipient address: {to_address}",
                "hint": "Recipient must be a valid 0x EVM address (42 characters).",
            }

        if not wallet_address.startswith("0x") or len(wallet_address) != 42:
            return {
                "success": False,
                "error": f"Invalid sender address: {wallet_address}",
            }

        # Resolve chain
        registry = await get_chain_registry()
        chain_id = registry.get_chain_id(chain.lower())
        if chain_id is None:
            return {
                "success": False,
                "error": f"Unknown chain: {chain}",
                "hint": "Supported chains include: ethereum, base, arbitrum, optimism, polygon",
            }

        if isinstance(chain_id, str) and chain_id.lower() == "solana":
            return {
                "success": False,
                "error": "Solana transfers are not supported by this tool",
                "hint": "This tool supports EVM chains only.",
            }

        # Resolve token
        def resolve_token(tok: str, cid: int) -> Optional[Dict[str, Any]]:
            chain_tokens = TOKEN_REGISTRY.get(cid, {})
            alias_map = TOKEN_ALIAS_MAP.get(cid, {})

            if tok.startswith("0x") and len(tok) == 42:
                for sym, meta in chain_tokens.items():
                    if str(meta.get("address", "")).lower() == tok.lower():
                        return {"symbol": sym, **meta}
                return {"symbol": tok[:8], "address": tok, "decimals": 18, "is_native": False}

            symbol = alias_map.get(tok.lower(), tok.upper())
            if symbol in chain_tokens:
                return {"symbol": symbol, **chain_tokens[symbol]}

            if tok.lower() in ("eth", "ether", "native"):
                return {
                    "symbol": "ETH",
                    "address": "0x0000000000000000000000000000000000000000",
                    "decimals": 18,
                    "is_native": True,
                }

            return None

        token_meta = resolve_token(token, chain_id)
        if not token_meta:
            return {
                "success": False,
                "error": f"Unknown token: {token}",
                "hint": "Try using the full contract address or a common symbol like ETH, USDC, USDT",
            }

        # Convert amount to base units
        amount_decimal = Decimal(str(amount))
        decimals = int(token_meta.get("decimals", 18))
        amount_base_units = int(amount_decimal * (Decimal(10) ** decimals))

        if amount_base_units <= 0:
            return {
                "success": False,
                "error": "Amount must be greater than 0",
            }

        chain_name = registry.get_chain_name(chain_id)
        is_native = token_meta.get("is_native", False)

        # Build transaction data
        if is_native:
            # Native ETH transfer: send value directly
            tx_data = {
                "to": to_address,
                "data": "0x",
                "value": hex(amount_base_units),
            }
        else:
            # ERC20 transfer(address,uint256) — selector 0xa9059cbb
            selector = "a9059cbb"
            encoded_address = to_address[2:].lower().zfill(64)
            encoded_amount = hex(amount_base_units)[2:].zfill(64)
            data = "0x" + selector + encoded_address + encoded_amount

            tx_data = {
                "to": token_meta["address"],  # ERC20 contract
                "data": data,
                "value": "0x0",
            }

        # Run policy check
        policy_result = await _check_action_allowed(
            wallet_address=wallet_address,
            action_type="transfer",
            value_usd=0,  # No USD estimate for raw transfers
            chain_id=chain_id,
            token_in=token_meta.get("symbol"),
            contract_address=token_meta.get("address") if not is_native else None,
        )

        if policy_result.get("success") and not policy_result.get("approved", True):
            violations = policy_result.get("violations", [])
            return {
                "success": False,
                "error": "Transfer blocked by policy",
                "violations": violations,
                "hint": "Check your risk policy settings or contact support.",
            }

        return {
            "success": True,
            "type": "transfer",
            "chain": chain_name,
            "chain_id": chain_id,
            "token": {
                "symbol": token_meta["symbol"],
                "address": token_meta["address"],
                "amount": str(amount),
                "amount_base_units": str(amount_base_units),
                "decimals": decimals,
            },
            "recipient": to_address,
            "tx_data": tx_data,
            "instructions": [
                f"Send {amount} {token_meta['symbol']} to {to_address[:6]}...{to_address[-4:]} on {chain_name}",
                "Sign the transaction in your wallet to execute.",
            ],
        }

    except Exception as e:
        logger.error(f"Error building transfer: {e}")
        return {"success": False, "error": str(e)}


# =========================================================================
# Cross-Chain Bridge (Relay)
# =========================================================================


@tool_spec(
    name="get_bridge_quote",
    description=(
        "REQUIRED: Call this tool for ANY cross-chain token operation - including swaps, bridges, or transfers "
        "where the source and destination are DIFFERENT blockchains. "
        "Supports ALL EVM chains (Ethereum/mainnet, Base, Arbitrum, Optimism, Polygon, Ink, zkSync, Scroll) and Solana. "
        "KEY PATTERN: If user mentions TWO DIFFERENT chains, use this tool. 'mainnet' = Ethereum. "
        "Examples: 'swap USDC.e on Ink to USDC on mainnet', 'bridge ETH to Base', 'transfer from Polygon to Arbitrum', "
        "'move USDC from Ink chain to Ethereum', 'swap tokens from Base to mainnet'. "
        "Returns real quotes with exact amounts and fees - NEVER guess or make up numbers. "
        "NOTE: For SAME-chain swaps only (e.g., ETH to USDC both on Ethereum), use get_swap_quote instead."
    ),
    parameters=[
        ToolParameter(
            name="wallet_address",
            type=ToolParameterType.STRING,
            description="The wallet address performing the bridge (0x... for EVM, base58 for Solana)",
            required=True,
        ),
        ToolParameter(
            name="from_chain",
            type=ToolParameterType.STRING,
            description=(
                "The source blockchain name (e.g., 'ethereum', 'base', 'arbitrum', "
                "'optimism', 'polygon', 'ink', 'zksync', 'solana', etc.)"
            ),
            required=True,
        ),
        ToolParameter(
            name="to_chain",
            type=ToolParameterType.STRING,
            description=(
                "The destination blockchain name (e.g., 'ethereum', 'base', 'arbitrum', "
                "'optimism', 'polygon', 'ink', 'zksync', 'solana', etc.)"
            ),
            required=True,
        ),
        ToolParameter(
            name="token",
            type=ToolParameterType.STRING,
            description=(
                "The token to bridge (e.g., 'ETH', 'USDC', 'USDT'). "
                "The same token type will be received on the destination chain."
            ),
            required=True,
        ),
        ToolParameter(
            name="amount",
            type=ToolParameterType.NUMBER,
            description="Amount of token to bridge (in human-readable units)",
            required=True,
        ),
        ToolParameter(
            name="destination_address",
            type=ToolParameterType.STRING,
            description=(
                "Optional: Recipient address on the destination chain. "
                "If not provided, tokens are sent to the same address (if compatible) "
                "or the user must specify for cross-ecosystem bridges (EVM ↔ Solana)."
            ),
            required=False,
        ),
    ],
    requires_address=True,
)
async def handle_get_bridge_quote(
    wallet_address: str,
    from_chain: str,
    to_chain: str,
    token: str,
    amount: float,
    destination_address: Optional[str] = None,
) -> Dict[str, Any]:
    """Handle cross-chain bridge quote request via Relay."""
    from decimal import Decimal
    from ....providers.relay import RelayProvider
    from ...bridge.chain_registry import get_chain_registry
    from ...swap.constants import TOKEN_REGISTRY, NATIVE_SOL_MINT

    try:
        # Resolve chain names to chain IDs
        registry = await get_chain_registry()

        from_chain_id = registry.get_chain_id(from_chain.lower())
        to_chain_id = registry.get_chain_id(to_chain.lower())

        if from_chain_id is None:
            supported = registry.get_supported_chain_names(10)
            return {
                "success": False,
                "error": f"Unknown source chain: {from_chain}",
                "hint": f"Supported chains include: {', '.join(supported)}",
            }

        if to_chain_id is None:
            supported = registry.get_supported_chain_names(10)
            return {
                "success": False,
                "error": f"Unknown destination chain: {to_chain}",
                "hint": f"Supported chains include: {', '.join(supported)}",
            }

        # Same chain = not a bridge
        if from_chain_id == to_chain_id:
            return {
                "success": False,
                "error": "Source and destination chains are the same",
                "hint": "For same-chain swaps, use get_swap_quote (EVM) or get_solana_swap_quote (Solana)",
            }

        # Determine if either chain is Solana
        from_is_solana = isinstance(from_chain_id, str) and from_chain_id.lower() == "solana"
        to_is_solana = isinstance(to_chain_id, str) and to_chain_id.lower() == "solana"

        # For cross-ecosystem bridges (EVM ↔ Solana), destination address is required
        if (from_is_solana or to_is_solana) and not destination_address:
            if from_is_solana and not to_is_solana:
                return {
                    "success": False,
                    "error": "Destination EVM address required for Solana → EVM bridge",
                    "hint": "Please provide the destination_address parameter with the EVM wallet address (0x...)",
                }
            elif to_is_solana and not from_is_solana:
                return {
                    "success": False,
                    "error": "Destination Solana address required for EVM → Solana bridge",
                    "hint": "Please provide the destination_address parameter with the Solana wallet address",
                }

        recipient = destination_address or wallet_address

        # Equivalent token mapping for bridged variants
        # Maps bridged token symbols to their canonical form
        EQUIVALENT_TOKENS = {
            "usdc.e": "usdc",
            "usdc.b": "usdc",
            "usdce": "usdc",
            "usdt.e": "usdt",
            "usdt.b": "usdt",
            "weth.e": "weth",
            "dai.e": "dai",
        }

        def get_token_address(token: str, chain_id, is_solana: bool, allow_equivalent: bool = False) -> tuple:
            """Get token address and decimals for a chain.

            Args:
                token: Token symbol or address
                chain_id: Target chain ID
                is_solana: Whether the chain is Solana
                allow_equivalent: If True, try equivalent tokens (e.g., USDC.e → USDC)

            Returns:
                Tuple of (address, decimals) or (None, None) if not found
            """
            token_lower = token.lower()

            if is_solana:
                # Solana token addresses
                solana_tokens = {
                    "sol": (NATIVE_SOL_MINT, 9),
                    "usdc": ("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", 6),
                    "usdt": ("Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB", 6),
                }
                if token_lower in solana_tokens:
                    return solana_tokens[token_lower]
                # Try equivalent token
                if allow_equivalent and token_lower in EQUIVALENT_TOKENS:
                    equiv = EQUIVALENT_TOKENS[token_lower]
                    if equiv in solana_tokens:
                        return solana_tokens[equiv]
                # Check if it's already a valid Solana address (base58, 32-44 chars)
                if len(token) >= 32 and len(token) <= 44 and token[0].isalnum():
                    return (token, 9)
                return (None, None)
            else:
                # EVM token addresses
                chain_tokens = TOKEN_REGISTRY.get(chain_id, {})
                for sym, meta in chain_tokens.items():
                    if sym.lower() == token_lower or token_lower in [a.lower() for a in meta.get("aliases", [])]:
                        return (meta["address"], meta["decimals"])

                # Try equivalent token (e.g., USDC.e → USDC)
                if allow_equivalent and token_lower in EQUIVALENT_TOKENS:
                    equiv = EQUIVALENT_TOKENS[token_lower]
                    for sym, meta in chain_tokens.items():
                        if sym.lower() == equiv or equiv in [a.lower() for a in meta.get("aliases", [])]:
                            return (meta["address"], meta["decimals"])

                # Native ETH
                if token_lower in ("eth", "ether", "native"):
                    return ("0x0000000000000000000000000000000000000000", 18)

                # Check if it's already a valid EVM address
                if token.startswith("0x") and len(token) == 42:
                    return (token, 18)

                return (None, None)

        # Resolve source token (exact match required)
        from_address, from_decimals = get_token_address(token, from_chain_id, from_is_solana, allow_equivalent=False)
        if from_address is None:
            return {
                "success": False,
                "error": f"Token '{token}' not found on source chain",
                "hint": f"Make sure the token exists on the source chain",
            }

        # Resolve destination token (allow equivalent tokens for bridging)
        to_address, _ = get_token_address(token, to_chain_id, to_is_solana, allow_equivalent=True)
        if to_address is None:
            return {
                "success": False,
                "error": f"Token '{token}' (or equivalent) not found on destination chain",
                "hint": f"The destination chain may not support this token",
            }

        # Convert amount to base units
        amount_decimal = Decimal(str(amount))
        amount_base_units = int(amount_decimal * (Decimal(10) ** from_decimals))

        if amount_base_units <= 0:
            return {
                "success": False,
                "error": "Amount must be greater than 0",
            }

        # Build Relay quote request
        from ...chain_types import RELAY_SOLANA_CHAIN_ID

        relay = RelayProvider()
        relay_payload = {
            "user": wallet_address,
            "originChainId": from_chain_id if not from_is_solana else RELAY_SOLANA_CHAIN_ID,
            "destinationChainId": to_chain_id if not to_is_solana else RELAY_SOLANA_CHAIN_ID,
            "originCurrency": from_address,
            "destinationCurrency": to_address,
            "recipient": recipient,
            "tradeType": "EXACT_INPUT",
            "amount": str(amount_base_units),
            "referrer": "sherpa.chat",
        }

        quote = await relay.quote(relay_payload)

        # Parse quote response
        details = quote.get("details", {})

        # Get output amount
        currency_out = details.get("currencyOut", {})
        output_amount_raw = currency_out.get("amount", "0")
        to_decimals = currency_out.get("decimals", from_decimals)
        output_amount = Decimal(output_amount_raw) / (Decimal(10) ** to_decimals)

        # Get fees breakdown
        fees = quote.get("fees", {})
        gas_fee = fees.get("gas", {})
        relayer_fee = fees.get("relayer", {})
        total_fee_usd = Decimal(str(details.get("totalFeeUsd", "0")))

        # Get time estimate
        time_estimate = details.get("timeEstimate", 0)  # seconds

        # Extract transactions from steps (frontend needs tx, approvals, etc.)
        steps = quote.get("steps", [])
        transactions = []
        approvals = []
        primary_tx = None

        for step in steps:
            for item in step.get("items", []):
                data_obj = item.get("data") or {}
                if isinstance(data_obj, dict):
                    # Transaction data (has 'to' and 'data' or 'value')
                    if "to" in data_obj and ("data" in data_obj or "value" in data_obj):
                        tx_entry = {
                            "to": data_obj.get("to"),
                            "data": data_obj.get("data"),
                            "value": data_obj.get("value", "0"),
                            "chainId": data_obj.get("chainId", from_chain_id),
                            "gas": data_obj.get("gas") or data_obj.get("gasLimit"),
                            "maxFeePerGas": data_obj.get("maxFeePerGas"),
                            "maxPriorityFeePerGas": data_obj.get("maxPriorityFeePerGas"),
                        }
                        transactions.append(tx_entry)
                        if primary_tx is None:
                            primary_tx = tx_entry
                    # Approval data
                    elif "spender" in data_obj and ("amount" in data_obj or "value" in data_obj):
                        approvals.append({
                            "tokenAddress": data_obj.get("token") or from_address,
                            "spenderAddress": data_obj.get("spender"),
                            "amount": data_obj.get("amount") or data_obj.get("value"),
                        })

        from_chain_name = registry.get_chain_name(from_chain_id) if not from_is_solana else "Solana"
        to_chain_name = registry.get_chain_name(to_chain_id) if not to_is_solana else "Solana"

        # Build response in format frontend expects
        result = {
            "success": True,
            "provider": "relay",
            "bridge_type": "cross_chain",
            "quote_type": "bridge",
            "from_chain": {
                "name": from_chain_name,
                "chain_id": from_chain_id,
                "is_solana": from_is_solana,
            },
            "to_chain": {
                "name": to_chain_name,
                "chain_id": to_chain_id,
                "is_solana": to_is_solana,
            },
            "token": {
                "symbol": token.upper(),
                "input_address": from_address,
                "amount": str(amount),
                "amount_base_units": str(amount_base_units),
            },
            "output": {
                "amount_estimate": str(output_amount),
                "recipient": recipient,
            },
            "fees": {
                "total_usd": str(total_fee_usd),
                "gas_usd": gas_fee.get("amountUsd"),
                "relayer_usd": relayer_fee.get("amountUsd"),
            },
            "time_estimate_seconds": time_estimate,
            "steps_count": len(steps),
            # Frontend execution fields
            "wallet": {"address": wallet_address},
            "amounts": {
                "input_amount_wei": str(amount_base_units),
                "input_base_units": str(amount_base_units),
            },
            "breakdown": {
                "input": {
                    "symbol": token.upper(),
                    "amount": str(amount),
                    "token_address": from_address,
                },
                "output": {
                    "symbol": token.upper(),
                    "amount_estimate": str(output_amount),
                    "token_address": to_address,
                },
            },
            "instructions": [
                f"Bridge {amount} {token.upper()} from {from_chain_name} to {to_chain_name}",
                f"Expected output: ~{output_amount:.6f} {token.upper()}",
                f"Estimated fees: ${total_fee_usd:.2f}",
                f"Estimated time: {time_estimate // 60} min {time_estimate % 60} sec" if time_estimate else "Time varies",
                "Review and sign the transaction(s) in your wallet to execute the bridge.",
            ],
            "quote_data": quote,  # Full quote for debugging
        }

        # Add transaction data if available (required for execution)
        if primary_tx:
            result["tx"] = primary_tx
            result["tx_ready"] = True
        else:
            result["tx_ready"] = False

        if approvals:
            result["approval_data"] = approvals[0]  # Frontend expects single approval
            result["approvals"] = approvals

        if transactions:
            result["transactions"] = transactions

        return result

    except Exception as e:
        logger.error(f"Error in bridge quote: {e}")
        return {"success": False, "error": str(e)}
