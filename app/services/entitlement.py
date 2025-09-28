from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal, ROUND_FLOOR
from typing import Optional

from ..cache import cache
from ..config import settings
from ..providers.alchemy import AlchemyProvider
from ..types import EntitlementResponse

_ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")
_VALID_STANDARDS = {"erc20", "erc721", "erc1155"}


class EntitlementError(Exception):
    """Raised when the entitlement configuration is invalid."""


def _normalize_address(address: str) -> str:
    if not address:
        raise EntitlementError("Wallet address is required")
    address = address.strip()
    if not _ADDRESS_RE.fullmatch(address):
        raise EntitlementError("Invalid wallet address")
    return address.lower()


def _normalize_token_address(address: str) -> str:
    address = address.strip()
    if not _ADDRESS_RE.fullmatch(address):
        raise EntitlementError("Invalid entitlement token address")
    return address.lower()


def _normalize_token_id(token_id: str) -> str:
    value = token_id.strip().lower()
    if value.startswith("0x"):
        return format(int(value, 16), "x")
    return format(int(value, 10), "x")


def _parse_int(value: Optional[str]) -> int:
    if value is None:
        return 0
    raw = value.strip()
    if not raw:
        return 0
    if raw.startswith("0x"):
        return int(raw, 16)
    return int(raw, 10)


async def evaluate_entitlement(address: str, chain: Optional[str] = None) -> EntitlementResponse:
    """Evaluate Pro entitlement for a wallet address."""

    normalized_chain = (chain or settings.pro_token_chain or "ethereum").lower()
    normalized_address = _normalize_address(address)

    token_address_raw = settings.pro_token_address.strip()
    if not token_address_raw:
        return EntitlementResponse(
            address=normalized_address,
            chain=normalized_chain,
            pro=False,
            gating="disabled",
            reason="Entitlement token not configured",
        )

    try:
        token_address = _normalize_token_address(token_address_raw)
    except EntitlementError as exc:  # Surface config issues
        return EntitlementResponse(
            address=normalized_address,
            chain=normalized_chain,
            pro=False,
            gating="error",
            reason=str(exc),
        )

    standard = (settings.pro_token_standard or "erc20").lower()
    if standard not in _VALID_STANDARDS:
        return EntitlementResponse(
            address=normalized_address,
            chain=normalized_chain,
            pro=False,
            gating="error",
            reason=f"Unsupported entitlement standard: {standard}",
            token_address=token_address,
            standard=standard,
        )

    cache_key = f"entitlement:{standard}:{normalized_chain}:{normalized_address}"
    cached_response = await cache.get(cache_key)
    if cached_response:
        return cached_response.model_copy(update={"cached": True})

    provider = AlchemyProvider()
    if not await provider.ready():
        response = EntitlementResponse(
            address=normalized_address,
            chain=normalized_chain,
            pro=False,
            gating="token",
            standard=standard,
            token_address=token_address,
            reason="Indexing provider unavailable",
            metadata={"provider": "alchemy", "ready": False},
        )
        await cache.set(cache_key, response, ttl=60)
        return response

    checked_at = datetime.utcnow()
    pro = False
    reason: Optional[str] = None
    metadata: dict[str, str | int | float | Decimal] = {}

    try:
        if standard == "erc20":
            token_balance = await provider.get_token_balance_for_contract(
                normalized_address,
                token_address,
                normalized_chain,
            )
            balance_hex = token_balance.get("tokenBalance", "0x0")
            balance_int = _parse_int(balance_hex)
            decimals = max(settings.pro_token_decimals, 0)
            min_balance = settings.pro_token_min_balance
            multiplier = Decimal(10) ** decimals
            min_units = int((min_balance * multiplier).to_integral_value(rounding=ROUND_FLOOR)) if min_balance > 0 else 1 if min_balance == Decimal("0") else 0
            if min_balance == Decimal("0"):
                # Any balance unlocks
                pro = balance_int > 0
            else:
                pro = balance_int >= min_units
            if not pro:
                human_requirement = (
                    f"Hold at least {min_balance.normalize()} tokens"
                    if min_balance > 0
                    else "Hold any balance"
                )
                reason = f"{human_requirement} of {token_address}"
            metadata = {
                "balance_hex": balance_hex,
                "balance_int": balance_int,
                "min_units": min_units,
                "decimals": decimals,
                "min_balance": str(min_balance),
            }
        elif standard == "erc721":
            nft_data = await provider.get_owned_nfts(
                normalized_address,
                token_address,
                normalized_chain,
            )
            total_owned = int(nft_data.get("total", 0))
            min_required = int(settings.pro_token_min_balance.to_integral_value(rounding=ROUND_FLOOR)) if settings.pro_token_min_balance > 0 else 1
            pro = total_owned >= max(min_required, 1)
            if not pro:
                target = max(min_required, 1)
                reason = f"Requires owning {target} NFT(s) from {token_address}"
            metadata = {
                "owned": total_owned,
                "min_required": max(min_required, 1),
            }
        else:  # ERC-1155
            token_id_raw = settings.pro_token_id
            if not token_id_raw:
                raise EntitlementError("Token ID required for ERC-1155 entitlement")
            target_token_id = _normalize_token_id(token_id_raw)
            nft_data = await provider.get_owned_nfts(
                normalized_address,
                token_address,
                normalized_chain,
                page_size=100,
            )
            owned_nfts = nft_data.get("owned_nfts", [])
            owned_balance = 0
            for nft in owned_nfts:
                nft_id = nft.get("id", {}).get("tokenId")
                if nft_id and _normalize_token_id(nft_id) == target_token_id:
                    owned_balance = _parse_int(nft.get("balance"))
                    break
            min_required = int(settings.pro_token_min_balance.to_integral_value(rounding=ROUND_FLOOR)) if settings.pro_token_min_balance > 0 else 1
            pro = owned_balance >= max(min_required, 1)
            if not pro:
                reason = f"Requires at least {max(min_required, 1)} unit(s) of token ID {token_id_raw}"
            metadata = {
                "owned_balance": owned_balance,
                "min_required": max(min_required, 1),
                "token_id_normalized": target_token_id,
            }
    except EntitlementError as exc:
        response = EntitlementResponse(
            address=normalized_address,
            chain=normalized_chain,
            pro=False,
            gating="error",
            standard=standard,
            token_address=token_address,
            token_id=settings.pro_token_id,
            reason=str(exc),
        )
        await cache.set(cache_key, response, ttl=60)
        return response
    except Exception as exc:  # catch provider issues
        response = EntitlementResponse(
            address=normalized_address,
            chain=normalized_chain,
            pro=False,
            gating="token",
            standard=standard,
            token_address=token_address,
            token_id=settings.pro_token_id,
            reason=f"Entitlement check failed: {exc}",
            metadata={"error": str(exc)},
        )
        await cache.set(cache_key, response, ttl=30)
        return response

    response = EntitlementResponse(
        address=normalized_address,
        chain=normalized_chain,
        pro=pro,
        gating="token",
        standard=standard,
        token_address=token_address,
        token_id=settings.pro_token_id,
        reason=reason,
        checked_at=checked_at,
        metadata=metadata,
    )
    await cache.set(cache_key, response, ttl=180)
    return response
