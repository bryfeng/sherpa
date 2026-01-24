"""
Solana wallet sign-in message parsing and signature verification.
"""

from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from nacl.exceptions import BadSignatureError
from nacl.signing import VerifyKey

from app.services.address import is_valid_solana_address


_BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_BASE58_INDEX = {char: index for index, char in enumerate(_BASE58_ALPHABET)}

_HEADER_RE = re.compile(
    r"^(?P<domain>.+?) wants you to sign in with your Solana (account|address):$",
    re.IGNORECASE,
)
_FIELD_RE = re.compile(r"^(?P<key>[A-Za-z][A-Za-z0-9 ]+):\s*(?P<value>.*)$")


@dataclass
class SolanaSignInMessage:
    domain: str
    address: str
    nonce: str
    statement: Optional[str] = None
    uri: Optional[str] = None
    version: Optional[str] = None
    chain_id: Optional[str] = None
    issued_at: Optional[str] = None
    expiration_time: Optional[str] = None
    not_before: Optional[str] = None
    request_id: Optional[str] = None
    resources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "domain": self.domain,
            "address": self.address,
            "nonce": self.nonce,
            "statement": self.statement,
            "uri": self.uri,
            "version": self.version,
            "chain_id": self.chain_id,
            "issued_at": self.issued_at,
            "expiration_time": self.expiration_time,
            "not_before": self.not_before,
            "request_id": self.request_id,
        }


def parse_solana_signin_message(message: str) -> SolanaSignInMessage:
    """
    Parse a Sign-In with Solana message.

    Expected format (flexible, fields optional):
        <domain> wants you to sign in with your Solana account:
        <address>

        <statement>

        URI: <uri>
        Version: <version>
        Chain ID: <chain-id>
        Nonce: <nonce>
        Issued At: <timestamp>
        Resources:
        - <resource>
    """
    lines = message.splitlines()
    if not lines:
        raise ValueError("Empty sign-in message")

    header_index = None
    domain = None
    for idx, line in enumerate(lines):
        match = _HEADER_RE.match(line.strip())
        if match:
            header_index = idx
            domain = match.group("domain").strip()
            break

    if header_index is None or not domain:
        raise ValueError("Invalid Solana sign-in header")

    address_index = _next_non_empty_index(lines, header_index + 1)
    if address_index is None:
        raise ValueError("Missing Solana address line")

    address = lines[address_index].strip()
    if not is_valid_solana_address(address):
        raise ValueError("Invalid Solana address in message")

    idx = address_index + 1
    if idx < len(lines) and lines[idx].strip() == "":
        idx += 1

    statement_lines: List[str] = []
    while idx < len(lines):
        line = lines[idx]
        if line.strip() == "":
            idx += 1
            break
        if _FIELD_RE.match(line.strip()):
            break
        statement_lines.append(line.rstrip())
        idx += 1

    statement = "\n".join(statement_lines).strip() if statement_lines else None

    fields: Dict[str, str] = {}
    resources: List[str] = []
    while idx < len(lines):
        line = lines[idx].rstrip()
        if line.strip() == "":
            idx += 1
            continue
        if line.strip().lower().startswith("resources:"):
            idx += 1
            while idx < len(lines):
                resource_line = lines[idx].strip()
                if not resource_line.startswith("- "):
                    break
                resources.append(resource_line[2:].strip())
                idx += 1
            continue
        match = _FIELD_RE.match(line.strip())
        if match:
            key = _normalize_key(match.group("key"))
            fields[key] = match.group("value").strip()
        idx += 1

    nonce = fields.get("nonce")
    if not nonce:
        raise ValueError("Solana sign-in message missing nonce")

    return SolanaSignInMessage(
        domain=domain,
        address=address,
        nonce=nonce,
        statement=statement,
        uri=fields.get("uri"),
        version=fields.get("version"),
        chain_id=fields.get("chain_id"),
        issued_at=fields.get("issued_at"),
        expiration_time=fields.get("expiration_time"),
        not_before=fields.get("not_before"),
        request_id=fields.get("request_id"),
        resources=resources,
    )


def verify_solana_signature(
    message: str,
    signature: str,
    address: str,
) -> None:
    """
    Verify a Solana signMessage signature.
    Raises ValueError if verification fails.
    """
    public_key = base58_decode(address)
    if len(public_key) != 32:
        raise ValueError("Invalid Solana public key length")

    signature_bytes = _decode_signature(signature)
    verify_key = VerifyKey(public_key)
    try:
        verify_key.verify(message.encode("utf-8"), signature_bytes)
    except BadSignatureError as exc:
        raise ValueError("Invalid Solana signature") from exc


def base58_decode(value: str) -> bytes:
    if not value:
        return b""
    num = 0
    for char in value:
        if char not in _BASE58_INDEX:
            raise ValueError("Invalid base58 character")
        num = num * 58 + _BASE58_INDEX[char]
    combined = num.to_bytes((num.bit_length() + 7) // 8, "big") if num else b""
    pad = len(value) - len(value.lstrip("1"))
    return b"\x00" * pad + combined


def base58_encode(data: bytes) -> str:
    if not data:
        return ""
    num = int.from_bytes(data, "big")
    encoded = ""
    while num > 0:
        num, rem = divmod(num, 58)
        encoded = _BASE58_ALPHABET[rem] + encoded
    pad = 0
    for byte in data:
        if byte == 0:
            pad += 1
        else:
            break
    return "1" * pad + encoded


def _decode_signature(signature: str) -> bytes:
    candidate = signature.strip()
    if not candidate:
        raise ValueError("Signature is empty")

    if _looks_base58(candidate):
        try:
            return base58_decode(candidate)
        except ValueError:
            pass

    try:
        return base64.b64decode(candidate, validate=True)
    except (ValueError, binascii.Error):
        try:
            padded = candidate + "=" * (-len(candidate) % 4)
            return base64.urlsafe_b64decode(padded)
        except Exception as exc:
            raise ValueError("Unsupported signature encoding") from exc


def _looks_base58(value: str) -> bool:
    return all(char in _BASE58_INDEX for char in value)


def _normalize_key(key: str) -> str:
    return key.strip().lower().replace(" ", "_")


def _next_non_empty_index(lines: List[str], start: int) -> Optional[int]:
    for idx in range(start, len(lines)):
        if lines[idx].strip():
            return idx
    return None
