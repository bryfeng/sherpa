from nacl.signing import SigningKey
import pytest

from app.auth.solana_signin import (
    base58_encode,
    parse_solana_signin_message,
    verify_solana_signature,
)


def _build_message(domain: str, address: str, nonce: str) -> str:
    return (
        f"{domain} wants you to sign in with your Solana account:\n"
        f"{address}\n"
        "\n"
        "Sign in to Sherpa.\n"
        "\n"
        f"URI: https://{domain}\n"
        "Version: 1\n"
        "Chain ID: solana\n"
        f"Nonce: {nonce}\n"
        "Issued At: 2026-01-21T00:00:00Z"
    )


def test_parse_solana_signin_message_extracts_fields() -> None:
    signing_key = SigningKey.generate()
    address = base58_encode(signing_key.verify_key.encode())
    message = _build_message("example.com", address, "nonce-123")

    parsed = parse_solana_signin_message(message)

    assert parsed.address == address
    assert parsed.nonce == "nonce-123"
    assert parsed.chain_id == "solana"
    assert parsed.domain == "example.com"


def test_verify_solana_signature_accepts_valid_signature() -> None:
    signing_key = SigningKey.generate()
    address = base58_encode(signing_key.verify_key.encode())
    message = _build_message("example.com", address, "nonce-abc")

    signature = signing_key.sign(message.encode("utf-8")).signature
    signature_base58 = base58_encode(signature)

    verify_solana_signature(message, signature_base58, address)


def test_verify_solana_signature_rejects_invalid_signature() -> None:
    signing_key = SigningKey.generate()
    other_key = SigningKey.generate()
    address = base58_encode(signing_key.verify_key.encode())
    message = _build_message("example.com", address, "nonce-xyz")

    signature = other_key.sign(message.encode("utf-8")).signature
    signature_base58 = base58_encode(signature)

    with pytest.raises(ValueError):
        verify_solana_signature(message, signature_base58, address)
