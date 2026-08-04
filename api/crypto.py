"""Token-at-rest encryption for stored OAuth credentials (Yahoo, etc.)."""
from __future__ import annotations

import os

from cryptography.fernet import Fernet


def _cipher() -> Fernet:
    key = os.getenv("TOKEN_ENCRYPTION_KEY")
    if not key:
        raise RuntimeError("TOKEN_ENCRYPTION_KEY not set — cannot handle OAuth tokens")
    return Fernet(key.encode() if isinstance(key, str) else key)


def encrypt(plaintext: str) -> bytes:
    return _cipher().encrypt(plaintext.encode())


def decrypt(token: bytes) -> str:
    return _cipher().decrypt(token).decode()
