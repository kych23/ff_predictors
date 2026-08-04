"""Fernet-based token-at-rest encryption (api/crypto.py)."""
import pytest
from cryptography.fernet import Fernet


def test_encrypt_roundtrip(monkeypatch):
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", Fernet.generate_key().decode())
    from importlib import reload
    import api.crypto as c
    reload(c)
    token = "ya29.secret-access-token"
    assert c.decrypt(c.encrypt(token)) == token


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("TOKEN_ENCRYPTION_KEY", raising=False)
    from importlib import reload
    import api.crypto as c
    reload(c)
    with pytest.raises(RuntimeError):
        c.encrypt("x")
