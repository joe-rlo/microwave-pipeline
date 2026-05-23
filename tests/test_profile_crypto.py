"""Tests for the profile encryption substrate (Phase G.1.a).

We use the real `cryptography` library — there's no useful way to
mock crypto primitives. `keyring` is mocked because real keychain
access would be machine-dependent.

Coverage:
- derive_user_key: deterministic for same inputs, different user_ids
  produce different keys, empty user_id rejected, short master rejected
- encrypt/decrypt: roundtrip, wrong key fails with tag mismatch,
  tampered blob fails with tag mismatch, key-length validation,
  blob-length validation, nonces are unique per call
- get_master_key("env"): reads base64-decoded key, missing env raises,
  invalid base64 raises, wrong length raises
- get_master_key("keychain"): reads existing entry, generates+stores
  if missing, corrupt entry raises, length-mismatch entry raises
- get_master_key("kms"): raises clear "not yet implemented" error
- Unknown source raises
- encrypt_for_user / decrypt_for_user end-to-end
"""

from __future__ import annotations

import base64
import sys

import pytest

from src.health.profile.crypto import (
    ENV_MASTER_KEY,
    KEY_LEN,
    MIN_BLOB_LEN,
    NONCE_LEN,
    CryptoError,
    decrypt,
    decrypt_for_user,
    derive_user_key,
    encrypt,
    encrypt_for_user,
    get_master_key,
)


# --- derive_user_key ---


class TestDeriveUserKey:
    def test_deterministic_for_same_inputs(self):
        master = b"x" * 32
        a = derive_user_key(master, "self")
        b = derive_user_key(master, "self")
        assert a == b
        assert len(a) == KEY_LEN

    def test_different_user_ids_yield_different_keys(self):
        master = b"x" * 32
        a = derive_user_key(master, "self")
        b = derive_user_key(master, "other")
        assert a != b

    def test_different_masters_yield_different_keys(self):
        a = derive_user_key(b"x" * 32, "self")
        b = derive_user_key(b"y" * 32, "self")
        assert a != b

    def test_empty_user_id_rejected(self):
        with pytest.raises(CryptoError, match="user_id"):
            derive_user_key(b"x" * 32, "")

    def test_short_master_rejected(self):
        with pytest.raises(CryptoError, match="too short"):
            derive_user_key(b"x" * 10, "self")


# --- encrypt / decrypt ---


class TestEncryptDecrypt:
    def test_roundtrip(self):
        key = b"k" * KEY_LEN
        pt = b"hello world"
        blob = encrypt(pt, key)
        assert decrypt(blob, key) == pt

    def test_roundtrip_with_empty_plaintext(self):
        key = b"k" * KEY_LEN
        blob = encrypt(b"", key)
        # Should still produce a valid blob (nonce + tag)
        assert len(blob) >= MIN_BLOB_LEN
        assert decrypt(blob, key) == b""

    def test_roundtrip_with_large_plaintext(self):
        key = b"k" * KEY_LEN
        pt = b"x" * 1_000_000  # 1MB
        blob = encrypt(pt, key)
        assert decrypt(blob, key) == pt

    def test_wrong_key_fails(self):
        key_a = b"a" * KEY_LEN
        key_b = b"b" * KEY_LEN
        blob = encrypt(b"secret", key_a)
        with pytest.raises(CryptoError, match="tag mismatch"):
            decrypt(blob, key_b)

    def test_tampered_blob_fails(self):
        key = b"k" * KEY_LEN
        blob = bytearray(encrypt(b"important", key))
        # Flip a bit deep in the ciphertext (past the nonce)
        blob[NONCE_LEN + 2] ^= 0x01
        with pytest.raises(CryptoError, match="tag mismatch"):
            decrypt(bytes(blob), key)

    def test_truncated_blob_fails(self):
        key = b"k" * KEY_LEN
        with pytest.raises(CryptoError, match="too short"):
            decrypt(b"x" * (MIN_BLOB_LEN - 1), key)

    def test_wrong_key_length_rejected(self):
        with pytest.raises(CryptoError, match="must be"):
            encrypt(b"hi", b"k" * 16)  # 128-bit, not 256-bit
        with pytest.raises(CryptoError, match="must be"):
            decrypt(b"k" * 32, b"k" * 16)

    def test_each_encrypt_uses_fresh_nonce(self):
        # Same plaintext + key → different blobs because the nonce
        # is random each call. This is what makes GCM safe.
        key = b"k" * KEY_LEN
        a = encrypt(b"same input", key)
        b = encrypt(b"same input", key)
        assert a != b
        # But both decrypt to the same plaintext
        assert decrypt(a, key) == decrypt(b, key) == b"same input"

    def test_non_bytes_inputs_rejected(self):
        key = b"k" * KEY_LEN
        with pytest.raises(CryptoError, match="bytes"):
            encrypt("string-not-bytes", key)  # type: ignore[arg-type]
        with pytest.raises(CryptoError, match="bytes"):
            decrypt("string-not-bytes", key)  # type: ignore[arg-type]


# --- get_master_key("env") ---


class TestEnvMasterKey:
    def test_reads_base64_key(self, monkeypatch):
        key = b"x" * KEY_LEN
        monkeypatch.setenv(ENV_MASTER_KEY, base64.b64encode(key).decode())
        assert get_master_key("env") == key

    def test_missing_raises(self, monkeypatch):
        monkeypatch.delenv(ENV_MASTER_KEY, raising=False)
        with pytest.raises(CryptoError, match="not set"):
            get_master_key("env")

    def test_invalid_base64_raises(self, monkeypatch):
        monkeypatch.setenv(ENV_MASTER_KEY, "!@#$%^&*not-base64")
        with pytest.raises(CryptoError, match="not valid base64"):
            get_master_key("env")

    def test_wrong_length_raises(self, monkeypatch):
        # 16-byte key, but we require 32
        monkeypatch.setenv(ENV_MASTER_KEY, base64.b64encode(b"x" * 16).decode())
        with pytest.raises(CryptoError, match="expected 32"):
            get_master_key("env")


# --- get_master_key("keychain") ---


class _FakeKeyring:
    """Minimal in-memory keyring substitute."""

    def __init__(self):
        self.store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, account: str) -> str | None:
        return self.store.get((service, account))

    def set_password(self, service: str, account: str, password: str) -> None:
        self.store[(service, account)] = password

    def delete_password(self, service: str, account: str) -> None:
        self.store.pop((service, account), None)


@pytest.fixture
def fake_keyring(monkeypatch):
    fake = _FakeKeyring()
    monkeypatch.setitem(sys.modules, "keyring", fake)
    return fake


class TestKeychainMasterKey:
    def test_generates_fresh_key_on_first_access(self, fake_keyring):
        # Empty keychain → first call generates + stores
        key = get_master_key("keychain")
        assert len(key) == KEY_LEN
        # Verify stored in the (microwaveos-phi, master-key-v1) slot
        from src.health.profile.crypto import _KEYRING_ACCOUNT, _KEYRING_SERVICE
        stored = fake_keyring.get_password(_KEYRING_SERVICE, _KEYRING_ACCOUNT)
        assert stored is not None
        assert base64.b64decode(stored) == key

    def test_second_access_returns_same_key(self, fake_keyring):
        key1 = get_master_key("keychain")
        key2 = get_master_key("keychain")
        assert key1 == key2

    def test_corrupt_entry_raises(self, fake_keyring):
        from src.health.profile.crypto import _KEYRING_ACCOUNT, _KEYRING_SERVICE
        fake_keyring.set_password(_KEYRING_SERVICE, _KEYRING_ACCOUNT, "not-base64!!!")
        with pytest.raises(CryptoError, match="corrupt"):
            get_master_key("keychain")

    def test_wrong_length_entry_raises(self, fake_keyring):
        from src.health.profile.crypto import _KEYRING_ACCOUNT, _KEYRING_SERVICE
        # 16-byte key, but we require 32
        fake_keyring.set_password(
            _KEYRING_SERVICE, _KEYRING_ACCOUNT,
            base64.b64encode(b"x" * 16).decode(),
        )
        with pytest.raises(CryptoError, match="Manually delete"):
            get_master_key("keychain")

    def test_missing_package_raises_clean_error(self, monkeypatch):
        # Simulate `import keyring` failing
        monkeypatch.setitem(sys.modules, "keyring", None)
        with pytest.raises(CryptoError, match="`keyring` package"):
            get_master_key("keychain")


# --- get_master_key("kms") and unknown source ---


class TestKmsAndUnknown:
    def test_kms_raises_not_implemented(self):
        with pytest.raises(CryptoError, match="not implemented"):
            get_master_key("kms")

    def test_unknown_source_raises(self):
        with pytest.raises(CryptoError, match="Unknown PHI key source"):
            get_master_key("bogus")  # type: ignore[arg-type]


# --- encrypt_for_user / decrypt_for_user end-to-end ---


class TestForUserHelpers:
    def test_roundtrip_via_keychain(self, fake_keyring):
        pt = b"my profile data"
        blob = encrypt_for_user(pt, "self", source="keychain")
        assert decrypt_for_user(blob, "self", source="keychain") == pt

    def test_different_users_cant_decrypt(self, fake_keyring):
        pt = b"alice's data"
        blob = encrypt_for_user(pt, "alice", source="keychain")
        # Bob can't decrypt — different derived key
        with pytest.raises(CryptoError):
            decrypt_for_user(blob, "bob", source="keychain")

    def test_roundtrip_via_env(self, monkeypatch):
        master = base64.b64encode(b"k" * KEY_LEN).decode()
        monkeypatch.setenv(ENV_MASTER_KEY, master)
        pt = b"env-source roundtrip"
        blob = encrypt_for_user(pt, "self", source="env")
        assert decrypt_for_user(blob, "self", source="env") == pt
