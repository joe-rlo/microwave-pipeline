"""Encryption substrate for the Health Profile (Phase G.1.a).

The profile holds PHI. Storage is one encrypted JSON blob per user in
SQLite, with the encryption key derived from a master key + the
user_id via HKDF-SHA256. This module owns the crypto and the master-
key plumbing; the storage layer (Phase G.1.b) just calls encrypt /
decrypt without caring where the key came from.

Three master-key sources, chosen by `phi_encryption_key_source` config:

- `keychain` (default on macOS / Linux): OS keychain via the `keyring`
  package. First access generates a fresh 32-byte key and stores it;
  subsequent accesses retrieve. Most users want this.

- `env`: read base64-encoded key from `PHI_ENCRYPTION_MASTER_KEY`.
  Useful for CI / containerized deployments where keyring access
  isn't practical. **Less secure** — the key sits in env / .env.

- `kms`: AWS KMS data-key envelope. Not implemented in Phase G.1.a
  (stubbed with a clear "not yet" error). Lands when there's a real
  multi-user / production deployment that needs it.

Crypto choices:

- AES-GCM with random 12-byte (96-bit) nonces. Industry standard,
  authenticated. Tag is 16 bytes (default).
- HKDF-SHA256 for per-user key derivation. info carries a versioned
  prefix + user_id so a future scheme rotation has a clean upgrade
  path (`v1:user=` → `v2:user=`).
- The encrypted blob format is `nonce || ciphertext_with_tag`. Self-
  describing (no separate IV column), simple to decrypt.

Why a custom module instead of `cryptography.fernet`: Fernet uses
AES-CBC + HMAC which is older and not authenticated in one pass.
AES-GCM is the modern choice. The wrapping here is intentionally thin
so the actual crypto library does the heavy lifting.
"""

from __future__ import annotations

import base64
import logging
import os
import secrets
from typing import Literal

log = logging.getLogger(__name__)


# AES-GCM standard parameters. 256-bit key, 96-bit nonce — the
# RFC 5116 recommendation.
KEY_LEN = 32
NONCE_LEN = 12
# Minimum size of an encrypted blob: nonce + GCM tag (16 bytes) with
# zero-byte plaintext. Anything smaller can't be valid.
MIN_BLOB_LEN = NONCE_LEN + 16

# HKDF info prefix — versioned so a future key-scheme change has a
# clean migration path (existing blobs stay readable; new ones use
# the new derivation).
HKDF_INFO_V1 = b"microwaveos:profile:v1:user="

# Keychain service / account names. Match `keyring`'s
# (service, username) addressing.
_KEYRING_SERVICE = "microwaveos-phi"
_KEYRING_ACCOUNT = "master-key-v1"

# Env-var name used by the `env` source.
ENV_MASTER_KEY = "PHI_ENCRYPTION_MASTER_KEY"


KeySource = Literal["keychain", "env", "kms"]


class CryptoError(RuntimeError):
    """Raised on any crypto / key-source failure. Safe to surface to
    callers; messages never include key material."""


# --- Key derivation -------------------------------------------------------


def derive_user_key(master_key: bytes, user_id: str) -> bytes:
    """Derive a 256-bit per-user key from the master via HKDF-SHA256.

    Deterministic for a given (master, user_id) pair — same inputs
    always yield the same key, so a stored blob is readable across
    restarts.
    """
    if len(master_key) < 16:
        raise CryptoError(
            f"Master key too short ({len(master_key)} bytes); expected >= 16"
        )
    if not user_id:
        raise CryptoError("user_id must be non-empty")

    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    info = HKDF_INFO_V1 + user_id.encode("utf-8")
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=KEY_LEN,
        salt=None,  # HKDF without salt is fine when info carries entropy
        info=info,
    )
    return hkdf.derive(master_key)


# --- AES-GCM encrypt / decrypt -------------------------------------------


def encrypt(plaintext: bytes, key: bytes) -> bytes:
    """AES-GCM encrypt. Returns `nonce || ciphertext_with_tag`.

    Each call uses a fresh random nonce so the same plaintext under
    the same key produces different blobs. This is required for
    GCM security.
    """
    if len(key) != KEY_LEN:
        raise CryptoError(f"Key must be {KEY_LEN} bytes, got {len(key)}")
    if not isinstance(plaintext, (bytes, bytearray)):
        raise CryptoError("plaintext must be bytes")

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = secrets.token_bytes(NONCE_LEN)
    aesgcm = AESGCM(key)
    ct = aesgcm.encrypt(nonce, bytes(plaintext), associated_data=None)
    return nonce + ct


def decrypt(blob: bytes, key: bytes) -> bytes:
    """AES-GCM decrypt. Expects `nonce || ciphertext_with_tag`.

    Raises `CryptoError` on any failure — wrong key, truncated blob,
    or authentication-tag mismatch. The GCM tag catches tampering;
    a single bit flip in the ciphertext makes decryption fail.
    """
    if len(key) != KEY_LEN:
        raise CryptoError(f"Key must be {KEY_LEN} bytes, got {len(key)}")
    if not isinstance(blob, (bytes, bytearray)):
        raise CryptoError("blob must be bytes")
    if len(blob) < MIN_BLOB_LEN:
        raise CryptoError(
            f"Blob too short ({len(blob)} bytes); expected >= {MIN_BLOB_LEN}"
        )

    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce, ct = blob[:NONCE_LEN], blob[NONCE_LEN:]
    aesgcm = AESGCM(key)
    try:
        return aesgcm.decrypt(nonce, bytes(ct), associated_data=None)
    except InvalidTag as e:
        # Don't include key or ciphertext bytes in the message — even
        # the failure surface shouldn't leak material.
        raise CryptoError("Decryption failed (tag mismatch — wrong key or tampered blob)") from e


# --- Master-key sources ---------------------------------------------------


def get_master_key(source: KeySource = "keychain") -> bytes:
    """Load the master key from the configured source.

    Generates + stores a fresh key on first access for `keychain`. The
    `env` source expects the key to already exist; missing key is a
    hard error (no silent fallback).

    Returns raw bytes — never a base64 / hex string. Callers should
    avoid logging or pickling the result.
    """
    if source == "env":
        return _master_key_from_env()
    if source == "keychain":
        return _master_key_from_keychain()
    if source == "kms":
        raise CryptoError(
            "KMS key source is not implemented yet. Use keychain (default) "
            "or env for now. Track Phase G.x for KMS."
        )
    raise CryptoError(f"Unknown PHI key source: {source!r}")


def _master_key_from_env() -> bytes:
    raw = os.environ.get(ENV_MASTER_KEY, "").strip()
    if not raw:
        raise CryptoError(
            f"{ENV_MASTER_KEY} is not set. Generate one with: "
            "python3 -c 'import secrets, base64; "
            "print(base64.b64encode(secrets.token_bytes(32)).decode())'"
        )
    try:
        key = base64.b64decode(raw)
    except Exception as e:
        raise CryptoError(
            f"{ENV_MASTER_KEY} is not valid base64: {e}"
        ) from e
    if len(key) != KEY_LEN:
        raise CryptoError(
            f"{ENV_MASTER_KEY} decoded to {len(key)} bytes; expected {KEY_LEN}"
        )
    return key


def _master_key_from_keychain() -> bytes:
    """Read the master key from the OS keychain, generating + storing
    a fresh one if no existing entry."""
    try:
        import keyring
    except ImportError as e:
        raise CryptoError(
            "keychain key source requires the `keyring` package. "
            "Install with `pip install keyring`, or set "
            "phi_encryption_key_source='env'."
        ) from e

    try:
        stored = keyring.get_password(_KEYRING_SERVICE, _KEYRING_ACCOUNT)
    except Exception as e:
        # macOS Keychain occasionally surfaces RuntimeError on locked
        # screens / permission denials. Bubble up with context.
        raise CryptoError(f"Keychain read failed: {e}") from e

    if stored:
        try:
            key = base64.b64decode(stored)
        except Exception as e:
            raise CryptoError(
                f"Keychain entry for {_KEYRING_SERVICE!r} is corrupt: {e}"
            ) from e
        if len(key) != KEY_LEN:
            raise CryptoError(
                f"Keychain entry decoded to {len(key)} bytes; expected {KEY_LEN}. "
                "Manually delete the entry to regenerate."
            )
        return key

    # First access — generate, store, return.
    fresh = secrets.token_bytes(KEY_LEN)
    encoded = base64.b64encode(fresh).decode("ascii")
    try:
        keyring.set_password(_KEYRING_SERVICE, _KEYRING_ACCOUNT, encoded)
    except Exception as e:
        raise CryptoError(
            f"Could not store fresh master key in keychain: {e}"
        ) from e
    log.info(
        "Generated fresh PHI master key in keychain (%s/%s)",
        _KEYRING_SERVICE, _KEYRING_ACCOUNT,
    )
    return fresh


# --- Convenience: full envelope -------------------------------------------


def encrypt_for_user(plaintext: bytes, user_id: str, *, source: KeySource = "keychain") -> bytes:
    """Convenience: derive the user key from the configured master and encrypt."""
    master = get_master_key(source)
    user_key = derive_user_key(master, user_id)
    return encrypt(plaintext, user_key)


def decrypt_for_user(blob: bytes, user_id: str, *, source: KeySource = "keychain") -> bytes:
    """Convenience: derive the user key and decrypt."""
    master = get_master_key(source)
    user_key = derive_user_key(master, user_id)
    return decrypt(blob, user_key)
