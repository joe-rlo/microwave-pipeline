"""Health module configuration.

Lives separately from the top-level Config so the health-specific
surface stays browsable and so a non-health install (most users) can
ignore the whole namespace. The top-level Config exposes this as
`config.health` for convenience.

Phase 1 implements the general path only — `phi_path_enabled`
ultimately depends on `baa_provider != "none"` AND the AWS creds being
present, but that wiring lands in Phase 2. The fields exist now so
.env files written today don't need to migrate later.
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel


class HealthConfig(BaseModel):
    # Master switch. False means the module's behavior is fully bypassed —
    # triage's PHI extension is still parsed but never acted on, the
    # router always returns "skip", and no health code runs in the hot
    # path. Default off so a fresh install behaves identically to today.
    enabled: bool = False

    # Default user-facing language for retrieved evidence and disclaimers.
    # Currently informational; not enforced by retrieval sources in Phase 1.
    default_language: str = "en"

    # --- Retrieval source toggles ---
    # In Phase 1 only PubMed and MedlinePlus have implementations. The
    # other toggles exist for forward compat — flipping them on without
    # a registered source is a no-op (logged at debug level).
    retrieval_pubmed: bool = True
    retrieval_openfda: bool = True
    retrieval_medlineplus: bool = True
    retrieval_cdc: bool = True
    retrieval_clinicaltrials: bool = True
    retrieval_timeout_seconds: float = 5.0
    retrieval_cache_ttl_hours: int = 24

    # Optional NCBI API key — raises the PubMed E-utilities rate limit
    # from 3 req/s anonymous to 10 req/s. Personal-use volume is well
    # under the anonymous cap; only set if you start hitting limits.
    ncbi_api_key: str = ""

    # --- BAA path (Phase 2; declared now so config schemas don't churn) ---
    # "bedrock" wires through AWS Bedrock; "vertex" reserved for GCP
    # Vertex AI; "none" means there is no BAA path and personal queries
    # downgrade to a safety message asking the user to rephrase or
    # enable a BAA provider.
    baa_provider: Literal["bedrock", "vertex", "none"] = "none"
    baa_model_main: str = ""
    baa_model_escalation: str = ""

    # --- Audit ---
    # Default 7 years aligns with HIPAA's record-retention defaults; a
    # personal-use install can shorten this freely.
    audit_retention_days: int = 2555

    # Where the PHI encryption key lives. "keychain" = OS keychain (macOS
    # Keychain on darwin, libsecret on Linux); "env" = inline env var
    # (development convenience, not for real use); "kms" = AWS KMS.
    # Phase 2 wiring; Phase 1 doesn't touch encrypted storage.
    phi_encryption_key_source: Literal["keychain", "env", "kms"] = "keychain"

    @property
    def phi_path_available(self) -> bool:
        """True when the BAA path is configured well enough to use.

        The router consults this when deciding whether to route a
        personal query to the BAA LLM or fall back to a safety message.
        Phase 1 always returns False since no BAA client is wired yet.
        """
        return self.enabled and self.baa_provider != "none" and bool(self.baa_model_main)

    @property
    def any_retrieval_source_enabled(self) -> bool:
        return any([
            self.retrieval_pubmed,
            self.retrieval_openfda,
            self.retrieval_medlineplus,
            self.retrieval_cdc,
            self.retrieval_clinicaltrials,
        ])


def _bool_env(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def _str_env(key: str, default: str) -> str:
    """Env getter that treats empty strings as missing.

    `os.getenv("X", "default")` returns the literal `""` when the var
    is set-but-empty (a common shape in commented-out .env files).
    Numeric coercion downstream then crashes — handle here once.
    """
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return raw


def load_health_config() -> HealthConfig:
    """Build HealthConfig from environment variables.

    Mirrors the top-level `load_config()` pattern — same dotenv flow
    is assumed to have already populated `os.environ`. Safe to call
    standalone (e.g., from the `health` CLI subcommand).
    """
    return HealthConfig(
        enabled=_bool_env("HEALTH_MODULE_ENABLED", False),
        default_language=_str_env("HEALTH_DEFAULT_LANGUAGE", "en"),
        retrieval_pubmed=_bool_env("HEALTH_RETRIEVAL_PUBMED", True),
        retrieval_openfda=_bool_env("HEALTH_RETRIEVAL_OPENFDA", True),
        retrieval_medlineplus=_bool_env("HEALTH_RETRIEVAL_MEDLINEPLUS", True),
        retrieval_cdc=_bool_env("HEALTH_RETRIEVAL_CDC", True),
        retrieval_clinicaltrials=_bool_env("HEALTH_RETRIEVAL_CLINICALTRIALS", True),
        retrieval_timeout_seconds=float(
            _str_env("HEALTH_RETRIEVAL_TIMEOUT_SECONDS", "5.0")
        ),
        retrieval_cache_ttl_hours=int(
            _str_env("HEALTH_RETRIEVAL_CACHE_TTL_HOURS", "24")
        ),
        ncbi_api_key=_str_env("NCBI_API_KEY", ""),
        baa_provider=_baa_provider_env(),
        baa_model_main=_str_env("HEALTH_BAA_MODEL_MAIN", ""),
        baa_model_escalation=_str_env("HEALTH_BAA_MODEL_ESCALATION", ""),
        audit_retention_days=int(
            _str_env("HEALTH_AUDIT_RETENTION_DAYS", "2555")
        ),
        phi_encryption_key_source=_key_source_env(),
    )


def _baa_provider_env() -> Literal["bedrock", "vertex", "none"]:
    raw = (os.getenv("HEALTH_BAA_PROVIDER", "none") or "none").lower()
    if raw in ("bedrock", "vertex", "none"):
        return raw  # type: ignore[return-value]
    return "none"


def _key_source_env() -> Literal["keychain", "env", "kms"]:
    raw = (os.getenv("PHI_ENCRYPTION_KEY_SOURCE", "keychain") or "keychain").lower()
    if raw in ("keychain", "env", "kms"):
        return raw  # type: ignore[return-value]
    return "keychain"
