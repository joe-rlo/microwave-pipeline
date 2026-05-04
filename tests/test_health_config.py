"""Health config tests — env loading and the derived properties.

`phi_path_available` is the key property gating Phase 2 routing
decisions; getting its boolean false-positive rate to zero matters
because the router uses it to decide whether to send personal
queries down the BAA path or fall back to a safety message.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from src.health.config import HealthConfig, load_health_config


def _clean_env(**overrides):
    """Temporarily clear all HEALTH_* and a few related env vars,
    then apply `overrides`. Lets each test see a deterministic env."""
    keys_to_clear = [
        "HEALTH_MODULE_ENABLED", "HEALTH_DEFAULT_LANGUAGE",
        "HEALTH_RETRIEVAL_PUBMED", "HEALTH_RETRIEVAL_MEDLINEPLUS",
        "HEALTH_RETRIEVAL_OPENFDA", "HEALTH_RETRIEVAL_CDC",
        "HEALTH_RETRIEVAL_CLINICALTRIALS",
        "HEALTH_RETRIEVAL_TIMEOUT_SECONDS", "HEALTH_RETRIEVAL_CACHE_TTL_HOURS",
        "NCBI_API_KEY", "HEALTH_BAA_PROVIDER",
        "HEALTH_BAA_MODEL_MAIN", "HEALTH_BAA_MODEL_ESCALATION",
        "HEALTH_AUDIT_RETENTION_DAYS", "PHI_ENCRYPTION_KEY_SOURCE",
    ]
    new = {k: "" for k in keys_to_clear}
    new.update(overrides)
    return patch.dict(os.environ, new, clear=False)


class TestLoadHealthConfig:
    def test_defaults_when_env_empty(self):
        with _clean_env():
            cfg = load_health_config()
        assert cfg.enabled is False  # off by default — fresh install no-op
        assert cfg.baa_provider == "none"
        assert cfg.retrieval_pubmed is True
        assert cfg.retrieval_timeout_seconds == 5.0
        assert cfg.audit_retention_days == 2555  # 7 years

    def test_enabled_flag(self):
        with _clean_env(HEALTH_MODULE_ENABLED="true"):
            assert load_health_config().enabled is True
        with _clean_env(HEALTH_MODULE_ENABLED="1"):
            assert load_health_config().enabled is True
        with _clean_env(HEALTH_MODULE_ENABLED="no"):
            assert load_health_config().enabled is False

    def test_retrieval_toggles(self):
        with _clean_env(
            HEALTH_RETRIEVAL_PUBMED="false",
            HEALTH_RETRIEVAL_MEDLINEPLUS="true",
        ):
            cfg = load_health_config()
        assert cfg.retrieval_pubmed is False
        assert cfg.retrieval_medlineplus is True

    def test_baa_provider_normalization(self):
        with _clean_env(HEALTH_BAA_PROVIDER="bedrock"):
            assert load_health_config().baa_provider == "bedrock"
        with _clean_env(HEALTH_BAA_PROVIDER="BEDROCK"):
            # Case-insensitive — env values often arrive uppercased
            assert load_health_config().baa_provider == "bedrock"
        with _clean_env(HEALTH_BAA_PROVIDER="garbage"):
            # Unknown values fall back to "none" rather than crash
            assert load_health_config().baa_provider == "none"

    def test_key_source_normalization(self):
        with _clean_env(PHI_ENCRYPTION_KEY_SOURCE="env"):
            assert load_health_config().phi_encryption_key_source == "env"
        with _clean_env(PHI_ENCRYPTION_KEY_SOURCE="garbage"):
            assert load_health_config().phi_encryption_key_source == "keychain"

    def test_numeric_envs_parse(self):
        with _clean_env(
            HEALTH_RETRIEVAL_TIMEOUT_SECONDS="2.5",
            HEALTH_RETRIEVAL_CACHE_TTL_HOURS="48",
            HEALTH_AUDIT_RETENTION_DAYS="365",
        ):
            cfg = load_health_config()
        assert cfg.retrieval_timeout_seconds == 2.5
        assert cfg.retrieval_cache_ttl_hours == 48
        assert cfg.audit_retention_days == 365


class TestPhiPathAvailable:
    """The router calls this before routing personal queries to BAA;
    a False here means fall back to safety message."""

    def test_disabled_module(self):
        cfg = HealthConfig(enabled=False, baa_provider="bedrock", baa_model_main="x")
        assert cfg.phi_path_available is False

    def test_no_baa_provider(self):
        cfg = HealthConfig(enabled=True, baa_provider="none", baa_model_main="x")
        assert cfg.phi_path_available is False

    def test_no_model(self):
        # provider set but no model — wiring incomplete, refuse
        cfg = HealthConfig(enabled=True, baa_provider="bedrock", baa_model_main="")
        assert cfg.phi_path_available is False

    def test_fully_configured(self):
        cfg = HealthConfig(
            enabled=True, baa_provider="bedrock",
            baa_model_main="anthropic.claude-sonnet-4-20250514-v1:0",
        )
        assert cfg.phi_path_available is True


class TestAnyRetrievalSourceEnabled:
    def test_default(self):
        # Defaults all-on, so even disabled-module config has sources enabled
        assert HealthConfig().any_retrieval_source_enabled is True

    def test_all_off(self):
        cfg = HealthConfig(
            retrieval_pubmed=False, retrieval_openfda=False,
            retrieval_medlineplus=False, retrieval_cdc=False,
            retrieval_clinicaltrials=False,
        )
        assert cfg.any_retrieval_source_enabled is False

    def test_only_one_on(self):
        cfg = HealthConfig(
            retrieval_pubmed=True, retrieval_openfda=False,
            retrieval_medlineplus=False, retrieval_cdc=False,
            retrieval_clinicaltrials=False,
        )
        assert cfg.any_retrieval_source_enabled is True
