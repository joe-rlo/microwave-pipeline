"""Health audit writer tests.

Audit is the only durable trace of a health-routed turn after the
session ends — its shape and what it deliberately doesn't capture
(prompt/response content) are the load-bearing assertions here.
"""

from __future__ import annotations

import json

import pytest

from src.health.audit import HealthAuditRow, HealthAuditWriter


@pytest.fixture
def writer(tmp_path):
    w = HealthAuditWriter(tmp_path / "audit.db")
    w.connect()
    yield w
    w.close()


class TestHealthAuditWriter:
    def test_write_minimal_row(self, writer):
        writer.write(HealthAuditRow(route="general"))
        rows = writer.list_recent()
        assert len(rows) == 1
        assert rows[0]["route"] == "general"
        assert rows[0]["timestamp"] > 0

    def test_full_row_roundtrip(self, writer):
        writer.write(
            HealthAuditRow(
                route="general",
                triage_phi_class="general",
                triage_health_topic="diabetes",
                sources_queried=["pubmed", "medlineplus"],
                sources_returned=[
                    {"name": "pubmed", "count": 3},
                    {"name": "medlineplus", "count": 1},
                ],
                llm_provider="anthropic",
                llm_model="sonnet",
                latency_ms=842,
            ),
            ts=1714780000,
        )
        rows = writer.list_recent()
        assert rows[0]["timestamp"] == 1714780000
        assert rows[0]["triage_health_topic"] == "diabetes"
        assert rows[0]["llm_provider"] == "anthropic"
        assert rows[0]["latency_ms"] == 842
        # JSON-encoded list survives roundtrip
        assert json.loads(rows[0]["sources_queried"]) == ["pubmed", "medlineplus"]
        returned = json.loads(rows[0]["sources_returned"])
        assert {"name": "pubmed", "count": 3} in returned

    def test_decline_row_has_null_llm(self, writer):
        """decline_phi turns don't make an LLM call — the provider/model
        columns must be NULL, not echoed from defaults."""
        writer.write(HealthAuditRow(
            route="decline_phi",
            triage_phi_class="personal",
            latency_ms=12,
        ))
        rows = writer.list_recent()
        assert rows[0]["route"] == "decline_phi"
        assert rows[0]["llm_provider"] is None
        assert rows[0]["llm_model"] is None

    def test_no_prompt_or_response_columns(self, writer):
        """Spec is explicit: never store prompt or response content.
        Lock that contract by verifying the schema columns."""
        writer.write(HealthAuditRow(route="general"))
        cols = list(writer.conn.execute("PRAGMA table_info(health_audit)"))
        names = {c["name"] for c in cols}
        assert "prompt" not in names
        assert "response" not in names
        assert "message" not in names
        assert "content" not in names

    def test_recent_returns_newest_first(self, writer):
        writer.write(HealthAuditRow(route="general"), ts=100)
        writer.write(HealthAuditRow(route="phi"), ts=200)
        writer.write(HealthAuditRow(route="general"), ts=150)
        rows = writer.list_recent()
        assert [r["timestamp"] for r in rows] == [200, 150, 100]

    def test_recent_respects_limit(self, writer):
        for i in range(20):
            writer.write(HealthAuditRow(route="general"), ts=i)
        assert len(writer.list_recent(limit=5)) == 5

    def test_write_without_connect_logs_and_drops(self, tmp_path, caplog):
        """Pipeline robustness: an audit-DB problem must not break the
        turn. Writer drops silently with a warning."""
        unconnected = HealthAuditWriter(tmp_path / "doesnt-matter.db")
        # Don't call connect()
        with caplog.at_level("WARNING"):
            unconnected.write(HealthAuditRow(route="general"))
        # No exception raised; warning logged
        assert any("not connected" in r.message for r in caplog.records)

    def test_empty_lists_stored_as_null_not_empty_string(self, writer):
        """Spec uses JSON arrays; empty arrays should write as NULL so
        downstream queries can `WHERE sources_queried IS NOT NULL`
        without false positives on `'[]'`."""
        writer.write(HealthAuditRow(route="general"))
        rows = writer.list_recent()
        assert rows[0]["sources_queried"] is None
        assert rows[0]["sources_returned"] is None
