"""Tests for memory contradiction surfacing.

Uses a scripted SingleTurnClient stand-in so the tests don't need a
real LLM call; we're verifying the detection wiring (parse, project
to Contradiction, format), not the model's judgment. Real behavior
needs to be sanity-checked manually against a real MEMORY.md once.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.memory.health import (
    Contradiction,
    detect_contradictions,
    format_for_cli,
)


class _ScriptedSingleTurn:
    """SingleTurnClient stand-in. Records calls and returns the canned
    response in order, raising on exhausted/error entries to mirror real
    failure shapes."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def query(self, system, user):
        self.calls.append((system, user))
        if not self.responses:
            raise AssertionError("Scripted client ran out of responses")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _patch_client(scripted):
    """Patch SingleTurnClient inside health.py so detect_contradictions
    uses the scripted instance regardless of constructor args."""
    return patch(
        "src.memory.health.SingleTurnClient",
        return_value=scripted,
    )


class TestDetectContradictions:
    @pytest.mark.asyncio
    async def test_empty_memory_returns_empty(self):
        """Don't even call the LLM for an empty doc — would just waste tokens."""
        scripted = _ScriptedSingleTurn([])
        with _patch_client(scripted):
            result = await detect_contradictions("")
        assert result == []
        assert len(scripted.calls) == 0

    @pytest.mark.asyncio
    async def test_no_contradictions_parsed_as_empty_list(self):
        scripted = _ScriptedSingleTurn(['{"contradictions": []}'])
        with _patch_client(scripted):
            result = await detect_contradictions("- Lives in Berlin")
        assert result == []
        assert len(scripted.calls) == 1

    @pytest.mark.asyncio
    async def test_one_contradiction(self):
        scripted = _ScriptedSingleTurn([
            '{"contradictions": [{"a": "Dog is Biscuit", '
            '"b": "Dog is Max", "summary": "Dog name conflict"}]}'
        ])
        with _patch_client(scripted):
            result = await detect_contradictions(
                "- Dog is Biscuit\n- Dog is Max"
            )
        assert len(result) == 1
        assert result[0].a == "Dog is Biscuit"
        assert result[0].b == "Dog is Max"
        assert result[0].summary == "Dog name conflict"

    @pytest.mark.asyncio
    async def test_multiple_contradictions(self):
        scripted = _ScriptedSingleTurn([
            '{"contradictions": ['
            '{"a": "Dog A", "b": "Dog B", "summary": "S1"},'
            '{"a": "Cat A", "b": "Cat B", "summary": "S2"}'
            ']}'
        ])
        with _patch_client(scripted):
            result = await detect_contradictions("anything")
        assert len(result) == 2
        assert {c.summary for c in result} == {"S1", "S2"}

    @pytest.mark.asyncio
    async def test_skips_items_missing_quotes(self):
        """Defensive: model returned a malformed entry (no `a`)."""
        scripted = _ScriptedSingleTurn([
            '{"contradictions": ['
            '{"a": "", "b": "x", "summary": "broken"},'
            '{"a": "real-a", "b": "real-b", "summary": "ok"}'
            ']}'
        ])
        with _patch_client(scripted):
            result = await detect_contradictions("anything")
        assert len(result) == 1
        assert result[0].a == "real-a"

    @pytest.mark.asyncio
    async def test_invalid_top_level_returns_empty(self):
        """Model returned JSON but not the expected shape."""
        scripted = _ScriptedSingleTurn(['{"contradictions": "not a list"}'])
        with _patch_client(scripted):
            result = await detect_contradictions("anything")
        assert result == []

    @pytest.mark.asyncio
    async def test_parse_failure_then_retry_recovers(self):
        """Verifies the helper from item #2 is in the loop — first
        response is unparseable, retry returns valid JSON."""
        scripted = _ScriptedSingleTurn([
            "I think there is one contradiction",  # not JSON
            '{"contradictions": [{"a": "x", "b": "y", "summary": "z"}]}',
        ])
        with _patch_client(scripted):
            result = await detect_contradictions("anything")
        assert len(result) == 1
        assert len(scripted.calls) == 2  # one retry happened

    @pytest.mark.asyncio
    async def test_two_failures_returns_empty(self):
        """User-friendly behavior on irrecoverable failure: empty list,
        not an exception."""
        scripted = _ScriptedSingleTurn([
            "still not json",
            "still not json on retry either",
        ])
        with _patch_client(scripted):
            result = await detect_contradictions("anything")
        assert result == []


class TestFormatForCli:
    def test_no_contradictions(self):
        out = format_for_cli([])
        assert "No contradictions" in out

    def test_one_contradiction_format(self):
        out = format_for_cli([
            Contradiction(a="A line", b="B line", summary="S"),
        ])
        assert "1 contradiction in MEMORY.md" in out
        assert "1. S" in out
        assert "A line" in out
        assert "B line" in out
        assert "edit workspace/MEMORY.md directly" in out

    def test_plural_count(self):
        out = format_for_cli([
            Contradiction(a="a", b="b", summary="s"),
            Contradiction(a="c", b="d", summary="t"),
        ])
        assert "2 contradictions" in out
