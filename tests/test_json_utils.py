"""Tests for JSON extraction from LLM responses."""

import pytest

from src.pipeline.json_utils import extract_json, query_json_with_retry


class TestExtractJson:
    def test_clean_json(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_code_fence(self):
        text = '```json\n{"a": 1}\n```'
        assert extract_json(text) == {"a": 1}

    def test_json_with_trailing_text(self):
        text = '{"confidence": 0.9, "action": "deliver"}\n\nThe response looks good.'
        result = extract_json(text)
        assert result["confidence"] == 0.9
        assert result["action"] == "deliver"

    def test_json_with_preamble(self):
        text = 'Here is my analysis:\n{"intent": "recall", "complexity": "simple"}'
        result = extract_json(text)
        assert result["intent"] == "recall"

    def test_code_fence_with_trailing_text(self):
        text = '```json\n{"a": 1}\n```\n\nSome explanation here.'
        assert extract_json(text) == {"a": 1}

    def test_nested_braces(self):
        text = '{"params": {"decay": 7, "count": 5}}'
        result = extract_json(text)
        assert result["params"]["decay"] == 7

    def test_no_json(self):
        assert extract_json("no json here") is None

    def test_empty_string(self):
        assert extract_json("") is None

    def test_string_with_braces(self):
        text = '{"msg": "hello {world}"}'
        result = extract_json(text)
        assert result["msg"] == "hello {world}"


class _ScriptedClient:
    """Test double for `SingleTurnClient.query` — returns canned responses
    in order, recording each (system, user) call so the retry prompt can
    be inspected.
    """

    def __init__(self, responses: list[str | Exception]):
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    async def query(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self.responses:
            raise AssertionError("Scripted client ran out of responses")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class TestQueryJsonWithRetry:
    @pytest.mark.asyncio
    async def test_first_try_success(self):
        client = _ScriptedClient(['{"intent": "recall"}'])
        result = await query_json_with_retry(
            client.query, "system", "user", "schema", label="t",
        )
        assert result == {"intent": "recall"}
        assert len(client.calls) == 1  # no retry

    @pytest.mark.asyncio
    async def test_retry_recovers(self):
        client = _ScriptedClient([
            "I think the answer is recall",  # not JSON
            '{"intent": "recall"}',          # valid on retry
        ])
        result = await query_json_with_retry(
            client.query, "system", "user", "schema", label="t",
        )
        assert result == {"intent": "recall"}
        assert len(client.calls) == 2
        # Retry user prompt must show the bad response back to the model
        # so it has something concrete to correct against.
        retry_user = client.calls[1][1]
        assert "Previous response" in retry_user
        assert "I think the answer is recall" in retry_user
        assert "schema" in retry_user

    @pytest.mark.asyncio
    async def test_two_failures_returns_none(self):
        client = _ScriptedClient([
            "still not json",
            "still not json on retry either",
        ])
        result = await query_json_with_retry(
            client.query, "system", "user", "schema", label="t",
        )
        assert result is None
        assert len(client.calls) == 2

    @pytest.mark.asyncio
    async def test_first_call_exception_returns_none(self):
        client = _ScriptedClient([RuntimeError("network down")])
        result = await query_json_with_retry(
            client.query, "system", "user", "schema", label="t",
        )
        # Exception on first call is treated as a hard failure — we
        # don't retry network errors, only parse errors. The caller's
        # fallback defaults will pick up.
        assert result is None
        assert len(client.calls) == 1

    @pytest.mark.asyncio
    async def test_retry_call_exception_returns_none(self):
        client = _ScriptedClient([
            "not json",
            RuntimeError("network down"),
        ])
        result = await query_json_with_retry(
            client.query, "system", "user", "schema", label="t",
        )
        assert result is None
        assert len(client.calls) == 2
