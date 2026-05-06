"""Query-rewrite tests — fallback paths + output cleaning.

We don't drive a real Haiku call; we mock the SingleTurnClient at
the module boundary and verify (a) successful rewrite passes through
cleanly, (b) every failure mode falls back to the original message
rather than dropping the turn.
"""

from __future__ import annotations

from unittest.mock import patch, AsyncMock

import pytest

from src.health.retrieval.query_rewrite import (
    _clean_output,
    rewrite_query,
)


class TestCleanOutput:
    def test_passes_through_clean_query(self):
        assert _clean_output("antihypertensives gastroesophageal reflux") == (
            "antihypertensives gastroesophageal reflux"
        )

    def test_strips_surrounding_whitespace(self):
        assert _clean_output("  metformin pharmacology  ") == "metformin pharmacology"

    def test_strips_wrapping_double_quotes(self):
        assert _clean_output('"statins side effects"') == "statins side effects"

    def test_strips_wrapping_single_quotes(self):
        assert _clean_output("'statins side effects'") == "statins side effects"

    def test_strips_wrapping_backticks(self):
        assert _clean_output("`statins side effects`") == "statins side effects"

    def test_strips_label_prefixes(self):
        """Model sometimes ignores 'no explanation' instruction and
        prepends 'Query:' or similar — strip those."""
        assert _clean_output("Query: metformin lactic acidosis") == (
            "metformin lactic acidosis"
        )
        assert _clean_output("Output: x y z") == "x y z"
        assert _clean_output("REWRITTEN: x y") == "x y"

    def test_drops_trailing_period(self):
        # Keyword queries don't end with sentence punctuation
        assert _clean_output("metformin side effects.") == "metformin side effects"

    def test_empty_returns_empty(self):
        assert _clean_output("") == ""
        assert _clean_output(None) == ""  # type: ignore[arg-type]


class _FakeClient:
    """Stand-in for SingleTurnClient — captures the prompt and returns
    a scripted response or raises."""

    def __init__(self, response):
        self.response = response
        self.calls: list[tuple[str, str]] = []

    async def query(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _patch_client(response):
    return patch(
        "src.health.retrieval.query_rewrite.SingleTurnClient",
        return_value=_FakeClient(response),
    )


class TestRewriteQuery:
    @pytest.mark.asyncio
    async def test_empty_message_passes_through(self):
        """Don't burn a Haiku call on no input."""
        with patch(
            "src.health.retrieval.query_rewrite.SingleTurnClient"
        ) as mk:
            result = await rewrite_query("")
        assert result == ""
        mk.assert_not_called()

    @pytest.mark.asyncio
    async def test_successful_rewrite(self):
        with _patch_client("antihypertensives statins gastroesophageal reflux"):
            result = await rewrite_query(
                "Do blood pressure medications or statins sometimes cause heartburn?"
            )
        assert result == "antihypertensives statins gastroesophageal reflux"

    @pytest.mark.asyncio
    async def test_topic_threaded_into_user_input(self):
        """The triage health_topic gets passed through so the model can
        bias its rewrite toward the right vocabulary."""
        fake = _FakeClient("metformin gi side effects")
        with patch(
            "src.health.retrieval.query_rewrite.SingleTurnClient",
            return_value=fake,
        ):
            await rewrite_query(
                "Does metformin cause stomach issues?",
                topic="diabetes",
            )
        assert len(fake.calls) == 1
        _, user_text = fake.calls[0]
        assert "Topic tag: diabetes" in user_text
        assert "Does metformin cause stomach issues?" in user_text

    @pytest.mark.asyncio
    async def test_no_topic_no_topic_label(self):
        fake = _FakeClient("metformin")
        with patch(
            "src.health.retrieval.query_rewrite.SingleTurnClient",
            return_value=fake,
        ):
            await rewrite_query("metformin?")
        _, user_text = fake.calls[0]
        assert "Topic tag" not in user_text

    @pytest.mark.asyncio
    async def test_call_exception_falls_back_to_original(self):
        """Network / API failure must not drop the turn."""
        original = "Do statins cause heartburn?"
        with _patch_client(RuntimeError("network down")):
            result = await rewrite_query(original)
        assert result == original

    @pytest.mark.asyncio
    async def test_empty_response_falls_back(self):
        original = "Do statins cause heartburn?"
        with _patch_client("   \n  "):  # whitespace only
            result = await rewrite_query(original)
        assert result == original

    @pytest.mark.asyncio
    async def test_excessively_long_response_falls_back(self):
        """If the model misunderstands and returns a paragraph, fall
        back rather than blow the search-API URL length budget."""
        original = "Does Cialis interact with grapefruit?"
        long_response = "tadalafil " * 50  # ~500 chars
        with _patch_client(long_response):
            result = await rewrite_query(original)
        assert result == original

    @pytest.mark.asyncio
    async def test_quoted_response_unwrapped(self):
        """Common model failure mode: wraps the answer in quotes
        despite 'no quotes' instruction. We strip them."""
        with _patch_client('"metformin side effects"'):
            result = await rewrite_query("does metformin have side effects?")
        assert result == "metformin side effects"

    @pytest.mark.asyncio
    async def test_label_prefixed_response_cleaned(self):
        with _patch_client("Query: statins adverse effects"):
            result = await rewrite_query("statins side effects?")
        assert result == "statins adverse effects"
