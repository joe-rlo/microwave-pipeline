"""Tests for Stage 1: Triage."""

import pytest
from src.pipeline.triage import _parse_triage_response


class TestTriageParsing:
    def test_valid_json(self):
        raw = '{"intent": "recall", "complexity": "simple", "needs_memory": true, "search_params": {"decay_half_life": 7, "result_count": 5, "weight_recency": 0.8, "mmr_lambda": 0.7}}'
        result = _parse_triage_response(raw)
        assert result.intent == "recall"
        assert result.complexity == "simple"
        assert result.needs_memory is True
        assert result.search_params["decay_half_life"] == 7

    def test_json_with_code_fence(self):
        raw = '```json\n{"intent": "social", "complexity": "simple", "needs_memory": false, "search_params": {}}\n```'
        result = _parse_triage_response(raw)
        assert result.intent == "social"
        assert result.needs_memory is False

    def test_invalid_json_returns_defaults(self):
        result = _parse_triage_response("not json at all")
        assert result.intent == "question"
        assert result.complexity == "moderate"
        assert result.needs_memory is True

    def test_empty_string_returns_defaults(self):
        result = _parse_triage_response("")
        assert result.intent == "question"
