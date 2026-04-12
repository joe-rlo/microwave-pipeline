"""Tests for Stage 4: Reflection."""

import pytest
from src.pipeline.reflection import _parse_reflection_response


class TestReflectionParsing:
    def test_valid_deliver(self):
        raw = '{"confidence": 0.95, "hedging_detected": false, "action": "deliver", "memory_gap": null}'
        result = _parse_reflection_response(raw, "test response")
        assert result.action == "deliver"
        assert result.confidence == 0.95
        assert result.response == "test response"

    def test_re_search_with_gap(self):
        raw = '{"confidence": 0.3, "hedging_detected": true, "action": "re-search", "memory_gap": "user birthday"}'
        result = _parse_reflection_response(raw, "I think maybe...")
        assert result.action == "re-search"
        assert result.memory_gap == "user birthday"
        assert result.hedging_detected is True

    def test_invalid_json_defaults_to_deliver(self):
        result = _parse_reflection_response("broken", "some response")
        assert result.action == "deliver"
        assert result.response == "some response"
