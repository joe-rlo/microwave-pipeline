"""Tests for JSON extraction from LLM responses."""

from src.pipeline.json_utils import extract_json


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
