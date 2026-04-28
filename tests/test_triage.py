"""Tests for Stage 1: Triage."""

import pytest
from src.pipeline.triage import (
    TRIAGE_PROMPT_BASE,
    _build_prompt,
    _parse_triage_response,
)


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

    def test_matched_skill_defaults_to_none(self):
        # Legacy responses without the field parse to matched_skill=None,
        # not a KeyError.
        raw = '{"intent": "question", "complexity": "moderate"}'
        result = _parse_triage_response(raw)
        assert result.matched_skill is None


class TestMatchedSkillParsing:
    def test_valid_skill_preserved(self):
        raw = (
            '{"intent": "task", "complexity": "moderate", '
            '"matched_skill": "substack-writer"}'
        )
        result = _parse_triage_response(
            raw, available_skills={"substack-writer", "blog-writing"}
        )
        assert result.matched_skill == "substack-writer"

    def test_unknown_skill_discarded(self):
        # Haiku might hallucinate skill names — anything not in the
        # allowlist must be dropped rather than trusted.
        raw = (
            '{"intent": "task", "complexity": "moderate", '
            '"matched_skill": "writing-tool-that-does-not-exist"}'
        )
        result = _parse_triage_response(
            raw, available_skills={"substack-writer"}
        )
        assert result.matched_skill is None

    def test_no_allowlist_means_no_validation(self):
        # When the caller doesn't provide a catalog (tests or legacy code),
        # we keep whatever Haiku returned — the orchestrator does a second
        # check when it tries to load the skill anyway.
        raw = '{"intent": "task", "complexity": "moderate", "matched_skill": "foo"}'
        result = _parse_triage_response(raw, available_skills=None)
        assert result.matched_skill == "foo"

    def test_null_stays_null(self):
        raw = (
            '{"intent": "social", "complexity": "simple", '
            '"matched_skill": null}'
        )
        result = _parse_triage_response(raw, available_skills={"x"})
        assert result.matched_skill is None

    def test_empty_string_becomes_null(self):
        raw = (
            '{"intent": "social", "complexity": "simple", '
            '"matched_skill": "   "}'
        )
        result = _parse_triage_response(raw, available_skills={"x"})
        assert result.matched_skill is None


class TestPromptBuilding:
    def test_no_skills_matches_legacy_shape(self):
        # The zero-skill install should get the original prompt text —
        # no `matched_skill` field mentioned. Keeps token counts stable
        # for users who aren't using skills.
        prompt = _build_prompt(skills=None)
        assert "matched_skill" not in prompt
        assert "Available skills:" not in prompt

    def test_empty_list_behaves_like_none(self):
        assert _build_prompt(skills=[]) == _build_prompt(skills=None)

    def test_catalog_injected(self):
        prompt = _build_prompt(skills=[
            ("substack-writer", "Short notes in Joe's voice."),
            ("blog-writing", "Long-form posts."),
        ])
        assert "matched_skill" in prompt
        assert "Available skills:" in prompt
        assert "substack-writer: Short notes in Joe's voice." in prompt
        assert "blog-writing: Long-form posts." in prompt
        # "Prefer null" guidance is load-bearing against wrong-match risk
        assert "null" in prompt.lower()

    def test_missing_description_handled(self):
        prompt = _build_prompt(skills=[("x", "")])
        assert "x: (no description)" in prompt
