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


class TestMetaIntent:
    """1.2 — `meta` intent for bot-self / conversation-meta questions.
    Extending `intent` (not adding a separate `register` axis) is the
    chosen path; gating search via `needs_memory=false` is reused from
    the social path.
    """

    def test_meta_in_prompt_enum(self):
        """The triage prompt enumerates valid intent values — `meta`
        must be listed so Haiku knows it's available."""
        prompt = _build_prompt(skills=None)
        # JSON schema line
        assert '"meta"' in prompt
        # Definition section
        assert '- "meta"' in prompt

    def test_meta_definition_distinguishes_from_recall_and_question(self):
        """The risk on this classification is `meta` collapsing into
        `recall` ("did we talk about X?") or `question` ("what's the
        capital of France?"). The prompt must spell out the boundary."""
        prompt = _build_prompt(skills=None)
        meta_section = prompt[prompt.find('- "meta"'):]
        # Must contrast with recall AND question, or the model will drift
        assert "recall" in meta_section.lower()
        assert "question" in meta_section.lower()

    def test_meta_skips_search(self):
        """Per the spec acceptance: meta-classified turns set
        needs_memory=false so Stage 2 short-circuits."""
        prompt = _build_prompt(skills=None)
        # Search-params guideline must include the meta skip
        guideline_section = prompt[prompt.find("Search parameter guidelines"):]
        assert "meta" in guideline_section.lower()
        assert "needs_memory=false" in guideline_section

    def test_meta_parses_through(self):
        """A response with intent=meta and needs_memory=false should
        round-trip — no validation collapses it to a default."""
        raw = (
            '{"intent": "meta", "complexity": "simple", '
            '"needs_memory": false, "search_params": {}}'
        )
        result = _parse_triage_response(raw)
        assert result.intent == "meta"
        assert result.needs_memory is False

    def test_schema_hint_advertises_meta(self):
        """The retry schema hint is what the model sees on the JSON
        recovery turn — `meta` must be in the enum there too, otherwise
        the retry path would silently push the model back toward the
        old five values."""
        from src.pipeline.triage import (
            _TRIAGE_SCHEMA_HINT,
            _TRIAGE_SCHEMA_HINT_HEALTH,
        )
        assert "meta" in _TRIAGE_SCHEMA_HINT
        assert "meta" in _TRIAGE_SCHEMA_HINT_HEALTH


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


class TestHealthClassification:
    """Health module extension to triage. The PHI fields are opt-in via
    health_enabled — the prompt and parser must keep their pre-health
    shape and behavior when the flag is off."""

    def test_health_disabled_prompt_omits_block(self):
        prompt = _build_prompt(skills=None, health_enabled=False)
        assert "phi_class" not in prompt
        assert "Health classification" not in prompt

    def test_health_enabled_prompt_includes_block(self):
        prompt = _build_prompt(skills=None, health_enabled=True)
        assert "phi_class" in prompt
        assert "health_topic" in prompt
        assert "personal" in prompt and "general" in prompt
        # First-person-marker rule is the load-bearing change vs. the
        # original "default to personal in doubt" bias — pin it.
        assert "first-person marker" in prompt or "first-person reference" in prompt

    def test_health_prompt_has_meta_question_carveout(self):
        """Meta questions about the bot itself ('did you find anything?',
        'no results from the pipeline?') must classify as 'none' — not
        as personal just because they came after a health turn."""
        prompt = _build_prompt(skills=None, health_enabled=True)
        # Look for the explicit "meta-questions" carve-out language
        assert "meta-questions" in prompt.lower() or "meta question" in prompt.lower()
        # And at least one example phrasing the model can pattern-match
        assert "no results" in prompt.lower() or "did you find" in prompt.lower()

    def test_health_prompt_carries_topic_over_followups(self):
        """Brief follow-ups without first-person markers should inherit
        the general classification from the previous turn — locks the
        rule against the screenshot scenario where 'what about ED
        medications?' got mis-classified."""
        prompt = _build_prompt(skills=None, health_enabled=True)
        assert "carryover" in prompt.lower() or "follow-up" in prompt.lower()
        # The clarifying example phrasings the prompt uses
        assert "what about" in prompt.lower() or "and statins" in prompt.lower()

    def test_health_prompt_distinguishes_topic_from_marker(self):
        """A sensitive TOPIC alone (ED, cancer, mental health) must NOT
        trigger 'personal' without a first-person marker. The prompt
        spells this out so Haiku doesn't keyword-match its way into
        false positives."""
        prompt = _build_prompt(skills=None, health_enabled=True)
        # Look for the disambiguating contrast pair the prompt uses
        assert "What does Cialis do" in prompt
        assert "safe for me" in prompt

    def test_health_enabled_with_skills(self):
        """Both extensions should compose without trampling each other."""
        prompt = _build_prompt(
            skills=[("blog-writing", "Long-form posts.")],
            health_enabled=True,
        )
        assert "matched_skill" in prompt
        assert "phi_class" in prompt
        assert "blog-writing" in prompt

    def test_active_project_adds_fictional_carveout(self):
        prompt = _build_prompt(
            skills=None, health_enabled=True, active_project="the-heist",
        )
        assert "the-heist" in prompt
        # The carveout must mention "fictional" framing so Haiku catches the
        # novel-character-with-diabetes case the spec calls out
        assert "fictional" in prompt.lower()

    def test_no_active_project_no_fiction_note(self):
        prompt = _build_prompt(skills=None, health_enabled=True, active_project=None)
        # Nothing about fictional content when no project is active
        assert "fictional" not in prompt.lower()

    def test_parse_phi_class_default_is_none(self):
        """Legacy responses without phi_class get the safe default."""
        raw = '{"intent": "question", "complexity": "moderate"}'
        result = _parse_triage_response(raw)
        assert result.phi_class == "none"
        assert result.health_topic is None

    def test_parse_phi_class_present(self):
        raw = (
            '{"intent": "question", "complexity": "moderate", '
            '"phi_class": "personal", "health_topic": "diabetes"}'
        )
        result = _parse_triage_response(raw)
        assert result.phi_class == "personal"
        assert result.health_topic == "diabetes"

    def test_unknown_phi_class_value_falls_to_none(self):
        """Defensive: if Haiku returns 'phi' or some other unmapped string,
        we don't surface it — defaults to 'none' rather than letting the
        router mis-route."""
        raw = (
            '{"intent": "question", "complexity": "moderate", '
            '"phi_class": "garbage"}'
        )
        result = _parse_triage_response(raw)
        assert result.phi_class == "none"

    def test_phi_class_case_normalized(self):
        raw = (
            '{"intent": "question", "complexity": "moderate", '
            '"phi_class": "PERSONAL"}'
        )
        result = _parse_triage_response(raw)
        assert result.phi_class == "personal"

    def test_health_topic_blank_becomes_none(self):
        raw = (
            '{"intent": "question", "complexity": "moderate", '
            '"phi_class": "general", "health_topic": "   "}'
        )
        result = _parse_triage_response(raw)
        assert result.health_topic is None
