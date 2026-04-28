"""Stage 1: Triage — intent classification and search parameter routing.

Uses Haiku for fast classification (~200ms).
Dynamically configures Stage 2 search parameters based on intent.
"""

from __future__ import annotations

import logging

from src.llm.client import SingleTurnClient
from src.session.models import TriageResult, Turn

log = logging.getLogger(__name__)

TRIAGE_PROMPT_BASE = """\
You are a triage classifier for a personal AI assistant's memory system.
Given a user message and recent conversation context, classify the intent and output search parameters.

Respond with ONLY valid JSON, no other text:

{
  "intent": "recall" | "preference" | "task" | "question" | "social",
  "complexity": "simple" | "moderate" | "complex",
  "needs_memory": true | false,
  "search_params": {
    "decay_half_life": <float, days — short for recent recall, long for preferences>,
    "result_count": <int, 3-10>,
    "weight_recency": <float, 0.0-1.0 — high for recall, low for preferences>,
    "mmr_lambda": <float, 0.5-0.9 — diversity vs relevance tradeoff>
  },
  "matched_skill": <string or null>
}

Intent definitions:
- "recall": user is asking about something previously discussed or a past event
- "preference": user is asking about their own preferences, habits, or opinions
- "task": user wants help doing something (writing, coding, planning)
- "question": user is asking a factual or analytical question
- "social": greeting, small talk, emotional exchange

Complexity definitions:
- "simple": greetings, yes/no questions, quick factual lookups, one-line answers
- "moderate": standard questions, summaries, short writing tasks, most everyday requests
- "complex": multi-step reasoning, deep analysis, architecture design, debugging tricky problems, long-form writing, math proofs, anything requiring careful step-by-step thought

Reserve "complex" for tasks that genuinely need deep thinking. Most messages are "moderate".

Search parameter guidelines:
- recall: short decay (7-14 days), high recency (0.7-0.9), moderate results (5)
- preference: long decay (90-180 days), low recency (0.1-0.3), more results (7)
- task: moderate decay (30 days), moderate recency (0.5), fewer results (3)
- question: moderate decay (30 days), moderate recency (0.5), moderate results (5)
- social: set needs_memory=false unless referencing something specific\
"""

_SKILLS_PROMPT_TEMPLATE = """\

Skills available — reusable instruction bundles the assistant can activate for
specialized tasks. If (and only if) the current message clearly calls for one
of them, set `matched_skill` to that skill's exact name. Otherwise set it to
null. Strongly prefer null over a weak or uncertain match — a wrong skill is
worse than no skill.

Guidelines for matching:
- The user's CURRENT intent must match the skill's purpose, not merely touch
  the topic. Asking "what's a Substack Note?" is a question (null), not a
  substack-writer job.
- Recent conversation context can clarify intent — if the last few turns are
  about drafting a blog post, a message like "cut the third paragraph" is
  still blog-writing.
- Skills are opt-in for generative work. Never match a skill for greetings,
  small talk, debugging the bot itself, or meta-questions about the system.

Available skills:
{catalog}
"""


def _build_prompt(skills: list[tuple[str, str]] | None) -> str:
    """Compose the triage prompt. Skill catalog is appended only when
    skills are available so a zero-skill install keeps the original prompt
    shape (and token count)."""
    if not skills:
        # Keep behavior identical to pre-skill triage — same prompt text.
        # Return the base unmodified so unrelated tests stay stable.
        return TRIAGE_PROMPT_BASE.replace(
            ',\n  "matched_skill": <string or null>', ""
        )
    catalog_lines = [f"- {name}: {desc or '(no description)'}" for name, desc in skills]
    return TRIAGE_PROMPT_BASE + _SKILLS_PROMPT_TEMPLATE.format(
        catalog="\n".join(catalog_lines)
    )


def _format_triage_input(message: str, recent_turns: list[Turn]) -> str:
    parts = []
    if recent_turns:
        parts.append("Recent conversation:")
        for turn in recent_turns[-4:]:  # Last 4 turns for context
            parts.append(f"  {turn.role}: {turn.content[:200]}")
        parts.append("")
    parts.append(f"Current message: {message}")
    return "\n".join(parts)


def _default_triage() -> TriageResult:
    return TriageResult(
        intent="question",
        complexity="moderate",
        search_params={
            "decay_half_life": 30.0,
            "result_count": 5,
            "weight_recency": 0.5,
            "mmr_lambda": 0.7,
        },
        needs_memory=True,
    )


def _parse_triage_response(
    response: str, available_skills: set[str] | None = None
) -> TriageResult:
    """Parse JSON response from triage model, with fallback defaults.

    `available_skills` is used to validate matched_skill — if Haiku
    returns a name that doesn't match any real skill (hallucination or
    case mismatch), we drop it to None rather than trust it.
    """
    from src.pipeline.json_utils import extract_json

    data = extract_json(response)
    if data is None:
        log.warning(f"Triage parse failed, using defaults (response: {response[:100]!r})")
        return _default_triage()

    matched = data.get("matched_skill")
    if matched is not None:
        matched = str(matched).strip() or None
    if matched and available_skills is not None and matched not in available_skills:
        log.info(f"Triage returned unknown skill {matched!r}; discarding")
        matched = None

    return TriageResult(
        intent=data.get("intent", "question"),
        complexity=data.get("complexity", "moderate"),
        search_params=data.get("search_params", {
            "decay_half_life": 30.0,
            "result_count": 5,
            "weight_recency": 0.5,
            "mmr_lambda": 0.7,
        }),
        needs_memory=data.get("needs_memory", True),
        matched_skill=matched,
    )


async def triage(
    message: str,
    recent_turns: list[Turn],
    model: str = "haiku",
    auth_mode: str = "max",
    api_key: str = "",
    cli_path: str = "",
    skills: list[tuple[str, str]] | None = None,
) -> TriageResult:
    """Run triage classification on a user message.

    `skills` is an optional catalog of (name, description) tuples.
    When provided, the prompt asks Haiku to also return a matched_skill
    (or null) — enables the adaptive-skill activation feature without
    any second model call.
    """
    client = SingleTurnClient(model=model, auth_mode=auth_mode, api_key=api_key, cli_path=cli_path)
    input_text = _format_triage_input(message, recent_turns)
    prompt = _build_prompt(skills)
    available = {name for name, _ in skills} if skills else None

    try:
        response = await client.query(prompt, input_text)
        result = _parse_triage_response(response, available_skills=available)
        skill_note = f", skill={result.matched_skill!r}" if result.matched_skill else ""
        log.info(
            f"Triage: intent={result.intent}, complexity={result.complexity}, "
            f"needs_memory={result.needs_memory}{skill_note}"
        )
        return result
    except Exception as e:
        log.error(f"Triage failed: {e}")
        return _parse_triage_response("", available_skills=available)  # returns defaults
