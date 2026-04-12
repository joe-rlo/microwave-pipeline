"""Stage 1: Triage — intent classification and search parameter routing.

Uses Haiku for fast classification (~200ms).
Dynamically configures Stage 2 search parameters based on intent.
"""

from __future__ import annotations

import logging

from src.llm.client import SingleTurnClient
from src.session.models import TriageResult, Turn

log = logging.getLogger(__name__)

TRIAGE_PROMPT = """\
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
  }
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


def _parse_triage_response(response: str) -> TriageResult:
    """Parse JSON response from triage model, with fallback defaults."""
    from src.pipeline.json_utils import extract_json

    data = extract_json(response)
    if data is None:
        log.warning(f"Triage parse failed, using defaults (response: {response[:100]!r})")
        return _default_triage()

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
    )


async def triage(
    message: str,
    recent_turns: list[Turn],
    model: str = "haiku",
    auth_mode: str = "max",
    api_key: str = "",
    cli_path: str = "",
) -> TriageResult:
    """Run triage classification on a user message."""
    client = SingleTurnClient(model=model, auth_mode=auth_mode, api_key=api_key, cli_path=cli_path)
    input_text = _format_triage_input(message, recent_turns)

    try:
        response = await client.query(TRIAGE_PROMPT, input_text)
        result = _parse_triage_response(response)
        log.info(f"Triage: intent={result.intent}, complexity={result.complexity}, "
                 f"needs_memory={result.needs_memory}")
        return result
    except Exception as e:
        log.error(f"Triage failed: {e}")
        return _parse_triage_response("")  # returns defaults
