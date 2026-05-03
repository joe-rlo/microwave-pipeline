"""Stage 4: Reflection — quality gate on LLM output before delivery.

Uses Haiku for fast evaluation (~300ms).
Checks for hedging, contradictions, or insufficient context.
Caps at one retry to avoid loops.
"""

from __future__ import annotations

import logging

from src.llm.client import SingleTurnClient
from src.session.models import ReflectionResult

log = logging.getLogger(__name__)

REFLECTION_PROMPT = """\
You are a quality gate for an AI assistant's responses. Evaluate the response for quality issues.

Respond with ONLY valid JSON:

{
  "confidence": <float, 0.0-1.0>,
  "hedging_detected": <bool — did the response hedge excessively?>,
  "action": "deliver" | "re-search" | "clarify",
  "memory_gap": <string | null — what info was missing if quality is low?>
}

Evaluation criteria:
- confidence: How confident is the response? 0.0 = completely unsure, 1.0 = fully confident
- hedging_detected: Does the response use excessive hedging ("I think maybe", "I'm not sure but")?
- action:
  - "deliver": Response is good enough to send
  - "re-search": Response lacks specific facts that memory might have (retry with broader search)
  - "clarify": Response cannot be improved with memory; ask user for clarification
- memory_gap: If action is "re-search", what information was missing?

Be lenient — most responses should be "deliver". Only flag real quality issues.\
"""


def _format_reflection_input(response: str, context: str) -> str:
    parts = [
        "Context provided to assistant:",
        context[:1000] if context else "(no context)",
        "",
        "Assistant's response:",
        response[:2000],
    ]
    return "\n".join(parts)


# Schema hint for the retry prompt — keep in sync with the JSON shape in
# REFLECTION_PROMPT.
_REFLECTION_SCHEMA_HINT = (
    '{"confidence": float, "hedging_detected": bool, '
    '"action": "deliver|re-search|clarify", '
    '"memory_gap": string|null}'
)


def _parse_reflection_response(raw: str, original_response: str) -> ReflectionResult:
    """Single-shot parse with fallback — used by unit tests. Production
    path in `reflect()` uses `query_json_with_retry` for one corrective
    re-prompt before falling back."""
    from src.pipeline.json_utils import extract_json

    data = extract_json(raw)
    if data is None:
        log.warning(f"Reflection parse failed, defaulting to deliver (response: {raw[:100]!r})")
        return _default_reflection(original_response)
    return _result_from_data(data, original_response)


def _result_from_data(data: dict, original_response: str) -> ReflectionResult:
    """Project a parsed JSON dict into a ReflectionResult with safe defaults
    for any missing fields."""
    return ReflectionResult(
        response=original_response,
        confidence=data.get("confidence", 0.8),
        hedging_detected=data.get("hedging_detected", False),
        action=data.get("action", "deliver"),
        memory_gap=data.get("memory_gap"),
    )


def _default_reflection(original_response: str) -> ReflectionResult:
    return ReflectionResult(
        response=original_response,
        confidence=0.8,
        hedging_detected=False,
        action="deliver",
        memory_gap=None,
    )


async def reflect(
    response: str,
    context: str,
    model: str = "haiku",
    auth_mode: str = "max",
    api_key: str = "",
    cli_path: str = "",
) -> ReflectionResult:
    """Run quality gate on a response. Returns action recommendation."""
    from src.pipeline.json_utils import query_json_with_retry

    client = SingleTurnClient(model=model, auth_mode=auth_mode, api_key=api_key, cli_path=cli_path)
    input_text = _format_reflection_input(response, context)

    data = await query_json_with_retry(
        client.query,
        REFLECTION_PROMPT,
        input_text,
        _REFLECTION_SCHEMA_HINT,
        label="reflection",
    )
    if data is None:
        log.warning("Reflection falling back to deliver after parse exhausted")
        return _default_reflection(response)

    result = _result_from_data(data, response)
    log.info(
        f"Reflection: confidence={result.confidence:.2f}, "
        f"action={result.action}, hedging={result.hedging_detected}"
    )
    if result.memory_gap:
        log.info(f"Memory gap: {result.memory_gap}")
    return result
