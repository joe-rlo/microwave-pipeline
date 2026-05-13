"""Stage 4: Reflection — quality gate on LLM output before delivery.

Uses Haiku for fast evaluation (~300ms).
Checks for hedging, contradictions, or insufficient context.
Caps at one retry to avoid loops.

Variants:
- "normal" (default) — original prompt, used for moderate-complexity turns
- "deep" — used for complex-complexity turns. Adds an unsupported-claim
  check on top of the hedging check.

For simple-complexity turns the orchestrator skips this stage entirely
and uses `simple_hedge_check()` — a regex-only fast path that produces
a ReflectionResult-shaped object so downstream code (audit metadata,
re-search guard) keeps working unchanged.
"""

from __future__ import annotations

import logging
import re

from src.llm.client import SingleTurnClient
from src.session.models import ReflectionResult

log = logging.getLogger(__name__)

REFLECTION_PROMPT_NORMAL = """\
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

# Used for complex-tier turns. Adds an unsupported-claims check on top
# of the normal hedging check. Complex turns are where the model is
# more likely to confidently assert specifics that aren't grounded —
# worth the extra ~300ms to catch them before delivery.
REFLECTION_PROMPT_DEEP = """\
You are a quality gate for an AI assistant's responses on a complex,
multi-step question. Evaluate the response with extra scrutiny.

Respond with ONLY valid JSON:

{
  "confidence": <float, 0.0-1.0>,
  "hedging_detected": <bool — did the response hedge excessively?>,
  "unsupported_claims": <bool — did the response state specifics that
    aren't grounded in the provided context or commonly-known facts?>,
  "action": "deliver" | "re-search" | "clarify",
  "memory_gap": <string | null — what info was missing if quality is low?>
}

Evaluation criteria:
- confidence: How confident is the response? 0.0 = completely unsure, 1.0 = fully confident
- hedging_detected: Does the response use excessive hedging?
- unsupported_claims: Does the response confidently assert specific
  facts (numbers, dates, attributions, quotations) that don't trace
  back to the provided context or to widely-known general knowledge?
  Flag confidently-asserted hallucinations. Do NOT flag opinion or
  analysis presented as the assistant's own view.
- action:
  - "deliver": Response is good enough to send
  - "re-search": Response lacks specific facts that memory might have
  - "clarify": Response cannot be improved with memory; ask user
- memory_gap: If action is "re-search", what information was missing?

Be strict on unsupported_claims for complex turns — the cost of
catching a hallucination here is one regeneration; the cost of
delivering one is the user mistrusting the bot.\
"""

# Backward-compat alias — some external code / tests still import
# REFLECTION_PROMPT directly. Points at the normal variant.
REFLECTION_PROMPT = REFLECTION_PROMPT_NORMAL


# Hedge tokens we scan for on simple-tier turns where we've skipped
# the model-call reflection. These are the same patterns the model
# would catch under `hedging_detected: true` but cheaper. Whole-word
# matching via word boundaries to avoid catching e.g. "thinker" → "I think".
_HEDGE_PATTERNS = re.compile(
    r"\b("
    r"perhaps|maybe|i\s+think|i\s+believe|it\s+seems|it\s+appears|"
    r"might\s+be|could\s+be|i'?m\s+not\s+sure|i'?m\s+not\s+certain|"
    r"sort\s+of|kind\s+of|i\s+suppose|i\s+guess|probably|presumably"
    r")\b",
    re.IGNORECASE,
)


def simple_hedge_check(response: str) -> ReflectionResult:
    """Regex-only stand-in for reflection on simple-tier turns.

    Detects hedging via word-boundary token match. Produces a
    ReflectionResult shaped exactly like the model-call path so
    downstream callers (audit metadata, re-search guard) don't
    branch on which path produced the result.

    By design this NEVER triggers re-search — a simple-tier turn that
    hedged doesn't have richer context to retrieve; the answer just
    needs to be more direct. The hedging signal lands in
    `hedging_detected` for audit visibility, and reflection's `action`
    stays "deliver" to keep the simple path fast.
    """
    hedged = bool(_HEDGE_PATTERNS.search(response or ""))
    return ReflectionResult(
        response=response,
        confidence=0.85 if not hedged else 0.65,
        hedging_detected=hedged,
        action="deliver",
        memory_gap=None,
        path="skipped",
    )


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
    workspace_dir: str = "",
    variant: str = "normal",
) -> ReflectionResult:
    """Run quality gate on a response. Returns action recommendation.

    `variant`:
    - "normal" — original prompt, used for moderate-complexity turns
    - "deep"   — adds unsupported-claims check, used for complex turns

    For simple-complexity turns, callers should use `simple_hedge_check()`
    directly rather than calling this function with a "skip" variant —
    the regex helper avoids the model round-trip entirely.
    """
    from src.pipeline.json_utils import query_json_with_retry

    if variant == "deep":
        prompt = REFLECTION_PROMPT_DEEP
    else:
        prompt = REFLECTION_PROMPT_NORMAL

    client = SingleTurnClient(
        model=model, auth_mode=auth_mode, api_key=api_key, cli_path=cli_path,
        workspace_dir=workspace_dir,
    )
    input_text = _format_reflection_input(response, context)

    data = await query_json_with_retry(
        client.query,
        prompt,
        input_text,
        _REFLECTION_SCHEMA_HINT,
        label=f"reflection_{variant}",
    )
    if data is None:
        log.warning("Reflection falling back to deliver after parse exhausted")
        fallback = _default_reflection(response)
        fallback.path = variant
        return fallback

    result = _result_from_data(data, response)
    result.path = variant
    deep_note = ""
    if variant == "deep" and data.get("unsupported_claims"):
        # Surface the unsupported-claims signal in logs even though
        # the ReflectionResult dataclass doesn't have a dedicated
        # field for it yet. Future: add the field if this flag turns
        # out to be the right place to trigger regeneration.
        deep_note = " unsupported_claims=true"
    log.info(
        f"Reflection ({variant}): confidence={result.confidence:.2f}, "
        f"action={result.action}, hedging={result.hedging_detected}{deep_note}"
    )
    if result.memory_gap:
        log.info(f"Memory gap: {result.memory_gap}")
    return result
