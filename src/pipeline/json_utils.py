"""Robust JSON extraction from LLM responses.

Models sometimes return JSON wrapped in markdown fences,
or followed by explanatory text. This handles all those cases.

`query_json_with_retry` adds a one-shot retry on parse failure for the
small-model stages (triage, reflection) where Haiku occasionally returns
malformed JSON. The retry shows the bad response back to the model and
asks for valid JSON only — in practice this recovers the vast majority
of failures, with the second-failure path falling through to the
caller's defaults.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Awaitable, Callable

log = logging.getLogger(__name__)


def extract_json(text: str) -> dict | None:
    """Extract the first JSON object from text, ignoring surrounding content.

    Handles:
    - Clean JSON
    - JSON in ```json ... ``` code fences
    - JSON followed by extra text/explanation
    - JSON preceded by preamble
    """
    text = text.strip()

    # Try 1: Strip code fences and parse
    stripped = _strip_code_fence(text)
    result = _try_parse(stripped)
    if result is not None:
        return result

    # Try 2: Find first { and match to its closing }
    result = _extract_braced(text)
    if result is not None:
        return result

    return None


def _strip_code_fence(text: str) -> str:
    """Remove markdown code fences if present."""
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _try_parse(text: str) -> dict | None:
    """Try to parse text as JSON, return None on failure."""
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return None


async def query_json_with_retry(
    client_call: Callable[[str, str], Awaitable[str]],
    system_prompt: str,
    user_input: str,
    schema_hint: str,
    *,
    label: str = "json",
) -> dict | None:
    """Query an LLM and parse the response as JSON, with one corrective retry.

    `client_call(system, user) -> response_text` is awaited for each attempt.
    On parse failure the second attempt re-prompts with the original bad
    response shown back to the model, asking for valid JSON only.

    Returns the parsed dict on success, or None after exhausting retries
    (one retry max). Callers are responsible for fallback defaults — None
    is the signal to use them.

    `label` is used for structured logging so failures can be audited
    later (greppable via `json_parse status=failed label=<label>`).

    Why one retry instead of zero or many: zero leaves Haiku's noise as a
    silent triage/reflection failure with wrong defaults; many delays the
    diagnosis when the prompt itself is the problem. One catches transient
    formatting noise without obscuring real prompt issues.
    """
    response = ""
    try:
        response = await client_call(system_prompt, user_input)
    except Exception as e:
        log.warning(
            "json_parse status=client_error label=%s error=%s",
            label, e,
        )
        return None

    parsed = extract_json(response)
    if parsed is not None:
        log.debug("json_parse status=parsed label=%s", label)
        return parsed

    # First attempt failed — re-prompt with the bad response shown back.
    log.info(
        "json_parse status=retrying label=%s response_len=%d preview=%r",
        label, len(response), response[:120],
    )
    retry_user = (
        f"Your previous response was not valid JSON. "
        f"Previous response (between triple backticks):\n```\n{response}\n```\n\n"
        f"Return ONLY valid JSON matching this schema. No explanation, "
        f"no markdown fences, no surrounding text:\n{schema_hint}"
    )
    retry_response = ""
    try:
        retry_response = await client_call(system_prompt, retry_user)
    except Exception as e:
        log.warning(
            "json_parse status=retry_error label=%s error=%s",
            label, e,
        )
        return None

    parsed = extract_json(retry_response)
    if parsed is not None:
        log.info("json_parse status=retried label=%s", label)
        return parsed

    log.warning(
        "json_parse status=failed label=%s "
        "first_preview=%r retry_preview=%r",
        label, response[:120], retry_response[:120],
    )
    return None


def _extract_braced(text: str) -> dict | None:
    """Find the first top-level { ... } in text and parse it."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        c = text[i]

        if escape:
            escape = False
            continue

        if c == "\\":
            escape = True
            continue

        if c == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return _try_parse(text[start : i + 1])

    return None
