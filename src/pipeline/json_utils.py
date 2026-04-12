"""Robust JSON extraction from LLM responses.

Models sometimes return JSON wrapped in markdown fences,
or followed by explanatory text. This handles all those cases.
"""

from __future__ import annotations

import json
import re


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
