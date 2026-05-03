"""Memory contradiction surfacing — auditing aid for MEMORY.md.

Append-only memory accumulates over time, and the user has no built-in
way to spot when newer entries quietly contradict older ones ("dog's
name is Biscuit" → "dog's name is Max"). The agent silently has both
in its head and gets confused; the user can't tell why.

This module provides one function — `detect_contradictions` — that
runs an LLM pass over `MEMORY.md` and returns a list of likely
conflicts for the user to resolve manually.

It deliberately does NOT auto-merge or auto-delete. Sovereignty over
memory is the design value: the user owns their own memory file, the
agent's job is to flag, not rewrite. Even "obvious" contradictions
might be context-dependent (different time periods, different
referents) — only the user knows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.llm.client import SingleTurnClient
from src.pipeline.json_utils import query_json_with_retry

log = logging.getLogger(__name__)


_DETECTION_PROMPT = """\
You are auditing a personal memory document for contradictions. Identify
entries that disagree about a stable fact — same entity (person, place,
thing, preference) with conflicting attributes.

Respond with ONLY valid JSON. No explanation, no markdown fences:

{
  "contradictions": [
    {
      "a": "<exact quoted line from the document>",
      "b": "<exact quoted line from the document>",
      "summary": "<one short sentence describing the conflict>"
    }
  ]
}

Return an empty `"contradictions": []` if there are none.

Important — what NOT to flag:
- Entries that *update* an entity rather than contradict it
  (e.g. "moved from NYC" → "now lives in Berlin" is an update,
  not a contradiction).
- Entries about different referents that share a name.
- Entries that are simply unrelated.

When uncertain, do NOT flag. False positives are worse than missing
some contradictions; the user has to triage every flagged item.\
"""


_DETECTION_SCHEMA_HINT = (
    '{"contradictions": [{"a": string, "b": string, "summary": string}]}'
)


@dataclass
class Contradiction:
    """Two lines from MEMORY.md that disagree about a stable fact.

    `a` and `b` are the verbatim lines (as the model returned them);
    `summary` is a one-sentence description of the conflict for display.
    """
    a: str
    b: str
    summary: str


async def detect_contradictions(
    memory_text: str,
    *,
    model: str = "haiku",
    auth_mode: str = "max",
    api_key: str = "",
    cli_path: str = "",
) -> list[Contradiction]:
    """Scan a memory document for likely contradictions.

    Returns an empty list if the document is empty, the model returns
    no contradictions, or the parse exhausts retries — in all three
    cases the right user-facing answer is "no flagged conflicts."
    """
    text = (memory_text or "").strip()
    if not text:
        return []

    client = SingleTurnClient(
        model=model,
        auth_mode=auth_mode,
        api_key=api_key,
        cli_path=cli_path,
    )

    data = await query_json_with_retry(
        client.query,
        _DETECTION_PROMPT,
        text,
        _DETECTION_SCHEMA_HINT,
        label="memory_health",
    )
    if data is None:
        log.warning("Memory health: parse exhausted; returning no contradictions")
        return []

    raw_items = data.get("contradictions") or []
    if not isinstance(raw_items, list):
        log.warning(
            "Memory health: 'contradictions' was not a list (got %r); "
            "returning no contradictions",
            type(raw_items).__name__,
        )
        return []

    out: list[Contradiction] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        a = str(item.get("a", "")).strip()
        b = str(item.get("b", "")).strip()
        summary = str(item.get("summary", "")).strip()
        if not a or not b:
            continue
        out.append(Contradiction(a=a, b=b, summary=summary or "(no summary)"))
    return out


def format_for_cli(contradictions: list[Contradiction]) -> str:
    """Render a contradictions list for terminal output.

    Empty list → a single "no contradictions" line. Non-empty → a
    numbered list with quoted entries and a summary, plus a one-line
    nudge that resolution is manual (the user edits `MEMORY.md`).
    """
    if not contradictions:
        return "✓ No contradictions detected in MEMORY.md."

    n = len(contradictions)
    lines = [f"⚠ {n} contradiction{'s' if n != 1 else ''} in MEMORY.md", ""]
    for i, c in enumerate(contradictions, 1):
        lines.append(f"{i}. {c.summary}")
        lines.append(f"   A: {c.a!r}")
        lines.append(f"   B: {c.b!r}")
        lines.append("")
    lines.append("To resolve: edit workspace/MEMORY.md directly. Re-run")
    lines.append("`microwaveos memory health` to verify.")
    return "\n".join(lines)
