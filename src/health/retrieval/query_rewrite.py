"""Convert natural-language health questions into keyword queries.

PubMed's E-utilities and MedlinePlus's Web Service both index by
clinical concepts and MeSH terms — they do okay with natural language
but FAR better with short keyword queries. Sending the user's verbatim
sentence ("Do blood pressure medications or statins sometimes cause
heartburn?") returns generic "blood pressure measurement" abstracts
because the search engine weights the dominant noun phrase. A
rewritten query like "antihypertensives statins gastroesophageal
reflux side-effects" lands the right literature on the first hit.

This module runs one short Haiku call to do that rewrite. ~200ms of
added latency, much better recall. On any failure (transport, empty
output, suspicious length) we fall back to the original message —
worse retrieval is preferable to a dropped turn.
"""

from __future__ import annotations

import logging

from src.llm.client import SingleTurnClient

log = logging.getLogger(__name__)


_REWRITE_PROMPT = """\
You are a search-query optimizer for medical literature databases
(PubMed, MedlinePlus, openFDA, ClinicalTrials.gov). Convert the user's
natural-language health question into a short keyword-style query
that PubMed's search engine indexes well.

Rules:
- 3 to 10 words. No full sentences, no question marks.
- Use clinical / pharmacological vocabulary when it's clearer:
  "antihypertensives" over "blood pressure pills", "gastroesophageal
  reflux" over "heartburn" when discussing mechanisms, "PDE5
  inhibitors" over "ED medications" when the class matters.
- Include adverse-effect / mechanism / pharmacology terms when the
  question is about side effects ("adverse effects", "side effects",
  "drug interactions", "contraindications" — pick what fits).
- Drop conversational filler ("can you tell me", "what about",
  "I'm wondering if", "sometimes", "do they ever").
- DO NOT invent symptoms or conditions the user didn't mention.
- DO NOT translate clinical terms the user already used into laypers
  prose — go the other direction.

Output ONLY the rewritten query, plain text. No quotes, no
explanation, no JSON, no surrounding markdown.

Examples:
  "Do blood pressure medications or statins sometimes cause heartburn?"
  → antihypertensives statins gastroesophageal reflux side effects

  "What does Cialis do?"
  → tadalafil mechanism of action pharmacology

  "How does the flu spread?"
  → influenza transmission epidemiology
"""

# Defensive cap: if the model returns something this long it
# misunderstood the task, and we'd rather fall back to the original
# than blow the search-API URL length budget.
_MAX_REWRITTEN_CHARS = 200


async def rewrite_query(
    message: str,
    topic: str | None = None,
    *,
    model: str = "haiku",
    auth_mode: str = "max",
    api_key: str = "",
    cli_path: str = "",
    workspace_dir: str = "",
) -> str:
    """Return a keyword-style search query rewritten from `message`.

    Falls back to `message` verbatim on any failure path (network,
    empty output, output too long). Worse retrieval is acceptable
    when the rewrite path is broken; a dropped turn is not.

    `topic` (optional) is the triage `health_topic` tag. When set
    we include it in the user input so the model can bias its
    rewrite toward the right vocabulary domain.
    """
    if not message or not message.strip():
        return message

    client = SingleTurnClient(
        model=model,
        auth_mode=auth_mode,
        api_key=api_key,
        cli_path=cli_path,
        workspace_dir=workspace_dir,
    )

    if topic:
        user_input = f"Topic tag: {topic}\nUser question: {message}"
    else:
        user_input = f"User question: {message}"

    try:
        raw = await client.query(_REWRITE_PROMPT, user_input)
    except Exception as e:
        log.warning(f"Query rewrite call failed: {e}; using original message")
        return message

    rewritten = _clean_output(raw)
    if not rewritten:
        log.info("Query rewrite returned empty; using original message")
        return message
    if len(rewritten) > _MAX_REWRITTEN_CHARS:
        log.info(
            "Query rewrite returned suspiciously long output (%d chars); "
            "using original message",
            len(rewritten),
        )
        return message

    log.info(f"Query rewrite: {message!r} -> {rewritten!r}")
    return rewritten


def _clean_output(raw: str) -> str:
    """Strip the wrapping junk the model might emit despite instructions:
    surrounding quotes, leading/trailing whitespace, a stray period at
    the end, accidental "Query: " prefix.
    """
    text = (raw or "").strip()
    # Strip leading "Query:", "Rewritten:", "Output:" labels
    for prefix in ("Query:", "Rewritten:", "Output:", "Search:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].lstrip()
            break
    # Strip wrapping quotes (single or double)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"', "`"):
        text = text[1:-1].strip()
    # Drop trailing period — keyword queries don't end in punctuation
    text = text.rstrip(".")
    return text.strip()
