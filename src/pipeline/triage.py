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

# Health classification block — appended to the JSON schema and the
# definitions section when the health module is enabled. Tightened
# in May 2026 after observing real over-triggering on benign generic
# questions ("what about ED medications?" → personal) and meta
# questions about the bot itself ("no results from the pipeline?" →
# personal). The rule is now: "personal" requires an EXPLICIT
# first-person marker, not just a sensitive topic.
_HEALTH_PROMPT_BLOCK = """\

Health classification (set phi_class):

- "none" — the message is not about health, medicine, symptoms, or
  treatments. THIS INCLUDES meta-questions about the bot itself or
  the conversation, even when the surrounding topic is health
  ("did you find anything?", "no results?", "can you search again?",
  "what tools do you have?", "so the pipeline returned nothing?").

- "general" — health concepts, conditions, drugs, or research in
  the abstract. Includes brief follow-ups that pivot the topic but
  stay generic ("what about X?", "and statins?", "how about ED
  medications?", "does Cialis cause heartburn?"). The defining
  property is the ABSENCE of a first-person reference to the user's
  body, history, meds, or labs.

- "personal" — REQUIRES an explicit first-person marker referring
  to the user's own body, symptoms, history, medications, or test
  results. Look for: "my <body part / lab>", "I'm <symptom>",
  "I take", "I have", "I had", "should I", "is this <my body part>",
  "my doctor said", a specific test value the user reported as theirs.

  Brushing a sensitive topic (cancer, ED, mental health, sex,
  pregnancy, addiction) does NOT make a question "personal" without
  one of those markers. "What does Cialis do?" is general; "is Cialis
  safe for me?" is personal. "Do statins cause heartburn?" is general;
  "am I getting heartburn from my statin?" is personal.

- "unknown" — only when there's a genuine ambiguity about whether
  the user is asking about themselves vs. a hypothetical / friend /
  news story. A bare topic mention in isolation is NOT ambiguous;
  it's general until a first-person marker appears.

Topic carryover from recent conversation: if the previous turn was
classified general and the current message is a brief follow-up
with no first-person markers, the follow-up stays "general".
Conversation context wins over isolated keyword patterns.

Also set `health_topic` to a short tag like "diabetes", "medication",
"side-effects", "vaccines", "nutrition" when the message has a clear
topic; otherwise null.{project_note}
"""

_HEALTH_PROJECT_NOTE = """

An active writing project named "{project}" is in scope. References to
fictional characters' bodies, symptoms, medications, or conditions inside
that project's content classify as "general" (not "personal") — a novel
about a character with diabetes is research, not the user's own PHI."""


def _build_prompt(
    skills: list[tuple[str, str]] | None,
    health_enabled: bool = False,
    active_project: str | None = None,
) -> str:
    """Compose the triage prompt.

    Skill catalog and health classification are both opt-in: when off
    they don't appear in the prompt at all, so a zero-skill / health-off
    install keeps the original prompt shape and token count.
    """
    base = TRIAGE_PROMPT_BASE
    if not skills:
        base = base.replace(',\n  "matched_skill": <string or null>', "")

    if health_enabled:
        # Inject phi_class + health_topic into the JSON shape and append
        # the classification rules. We splice into the JSON schema before
        # the closing brace so the model sees one coherent shape, not
        # two trailing addenda.
        health_fields = ',\n  "phi_class": "none" | "general" | "personal" | "unknown",\n  "health_topic": <string or null>'
        base = base.rstrip()
        # Find the last `}` in the JSON example and inject before it.
        last_brace = base.rfind("}")
        if last_brace != -1:
            base = base[:last_brace].rstrip() + health_fields + "\n" + base[last_brace:]

        project_note = (
            _HEALTH_PROJECT_NOTE.format(project=active_project)
            if active_project
            else ""
        )
        base = base + _HEALTH_PROMPT_BLOCK.format(project_note=project_note)

    if not skills:
        return base
    catalog_lines = [f"- {name}: {desc or '(no description)'}" for name, desc in skills]
    return base + _SKILLS_PROMPT_TEMPLATE.format(catalog="\n".join(catalog_lines))


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


# Schema hint shown to the model on the retry prompt — terse on purpose so
# the retry call stays cheap. Keep in sync with the JSON shape in
# TRIAGE_PROMPT_BASE; if you add a field to the prompt, mirror it here.
_TRIAGE_SCHEMA_HINT = (
    '{"intent": "recall|preference|task|question|social", '
    '"complexity": "simple|moderate|complex", '
    '"needs_memory": true|false, '
    '"search_params": {"decay_half_life": float, "result_count": int, '
    '"weight_recency": float, "mmr_lambda": float}, '
    '"matched_skill": string|null}'
)

# Schema hint with health fields appended — used for the retry prompt
# when the health module is enabled. Same compactness rule applies.
_TRIAGE_SCHEMA_HINT_HEALTH = (
    '{"intent": "recall|preference|task|question|social", '
    '"complexity": "simple|moderate|complex", '
    '"needs_memory": true|false, '
    '"search_params": {"decay_half_life": float, "result_count": int, '
    '"weight_recency": float, "mmr_lambda": float}, '
    '"matched_skill": string|null, '
    '"phi_class": "none|general|personal|unknown", '
    '"health_topic": string|null}'
)


def _parse_triage_response(
    response: str, available_skills: set[str] | None = None
) -> TriageResult:
    """Single-shot parse with fallback to defaults — used by unit tests
    and as the base projection for the retry helper. The production path
    in `triage()` goes through `query_json_with_retry`, which calls into
    `_result_from_data` directly after a successful parse."""
    from src.pipeline.json_utils import extract_json

    data = extract_json(response)
    if data is None:
        log.warning(f"Triage parse failed, using defaults (response: {response[:100]!r})")
        return _default_triage()
    return _result_from_data(data, available_skills=available_skills)


def _result_from_data(
    data: dict, available_skills: set[str] | None
) -> TriageResult:
    """Project a parsed JSON dict into a TriageResult with safe defaults
    for any missing fields. Centralized so the parse path and the retry
    path share the same coercion logic."""
    matched = data.get("matched_skill")
    if matched is not None:
        matched = str(matched).strip() or None
    if matched and available_skills is not None and matched not in available_skills:
        log.info(f"Triage returned unknown skill {matched!r}; discarding")
        matched = None

    # Health classification — defaults to safe values when fields are
    # absent (i.e., health module disabled or model dropped them).
    phi_class = str(data.get("phi_class", "none") or "none").lower()
    if phi_class not in ("none", "general", "personal", "unknown"):
        log.info(f"Triage returned unknown phi_class {phi_class!r}; defaulting to none")
        phi_class = "none"
    health_topic = data.get("health_topic")
    if health_topic is not None:
        health_topic = str(health_topic).strip() or None

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
        phi_class=phi_class,
        health_topic=health_topic,
    )


async def triage(
    message: str,
    recent_turns: list[Turn],
    model: str = "haiku",
    auth_mode: str = "max",
    api_key: str = "",
    cli_path: str = "",
    skills: list[tuple[str, str]] | None = None,
    workspace_dir: str = "",
    health_enabled: bool = False,
    active_project: str | None = None,
) -> TriageResult:
    """Run triage classification on a user message.

    `skills` is an optional catalog of (name, description) tuples.
    When provided, the prompt asks Haiku to also return a matched_skill
    (or null) — enables the adaptive-skill activation feature without
    any second model call.

    `health_enabled` toggles the PHI classification block in the prompt.
    When False the prompt's shape is identical to before the health
    module existed; the parser still tolerates phi_class fields and
    defaults to "none" so callers don't need branching.

    `active_project` is the user's currently-active writing project (if
    any). When set, fictional-content references inside that project's
    scope classify as "general" rather than "personal" — the agreed
    policy for a novel about a character with diabetes.
    """
    from src.pipeline.json_utils import query_json_with_retry

    client = SingleTurnClient(
        model=model, auth_mode=auth_mode, api_key=api_key, cli_path=cli_path,
        workspace_dir=workspace_dir,
    )
    input_text = _format_triage_input(message, recent_turns)
    prompt = _build_prompt(
        skills,
        health_enabled=health_enabled,
        active_project=active_project,
    )
    available = {name for name, _ in skills} if skills else None

    schema_hint = _TRIAGE_SCHEMA_HINT_HEALTH if health_enabled else _TRIAGE_SCHEMA_HINT
    data = await query_json_with_retry(
        client.query,
        prompt,
        input_text,
        schema_hint,
        label="triage",
    )
    if data is None:
        log.warning("Triage falling back to defaults after parse exhausted")
        return _default_triage()

    result = _result_from_data(data, available_skills=available)
    skill_note = f", skill={result.matched_skill!r}" if result.matched_skill else ""
    health_note = (
        f", phi_class={result.phi_class}"
        + (f"/{result.health_topic}" if result.health_topic else "")
        if result.phi_class != "none" else ""
    )
    log.info(
        f"Triage: intent={result.intent}, complexity={result.complexity}, "
        f"needs_memory={result.needs_memory}{skill_note}{health_note}"
    )
    return result
