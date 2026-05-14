"""Data models for session and pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TriageResult:
    intent: str  # "recall", "preference", "task", "question", "social", "meta"
    complexity: str  # "simple", "moderate", "complex"
    search_params: dict = field(default_factory=dict)
    needs_memory: bool = True
    # Name of a skill the triage stage matched to this message, or None.
    # Ephemeral — this drives per-turn skill activation only; it doesn't
    # persist anywhere. Orchestrator ignores it when an explicit skill is
    # already pinned. Defaults to None for back-compat with older callers.
    matched_skill: str | None = None

    # --- Health module classification ---
    # `phi_class` distinguishes non-health input from general health
    # questions from personal health queries that contain PHI:
    #   "none"     — not health-related; route to existing pipeline
    #   "general"  — health concepts in the abstract; standard LLM + retrieval
    #   "personal" — references the user's own body / symptoms / meds / labs
    #   "unknown"  — classifier is uncertain; treated as "personal" for safety
    # The triage prompt biases toward `personal`/`unknown` over `general`
    # because false positives cost a Bedrock call but false negatives could
    # leak PHI to a non-BAA endpoint. Defaults to "none" so callers that
    # don't enable the health module see no behavioral change.
    phi_class: str = "none"
    # Short topic tag like "diabetes", "medication", "symptoms" — used by
    # retrieval to rank source matches and by audit to dedupe similar queries.
    # Optional; the model returns null when no clear topic applies.
    health_topic: str | None = None


@dataclass
class MemoryFragment:
    id: int
    content: str
    source: str  # file path or "sqlite"
    timestamp: datetime
    score: float = 0.0
    retrieval_count: int = 0
    # "fragment" = durable indexed content (MEMORY.md, identity, daily notes).
    # "turn" = recent conversation turn pulled live from the turns table.
    # Used by assembly to label the two retrieval sources distinctly in the prompt.
    source_type: str = "fragment"


@dataclass
class SearchResult:
    fragments: list[MemoryFragment] = field(default_factory=list)
    strategy_used: str = ""
    search_time_ms: int = 0


@dataclass
class AssemblyResult:
    stable_prompt: str = ""
    memory_context: str = ""
    token_budget_used: int = 0
    promote_candidates: list[MemoryFragment] = field(default_factory=list)


@dataclass
class ReflectionResult:
    response: str = ""
    confidence: float = 1.0
    hedging_detected: bool = False
    action: str = "deliver"  # "deliver", "re-search", "clarify"
    memory_gap: str | None = None
    # Which reflection lane produced this result — used by /debug and
    # the audit log to see at a glance whether the model-call path
    # fired or the regex shortcut did.
    #   "skipped" — simple-tier turns; regex hedge-check only, no model call
    #   "normal"  — moderate-tier; standard reflection prompt
    #   "deep"    — complex-tier; adds unsupported-claim check
    path: str = "normal"


@dataclass
class Turn:
    id: int | None = None
    session_id: str = ""
    channel: str = ""
    user_id: str = ""
    role: str = ""  # "user" or "assistant"
    content: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    token_count: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class PipelineMetadata:
    triage: TriageResult | None = None
    search: SearchResult | None = None
    reflection: ReflectionResult | None = None
    escalated: bool = False
    escalated_model: str = ""
    total_time_ms: int = 0
    cost_usd: float = 0.0
    # Skill-driven pipeline overrides applied to this turn (2.3). Empty
    # when no skill is active or the active skill omits a pipeline block.
    # Stored on metadata so `/debug` and the audit log can show *which*
    # overrides actually shaped the turn — important when a misconfigured
    # skill silently flips behavior the user wasn't expecting.
    skill_overrides: dict = field(default_factory=dict)
