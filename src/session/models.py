"""Data models for session and pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TriageResult:
    intent: str  # "recall", "preference", "task", "question", "social"
    complexity: str  # "simple", "moderate", "complex"
    search_params: dict = field(default_factory=dict)
    needs_memory: bool = True


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
