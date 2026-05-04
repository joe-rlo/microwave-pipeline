"""Health router — picks a downstream path based on triage classification.

Pure function, no I/O. The orchestrator calls `route()` after triage
and consults the returned `HealthRoute` to decide:

- whether to run the standard pipeline unchanged ("skip")
- whether to enable evidence retrieval and the health-qa skill
  ("general" or "phi")
- whether to use the BAA-covered LLM client ("phi" only)
- whether to surface a "this isn't medical advice" disclaimer
- whether to *decline* the turn entirely with a safety message because
  the user sent PHI but the BAA path isn't configured ("decline_phi")

Why a "decline_phi" path. The spec says: "If HEALTH_BAA_PROVIDER=none,
the module refuses to enable the PHI path and logs a warning. Personal
queries fall back to a safety message." We model that as a routing
decision rather than letting the orchestrator infer it from the
absence of BAA config — cleaner, and the audit log can record the
specific "we declined to process this" outcome rather than a hole.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.health.config import HealthConfig
from src.session.models import TriageResult


HealthPath = Literal["skip", "general", "phi", "decline_phi"]


@dataclass(frozen=True)
class HealthRoute:
    path: HealthPath
    # Short human-readable phrase — surfaced in audit and `/debug`.
    # Not user-facing copy.
    reason: str
    # True only on the BAA path. The orchestrator selects the Bedrock
    # client when this is set; otherwise the standard LLM client runs.
    use_baa_llm: bool
    # General + phi paths fan out to PubMed/MedlinePlus/etc.; skip
    # and decline_phi don't.
    enable_retrieval: bool
    # When True, the response gets the health.md channel rules appended
    # so disclaimers fire. Always True for any non-skip path; False for
    # skip and decline_phi (decline_phi has its own safety message
    # instead, no disclaimer needed).
    require_disclaimer: bool

    @property
    def is_health(self) -> bool:
        """Convenience for callers that just need 'did the route fire?'.

        decline_phi counts — even though we won't run the pipeline,
        the audit log treats it as a health-touched turn."""
        return self.path != "skip"


def route(triage: TriageResult, config: HealthConfig) -> HealthRoute:
    """Pick a path for a triage-classified message.

    Decision tree:
    1. Module disabled → skip (zero behavior change vs. pre-health build)
    2. phi_class == "none" → skip (not health-related)
    3. phi_class == "general" → general path (retrieval + disclaimer,
       standard LLM)
    4. phi_class in ("personal", "unknown") AND BAA configured → phi path
       (retrieval + disclaimer + Bedrock LLM)
    5. phi_class in ("personal", "unknown") AND BAA NOT configured →
       decline_phi (safety message, no LLM call, no retrieval)

    Step 5 is the Phase 1 reality: we don't ship a Bedrock client
    until Phase 2, so personal queries fall back to a safety message
    by default. Once Phase 2 lands and `phi_path_available` flips True,
    the same router code starts routing personal queries to BAA without
    any caller-side change.
    """
    if not config.enabled:
        return HealthRoute(
            path="skip",
            reason="module disabled",
            use_baa_llm=False,
            enable_retrieval=False,
            require_disclaimer=False,
        )

    if triage.phi_class == "none":
        return HealthRoute(
            path="skip",
            reason="not health-related",
            use_baa_llm=False,
            enable_retrieval=False,
            require_disclaimer=False,
        )

    if triage.phi_class == "general":
        return HealthRoute(
            path="general",
            reason="general health query",
            use_baa_llm=False,
            enable_retrieval=True,
            require_disclaimer=True,
        )

    # phi_class in ("personal", "unknown")
    if config.phi_path_available:
        return HealthRoute(
            path="phi",
            reason=f"phi_class={triage.phi_class}",
            use_baa_llm=True,
            enable_retrieval=True,
            require_disclaimer=True,
        )

    # Personal/unknown but no BAA path — decline cleanly. The orchestrator
    # interprets this as "respond with a safety message instead of running
    # the pipeline." The audit log records the decline so we know the
    # module declined-rather-than-leaked.
    return HealthRoute(
        path="decline_phi",
        reason=f"phi_class={triage.phi_class} but no BAA provider configured",
        use_baa_llm=False,
        enable_retrieval=False,
        require_disclaimer=False,
    )


# User-facing safety message for the decline_phi path. Channels surface
# this verbatim instead of running the pipeline. Kept short and
# action-oriented — the user knows their query touched personal info,
# we want them to either rephrase abstractly or enable the BAA path.
DECLINE_PHI_MESSAGE = (
    "Your message looks like it includes personal health details, and the "
    "BAA-covered processing path isn't configured. To answer responsibly "
    "I'd need either:\n\n"
    "• A rephrasing without specific personal details (treat it as a "
    "general question — \"what does X do\" rather than \"should I take X\"), or\n"
    "• A BAA provider enabled in your config "
    "(see HEALTH_BAA_PROVIDER in the docs).\n\n"
    "For anything urgent, please contact a clinician or call emergency services."
)
