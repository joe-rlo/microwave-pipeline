"""Health module — privacy-aware health Q&A.

Extends the existing pipeline with a triage-driven router that selects
between three downstream paths:

- "skip"    — non-health input, runs the default pipeline unchanged
- "general" — health content with no personal context; uses standard
              LLM client and the evidence-retrieval layer
- "phi"     — personal health content; routes through a BAA-covered
              LLM client (Phase 2; falls back to "general" with a
              safety message in Phase 1)

Phase 1 ships the general path: triage extension, retrieval against
PubMed and MedlinePlus, the health-qa skill, channel disclaimers, and
a non-PHI audit log. Disabling `HEALTH_MODULE_ENABLED` returns the
system to its current behavior with no health-aware code in any hot
path.

See `microwave-health-spec.md` for the full architecture.
"""
