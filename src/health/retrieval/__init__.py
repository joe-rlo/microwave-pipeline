"""Evidence retrieval — parallel fan-out across authoritative health sources.

Each source (PubMed, MedlinePlus, openFDA, CDC, ClinicalTrials.gov) is an
async-friendly client implementing `EvidenceSource`. The orchestrator
runs them concurrently with `asyncio.gather`, deduplicates by URL, and
ranks results so the LLM gets the most relevant + most authoritative
fragments first.

Phase 1 ships PubMed and MedlinePlus implementations. The base classes
and orchestrator are source-agnostic — Phase 3 sources just register
themselves and inherit the same fan-out / rank / cache plumbing.
"""

from src.health.retrieval.base import Evidence, EvidenceSource

__all__ = ["Evidence", "EvidenceSource"]
