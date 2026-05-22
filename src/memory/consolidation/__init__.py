"""Memory consolidation pipeline (Phase F.2).

Public surface:
    - init_tables(conn): create the consolidation schema
    - run_extract(...): extract structured facts from recent inputs
    - run_link(...): build the knowledge-graph edges + contradictions
    - run_brief(...): generate BRIEFING.md
    - run_consolidation(...): orchestrate all three stages
    - ConsolidationResult: counts surface for CLI / scheduler
    - ExtractedFact / FactEdge / PendingContradiction: dataclass shapes
"""

from src.memory.consolidation.brief import run_brief
from src.memory.consolidation.extract import run_extract
from src.memory.consolidation.link import run_link
from src.memory.consolidation.pipeline import (
    ConsolidationResult,
    run_consolidation,
)
from src.memory.consolidation.scheduler import (
    run_catchup_if_due,
    should_run,
    touch_marker,
)
from src.memory.consolidation.schema import (
    EDGE_RELATIONS,
    FACT_TYPES,
    ExtractedFact,
    FactEdge,
    PendingContradiction,
    init_tables,
)

__all__ = [
    "ConsolidationResult",
    "EDGE_RELATIONS",
    "ExtractedFact",
    "FACT_TYPES",
    "FactEdge",
    "PendingContradiction",
    "init_tables",
    "run_brief",
    "run_catchup_if_due",
    "run_consolidation",
    "run_extract",
    "run_link",
    "should_run",
    "touch_marker",
]
