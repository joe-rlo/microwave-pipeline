"""Startup-catchup hook for the consolidation pipeline.

Per Open Question 14.10 in the spec, the bot's host (laptop) often
isn't on at 3 AM when a typical cron would fire. Rather than relying
on an external scheduler, the orchestrator checks at startup whether
the last consolidation run was more than `lookback_hours` ago and
fires one if so. The check is cheap (one file mtime); the run itself
goes to a background task so startup isn't blocked.

The "last run" timestamp lives as the mtime of
`~/.microwaveos/data/.last_consolidation`. The pipeline is
idempotent (fact IDs are content-hashed) so a "missed" cron firing
at 3 AM gets picked up at next startup with no duplication.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

import apsw

log = logging.getLogger(__name__)


# Marker file mtime represents the last successful consolidation run.
_MARKER_BASENAME = ".last_consolidation"


def marker_path(data_dir: Path) -> Path:
    return Path(data_dir) / _MARKER_BASENAME


def should_run(
    *,
    data_dir: Path,
    interval_hours: int = 24,
    now: float | None = None,
) -> bool:
    """True if consolidation has not run in the last `interval_hours`.

    Fresh installs (no marker file) ALWAYS return True — first start
    populates the graph from whatever's there.
    """
    marker = marker_path(data_dir)
    if not marker.exists():
        return True
    now_ts = now if now is not None else time.time()
    age_hours = (now_ts - marker.stat().st_mtime) / 3600
    return age_hours >= interval_hours


def touch_marker(data_dir: Path) -> None:
    """Update the marker file's mtime to now. Called after a successful run."""
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    marker_path(data_dir).touch()


async def run_catchup_if_due(
    *,
    conn: apsw.Connection,
    config: Any,
    interval_hours: int = 24,
) -> bool:
    """Fire a consolidation run if one is overdue. Returns True if it ran.

    Designed to be awaited from `Orchestrator.start()` — but since
    the orchestrator's startup is already async, callers usually wrap
    this in `asyncio.create_task()` to avoid blocking the bot's
    first-message latency on a multi-second consolidation.
    """
    data_dir = getattr(config, "data_dir", None)
    if data_dir is None:
        log.debug("Catchup skipped (config has no data_dir)")
        return False

    if not should_run(data_dir=data_dir, interval_hours=interval_hours):
        log.debug("Catchup skipped — last run within %d h", interval_hours)
        return False

    workspace_dir = getattr(config, "workspace_dir", None)
    daily_dir = None
    briefing_path = None
    if workspace_dir is not None:
        daily_dir = Path(workspace_dir) / "memory"
        briefing_path = Path(workspace_dir) / "BRIEFING.md"

    from src.memory.consolidation import run_consolidation

    try:
        result = await run_consolidation(
            conn=conn,
            config=config,
            daily_notes_dir=daily_dir if daily_dir and daily_dir.exists() else None,
            briefing_path=briefing_path,
            lookback_hours=interval_hours,
        )
        log.info(
            "Catchup consolidation: %d new facts, %d edges, %d contradictions, "
            "%d ms",
            result.new_facts, result.edges, result.contradictions,
            result.duration_ms,
        )
        touch_marker(data_dir)
        return True
    except Exception as e:
        log.warning("Catchup consolidation failed: %s", e)
        return False
