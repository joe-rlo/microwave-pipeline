"""Per-hook state persistence under workspace/heartbeat/<hook>.json.

Each tick saves the raw snapshot; the next tick loads it as `previous`
for the judge to compare against. Survives process restart so the
judge has continuity.

JSON, not SQLite: snapshots are small, atomic-write via rename is
trivial, and `cat workspace/heartbeat/blink.json` is human-readable
for debugging. SQLite would be overkill for this surface.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def state_path(state_dir: Path, hook_name: str) -> Path:
    """Where one hook's snapshot lives."""
    return Path(state_dir) / f"{hook_name}.json"


def load_hook_state(state_dir: Path, hook_name: str) -> Optional[dict]:
    """Return the most-recent saved snapshot, or None on first run /
    corrupt file. Never raises — judge must tolerate either."""
    p = state_path(state_dir, hook_name)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning(
            "[heartbeat-state] read failed for %s (%s); treating as fresh",
            hook_name, e,
        )
        return None


def save_hook_state(state_dir: Path, hook_name: str, data: dict) -> None:
    """Persist the current snapshot atomically (.tmp → rename)."""
    p = state_path(state_dir, hook_name)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8",
        )
        tmp.replace(p)
    except Exception as e:
        log.warning(
            "[heartbeat-state] write failed for %s: %s", hook_name, e,
        )
