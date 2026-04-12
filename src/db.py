"""Shared database connection factory using apsw.

apsw supports loadable extensions (unlike stdlib sqlite3 on macOS python.org builds),
which is required for sqlite-vec.
"""

from __future__ import annotations

import logging
from pathlib import Path

import apsw
import apsw.bestpractice

log = logging.getLogger(__name__)

# Apply apsw best practices (WAL mode, foreign keys, etc.)
apsw.bestpractice.apply(apsw.bestpractice.recommended)


class Row(dict):
    """Dict-like row that supports both dict[key] and row.key access."""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def _row_factory(cursor, row):
    """Convert rows to Row dicts keyed by column name."""
    description = cursor.getdescription()
    return Row(zip([d[0] for d in description], row))


def connect(db_path: Path) -> apsw.Connection:
    """Create an apsw connection with row factory configured."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = apsw.Connection(str(db_path))
    conn.setrowtrace(_row_factory)
    return conn
