"""Chunking, embedding, and FTS indexing for memory fragments.

SQLite (via apsw) with sqlite-vec (vector similarity) and FTS5 (keyword search).
"""

from __future__ import annotations

import logging
import struct
from datetime import datetime
from pathlib import Path

import apsw

from src.db import connect as db_connect
from src.memory.embeddings import EMBEDDING_DIMENSION, EMBEDDING_VERSION, EmbeddingClient

log = logging.getLogger(__name__)

CHUNK_TARGET_TOKENS = 200


def _serialize_vector(vec: list[float]) -> bytes:
    """Serialize a float vector to bytes for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


def _deserialize_vector(data: bytes) -> list[float]:
    """Deserialize bytes back to a float vector."""
    n = len(data) // 4
    return list(struct.unpack(f"{n}f", data))


class MemoryIndex:
    def __init__(self, db_path: Path, embedding_client: EmbeddingClient):
        self.db_path = db_path
        self.embedder = embedding_client
        self.conn: apsw.Connection | None = None
        self._has_vec = False

    def connect(self) -> None:
        self.conn = db_connect(self.db_path)
        self._load_extensions()  # load vec extension first so all tables are created in one pass
        self._init_tables()

    def _load_extensions(self) -> None:
        """Load sqlite-vec extension if available."""
        try:
            import sqlite_vec

            self.conn.enableloadextension(True)
            sqlite_vec.load(self.conn)
            self.conn.enableloadextension(False)
            self._has_vec = True
            log.info("sqlite-vec extension loaded")
        except (ImportError, Exception) as e:
            log.warning(f"sqlite-vec not available, vector search disabled: {e}")

    def _init_tables(self) -> None:
        """Create all tables in one pass to avoid schema-change errors."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS fragments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                embedding_version TEXT,
                retrieval_count INTEGER DEFAULT 0,
                metadata JSON
            )
        """)
        self.conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS fragments_fts USING fts5(
                content,
                content_rowid='id'
            )
        """)
        if self._has_vec:
            try:
                self.conn.execute(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS fragments_vec "
                    f"USING vec0(id INTEGER PRIMARY KEY, embedding float[{EMBEDDING_DIMENSION}])"
                )
            except Exception as e:
                log.warning(f"Could not create vec table: {e}")
                self._has_vec = False

    def index_file(self, path: Path, force: bool = False) -> int:
        """Index a file into fragments, skipping if unchanged since last index.

        Compares file mtime against the timestamp of the most recent fragment
        for that source. If the file is newer (or has never been indexed),
        deletes stale fragments and reindexes from scratch.

        Returns the number of new fragments created (0 = skipped/empty).
        """
        if not path.exists():
            return 0

        source = str(path)
        file_mtime = path.stat().st_mtime

        if not force:
            rows = list(self.conn.execute(
                "SELECT MAX(timestamp) FROM fragments WHERE source = ?",
                (source,),
            ))
            if rows and rows[0][0]:
                try:
                    last_indexed = datetime.fromisoformat(rows[0][0]).timestamp()
                    if file_mtime <= last_indexed:
                        log.debug(f"Skipping {path.name} (unchanged)")
                        return 0
                except (ValueError, OSError):
                    pass

        self._delete_source(source)

        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return 0

        return len(self.index_text(text, source))

    def _delete_source(self, source: str) -> None:
        """Remove all fragments and their FTS/vec entries for a given source."""
        rows = list(self.conn.execute(
            "SELECT id FROM fragments WHERE source = ?", (source,)
        ))
        ids = [r[0] for r in rows]
        if not ids:
            return
        for fid in ids:
            self.conn.execute("DELETE FROM fragments_fts WHERE rowid = ?", (fid,))
        if self._has_vec:
            for fid in ids:
                try:
                    self.conn.execute("DELETE FROM fragments_vec WHERE id = ?", (fid,))
                except Exception:
                    pass
        self.conn.execute("DELETE FROM fragments WHERE source = ?", (source,))
        log.debug(f"Deleted {len(ids)} fragments for source {source!r}")

    def index_text(self, text: str, source: str, timestamp: datetime | None = None) -> list[int]:
        """Chunk text and index each chunk. Returns list of fragment IDs."""
        timestamp = timestamp or datetime.now()
        chunks = self._chunk(text)
        if not chunks:
            return []

        embeddings = self.embedder.embed_batch(chunks)
        ids = []

        for chunk, embedding in zip(chunks, embeddings):
            self.conn.execute(
                "INSERT INTO fragments (content, source, timestamp, embedding_version) "
                "VALUES (?, ?, ?, ?)",
                (chunk, source, timestamp.isoformat(), EMBEDDING_VERSION),
            )
            frag_id = self.conn.last_insert_rowid()
            ids.append(frag_id)

            # FTS index
            self.conn.execute(
                "INSERT INTO fragments_fts (rowid, content) VALUES (?, ?)",
                (frag_id, chunk),
            )

            # Vector index
            if self._has_vec:
                try:
                    self.conn.execute(
                        "INSERT INTO fragments_vec (id, embedding) VALUES (?, ?)",
                        (frag_id, _serialize_vector(embedding)),
                    )
                except Exception:
                    pass

        log.info(f"Indexed {len(ids)} fragments from {source}")
        return ids

    def increment_retrieval(self, fragment_id: int) -> None:
        self.conn.execute(
            "UPDATE fragments SET retrieval_count = retrieval_count + 1 WHERE id = ?",
            (fragment_id,),
        )

    def get_promotion_candidates(self, min_retrievals: int = 3) -> list[dict]:
        """Find fragments retrieved 3+ times — candidates for MEMORY.md promotion."""
        rows = list(self.conn.execute(
            "SELECT id, content, source, retrieval_count FROM fragments "
            "WHERE retrieval_count >= ? ORDER BY retrieval_count DESC",
            (min_retrievals,),
        ))
        return [dict(row) for row in rows]

    def _chunk(self, text: str) -> list[str]:
        """Split text at paragraph boundaries, targeting ~200 tokens per chunk."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return [text.strip()] if text.strip() else []

        chunks = []
        current = []
        current_len = 0

        for para in paragraphs:
            para_tokens = len(para) // 4
            if current_len + para_tokens > CHUNK_TARGET_TOKENS and current:
                chunks.append("\n\n".join(current))
                current = [para]
                current_len = para_tokens
            else:
                current.append(para)
                current_len += para_tokens

        if current:
            chunks.append("\n\n".join(current))

        return chunks

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None
