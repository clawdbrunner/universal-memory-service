"""Async SQLite database layer for Universal Memory Service."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, AsyncIterator

import aiosqlite

from .config import get_config
from .models import Chunk

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    content_hash TEXT NOT NULL,
    modified_at TEXT,
    size_bytes INTEGER DEFAULT 0,
    indexed_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    line_start INTEGER NOT NULL,
    line_end INTEGER NOT NULL,
    content TEXT NOT NULL,
    header_path TEXT DEFAULT '',
    file_modified_at TEXT,
    indexed_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    token_count INTEGER DEFAULT 0,
    embedding_hash TEXT DEFAULT '',
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_file_path ON chunks(file_path);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    file_path,
    header_path,
    content='chunks',
    content_rowid='rowid'
);

-- Triggers to keep FTS in sync with chunks table
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content, file_path, header_path)
    VALUES (new.rowid, new.content, new.file_path, new.header_path);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content, file_path, header_path)
    VALUES ('delete', old.rowid, old.content, old.file_path, old.header_path);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content, file_path, header_path)
    VALUES ('delete', old.rowid, old.content, old.file_path, old.header_path);
    INSERT INTO chunks_fts(rowid, content, file_path, header_path)
    VALUES (new.rowid, new.content, new.file_path, new.header_path);
END;

CREATE TABLE IF NOT EXISTS file_state (
    file_path TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    last_indexed_at TEXT,
    chunk_count INTEGER DEFAULT 0
);
"""


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------


def _db_path() -> str:
    return get_config().index.db_path


async def init_db(db_path: str | None = None) -> None:
    """Create the database and all tables/triggers.

    Creates parent directories if needed.
    """
    path = db_path or _db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(path) as db:
        await db.executescript(_SCHEMA)
        await db.commit()


@contextlib.asynccontextmanager
async def get_connection(db_path: str | None = None) -> AsyncIterator[aiosqlite.Connection]:
    """Yield an async SQLite connection."""
    path = db_path or _db_path()
    db = await aiosqlite.connect(path)
    db.row_factory = aiosqlite.Row
    try:
        yield db
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------


async def insert_chunks(chunks: list[Chunk], db_path: str | None = None) -> int:
    """Insert chunks into the chunks table. Returns the number inserted."""
    if not chunks:
        return 0
    async with get_connection(db_path) as db:
        await db.executemany(
            """INSERT OR REPLACE INTO chunks
               (id, document_id, file_path, line_start, line_end,
                content, header_path, token_count, embedding_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    c.id,
                    c.document_id,
                    c.file_path,
                    c.line_start,
                    c.line_end,
                    c.content,
                    c.header_path,
                    c.token_count,
                    c.embedding_hash,
                )
                for c in chunks
            ],
        )
        await db.commit()
    return len(chunks)


async def delete_chunks_for_file(file_path: str, db_path: str | None = None) -> int:
    """Delete all chunks for a given file. Returns rows deleted."""
    async with get_connection(db_path) as db:
        cursor = await db.execute(
            "DELETE FROM chunks WHERE file_path = ?", (file_path,)
        )
        await db.commit()
        return cursor.rowcount


async def get_chunks_for_file(
    file_path: str, db_path: str | None = None
) -> list[dict[str, Any]]:
    """Return all chunks for a file as dicts."""
    async with get_connection(db_path) as db:
        cursor = await db.execute(
            "SELECT * FROM chunks WHERE file_path = ? ORDER BY line_start",
            (file_path,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def search_bm25(
    query: str,
    limit: int = 20,
    filter_paths: list[str] | None = None,
    db_path: str | None = None,
) -> list[dict[str, Any]]:
    """Run a BM25 search against FTS5. Returns ranked results with scores."""
    async with get_connection(db_path) as db:
        if filter_paths:
            placeholders = ",".join("?" for _ in filter_paths)
            sql = f"""
                SELECT c.*, bm25(chunks_fts) AS score
                FROM chunks_fts f
                JOIN chunks c ON c.rowid = f.rowid
                WHERE chunks_fts MATCH ?
                  AND c.file_path IN ({placeholders})
                ORDER BY score
                LIMIT ?
            """
            params: list[Any] = [query, *filter_paths, limit]
        else:
            sql = """
                SELECT c.*, bm25(chunks_fts) AS score
                FROM chunks_fts f
                JOIN chunks c ON c.rowid = f.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY score
                LIMIT ?
            """
            params = [query, limit]

        cursor = await db.execute(sql, params)
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def get_file_state(
    file_path: str, db_path: str | None = None
) -> dict[str, Any] | None:
    """Return file_state row for a path, or None."""
    async with get_connection(db_path) as db:
        cursor = await db.execute(
            "SELECT * FROM file_state WHERE file_path = ?", (file_path,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None


async def update_file_state(
    file_path: str,
    content_hash: str,
    chunk_count: int,
    db_path: str | None = None,
) -> None:
    """Upsert file_state for a given path."""
    async with get_connection(db_path) as db:
        await db.execute(
            """INSERT INTO file_state (file_path, content_hash, last_indexed_at, chunk_count)
               VALUES (?, ?, strftime('%Y-%m-%dT%H:%M:%SZ', 'now'), ?)
               ON CONFLICT(file_path) DO UPDATE SET
                 content_hash = excluded.content_hash,
                 last_indexed_at = excluded.last_indexed_at,
                 chunk_count = excluded.chunk_count""",
            (file_path, content_hash, chunk_count),
        )
        await db.commit()


async def get_stats(db_path: str | None = None) -> dict[str, Any]:
    """Return index statistics."""
    async with get_connection(db_path) as db:
        files = await (await db.execute("SELECT COUNT(*) FROM file_state")).fetchone()
        chunks = await (await db.execute("SELECT COUNT(*) FROM chunks")).fetchone()
        embeddings = await (
            await db.execute(
                "SELECT COUNT(*) FROM chunks WHERE embedding_hash != ''"
            )
        ).fetchone()
        last = await (
            await db.execute(
                "SELECT MAX(last_indexed_at) FROM file_state"
            )
        ).fetchone()

        return {
            "files_indexed": files[0] if files else 0,
            "chunks": chunks[0] if chunks else 0,
            "embeddings": embeddings[0] if embeddings else 0,
            "last_indexed_at": last[0] if last else None,
        }
