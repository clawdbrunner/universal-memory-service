"""Tests for async SQLite database layer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from universal_memory.db import (
    delete_chunks_for_file,
    get_file_state,
    get_stats,
    init_db,
    insert_chunks,
    update_file_state,
)
from universal_memory.models import Chunk


@pytest.fixture
async def db_path(tmp_path):
    path = str(tmp_path / "test.db")
    await init_db(path)
    return path


def _make_chunk(
    file_path: str = "test.md",
    content: str = "hello world",
    chunk_id: str | None = None,
) -> Chunk:
    return Chunk(
        document_id="doc1",
        file_path=file_path,
        line_start=1,
        line_end=5,
        content=content,
        header_path="## Section",
        token_count=10,
        id=chunk_id or f"chunk-{content[:8]}",
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestInitDb:
    @pytest.mark.asyncio
    async def test_creates_tables(self, tmp_path):
        path = str(tmp_path / "fresh.db")
        await init_db(path)
        assert Path(path).exists()

        # Verify tables by inserting and reading
        chunk = _make_chunk()
        count = await insert_chunks([chunk], db_path=path)
        assert count == 1

    @pytest.mark.asyncio
    async def test_idempotent(self, tmp_path):
        path = str(tmp_path / "idem.db")
        await init_db(path)
        await init_db(path)  # should not raise
        stats = await get_stats(db_path=path)
        assert "chunks" in stats


# ---------------------------------------------------------------------------
# Chunks CRUD
# ---------------------------------------------------------------------------


class TestChunkOperations:
    @pytest.mark.asyncio
    async def test_insert_and_count(self, db_path):
        chunks = [_make_chunk(content=f"chunk {i}", chunk_id=f"c{i}") for i in range(3)]
        inserted = await insert_chunks(chunks, db_path=db_path)
        assert inserted == 3

        stats = await get_stats(db_path=db_path)
        assert stats["chunks"] == 3

    @pytest.mark.asyncio
    async def test_delete_chunks_for_file(self, db_path):
        chunks = [_make_chunk(content=f"c{i}", chunk_id=f"c{i}") for i in range(2)]
        await insert_chunks(chunks, db_path=db_path)

        deleted = await delete_chunks_for_file("test.md", db_path=db_path)
        assert deleted == 2

        stats = await get_stats(db_path=db_path)
        assert stats["chunks"] == 0

    @pytest.mark.asyncio
    async def test_insert_empty_list(self, db_path):
        count = await insert_chunks([], db_path=db_path)
        assert count == 0


# ---------------------------------------------------------------------------
# File state
# ---------------------------------------------------------------------------


class TestFileState:
    @pytest.mark.asyncio
    async def test_track_file_state(self, db_path):
        await update_file_state("a.md", "abc123", 5, db_path=db_path)
        state = await get_file_state("a.md", db_path=db_path)
        assert state is not None
        assert state["content_hash"] == "abc123"
        assert state["chunk_count"] == 5

    @pytest.mark.asyncio
    async def test_upsert_file_state(self, db_path):
        await update_file_state("a.md", "hash1", 3, db_path=db_path)
        await update_file_state("a.md", "hash2", 7, db_path=db_path)
        state = await get_file_state("a.md", db_path=db_path)
        assert state["content_hash"] == "hash2"
        assert state["chunk_count"] == 7

    @pytest.mark.asyncio
    async def test_nonexistent_file_state(self, db_path):
        state = await get_file_state("missing.md", db_path=db_path)
        assert state is None
