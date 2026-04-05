"""File indexer — hash, chunk, embed, store."""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import time
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

from .chunker import chunk_markdown
from .config import get_config
from .db import (
    delete_chunks_for_file,
    get_file_state,
    insert_chunks,
    update_file_state,
)
from .retrieval.embeddings import EmbeddingService
from .retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _should_ignore(file_path: str, patterns: list[str]) -> bool:
    """Check if a file path matches any ignore pattern.

    For patterns starting with ``*``, use fnmatch against the filename.
    For other patterns, check if any path segment equals the pattern.
    """
    segments = Path(file_path).parts
    for pattern in patterns:
        if pattern.startswith("*"):
            if fnmatch.fnmatch(segments[-1], pattern):
                return True
        else:
            if pattern in segments:
                return True
    return False


class IndexResult:
    """Result of an indexing operation."""

    __slots__ = ("chunks_stored", "embeddings_ok")

    def __init__(self, chunks_stored: int, embeddings_ok: bool) -> None:
        self.chunks_stored = chunks_stored
        self.embeddings_ok = embeddings_ok

    @property
    def is_partial(self) -> bool:
        """True when chunks were stored but embeddings failed."""
        return self.chunks_stored > 0 and not self.embeddings_ok


class Indexer:
    """Indexes files into the chunk/vector/FTS stores."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        self._embeddings = embedding_service
        self._vectors = vector_store
        self._config = get_config()
        self._last_embedding_success: str | None = None
        self._last_embedding_failure: str | None = None
        self._recently_indexed: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def index_file(self, file_path: str) -> IndexResult:
        """Index a single file if its content has changed.

        Returns an IndexResult with chunks_stored count and embedding status.
        """
        # Debounce: skip if this file was indexed less than 2 seconds ago
        # Exception: daily logs (agents/*/logs/*.md) are always re-indexed
        now = time.time()
        is_daily_log = "/logs/" in file_path and file_path.endswith(".md")
        last = self._recently_indexed.get(file_path, 0)
        if not is_daily_log and now - last < 2.0:
            logger.debug("Skipping recently indexed file: %s", file_path)
            return IndexResult(0, True)
        self._recently_indexed[file_path] = now

        # Periodically clean up old entries
        if len(self._recently_indexed) > 100:
            cutoff = now - 60
            self._recently_indexed = {k: v for k, v in self._recently_indexed.items() if v > cutoff}

        path = Path(file_path)
        if not path.is_file():
            logger.warning("index_file: not a file: %s", file_path)
            return IndexResult(0, True)

        content = path.read_text(encoding="utf-8", errors="replace")
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        state = await get_file_state(file_path)
        if state and state["content_hash"] == content_hash:
            return IndexResult(0, True)  # unchanged

        # Remove stale data
        await delete_chunks_for_file(file_path)
        await self._vectors.delete_for_file(file_path)

        # Chunk
        chunks = chunk_markdown(
            content,
            file_path,
            chunk_size=self._config.index.chunk_size_tokens,
            overlap=self._config.index.chunk_overlap_tokens,
        )
        if not chunks:
            await update_file_state(file_path, content_hash, 0)
            return IndexResult(0, True)

        # Store chunks (triggers FTS via DB triggers)
        await insert_chunks(chunks)

        # Embeddings
        texts = [c.content for c in chunks]
        embeddings = await self._embeddings.generate(texts)
        embeddings_ok = len(embeddings) > 0
        if embeddings_ok:
            for chunk, emb in zip(chunks, embeddings):
                if emb:
                    await self._vectors.upsert(chunk.id, emb)
            self._last_embedding_success = _now()
        else:
            self._last_embedding_failure = _now()
            logger.warning("Embeddings failed for %s; chunks stored for BM25 only", file_path)

        await update_file_state(file_path, content_hash, len(chunks))
        logger.info("Indexed %s → %d chunks (embeddings=%s)", file_path, len(chunks), embeddings_ok)
        return IndexResult(len(chunks), embeddings_ok)

    @property
    def embedding_health(self) -> dict[str, str | None]:
        """Return last embedding success/failure timestamps."""
        return {
            "last_success": self._last_embedding_success,
            "last_failure": self._last_embedding_failure,
        }

    async def index_directory(self, directory: str | None = None) -> int:
        """Walk a directory and index all matching files.

        Returns total chunks stored.
        """
        root = Path(directory) if directory else Path(self._config.memory.data_dir)
        extensions = set(self._config.memory.extensions)
        ignore = self._config.memory.ignore_patterns
        total = 0
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in extensions:
                if _should_ignore(str(path), ignore):
                    continue
                result = await self.index_file(str(path))
                total += result.chunks_stored
        return total

    async def remove_file(self, file_path: str) -> None:
        """Remove all indexed data for a file."""
        await delete_chunks_for_file(file_path)
        await self._vectors.delete_for_file(file_path)
        # Remove file_state row
        from .db import get_connection

        async with get_connection() as db:
            await db.execute(
                "DELETE FROM file_state WHERE file_path = ?", (file_path,)
            )
            await db.commit()
        logger.info("Removed index for %s", file_path)

    async def reindex_all(self) -> int:
        """Full reindex of the data directory."""
        logger.info("Starting full reindex of %s", self._config.memory.data_dir)
        return await self.index_directory()
