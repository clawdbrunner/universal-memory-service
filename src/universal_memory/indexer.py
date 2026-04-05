"""File indexer — hash, chunk, embed, store."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def index_file(self, file_path: str) -> int:
        """Index a single file if its content has changed.

        Returns the number of chunks stored (0 if unchanged).
        """
        path = Path(file_path)
        if not path.is_file():
            logger.warning("index_file: not a file: %s", file_path)
            return 0

        content = path.read_text(encoding="utf-8", errors="replace")
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        state = await get_file_state(file_path)
        if state and state["content_hash"] == content_hash:
            return 0  # unchanged

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
            return 0

        # Store chunks (triggers FTS via DB triggers)
        await insert_chunks(chunks)

        # Embeddings
        texts = [c.content for c in chunks]
        embeddings = await self._embeddings.generate(texts)
        for chunk, emb in zip(chunks, embeddings):
            if emb:
                await self._vectors.upsert(chunk.id, emb)

        await update_file_state(file_path, content_hash, len(chunks))
        logger.info("Indexed %s → %d chunks", file_path, len(chunks))
        return len(chunks)

    async def index_directory(self, directory: str | None = None) -> int:
        """Walk a directory and index all matching files.

        Returns total chunks stored.
        """
        root = Path(directory) if directory else Path(self._config.memory.data_dir)
        extensions = set(self._config.memory.extensions)
        total = 0
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in extensions:
                total += await self.index_file(str(path))
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
