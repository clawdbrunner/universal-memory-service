"""Vector store with SQLite persistence and in-memory cosine similarity search."""

from __future__ import annotations

import json
import logging
import math

from ..db import get_connection

logger = logging.getLogger(__name__)


class VectorStore:
    """Store and search embeddings using cosine similarity.

    Embeddings are stored in SQLite (as BLOB, JSON-encoded float array)
    and cached in memory for fast search. Cosine similarity is computed
    in Python against all stored embeddings.
    """

    def __init__(self) -> None:
        self._cache: dict[str, list[float]] = {}
        self._file_index: dict[str, set[str]] = {}  # file_path -> chunk_ids
        self._loaded = False

    async def _ensure_loaded(self) -> None:
        """Load all embeddings from SQLite into memory on first access."""
        if self._loaded:
            return
        async with get_connection() as db:
            cursor = await db.execute(
                "SELECT e.chunk_id, e.embedding, c.file_path "
                "FROM embeddings e JOIN chunks c ON e.chunk_id = c.id"
            )
            rows = await cursor.fetchall()
        for row in rows:
            chunk_id = row["chunk_id"]
            embedding = json.loads(row["embedding"])
            file_path = row["file_path"]
            self._cache[chunk_id] = embedding
            self._file_index.setdefault(file_path, set()).add(chunk_id)
        self._loaded = True
        logger.info("Loaded %d embeddings into memory", len(self._cache))

    async def upsert(self, chunk_id: str, embedding: list[float]) -> None:
        """Insert or update an embedding in SQLite and in-memory cache."""
        blob = json.dumps(embedding)
        async with get_connection() as db:
            await db.execute(
                "INSERT OR REPLACE INTO embeddings (chunk_id, embedding) VALUES (?, ?)",
                (chunk_id, blob),
            )
            await db.commit()
            # Update file index
            cursor = await db.execute(
                "SELECT file_path FROM chunks WHERE id = ?", (chunk_id,)
            )
            row = await cursor.fetchone()
        self._cache[chunk_id] = embedding
        if row:
            self._file_index.setdefault(row["file_path"], set()).add(chunk_id)

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        filter_paths: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Search for similar embeddings using cosine similarity.

        Returns list of (chunk_id, similarity_score) sorted by score descending.
        """
        await self._ensure_loaded()

        logger.debug("VectorStore.search: cache=%d items, filter_paths=%s", len(self._cache), filter_paths)

        qnorm = math.sqrt(sum(x * x for x in query_embedding))
        logger.debug("VectorStore.search: query embedding norm=%.4f, dims=%d", qnorm, len(query_embedding))

        if not self._cache:
            return []

        candidates = self._cache
        if filter_paths:
            allowed_ids: set[str] = set()
            for file_path, chunk_ids in self._file_index.items():
                for prefix in filter_paths:
                    if prefix in file_path:
                        allowed_ids.update(chunk_ids)
                        break
            candidates = {k: v for k, v in candidates.items() if k in allowed_ids}

        scores: list[tuple[str, float]] = []
        for chunk_id, stored_emb in candidates.items():
            score = _cosine_similarity(query_embedding, stored_emb)
            scores.append((chunk_id, score))

        score_vals = [s for _, s in scores]
        logger.debug(
            "VectorStore.search: after filter %d candidates, top scores: %s",
            len(candidates),
            [f"{s:.3f}" for s in sorted(score_vals, reverse=True)[:3]] if score_vals else "none",
        )

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    async def delete_for_file(self, file_path: str) -> None:
        """Delete all embeddings for a given file path."""
        chunk_ids = self._file_index.pop(file_path, set())
        for cid in chunk_ids:
            self._cache.pop(cid, None)
        if chunk_ids:
            placeholders = ",".join("?" for _ in chunk_ids)
            async with get_connection() as db:
                await db.execute(
                    f"DELETE FROM embeddings WHERE chunk_id IN ({placeholders})",
                    list(chunk_ids),
                )
                await db.commit()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
