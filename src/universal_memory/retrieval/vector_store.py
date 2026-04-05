"""Vector store with SQLite persistence and cosine similarity search."""

from __future__ import annotations

import json
import logging
import math

from ..db import get_connection

logger = logging.getLogger(__name__)


class VectorStore:
    """Store and search embeddings using cosine similarity.

    Embeddings are stored in SQLite (as JSON-encoded float arrays).
    Each search queries SQLite directly — no in-memory cache.
    """

    async def upsert(self, chunk_id: str, embedding: list[float]) -> None:
        """Insert or update an embedding in SQLite."""
        blob = json.dumps(embedding)
        async with get_connection() as db:
            await db.execute(
                "INSERT OR REPLACE INTO embeddings (chunk_id, embedding) VALUES (?, ?)",
                (chunk_id, blob),
            )
            await db.commit()

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        filter_paths: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Search for similar embeddings using cosine similarity.

        Returns list of (chunk_id, similarity_score) sorted by score descending.
        """
        async with get_connection() as db:
            cursor = await db.execute(
                "SELECT e.chunk_id, e.embedding, c.file_path "
                "FROM embeddings e JOIN chunks c ON e.chunk_id = c.id"
            )
            rows = await cursor.fetchall()

        logger.debug("VectorStore.search: %d rows from DB, filter_paths=%s", len(rows), filter_paths)

        candidates: dict[str, list[float]] = {}
        for row in rows:
            chunk_id = row["chunk_id"]
            embedding = json.loads(row["embedding"])
            file_path = row["file_path"]

            if filter_paths:
                if not any(prefix in file_path for prefix in filter_paths):
                    continue

            candidates[chunk_id] = embedding

        scores: list[tuple[str, float]] = []
        for chunk_id, stored_emb in candidates.items():
            score = _cosine_similarity(query_embedding, stored_emb)
            scores.append((chunk_id, score))

        score_vals = [s for _, s in scores]
        logger.debug(
            "VectorStore.search: %d candidates, top scores: %s",
            len(candidates),
            [f"{s:.3f}" for s in sorted(score_vals, reverse=True)[:3]] if score_vals else "none",
        )

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    async def delete_for_file(self, file_path: str) -> None:
        """Delete all embeddings for a given file path."""
        async with get_connection() as db:
            await db.execute(
                "DELETE FROM embeddings WHERE chunk_id IN "
                "(SELECT id FROM chunks WHERE file_path = ?)",
                (file_path,),
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
