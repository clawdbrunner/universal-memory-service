"""BM25 search wrapper using SQLite FTS5."""

from __future__ import annotations

import logging

from ..db import search_bm25

logger = logging.getLogger(__name__)


class BM25Search:
    """BM25 search via FTS5 with score normalization.

    Uses db.search_bm25 from Phase 1 and normalizes scores to 0-1 range.
    """

    async def search(
        self,
        query: str,
        limit: int = 20,
        filter_paths: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Search and return (chunk_id, normalized_score) pairs."""
        try:
            results = await search_bm25(query, limit=limit, filter_paths=filter_paths)
        except Exception:
            logger.warning("BM25 search failed", exc_info=True)
            return []

        if not results:
            return []

        # FTS5 bm25() returns negative scores (lower = better match)
        raw_scores = [abs(r["score"]) for r in results]
        max_score = max(raw_scores) if raw_scores else 1.0
        if max_score == 0:
            max_score = 1.0

        normalized: list[tuple[str, float]] = []
        for r, raw in zip(results, raw_scores):
            score = raw / max_score
            normalized.append((r["id"], score))

        return normalized
