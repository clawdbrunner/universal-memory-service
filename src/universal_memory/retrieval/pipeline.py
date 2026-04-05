"""6-stage retrieval pipeline: expand -> vector -> bm25 -> graphiti -> merge -> rerank."""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone

from ..config import get_config
from ..db import get_chunk
from ..models import SearchRequest, SearchResponse, SearchResult
from .bm25 import BM25Search
from .embeddings import EmbeddingService
from .expander import QueryExpanderService
from .graphiti import GraphitiClient
from .reranker import RerankerService
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """6-stage retrieval pipeline with graceful degradation at every stage.

    Stages: expand -> vector search -> bm25 search -> graphiti search -> merge -> rerank
    """

    def __init__(self, vector_store: VectorStore | None = None, embeddings: EmbeddingService | None = None) -> None:
        self._config = get_config()
        self.embeddings = embeddings or EmbeddingService()
        self.vector_store = vector_store or VectorStore()
        self.bm25 = BM25Search()
        self.graphiti = GraphitiClient()
        self.reranker = RerankerService()
        self.expander = QueryExpanderService()

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Execute the full 6-stage retrieval pipeline."""
        timing_ms: dict[str, float] = {}
        sources_queried: list[str] = []

        # Stage 1: Query expansion
        t0 = time.perf_counter()
        try:
            if request.expand:
                queries = await self.expander.expand(request.query)
            else:
                queries = [request.query]
        except Exception:
            logger.warning("Expand stage failed", exc_info=True)
            queries = [request.query]
        timing_ms["expand"] = (time.perf_counter() - t0) * 1000

        # Derive filter_paths and group_ids from request scope
        filter_paths: list[str] | None = None
        group_ids: list[str] | None = None
        if request.author:
            agent_info = self._config.agents.get(request.author)
            dept = agent_info.department if agent_info else None
            filter_paths = [f"agents/{request.author}/", "shared/"]
            group_ids = [f"memory-{request.author}"]
            if dept:
                filter_paths.append(f"departments/{dept}/")
                group_ids.extend([f"memory-{dept}", "memory-shared"])
        elif request.department:
            filter_paths = [f"departments/{request.department}/", "shared/"]
            group_ids = [f"memory-{request.department}", "memory-shared"]

        # Stage 2: Vector search
        t0 = time.perf_counter()
        vector_results: list[tuple[str, float]] = []
        if "files" in request.sources:
            try:
                for q in queries:
                    embs = await self.embeddings.generate([q])
                    if embs:
                        hits = await self.vector_store.search(
                            embs[0], top_k=20, filter_paths=filter_paths
                        )
                        vector_results.extend(hits)
                sources_queried.append("vector")
            except Exception:
                logger.warning("Vector search stage failed", exc_info=True)
        timing_ms["vector"] = (time.perf_counter() - t0) * 1000

        # Stage 3: BM25 search
        t0 = time.perf_counter()
        bm25_results: list[tuple[str, float]] = []
        if "files" in request.sources:
            try:
                for q in queries:
                    hits = await self.bm25.search(q, limit=20, filter_paths=filter_paths)
                    bm25_results.extend(hits)
                sources_queried.append("bm25")
            except Exception:
                logger.warning("BM25 search stage failed", exc_info=True)
        timing_ms["bm25"] = (time.perf_counter() - t0) * 1000

        # Stage 4: Graphiti search
        t0 = time.perf_counter()
        graphiti_results: list[SearchResult] = []
        if "graphiti" in request.sources:
            try:
                graphiti_results = await self.graphiti.search(
                    request.query, group_ids=group_ids, limit=10
                )
                sources_queried.append("graphiti")
            except Exception:
                logger.warning("Graphiti search stage failed", exc_info=True)
        timing_ms["graphiti"] = (time.perf_counter() - t0) * 1000

        # Stage 5: Merge
        t0 = time.perf_counter()
        try:
            merged = await self._merge(vector_results, bm25_results, graphiti_results)
        except Exception:
            logger.warning("Merge stage failed", exc_info=True)
            merged = []
        timing_ms["merge"] = (time.perf_counter() - t0) * 1000

        total_candidates = len(merged)

        # Stage 6: Rerank
        t0 = time.perf_counter()
        try:
            if request.rerank:
                candidates_n = self._config.models.reranker.candidates
                reranked = await self.reranker.rerank(
                    request.query, merged[:candidates_n], top_n=request.max_results
                )
            else:
                reranked = merged[: request.max_results]
        except Exception:
            logger.warning("Rerank stage failed", exc_info=True)
            reranked = merged[: request.max_results]
        timing_ms["rerank"] = (time.perf_counter() - t0) * 1000

        # Filter by min_score
        final = [r for r in reranked if r.score >= request.min_score]

        scope: dict[str, object] = {}
        if filter_paths is not None:
            scope["directories_searched"] = filter_paths
        if group_ids is not None:
            scope["graphiti_group_ids"] = group_ids

        return SearchResponse(
            results=final,
            query=request.query,
            scope=scope,
            expanded_queries=queries,
            sources_queried=sources_queried,
            timing_ms=timing_ms,
        )

    async def _merge(
        self,
        vector_results: list[tuple[str, float]],
        bm25_results: list[tuple[str, float]],
        graphiti_results: list[SearchResult],
    ) -> list[SearchResult]:
        """Merge results: normalize scores, weighted combine, temporal decay, MMR dedup."""
        weights = self._config.search.weights
        merged: dict[str, SearchResult] = {}

        # Normalize and add vector results
        for chunk_id, score in _normalize_scores(vector_results):
            chunk_data = await get_chunk(chunk_id)
            if not chunk_data:
                continue
            sr = _chunk_to_result(chunk_data, score * weights.vector, "vector")
            if chunk_id in merged:
                merged[chunk_id].score += sr.score
            else:
                merged[chunk_id] = sr

        # Normalize and add BM25 results
        for chunk_id, score in _normalize_scores(bm25_results):
            if chunk_id in merged:
                merged[chunk_id].score += score * weights.bm25
            else:
                chunk_data = await get_chunk(chunk_id)
                if not chunk_data:
                    continue
                sr = _chunk_to_result(chunk_data, score * weights.bm25, "bm25")
                merged[chunk_id] = sr

        # Add Graphiti results (already SearchResult objects)
        for gr in graphiti_results:
            key = gr.chunk_id or f"graphiti_{id(gr)}"
            if key in merged:
                merged[key].score += gr.score * weights.graphiti
            else:
                gr.score *= weights.graphiti
                merged[key] = gr

        # Temporal decay
        td_cfg = self._config.search.temporal_decay
        if td_cfg.enabled:
            now = datetime.now(timezone.utc)
            for sr in merged.values():
                if sr.file_path and any(
                    sr.file_path.endswith(ex) for ex in td_cfg.exempt_files
                ):
                    continue
                modified = sr.metadata.get("file_modified_at")
                if modified:
                    try:
                        mod_dt = datetime.fromisoformat(modified)
                        if mod_dt.tzinfo is None:
                            mod_dt = mod_dt.replace(tzinfo=timezone.utc)
                        age_days = (now - mod_dt).total_seconds() / 86400
                        decay = math.pow(0.5, age_days / td_cfg.half_life_days)
                        sr.score *= decay
                    except (ValueError, TypeError):
                        pass

        # Sort by score descending
        results = sorted(merged.values(), key=lambda r: r.score, reverse=True)

        # MMR dedup
        mmr_cfg = self._config.search.mmr
        if mmr_cfg.enabled and results:
            results = _mmr_dedup(results, mmr_cfg.lambda_)

        return results


def _normalize_scores(pairs: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Normalize scores to 0-1 range, deduplicating by chunk_id (keep highest)."""
    if not pairs:
        return []
    best: dict[str, float] = {}
    for chunk_id, score in pairs:
        if chunk_id not in best or score > best[chunk_id]:
            best[chunk_id] = score
    scores = list(best.values())
    max_s = max(scores) if scores else 1.0
    min_s = min(scores) if scores else 0.0
    range_s = max_s - min_s if max_s != min_s else 1.0
    return [(cid, (s - min_s) / range_s) for cid, s in best.items()]


def _chunk_to_result(chunk_data: dict, score: float, source: str) -> SearchResult:
    """Convert a chunk dict from the database to a SearchResult."""
    return SearchResult(
        chunk_id=chunk_data["id"],
        score=score,
        source=source,
        content=chunk_data["content"],
        file_path=chunk_data.get("file_path", ""),
        line_start=chunk_data.get("line_start", 0),
        line_end=chunk_data.get("line_end", 0),
        header_path=chunk_data.get("header_path", ""),
        metadata={
            "document_id": chunk_data.get("document_id", ""),
            "file_modified_at": chunk_data.get("file_modified_at"),
        },
    )


def _mmr_dedup(results: list[SearchResult], lambda_: float) -> list[SearchResult]:
    """Maximal Marginal Relevance deduplication based on content overlap."""
    if not results:
        return []

    selected: list[SearchResult] = [results[0]]
    remaining = list(results[1:])

    while remaining:
        best_idx = -1
        best_mmr = -1.0

        for i, candidate in enumerate(remaining):
            relevance = candidate.score
            max_sim = max(_text_similarity(candidate.content, s.content) for s in selected)
            mmr_score = lambda_ * relevance - (1 - lambda_) * max_sim
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = i

        if best_idx >= 0:
            selected.append(remaining.pop(best_idx))
        else:
            break

    return selected


def _text_similarity(a: str, b: str) -> float:
    """Simple Jaccard similarity on word sets."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)
