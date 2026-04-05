"""Comprehensive integration tests for the 6-stage retrieval pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from universal_memory.models import SearchRequest, SearchResponse, SearchResult
from universal_memory.retrieval.pipeline import (
    RetrievalPipeline,
    _chunk_to_result,
    _mmr_dedup,
    _normalize_scores,
    _text_similarity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_search_result(
    chunk_id="c1",
    score=0.8,
    source="vector",
    content="test content",
    file_path="test.md",
):
    return SearchResult(
        chunk_id=chunk_id,
        score=score,
        source=source,
        content=content,
        file_path=file_path,
    )


def _make_chunk_data(chunk_id="c1", content="test content", file_path="test.md"):
    return {
        "id": chunk_id,
        "content": content,
        "file_path": file_path,
        "line_start": 1,
        "line_end": 5,
        "header_path": "Test",
        "document_id": "doc1",
        "file_modified_at": None,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.search_weights.vector = 0.40
    cfg.search_weights.bm25 = 0.20
    cfg.search_weights.graphiti = 0.25
    cfg.models.reranker.candidates = 30
    return cfg


@pytest.fixture
def pipeline(mock_config):
    """Create a RetrievalPipeline with every sub-service mocked out."""
    with patch(
        "universal_memory.retrieval.pipeline.get_config", return_value=mock_config
    ):
        p = RetrievalPipeline()
    p.embeddings = AsyncMock()
    p.vector_store = AsyncMock()
    p.bm25 = AsyncMock()
    p.graphiti = AsyncMock()
    p.reranker = AsyncMock()
    p.expander = AsyncMock()
    return p


# ---------------------------------------------------------------------------
# _normalize_scores
# ---------------------------------------------------------------------------


class TestNormalizeScores:
    def test_empty(self):
        assert _normalize_scores([]) == []

    def test_single_item(self):
        result = _normalize_scores([("c1", 5.0)])
        assert len(result) == 1
        assert result[0][0] == "c1"
        # (5 - 5) / 1.0 = 0.0
        assert result[0][1] == pytest.approx(0.0)

    def test_two_items(self):
        result = _normalize_scores([("c1", 1.0), ("c2", 3.0)])
        scores = {cid: s for cid, s in result}
        assert scores["c1"] == pytest.approx(0.0)
        assert scores["c2"] == pytest.approx(1.0)

    def test_deduplicates_keeping_highest(self):
        result = _normalize_scores([("c1", 1.0), ("c1", 5.0), ("c2", 3.0)])
        ids = [cid for cid, _ in result]
        assert ids.count("c1") == 1
        assert len(result) == 2

    def test_all_same_score(self):
        result = _normalize_scores([("c1", 5.0), ("c2", 5.0)])
        for _, score in result:
            assert score == pytest.approx(0.0)

    def test_negative_scores(self):
        result = _normalize_scores([("c1", -3.0), ("c2", -1.0)])
        scores = {cid: s for cid, s in result}
        assert scores["c1"] == pytest.approx(0.0)
        assert scores["c2"] == pytest.approx(1.0)

    def test_many_items(self):
        pairs = [(f"c{i}", float(i)) for i in range(10)]
        result = _normalize_scores(pairs)
        assert len(result) == 10
        scores = {cid: s for cid, s in result}
        assert scores["c0"] == pytest.approx(0.0)
        assert scores["c9"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _chunk_to_result
# ---------------------------------------------------------------------------


class TestChunkToResult:
    def test_basic_conversion(self):
        data = _make_chunk_data()
        sr = _chunk_to_result(data, 0.9, "vector")
        assert sr.chunk_id == "c1"
        assert sr.score == 0.9
        assert sr.source == "vector"
        assert sr.content == "test content"
        assert sr.file_path == "test.md"
        assert sr.line_start == 1
        assert sr.line_end == 5
        assert sr.header_path == "Test"
        assert sr.metadata["document_id"] == "doc1"

    def test_missing_optional_fields(self):
        data = {"id": "x", "content": "hello"}
        sr = _chunk_to_result(data, 0.5, "bm25")
        assert sr.file_path == ""
        assert sr.line_start == 0
        assert sr.line_end == 0

    def test_different_sources(self):
        for source in ("vector", "bm25", "graphiti"):
            sr = _chunk_to_result(_make_chunk_data(), 0.5, source)
            assert sr.source == source


# ---------------------------------------------------------------------------
# _text_similarity
# ---------------------------------------------------------------------------


class TestTextSimilarity:
    def test_identical(self):
        assert _text_similarity("hello world", "hello world") == pytest.approx(1.0)

    def test_disjoint(self):
        assert _text_similarity("hello world", "foo bar") == pytest.approx(0.0)

    def test_partial_overlap(self):
        sim = _text_similarity("the quick brown fox", "the lazy brown dog")
        assert 0.0 < sim < 1.0

    def test_empty_strings(self):
        assert _text_similarity("", "hello") == 0.0
        assert _text_similarity("hello", "") == 0.0
        assert _text_similarity("", "") == 0.0

    def test_case_insensitive(self):
        assert _text_similarity("Hello World", "hello world") == pytest.approx(1.0)

    def test_single_word_overlap(self):
        sim = _text_similarity("apple banana cherry", "apple orange grape")
        # Jaccard: {apple} / {apple, banana, cherry, orange, grape} = 1/5
        assert sim == pytest.approx(1.0 / 5.0)


# ---------------------------------------------------------------------------
# _mmr_dedup
# ---------------------------------------------------------------------------


class TestMMRDedup:
    def test_empty(self):
        assert _mmr_dedup([], 0.7) == []

    def test_single_result(self):
        r = _make_search_result()
        result = _mmr_dedup([r], 0.7)
        assert len(result) == 1
        assert result[0] is r

    def test_preserves_diverse_results(self):
        results = [
            _make_search_result("c1", 0.9, content="deployment pipeline broken"),
            _make_search_result("c2", 0.8, content="auth middleware updated"),
            _make_search_result("c3", 0.7, content="database schema migration"),
        ]
        deduped = _mmr_dedup(results, 0.7)
        assert len(deduped) == 3

    def test_first_result_always_highest(self):
        results = [
            _make_search_result("c1", 0.9, content="deploy service staging"),
            _make_search_result("c2", 0.88, content="deploy service production"),
            _make_search_result("c3", 0.7, content="database migration done"),
        ]
        deduped = _mmr_dedup(results, 0.7)
        assert deduped[0].chunk_id == "c1"

    def test_all_identical_content(self):
        results = [
            _make_search_result(f"c{i}", 0.9 - i * 0.1, content="same content here")
            for i in range(5)
        ]
        deduped = _mmr_dedup(results, 0.7)
        assert len(deduped) == 5
        assert deduped[0].chunk_id == "c0"

    def test_lambda_zero_favors_diversity(self):
        results = [
            _make_search_result("c1", 0.95, content="the cat sat on the mat"),
            _make_search_result("c2", 0.90, content="the cat sat on the rug"),
            _make_search_result("c3", 0.50, content="dogs play in the park"),
        ]
        deduped = _mmr_dedup(results, 0.0)
        # With lambda=0, diversity is maximized — c3 (most different) should be picked 2nd
        assert deduped[0].chunk_id == "c1"
        assert deduped[1].chunk_id == "c3"


# ---------------------------------------------------------------------------
# Full pipeline search — basic
# ---------------------------------------------------------------------------


class TestPipelineSearch:
    @pytest.mark.asyncio
    async def test_basic_search(self, pipeline):
        pipeline.expander.expand = AsyncMock(return_value=["test query"])
        pipeline.embeddings.generate = AsyncMock(return_value=[[0.1, 0.2]])
        pipeline.vector_store.search = AsyncMock(return_value=[("c1", 0.9)])
        pipeline.bm25.search = AsyncMock(return_value=[("c1", 0.8)])
        pipeline.graphiti.search = AsyncMock(return_value=[])
        pipeline.reranker.rerank = AsyncMock(
            side_effect=lambda q, candidates, top_n: candidates[:top_n]
        )

        with patch(
            "universal_memory.retrieval.pipeline.get_chunk",
            new_callable=AsyncMock,
            return_value=_make_chunk_data(),
        ):
            req = SearchRequest(query="test query")
            resp = await pipeline.search(req)

        assert isinstance(resp, SearchResponse)
        assert resp.query == "test query"

    @pytest.mark.asyncio
    async def test_expansion_disabled(self, pipeline):
        pipeline.expander.expand = AsyncMock()
        pipeline.embeddings.generate = AsyncMock(return_value=[[0.1]])
        pipeline.vector_store.search = AsyncMock(return_value=[])
        pipeline.bm25.search = AsyncMock(return_value=[])
        pipeline.graphiti.search = AsyncMock(return_value=[])
        pipeline.reranker.rerank = AsyncMock(return_value=[])

        req = SearchRequest(query="test", expand=False)
        resp = await pipeline.search(req)

        pipeline.expander.expand.assert_not_called()
        assert resp.expanded_queries == ["test"]

    @pytest.mark.asyncio
    async def test_rerank_disabled(self, pipeline):
        pipeline.expander.expand = AsyncMock(return_value=["q"])
        pipeline.embeddings.generate = AsyncMock(return_value=[[0.1]])
        pipeline.vector_store.search = AsyncMock(return_value=[("c1", 0.9)])
        pipeline.bm25.search = AsyncMock(return_value=[])
        pipeline.graphiti.search = AsyncMock(return_value=[])

        with patch(
            "universal_memory.retrieval.pipeline.get_chunk",
            new_callable=AsyncMock,
            return_value=_make_chunk_data(),
        ):
            req = SearchRequest(query="q", rerank=False, min_score=0.0)
            await pipeline.search(req)

        pipeline.reranker.rerank.assert_not_called()

    @pytest.mark.asyncio
    async def test_min_score_filters_low_results(self, pipeline):
        pipeline.expander.expand = AsyncMock(return_value=["q"])
        pipeline.embeddings.generate = AsyncMock(return_value=[[0.1]])
        pipeline.vector_store.search = AsyncMock(return_value=[("c1", 0.05)])
        pipeline.bm25.search = AsyncMock(return_value=[])
        pipeline.graphiti.search = AsyncMock(return_value=[])
        pipeline.reranker.rerank = AsyncMock(
            side_effect=lambda q, candidates, top_n: candidates[:top_n]
        )

        with patch(
            "universal_memory.retrieval.pipeline.get_chunk",
            new_callable=AsyncMock,
            return_value=_make_chunk_data(),
        ):
            req = SearchRequest(query="q", min_score=0.9)
            resp = await pipeline.search(req)

        assert all(r.score >= 0.9 for r in resp.results)

    @pytest.mark.asyncio
    async def test_sources_only_files(self, pipeline):
        pipeline.expander.expand = AsyncMock(return_value=["q"])
        pipeline.embeddings.generate = AsyncMock(return_value=[[0.1]])
        pipeline.vector_store.search = AsyncMock(return_value=[])
        pipeline.bm25.search = AsyncMock(return_value=[])
        pipeline.graphiti.search = AsyncMock(return_value=[])
        pipeline.reranker.rerank = AsyncMock(return_value=[])

        req = SearchRequest(query="q", sources=["files"])
        resp = await pipeline.search(req)

        pipeline.graphiti.search.assert_not_called()
        assert "graphiti" not in resp.sources_queried

    @pytest.mark.asyncio
    async def test_sources_only_graphiti(self, pipeline):
        pipeline.expander.expand = AsyncMock(return_value=["q"])
        pipeline.graphiti.search = AsyncMock(
            return_value=[_make_search_result(source="graphiti")]
        )
        pipeline.reranker.rerank = AsyncMock(
            side_effect=lambda q, candidates, top_n: candidates[:top_n]
        )

        req = SearchRequest(query="q", sources=["graphiti"], min_score=0.0)
        resp = await pipeline.search(req)

        pipeline.vector_store.search.assert_not_called()
        pipeline.bm25.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_timing_keys_always_present(self, pipeline):
        pipeline.expander.expand = AsyncMock(return_value=["q"])
        pipeline.embeddings.generate = AsyncMock(return_value=[])
        pipeline.vector_store.search = AsyncMock(return_value=[])
        pipeline.bm25.search = AsyncMock(return_value=[])
        pipeline.graphiti.search = AsyncMock(return_value=[])
        pipeline.reranker.rerank = AsyncMock(return_value=[])

        resp = await pipeline.search(SearchRequest(query="q"))

        for key in ("expand", "vector", "bm25", "graphiti", "merge", "rerank"):
            assert key in resp.timing_ms
            assert resp.timing_ms[key] >= 0

    @pytest.mark.asyncio
    async def test_department_passed_to_graphiti(self, pipeline):
        pipeline.expander.expand = AsyncMock(return_value=["q"])
        pipeline.embeddings.generate = AsyncMock(return_value=[])
        pipeline.vector_store.search = AsyncMock(return_value=[])
        pipeline.bm25.search = AsyncMock(return_value=[])
        pipeline.graphiti.search = AsyncMock(return_value=[])
        pipeline.reranker.rerank = AsyncMock(return_value=[])

        req = SearchRequest(query="q", department="engineering")
        await pipeline.search(req)

        pipeline.graphiti.search.assert_called_once_with(
            "q", group_ids=["engineering"], limit=10
        )


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    @pytest.mark.asyncio
    async def test_expand_failure(self, pipeline):
        pipeline.expander.expand = AsyncMock(side_effect=RuntimeError("boom"))
        pipeline.embeddings.generate = AsyncMock(return_value=[])
        pipeline.vector_store.search = AsyncMock(return_value=[])
        pipeline.bm25.search = AsyncMock(return_value=[])
        pipeline.graphiti.search = AsyncMock(return_value=[])
        pipeline.reranker.rerank = AsyncMock(return_value=[])

        resp = await pipeline.search(SearchRequest(query="test"))

        assert isinstance(resp, SearchResponse)
        assert resp.expanded_queries == ["test"]

    @pytest.mark.asyncio
    async def test_vector_failure(self, pipeline):
        pipeline.expander.expand = AsyncMock(return_value=["q"])
        pipeline.embeddings.generate = AsyncMock(
            side_effect=RuntimeError("embedding crash")
        )
        pipeline.bm25.search = AsyncMock(return_value=[("c1", 0.8)])
        pipeline.graphiti.search = AsyncMock(return_value=[])
        pipeline.reranker.rerank = AsyncMock(
            side_effect=lambda q, candidates, top_n: candidates[:top_n]
        )

        with patch(
            "universal_memory.retrieval.pipeline.get_chunk",
            new_callable=AsyncMock,
            return_value=_make_chunk_data(),
        ):
            resp = await pipeline.search(
                SearchRequest(query="q", min_score=0.0)
            )

        assert isinstance(resp, SearchResponse)

    @pytest.mark.asyncio
    async def test_bm25_failure(self, pipeline):
        pipeline.expander.expand = AsyncMock(return_value=["q"])
        pipeline.embeddings.generate = AsyncMock(return_value=[[0.1]])
        pipeline.vector_store.search = AsyncMock(return_value=[("c1", 0.9)])
        pipeline.bm25.search = AsyncMock(side_effect=RuntimeError("fts5 broken"))
        pipeline.graphiti.search = AsyncMock(return_value=[])
        pipeline.reranker.rerank = AsyncMock(
            side_effect=lambda q, candidates, top_n: candidates[:top_n]
        )

        with patch(
            "universal_memory.retrieval.pipeline.get_chunk",
            new_callable=AsyncMock,
            return_value=_make_chunk_data(),
        ):
            resp = await pipeline.search(
                SearchRequest(query="q", min_score=0.0)
            )

        assert isinstance(resp, SearchResponse)

    @pytest.mark.asyncio
    async def test_graphiti_failure(self, pipeline):
        pipeline.expander.expand = AsyncMock(return_value=["q"])
        pipeline.embeddings.generate = AsyncMock(return_value=[])
        pipeline.vector_store.search = AsyncMock(return_value=[])
        pipeline.bm25.search = AsyncMock(return_value=[])
        pipeline.graphiti.search = AsyncMock(side_effect=RuntimeError("down"))
        pipeline.reranker.rerank = AsyncMock(return_value=[])

        resp = await pipeline.search(SearchRequest(query="q"))

        assert isinstance(resp, SearchResponse)

    @pytest.mark.asyncio
    async def test_rerank_failure_returns_unranked(self, pipeline):
        pipeline.expander.expand = AsyncMock(return_value=["q"])
        pipeline.embeddings.generate = AsyncMock(return_value=[[0.1]])
        pipeline.vector_store.search = AsyncMock(return_value=[("c1", 0.9)])
        pipeline.bm25.search = AsyncMock(return_value=[])
        pipeline.graphiti.search = AsyncMock(return_value=[])
        pipeline.reranker.rerank = AsyncMock(side_effect=RuntimeError("model fail"))

        with patch(
            "universal_memory.retrieval.pipeline.get_chunk",
            new_callable=AsyncMock,
            return_value=_make_chunk_data(),
        ):
            resp = await pipeline.search(
                SearchRequest(query="q", min_score=0.0)
            )

        assert isinstance(resp, SearchResponse)

    @pytest.mark.asyncio
    async def test_merge_failure(self, pipeline):
        pipeline.expander.expand = AsyncMock(return_value=["q"])
        pipeline.embeddings.generate = AsyncMock(return_value=[[0.1]])
        pipeline.vector_store.search = AsyncMock(return_value=[("c1", 0.9)])
        pipeline.bm25.search = AsyncMock(return_value=[])
        pipeline.graphiti.search = AsyncMock(return_value=[])
        pipeline.reranker.rerank = AsyncMock(return_value=[])
        pipeline._merge = AsyncMock(side_effect=RuntimeError("merge exploded"))

        resp = await pipeline.search(SearchRequest(query="q"))

        assert isinstance(resp, SearchResponse)
        assert resp.results == []

    @pytest.mark.asyncio
    async def test_all_stages_fail(self, pipeline):
        pipeline.expander.expand = AsyncMock(side_effect=RuntimeError("1"))
        pipeline.embeddings.generate = AsyncMock(side_effect=RuntimeError("2"))
        pipeline.bm25.search = AsyncMock(side_effect=RuntimeError("3"))
        pipeline.graphiti.search = AsyncMock(side_effect=RuntimeError("4"))
        pipeline.reranker.rerank = AsyncMock(side_effect=RuntimeError("5"))

        resp = await pipeline.search(SearchRequest(query="anything"))

        assert isinstance(resp, SearchResponse)
        assert resp.results == []


# ---------------------------------------------------------------------------
# Embedding stage
# ---------------------------------------------------------------------------


class TestEmbeddingStage:
    @pytest.mark.asyncio
    async def test_multiple_queries_each_embedded(self, pipeline):
        pipeline.expander.expand = AsyncMock(return_value=["q1", "q2", "q3"])
        pipeline.embeddings.generate = AsyncMock(return_value=[[0.1, 0.2]])
        pipeline.vector_store.search = AsyncMock(return_value=[])
        pipeline.bm25.search = AsyncMock(return_value=[])
        pipeline.graphiti.search = AsyncMock(return_value=[])
        pipeline.reranker.rerank = AsyncMock(return_value=[])

        await pipeline.search(SearchRequest(query="q1"))

        assert pipeline.embeddings.generate.call_count == 3

    @pytest.mark.asyncio
    async def test_empty_embedding_skips_vector_search(self, pipeline):
        pipeline.expander.expand = AsyncMock(return_value=["q"])
        pipeline.embeddings.generate = AsyncMock(return_value=[])
        pipeline.vector_store.search = AsyncMock(return_value=[])
        pipeline.bm25.search = AsyncMock(return_value=[])
        pipeline.graphiti.search = AsyncMock(return_value=[])
        pipeline.reranker.rerank = AsyncMock(return_value=[])

        await pipeline.search(SearchRequest(query="q"))

        pipeline.vector_store.search.assert_not_called()


# ---------------------------------------------------------------------------
# Merge stage
# ---------------------------------------------------------------------------


class TestMergeStage:
    @pytest.mark.asyncio
    async def test_vector_and_bm25_scores_combined(self, pipeline):
        pipeline.expander.expand = AsyncMock(return_value=["q"])
        pipeline.embeddings.generate = AsyncMock(return_value=[[0.1]])
        pipeline.vector_store.search = AsyncMock(return_value=[("c1", 0.9)])
        pipeline.bm25.search = AsyncMock(return_value=[("c1", 0.8)])
        pipeline.graphiti.search = AsyncMock(return_value=[])
        pipeline.reranker.rerank = AsyncMock(
            side_effect=lambda q, candidates, top_n: candidates[:top_n]
        )

        with patch(
            "universal_memory.retrieval.pipeline.get_chunk",
            new_callable=AsyncMock,
            return_value=_make_chunk_data(),
        ):
            resp = await pipeline.search(
                SearchRequest(query="q", min_score=0.0)
            )

        # Chunk c1 appears in both vector and bm25 — should merge into one result
        c1_results = [r for r in resp.results if r.chunk_id == "c1"]
        assert len(c1_results) <= 1

    @pytest.mark.asyncio
    async def test_graphiti_results_included(self, pipeline):
        pipeline.expander.expand = AsyncMock(return_value=["q"])
        pipeline.embeddings.generate = AsyncMock(return_value=[])
        pipeline.vector_store.search = AsyncMock(return_value=[])
        pipeline.bm25.search = AsyncMock(return_value=[])
        pipeline.graphiti.search = AsyncMock(
            return_value=[
                _make_search_result("g1", 0.85, "graphiti", "knowledge graph fact")
            ]
        )
        pipeline.reranker.rerank = AsyncMock(
            side_effect=lambda q, candidates, top_n: candidates[:top_n]
        )

        resp = await pipeline.search(
            SearchRequest(query="q", min_score=0.0)
        )

        assert any("knowledge graph" in r.content for r in resp.results)

    @pytest.mark.asyncio
    async def test_no_results_from_any_source(self, pipeline):
        pipeline.expander.expand = AsyncMock(return_value=["q"])
        pipeline.embeddings.generate = AsyncMock(return_value=[])
        pipeline.vector_store.search = AsyncMock(return_value=[])
        pipeline.bm25.search = AsyncMock(return_value=[])
        pipeline.graphiti.search = AsyncMock(return_value=[])
        pipeline.reranker.rerank = AsyncMock(return_value=[])

        resp = await pipeline.search(SearchRequest(query="q"))

        assert resp.results == []
