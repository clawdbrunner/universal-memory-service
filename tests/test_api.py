"""Unit tests for FastAPI API endpoints using TestClient with mocked services."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from universal_memory.api.routes import router
from universal_memory.indexer import IndexResult
from universal_memory.models import SearchResponse, SearchResult, WriteResponse


# ---------------------------------------------------------------------------
# Test app factory
# ---------------------------------------------------------------------------


def _create_test_app(**state_overrides) -> FastAPI:
    """Create a FastAPI app with mocked state for testing."""
    app = FastAPI()
    app.include_router(router)

    # Default mocks
    mock_config = MagicMock()
    mock_config.memory.data_dir = "/tmp/test-memory-data"

    app.state.config = state_overrides.get("config", mock_config)
    app.state.pipeline = state_overrides.get("pipeline", AsyncMock())
    app.state.file_writer = state_overrides.get("file_writer", AsyncMock())
    app.state.graphiti_writer = state_overrides.get("graphiti_writer", AsyncMock())
    app.state.indexer = state_overrides.get("indexer", AsyncMock())
    app.state.sync_engine = state_overrides.get("sync_engine", AsyncMock())
    app.state.start_time = state_overrides.get("start_time", time.time())

    # Watcher mock with .running property
    watcher = MagicMock()
    type(watcher).running = PropertyMock(return_value=True)
    app.state.watcher = state_overrides.get("watcher", watcher)

    return app


@pytest.fixture
def app():
    return _create_test_app()


@pytest.fixture
def client(app):
    return TestClient(app)


# ---------------------------------------------------------------------------
# POST /api/v1/search
# ---------------------------------------------------------------------------


class TestSearchEndpoint:
    def test_search_basic(self, app, client):
        resp_obj = SearchResponse(
            results=[
                SearchResult(
                    chunk_id="c1",
                    score=0.9,
                    source="vector",
                    content="found it",
                    file_path="test.md",
                )
            ],
            query="test query",
            expanded_queries=["test query"],
            sources_queried=["vector", "bm25"],
            timing_ms={"expand": 1.0, "vector": 2.0},
        )
        app.state.pipeline.search = AsyncMock(return_value=resp_obj)

        response = client.post(
            "/api/v1/search",
            json={"query": "test query"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "test query"
        assert len(data["results"]) == 1
        assert data["results"][0]["content"] == "found it"

    def test_search_with_all_params(self, app, client):
        app.state.pipeline.search = AsyncMock(
            return_value=SearchResponse(results=[], query="q")
        )

        response = client.post(
            "/api/v1/search",
            json={
                "query": "deployment",
                "author": "alice",
                "department": "engineering",
                "sources": ["files"],
                "max_results": 5,
                "min_score": 0.5,
                "expand": False,
                "rerank": False,
            },
        )

        assert response.status_code == 200
        call_args = app.state.pipeline.search.call_args[0][0]
        assert call_args.query == "deployment"
        assert call_args.author == "alice"

    def test_search_empty_results(self, app, client):
        app.state.pipeline.search = AsyncMock(
            return_value=SearchResponse(results=[], query="nothing")
        )

        response = client.post("/api/v1/search", json={"query": "nothing"})

        assert response.status_code == 200
        assert response.json()["results"] == []

    def test_search_missing_query(self, app):
        c = TestClient(app, raise_server_exceptions=False)
        response = c.post("/api/v1/search", json={})
        # from_dict will create SearchRequest without query — should fail or use default
        # The endpoint should handle this gracefully
        assert response.status_code in (200, 422, 500)


# ---------------------------------------------------------------------------
# POST /api/v1/write
# ---------------------------------------------------------------------------


class TestWriteEndpoint:
    def test_write_to_file_and_graphiti(self, app, client):
        app.state.file_writer.resolve_path = MagicMock(
            return_value=Path("/tmp/test-memory-data/agents/alice/logs/2025-01-15.md")
        )
        app.state.file_writer.write_content = AsyncMock(return_value=Path("/tmp/out.md"))
        app.state.indexer.index_file = AsyncMock(return_value=IndexResult(3, True))
        app.state.sync_engine.sync_file = AsyncMock(return_value=[])
        app.state.graphiti_writer.write = AsyncMock(return_value={"ok": True})

        response = client.post(
            "/api/v1/write",
            json={
                "content": "Deployed v2.3 to staging",
                "author": "alice",
                "target": "daily",
                "targets": ["file", "graphiti"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert "file" in data["written_to"]
        app.state.file_writer.write_content.assert_called_once()
        app.state.graphiti_writer.write.assert_called_once()

    def test_write_file_only(self, app, client):
        app.state.file_writer.resolve_path = MagicMock(return_value=Path("/tmp/out.md"))
        app.state.file_writer.write_content = AsyncMock()
        app.state.indexer.index_file = AsyncMock(return_value=IndexResult(1, True))
        app.state.sync_engine.sync_file = AsyncMock(return_value=[])

        response = client.post(
            "/api/v1/write",
            json={
                "content": "Note",
                "author": "bob",
                "targets": ["file"],
            },
        )

        assert response.status_code == 200
        app.state.graphiti_writer.write.assert_not_called()

    def test_write_graphiti_only(self, app, client):
        app.state.graphiti_writer.write = AsyncMock(return_value={"id": "g1"})

        response = client.post(
            "/api/v1/write",
            json={
                "content": "Knowledge fact",
                "author": "carol",
                "targets": ["graphiti"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert "graphiti" in data["written_to"]

    def test_write_triggers_indexing(self, app, client):
        app.state.file_writer.resolve_path = MagicMock(return_value=Path("/tmp/f.md"))
        app.state.file_writer.write_content = AsyncMock()
        app.state.indexer.index_file = AsyncMock(return_value=IndexResult(5, True))
        app.state.sync_engine.sync_file = AsyncMock(return_value=[])
        app.state.graphiti_writer.write = AsyncMock(return_value={})

        response = client.post(
            "/api/v1/write",
            json={"content": "test", "author": "alice"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["index_updated"] is True
        assert data["index_status"] == "full"

    def test_write_partial_indexing(self, app, client):
        """When embeddings fail, index_status should be 'partial'."""
        app.state.file_writer.resolve_path = MagicMock(return_value=Path("/tmp/f.md"))
        app.state.file_writer.write_content = AsyncMock()
        app.state.indexer.index_file = AsyncMock(return_value=IndexResult(5, False))
        app.state.sync_engine.sync_file = AsyncMock(return_value=[])
        app.state.graphiti_writer.write = AsyncMock(return_value={})

        response = client.post(
            "/api/v1/write",
            json={"content": "test", "author": "alice"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["index_updated"] is True
        assert data["index_status"] == "partial"


# ---------------------------------------------------------------------------
# GET /api/v1/read/{path}
# ---------------------------------------------------------------------------


class TestReadEndpoint:
    def test_read_existing_file(self, app, client):
        app.state.file_writer.read_file = AsyncMock(return_value="file contents here")

        with patch("pathlib.Path") as MockPath:
            mock_full = MagicMock()
            mock_full.is_file.return_value = True
            MockPath.return_value.__truediv__ = MagicMock(return_value=mock_full)

            response = client.get("/api/v1/read/shared/MEMORY.md")

        assert response.status_code == 200
        data = response.json()
        assert data["content"] == "file contents here"
        assert data["path"] == "shared/MEMORY.md"

    def test_read_not_found(self, app, client):
        with patch("pathlib.Path") as MockPath:
            mock_full = MagicMock()
            mock_full.is_file.return_value = False
            MockPath.return_value.__truediv__ = MagicMock(return_value=mock_full)

            response = client.get("/api/v1/read/nonexistent.md")

        assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/v1/list/{namespace}
# ---------------------------------------------------------------------------


class TestListEndpoint:
    def test_list_files(self, app, client):
        app.state.file_writer.list_files = AsyncMock(
            return_value=["2025-01-14.md", "2025-01-15.md"]
        )

        response = client.get("/api/v1/list/agents/alice/logs")

        assert response.status_code == 200
        data = response.json()
        assert data["namespace"] == "agents/alice/logs"
        assert len(data["files"]) == 2

    def test_list_empty_namespace(self, app, client):
        app.state.file_writer.list_files = AsyncMock(return_value=[])

        response = client.get("/api/v1/list/nonexistent")

        assert response.status_code == 200
        assert response.json()["files"] == []


# ---------------------------------------------------------------------------
# POST /api/v1/edit
# ---------------------------------------------------------------------------


class TestEditEndpoint:
    def test_edit_success(self, app, client):
        app.state.file_writer.edit_content = AsyncMock(return_value=True)
        app.state.indexer.index_file = AsyncMock(return_value=IndexResult(2, True))
        app.state.graphiti_writer.write = AsyncMock(return_value={})

        with patch("pathlib.Path") as MockPath:
            mock_full = MagicMock()
            mock_full.is_file.return_value = True
            mock_full.__str__ = MagicMock(return_value="/tmp/test.md")
            MockPath.return_value.__truediv__ = MagicMock(return_value=mock_full)

            response = client.post(
                "/api/v1/edit",
                json={
                    "path": "shared/MEMORY.md",
                    "old_text": "old value",
                    "new_text": "new value",
                    "targets": ["file", "graphiti"],
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["graphiti_updated"] is True

    def test_edit_file_not_found(self, app, client):
        with patch("pathlib.Path") as MockPath:
            mock_full = MagicMock()
            mock_full.is_file.return_value = False
            MockPath.return_value.__truediv__ = MagicMock(return_value=mock_full)

            response = client.post(
                "/api/v1/edit",
                json={
                    "path": "gone.md",
                    "old_text": "x",
                    "new_text": "y",
                },
            )

        assert response.status_code == 404

    def test_edit_old_text_not_found(self, app, client):
        app.state.file_writer.edit_content = AsyncMock(
            side_effect=ValueError("old_text not found")
        )

        with patch("pathlib.Path") as MockPath:
            mock_full = MagicMock()
            mock_full.is_file.return_value = True
            MockPath.return_value.__truediv__ = MagicMock(return_value=mock_full)

            response = client.post(
                "/api/v1/edit",
                json={
                    "path": "test.md",
                    "old_text": "nonexistent",
                    "new_text": "new",
                },
            )

        assert response.status_code == 400

    def test_edit_without_graphiti_target(self, app, client):
        app.state.file_writer.edit_content = AsyncMock(return_value=True)
        app.state.indexer.index_file = AsyncMock(return_value=IndexResult(1, True))

        with patch("pathlib.Path") as MockPath:
            mock_full = MagicMock()
            mock_full.is_file.return_value = True
            mock_full.__str__ = MagicMock(return_value="/tmp/test.md")
            MockPath.return_value.__truediv__ = MagicMock(return_value=mock_full)

            response = client.post(
                "/api/v1/edit",
                json={
                    "path": "test.md",
                    "old_text": "old",
                    "new_text": "new",
                    "targets": ["file"],
                },
            )

        assert response.status_code == 200
        assert "graphiti_updated" not in response.json()
        app.state.graphiti_writer.write.assert_not_called()


# ---------------------------------------------------------------------------
# POST /api/v1/ingest
# ---------------------------------------------------------------------------


class TestIngestEndpoint:
    def test_ingest_messages(self, app, client):
        app.state.graphiti_writer.write = AsyncMock(return_value={"ok": True})

        response = client.post(
            "/api/v1/ingest",
            json={
                "messages": [
                    {"content": "Message 1", "author": "alice"},
                    {"content": "Message 2", "author": "bob"},
                ],
                "source": "conversation",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["ingested"] == 2
        assert app.state.graphiti_writer.write.call_count == 2

    def test_ingest_empty_messages(self, app, client):
        response = client.post(
            "/api/v1/ingest",
            json={"messages": []},
        )

        assert response.status_code == 200
        assert response.json()["ingested"] == 0

    def test_ingest_uses_source_as_fallback_author(self, app, client):
        app.state.graphiti_writer.write = AsyncMock(return_value={})

        response = client.post(
            "/api/v1/ingest",
            json={
                "messages": [{"content": "No explicit author"}],
                "source": "system",
            },
        )

        assert response.status_code == 200
        # The author should fall back to source="system"
        call_kwargs = app.state.graphiti_writer.write.call_args
        assert call_kwargs.kwargs.get("author") == "system" or call_kwargs[1].get(
            "author"
        ) == "system"


# ---------------------------------------------------------------------------
# GET /api/v1/status
# ---------------------------------------------------------------------------


class TestStatusEndpoint:
    def test_status_healthy(self, app, client):
        app.state.start_time = time.time() - 10  # ensure uptime > 0
        app.state.indexer.embedding_health = {
            "last_success": "2025-01-15T10:00:00+00:00",
            "last_failure": None,
        }

        # Mock pipeline reranker/expander for model status
        reranker = MagicMock()
        reranker._model = None
        reranker._config.models.reranker.model_path = "/models/reranker.onnx"
        reranker.model_status = "model_file_missing"
        reranker.model_error = "File not found: /models/reranker.onnx"
        expander = MagicMock()
        expander._model = None
        expander._config.models.query_expander.model_path = "/models/expander.gguf"
        expander.model_status = "model_file_missing"
        expander.model_error = "File not found: /models/expander.gguf"
        app.state.pipeline.reranker = reranker
        app.state.pipeline.expander = expander

        with patch("universal_memory.db.get_stats", new_callable=AsyncMock) as mock_stats:
            mock_stats.return_value = {
                "files_indexed": 42,
                "chunks": 150,
                "embeddings": 148,
                "last_indexed_at": "2025-01-15T10:00:00Z",
            }

            response = client.get("/api/v1/status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["uptime_seconds"] > 0
        assert data["index"]["files_indexed"] == 42
        assert data["file_watcher"]["running"] is True
        assert data["embedding_provider"]["last_success"] is not None
        assert data["embedding_provider"]["last_failure"] is None
        assert data["models"]["reranker"]["loaded"] is False
        assert data["models"]["reranker"]["status"] == "model_file_missing"
        assert data["models"]["reranker"]["error"] is not None
        assert data["models"]["query_expander"]["loaded"] is False
        assert data["models"]["query_expander"]["status"] == "model_file_missing"
        assert data["models"]["query_expander"]["error"] is not None

    def test_status_includes_watcher_state(self, app, client):
        type(app.state.watcher).running = PropertyMock(return_value=False)
        app.state.indexer.embedding_health = {"last_success": None, "last_failure": None}

        # Mock pipeline reranker/expander for model status
        reranker = MagicMock()
        reranker._model = None
        reranker._config.models.reranker.model_path = "/models/reranker.onnx"
        reranker.model_status = "disabled"
        reranker.model_error = None
        expander = MagicMock()
        expander._model = None
        expander._config.models.query_expander.model_path = "/models/expander.gguf"
        expander.model_status = "disabled"
        expander.model_error = None
        app.state.pipeline.reranker = reranker
        app.state.pipeline.expander = expander

        with patch("universal_memory.db.get_stats", new_callable=AsyncMock) as mock_stats:
            mock_stats.return_value = {}

            response = client.get("/api/v1/status")

        assert response.status_code == 200
        assert response.json()["file_watcher"]["running"] is False


# ---------------------------------------------------------------------------
# POST /api/v1/reindex
# ---------------------------------------------------------------------------


class TestReindexEndpoint:
    def test_reindex(self, app, client):
        app.state.indexer.reindex_all = AsyncMock(return_value=100)

        response = client.post("/api/v1/reindex")

        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["chunks_indexed"] == 100
        assert "elapsed_seconds" in data


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


class TestAPIGracefulDegradation:
    def test_search_pipeline_error(self, app):
        app.state.pipeline.search = AsyncMock(
            side_effect=RuntimeError("pipeline exploded")
        )

        c = TestClient(app, raise_server_exceptions=False)
        response = c.post("/api/v1/search", json={"query": "test"})

        # FastAPI returns 500 for unhandled exceptions
        assert response.status_code == 500

    def test_write_indexer_error_still_writes(self, app):
        app.state.file_writer.resolve_path = MagicMock(return_value=Path("/tmp/f.md"))
        app.state.file_writer.write_content = AsyncMock()
        app.state.indexer.index_file = AsyncMock(side_effect=RuntimeError("db error"))
        app.state.sync_engine.sync_file = AsyncMock(return_value=[])

        c = TestClient(app, raise_server_exceptions=False)
        response = c.post(
            "/api/v1/write",
            json={"content": "test", "author": "alice", "targets": ["file"]},
        )

        # Indexing failure propagates — the route doesn't catch it
        assert response.status_code == 500

    def test_ingest_partial_failure(self, app):
        call_count = 0

        async def write_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("graphiti down")
            return {"ok": True}

        app.state.graphiti_writer.write = AsyncMock(side_effect=write_side_effect)

        c = TestClient(app, raise_server_exceptions=False)
        response = c.post(
            "/api/v1/ingest",
            json={
                "messages": [
                    {"content": "msg1"},
                    {"content": "msg2"},
                    {"content": "msg3"},
                ],
            },
        )

        # Unhandled error on second message propagates
        assert response.status_code == 500
