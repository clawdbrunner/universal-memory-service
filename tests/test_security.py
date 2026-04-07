"""Tests for auth token enforcement and path traversal protection."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from universal_memory.api.routes import router


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_app(auth_token: str | None = None, data_dir: str = "/tmp/test-data") -> FastAPI:
    app = FastAPI()
    app.include_router(router)

    mock_config = MagicMock()
    mock_config.service.auth_token = auth_token
    mock_config.memory.data_dir = data_dir

    app.state.config = mock_config
    app.state.pipeline = AsyncMock()
    app.state.file_writer = AsyncMock()
    app.state.graphiti_writer = AsyncMock()
    app.state.indexer = AsyncMock()
    app.state.sync_engine = AsyncMock()
    app.state.start_time = time.time()

    watcher = MagicMock()
    type(watcher).running = PropertyMock(return_value=True)
    app.state.watcher = watcher

    return app


# ---------------------------------------------------------------------------
# Auth token tests
# ---------------------------------------------------------------------------


class TestAuthTokenRequired:
    def test_no_token_configured_allows_access(self):
        """When auth_token is None, requests pass without credentials."""
        app = _create_app(auth_token=None)
        app.state.pipeline.search = AsyncMock(
            return_value=MagicMock(to_dict=lambda: {"results": []})
        )
        client = TestClient(app)
        resp = client.post("/api/v1/search", json={"query": "hi"})
        assert resp.status_code == 200

    def test_empty_token_allows_access(self):
        """When auth_token is empty string, requests pass without credentials."""
        app = _create_app(auth_token="")
        app.state.pipeline.search = AsyncMock(
            return_value=MagicMock(to_dict=lambda: {"results": []})
        )
        client = TestClient(app)
        resp = client.post("/api/v1/search", json={"query": "hi"})
        assert resp.status_code == 200

    def test_missing_token_returns_401(self):
        """When auth_token is set and no header sent, return 401."""
        app = _create_app(auth_token="secret-token-123")
        client = TestClient(app)
        resp = client.post("/api/v1/search", json={"query": "hi"})
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Unauthorized"

    def test_wrong_token_returns_401(self):
        app = _create_app(auth_token="secret-token-123")
        client = TestClient(app)
        resp = client.post(
            "/api/v1/search",
            json={"query": "hi"},
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 401

    def test_correct_token_allows_access(self):
        app = _create_app(auth_token="secret-token-123")
        app.state.pipeline.search = AsyncMock(
            return_value=MagicMock(to_dict=lambda: {"results": []})
        )
        client = TestClient(app)
        resp = client.post(
            "/api/v1/search",
            json={"query": "hi"},
            headers={"Authorization": "Bearer secret-token-123"},
        )
        assert resp.status_code == 200

    def test_auth_applies_to_status_endpoint(self):
        app = _create_app(auth_token="tok")
        client = TestClient(app)
        resp = client.get("/api/v1/status")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Path traversal tests
# ---------------------------------------------------------------------------


class TestPathTraversalBlocked:
    def test_read_traversal_blocked(self, tmp_path):
        """Path with .. that escapes data_dir should be rejected."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        app = _create_app(data_dir=str(data_dir))
        client = TestClient(app)
        # URL-encode the .. segments so httpx doesn't normalize them away.
        resp = client.get(
            "/api/v1/read/agents/%2e%2e/%2e%2e/etc/passwd"
        )
        assert resp.status_code == 403
        assert "Path outside" in resp.json()["detail"]

    def test_read_within_data_dir_ok(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        app = _create_app(data_dir=str(data_dir))
        app.state.file_writer.read_file = AsyncMock(return_value="content")
        client = TestClient(app)
        # The file won't exist, so we expect 404, not 403
        resp = client.get("/api/v1/read/agents/alice/logs/2025-01-01.md")
        assert resp.status_code in (200, 404)
        assert resp.status_code != 403

    def test_edit_traversal_blocked(self):
        app = _create_app(data_dir="/tmp/test-data")
        client = TestClient(app)
        resp = client.post(
            "/api/v1/edit",
            json={
                "path": "agents/../../etc/shadow",
                "old_text": "x",
                "new_text": "y",
            },
        )
        assert resp.status_code == 403

    def test_list_traversal_blocked(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        app = _create_app(data_dir=str(data_dir))
        client = TestClient(app)
        resp = client.get(
            "/api/v1/list/agents/%2e%2e/%2e%2e/etc"
        )
        assert resp.status_code == 403
