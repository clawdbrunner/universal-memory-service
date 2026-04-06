"""Unit tests for data models."""

from __future__ import annotations

import pytest

from universal_memory.models import (
    Chunk,
    Document,
    EditRequest,
    IngestRequest,
    SearchRequest,
    SearchResponse,
    SearchResult,
    StatusResponse,
    WriteRequest,
    WriteResponse,
)


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


class TestDocument:
    def test_creation(self):
        doc = Document(path="agents/alice/logs/2025-01-15.md", content="Hello")
        assert doc.path == "agents/alice/logs/2025-01-15.md"
        assert doc.content == "Hello"
        assert len(doc.id) == 32  # uuid hex

    def test_defaults(self):
        doc = Document(path="test.md", content="")
        assert doc.metadata == {}
        assert doc.hash == ""
        assert doc.modified_at == ""
        assert doc.size_bytes == 0

    def test_to_dict(self):
        doc = Document(path="p.md", content="c", id="abc123")
        d = doc.to_dict()
        assert d["id"] == "abc123"
        assert d["path"] == "p.md"
        assert d["content"] == "c"
        assert "metadata" in d
        assert "hash" in d
        assert "size_bytes" in d

    def test_from_dict(self):
        data = {"path": "test.md", "content": "hello", "hash": "sha256", "id": "myid"}
        doc = Document.from_dict(data)
        assert doc.path == "test.md"
        assert doc.hash == "sha256"
        assert doc.id == "myid"

    def test_from_dict_ignores_unknown(self):
        data = {"path": "t.md", "content": "c", "unknown_field": 42}
        doc = Document.from_dict(data)
        assert doc.path == "t.md"
        assert not hasattr(doc, "unknown_field")

    def test_roundtrip(self):
        doc = Document(path="r.md", content="roundtrip", metadata={"key": "val"})
        d = doc.to_dict()
        doc2 = Document.from_dict(d)
        assert doc2.path == doc.path
        assert doc2.content == doc.content
        assert doc2.metadata == doc.metadata


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


class TestChunk:
    def test_creation(self):
        chunk = Chunk(
            document_id="doc1",
            file_path="test.md",
            line_start=1,
            line_end=10,
            content="Some content",
        )
        assert chunk.document_id == "doc1"
        assert chunk.line_start == 1
        assert chunk.line_end == 10
        assert len(chunk.id) == 32

    def test_defaults(self):
        chunk = Chunk(document_id="d", file_path="f", line_start=1, line_end=1, content="c")
        assert chunk.header_path == ""
        assert chunk.token_count == 0
        assert chunk.embedding is None
        assert chunk.embedding_hash == ""

    def test_to_dict_excludes_embedding(self):
        chunk = Chunk(
            document_id="d", file_path="f", line_start=1, line_end=1,
            content="c", embedding=[0.1, 0.2],
        )
        d = chunk.to_dict()
        assert "embedding" not in d  # embedding excluded from dict
        assert d["content"] == "c"

    def test_from_dict(self):
        data = {
            "document_id": "doc1", "file_path": "f.md",
            "line_start": 5, "line_end": 10, "content": "text",
            "header_path": "Section A", "id": "chunk1",
        }
        chunk = Chunk.from_dict(data)
        assert chunk.id == "chunk1"
        assert chunk.header_path == "Section A"


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------


class TestSearchResult:
    def test_creation(self):
        sr = SearchResult(chunk_id="c1", score=0.95, source="vector", content="match")
        assert sr.score == 0.95
        assert sr.source == "vector"

    def test_defaults(self):
        sr = SearchResult(chunk_id="c1", score=0.5, source="bm25", content="x")
        assert sr.file_path == ""
        assert sr.line_start == 0
        assert sr.line_end == 0
        assert sr.header_path == ""
        assert sr.metadata == {}

    def test_to_dict(self):
        sr = SearchResult(
            chunk_id="c1", score=0.88, source="merged",
            content="text", file_path="test.md",
            line_start=5, line_end=10, header_path="H",
            metadata={"key": "val"},
        )
        d = sr.to_dict()
        assert d["chunk_id"] == "c1"
        assert d["score"] == 0.88
        assert d["source"] == "merged"
        assert d["metadata"] == {"key": "val"}

    def test_all_sources_valid(self):
        for source in ("vector", "bm25", "graphiti", "merged"):
            sr = SearchResult(chunk_id="c", score=0.5, source=source, content="x")
            assert sr.source == source


# ---------------------------------------------------------------------------
# SearchRequest
# ---------------------------------------------------------------------------


class TestSearchRequest:
    def test_defaults(self):
        req = SearchRequest(query="test")
        assert req.query == "test"
        assert req.author is None
        assert req.department is None
        assert req.sources == ["files", "graphiti"]
        assert req.max_results == 10
        assert req.min_score == 0.3
        assert req.temporal_filter is None
        assert req.expand is True
        assert req.rerank is True

    def test_from_dict(self):
        data = {
            "query": "electric bill",
            "author": "alice",
            "max_results": 5,
            "expand": False,
        }
        req = SearchRequest.from_dict(data)
        assert req.query == "electric bill"
        assert req.author == "alice"
        assert req.max_results == 5
        assert req.expand is False
        assert req.rerank is True  # default

    def test_from_dict_ignores_unknown(self):
        data = {"query": "test", "nonexistent": True}
        req = SearchRequest.from_dict(data)
        assert req.query == "test"

    def test_temporal_filter(self):
        req = SearchRequest(
            query="q",
            temporal_filter={"after": "2025-01-01", "before": "2025-02-01"},
        )
        assert req.temporal_filter["after"] == "2025-01-01"


# ---------------------------------------------------------------------------
# SearchResponse
# ---------------------------------------------------------------------------


class TestSearchResponse:
    def test_empty_response(self):
        resp = SearchResponse(results=[], query="test")
        assert resp.results == []
        assert resp.query == "test"
        assert resp.timing_ms == {}

    def test_expansion_status_default(self):
        resp = SearchResponse(results=[], query="test")
        assert resp.expansion_status == "success"

    def test_expansion_status_in_dict(self):
        resp = SearchResponse(results=[], query="q", expansion_status="skipped_pattern")
        d = resp.to_dict()
        assert d["expansion_status"] == "skipped_pattern"

    def test_timing_ms_field(self):
        resp = SearchResponse(
            results=[],
            query="q",
            timing_ms={"expansion": 82, "vector": 45, "total": 537},
        )
        assert resp.timing_ms["expansion"] == 82
        assert resp.timing_ms["total"] == 537

    def test_to_dict(self):
        sr = SearchResult(chunk_id="c1", score=0.9, source="vector", content="text")
        resp = SearchResponse(
            results=[sr],
            query="q",
            scope={"author": "alice"},
            expanded_queries=["q alt"],
            sources_queried=["files"],
            timing_ms={"total": 100},
        )
        d = resp.to_dict()
        assert len(d["results"]) == 1
        assert d["results"][0]["chunk_id"] == "c1"
        assert d["query"] == "q"
        assert d["scope"] == {"author": "alice"}
        assert d["expanded_queries"] == ["q alt"]
        assert d["sources_queried"] == ["files"]
        assert d["timing_ms"] == {"total": 100}
        assert d["expansion_status"] == "success"

    def test_multiple_results(self):
        results = [
            SearchResult(chunk_id=f"c{i}", score=1.0 - i * 0.1, source="merged", content=f"r{i}")
            for i in range(5)
        ]
        resp = SearchResponse(results=results, query="q")
        d = resp.to_dict()
        assert len(d["results"]) == 5
        assert d["results"][0]["score"] == 1.0
        assert d["results"][4]["score"] == pytest.approx(0.6, abs=0.01)


# ---------------------------------------------------------------------------
# WriteRequest
# ---------------------------------------------------------------------------


class TestWriteRequest:
    def test_defaults(self):
        req = WriteRequest(content="note", author="alice")
        assert req.content == "note"
        assert req.author == "alice"
        assert req.department is None
        assert req.role == "assistant"
        assert req.target == "daily"
        assert req.file_path is None
        assert req.targets == ["file", "graphiti"]
        assert req.timestamp is None

    def test_from_dict(self):
        data = {
            "content": "hello",
            "author": "bob",
            "department": "engineering",
            "target": "department",
        }
        req = WriteRequest.from_dict(data)
        assert req.department == "engineering"
        assert req.target == "department"

    def test_custom_targets(self):
        req = WriteRequest(content="x", author="a", targets=["file"])
        assert req.targets == ["file"]


# ---------------------------------------------------------------------------
# WriteResponse
# ---------------------------------------------------------------------------


class TestWriteResponse:
    def test_defaults(self):
        resp = WriteResponse(ok=True)
        assert resp.ok is True
        assert resp.written_to == {}
        assert resp.synced_to == []
        assert resp.index_updated is False

    def test_to_dict(self):
        resp = WriteResponse(
            ok=True,
            written_to={"file": "agents/alice/logs/2025-01-15.md"},
            synced_to=["openclaw:alice"],
            index_updated=True,
        )
        d = resp.to_dict()
        assert d["ok"] is True
        assert d["written_to"]["file"] == "agents/alice/logs/2025-01-15.md"
        assert d["synced_to"] == ["openclaw:alice"]
        assert d["index_updated"] is True


# ---------------------------------------------------------------------------
# IngestRequest
# ---------------------------------------------------------------------------


class TestIngestRequest:
    def test_defaults(self):
        req = IngestRequest(messages=[{"content": "hi"}])
        assert len(req.messages) == 1
        assert req.group_id == ""
        assert req.source == ""
        assert req.targets == ["graphiti"]

    def test_from_dict(self):
        data = {
            "messages": [{"content": "a"}, {"content": "b"}],
            "group_id": "memory-alice",
            "source": "openclaw",
        }
        req = IngestRequest.from_dict(data)
        assert len(req.messages) == 2
        assert req.group_id == "memory-alice"


# ---------------------------------------------------------------------------
# EditRequest
# ---------------------------------------------------------------------------


class TestEditRequest:
    def test_creation(self):
        req = EditRequest(path="shared/MEMORY.md", old_text="old", new_text="new")
        assert req.path == "shared/MEMORY.md"
        assert req.targets == ["file", "graphiti"]

    def test_from_dict(self):
        data = {"path": "p.md", "old_text": "a", "new_text": "b", "targets": ["file"]}
        req = EditRequest.from_dict(data)
        assert req.targets == ["file"]


# ---------------------------------------------------------------------------
# StatusResponse
# ---------------------------------------------------------------------------


class TestStatusResponse:
    def test_defaults(self):
        resp = StatusResponse()
        assert resp.status == "healthy"
        assert resp.uptime_seconds == 0.0
        assert resp.index == {}
        assert resp.models == {}
        assert resp.graphiti == {}
        assert resp.embedding_provider == {}
        assert resp.file_watcher == {}

    def test_to_dict(self):
        resp = StatusResponse(
            status="healthy",
            uptime_seconds=3600.5,
            index={"files_indexed": 52, "chunks": 847},
            file_watcher={"running": True},
        )
        d = resp.to_dict()
        assert d["status"] == "healthy"
        assert d["uptime_seconds"] == 3600.5
        assert d["index"]["files_indexed"] == 52
        assert d["file_watcher"]["running"] is True

    def test_all_fields_in_dict(self):
        resp = StatusResponse()
        d = resp.to_dict()
        expected_keys = {
            "status", "uptime_seconds", "index", "models",
            "graphiti", "embedding_provider", "file_watcher",
        }
        assert set(d.keys()) == expected_keys
