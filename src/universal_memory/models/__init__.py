"""Data models for Universal Memory Service."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def _new_id() -> str:
    return uuid.uuid4().hex


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


# ---------------------------------------------------------------------------
# Core data
# ---------------------------------------------------------------------------


@dataclass
class Document:
    path: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    hash: str = ""
    modified_at: str = ""
    size_bytes: int = 0
    id: str = field(default_factory=_new_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "content": self.content,
            "metadata": self.metadata,
            "hash": self.hash,
            "modified_at": self.modified_at,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Document:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Chunk:
    document_id: str
    file_path: str
    line_start: int
    line_end: int
    content: str
    header_path: str = ""
    token_count: int = 0
    embedding: list[float] | None = None
    embedding_hash: str = ""
    id: str = field(default_factory=_new_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "content": self.content,
            "header_path": self.header_path,
            "token_count": self.token_count,
            "embedding_hash": self.embedding_hash,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Chunk:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SearchResult:
    chunk_id: str
    score: float
    source: str  # "vector" | "bm25" | "graphiti" | "merged"
    content: str
    file_path: str = ""
    line_start: int = 0
    line_end: int = 0
    header_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "source": self.source,
            "content": self.content,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "header_path": self.header_path,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# API request / response
# ---------------------------------------------------------------------------


@dataclass
class WriteRequest:
    content: str
    author: str
    department: str | None = None
    role: str = "assistant"
    target: str = "daily"
    file_path: str | None = None
    targets: list[str] = field(default_factory=lambda: ["file", "graphiti"])
    timestamp: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WriteRequest:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SearchRequest:
    query: str
    author: str | None = None
    department: str | None = None
    sources: list[str] = field(default_factory=lambda: ["files", "graphiti"])
    max_results: int = 10
    min_score: float = 0.5
    temporal_filter: dict[str, str] | None = None
    expand: bool = True
    rerank: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SearchRequest:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SearchResponse:
    results: list[SearchResult]
    query: str
    scope: dict[str, Any] = field(default_factory=dict)
    expanded_queries: list[str] = field(default_factory=list)
    sources_queried: list[str] = field(default_factory=list)
    timing_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "query": self.query,
            "scope": self.scope,
            "expanded_queries": self.expanded_queries,
            "sources_queried": self.sources_queried,
            "timing_ms": self.timing_ms,
        }


@dataclass
class WriteResponse:
    ok: bool
    written_to: dict[str, Any] = field(default_factory=dict)
    synced_to: list[str] = field(default_factory=list)
    index_updated: bool = False
    index_status: str = "full"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "written_to": self.written_to,
            "synced_to": self.synced_to,
            "index_updated": self.index_updated,
            "index_status": self.index_status,
        }


@dataclass
class IngestRequest:
    messages: list[dict[str, Any]]
    group_id: str = ""
    source: str = ""
    session_id: str = ""
    targets: list[str] = field(default_factory=lambda: ["graphiti"])

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IngestRequest:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class EditRequest:
    path: str
    old_text: str
    new_text: str
    targets: list[str] = field(default_factory=lambda: ["file", "graphiti"])

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EditRequest:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class StatusResponse:
    status: str = "healthy"
    uptime_seconds: float = 0.0
    index: dict[str, Any] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)
    graphiti: dict[str, Any] = field(default_factory=dict)
    embedding_provider: dict[str, Any] = field(default_factory=dict)
    file_watcher: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "uptime_seconds": self.uptime_seconds,
            "index": self.index,
            "models": self.models,
            "graphiti": self.graphiti,
            "embedding_provider": self.embedding_provider,
            "file_watcher": self.file_watcher,
        }
