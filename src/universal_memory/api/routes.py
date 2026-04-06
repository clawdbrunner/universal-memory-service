"""FastAPI route definitions for Universal Memory Service."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ..models import (
    EditRequest,
    IngestRequest,
    SearchRequest,
    SearchResponse,
    StatusResponse,
    WriteRequest,
    WriteResponse,
)

router = APIRouter(prefix="/api/v1")


# ------------------------------------------------------------------
# Helpers — pull services from app.state (set during startup)
# ------------------------------------------------------------------


def _state(request: Request) -> Any:
    return request.app.state


# ------------------------------------------------------------------
# Search
# ------------------------------------------------------------------


@router.post("/search", response_model=None)
async def search(request: Request, body: dict) -> dict:
    """Run the hybrid retrieval pipeline."""
    state = _state(request)
    req = SearchRequest.from_dict(body)
    response: SearchResponse = await state.pipeline.search(req)
    return response.to_dict()


# ------------------------------------------------------------------
# Write
# ------------------------------------------------------------------


@router.post("/write", response_model=None)
async def write(request: Request, body: dict) -> dict:
    """Write content to file and/or Graphiti."""
    state = _state(request)
    req = WriteRequest.from_dict(body)
    result = WriteResponse(ok=True)

    if "file" in req.targets:
        path = state.file_writer.resolve_path(
            author=req.author,
            target=req.target,
            file_path=req.file_path,
            config=state.config,
        )
        await state.file_writer.write_content(
            path=path,
            content=req.content,
            header_format=state.config.write.daily_log_header_format,
            author=req.author,
        )
        result.written_to["file"] = str(path)

        # Index the newly written file
        index_result = await state.indexer.index_file(str(path))
        result.index_updated = index_result.chunks_stored > 0
        result.index_status = "partial" if index_result.is_partial else "full"

        # Sync
        synced = await state.sync_engine.sync_file(str(path))
        result.synced_to = synced

    if "graphiti" in req.targets:
        gresult = await state.graphiti_writer.write(
            content=req.content,
            author=req.author,
        )
        if gresult:
            result.written_to["graphiti"] = True

    return result.to_dict()


# ------------------------------------------------------------------
# Read
# ------------------------------------------------------------------


@router.get("/read/{path:path}")
async def read_file(request: Request, path: str) -> dict:
    """Read a file from the data directory."""
    state = _state(request)
    from pathlib import Path

    full = Path(state.config.memory.data_dir) / path
    if not full.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    content = await state.file_writer.read_file(full)
    return {"path": path, "content": content}


# ------------------------------------------------------------------
# List
# ------------------------------------------------------------------


@router.get("/list/{namespace:path}")
async def list_files(request: Request, namespace: str) -> dict:
    """List files under a namespace."""
    state = _state(request)
    files = await state.file_writer.list_files(namespace)
    return {"namespace": namespace, "files": files}


# ------------------------------------------------------------------
# Edit
# ------------------------------------------------------------------


@router.post("/edit", response_model=None)
async def edit(request: Request, body: dict) -> dict:
    """Surgical find-and-replace in a file."""
    state = _state(request)
    req = EditRequest.from_dict(body)
    from pathlib import Path

    full = Path(state.config.memory.data_dir) / req.path
    if not full.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        await state.file_writer.edit_content(full, req.old_text, req.new_text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Re-index after edit
    index_result = await state.indexer.index_file(str(full))

    result: dict[str, Any] = {
        "ok": True,
        "path": req.path,
        "index_updated": index_result.chunks_stored > 0,
        "index_status": "partial" if index_result.is_partial else "full",
    }

    if "graphiti" in req.targets:
        await state.graphiti_writer.write(
            content=req.new_text,
            author="system",
        )
        result["graphiti_updated"] = True

    return result


# ------------------------------------------------------------------
# Ingest (Graphiti batch)
# ------------------------------------------------------------------


@router.post("/ingest", response_model=None)
async def ingest(request: Request, body: dict) -> dict:
    """Ingest messages into Graphiti."""
    state = _state(request)
    req = IngestRequest.from_dict(body)
    results = []
    for msg in req.messages:
        content = msg.get("content", "")
        author = msg.get("author", req.source)
        r = await state.graphiti_writer.write(
            content=content,
            author=author,
        )
        results.append(r)
    return {"ok": True, "ingested": len(results)}


# ------------------------------------------------------------------
# Status
# ------------------------------------------------------------------


@router.get("/status", response_model=None)
async def status(request: Request) -> dict:
    """Health check and service status."""
    state = _state(request)
    from ..db import get_stats

    stats = await get_stats()
    uptime = time.time() - state.start_time

    resp = StatusResponse(
        status="healthy",
        uptime_seconds=round(uptime, 2),
        index=stats,
        embedding_provider=state.indexer.embedding_health,
        models={
            "reranker": {
                "loaded": state.pipeline.reranker._model is not None,
                "status": state.pipeline.reranker.model_status,
                "error": state.pipeline.reranker.model_error,
                "model_path": str(state.pipeline.reranker._config.models.reranker.model_path),
            },
            "query_expander": {
                "loaded": state.pipeline.expander._model is not None,
                "status": state.pipeline.expander.model_status,
                "error": state.pipeline.expander.model_error,
                "model_path": str(state.pipeline.expander._config.models.query_expander.model_path),
            },
        },
        file_watcher={"running": state.watcher.running},
    )
    return resp.to_dict()


# ------------------------------------------------------------------
# Reindex
# ------------------------------------------------------------------


@router.post("/reindex", response_model=None)
async def reindex(request: Request) -> dict:
    """Trigger a full reindex."""
    state = _state(request)
    t0 = time.time()
    chunks = await state.indexer.reindex_all()
    elapsed = round(time.time() - t0, 2)
    return {"ok": True, "chunks_indexed": chunks, "elapsed_seconds": elapsed}
