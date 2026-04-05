"""MCP stdio transport server for Universal Memory Service.

Exposes the core memory operations as MCP tools so that Claude Desktop,
Cursor, and any MCP-compatible client can interact with the service.

Usage:
    python -m universal_memory.mcp_server
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .config import get_config, load_config
from .db import get_stats, init_db
from .indexer import Indexer
from .models import EditRequest, SearchRequest, WriteRequest
from .retrieval.embeddings import EmbeddingService
from .retrieval.pipeline import RetrievalPipeline
from .retrieval.vector_store import VectorStore
from .writers.file_writer import FileWriter
from .writers.graphiti_writer import GraphitiWriter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

server = Server("universal-memory")

_pipeline: RetrievalPipeline | None = None
_file_writer: FileWriter | None = None
_graphiti_writer: GraphitiWriter | None = None
_indexer: Indexer | None = None
_config = None


def _default_author() -> str:
    return os.environ.get("MEMORY_AUTHOR", "")


async def _ensure_init() -> None:
    """Lazy-initialise services on first tool call."""
    global _pipeline, _file_writer, _graphiti_writer, _indexer, _config

    if _pipeline is not None:
        return

    _config = load_config()
    await init_db(_config.index.db_path)

    embedding = EmbeddingService()
    vector = VectorStore()
    _pipeline = RetrievalPipeline()
    _file_writer = FileWriter()
    _graphiti_writer = GraphitiWriter()
    _indexer = Indexer(embedding, vector)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS = [
    Tool(
        name="memory_search",
        description=(
            "Search across memory files and Graphiti knowledge graph. "
            "Returns ranked results from vector search, BM25, and Graphiti."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "author": {"type": "string", "description": "Agent name for scoping"},
                "department": {"type": "string", "description": "Department scope override"},
                "max_results": {"type": "integer", "description": "Max results (default 10)"},
                "min_score": {"type": "number", "description": "Minimum score 0-1 (default 0.3)"},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="memory_write",
        description=(
            "Write content to memory files and/or Graphiti. "
            "Automatically resolves file paths from author and target."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Content to write"},
                "author": {"type": "string", "description": "Agent name"},
                "target": {
                    "type": "string",
                    "description": "Where to write: daily, long-term, department, shared, file",
                    "default": "daily",
                },
                "department": {"type": "string", "description": "Department override"},
                "file_path": {
                    "type": "string",
                    "description": "Custom file path (when target=file)",
                },
            },
            "required": ["content"],
        },
    ),
    Tool(
        name="memory_read",
        description="Read a specific file from the memory store.",
        inputSchema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to memory root",
                },
            },
            "required": ["path"],
        },
    ),
    Tool(
        name="memory_list",
        description="List files under a namespace in the memory store.",
        inputSchema={
            "type": "object",
            "properties": {
                "namespace": {
                    "type": "string",
                    "description": "Namespace path (e.g. agents/alice/logs, shared)",
                },
            },
            "required": ["namespace"],
        },
    ),
    Tool(
        name="memory_edit",
        description="Surgical find-and-replace in a memory file. old_text must match exactly once.",
        inputSchema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to memory root"},
                "old_text": {"type": "string", "description": "Exact text to find"},
                "new_text": {"type": "string", "description": "Replacement text"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    ),
    Tool(
        name="memory_status",
        description="Get service health, index statistics, and component status.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
]


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


@server.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    await _ensure_init()

    try:
        if name == "memory_search":
            return await _handle_search(arguments)
        elif name == "memory_write":
            return await _handle_write(arguments)
        elif name == "memory_read":
            return await _handle_read(arguments)
        elif name == "memory_list":
            return await _handle_list(arguments)
        elif name == "memory_edit":
            return await _handle_edit(arguments)
        elif name == "memory_status":
            return await _handle_status(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as exc:
        logger.exception("Tool %s failed", name)
        return [TextContent(type="text", text=f"Error: {exc}")]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


async def _handle_search(args: dict) -> list[TextContent]:
    author = args.get("author") or _default_author()
    req = SearchRequest(
        query=args["query"],
        author=author or None,
        department=args.get("department"),
        max_results=args.get("max_results", 10),
        min_score=args.get("min_score", 0.3),
    )
    resp = await _pipeline.search(req)
    lines = [f"Found {len(resp.results)} results for: {resp.query}"]
    if resp.expanded_queries:
        lines.append(f"Expanded queries: {', '.join(resp.expanded_queries)}")
    lines.append("")
    for i, r in enumerate(resp.results, 1):
        lines.append(f"--- Result {i} (score: {r.score:.2f}, source: {r.source}) ---")
        if r.file_path:
            loc = f"{r.file_path}:{r.line_start}-{r.line_end}"
            if r.header_path:
                loc += f" [{r.header_path}]"
            lines.append(loc)
        lines.append(r.content)
        lines.append("")
    if resp.timing_ms:
        parts = [f"{k}: {v:.0f}ms" for k, v in resp.timing_ms.items()]
        lines.append(f"Timing: {', '.join(parts)}")
    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_write(args: dict) -> list[TextContent]:
    author = args.get("author") or _default_author()
    if not author:
        return [TextContent(type="text", text="Error: author is required (set MEMORY_AUTHOR env)")]

    req = WriteRequest(
        content=args["content"],
        author=author,
        target=args.get("target", "daily"),
        department=args.get("department"),
        file_path=args.get("file_path"),
    )
    path = _file_writer.resolve_path(
        author=req.author, target=req.target,
        file_path=req.file_path, config=_config,
    )
    await _file_writer.write_content(
        path=path, content=req.content,
        header_format=_config.write.daily_log_header_format, author=req.author,
    )
    chunks = await _indexer.index_file(str(path))
    await _graphiti_writer.write(content=req.content, author=req.author)
    return [TextContent(type="text", text=f"Written to {path} ({chunks} chunks indexed)")]


async def _handle_read(args: dict) -> list[TextContent]:
    full = Path(_config.memory.data_dir) / args["path"]
    if not full.is_file():
        return [TextContent(type="text", text=f"File not found: {args['path']}")]
    content = await _file_writer.read_file(full)
    return [TextContent(type="text", text=content)]


async def _handle_list(args: dict) -> list[TextContent]:
    files = await _file_writer.list_files(args["namespace"])
    if not files:
        return [TextContent(type="text", text=f"No files found under {args['namespace']}")]
    return [TextContent(type="text", text="\n".join(files))]


async def _handle_edit(args: dict) -> list[TextContent]:
    full = Path(_config.memory.data_dir) / args["path"]
    if not full.is_file():
        return [TextContent(type="text", text=f"File not found: {args['path']}")]
    try:
        await _file_writer.edit_content(full, args["old_text"], args["new_text"])
    except ValueError as exc:
        return [TextContent(type="text", text=f"Edit failed: {exc}")]
    chunks = await _indexer.index_file(str(full))
    return [TextContent(
        type="text",
        text=f"Edited {args['path']} ({chunks} chunks re-indexed)",
    )]


async def _handle_status(args: dict) -> list[TextContent]:
    stats = await get_stats()
    lines = [
        "Universal Memory Service — Status",
        f"  Index: {stats.get('chunks', 0)} chunks, {stats.get('documents', 0)} documents",
        f"  Embedding provider: {_config.embedding.provider} / {_config.embedding.model}",
        f"  Graphiti: {_config.graphiti.url}",
        f"  Data dir: {_config.memory.data_dir}",
    ]
    return [TextContent(type="text", text="\n".join(lines))]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
