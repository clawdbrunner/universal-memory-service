# Universal Memory Service

A self-hosted memory service for AI agents. One API for semantic search, keyword search, and temporal knowledge graphs — with built-in reranking and query expansion.

**Status:** 🚧 Under Development

## Why?

AI agents need memory, but the current state is fragmented:

- **Vector search** lives in one system (QMD)
- **Keyword search** lives in another (SQLite FTS)
- **Temporal knowledge** lives in Graphiti (Neo4j)
- **File management** is scattered across shell scripts and launchd daemons

This service unifies all of that behind a single HTTP API (or MCP server). One call to search, one call to write, and everything stays in sync.

## Features

- **Hybrid retrieval** — Vector (semantic) + BM25 (keyword) + Graphiti (temporal) search with score fusion
- **Local reranking** — Cross-encoder model re-scores results for better precision
- **Query expansion** — Small local LLM rewrites queries for better recall
- **Multi-platform** — Works with OpenClaw, Hermes, Claude Desktop, or any MCP client
- **Namespace isolation** — Per-agent private memory, department-level sharing, global shared knowledge
- **Platform sync** — Auto-syncs in-context files (MEMORY.md, daily logs) to each platform
- **Single write path** — One API call persists to both files and Graphiti
- **Graceful degradation** — Each component fails independently; the service stays up

## Architecture

```
┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌──────────┐
│ OpenClaw │  │  Hermes  │  │ Claude Desktop│  │ Any MCP  │
│ (HTTP)   │  │  (HTTP)  │  │  (MCP stdio)  │  │  Client  │
└────┬─────┘  └────┬─────┘  └──────┬────────┘  └────┬─────┘
     └──────────────┴──────────────┴────────────────┘
                           │
              ┌────────────▼────────────┐
              │  Universal Memory Svc   │
              │  FastAPI :8002 + MCP    │
              ├─────────────────────────┤
              │  Retrieval Pipeline     │
              │  1. Query Expansion     │
              │  2. Vector Search       │
              │  3. BM25 Search         │
              │  4. Graphiti Search     │
              │  5. Merge & Rank        │
              │  6. Rerank              │
              ├─────────────────────────┤
              │  Write Engine           │
              │  File Writer + Graphiti │
              ├─────────────────────────┤
              │  Platform Sync          │
              │  OpenClaw ↔ Hermes ↔ …  │
              └────────┬────────────────┘
                       │
            ┌──────────┴──────────┐
            │                     │
      ┌─────▼─────┐        ┌─────▼──────┐
      │  SQLite   │        │  Graphiti  │
      │ vec + FTS5│        │  API:8001  │
      └───────────┘        └────────────┘
```

## Quick Start

```bash
# Clone
git clone https://github.com/clawdbrunner/universal-memory-service.git
cd universal-memory-service

# Configure
cp config/config.example.yaml config.yaml
# Edit config.yaml with your settings

# Run
uv run python -m uvicorn universal_memory.main:app --port 8002
```

### Docker

```bash
docker compose up -d
```

## API

### Search

```bash
curl -s localhost:8002/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "electric bill due date", "author": "alice"}'
```

The retrieval pipeline runs all enabled stages and returns merged, reranked results:

1. **Query expansion** — Rewrites query into 2–3 semantic variants
2. **Vector search** — Embeds variants with Gemini, searches sqlite-vec
3. **BM25 search** — Keyword matching via SQLite FTS5
4. **Graphiti search** — Temporal/relational facts from Neo4j
5. **Merge & rank** — Score normalization, weighted fusion, temporal decay, MMR dedup
6. **Rerank** — Local cross-encoder re-scores top candidates

### Write

```bash
curl -s localhost:8002/api/v1/write \
  -H "Content-Type: application/json" \
  -d '{"content": "Electric bill due on the 5th", "author": "alice"}'
```

A single write call:
- Appends to the agent's daily log file
- Logs to Graphiti with proper group scoping
- Triggers platform sync
- Updates the search index

### Other Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /search` | Hybrid search across all memory sources |
| `POST /write` | Write to files and Graphiti |
| `GET /read/{path}` | Read a file from the canonical store |
| `GET /list/{namespace}` | List files in a namespace |
| `POST /edit` | Surgical find-and-replace in a file |
| `POST /ingest` | Bulk ingest (session transcripts, batch imports) |
| `POST /reindex` | Force full re-index |
| `GET /status` | Health check and index stats |

### MCP (Claude Desktop)

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": ["-m", "universal_memory.mcp_server"],
      "env": { "MEMORY_AUTHOR": "alice" }
    }
  }
}
```

## Namespaces

Agents don't think in directory paths — they think in terms of who they are. The API uses `author` and `department` fields, and the service resolves file locations and Graphiti group IDs automatically.

| Target | File Location | Graphiti Groups |
|--------|--------------|-----------------|
| `daily` (default) | `agents/{author}/logs/YYYY-MM-DD.md` | `memory-{author}` + `memory-{dept}` |
| `long-term` | `agents/{author}/MEMORY.md` | `memory-{author}` |
| `department` | `departments/{dept}/YYYY-MM-DD.md` | `memory-{dept}` |
| `shared` | `shared/YYYY-MM-DD.md` | `memory-shared` |

Search is scoped automatically: an agent sees their own files, their department's files, and shared files — never another agent's private memory.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| HTTP API | FastAPI (Python 3.11+) |
| MCP Server | Python `mcp` SDK (stdio transport) |
| Vector Search | SQLite + sqlite-vec (fallback: FAISS) |
| Keyword Search | SQLite FTS5 (BM25) |
| Embeddings | Gemini `gemini-embedding-001` (free tier) |
| Reranker | Local GGUF cross-encoder (~400 MB) |
| Query Expander | Local GGUF LLM (~1 GB) |
| Knowledge Graph | Graphiti API → Neo4j |
| File Watching | `watchfiles` with polling fallback |

## Resource Usage

| Component | Memory | Disk |
|-----------|--------|------|
| FastAPI service | ~50 MB | — |
| Reranker model | ~400 MB | 312 MB |
| Query expander model | ~1.1 GB | 980 MB |
| SQLite index | ~10 MB | ~5 MB |
| **Total** | **~1.6 GB** | **~1.3 GB** |

## Configuration

Copy `config/config.example.yaml` and customize. Key sections:

- **agents** — Agent names, departments, Graphiti group mappings
- **sync** — Platform sync targets (OpenClaw, Hermes, etc.)
- **index** — Chunk size, overlap, DB path
- **embedding** — Provider, model, fallback chain
- **models** — Reranker and query expander paths and settings
- **search** — Source weights, temporal decay, MMR params
- **graphiti** — Graphiti API URL and timeout

See [`config/config.example.yaml`](config/config.example.yaml) for the full reference.

## Graceful Degradation

The service never becomes a single point of failure:

| Component Down | Behavior |
|----------------|----------|
| Gemini API | BM25 + Graphiti only; embeddings queued for retry |
| Reranker model | Returns Stage 5 merged results without reranking |
| Query expander | Uses original query only |
| Graphiti API | Returns file search results only |
| SQLite index corrupt | Auto-reindex; returns Graphiti-only results meanwhile |

## License

MIT
