# Universal Memory Service

Self-hosted service providing unified memory search and write operations across file-based memory, vector embeddings, and the [Graphiti](https://github.com/getzep/graphiti) temporal knowledge graph. Platform-agnostic — works with any client via HTTP API or MCP stdio transport.

## Features

- **Unified search** — One query searches vector embeddings (Gemini), BM25 full-text, and Graphiti temporal facts, merged and reranked
- **Unified write** — One call persists to markdown files and Graphiti simultaneously
- **6-stage retrieval pipeline** — Query expansion → vector → BM25 → Graphiti → merge & rank → cross-encoder rerank
- **Local models** — Reranker and query expander run locally via GGUF (no API dependency for search)
- **Platform sync** — Canonical files auto-sync to OpenClaw, Hermes, and other platforms
- **MCP server** — Stdio transport for Claude Desktop, Cursor, and any MCP client
- **Graceful degradation** — Every component fails independently; the service never fully breaks

## Architecture

```
┌───────────┐  ┌───────────┐  ┌───────────────┐  ┌───────────┐
│  OpenClaw │  │  Hermes   │  │Claude Desktop │  │  Any MCP  │
│  (skill)  │  │  (skill)  │  │  (MCP client) │  │  Client   │
└─────┬─────┘  └─────┬─────┘  └──────┬────────┘  └─────┬─────┘
      │              │               │                  │
      └──────────────┴───── HTTP ────┴──── MCP stdio ───┘
                             │
                ┌────────────▼────────────┐
                │  Universal Memory Svc   │
                │  FastAPI :8002 + MCP    │
                ├─────────────────────────┤
                │  Retrieval Pipeline     │
                │  File Writer + Sync     │
                │  Indexer + Watcher      │
                │  Local GGUF Models      │
                └──────┬──────────┬───────┘
                       │          │
                ┌──────▼──┐  ┌───▼────────┐
                │ SQLite  │  │ Graphiti   │
                │ vec+FTS │  │ API :8001  │
                └─────────┘  └────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- [Graphiti API](https://github.com/getzep/graphiti) running on port 8001 (optional)
- Gemini API key for embeddings (optional — falls back to OpenAI, then BM25-only)

### Install

```bash
git clone <repo-url> && cd universal-memory-service
pip install -e ".[dev]"
```

### Configure

```bash
cp config/config.example.yaml ~/.memory-service/config.yaml
# Edit to set your data_dir, API keys, agent mappings

# Required for vector embeddings:
export GEMINI_API_KEY=your-key-here
# Without this key, the service falls back to BM25-only search (no vector embeddings).
```

### Run

```bash
# HTTP server
python -m universal_memory.main

# MCP server (for Claude Desktop / Cursor)
python -m universal_memory.mcp_server
```

## API

Base URL: `http://localhost:8002/api/v1`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/search` | POST | Hybrid search across files + Graphiti |
| `/write` | POST | Write to files and/or Graphiti |
| `/read/{path}` | GET | Read a file from the memory store |
| `/list/{namespace}` | GET | List files under a namespace |
| `/edit` | POST | Surgical find-and-replace in a file |
| `/ingest` | POST | Batch ingest messages into Graphiti |
| `/status` | GET | Health check and index stats |
| `/reindex` | POST | Trigger full re-index |

### Search

```bash
curl -s localhost:8002/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "deployment process", "author": "alice"}' | jq
```

### Write

```bash
curl -s localhost:8002/api/v1/write \
  -H "Content-Type: application/json" \
  -d '{"content": "Deployed v2.3 to staging", "author": "bob"}'
```

## MCP Server

The MCP server exposes 6 tools over stdio transport:

| Tool | Maps to | Description |
|------|---------|-------------|
| `memory_search` | POST /search | Search files + Graphiti |
| `memory_write` | POST /write | Write to files + Graphiti |
| `memory_read` | GET /read | Read a specific file |
| `memory_list` | GET /list | List files in a namespace |
| `memory_edit` | POST /edit | Find-and-replace in a file |
| `memory_status` | GET /status | Service health and stats |

### Claude Desktop config

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

## Retrieval Pipeline

Every search runs through a 6-stage pipeline:

1. **Query Expansion** — Local LLM rewrites the query into 2-3 semantic variants
2. **Vector Search** — Embed all variants via Gemini, cosine similarity against SQLite-vec
3. **BM25 Search** — Full-text search via SQLite FTS5
4. **Graphiti Search** — Temporal fact retrieval from the knowledge graph
5. **Merge & Rank** — Normalize scores, weighted merge (vector 0.40, BM25 0.20, Graphiti 0.25), temporal decay, MMR dedup
6. **Rerank** — Local cross-encoder re-scores top-N candidates for precision

## File Namespaces

```
~/.memory-service/data/
├── shared/              # Cross-agent knowledge (MEMORY.md, USER.md)
├── agents/{name}/logs/  # Per-agent daily logs
├── departments/{dept}/  # Department-level knowledge
├── projects/            # Cross-cutting project docs
├── guides/              # How-to docs
└── system/              # Internal state
```

Agents write using `author` and `target` fields — the service resolves file paths automatically.

## Configuration

See [`config/config.example.yaml`](config/config.example.yaml) for all options:

- **Service** — Host, port, auth token
- **Memory** — Data directory, file extensions
- **Agents** — Name-to-department mapping
- **Index** — Chunk size (400 tokens), overlap (80 tokens), DB path
- **Embedding** — Provider (Gemini/OpenAI), model, batch size
- **Models** — Reranker and query expander GGUF paths
- **Search** — Weights, temporal decay, MMR lambda
- **Graphiti** — URL, timeout
- **Sync** — Platform sync targets

## Local Models

| Model | Purpose | Size | Latency |
|-------|---------|------|---------|
| bge-reranker-v2-m3 (GGUF Q4) | Cross-encoder reranking | ~312 MB | ~165ms for 30 candidates |
| Qwen3-1.7B (GGUF Q4) | Query expansion | ~980 MB | ~80-100ms per query |

Both are optional — the service degrades gracefully without them.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Lint
ruff check src/ tests/
```

## License

MIT
