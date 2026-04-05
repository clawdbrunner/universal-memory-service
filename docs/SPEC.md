"C""LYU  # Universal Memory Service — Specification

**Author:**
**Date:**
**Status:** Draft v4 — Under Review
**Reviewers:**

---

## Overview

A self-hosted HTTP service that provides unified memory search and write operations across file-based memory, vector embeddings, and the Graphiti temporal knowledge graph. Platform-agnostic by design — works with OpenClaw, Hermes Agent, Claude Desktop, or any client that can make HTTP calls.

### Goals

1. **Single write path** — One API call persists to both files and Graphiti
2. **Single search path** — One API call searches files (vector + BM25 + reranking) and Graphiti, returns merged results
3. **Platform-independent** — HTTP API with MCP server wrapper, no framework coupling
4. **Best-in-class retrieval** — Reranking and query expansion so the service stands alone without QMD or any framework-specific search
5. **Service owns the files** — Canonical memory files live in the service's data directory; platforms receive synced copies for system prompt injection
6. **Drop-in replacement** — Replaces OpenClaw's QMD/`memory_search`, existing shell scripts, and the file-sync launchd daemons
7. **Zero downtime migration** — Any combination of platforms can use it simultaneously

### Non-Goals

- Replacing Graphiti itself (Neo4j + Graphiti API continue running as-is)
- Replacing Hermes's session_search (that handles conversation recall natively)
- Hosting a UI (the Agent Dashboard can query the API if we want visualization later)

---

## Architecture

```
┌───────────┐  ┌───────────┐  ┌───────────────┐  ┌───────────┐
│  OpenClaw │  │  Hermes   │  │Claude Desktop │  │  Any MCP  │
│  (skill)  │  │  (skill)  │  │  (MCP client) │  │  Client   │
└─────┬─────┘  └─────┬─────┘  └──────┬────────┘  └─────┬─────┘
      │              │               │                  │
      │  ┌───────────┘    HTTP API (:8002) / MCP (stdio)│
      │  │   ┌───────────────────────┘                  │
      │  │   │   ┌──────────────────────────────────────┘
      │  │   │   │
      ▼  ▼   ▼   ▼
┌──────────────────────────────────────────┐
│         Universal Memory Service         │
│         (FastAPI :8002 + MCP stdio)      │
├──────────────────────────────────────────┤
│                                          │
│  ┌──────────────────┐  ┌──────────────┐  │
│  │ Retrieval Pipeline│  │ File Store   │  │
│  │ 1. Query Expand  │  │ (canonical)  │  │
│  │ 2. Vector Search │  │              │  │
│  │ 3. BM25 Search   │  │ ~/alice/data/│  │
│  │ 4. Graphiti      │  │   memory/    │  │
│  │ 5. Merge & Rank  │  │              │  │
│  │ 6. Rerank        │  └──────┬───────┘  │
│  └──────────────────┘         │          │
│                               │          │
│  ┌──────────────────┐  ┌──────▼───────┐  │
│  │ Write Engine     │  │ Platform Sync│  │
│  │ - File writer    │  │ - OpenClaw   │  │
│  │ - Graphiti log   │  │ - Hermes     │  │
│  │ - Index updater  │  │ - Custom     │  │
│  └──────────────────┘  └──────────────┘  │
│                                          │
│  ┌──────────────────┐  ┌──────────────┐  │
│  │ Local Models     │  │ File Watcher │  │
│  │ - Reranker GGUF  │  │ - Change     │  │
│  │ - Query Expander │  │   detect     │  │
│  └──────────────────┘  └──────────────┘  │
│                                          │
└─────────────┬────────────────────────────┘
              │
      ┌───────┴───────┐
      │               │
┌─────▼─────┐  ┌──────▼──────┐
│  SQLite   │  │  Graphiti   │
│ (vectors  │  │  API :8001  │
│  + FTS5)  │  │  → Neo4j    │
└───────────┘  └─────────────┘
```

### Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| **HTTP API** | Request handling | FastAPI (Python 3.11) |
| **MCP Server** | Claude Desktop / MCP client access | MCP stdio transport (Python `mcp` SDK) |
| **Vector Index** | Semantic similarity search | SQLite + `sqlite-vec` (fallback: FAISS) |
| **FTS Index** | Keyword/BM25 search | SQLite FTS5 |
| **Embedding Client** | Generate embeddings | Gemini `gemini-embedding-001` (free tier) |
| **Reranker** | Re-score candidate results for precision | Local GGUF cross-encoder via `llama-cpp-python` |
| **Query Expander** | Rewrite queries for better recall | Local GGUF LLM via `llama-cpp-python` |
| **Graphiti Client** | Temporal knowledge search & write | HTTP client to existing Graphiti API on :8001 |
| **File Watcher** | Detect memory file changes, trigger re-indexing | `watchfiles` (Python) with polling fallback |
| **File Writer** | Append/edit memory markdown files | Direct filesystem writes with `fcntl.flock()` |

---

## File Ownership & Namespaces

### Canonical File Store

The service owns all memory files. They live in a single directory tree:

```
~/.memory-service/data/
├── shared/                        # Cross-agent knowledge
│   ├── MEMORY.md                  # Long-term curated memory
│   ├── USER.md                    # User profile & preferences
│   └── IDENTITY.md                # Agent identity
│
├── agents/                        # Per-agent private memory
│   ├── alice/
│   │   └── logs/
│   │       ├── 2025-01-15.md
│   │       └── ...
│   ├── carol/
│   │   └── logs/
│   ├── sam/
│   │   └── logs/
│   └── .../                       # All multiple agents
│
├── departments/                   # Shared within teams
│   ├── engineering/               # Engineering team agents
│   ├── research/                  # Research team agents
│   ├── operations/                # Operations team agents
│   ├── finance/                   # Finance team agents
│   ├── comms/                     # Communications team agents
│   └── security/                  # Security team agents
│
├── projects/                      # Cross-cutting project docs
│   ├── GOALS.md
│   ├── mission-control-patterns.md
│   └── ...
│
├── guides/                        # How-to docs
│   ├── autonomous-operation.md
│   └── ...
│
└── system/                        # Internal state
    ├── heartbeat-state.json
    └── ...
```

No other system writes to this directory. Platforms access memory through the API, and the service syncs relevant files out to each platform (see Platform Sync below).

### Author/Department Model

Agents don't think in directory paths — they think in terms of who they are and what team they're on. The API uses `author` and `department` fields, and the service resolves everything else.

**Agent→department mapping (service config):**

```yaml
agents:
  alice:    { department: null }            # leader — searches everything by default
  carol:    { department: "comms" }
  dave:     { department: "research" }
  eve:      { department: "research" }
  frank:    { department: "operations" }
  grace:    { department: "operations" }
  heidi:    { department: "operations" }
  ivan:     { department: "comms" }
  judy:     { department: "comms" }
  karl:     { department: "comms" }
  leo:      { department: "finance" }
  mallory:  { department: "comms" }
  niaj:     { department: "engineering" }
  oscar:    { department: "engineering" }
  bob:      { department: "engineering" }
  pat:      { department: "security" }
  sam:      { department: "engineering" }
  rachel:   { department: "operations" }
  sam:      { department: "engineering" }
  heidi:    { department: "operations" }
  trent:    { department: "finance" }
  uma:      { department: "engineering" }
  victor:   { department: "engineering" }
```

**Write resolution — how `target` maps to file + Graphiti:**

| `target` | File location | Graphiti group_ids |
|----------|---------------|-------------------|
| `daily` (default) | `agents/{author}/logs/YYYY-MM-DD.md` | `memory-{author}` + `memory-{department}` (if known) |
| `long-term` | `agents/{author}/MEMORY.md` | `memory-{author}` |
| `department` | `departments/{department}/YYYY-MM-DD.md` | `memory-{department}` |
| `shared` | `shared/YYYY-MM-DD.md` | `memory-shared` |
| `file` | Custom `file_path` (relative to memory root) | Derived from path location |

**Examples:**

Simplest possible write — service knows the agent is in engineering:
```json
{
  "content": "Deployed v2.3 to staging",
  "author": "bob"
}
```
→ File: `agents/bob/logs/2025-01-15.md`
→ Graphiti: `memory-bob` + `memory-engineering`

Department-level knowledge:
```json
{
  "content": "Staging deploy process: always run smoke tests before promoting",
  "author": "bob",
  "target": "department"
}
```
→ File: `departments/engineering/2025-01-15.md`
→ Graphiti: `memory-engineering`

Override department (e.g., contributing to a different team's docs):
```json
{
  "content": "Security review checklist for deployments",
  "author": "bob",
  "department": "security",
  "target": "department"
}
```
→ File: `departments/security/2025-01-15.md`
→ Graphiti: `memory-security`

**Search resolution — how `author` and `department` determine scope:**

| Fields provided | Directories searched | Graphiti group_ids |
|---|---|---|
| `author` only | `agents/{author}/` + `departments/{default_dept}/` + `shared/` | `memory-{author}`, `memory-{dept}`, `memory-shared` |
| `department` only | `departments/{department}/` + `shared/` | `memory-{department}`, `memory-shared` |
| `author` + `department` | `agents/{author}/` + `departments/{department}/` + `shared/` | All three |
| Neither | Everything (global search) | All group_ids |

The agent never needs to construct a path. It just says who it is, and optionally which department context it cares about.

### Platform Sync

The service syncs designated in-context files to each platform's expected location. This runs automatically when the canonical files change.

**Sync targets are configured per platform:**

```yaml
sync:
  targets:
    - platform: "openclaw"
      agent: "alice"
      files:
        - source: "shared/MEMORY.md"
          dest: "~/.openclaw/workspace/MEMORY.md"
        - source: "shared/USER.md"
          dest: "~/.openclaw/workspace/USER.md"
        - source: "agents/alice/logs/{today}.md"
          dest: "~/.openclaw/workspace/memory/logs/{today}.md"
        - source: "agents/alice/logs/{yesterday}.md"
          dest: "~/.openclaw/workspace/memory/logs/{yesterday}.md"

    - platform: "hermes"
      agent: "alice"
      files:
        - source: "shared/MEMORY.md"
          dest: "~/.hermes/memories/MEMORY.md"
        - source: "shared/USER.md"
          dest: "~/.hermes/memories/USER.md"

    - platform: "hermes"
      agent: "carol"
      files:
        - source: "agents/carol/MEMORY.md"
          dest: "~/.hermes/profiles/carol/memories/MEMORY.md"
        - source: "shared/USER.md"
          dest: "~/.hermes/profiles/carol/memories/USER.md"

    # Add more platforms as needed (Claude Desktop doesn't need sync —
    # it uses MCP tools exclusively)
```

**Sync behavior:**

| Trigger | Action |
|---------|--------|
| Canonical file changes (write via API) | Immediate sync to all configured targets |
| File watcher detects external edit to canonical | Immediate sync to targets |
| Service startup | Full sync of all configured files |
| Target file edited directly by platform | **Conflict** — see below |

**Conflict resolution:**

If a platform edits a synced file directly (e.g., OpenClaw's agent writes to MEMORY.md via `edit` tool), the file watcher on the target detects the change and syncs it BACK to canonical. This makes the canonical copy eventually consistent.

However, the preferred path is: platforms write through the API, which updates canonical and syncs out. Direct file edits are a fallback that works but may have brief inconsistency.

**What gets synced vs. what stays API-only:**

| Content | Synced to platforms? | Why |
|---------|---------------------|-----|
| MEMORY.md, USER.md | ✅ Yes | Platforms inject these into system prompts at session start |
| Today/yesterday daily logs | ✅ Yes (OpenClaw) | OpenClaw auto-loads recent daily logs |
| All other memory files | ❌ No | Accessed via `memory_search` / `memory_read` API calls |
| Index, embeddings | ❌ No | Service-internal |

This keeps sync lightweight — only 2-4 small files per platform, updated infrequently.

---

### Why Local Models for Reranking & Query Expansion?

- **No API dependency for search** — Search works offline and with zero latency to external services
- **No per-query cost** — Reranking and expansion run thousands of times; API calls would add up
- **Proven approach** — QMD uses the same pattern (local GGUF models) successfully
- **Privacy** — Memory content never leaves the machine for search operations
- **Embeddings stay API-based** — Indexing is infrequent (on file changes), so Gemini API calls are cheap and higher quality than local embedding models

### Why SQLite (not a separate vector DB)?

- Already proven in many agent stacks
- Single file, no extra service to manage
- `sqlite-vec` provides vector search in-process
- FTS5 is built into SQLite
- Backup = copying one file
- Keeps total service count at 3 (this + Graphiti + Neo4j)

---

## Retrieval Pipeline

This is the core of the service — a 6-stage pipeline that runs on every search request.

```
                        ┌─────────────┐
                        │    Query    │
                        └──────┬──────┘
                               │
                   ┌───────────▼───────────┐
            Stage 1│   Query Expansion     │
                   │  Local LLM rewrites   │
                   │  query into 2-3       │
                   │  semantic variants    │
                   └───────────┬───────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                  │
    ┌────────▼────────┐ ┌─────▼──────┐ ┌────────▼────────┐
    │  Stage 2:       │ │ Stage 3:   │ │  Stage 4:       │
    │  Vector Search  │ │ BM25       │ │  Graphiti       │
    │  (all variants) │ │ (original  │ │  (original      │
    │                 │ │  + expand) │ │   query)        │
    └────────┬────────┘ └─────┬──────┘ └────────┬────────┘
             │                │                  │
             └────────────────┼──────────────────┘
                              │
                   ┌──────────▼──────────┐
            Stage 5│   Merge & Rank      │
                   │  - Score normalize  │
                   │  - Weighted merge   │
                   │  - Temporal decay   │
                   │  - MMR deduplicate  │
                   └──────────┬──────────┘
                              │
                   ┌──────────▼──────────┐
            Stage 6│   Rerank            │
                   │  Local cross-encoder│
                   │  re-scores top N    │
                   │  candidates         │
                   └──────────┬──────────┘
                              │
                        ┌─────▼─────┐
                        │  Results  │
                        └───────────┘
```

### Stage 1: Query Expansion

A local LLM rewrites the user's query into 2-3 semantic variants to improve recall.

**Example:**
```
Input:  "electric bill due date"
Output: [
  "electric bill due date",                          // original preserved
  "electricity payment schedule utility company",    // synonym expansion
  "when is the monthly utility payment due"    // natural language variant
]
```

**Model:** Small local GGUF (~1-2B params, e.g., Qwen3-1.7B or Phi-4-mini). Loaded once at startup, stays in memory. The prompt is a simple few-shot template:

```
Rewrite this search query into 2 alternative phrasings that might match relevant documents. Keep the original meaning. Return only the rewrites, one per line.

Query: {query}
```

**Performance target:** <100ms per expansion. The model is tiny and the prompt is short.

**When to skip:** If the query is already very specific (contains IDs, error codes, file paths), expansion can hurt precision. The service detects these patterns and skips expansion, using the original query only.

### Stage 2: Vector Search

Embed all query variants (original + expanded) using Gemini, then search sqlite-vec for cosine similarity. Results from all variants are pooled and deduplicated by chunk_id (keep highest score).

**Top-K per variant:** 20 candidates (before merge).

### Stage 3: BM25 Search

Run the original query + expanded variants against SQLite FTS5. Standard BM25 scoring.

**Top-K:** 20 candidates.

### Stage 4: Graphiti Search

POST to Graphiti's `/search` endpoint with the original query. Returns temporal facts with entity relationships and validity timestamps.

**Max facts:** 10.

### Stage 5: Merge & Rank

1. **Normalize scores** — Each source's scores are min-max normalized to 0-1
2. **Weighted merge** — Combine by source weight:

| Source | Weight | Rationale |
|--------|--------|-----------|
| Vector | 0.40 | Primary semantic understanding |
| BM25 | 0.20 | Precision backstop for exact terms |
| Graphiti | 0.25 | High-signal temporal/relational facts |
| *(reserved for reranker adjustment)* | 0.15 | Applied in Stage 6 |

3. **Temporal decay** — File results older than 30 days lose ranking weight (half-life = 30 days). `MEMORY.md`, `IDENTITY.md`, `USER.md` are exempt.
4. **MMR deduplication** — λ=0.7 to reduce near-duplicate results from overlapping chunks

**Output:** Top 30 candidates passed to reranker.

### Stage 6: Rerank

A local cross-encoder model re-scores each (query, candidate) pair. This is the precision step — it reads both the query and the full chunk text together, so it understands context far better than vector similarity alone.

**Model:** GGUF cross-encoder (~100-400M params). Options:
- `bge-reranker-v2-m3` (GGUF, ~568M) — multilingual, strong
- `jina-reranker-v2-base-multilingual` (GGUF, ~278M) — lighter, still good
- Same model QMD uses, for parity

**Process:**
1. Take top 30 candidates from Stage 5
2. Score each with cross-encoder: `score(query, chunk_text)` → 0-1
3. Blend reranker score with Stage 5 score: `final = 0.85 * rerank + 0.15 * stage5_score`
4. Sort by final score
5. Return top `max_results`

**Performance target:** <200ms for 30 candidates on Apple Silicon. Cross-encoders are fast for short texts.

### Why This Pipeline Order Matters

- **Expansion first** — Casts a wider net before any retrieval happens
- **Three parallel retrievers** — Vector catches semantic matches, BM25 catches exact matches, Graphiti catches temporal facts. Together they have very high recall.
- **Merge normalizes across sources** — Graphiti scores aren't comparable to vector cosine distances without normalization
- **Reranker last** — Expensive per-candidate, so we only run it on the merged top-N. This is the standard retrieve-then-rerank pattern used in production RAG systems.

---

## API Design

### Base URL

`http://localhost:8002/api/v1`

### Authentication

Bearer token via `Authorization` header. Optional for localhost-only (typical case), configurable for future multi-machine deployments.

---

### `POST /search`

Unified search across all memory sources.

**Request:**
```json
{
  "query": "electric bill due date",
  "author": "alice",
  "department": null,
  "sources": ["files", "graphiti"],
  "max_results": 10,
  "min_score": 0.3,
  "temporal_filter": null,
  "expand": true,
  "rerank": true
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | required | Search query |
| `author` | string | `null` | Agent name — scopes to agent's files + default department + shared. Null = global. |
| `department` | string | `null` | Override/specify department scope. If omitted, uses author's default department from config. |
| `sources` | string[] | `["files", "graphiti"]` | Backends: `files`, `graphiti`, `all` |
| `max_results` | int | `10` | Maximum results to return |
| `min_score` | float | `0.3` | Minimum relevance score (0-1) |
| `temporal_filter` | object | `null` | `{"after": "2024-12-01", "before": "2025-01-01"}` |
| `expand` | bool | `true` | Enable query expansion |
| `rerank` | bool | `true` | Enable cross-encoder reranking |

**Response:**
```json
{
  "results": [
    {
      "source": "files",
      "score": 0.92,
      "content": "Electric company — Due on the 5th, manual pay required. Need to enroll in auto-pay.",
      "file": "memory/logs/2025-01-15.md",
      "line_start": 15,
      "line_end": 16,
      "header_path": "## Daily > ### Tasks",
      "chunk_id": "a3f2c1"
    },
    {
      "source": "graphiti",
      "score": 0.85,
      "content": "Electric bill is due on the 5th of each month",
      "fact_id": "f-12345",
      "valid_at": "2025-01-14T00:00:00Z",
      "entities": ["Utility Co", "User"]
    }
  ],
  "query": "electric bill due date",
  "scope": {
    "author": "alice",
    "department": null,
    "directories_searched": ["agents/alice/", "shared/"],
    "graphiti_group_ids": ["memory-alice", "memory-shared"]
  },
  "expanded_queries": [
    "electricity payment schedule utility company",
    "when is the electric bill due"
  ],
  "sources_queried": ["files", "graphiti"],
  "timing_ms": {
    "expansion": 82,
    "files_vector": 45,
    "files_bm25": 12,
    "graphiti": 230,
    "merge": 3,
    "rerank": 165,
    "total": 537
  }
}
```

---

### `POST /write`

Unified write to files and Graphiti.

**Request:**
```json
{
  "content": "electric bill is due on the 5th, manual pay required.",
  "author": "alice",
  "department": null,
  "role": "assistant",
  "target": "daily",
  "targets": ["file", "graphiti"],
  "file_path": null,
  "timestamp": null
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `content` | string | required | Content to write |
| `author` | string | required | Agent name — determines file path and Graphiti group_id |
| `department` | string | `null` | Override department. If omitted, uses author's default from config. |
| `role` | string | `"assistant"` | Role: `user`, `assistant`, `system` |
| `target` | string | `"daily"` | Where to write: `daily`, `long-term`, `department`, `shared`, `file` |
| `targets` | string[] | `["file", "graphiti"]` | Backends: `file`, `graphiti`, `all` |
| `file_path` | string | `null` | Custom path (relative to memory root) when `target` is `file` |
| `timestamp` | string | `null` | ISO 8601 (defaults to now) |

**How `target` resolves:**

| `target` | File location | Graphiti group_ids |
|----------|---------------|-------------------|
| `daily` (default) | `agents/{author}/logs/YYYY-MM-DD.md` | `memory-{author}` + `memory-{department}` |
| `long-term` | `agents/{author}/MEMORY.md` | `memory-{author}` |
| `department` | `departments/{department}/YYYY-MM-DD.md` | `memory-{department}` |
| `shared` | `shared/YYYY-MM-DD.md` | `memory-shared` |
| `file` | Custom `file_path` | Derived from path location |

**Response:**
```json
{
  "ok": true,
  "written_to": {
    "file": {
      "path": "agents/alice/logs/2025-01-15.md",
      "line": 42,
      "bytes_written": 87
    },
    "graphiti": {
      "status": "accepted",
      "group_ids": ["memory-alice"],
      "message_id": "msg-abc123"
    }
  },
  "synced_to": ["openclaw:alice", "hermes:alice"],
  "index_updated": true
}
```

**File format for daily logs:**
```markdown
## [09:15] alice
electric bill is due on the 5th, manual pay required.
```

---

### `GET /read/{path}`

Read a file's content from the canonical store.

**Example:** `GET /read/agents/alice/logs/2025-01-15.md`

**Query parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `line_start` | int | `null` | Start reading from this line (1-indexed) |
| `line_end` | int | `null` | Stop reading at this line |

**Response:**
```json
{
  "path": "agents/alice/logs/2025-01-15.md",
  "namespace": "agents/alice",
  "content": "# Daily Log — 2025-01-15\n\n## Morning (9:00 AM)\n...",
  "size_bytes": 2847,
  "last_modified": "2025-01-15T21:23:00Z",
  "lines": 87
}
```

---

### `GET /list/{namespace}`

List files in a namespace.

**Example:** `GET /list/agents/alice/logs`

**Query parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `recursive` | bool | `true` | Include subdirectories |
| `pattern` | string | `"*.md"` | Glob filter |

**Response:**
```json
{
  "namespace": "agents/alice",
  "path": "logs",
  "files": [
    {
      "path": "agents/alice/logs/2025-01-15.md",
      "size_bytes": 2847,
      "last_modified": "2025-01-15T21:23:00Z"
    },
    {
      "path": "agents/alice/logs/2025-01-14.md",
      "size_bytes": 4521,
      "last_modified": "2025-01-14T06:00:00Z"
    }
  ],
  "total": 45
}
```

---

### `POST /edit`

Surgical edit of a file (find and replace).

**Request:**
```json
{
  "path": "shared/MEMORY.md",
  "old_text": "Status: active",
  "new_text": "Status: inactive (as of Jan 15)",
  "targets": ["file", "graphiti"]
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | string | required | File path relative to memory root |
| `old_text` | string | required | Exact text to find (must match uniquely) |
| `new_text` | string | required | Replacement text |
| `targets` | string[] | `["file", "graphiti"]` | Also log the edit to Graphiti as a fact update |

**Response:**
```json
{
  "ok": true,
  "path": "shared/MEMORY.md",
  "line": 34,
  "synced_to": ["openclaw:alice", "hermes:alice"],
  "index_updated": true,
  "graphiti": {
    "status": "accepted",
    "group_id": "memory-shared"
  }
}
```

**Error if `old_text` matches 0 or 2+ locations:**
```json
{
  "ok": false,
  "error": "old_text matched 0 locations in shared/MEMORY.md"
}
```

---

### `POST /ingest`

Bulk ingest content (session transcripts, batch imports).

**Request:**
```json
{
  "messages": [
    {
      "role": "user",
      "author": "User",
      "content": "Can you check on the electric bill?",
      "timestamp": "2025-01-15T21:00:00Z"
    },
    {
      "role": "assistant",
      "author": "alice",
      "content": "Electric bill is due tomorrow. Manual payment required.",
      "timestamp": "2025-01-15T21:00:15Z"
    }
  ],
  "group_id": "memory-alice",
  "source": "openclaw-session",
  "session_id": "abc-123",
  "targets": ["graphiti"]
}
```

**Response:**
```json
{
  "ok": true,
  "ingested": 2,
  "graphiti_status": "accepted"
}
```

---

### `GET /status`

Health check and index statistics.

**Response:**
```json
{
  "status": "healthy",
  "uptime_seconds": 86400,
  "index": {
    "files_indexed": 52,
    "chunks": 847,
    "embeddings": 847,
    "last_indexed_at": "2025-01-15T21:15:00Z",
    "index_size_bytes": 4521984
  },
  "models": {
    "reranker": {
      "loaded": true,
      "model": "bge-reranker-v2-m3-Q4_K_M.gguf",
      "memory_mb": 312
    },
    "query_expander": {
      "loaded": true,
      "model": "qwen3-1.7b-Q4_K_M.gguf",
      "memory_mb": 980
    }
  },
  "graphiti": {
    "status": "connected",
    "url": "http://localhost:8001",
    "last_ping_ms": 12
  },
  "embedding_provider": {
    "provider": "gemini",
    "model": "gemini-embedding-001",
    "requests_today": 342
  },
  "file_watcher": {
    "watching": "/home/user/.memory-service/data",
    "last_change_detected": "2025-01-15T21:10:00Z"
  }
}
```

---

### `POST /reindex`

Force a full re-index of all memory files.

**Response:**
```json
{
  "ok": true,
  "files_indexed": 52,
  "chunks_created": 847,
  "embeddings_generated": 847,
  "duration_ms": 12500
}
```

---

## Local Models

### Reranker

| Parameter | Value |
|-----------|-------|
| Model | `bge-reranker-v2-m3` (GGUF, Q4_K_M quantization) |
| Size | ~312 MB on disk, ~400 MB in memory |
| Purpose | Cross-encoder scoring of (query, chunk) pairs |
| Latency | ~5-7ms per pair on M-series Apple Silicon |
| Throughput | 30 candidates in ~165ms |
| Loaded at | Service startup, stays in memory |

### Query Expander

| Parameter | Value |
|-----------|-------|
| Model | Qwen3-1.7B (GGUF, Q4_K_M quantization) |
| Size | ~980 MB on disk, ~1.1 GB in memory |
| Purpose | Rewrite queries into 2-3 semantic variants |
| Latency | ~80-100ms per query |
| Loaded at | Service startup, stays in memory |

### Total Model Memory

~1.5 GB for both models. On our host machine with ~8 GB used by other services, this fits comfortably alongside the multiple Hermes gateway processes (~1 GB).

### Model Management

Models are downloaded on first startup to `~/.memory-service/models/`:

```bash
# Auto-downloaded from HuggingFace on first run
~/.memory-service/models/
├── bge-reranker-v2-m3-Q4_K_M.gguf        # 312 MB
└── qwen3-1.7b-Q4_K_M.gguf                 # 980 MB
```

If download fails or hardware can't support local models, the service degrades gracefully:
- **No reranker:** Skip Stage 6, return Stage 5 results (still good, just less precise)
- **No query expander:** Skip Stage 1, use original query only (still works, just lower recall)
- Both are independent — one can fail without affecting the other

---

## Indexing

### File Discovery

Configurable watched paths:

```yaml
watch:
  directories:
    - ~/alice/memory/
  files:
    - ~/alice/MEMORY.md
    - ~/alice/IDENTITY.md
    - ~/alice/USER.md
  extensions: [".md"]
  ignore_patterns: ["*.tmp", "*.bak", ".git/"]
```

### Chunking Strategy

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Target chunk size | ~400 tokens (~1,600 chars) | Same as OpenClaw/QMD, balances context vs precision |
| Overlap | 80 tokens (~320 chars) | Preserves context across chunk boundaries |
| Split strategy | Markdown-aware | Split on `##` headers first, then `\n\n`, then sentences |
| Metadata per chunk | file path, line range, header hierarchy, last modified | Source attribution |

### Embedding

| Parameter | Value |
|-----------|-------|
| Provider | Gemini |
| Model | `gemini-embedding-001` |
| Dimensions | 768 |
| Rate limit | Free tier: 1,500 RPM / 1.5M RPD |
| Batch size | 100 chunks per request |
| Cost | Free |

Fallback chain: Gemini → OpenAI `text-embedding-3-small` → skip vectors (BM25 + Graphiti only).

### Index Storage

Single SQLite database: `~/.memory-service/data-index.db`

```sql
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    line_start INTEGER NOT NULL,
    line_end INTEGER NOT NULL,
    content TEXT NOT NULL,
    header_path TEXT,
    file_modified_at TIMESTAMP,
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    token_count INTEGER,
    embedding_hash TEXT
);

CREATE VIRTUAL TABLE chunk_embeddings USING vec0(
    chunk_id TEXT PRIMARY KEY,
    embedding FLOAT[768]
);

CREATE VIRTUAL TABLE chunks_fts USING fts5(
    content,
    file_path,
    header_path,
    content='chunks',
    content_rowid='rowid'
);

CREATE TABLE file_state (
    file_path TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    last_indexed_at TIMESTAMP,
    chunk_count INTEGER
);
```

### Incremental Indexing

On file change:
1. Hash file content → compare to `file_state.content_hash`
2. If changed: delete old chunks, re-chunk, embed new chunks (reuse embeddings for unchanged chunks via `embedding_hash`), update FTS
3. If unchanged: skip

Full re-index only on `/reindex` or first startup.

---

## Platform Integration

The service exposes the same capabilities through two transports: HTTP and MCP. Every platform uses one or both.

### MCP Server (Claude Desktop, Cursor, any MCP client)

MCP stdio transport, compatible with any MCP-capable client.

**MCP Tools exposed:**

| Tool | Maps to | Description |
|------|---------|-------------|
| `memory_search` | `POST /search` | Search across files and Graphiti |
| `memory_write` | `POST /write` | Write to files and Graphiti |
| `memory_read` | `GET /read/{path}` | Read a specific file |
| `memory_list` | `GET /list/{namespace}` | List files in a namespace |
| `memory_edit` | `POST /edit` | Surgical find/replace in a file |
| `memory_status` | `GET /status` | Service health and index stats |

**Claude Desktop config (`claude_desktop_config.json`):**
```json
{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": ["-m", "universal_memory.mcp_server"],
      "cwd": "~/.memory-service",
      "env": {
        "MEMORY_AUTHOR": "alice"
      }
    }
  }
}
```

The `MEMORY_AUTHOR` env var sets the default author for that client, so every tool call is automatically scoped without the user having to specify it. Department is resolved from the agent config.

MCP server is a thin wrapper calling the same internal functions as HTTP. No separate process.

### OpenClaw Skill

Skill at `~/skills/universal-memory/SKILL.md`.

Registers tools that call the HTTP API. Replaces QMD as the memory backend.

**Sync config:** The service syncs `shared/MEMORY.md` + today/yesterday daily logs → OpenClaw workspace, so OpenClaw's auto-load at session start still works.

**Tools registered:**
- `memory_search` — replaces built-in memory_search
- `memory_write` — new tool for dual file+Graphiti writes
- `memory_read` — read any file in the canonical store
- `memory_edit` — surgical edits with sync + Graphiti logging

### Hermes Agent Skill

Skill at `~/.hermes/skills/universal-memory/` (agentskills.io format).

Same tools as OpenClaw. Fills the semantic search gap Hermes doesn't have natively.

**Sync config:** The service syncs `shared/MEMORY.md` + `shared/USER.md` → Hermes profile memories dir, so Hermes's auto-injection at session start still works.

### How Each Platform Accesses Memory

| Platform | Transport | In-context files | Knowledge search/write | Namespace |
|----------|-----------|-----------------|----------------------|-----------|
| **OpenClaw** | HTTP (skill) | Synced MEMORY.md + daily logs auto-loaded | `memory_search` / `memory_write` skill tools | `author` per-agent config |
| **Hermes** | HTTP (skill) | Synced MEMORY.md + USER.md auto-injected | `memory_search` / `memory_write` skill tools | `author` per-profile config |
| **Claude Desktop** | MCP (stdio) | None (uses `memory_read` tool) | All 6 MCP tools | `MEMORY_AUTHOR` env var |
| **Any MCP client** | MCP (stdio) | None (uses `memory_read` tool) | All 6 MCP tools | `MEMORY_AUTHOR` env var |
| **CLI / scripts** | HTTP (curl) | N/A | Direct API calls | Per-request `author` field |

### Direct CLI

```bash
# Search (scoped to agent + default department + shared)
curl -s localhost:8002/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "electric bill", "author": "alice"}' | jq

# Search (global — no scoping)
curl -s localhost:8002/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "deployment process"}' | jq

# Write (simplest — service resolves everything from author)
curl -s localhost:8002/api/v1/write \
  -H "Content-Type: application/json" \
  -d '{"content": "electric bill auto-pay enrolled", "author": "alice"}'

# Write to department knowledge
curl -s localhost:8002/api/v1/write \
  -H "Content-Type: application/json" \
  -d '{"content": "Always run smoke tests before promoting", "author": "bob", "target": "department"}'

# Read
curl -s localhost:8002/api/v1/read/shared/MEMORY.md | jq

# List
curl -s localhost:8002/api/v1/list/agents/alice/logs | jq

# Edit
curl -s localhost:8002/api/v1/edit \
  -H "Content-Type: application/json" \
  -d '{"path": "shared/MEMORY.md", "old_text": "Status: active", "new_text": "Status: inactive"}'

# Status
curl -s localhost:8002/api/v1/status | jq
```

---

## Configuration

```yaml
# ~/.memory-service/config.yaml

service:
  host: "127.0.0.1"
  port: 8002
  auth_token: null

memory:
  data_dir: "~/.memory-service/data"      # Canonical file store
  extensions: [".md"]
  daily_log_format: "{date}.md"         # YYYY-MM-DD.md

agents:
  alice:    { department: null, graphiti_group: "memory-alice" }  # custom group name
  carol:    { department: "comms" }
  dave:     { department: "research" }
  eve:      { department: "research" }
  frank:    { department: "operations" }
  grace:    { department: "operations" }
  heidi:    { department: "operations" }
  ivan:     { department: "comms" }
  judy:     { department: "comms" }
  karl:     { department: "comms" }
  leo:      { department: "finance" }
  mallory:  { department: "comms" }
  niaj:     { department: "engineering" }
  oscar:    { department: "engineering" }
  bob:      { department: "engineering" }
  pat:      { department: "security" }
  sam:      { department: "engineering" }
  rachel:   { department: "operations" }
  sam:      { department: "engineering" }
  trent:    { department: "finance" }
  uma:      { department: "engineering" }
  victor:   { department: "engineering" }

# Graphiti group_id defaults:
#   agents/{name} → memory-{name}  (with optional per-agent overrides)
#   departments/{dept} → memory-{dept}
#   shared → memory-shared

sync:
  enabled: true
  debounce_ms: 500                       # Wait for writes to settle before syncing
  targets:
    - platform: "openclaw"
      agent: "alice"
      files:
        - source: "shared/MEMORY.md"
          dest: "~/.openclaw/workspace/MEMORY.md"
        - source: "shared/USER.md"
          dest: "~/.openclaw/workspace/USER.md"
        - source: "agents/alice/logs/{today}.md"
          dest: "~/.openclaw/workspace/memory/logs/{today}.md"
        - source: "agents/alice/logs/{yesterday}.md"
          dest: "~/.openclaw/workspace/memory/logs/{yesterday}.md"
    - platform: "hermes"
      agent: "alice"
      files:
        - source: "shared/MEMORY.md"
          dest: "~/.hermes/memories/MEMORY.md"
        - source: "shared/USER.md"
          dest: "~/.hermes/memories/USER.md"
    # Add per-agent Hermes profiles as they're created

index:
  db_path: "~/.memory-service/data-index.db"
  chunk_size_tokens: 400
  chunk_overlap_tokens: 80

embedding:
  provider: "gemini"
  model: "gemini-embedding-001"
  api_key_env: "GEMINI_API_KEY"
  batch_size: 100
  fallback_providers:
    - provider: "openai"
      model: "text-embedding-3-small"
      api_key_env: "OPENAI_API_KEY"

models:
  reranker:
    enabled: true
    model_path: "~/.memory-service/models/bge-reranker-v2-m3-Q4_K_M.gguf"
    model_url: "https://huggingface.co/BAAI/bge-reranker-v2-m3-GGUF/resolve/main/bge-reranker-v2-m3-Q4_K_M.gguf"
    candidates: 30
    blend_weight: 0.85
  query_expander:
    enabled: true
    model_path: "~/.memory-service/models/qwen3-1.7b-Q4_K_M.gguf"
    model_url: "https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/qwen3-1.7b-Q4_K_M.gguf"
    max_expansions: 2
    skip_patterns: ["^[A-Z0-9_-]+$", "\\.[a-z]+$", "error:", "traceback"]

search:
  weights:
    vector: 0.40
    bm25: 0.20
    graphiti: 0.25
  temporal_decay:
    enabled: true
    half_life_days: 30
    exempt_files: ["MEMORY.md", "IDENTITY.md", "USER.md"]
  mmr:
    enabled: true
    lambda: 0.7
  default_max_results: 10
  default_min_score: 0.3

graphiti:
  url: "http://localhost:8001"
  default_group_id: "memory-alice"
  timeout_seconds: 10

write:
  daily_log_header_format: "## [{time}] {author}"
  append_newlines: 2
  file_lock: true

logging:
  level: "INFO"
  file: "~/.memory-service/logs/service.log"
```

---

## Deployment

### Docker (recommended for production)

```yaml
# ~/.memory-service/docker-compose.yml
version: '3.8'

services:
  memory-service:
    build: .
    container_name: memory-service
    ports:
      - "8002:8002"
    volumes:
      - ~/workspace:/workspace
      - ~/.memory-service/data:/data
      - ./config.yaml:/app/config.yaml:ro
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - GRAPHITI_URL=http://host.docker.internal:8001
    deploy:
      resources:
        limits:
          memory: 3G
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8002/api/v1/status"]
      interval: 30s
      timeout: 5s
      retries: 3
```

### Launchd Service (macOS)

```xml
<!-- ~/Library/LaunchAgents/com.memory-service.plist -->
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.memory-service</string>
    <key>ProgramArguments</key>
    <array>
        <string>uv</string>
        <string>run</string>
        <string>python</string>
        <string>-m</string>
        <string>uvicorn</string>
        <string>main:app</string>
        <string>--host</string>
        <string>127.0.0.1</string>
        <string>--port</string>
        <string>8002</string>
    </array>
    <key>WorkingDirectory</key>
    <string>~/.memory-service</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
```

---

## Replaces / Deprecates

| Current Component | Replaced By | Notes |
|-------------------|-------------|-------|
| OpenClaw QMD sidecar | Service's retrieval pipeline | Full replacement with reranking + query expansion + Graphiti |
| OpenClaw builtin memory_search | Service's `/search` endpoint | Via OpenClaw skill wrapper |
| `graphiti-search.sh` | `POST /search` with `sources: ["graphiti"]` | Keep script as convenience alias |
| `graphiti-log.sh` | `POST /write` with `targets: ["graphiti"]` | Keep script as convenience alias |
| `memory-hybrid-search.sh` | `POST /search` with `sources: ["all"]` | Direct replacement |
| `graphiti-file-sync daemon` launchd | Built-in file watcher | Remove launchd job |
| `graphiti-sync daemon` session sync | `POST /ingest` | Remove launchd job |

---

## Resource Budget

| Component | Memory | Disk | CPU |
|-----------|--------|------|-----|
| FastAPI service | ~50 MB | — | Negligible idle |
| Reranker model (loaded) | ~400 MB | 312 MB | Bursty on search |
| Query expander model (loaded) | ~1.1 GB | 980 MB | Bursty on search |
| SQLite index | ~10 MB | ~5 MB | Negligible |
| **Total** | **~1.6 GB** | **~1.3 GB** | **Low avg, bursty** |

On our host machine:
- Current usage: ~8 GB (Docker, Graphiti, Neo4j, OpenClaw gateway, misc)
- This service: ~1.6 GB
- multiple Hermes gateways (future): ~1 GB
- Remaining headroom depends on host configuration

---

## Graceful Degradation

The service should never be a single point of failure. Each component degrades independently:

| Component Down | Behavior |
|----------------|----------|
| Gemini API | Skip vector search, use BM25 + Graphiti only. Queue embeddings for retry. |
| Reranker model | Skip Stage 6, return Stage 5 merged results. Log warning. |
| Query expander model | Skip Stage 1, use original query only. Log warning. |
| Graphiti API | Skip Graphiti results, return file search only. Log warning. |
| SQLite index corrupt | Trigger auto-reindex on next request. Return Graphiti-only results meanwhile. |
| Service itself down | Platform skills fall back to native search (OpenClaw builtin, Hermes session_search). |

---

## Migration Plan

### Phase 1: Build & Test (standalone)
- Build the service with all 6 pipeline stages
- Index existing memory files
- Download and test local models (reranker + expander)
- Benchmark search quality against current QMD + Graphiti results
- Verify graceful degradation

### Phase 2: Integrate with Hermes
- Create Hermes skill wrapper
- Test search + write from Hermes sessions
- Verify daily log writes and Graphiti ingestion

### Phase 3: Integrate with OpenClaw
- Create OpenClaw skill wrapper
- Switch memory backend from QMD to our service
- Test dual-platform operation

### Phase 4: MCP Server
- Add MCP stdio transport
- Test with Claude Desktop
- Document MCP configuration for other clients

### Phase 5: Deprecate Old Components
- Remove file-sync and session-sync launchd daemons
- Disable QMD sidecar in OpenClaw config
- Update TOOLS.md documentation
- Archive old shell scripts (keep as convenience aliases)

---

## Open Questions

1. **sqlite-vec on macOS ARM:** Is `sqlite-vec` installable via pip? If not, use FAISS for vectors with SQLite for metadata/FTS only.

2. **Reranker model choice:** `bge-reranker-v2-m3` vs `jina-reranker-v2-base-multilingual` — need to benchmark both on your actual memory corpus. QMD's model choice may inform this.

3. **Query expander model choice:** Qwen3-1.7B vs Phi-4-mini vs Gemma-3n-2B — need to test which gives best expansions for memory-style queries at acceptable latency.

4. **Embedding cache:** Cache embeddings by content hash so unchanged chunks survive re-indexing without re-embedding? Likely yes — saves API calls.

5. **Session transcript format:** OpenClaw uses JSONL, Hermes has its own format. The `/ingest` endpoint needs parsers for both. Build adapters or standardize?

6. **Migration of existing files:** We have 45+ daily logs in `~/alice/memory/logs/`, MEMORY.md at `~/alice/MEMORY.md`, plus per-agent files in `~/alice/agents/*/`. Need a one-time migration script to reorganize into the new `~/.memory-service/data/` directory structure.

7. **Reverse sync conflicts:** If a platform edits a synced file directly (e.g., Hermes's memory tool updates MEMORY.md), the service needs to detect this and sync the change back to canonical. Should we use file hashes, inotify on sync targets, or accept eventual consistency on a timer?

8. **Agent config hot-reload:** Should changes to the agent roster (e.g., adding a new agent, changing departments) require a service restart, or should the config be watchable and hot-reloaded?

---

## Success Criteria

- [ ] Search returns relevant results for queries that work in QMD + Graphiti today
- [ ] Reranking measurably improves precision over vector-only search (test with 20 known queries)
- [ ] Query expansion measurably improves recall (test with 20 queries using alternate phrasing)
- [ ] Write successfully persists to both daily log and Graphiti in a single call
- [ ] Namespace scoping works: agent search returns own + shared + department results, not other agents' private logs
- [ ] Platform sync: editing MEMORY.md via API updates both OpenClaw and Hermes copies within 1 second
- [ ] Reverse sync: editing a synced file directly on a platform propagates back to canonical
- [ ] File watcher detects changes to canonical files and re-indexes within 30 seconds
- [ ] Search latency < 600ms p95 (with expansion + reranking)
- [ ] Search latency < 200ms p95 (without expansion + reranking, for fast mode)
- [ ] Service runs stable for 48+ hours without intervention
- [ ] All four platform integrations work (OpenClaw, Hermes, MCP/Claude Desktop, CLI)
- [ ] Graceful degradation verified for each failure mode
- [ ] Memory usage stays under 2 GB RSS
- [ ] Existing 45+ daily logs and memory files successfully migrated to new namespace structure
