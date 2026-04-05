# Hermes Memory Tools

Use these tools to search, write, read, edit, and manage the shared memory service. The memory service stores knowledge across sessions in markdown files and the Graphiti knowledge graph.

Base URL: `http://localhost:8002/api/v1`

## Tools

### memory_search

Search across memory files and the Graphiti knowledge graph. Returns ranked results from vector search, BM25 full-text, and Graphiti temporal facts.

**When to use:** When you need to recall prior context, find past decisions, look up project knowledge, or retrieve any previously stored information.

```
POST /api/v1/search
Content-Type: application/json

{
  "query": "<search query>",
  "author": "<agent name>",
  "department": "<optional department scope>",
  "sources": ["files", "graphiti"],
  "max_results": 10,
  "min_score": 0.3,
  "expand": true,
  "rerank": true
}
```

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| query | string | yes | | Natural language search query |
| author | string | no | | Agent name for scoping results |
| department | string | no | | Department scope override |
| sources | string[] | no | ["files", "graphiti"] | Which backends to query |
| max_results | int | no | 10 | Maximum results to return |
| min_score | float | no | 0.3 | Minimum relevance score (0-1) |
| expand | bool | no | true | Enable query expansion |
| rerank | bool | no | true | Enable cross-encoder reranking |

**Response:** `{ results: [{chunk_id, score, source, content, file_path, line_start, line_end, header_path, metadata}], query, expanded_queries, sources_queried, timing_ms }`

### memory_write

Write content to memory files and/or the Graphiti knowledge graph. Automatically resolves file paths from author and target.

**When to use:** After completing tasks, making decisions, learning user preferences, or any time you have information worth persisting across sessions.

```
POST /api/v1/write
Content-Type: application/json

{
  "content": "<content to write>",
  "author": "<agent name>",
  "target": "daily",
  "targets": ["file", "graphiti"]
}
```

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| content | string | yes | | Content to write |
| author | string | yes | | Agent name |
| target | string | no | "daily" | Where to write: daily, long-term, department, shared, file |
| department | string | no | | Department override |
| file_path | string | no | | Custom path (when target=file) |
| targets | string[] | no | ["file", "graphiti"] | Which backends to persist to |

**Targets:**
- `daily` — `agents/{author}/logs/YYYY-MM-DD.md`
- `long-term` — `agents/{author}/MEMORY.md`
- `department` — `departments/{dept}/YYYY-MM-DD.md`
- `shared` — `shared/YYYY-MM-DD.md`
- `file` — custom `file_path`

**Response:** `{ ok, written_to, synced_to, index_updated }`

### memory_read

Read a specific file from the memory store.

**When to use:** When you need to read the full contents of a known file, such as a MEMORY.md or a daily log.

```
GET /api/v1/read/{path}
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| path | string (URL path) | yes | File path relative to memory root |

**Response:** `{ path, content }`

### memory_list

List files under a namespace in the memory store.

**When to use:** To discover what files exist for an agent, department, or shared namespace.

```
GET /api/v1/list/{namespace}
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| namespace | string (URL path) | yes | Namespace path (e.g. agents/alice/logs, shared, departments/engineering) |

**Response:** `{ namespace, files: [string] }`

### memory_edit

Surgical find-and-replace in a memory file. The old_text must match exactly once.

**When to use:** When you need to update a specific piece of text in an existing memory file, such as correcting a fact in MEMORY.md.

```
POST /api/v1/edit
Content-Type: application/json

{
  "path": "shared/MEMORY.md",
  "old_text": "exact text to find",
  "new_text": "replacement text",
  "targets": ["file", "graphiti"]
}
```

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| path | string | yes | | File path relative to memory root |
| old_text | string | yes | | Exact text to find (must match once) |
| new_text | string | yes | | Replacement text |
| targets | string[] | no | ["file", "graphiti"] | Which backends to update |

**Response:** `{ ok, path, index_updated, graphiti_updated? }`

### memory_ingest

Batch ingest messages into the Graphiti knowledge graph.

**When to use:** When you have a batch of conversation messages or structured data to persist to the knowledge graph.

```
POST /api/v1/ingest
Content-Type: application/json

{
  "messages": [
    {"content": "message text", "author": "alice"},
    {"content": "another message", "author": "bob"}
  ],
  "source": "conversation",
  "group_id": "optional-group"
}
```

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| messages | object[] | yes | | List of {content, author?} objects |
| group_id | string | no | "" | Graphiti group ID |
| source | string | no | "" | Source identifier (used as fallback author) |
| session_id | string | no | "" | Session identifier |

**Response:** `{ ok, ingested }`

### memory_status

Get service health, index statistics, and component status.

**When to use:** To check if the memory service is running and inspect index statistics.

```
GET /api/v1/status
```

**Response:** `{ status, uptime_seconds, index: {files_indexed, chunks, embeddings}, file_watcher: {running} }`

## File Namespaces

```
~/.memory-service/data/
├── shared/              # Cross-agent knowledge (MEMORY.md, USER.md)
├── agents/{name}/logs/  # Per-agent daily logs
���── departments/{dept}/  # Department-level knowledge
├── projects/            # Cross-cutting project docs
├── guides/              # How-to docs
└── system/              # Internal state
```

## Best Practices

1. **Search before writing** — Check if knowledge already exists before creating duplicates.
2. **Use long-term for persistent facts** — User preferences, project context, and stable decisions go in MEMORY.md via `target: "long-term"`.
3. **Use daily for activity logs** — Session summaries, completed tasks, and ephemeral notes go in daily logs via `target: "daily"`.
4. **Use shared for cross-agent knowledge** — Facts that all agents should know go in `target: "shared"`.
5. **Edit, don't append** �� When updating a fact (e.g., a preference changed), use `memory_edit` to replace the old value rather than appending a contradiction.
