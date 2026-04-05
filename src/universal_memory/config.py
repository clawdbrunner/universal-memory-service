"""Configuration loader for Universal Memory Service."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_cached_config: FullConfig | None = None


def _expand(p: str) -> str:
    """Expand ~ and env vars in a path string."""
    return str(Path(os.path.expandvars(os.path.expanduser(p))).resolve())


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ServiceConfig:
    host: str = "127.0.0.1"
    port: int = 8002
    auth_token: str | None = None


@dataclass
class MemoryConfig:
    data_dir: str = "~/.memory-service/data"
    extensions: list[str] = field(default_factory=lambda: [".md"])
    daily_log_format: str = "{date}.md"

    def __post_init__(self) -> None:
        self.data_dir = _expand(self.data_dir)


@dataclass
class AgentInfo:
    department: str | None = None


@dataclass
class IndexConfig:
    db_path: str = "~/.memory-service/index.db"
    chunk_size_tokens: int = 400
    chunk_overlap_tokens: int = 80

    def __post_init__(self) -> None:
        self.db_path = _expand(self.db_path)


@dataclass
class EmbeddingConfig:
    provider: str = "gemini"
    model: str = "gemini-embedding-001"
    api_key_env: str = "GEMINI_API_KEY"
    batch_size: int = 100


@dataclass
class ModelSpec:
    enabled: bool = True
    model_path: str = ""
    candidates: int = 30
    blend_weight: float = 0.85
    max_expansions: int = 2

    def __post_init__(self) -> None:
        if self.model_path:
            self.model_path = _expand(self.model_path)


@dataclass
class ModelConfig:
    reranker: ModelSpec = field(default_factory=ModelSpec)
    query_expander: ModelSpec = field(default_factory=ModelSpec)


@dataclass
class SearchWeights:
    vector: float = 0.40
    bm25: float = 0.20
    graphiti: float = 0.25


@dataclass
class TemporalDecayConfig:
    enabled: bool = True
    half_life_days: int = 30
    exempt_files: list[str] = field(default_factory=lambda: ["MEMORY.md"])


@dataclass
class MMRConfig:
    enabled: bool = True
    lambda_: float = 0.7


@dataclass
class SearchConfig:
    weights: SearchWeights = field(default_factory=SearchWeights)
    temporal_decay: TemporalDecayConfig = field(default_factory=TemporalDecayConfig)
    mmr: MMRConfig = field(default_factory=MMRConfig)
    default_max_results: int = 10
    default_min_score: float = 0.3


@dataclass
class WriteConfig:
    daily_log_header_format: str = "## [{time}] {author}"
    append_newlines: int = 2
    file_lock: bool = True


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "~/.memory-service/logs/service.log"

    def __post_init__(self) -> None:
        if self.file:
            self.file = _expand(self.file)


@dataclass
class GraphitiConfig:
    url: str = "http://localhost:8001"
    timeout_seconds: int = 10


@dataclass
class SyncConfig:
    enabled: bool = True
    debounce_ms: int = 500
    targets: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class FullConfig:
    service: ServiceConfig = field(default_factory=ServiceConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    agents: dict[str, AgentInfo] = field(default_factory=dict)
    index: IndexConfig = field(default_factory=IndexConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    graphiti: GraphitiConfig = field(default_factory=GraphitiConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    write: WriteConfig = field(default_factory=WriteConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

_CONFIG_SEARCH_PATHS = [
    "~/.memory-service/config.yaml",
    "config/config.yaml",
]


def _dict_to_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Recursively convert a dict to a dataclass, ignoring unknown keys."""
    if not data:
        return cls()
    field_names = {f.name for f in cls.__dataclass_fields__.values()}
    return cls(**{k: v for k, v in data.items() if k in field_names})


def load_config(path: str | Path | None = None) -> FullConfig:
    """Load configuration from YAML.

    Search order:
      1. Explicit *path* argument
      2. MEMORY_SERVICE_CONFIG env var
      3. ~/.memory-service/config.yaml
      4. config/config.yaml  (repo-relative)

    Returns a FullConfig with sensible defaults for any missing keys.
    """
    raw: dict[str, Any] = {}

    candidates: list[str] = []
    if path:
        candidates.append(str(path))
    env = os.environ.get("MEMORY_SERVICE_CONFIG")
    if env:
        candidates.append(env)
    candidates.extend(_CONFIG_SEARCH_PATHS)

    for candidate in candidates:
        resolved = Path(os.path.expanduser(candidate)).resolve()
        if resolved.is_file():
            with open(resolved) as f:
                raw = yaml.safe_load(f) or {}
            break

    return _parse_raw(raw)


def _parse_raw(raw: dict[str, Any]) -> FullConfig:
    service = _dict_to_dataclass(ServiceConfig, raw.get("service", {}))
    memory = _dict_to_dataclass(MemoryConfig, raw.get("memory", {}))

    agents: dict[str, AgentInfo] = {}
    for name, info in raw.get("agents", {}).items():
        if isinstance(info, dict):
            agents[name] = AgentInfo(department=info.get("department"))
        else:
            agents[name] = AgentInfo()

    index = _dict_to_dataclass(IndexConfig, raw.get("index", {}))
    embedding = _dict_to_dataclass(EmbeddingConfig, raw.get("embedding", {}))

    models_raw = raw.get("models", {})
    reranker = _dict_to_dataclass(ModelSpec, models_raw.get("reranker", {}))
    expander = _dict_to_dataclass(ModelSpec, models_raw.get("query_expander", {}))
    model_config = ModelConfig(reranker=reranker, query_expander=expander)

    search_raw = raw.get("search", {})
    weights = _dict_to_dataclass(SearchWeights, search_raw.get("weights", {}))
    temporal_decay = _dict_to_dataclass(
        TemporalDecayConfig, search_raw.get("temporal_decay", {})
    )
    mmr_raw = search_raw.get("mmr", {})
    mmr = MMRConfig(
        enabled=mmr_raw.get("enabled", True),
        lambda_=mmr_raw.get("lambda", 0.7),
    ) if mmr_raw else MMRConfig()
    search_config = SearchConfig(
        weights=weights,
        temporal_decay=temporal_decay,
        mmr=mmr,
        default_max_results=search_raw.get("default_max_results", 10),
        default_min_score=search_raw.get("default_min_score", 0.5),
    )

    graphiti = _dict_to_dataclass(GraphitiConfig, raw.get("graphiti", {}))
    sync = _dict_to_dataclass(SyncConfig, raw.get("sync", {}))
    write = _dict_to_dataclass(WriteConfig, raw.get("write", {}))

    logging_raw = raw.get("logging", {})
    logging_config = LoggingConfig(
        level=logging_raw.get("level", "INFO"),
        file=logging_raw.get("file", "~/.memory-service/logs/service.log"),
    ) if logging_raw else LoggingConfig()

    return FullConfig(
        service=service,
        memory=memory,
        agents=agents,
        index=index,
        embedding=embedding,
        models=model_config,
        search=search_config,
        graphiti=graphiti,
        sync=sync,
        write=write,
        logging=logging_config,
    )


def get_config() -> FullConfig:
    """Return the cached global config, loading it on first call."""
    global _cached_config
    if _cached_config is None:
        _cached_config = load_config()
    return _cached_config


def reset_config() -> None:
    """Clear the cached config (useful for tests)."""
    global _cached_config
    _cached_config = None
