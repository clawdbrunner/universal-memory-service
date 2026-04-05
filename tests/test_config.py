"""Unit tests for configuration loading."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from universal_memory.config import (
    EmbeddingConfig,
    FullConfig,
    GraphitiConfig,
    IndexConfig,
    LoggingConfig,
    MMRConfig,
    MemoryConfig,
    ModelConfig,
    ModelSpec,
    SearchConfig,
    SearchWeights,
    ServiceConfig,
    SyncConfig,
    TemporalDecayConfig,
    WriteConfig,
    _dict_to_dataclass,
    _expand,
    _parse_raw,
    load_config,
    reset_config,
)


# ---------------------------------------------------------------------------
# Path expansion
# ---------------------------------------------------------------------------


class TestExpand:
    def test_tilde_expansion(self):
        result = _expand("~/foo")
        assert "~" not in result
        assert result.endswith("/foo")

    def test_env_var_expansion(self, monkeypatch):
        monkeypatch.setenv("MY_TEST_DIR", "/custom/path")
        result = _expand("$MY_TEST_DIR/data")
        assert result.endswith("/custom/path/data")

    def test_returns_resolved_path(self):
        result = _expand("/tmp/../tmp/test")
        assert ".." not in result


# ---------------------------------------------------------------------------
# Dataclass defaults
# ---------------------------------------------------------------------------


class TestDataclassDefaults:
    def test_service_defaults(self):
        cfg = ServiceConfig()
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 8002
        assert cfg.auth_token is None

    def test_memory_defaults(self):
        cfg = MemoryConfig()
        assert cfg.extensions == [".md"]
        assert cfg.daily_log_format == "{date}.md"
        # data_dir should be expanded (no tilde)
        assert "~" not in cfg.data_dir

    def test_memory_expands_data_dir(self):
        cfg = MemoryConfig(data_dir="~/test-dir")
        assert "~" not in cfg.data_dir
        assert cfg.data_dir.endswith("/test-dir")

    def test_index_defaults(self):
        cfg = IndexConfig()
        assert cfg.chunk_size_tokens == 400
        assert cfg.chunk_overlap_tokens == 80
        assert "~" not in cfg.db_path

    def test_index_expands_db_path(self):
        cfg = IndexConfig(db_path="~/my.db")
        assert "~" not in cfg.db_path

    def test_embedding_defaults(self):
        cfg = EmbeddingConfig()
        assert cfg.provider == "gemini"
        assert cfg.model == "gemini-embedding-001"
        assert cfg.batch_size == 100

    def test_model_spec_defaults(self):
        spec = ModelSpec()
        assert spec.enabled is True
        assert spec.model_path == ""
        assert spec.candidates == 30
        assert spec.blend_weight == 0.85
        assert spec.max_expansions == 2

    def test_model_spec_expands_path(self):
        spec = ModelSpec(model_path="~/models/test.gguf")
        assert "~" not in spec.model_path
        assert spec.model_path.endswith("/models/test.gguf")

    def test_search_weights_defaults(self):
        w = SearchWeights()
        assert w.vector == 0.40
        assert w.bm25 == 0.20
        assert w.graphiti == 0.25

    def test_graphiti_defaults(self):
        cfg = GraphitiConfig()
        assert cfg.url == "http://localhost:8001"
        assert cfg.timeout_seconds == 10

    def test_sync_defaults(self):
        cfg = SyncConfig()
        assert cfg.enabled is True
        assert cfg.debounce_ms == 500
        assert cfg.targets == []

    def test_temporal_decay_defaults(self):
        cfg = TemporalDecayConfig()
        assert cfg.enabled is True
        assert cfg.half_life_days == 30
        assert cfg.exempt_files == ["MEMORY.md"]

    def test_mmr_defaults(self):
        cfg = MMRConfig()
        assert cfg.enabled is True
        assert cfg.lambda_ == 0.7

    def test_search_config_defaults(self):
        cfg = SearchConfig()
        assert isinstance(cfg.weights, SearchWeights)
        assert isinstance(cfg.temporal_decay, TemporalDecayConfig)
        assert isinstance(cfg.mmr, MMRConfig)
        assert cfg.default_max_results == 10
        assert cfg.default_min_score == 0.5

    def test_write_defaults(self):
        cfg = WriteConfig()
        assert cfg.daily_log_header_format == "## [{time}] {author}"
        assert cfg.append_newlines == 2
        assert cfg.file_lock is True

    def test_logging_defaults(self):
        cfg = LoggingConfig()
        assert cfg.level == "INFO"
        assert "~" not in cfg.file

    def test_logging_expands_file_path(self):
        cfg = LoggingConfig(file="~/logs/test.log")
        assert "~" not in cfg.file
        assert cfg.file.endswith("/logs/test.log")

    def test_full_config_defaults(self):
        cfg = FullConfig()
        assert isinstance(cfg.service, ServiceConfig)
        assert isinstance(cfg.memory, MemoryConfig)
        assert isinstance(cfg.index, IndexConfig)
        assert isinstance(cfg.embedding, EmbeddingConfig)
        assert isinstance(cfg.models, ModelConfig)
        assert isinstance(cfg.search, SearchConfig)
        assert isinstance(cfg.graphiti, GraphitiConfig)
        assert isinstance(cfg.sync, SyncConfig)
        assert isinstance(cfg.write, WriteConfig)
        assert isinstance(cfg.logging, LoggingConfig)
        assert cfg.agents == {}


# ---------------------------------------------------------------------------
# _dict_to_dataclass
# ---------------------------------------------------------------------------


class TestDictToDataclass:
    def test_empty_dict(self):
        result = _dict_to_dataclass(ServiceConfig, {})
        assert result.host == "127.0.0.1"

    def test_none_returns_defaults(self):
        result = _dict_to_dataclass(ServiceConfig, None)
        assert result.port == 8002

    def test_partial_override(self):
        result = _dict_to_dataclass(ServiceConfig, {"port": 9000})
        assert result.port == 9000
        assert result.host == "127.0.0.1"  # default preserved

    def test_ignores_unknown_keys(self):
        result = _dict_to_dataclass(ServiceConfig, {"port": 9000, "unknown_key": "x"})
        assert result.port == 9000
        assert not hasattr(result, "unknown_key")


# ---------------------------------------------------------------------------
# _parse_raw (YAML dict → FullConfig)
# ---------------------------------------------------------------------------


class TestParseRaw:
    def test_empty_raw(self):
        cfg = _parse_raw({})
        assert isinstance(cfg, FullConfig)
        assert cfg.service.port == 8002

    def test_service_override(self):
        cfg = _parse_raw({"service": {"port": 3000}})
        assert cfg.service.port == 3000

    def test_agents_parsed(self):
        cfg = _parse_raw({
            "agents": {
                "alice": {"department": None},
                "bob": {"department": "engineering"},
            }
        })
        assert "alice" in cfg.agents
        assert cfg.agents["bob"].department == "engineering"
        assert cfg.agents["alice"].department is None

    def test_agents_without_dict_value(self):
        """Agent entries that aren't dicts should still parse."""
        cfg = _parse_raw({"agents": {"alice": "something"}})
        assert "alice" in cfg.agents
        assert cfg.agents["alice"].department is None

    def test_search_weights_nested(self):
        cfg = _parse_raw({"search": {"weights": {"vector": 0.5, "bm25": 0.3}}})
        assert cfg.search.weights.vector == 0.5
        assert cfg.search.weights.bm25 == 0.3
        assert cfg.search.weights.graphiti == 0.25  # default

    def test_search_temporal_decay_parsed(self):
        cfg = _parse_raw({"search": {"temporal_decay": {"half_life_days": 60, "enabled": False}}})
        assert cfg.search.temporal_decay.half_life_days == 60
        assert cfg.search.temporal_decay.enabled is False

    def test_search_mmr_lambda_parsed(self):
        cfg = _parse_raw({"search": {"mmr": {"lambda": 0.5, "enabled": False}}})
        assert cfg.search.mmr.lambda_ == 0.5
        assert cfg.search.mmr.enabled is False

    def test_search_defaults_parsed(self):
        cfg = _parse_raw({"search": {"default_max_results": 20, "default_min_score": 0.5}})
        assert cfg.search.default_max_results == 20
        assert cfg.search.default_min_score == 0.5

    def test_write_parsed(self):
        cfg = _parse_raw({"write": {"daily_log_header_format": "# {author}", "append_newlines": 1}})
        assert cfg.write.daily_log_header_format == "# {author}"
        assert cfg.write.append_newlines == 1

    def test_logging_parsed(self):
        cfg = _parse_raw({"logging": {"level": "DEBUG", "file": "/tmp/test.log"}})
        assert cfg.logging.level == "DEBUG"
        assert cfg.logging.file.endswith("test.log")

    def test_models_nested(self):
        cfg = _parse_raw({
            "models": {
                "reranker": {"enabled": False},
                "query_expander": {"max_expansions": 5},
            }
        })
        assert cfg.models.reranker.enabled is False
        assert cfg.models.query_expander.max_expansions == 5


# ---------------------------------------------------------------------------
# load_config (file-based)
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_load_from_explicit_path(self, tmp_path):
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(textwrap.dedent("""\
            service:
              port: 7777
            memory:
              data_dir: /tmp/test-memory-data
        """))
        cfg = load_config(config_file)
        assert cfg.service.port == 7777
        assert cfg.memory.data_dir.endswith("test-memory-data")

    def test_load_from_env_var(self, tmp_path, monkeypatch):
        config_file = tmp_path / "env_config.yaml"
        config_file.write_text("service:\n  port: 5555\n")
        monkeypatch.setenv("MEMORY_SERVICE_CONFIG", str(config_file))
        cfg = load_config()
        assert cfg.service.port == 5555

    def test_missing_file_returns_defaults(self, tmp_path, monkeypatch):
        # Point away from any real config
        monkeypatch.delenv("MEMORY_SERVICE_CONFIG", raising=False)
        cfg = load_config(tmp_path / "nonexistent.yaml")
        assert isinstance(cfg, FullConfig)
        assert cfg.service.port == 8002  # default

    def test_empty_yaml_returns_defaults(self, tmp_path):
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        cfg = load_config(config_file)
        assert cfg.service.port == 8002

    def test_full_config_roundtrip(self, tmp_path):
        config_file = tmp_path / "full.yaml"
        config_file.write_text(textwrap.dedent("""\
            service:
              host: "0.0.0.0"
              port: 9999
            memory:
              data_dir: /tmp/full-test
              extensions: [".md", ".txt"]
            agents:
              alice: { department: null }
              bob: { department: engineering }
            index:
              chunk_size_tokens: 200
            embedding:
              provider: openai
              model: text-embedding-3-small
            models:
              reranker:
                enabled: false
              query_expander:
                max_expansions: 3
            search:
              weights:
                vector: 0.5
                bm25: 0.3
                graphiti: 0.2
            graphiti:
              url: http://localhost:9001
              timeout_seconds: 5
            sync:
              enabled: false
        """))
        cfg = load_config(config_file)
        assert cfg.service.host == "0.0.0.0"
        assert cfg.service.port == 9999
        assert cfg.memory.extensions == [".md", ".txt"]
        assert cfg.agents["bob"].department == "engineering"
        assert cfg.index.chunk_size_tokens == 200
        assert cfg.embedding.provider == "openai"
        assert cfg.models.reranker.enabled is False
        assert cfg.models.query_expander.max_expansions == 3
        assert cfg.search.weights.vector == 0.5
        assert cfg.graphiti.url == "http://localhost:9001"
        assert cfg.sync.enabled is False
