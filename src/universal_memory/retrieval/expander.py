"""Query expansion using a small local GGUF LLM."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import get_config

logger = logging.getLogger(__name__)

# Patterns that should skip expansion (IDs, file paths, error codes)
SKIP_PATTERNS = [
    re.compile(r"^[a-f0-9]{8,}$", re.IGNORECASE),  # Hex IDs / commit hashes
    re.compile(r"[\\/]"),  # File paths
    re.compile(r"^[A-Z]+-\d+$"),  # Issue/ticket IDs like JIRA-123
    re.compile(r"^(ERR|ERROR|WARN)[-_]\d+", re.IGNORECASE),  # Error codes
    re.compile(r"^\S+\.\S+$"),  # Dotted identifiers like module.function
]


@dataclass
class ExpandResult:
    """Result of query expansion including status information."""
    queries: list[str] = field(default_factory=list)
    status: str = "success"  # "success" | "skipped_pattern" | "model_unavailable" | "error"
    error: str | None = None


class QueryExpanderService:
    """Expand queries with alternative phrasings using a local LLM.

    Generates 2 alternative phrasings. Skips expansion for queries
    matching skip patterns. Returns [query] only if model unavailable.
    """

    def __init__(self) -> None:
        self._model = None
        self._config = get_config()
        self._model_status: str = "disabled" if not self._config.models.query_expander.enabled else "loaded"
        self._model_error: str | None = None

    @property
    def model_status(self) -> str:
        return self._model_status

    @property
    def model_error(self) -> str | None:
        return self._model_error

    def _ensure_model(self) -> bool:
        """Lazy-load the query expander model. Returns True if available."""
        if self._model is not None:
            return True
        spec = self._config.models.query_expander
        if not spec.enabled:
            self._model_status = "disabled"
            return False
        model_path = Path(spec.model_path).expanduser()
        if not model_path.exists():
            self._model_status = "model_file_missing"
            self._model_error = f"File not found: {model_path}"
            logger.warning("Query expander model not found at %s", model_path)
            return False
        try:
            from llama_cpp import Llama

            self._model = Llama(
                model_path=str(model_path),
                n_ctx=512,
                n_threads=4,
                verbose=False,
            )
            self._model_status = "loaded"
            self._model_error = None
            logger.info("Query expander model loaded from %s", model_path)
            return True
        except Exception as e:
            self._model_status = "load_error"
            self._model_error = str(e)
            logger.warning("Failed to load query expander model", exc_info=True)
            return False

    def _should_skip(self, query: str) -> bool:
        """Return True if query matches skip patterns."""
        return any(p.search(query.strip()) for p in SKIP_PATTERNS)

    async def expand(self, query: str) -> ExpandResult:
        """Generate alternative phrasings for a query."""
        queries = [query]

        if self._should_skip(query):
            return ExpandResult(queries=queries, status="skipped_pattern")

        if not self._ensure_model():
            now = time.time()
            if now - getattr(self, "_last_expand_warn", 0) > 60:
                logger.warning("Query expansion skipped: model not loaded")
                self._last_expand_warn = now
            return ExpandResult(queries=queries, status="model_unavailable")

        try:
            max_expansions = self._config.models.query_expander.max_expansions
            output = self._model.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You generate alternative search queries. Return only the queries, one per line. /no_think"},
                    {"role": "user", "content": f'Generate {max_expansions} alternative phrasings for: "{query}" /no_think'},
                ],
                max_tokens=128,
                temperature=0.7,
            )
            text = output["choices"][0]["message"]["content"].strip()
            # Strip Qwen3 thinking tags if present
            import re as _re
            text = _re.sub(r'<think\b[^>]*>.*?</think\s*>', '', text, flags=_re.DOTALL).strip()
            for line in text.split("\n"):
                line = line.strip().strip("0123456789.-) ")
                # Skip preambles (lines ending with :), too short, or echoing original
                if not line or line == query or len(line) <= 3 or line.endswith(":"):
                    continue
                # Skip lines that are mostly the original query with noise appended
                if query.lower() in line.lower() and len(line) < len(query) + 20:
                    # Allow minor variations like "utility payment plans"
                    if line.lower().strip() == query.lower():
                        continue
                queries.append(line)
                if len(queries) >= max_expansions + 1:
                    break
        except Exception as exc:
            logger.warning("Query expansion failed: %s", exc, exc_info=True)
            return ExpandResult(queries=queries, status="error", error=str(exc))

        return ExpandResult(queries=queries, status="success")
