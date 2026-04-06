"""Query expansion using a small local GGUF LLM."""

from __future__ import annotations

import logging
import re
import time
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


class QueryExpanderService:
    """Expand queries with alternative phrasings using a local LLM.

    Generates 2 alternative phrasings. Skips expansion for queries
    matching skip patterns. Returns [query] only if model unavailable.
    """

    def __init__(self) -> None:
        self._model = None
        self._config = get_config()

    def _ensure_model(self) -> bool:
        """Lazy-load the query expander model. Returns True if available."""
        if self._model is not None:
            return True
        spec = self._config.models.query_expander
        if not spec.enabled:
            return False
        model_path = Path(spec.model_path).expanduser()
        if not model_path.exists():
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
            logger.info("Query expander model loaded from %s", model_path)
            return True
        except Exception:
            logger.warning("Failed to load query expander model", exc_info=True)
            return False

    def _should_skip(self, query: str) -> bool:
        """Return True if query matches skip patterns."""
        return any(p.search(query.strip()) for p in SKIP_PATTERNS)

    async def expand(self, query: str) -> list[str]:
        """Generate alternative phrasings for a query."""
        queries = [query]

        if self._should_skip(query):
            return queries

        if not self._ensure_model():
            now = time.time()
            if now - getattr(self, "_last_expand_warn", 0) > 60:
                logger.warning("Query expansion skipped: model not loaded")
                self._last_expand_warn = now
            return queries

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

        return queries
