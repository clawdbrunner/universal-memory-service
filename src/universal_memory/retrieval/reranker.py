"""Local GGUF cross-encoder reranker."""

from __future__ import annotations

import logging
import math
from pathlib import Path

from ..config import get_config
from ..models import SearchResult

logger = logging.getLogger(__name__)


class RerankerService:
    """Cross-encoder reranker using a GGUF model via llama-cpp-python.

    Blend: final_score = 0.85 * rerank_score + 0.15 * original_score
    Falls back to returning candidates as-is if model unavailable.
    """

    def __init__(self) -> None:
        self._model = None
        self._config = get_config()
        self._blend_weight = self._config.models.reranker.blend_weight
        self._model_status: str = "disabled" if not self._config.models.reranker.enabled else "loaded"
        self._model_error: str | None = None

    @property
    def model_status(self) -> str:
        return self._model_status

    @property
    def model_error(self) -> str | None:
        return self._model_error

    def _ensure_model(self) -> bool:
        """Lazy-load the reranker model. Returns True if available."""
        if self._model is not None:
            return True
        spec = self._config.models.reranker
        if not spec.enabled:
            self._model_status = "disabled"
            return False
        model_path = Path(spec.model_path).expanduser()
        if not model_path.exists():
            self._model_status = "model_file_missing"
            self._model_error = f"File not found: {model_path}"
            logger.warning("Reranker model not found at %s", model_path)
            return False
        try:
            from llama_cpp import Llama

            self._model = Llama(
                model_path=str(model_path),
                n_ctx=512,
                n_threads=4,
                verbose=False,
                embedding=True,
            )
            self._model_status = "loaded"
            self._model_error = None
            logger.info("Reranker model loaded from %s", model_path)
            return True
        except Exception as e:
            self._model_status = "load_error"
            self._model_error = str(e)
            logger.warning("Failed to load reranker model", exc_info=True)
            return False

    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_n: int = 30,
    ) -> list[SearchResult]:
        """Rerank candidates using cross-encoder scoring."""
        if not candidates:
            return []

        if not self._ensure_model():
            logger.info("Reranker unavailable, returning candidates as-is")
            return candidates[:top_n]

        scored: list[tuple[SearchResult, float]] = []
        for candidate in candidates:
            try:
                # bge-reranker-v2-m3 is a cross-encoder; in GGUF embedding
                # mode the first dimension of the output IS the relevance
                # logit.  The model expects a query-document pair separated
                # by </s></s> (BOS is added automatically by the tokenizer).
                pair_text = f"{query}</s></s>{candidate.content[:300]}</s>"
                output = self._model.embed(pair_text)
                if isinstance(output[0], list):
                    raw_score = float(output[0][0])
                else:
                    raw_score = float(output[0])
                # Sigmoid to convert logit to 0-1 relevance probability
                rerank_score = 1.0 / (1.0 + math.exp(-raw_score))
            except Exception:
                rerank_score = 0.5

            final_score = (
                self._blend_weight * rerank_score
                + (1 - self._blend_weight) * candidate.score
            )
            scored.append((candidate, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)

        results: list[SearchResult] = []
        for candidate, score in scored[:top_n]:
            candidate.score = score
            results.append(candidate)
        return results
