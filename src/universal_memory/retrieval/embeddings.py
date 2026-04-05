"""Embedding generation service with Gemini primary and OpenAI fallback."""

from __future__ import annotations

import hashlib
import logging
import os

from ..config import get_config

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generate embeddings via Gemini (primary) or OpenAI (fallback).

    - Primary: Google Gemini gemini-embedding-001 via google-genai SDK (3072 dims)
    - Fallback: OpenAI text-embedding-3-small (1536 dims)
    - Cache by SHA256 content hash (in-memory)
    - Batch up to config.embedding.batch_size items per request
    """

    GEMINI_DIMS = 3072
    OPENAI_DIMS = 1536

    def __init__(self) -> None:
        self._config = get_config()
        self._cache: dict[str, list[float]] = {}
        self._dimensions = self.GEMINI_DIMS
        self._provider_used: str | None = None

    @staticmethod
    def _content_hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    async def generate(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Returns empty list if both providers fail (graceful degradation).
        """
        logger.debug(
            "generate: texts=%d provider_used=%s cache_size=%d",
            len(texts), self._provider_used, len(self._cache),
        )

        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []

        # Check cache first
        for i, text in enumerate(texts):
            h = self._content_hash(text)
            if h in self._cache:
                results[i] = self._cache[h]
            else:
                uncached_indices.append(i)

        if not uncached_indices:
            return [r for r in results if r is not None]

        uncached_texts = [texts[i] for i in uncached_indices]

        # Stick with the provider that was used previously to avoid dimension mismatch
        if self._provider_used == "gemini":
            embeddings = await self._generate_gemini(uncached_texts)
            if embeddings:
                logger.debug("generate: gemini succeeded, count=%d", len(embeddings))
            else:
                logger.debug("generate: gemini failed, falling back to openai")
                logger.warning("Gemini failed with provider lock, falling back to OpenAI")
                embeddings = await self._generate_openai(uncached_texts)
                if embeddings:
                    logger.debug("generate: openai fallback succeeded, count=%d", len(embeddings))
                else:
                    logger.debug("generate: openai fallback also failed")
        elif self._provider_used == "openai":
            embeddings = await self._generate_openai(uncached_texts)
            if embeddings:
                logger.debug("generate: openai succeeded, count=%d", len(embeddings))
            else:
                logger.debug("generate: openai failed, falling back to gemini")
                logger.warning("OpenAI failed with provider lock, falling back to Gemini")
                embeddings = await self._generate_gemini(uncached_texts)
                if embeddings:
                    logger.debug("generate: gemini fallback succeeded, count=%d", len(embeddings))
                else:
                    logger.debug("generate: gemini fallback also failed")
        else:
            # First call: try Gemini first, fallback to OpenAI
            embeddings = await self._generate_gemini(uncached_texts)
            if embeddings:
                logger.debug("generate: gemini (first call) succeeded, count=%d", len(embeddings))
                self._provider_used = "gemini"
                self._dimensions = self.GEMINI_DIMS
            else:
                logger.debug("generate: gemini (first call) failed, trying openai")
                embeddings = await self._generate_openai(uncached_texts)
                if embeddings:
                    logger.debug("generate: openai (first call) succeeded, count=%d", len(embeddings))
                    self._provider_used = "openai"
                    self._dimensions = self.OPENAI_DIMS
                else:
                    logger.debug("generate: openai (first call) also failed")

        if not embeddings:
            logger.warning("All embedding providers failed, returning empty list")
            return []

        # Cache and assign results
        for idx, emb in zip(uncached_indices, embeddings):
            h = self._content_hash(texts[idx])
            self._cache[h] = emb
            results[idx] = emb

        final = [r for r in results if r is not None]
        logger.debug("generate: returning %d embeddings", len(final))
        return final

    async def _generate_gemini(self, texts: list[str]) -> list[list[float]] | None:
        """Generate embeddings via Google Gemini gemini-embedding-001.

        Runs the synchronous genai SDK call in a thread executor to avoid
        async/sync conflicts that cause AttributeError in event-loop contexts.
        """
        import asyncio

        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            try:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self._generate_gemini_sync, texts)
            except Exception:
                if attempt < max_attempts:
                    logger.info("Gemini embedding attempt %d failed, retrying in 1s", attempt)
                    await asyncio.sleep(1)
                else:
                    logger.warning("Gemini embedding failed after %d attempts", max_attempts, exc_info=True)
                    return None

    def _generate_gemini_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous Gemini embedding call (runs in thread executor)."""
        from google import genai

        cfg = self._config.embedding
        # Lookup order: configured env var -> GOOGLE_API_KEY -> GEMINI_API_KEY
        api_key = os.environ.get(cfg.api_key_env)
        if not api_key and cfg.api_key_env != "GOOGLE_API_KEY":
            api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key and cfg.api_key_env != "GEMINI_API_KEY":
            api_key = os.environ.get("GEMINI_API_KEY")

        # If no explicit key found, let the SDK try its own env lookup
        if api_key:
            client = genai.Client(api_key=api_key)
        else:
            # google-genai Client checks GOOGLE_API_KEY natively
            client = genai.Client()
        all_embeddings: list[list[float]] = []

        # Batch up to batch_size items per request
        for i in range(0, len(texts), cfg.batch_size):
            batch = texts[i : i + cfg.batch_size]
            result = client.models.embed_content(
                model=cfg.model,
                contents=batch,
            )
            for emb in result.embeddings:
                all_embeddings.append(list(emb.values))

        return all_embeddings

    async def _generate_openai(self, texts: list[str]) -> list[list[float]] | None:
        """Fallback: generate embeddings via OpenAI text-embedding-3-small."""
        try:
            import httpx

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OpenAI API key not found")
                return None

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": "text-embedding-3-small", "input": texts},
                    timeout=30.0,
                )
                resp.raise_for_status()
                data = resp.json()
                return [item["embedding"] for item in data["data"]]
        except Exception:
            logger.warning("OpenAI embedding failed", exc_info=True)
            return None

    @property
    def provider_used(self) -> str | None:
        """Return the name of the last successful provider ('gemini' or 'openai'), or None."""
        return self._provider_used

    @property
    def dimensions(self) -> int:
        """Return the actual dimension count used by the current provider."""
        return self._dimensions
