"""Graphiti HTTP client for knowledge graph search and write."""

from __future__ import annotations

import logging

import httpx

from ..config import get_config
from ..models import SearchResult

logger = logging.getLogger(__name__)


class GraphitiClient:
    """HTTP client for the Graphiti knowledge graph service.

    All async via httpx. Graceful: if Graphiti is unreachable,
    returns empty results and logs a warning.
    """

    def __init__(self) -> None:
        cfg = get_config()
        self._base_url = cfg.graphiti.url
        self._timeout = cfg.graphiti.timeout_seconds

    async def search(
        self,
        query: str,
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Search the knowledge graph."""
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/search",
                    json={
                        "query": query,
                        "group_ids": group_ids or [],
                        "num_results": limit,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                results: list[SearchResult] = []
                for item in data.get("results", []):
                    results.append(SearchResult(
                        chunk_id=item.get("id", ""),
                        score=float(item.get("score", 0.0)),
                        source="graphiti",
                        content=item.get("content", ""),
                        file_path=item.get("file_path", ""),
                        metadata=item.get("metadata", {}),
                    ))
                return results
        except Exception:
            logger.warning("Graphiti search unavailable", exc_info=True)
            return []

    async def write(
        self,
        content: str,
        group_ids: list[str] | None = None,
        author: str = "",
    ) -> dict:
        """Write content to the knowledge graph."""
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/messages",
                    json={
                        "content": content,
                        "group_ids": group_ids or [],
                        "author": author,
                    },
                )
                resp.raise_for_status()
                return resp.json()
        except Exception:
            logger.warning("Graphiti write failed", exc_info=True)
            return {}

    async def health(self) -> bool:
        """Check if Graphiti service is healthy."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self._base_url}/health")
                return resp.status_code == 200
        except Exception:
            return False
