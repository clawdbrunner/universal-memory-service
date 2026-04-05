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
                payload: dict = {"query": query, "max_facts": limit}
                if group_ids:
                    payload["group_ids"] = group_ids
                resp = await client.post(f"{self._base_url}/search", json=payload)
                resp.raise_for_status()
                data = resp.json()
                results: list[SearchResult] = []
                for item in data.get("facts", []):
                    results.append(SearchResult(
                        chunk_id="",
                        score=1.0,
                        source="graphiti",
                        content=item.get("fact", ""),
                        file_path="",
                        metadata={"valid_at": item.get("valid_at", ""), "entities": item.get("entities", [])},
                    ))
                return results
        except Exception:
            logger.warning("Graphiti search failed", exc_info=True)
            return []

    async def write(
        self,
        content: str,
        group_id: str = "",
        author: str = "assistant",
    ) -> dict:
        """Write content to the knowledge graph via /messages endpoint."""
        import datetime

        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/messages",
                    json={
                        "group_id": group_id,
                        "messages": [{
                            "role_type": "assistant",
                            "role": author,
                            "content": content,
                            "timestamp": timestamp,
                        }],
                    },
                )
                resp.raise_for_status()
                return {"status": "accepted", "group_id": group_id}
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
