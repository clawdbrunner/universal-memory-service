"""Graphiti write bridge — posts content to the knowledge graph."""

from __future__ import annotations

import logging

from ..config import get_config
from ..retrieval.graphiti import GraphitiClient

logger = logging.getLogger(__name__)


class GraphitiWriter:
    """Thin layer that resolves group IDs and delegates to GraphitiClient."""

    def __init__(self) -> None:
        self._client = GraphitiClient()
        self._config = get_config()

    async def write(
        self,
        content: str,
        author: str = "",
        group_ids: list[str] | None = None,
    ) -> dict:
        """Write *content* to Graphiti, resolving group IDs if needed."""
        ids = list(group_ids) if group_ids else []

        # Auto-resolve from agent config when no explicit IDs
        if not ids and author:
            agent = self._config.agents.get(author)
            if agent and agent.department:
                ids.append(agent.department)
            ids.append(author)

        try:
            result = await self._client.write(content, group_ids=ids, author=author)
            return result
        except Exception:
            logger.warning("GraphitiWriter.write failed", exc_info=True)
            return {}
