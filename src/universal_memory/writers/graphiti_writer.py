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
        group_id: str = "",
    ) -> dict:
        """Write *content* to Graphiti, resolving group ID if needed."""
        gid = group_id

        # Auto-resolve from agent config when no explicit ID
        if not gid and author:
            agent = self._config.agents.get(author)
            if agent and agent.department:
                gid = agent.department
            else:
                gid = author

        try:
            result = await self._client.write(content, group_id=gid, author=author)
            return result
        except Exception:
            logger.warning("GraphitiWriter.write failed", exc_info=True)
            return {}
