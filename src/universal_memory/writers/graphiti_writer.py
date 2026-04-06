"""Graphiti write bridge — posts content to the knowledge graph."""

from __future__ import annotations

import logging

from ..config import get_config, resolve_group_id
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

        # Auto-resolve using group_id_map / prefix when no explicit ID
        if not gid and author:
            gid = resolve_group_id(author, self._config)

        try:
            result = await self._client.write(content, group_id=gid, author=author)
            return result
        except Exception:
            logger.warning("GraphitiWriter.write failed", exc_info=True)
            return {}
