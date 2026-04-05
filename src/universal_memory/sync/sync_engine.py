"""Platform sync engine — copies canonical files to configured targets."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..config import get_config

logger = logging.getLogger(__name__)


class SyncEngine:
    """Copy canonical memory files to configured sync targets."""

    def __init__(self) -> None:
        self._config = get_config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def sync_file(self, source_path: str) -> list[str]:
        """Copy *source_path* to every matching sync target.

        Returns list of destination paths written.
        """
        if not self._config.sync.enabled:
            return []

        src = Path(source_path)
        if not src.is_file():
            return []

        written: list[str] = []
        for target in self._config.sync.targets:
            dest_template: str = target.get("dest", "")
            if not dest_template:
                continue
            dest = Path(self.resolve_templates(dest_template))
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dest))
                written.append(str(dest))
                logger.debug("Synced %s → %s", src, dest)
            except Exception:
                logger.warning("Sync failed %s → %s", src, dest, exc_info=True)

        return written

    async def sync_all(self) -> int:
        """Sync all configured files on startup. Returns count synced."""
        if not self._config.sync.enabled:
            return 0

        count = 0
        data_dir = Path(self._config.memory.data_dir)
        extensions = set(self._config.memory.extensions)
        for path in data_dir.rglob("*"):
            if path.is_file() and path.suffix in extensions:
                results = await self.sync_file(str(path))
                count += len(results)
        return count

    # ------------------------------------------------------------------
    # Template helpers
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_templates(path: str) -> str:
        """Replace ``{today}`` and ``{yesterday}`` in *path*."""
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")
        yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        return path.replace("{today}", today).replace("{yesterday}", yesterday)
