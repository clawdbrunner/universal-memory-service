"""File-based write operations for the memory service."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import aiofiles

from ..config import FullConfig, get_config

logger = logging.getLogger(__name__)


class FileWriter:
    """Handles all file write, edit, and read operations."""

    def __init__(self) -> None:
        self._config = get_config()

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_path(
        author: str,
        target: str,
        file_path: str | None = None,
        config: FullConfig | None = None,
    ) -> Path:
        """Map *author* + *target* to a canonical file path.

        Targets:
          daily      → agents/{author}/logs/YYYY-MM-DD.md
          long-term  → agents/{author}/MEMORY.md
          department → departments/{dept}/YYYY-MM-DD.md
          shared     → shared/YYYY-MM-DD.md
          file       → custom *file_path* (relative to data_dir)
        """
        cfg = config or get_config()
        base = Path(cfg.memory.data_dir)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if target == "daily":
            return base / "agents" / author / "logs" / f"{today}.md"
        if target == "long-term":
            return base / "agents" / author / "MEMORY.md"
        if target == "department":
            dept = "general"
            agent_info = cfg.agents.get(author)
            if agent_info and agent_info.department:
                dept = agent_info.department
            return base / "departments" / dept / f"{today}.md"
        if target == "shared":
            return base / "shared" / f"{today}.md"
        if target == "file" and file_path:
            p = Path(file_path)
            if p.is_absolute():
                return p
            return base / p
        # fallback to daily
        return base / "agents" / author / "logs" / f"{today}.md"

    # ------------------------------------------------------------------
    # Write / edit / read
    # ------------------------------------------------------------------

    async def write_content(
        self,
        path: Path,
        content: str,
        header_format: str | None = None,
        author: str = "",
    ) -> Path:
        """Append *content* to *path* with an optional header line."""
        path.parent.mkdir(parents=True, exist_ok=True)

        header = ""
        if header_format:
            now = datetime.now(timezone.utc).strftime("%H:%M:%S")
            header = header_format.replace("{time}", now).replace("{author}", author)

        block = ""
        if header:
            block += header + "\n\n"
        block += content + "\n\n"

        async with aiofiles.open(path, mode="a", encoding="utf-8") as f:
            await f.write(block)

        logger.debug("Wrote %d chars to %s", len(block), path)
        return path

    async def edit_content(
        self,
        path: Path,
        old_text: str,
        new_text: str,
    ) -> bool:
        """Surgical find-and-replace in *path*.

        The *old_text* must appear exactly once; raises ``ValueError``
        otherwise.
        """
        async with aiofiles.open(path, encoding="utf-8") as f:
            body = await f.read()

        count = body.count(old_text)
        if count == 0:
            raise ValueError(f"old_text not found in {path}")
        if count > 1:
            raise ValueError(f"old_text matches {count} times in {path}")

        body = body.replace(old_text, new_text, 1)

        async with aiofiles.open(path, mode="w", encoding="utf-8") as f:
            await f.write(body)

        return True

    async def read_file(
        self,
        path: Path,
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> str:
        """Read *path*, optionally returning only a line range."""
        async with aiofiles.open(path, encoding="utf-8") as f:
            lines = await f.readlines()

        if line_start is not None or line_end is not None:
            start = (line_start or 1) - 1  # 1-based → 0-based
            end = line_end or len(lines)
            lines = lines[start:end]

        return "".join(lines)

    async def list_files(
        self,
        namespace: str,
        recursive: bool = True,
        pattern: str = "*",
    ) -> list[str]:
        """List files under *namespace* inside the data dir."""
        base = Path(self._config.memory.data_dir) / namespace
        if not base.exists():
            return []
        glob_fn = base.rglob if recursive else base.glob
        return sorted(
            str(p.relative_to(base))
            for p in glob_fn(pattern)
            if p.is_file()
        )
