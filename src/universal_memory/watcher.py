"""File watcher for automatic re-indexing on changes."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import watchfiles

from .config import get_config
from .indexer import _should_ignore

logger = logging.getLogger(__name__)

ChangeCallback = Callable[[str, str], Coroutine[Any, Any, None]]


class FileWatcher:
    """Watch a directory for file changes and invoke a callback.

    Uses ``watchfiles.awatch`` (async, rust-backed) for efficient
    filesystem monitoring.  Rapid changes within *debounce_ms* are
    coalesced so the callback fires at most once per window.
    """

    def __init__(self, on_change: ChangeCallback) -> None:
        cfg = get_config()
        self._data_dir = Path(cfg.memory.data_dir)
        self._extensions: set[str] = set(cfg.memory.extensions)
        self._ignore_patterns: list[str] = cfg.memory.ignore_patterns
        self._debounce_ms: int = cfg.sync.debounce_ms
        self._on_change = on_change
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Begin watching in a background task."""
        if self._task is not None:
            return
        self._stop_event.clear()
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._task = asyncio.create_task(self._run(), name="file-watcher")
        logger.info("FileWatcher started on %s", self._data_dir)

    async def stop(self) -> None:
        """Signal the watcher to stop and wait for the task to finish."""
        if self._task is None:
            return
        self._stop_event.set()
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        logger.info("FileWatcher stopped")

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _match(self, path: str) -> bool:
        if _should_ignore(path, self._ignore_patterns):
            return False
        return Path(path).suffix in self._extensions

    @staticmethod
    def _change_type(change: watchfiles.Change) -> str:
        return {
            watchfiles.Change.added: "created",
            watchfiles.Change.modified: "modified",
            watchfiles.Change.deleted: "deleted",
        }.get(change, "modified")

    async def _run(self) -> None:
        try:
            async for changes in watchfiles.awatch(
                self._data_dir,
                debounce=self._debounce_ms,
                stop_event=self._stop_event,
                recursive=True,
            ):
                for change, path_str in changes:
                    if not self._match(path_str):
                        continue
                    ctype = self._change_type(change)
                    logger.debug("change %s %s", ctype, path_str)
                    try:
                        await self._on_change(path_str, ctype)
                    except Exception:
                        logger.exception("Callback error for %s", path_str)
        except asyncio.CancelledError:
            pass
