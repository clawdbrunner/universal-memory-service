"""Tests for FileWatcher filtering and lifecycle."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from universal_memory.watcher import FileWatcher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_config(data_dir: str = "/tmp/watch-test", ignore: list[str] | None = None):
    cfg = MagicMock()
    cfg.memory.data_dir = data_dir
    cfg.memory.extensions = [".md"]
    cfg.memory.ignore_patterns = ignore or ["node_modules", ".git", "*.pyc"]
    cfg.sync.debounce_ms = 100
    return cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWatcherFiltering:
    def test_match_valid_extension(self):
        with patch("universal_memory.watcher.get_config", return_value=_mock_config()):
            cb = AsyncMock()
            watcher = FileWatcher(on_change=cb)
            assert watcher._match("/tmp/watch-test/notes.md") is True

    def test_reject_wrong_extension(self):
        with patch("universal_memory.watcher.get_config", return_value=_mock_config()):
            cb = AsyncMock()
            watcher = FileWatcher(on_change=cb)
            assert watcher._match("/tmp/watch-test/image.png") is False

    def test_reject_ignored_directory(self):
        with patch("universal_memory.watcher.get_config", return_value=_mock_config()):
            cb = AsyncMock()
            watcher = FileWatcher(on_change=cb)
            assert watcher._match("/tmp/watch-test/node_modules/readme.md") is False

    def test_reject_ignored_glob_pattern(self):
        with patch("universal_memory.watcher.get_config", return_value=_mock_config()):
            cb = AsyncMock()
            watcher = FileWatcher(on_change=cb)
            assert watcher._match("/tmp/watch-test/script.pyc") is False

    def test_accept_nested_md(self):
        with patch("universal_memory.watcher.get_config", return_value=_mock_config()):
            cb = AsyncMock()
            watcher = FileWatcher(on_change=cb)
            assert watcher._match("/tmp/watch-test/agents/alice/logs/2025-01-01.md") is True


class TestWatcherLifecycle:
    def test_running_before_start(self):
        with patch("universal_memory.watcher.get_config", return_value=_mock_config()):
            watcher = FileWatcher(on_change=AsyncMock())
            assert watcher.running is False

    @pytest.mark.asyncio
    async def test_start_and_stop(self, tmp_path):
        cfg = _mock_config(data_dir=str(tmp_path))
        with patch("universal_memory.watcher.get_config", return_value=cfg):
            watcher = FileWatcher(on_change=AsyncMock())
            await watcher.start()
            assert watcher.running is True
            await watcher.stop()
            assert watcher.running is False
