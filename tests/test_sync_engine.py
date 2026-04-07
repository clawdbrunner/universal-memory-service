"""Tests for SyncEngine file copying and template resolution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from universal_memory.sync.sync_engine import SyncEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_config(enabled: bool = True, targets: list | None = None):
    cfg = MagicMock()
    cfg.sync.enabled = enabled
    cfg.sync.targets = targets or []
    cfg.memory.data_dir = "/tmp/sync-test"
    cfg.memory.extensions = [".md"]
    return cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSyncFile:
    @pytest.mark.asyncio
    async def test_sync_copies_file(self, tmp_path):
        src = tmp_path / "source.md"
        src.write_text("hello")
        dest = tmp_path / "dest" / "target.md"

        cfg = _mock_config(targets=[{"dest": str(dest)}])
        with patch("universal_memory.sync.sync_engine.get_config", return_value=cfg):
            engine = SyncEngine()
            results = await engine.sync_file(str(src))

        assert len(results) == 1
        assert dest.read_text() == "hello"

    @pytest.mark.asyncio
    async def test_sync_disabled(self, tmp_path):
        src = tmp_path / "source.md"
        src.write_text("hello")

        cfg = _mock_config(enabled=False)
        with patch("universal_memory.sync.sync_engine.get_config", return_value=cfg):
            engine = SyncEngine()
            results = await engine.sync_file(str(src))

        assert results == []

    @pytest.mark.asyncio
    async def test_sync_missing_source(self, tmp_path):
        cfg = _mock_config(targets=[{"dest": str(tmp_path / "out.md")}])
        with patch("universal_memory.sync.sync_engine.get_config", return_value=cfg):
            engine = SyncEngine()
            results = await engine.sync_file(str(tmp_path / "nonexistent.md"))

        assert results == []


class TestTemplateResolution:
    def test_today_template(self):
        from datetime import datetime, timezone

        result = SyncEngine.resolve_templates("/out/{today}.md")
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert today in result

    def test_yesterday_template(self):
        from datetime import datetime, timedelta, timezone

        result = SyncEngine.resolve_templates("/out/{yesterday}.md")
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        assert yesterday in result
