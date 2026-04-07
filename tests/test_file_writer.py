"""Tests for FileWriter path resolution, read, write, and edit."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from universal_memory.config import FullConfig, MemoryConfig, WriteConfig
from universal_memory.writers.file_writer import FileWriter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(data_dir: str = "/tmp/test-data") -> FullConfig:
    cfg = MagicMock(spec=FullConfig)
    cfg.memory = MemoryConfig(data_dir=data_dir)
    cfg.write = WriteConfig()
    cfg.agents = {}
    return cfg


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


class TestResolvePath:
    def test_daily_path(self):
        cfg = _make_config()
        p = FileWriter.resolve_path(author="alice", target="daily", config=cfg)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert str(p).endswith(f"agents/alice/logs/{today}.md")

    def test_long_term_path(self):
        cfg = _make_config()
        p = FileWriter.resolve_path(author="bob", target="long-term", config=cfg)
        assert str(p).endswith("agents/bob/MEMORY.md")

    def test_shared_path(self):
        cfg = _make_config()
        p = FileWriter.resolve_path(author="carol", target="shared", config=cfg)
        assert "shared" in str(p)
        assert str(p).endswith(".md")

    def test_custom_file_path(self):
        cfg = _make_config()
        p = FileWriter.resolve_path(
            author="alice",
            target="file",
            file_path="custom/notes.md",
            config=cfg,
        )
        assert str(p).endswith("custom/notes.md")

    def test_fallback_to_daily(self):
        cfg = _make_config()
        p = FileWriter.resolve_path(author="alice", target="unknown_target", config=cfg)
        assert "logs" in str(p)  # falls back to daily


# ---------------------------------------------------------------------------
# Write / edit / read
# ---------------------------------------------------------------------------


class TestWriteContent:
    @pytest.mark.asyncio
    async def test_write_creates_file(self, tmp_path):
        with patch("universal_memory.writers.file_writer.get_config") as mock_cfg:
            mock_cfg.return_value = _make_config(str(tmp_path))
            writer = FileWriter()

        target = tmp_path / "out.md"
        await writer.write_content(target, "Hello world")
        assert target.read_text().strip() == "Hello world"

    @pytest.mark.asyncio
    async def test_write_appends(self, tmp_path):
        with patch("universal_memory.writers.file_writer.get_config") as mock_cfg:
            mock_cfg.return_value = _make_config(str(tmp_path))
            writer = FileWriter()

        target = tmp_path / "append.md"
        await writer.write_content(target, "first")
        await writer.write_content(target, "second")
        content = target.read_text()
        assert "first" in content
        assert "second" in content

    @pytest.mark.asyncio
    async def test_write_with_header(self, tmp_path):
        with patch("universal_memory.writers.file_writer.get_config") as mock_cfg:
            mock_cfg.return_value = _make_config(str(tmp_path))
            writer = FileWriter()

        target = tmp_path / "headed.md"
        await writer.write_content(
            target, "content", header_format="## [{time}] {author}", author="alice"
        )
        text = target.read_text()
        assert "alice" in text
        assert "content" in text


class TestEditContent:
    @pytest.mark.asyncio
    async def test_edit_replaces_once(self, tmp_path):
        with patch("universal_memory.writers.file_writer.get_config") as mock_cfg:
            mock_cfg.return_value = _make_config(str(tmp_path))
            writer = FileWriter()

        target = tmp_path / "edit.md"
        target.write_text("hello world")
        await writer.edit_content(target, "hello", "goodbye")
        assert target.read_text() == "goodbye world"

    @pytest.mark.asyncio
    async def test_edit_not_found_raises(self, tmp_path):
        with patch("universal_memory.writers.file_writer.get_config") as mock_cfg:
            mock_cfg.return_value = _make_config(str(tmp_path))
            writer = FileWriter()

        target = tmp_path / "edit2.md"
        target.write_text("abc")
        with pytest.raises(ValueError, match="not found"):
            await writer.edit_content(target, "xyz", "new")

    @pytest.mark.asyncio
    async def test_edit_multiple_matches_raises(self, tmp_path):
        with patch("universal_memory.writers.file_writer.get_config") as mock_cfg:
            mock_cfg.return_value = _make_config(str(tmp_path))
            writer = FileWriter()

        target = tmp_path / "edit3.md"
        target.write_text("aa aa")
        with pytest.raises(ValueError, match="2 times"):
            await writer.edit_content(target, "aa", "bb")


class TestReadFile:
    @pytest.mark.asyncio
    async def test_read_full(self, tmp_path):
        with patch("universal_memory.writers.file_writer.get_config") as mock_cfg:
            mock_cfg.return_value = _make_config(str(tmp_path))
            writer = FileWriter()

        target = tmp_path / "read.md"
        target.write_text("line1\nline2\nline3\n")
        content = await writer.read_file(target)
        assert "line1" in content
        assert "line3" in content

    @pytest.mark.asyncio
    async def test_read_line_range(self, tmp_path):
        with patch("universal_memory.writers.file_writer.get_config") as mock_cfg:
            mock_cfg.return_value = _make_config(str(tmp_path))
            writer = FileWriter()

        target = tmp_path / "range.md"
        target.write_text("a\nb\nc\nd\n")
        content = await writer.read_file(target, line_start=2, line_end=3)
        assert content.strip() == "b\nc"


# ---------------------------------------------------------------------------
# Path traversal in file_writer resolve_path
# ---------------------------------------------------------------------------


class TestPathTraversalRejected:
    def test_absolute_custom_path_returned_as_is(self):
        """FileWriter.resolve_path returns absolute paths unchanged for target=file."""
        cfg = _make_config()
        p = FileWriter.resolve_path(
            author="alice", target="file", file_path="/etc/passwd", config=cfg
        )
        # The route-level _validate_path should catch this — but resolve_path
        # itself doesn't reject it, which is why the API layer is needed.
        assert p == Path("/etc/passwd")
