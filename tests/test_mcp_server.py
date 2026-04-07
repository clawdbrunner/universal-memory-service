"""Tests for MCP server tool listing and handler wiring."""

from __future__ import annotations

import pytest

from universal_memory.mcp_server import TOOLS, list_tools


# ---------------------------------------------------------------------------
# Tool listing
# ---------------------------------------------------------------------------


class TestMCPToolList:
    @pytest.mark.asyncio
    async def test_list_tools_returns_all(self):
        tools = await list_tools()
        names = {t.name for t in tools}
        assert "memory_search" in names
        assert "memory_write" in names
        assert "memory_read" in names
        assert "memory_list" in names
        assert "memory_edit" in names
        assert "memory_status" in names

    def test_tool_count(self):
        assert len(TOOLS) == 6

    def test_search_tool_schema(self):
        search = next(t for t in TOOLS if t.name == "memory_search")
        assert "query" in search.inputSchema["properties"]
        assert "query" in search.inputSchema["required"]

    def test_write_tool_schema(self):
        write = next(t for t in TOOLS if t.name == "memory_write")
        assert "content" in write.inputSchema["required"]

    def test_edit_tool_schema(self):
        edit = next(t for t in TOOLS if t.name == "memory_edit")
        required = edit.inputSchema["required"]
        assert "path" in required
        assert "old_text" in required
        assert "new_text" in required
