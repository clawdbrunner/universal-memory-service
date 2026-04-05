"""Unit tests for the markdown-aware chunker."""

from __future__ import annotations

import pytest

from universal_memory.chunker import chunk_markdown, estimate_tokens


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 1  # max(1, 0)

    def test_short_string(self):
        assert estimate_tokens("hi") == 1

    def test_typical_string(self):
        # 400 chars → ~100 tokens
        text = "a" * 400
        assert estimate_tokens(text) == 100

    def test_rough_ratio(self):
        text = "The quick brown fox jumps over the lazy dog."  # 44 chars
        tokens = estimate_tokens(text)
        assert tokens == 44 // 4


# ---------------------------------------------------------------------------
# chunk_markdown — basics
# ---------------------------------------------------------------------------


class TestChunkMarkdownBasics:
    def test_empty_text_returns_empty(self):
        assert chunk_markdown("", "test.md") == []

    def test_whitespace_only_returns_empty(self):
        assert chunk_markdown("   \n\n  ", "test.md") == []

    def test_single_paragraph(self):
        text = "Hello world, this is a simple paragraph."
        chunks = chunk_markdown(text, "notes.md")
        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].file_path == "notes.md"

    def test_document_id_assigned(self):
        chunks = chunk_markdown("Some text", "f.md", document_id="doc123")
        assert chunks[0].document_id == "doc123"

    def test_auto_document_id(self):
        chunks = chunk_markdown("Some text", "f.md")
        assert len(chunks[0].document_id) == 32  # uuid hex

    def test_chunk_ids_unique(self):
        text = "## A\nParagraph one.\n\n## B\nParagraph two."
        chunks = chunk_markdown(text, "f.md")
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_token_count_populated(self):
        chunks = chunk_markdown("Hello world", "f.md")
        assert chunks[0].token_count > 0


# ---------------------------------------------------------------------------
# chunk_markdown — header splitting
# ---------------------------------------------------------------------------


class TestChunkMarkdownHeaders:
    def test_splits_on_h2(self):
        text = "## Section A\nContent A.\n\n## Section B\nContent B."
        chunks = chunk_markdown(text, "f.md")
        assert len(chunks) == 2
        assert "Content A" in chunks[0].content
        assert "Content B" in chunks[1].content

    def test_header_path_tracked(self):
        text = "## My Section\nSome content here."
        chunks = chunk_markdown(text, "f.md")
        assert chunks[0].header_path == "My Section"

    def test_h3_subsection_split(self):
        # Make the h2 section large enough to trigger h3 splitting
        long_para = "Word " * 500  # ~2500 chars > default 1600 char budget
        text = f"## Big Section\n### Sub A\n{long_para}\n### Sub B\n{long_para}"
        chunks = chunk_markdown(text, "f.md")
        # Should have at least 2 chunks from the subsections
        assert len(chunks) >= 2
        # At least one chunk should have composite header path
        headers = [c.header_path for c in chunks]
        assert any("Big Section" in h and "Sub A" in h for h in headers) or \
               any("Big Section" in h for h in headers)

    def test_content_before_first_header(self):
        text = "Preamble text.\n\n## First Section\nSection content."
        chunks = chunk_markdown(text, "f.md")
        assert len(chunks) == 2
        assert "Preamble" in chunks[0].content

    def test_h1_not_split(self):
        """H1 headers should NOT cause splits (only ## and ### do)."""
        text = "# Title\nContent under title."
        chunks = chunk_markdown(text, "f.md")
        assert len(chunks) == 1
        assert "# Title" in chunks[0].content


# ---------------------------------------------------------------------------
# chunk_markdown — line numbers
# ---------------------------------------------------------------------------


class TestChunkMarkdownLineNumbers:
    def test_line_numbers_single_chunk(self):
        text = "Line one\nLine two\nLine three"
        chunks = chunk_markdown(text, "f.md")
        assert chunks[0].line_start == 1
        assert chunks[0].line_end == 3

    def test_line_numbers_multiple_sections(self):
        text = "## A\nLine 2\n\n## B\nLine 5"
        chunks = chunk_markdown(text, "f.md")
        assert chunks[0].line_start == 1
        assert chunks[1].line_start >= 4


# ---------------------------------------------------------------------------
# chunk_markdown — code blocks
# ---------------------------------------------------------------------------


class TestChunkMarkdownCodeBlocks:
    def test_code_block_preserved(self):
        text = "## Code\n```python\ndef hello():\n    print('hi')\n```"
        chunks = chunk_markdown(text, "f.md")
        assert len(chunks) >= 1
        full = " ".join(c.content for c in chunks)
        assert "def hello():" in full
        assert "print('hi')" in full

    def test_inline_code_preserved(self):
        text = "Use `memory_search` to find things."
        chunks = chunk_markdown(text, "f.md")
        assert "`memory_search`" in chunks[0].content


# ---------------------------------------------------------------------------
# chunk_markdown — large content splitting
# ---------------------------------------------------------------------------


class TestChunkMarkdownLargeContent:
    def test_large_paragraph_split(self):
        # A single paragraph larger than the char budget
        big = "This is a sentence. " * 200  # ~4000 chars
        chunks = chunk_markdown(big, "f.md", chunk_size=100)  # 400 char budget
        assert len(chunks) > 1

    def test_multiple_paragraphs_combined(self):
        """Small paragraphs should be combined into a single chunk."""
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = chunk_markdown(text, "f.md", chunk_size=400)
        assert len(chunks) == 1
        assert "Para one" in chunks[0].content
        assert "Para three" in chunks[0].content


# ---------------------------------------------------------------------------
# chunk_markdown — overlap
# ---------------------------------------------------------------------------


class TestChunkMarkdownOverlap:
    def test_overlap_applied(self):
        sections = []
        for i in range(5):
            sections.append(f"## Section {i}\n" + f"Content for section {i}. " * 50)
        text = "\n\n".join(sections)
        chunks_with = chunk_markdown(text, "f.md", chunk_size=200, overlap=40)
        chunks_without = chunk_markdown(text, "f.md", chunk_size=200, overlap=0)
        # With overlap, later chunks should be longer (have prepended tail)
        if len(chunks_with) > 1 and len(chunks_without) > 1:
            assert len(chunks_with[1].content) >= len(chunks_without[1].content)

    def test_no_overlap_when_single_chunk(self):
        text = "Just a short note."
        chunks = chunk_markdown(text, "f.md", overlap=80)
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# chunk_markdown — realistic markdown
# ---------------------------------------------------------------------------


class TestChunkMarkdownRealistic:
    def test_daily_log_format(self):
        text = (
            "# Daily Log — 2025-01-15\n\n"
            "## [09:15] alice\n"
            "Deployed v2.3 to staging.\n\n"
            "## [14:30] alice\n"
            "Fixed auth bug in middleware.\n\n"
            "## [17:00] alice\n"
            "Wrapped up for the day. Pushed all changes.\n"
        )
        chunks = chunk_markdown(text, "agents/alice/logs/2025-01-15.md")
        assert len(chunks) >= 3
        assert all(c.file_path == "agents/alice/logs/2025-01-15.md" for c in chunks)

    def test_memory_md_format(self):
        text = (
            "## User Preferences\n"
            "- Prefers dark mode\n"
            "- Uses vim keybindings\n\n"
            "## Project Context\n"
            "Working on memory service migration.\n"
        )
        chunks = chunk_markdown(text, "shared/MEMORY.md")
        assert len(chunks) == 2
        assert any("dark mode" in c.content for c in chunks)
        assert any("memory service" in c.content for c in chunks)
