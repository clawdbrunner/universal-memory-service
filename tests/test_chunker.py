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
        text = "a" * 400
        assert estimate_tokens(text) == 100

    def test_rough_ratio(self):
        text = "The quick brown fox jumps over the lazy dog."  # 44 chars
        assert estimate_tokens(text) == 44 // 4

    def test_single_char(self):
        assert estimate_tokens("x") == 1

    def test_whitespace_counted(self):
        text = "a b c d"  # 7 chars → 1 token
        assert estimate_tokens(text) == 7 // 4

    def test_large_text(self):
        text = "word " * 10_000  # 50_000 chars
        assert estimate_tokens(text) == 50_000 // 4


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

    def test_single_line(self):
        chunks = chunk_markdown("Just one line.", "f.md")
        assert len(chunks) == 1
        assert chunks[0].content == "Just one line."

    def test_preserves_full_content(self):
        text = "Line one.\nLine two.\nLine three."
        chunks = chunk_markdown(text, "f.md")
        full = "\n".join(c.content for c in chunks)
        assert "Line one." in full
        assert "Line three." in full


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
        long_para = "Word " * 500
        text = f"## Big Section\n### Sub A\n{long_para}\n### Sub B\n{long_para}"
        chunks = chunk_markdown(text, "f.md")
        assert len(chunks) >= 2
        headers = [c.header_path for c in chunks]
        assert any("Big Section" in h and "Sub A" in h for h in headers) or any(
            "Big Section" in h for h in headers
        )

    def test_content_before_first_header(self):
        text = "Preamble text.\n\n## First Section\nSection content."
        chunks = chunk_markdown(text, "f.md")
        assert len(chunks) == 2
        assert "Preamble" in chunks[0].content

    def test_h1_not_split(self):
        text = "# Title\nContent under title."
        chunks = chunk_markdown(text, "f.md")
        assert len(chunks) == 1
        assert "# Title" in chunks[0].content

    def test_many_h2_sections(self):
        sections = [f"## Section {i}\nContent {i}." for i in range(10)]
        text = "\n\n".join(sections)
        chunks = chunk_markdown(text, "f.md")
        assert len(chunks) == 10

    def test_empty_section_between_headers(self):
        text = "## A\nContent A.\n\n## B\n\n## C\nContent C."
        chunks = chunk_markdown(text, "f.md")
        assert any("Content A" in c.content for c in chunks)
        assert any("Content C" in c.content for c in chunks)


# ---------------------------------------------------------------------------
# chunk_markdown — nested headers
# ---------------------------------------------------------------------------


class TestNestedHeaders:
    def test_h2_with_h3_children(self):
        long = "Detail " * 200
        text = (
            f"## Parent Section\n"
            f"### Child A\n{long}\n"
            f"### Child B\n{long}\n"
        )
        chunks = chunk_markdown(text, "f.md")
        assert len(chunks) >= 2
        # At least one chunk should reference the parent
        all_headers = " ".join(c.header_path for c in chunks)
        assert "Parent Section" in all_headers

    def test_deeply_nested_h3_inherits_h2(self):
        long = "Text " * 300
        text = (
            f"## Top\n"
            f"### Mid\n{long}\n"
        )
        chunks = chunk_markdown(text, "f.md")
        # The h3 chunk should have composite header path
        for c in chunks:
            if "Mid" in c.header_path:
                assert "Top" in c.header_path

    def test_multiple_h2_each_with_h3(self):
        long = "Filler " * 200
        text = (
            f"## Alpha\n### A1\n{long}\n### A2\n{long}\n"
            f"## Beta\n### B1\n{long}\n### B2\n{long}\n"
        )
        chunks = chunk_markdown(text, "f.md")
        headers = [c.header_path for c in chunks]
        assert any("Alpha" in h for h in headers)
        assert any("Beta" in h for h in headers)


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

    def test_line_numbers_monotonic(self):
        sections = [f"## S{i}\nContent {i}." for i in range(5)]
        text = "\n\n".join(sections)
        chunks = chunk_markdown(text, "f.md")
        for i in range(1, len(chunks)):
            assert chunks[i].line_start >= chunks[i - 1].line_start

    def test_line_end_gte_line_start(self):
        text = "## A\nLine 1\nLine 2\n\n## B\nLine 1\nLine 2\nLine 3"
        chunks = chunk_markdown(text, "f.md")
        for c in chunks:
            assert c.line_end >= c.line_start


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

    def test_multiline_code_block_not_split_on_headers(self):
        text = "## Example\n```\n## This is not a header\nJust code\n```"
        chunks = chunk_markdown(text, "f.md")
        full = " ".join(c.content for c in chunks)
        assert "## This is not a header" in full


# ---------------------------------------------------------------------------
# chunk_markdown — large content splitting
# ---------------------------------------------------------------------------


class TestChunkMarkdownLargeContent:
    def test_large_paragraph_split(self):
        big = "This is a sentence. " * 200
        chunks = chunk_markdown(big, "f.md", chunk_size=100)
        assert len(chunks) > 1

    def test_multiple_paragraphs_combined(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = chunk_markdown(text, "f.md", chunk_size=400)
        assert len(chunks) == 1
        assert "Para one" in chunks[0].content
        assert "Para three" in chunks[0].content

    def test_very_long_single_line(self):
        text = "x" * 5000
        chunks = chunk_markdown(text, "f.md", chunk_size=100)
        # Should still produce chunks (sentence splitter may not split without punctuation,
        # but the chunker should handle it gracefully)
        assert len(chunks) >= 1

    def test_chunk_size_respected_approximately(self):
        big = "This is a test sentence. " * 500
        chunks = chunk_markdown(big, "f.md", chunk_size=100)
        char_budget = 100 * 4
        # Most chunks should be within ~2x the budget (overlap can add some)
        for c in chunks[:-1]:  # last chunk can be smaller
            assert len(c.content) < char_budget * 3


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
        if len(chunks_with) > 1 and len(chunks_without) > 1:
            assert len(chunks_with[1].content) >= len(chunks_without[1].content)

    def test_no_overlap_when_single_chunk(self):
        text = "Just a short note."
        chunks = chunk_markdown(text, "f.md", overlap=80)
        assert len(chunks) == 1

    def test_zero_overlap(self):
        sections = [f"## S{i}\n" + "Word " * 100 for i in range(3)]
        text = "\n\n".join(sections)
        chunks = chunk_markdown(text, "f.md", chunk_size=150, overlap=0)
        assert len(chunks) >= 3

    def test_overlap_does_not_duplicate_first_chunk(self):
        sections = [f"## S{i}\n" + "Filler " * 100 for i in range(3)]
        text = "\n\n".join(sections)
        chunks = chunk_markdown(text, "f.md", chunk_size=150, overlap=40)
        # First chunk should not have any overlap prepended
        assert chunks[0].content.startswith("## S0")


# ---------------------------------------------------------------------------
# chunk_markdown — token counting
# ---------------------------------------------------------------------------


class TestTokenCounting:
    def test_token_count_matches_estimate(self):
        text = "Hello world, this is a test of token counting."
        chunks = chunk_markdown(text, "f.md")
        expected = estimate_tokens(text)
        assert chunks[0].token_count == expected

    def test_token_count_positive_for_all_chunks(self):
        sections = [f"## S{i}\nContent {i}." for i in range(5)]
        text = "\n\n".join(sections)
        chunks = chunk_markdown(text, "f.md")
        for c in chunks:
            assert c.token_count > 0

    def test_larger_content_more_tokens(self):
        small = chunk_markdown("Short.", "f.md")
        big = chunk_markdown("A much longer piece of text " * 20, "f.md")
        assert big[0].token_count > small[0].token_count


# ---------------------------------------------------------------------------
# chunk_markdown — edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_only_headers_no_content(self):
        text = "## A\n\n## B\n\n## C"
        chunks = chunk_markdown(text, "f.md")
        assert len(chunks) >= 1

    def test_unicode_content(self):
        text = "## Section\nThis has unicode: \u00e9\u00e8\u00ea \u00fc\u00f6\u00e4 \u4e16\u754c \ud83d\ude80"
        chunks = chunk_markdown(text, "f.md")
        assert len(chunks) >= 1
        assert "\u00e9" in chunks[0].content

    def test_special_markdown_chars(self):
        text = "## Section\n**bold** _italic_ ~~strike~~ [link](url) ![img](src)"
        chunks = chunk_markdown(text, "f.md")
        assert "**bold**" in chunks[0].content

    def test_many_blank_lines(self):
        text = "Content A\n\n\n\n\n\nContent B"
        chunks = chunk_markdown(text, "f.md")
        assert len(chunks) >= 1
        full = " ".join(c.content for c in chunks)
        assert "Content A" in full
        assert "Content B" in full

    def test_tabs_and_mixed_whitespace(self):
        text = "## Section\n\tTabbed content\n  Spaced content"
        chunks = chunk_markdown(text, "f.md")
        assert len(chunks) >= 1

    def test_header_with_special_chars(self):
        text = "## Section: The \"Best\" (v2.0) [DRAFT]\nContent."
        chunks = chunk_markdown(text, "f.md")
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# chunk_markdown — realistic markdown
# ---------------------------------------------------------------------------


class TestChunkMarkdownRealistic:
    def test_daily_log_format(self):
        text = (
            "# Daily Log \u2014 2025-01-15\n\n"
            "## [09:15] alice\n"
            "Deployed v2.3 to staging.\n\n"
            "## [14:30] alice\n"
            "Fixed auth bug in middleware.\n\n"
            "## [17:00] alice\n"
            "Wrapped up for the day. Pushed all changes.\n"
        )
        chunks = chunk_markdown(text, "agents/alice/logs/2025-01-15.md")
        assert len(chunks) >= 3
        assert all(
            c.file_path == "agents/alice/logs/2025-01-15.md" for c in chunks
        )

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

    def test_mixed_content_types(self):
        text = (
            "## Notes\n"
            "Some text.\n\n"
            "```python\nprint('hello')\n```\n\n"
            "- List item 1\n"
            "- List item 2\n\n"
            "> Blockquote here\n\n"
            "| Col A | Col B |\n"
            "|-------|-------|\n"
            "| 1     | 2     |\n"
        )
        chunks = chunk_markdown(text, "f.md")
        full = " ".join(c.content for c in chunks)
        assert "print('hello')" in full
        assert "List item 1" in full
