"""Markdown-aware chunking for Universal Memory Service."""

from __future__ import annotations

import re
import uuid

from .models import Chunk


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def chunk_markdown(
    text: str,
    file_path: str,
    document_id: str = "",
    chunk_size: int = 400,
    overlap: int = 80,
) -> list[Chunk]:
    """Split markdown text into overlapping chunks.

    Strategy:
      1. Split on ## headers
      2. If a section is too large, split on ### headers
      3. If still too large, split on paragraphs (blank lines)
      4. If still too large, split on sentences

    Tracks line numbers and header hierarchy for each chunk.
    Overlap: the tail of the previous chunk is prepended to the next.
    """
    if not text.strip():
        return []

    doc_id = document_id or uuid.uuid4().hex
    lines = text.split("\n")
    sections = _split_by_headers(lines, level=2)

    chunks: list[Chunk] = []
    char_budget = chunk_size * 4  # convert token budget to char budget
    overlap_chars = overlap * 4

    for header_path, section_lines, section_start in sections:
        section_text = "\n".join(section_lines)
        if len(section_text) <= char_budget:
            chunks.append(
                _make_chunk(
                    doc_id,
                    file_path,
                    section_text,
                    section_start,
                    section_start + len(section_lines) - 1,
                    header_path,
                )
            )
        else:
            # Try splitting on ### within this section
            subsections = _split_by_headers(section_lines, level=3, base_line=section_start)
            for sub_header, sub_lines, sub_start in subsections:
                full_header = f"{header_path} > {sub_header}" if header_path and sub_header else (header_path or sub_header)
                sub_text = "\n".join(sub_lines)
                if len(sub_text) <= char_budget:
                    chunks.append(
                        _make_chunk(
                            doc_id, file_path, sub_text,
                            sub_start, sub_start + len(sub_lines) - 1,
                            full_header,
                        )
                    )
                else:
                    # Split on paragraphs, then sentences
                    para_chunks = _split_large_block(
                        sub_lines, sub_start, char_budget, full_header,
                        doc_id, file_path,
                    )
                    chunks.extend(para_chunks)

    # Apply overlap between consecutive chunks
    if overlap_chars > 0 and len(chunks) > 1:
        chunks = _apply_overlap(chunks, overlap_chars)

    return chunks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_by_headers(
    lines: list[str], level: int, base_line: int = 0
) -> list[tuple[str, list[str], int]]:
    """Split lines by markdown headers of a given level.

    Returns list of (header_text, lines, start_line_number).
    Line numbers are 1-indexed.
    """
    prefix = "#" * level + " "
    sections: list[tuple[str, list[str], int]] = []
    current_header = ""
    current_lines: list[str] = []
    current_start = base_line + 1  # 1-indexed

    for i, line in enumerate(lines):
        if line.startswith(prefix) and (level >= len(line) or not line[level].startswith("#")):
            # Found a header at this level
            if current_lines or sections:
                sections.append((current_header, current_lines, current_start))
            current_header = line[level + 1:].strip() if len(line) > level else ""
            current_lines = [line]
            current_start = base_line + i + 1
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_header, current_lines, current_start))

    # If no splits happened, return the whole thing as one section
    if not sections:
        sections = [("", lines, base_line + 1)]

    return sections


def _split_large_block(
    lines: list[str],
    start_line: int,
    char_budget: int,
    header_path: str,
    doc_id: str,
    file_path: str,
) -> list[Chunk]:
    """Split a large block by paragraphs, then by sentences if needed."""
    paragraphs = _split_paragraphs(lines, start_line)
    chunks: list[Chunk] = []
    current_text = ""
    current_start = start_line
    current_end = start_line

    for para_text, para_start, para_end in paragraphs:
        if not para_text.strip():
            continue
        if current_text and len(current_text) + len(para_text) + 2 > char_budget:
            # Flush current
            chunks.append(
                _make_chunk(doc_id, file_path, current_text, current_start, current_end, header_path)
            )
            current_text = ""
            current_start = para_start

        if len(para_text) > char_budget:
            # Flush anything buffered
            if current_text:
                chunks.append(
                    _make_chunk(doc_id, file_path, current_text, current_start, current_end, header_path)
                )
                current_text = ""
            # Split by sentences
            sentence_chunks = _split_sentences(
                para_text, para_start, para_end, char_budget, header_path, doc_id, file_path
            )
            chunks.extend(sentence_chunks)
            current_start = para_end + 1
        else:
            if current_text:
                current_text += "\n\n" + para_text
            else:
                current_text = para_text
            current_end = para_end

    if current_text.strip():
        chunks.append(
            _make_chunk(doc_id, file_path, current_text, current_start, current_end, header_path)
        )

    return chunks


def _split_paragraphs(
    lines: list[str], base_line: int
) -> list[tuple[str, int, int]]:
    """Split lines into paragraphs (separated by blank lines).

    Returns list of (text, start_line, end_line).
    """
    paragraphs: list[tuple[str, int, int]] = []
    current: list[str] = []
    start = base_line

    for i, line in enumerate(lines):
        line_num = base_line + i
        if line.strip() == "":
            if current:
                paragraphs.append(("\n".join(current), start, line_num - 1))
                current = []
            start = line_num + 1
        else:
            if not current:
                start = line_num
            current.append(line)

    if current:
        paragraphs.append(("\n".join(current), start, base_line + len(lines) - 1))

    return paragraphs


def _split_sentences(
    text: str,
    start_line: int,
    end_line: int,
    char_budget: int,
    header_path: str,
    doc_id: str,
    file_path: str,
) -> list[Chunk]:
    """Split text by sentences when paragraphs are still too large."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[Chunk] = []
    current = ""
    # Approximate line distribution across the text
    total_chars = len(text)
    line_span = max(1, end_line - start_line + 1)

    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > char_budget:
            char_offset = text.find(current)
            frac_start = char_offset / total_chars if total_chars else 0
            frac_end = (char_offset + len(current)) / total_chars if total_chars else 1
            cs = start_line + int(frac_start * line_span)
            ce = start_line + int(frac_end * line_span)
            chunks.append(_make_chunk(doc_id, file_path, current, cs, ce, header_path))
            current = ""

        current = f"{current} {sentence}".strip() if current else sentence

    if current.strip():
        chunks.append(
            _make_chunk(doc_id, file_path, current, start_line, end_line, header_path)
        )

    return chunks


def _apply_overlap(chunks: list[Chunk], overlap_chars: int) -> list[Chunk]:
    """Prepend tail of previous chunk to each subsequent chunk."""
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_content = chunks[i - 1].content
        tail = prev_content[-overlap_chars:] if len(prev_content) > overlap_chars else prev_content
        # Find a clean break point (newline or space)
        break_idx = tail.find("\n")
        if break_idx == -1:
            break_idx = tail.find(" ")
        if break_idx > 0:
            tail = tail[break_idx + 1:]

        chunk = chunks[i]
        new_content = tail + "\n" + chunk.content if tail.strip() else chunk.content
        result.append(
            Chunk(
                id=chunk.id,
                document_id=chunk.document_id,
                file_path=chunk.file_path,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                content=new_content,
                header_path=chunk.header_path,
                token_count=estimate_tokens(new_content),
                embedding=chunk.embedding,
                embedding_hash=chunk.embedding_hash,
            )
        )
    return result


def _make_chunk(
    doc_id: str,
    file_path: str,
    content: str,
    line_start: int,
    line_end: int,
    header_path: str,
) -> Chunk:
    return Chunk(
        id=uuid.uuid4().hex,
        document_id=doc_id,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        content=content,
        header_path=header_path,
        token_count=estimate_tokens(content),
    )
