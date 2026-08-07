"""Unit tests for ParagraphChunker."""
import pytest

from ParagraphChunker import ParagraphChunker


class TestParagraphChunker:
    def test_empty_text_returns_empty_list(self):
        chunker = ParagraphChunker(max_chunk_size=50)
        assert chunker.split_text("") == []
        assert chunker.split_text("   \n\n  ") == []

    def test_single_paragraph_under_limit(self):
        chunker = ParagraphChunker(max_chunk_size=100)
        text = "A short paragraph."
        assert chunker.split_text(text) == ["A short paragraph."]

    def test_single_oversized_paragraph_kept_intact(self):
        chunker = ParagraphChunker(max_chunk_size=10)
        text = "This paragraph is deliberately longer than the max chunk size."
        chunks = chunker.split_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_packs_paragraphs_until_max_size(self):
        chunker = ParagraphChunker(max_chunk_size=20)
        text = "aaaa\n\nbbbb\n\ncccc"
        chunks = chunker.split_text(text)
        # "aaaa" + "bbbb" = 8 chars, under 20; adding "cccc" (4) still under 20
        # Wait: current_size only counts paragraph lengths without separators
        # aaaa=4, bbbb=4 -> 8, +cccc=4 -> 12, all in one chunk
        assert chunks == ["aaaa\n\nbbbb\n\ncccc"]

    def test_splits_when_next_paragraph_would_exceed(self):
        chunker = ParagraphChunker(max_chunk_size=10)
        text = "abcdefghij\n\nklmnopqrst"  # 10 + 10
        chunks = chunker.split_text(text)
        assert chunks == ["abcdefghij", "klmnopqrst"]

    def test_whitespace_only_paragraphs_dropped(self):
        chunker = ParagraphChunker(max_chunk_size=100)
        text = "one\n\n   \n\ntwo"
        assert chunker.split_text(text) == ["one\n\ntwo"]

    def test_custom_separator(self):
        chunker = ParagraphChunker(max_chunk_size=5, paragraph_separator=r"\|")
        text = "aa|bbb|cccccc"
        chunks = chunker.split_text(text)
        # aa(2)+bbb(3)=5 fits exactly; oversized third paragraph stands alone
        assert chunks == ["aa\n\nbbb", "cccccc"]

    def test_strips_paragraph_whitespace(self):
        chunker = ParagraphChunker(max_chunk_size=100)
        text = "  hello  \n\n  world  "
        assert chunker.split_text(text) == ["hello\n\nworld"]
