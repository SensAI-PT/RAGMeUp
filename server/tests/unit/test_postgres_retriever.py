"""Unit tests for PostgresHybridRetriever helpers (no live DB)."""
from unittest.mock import MagicMock, patch
import json

import numpy as np
import pytest

from PostgresHybridRetriever import PostgresHybridRetriever


@pytest.fixture
def retriever():
    return PostgresHybridRetriever(connection_pool=MagicMock())


class TestEscapeQuery:
    def test_strips_punctuation(self, retriever, monkeypatch):
        # Ensure punkt is available; download quietly if needed
        import nltk

        try:
            nltk.word_tokenize("hello")
        except LookupError:
            nltk.download("punkt", quiet=True)
            try:
                nltk.download("punkt_tab", quiet=True)
            except Exception:
                pass

        sanitized = retriever.escape_query("What's the capital of France?!")
        assert "?" not in sanitized
        assert "!" not in sanitized
        assert "France" in sanitized

    def test_keeps_alnum_tokens(self, retriever):
        import nltk

        try:
            nltk.word_tokenize("abc")
        except LookupError:
            nltk.download("punkt", quiet=True)
            try:
                nltk.download("punkt_tab", quiet=True)
            except Exception:
                pass

        assert "hello" in retriever.escape_query("hello world").lower()


class TestGetRelevantDocumentsDispatch:
    def test_dispatches_to_rrf(self, retriever, monkeypatch):
        monkeypatch.setenv("use_rrf", "True")
        retriever._get_relevant_documents_rrf = MagicMock(return_value=["rrf"])
        retriever._get_relevant_documents_minmax = MagicMock(return_value=["mm"])
        out = retriever.get_relevant_documents("q", np.array([0.1]), [])
        assert out == ["rrf"]
        retriever._get_relevant_documents_rrf.assert_called_once()
        retriever._get_relevant_documents_minmax.assert_not_called()

    def test_dispatches_to_minmax(self, retriever, monkeypatch):
        monkeypatch.setenv("use_rrf", "False")
        retriever._get_relevant_documents_rrf = MagicMock(return_value=["rrf"])
        retriever._get_relevant_documents_minmax = MagicMock(return_value=["mm"])
        out = retriever.get_relevant_documents("q", np.array([0.1]), [])
        assert out == ["mm"]


class TestRrfFusion:
    def test_rrf_merges_and_ranks(self, retriever, monkeypatch):
        monkeypatch.setenv("use_rrf", "True")
        monkeypatch.setenv("use_re2", "False")
        monkeypatch.setenv("vector_store_k", "2")
        monkeypatch.setenv("rrf_k", "60")

        bm25_rows = [
            ("id-a", "content A", {"source": "a.txt", "dataset": "d"}),
            ("id-b", "content B", {"source": "b.txt", "dataset": "d"}),
        ]
        vector_rows = [
            ("id-b", "content B", {"source": "b.txt", "dataset": "d"}, 0.05),
            ("id-c", "content C", {"source": "c.txt", "dataset": "d"}, 0.1),
        ]

        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchall.side_effect = [bm25_rows, vector_rows]
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        pool = MagicMock()
        pool.getconn.return_value = conn
        retriever.connection_pool = pool
        retriever.escape_query = MagicMock(return_value="clean query")

        results = retriever._get_relevant_documents_rrf(
            "query", np.array([0.1, 0.2]), []
        )

        assert results is not None
        assert len(results) == 2
        # id-b appears in both lists → highest RRF score → first
        assert results[0]["content"] == "content B"
        assert "bm25" in results[0]["metadata"]["sources"]
        assert "vector" in results[0]["metadata"]["sources"]
        pool.putconn.assert_called_once_with(conn)

    def test_rrf_strips_re2_suffix(self, retriever, monkeypatch):
        monkeypatch.setenv("use_re2", "True")
        monkeypatch.setenv("re2_prompt", "Read again:")
        monkeypatch.setenv("vector_store_k", "1")
        monkeypatch.setenv("rrf_k", "60")

        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchall.side_effect = [[], []]
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        pool = MagicMock()
        pool.getconn.return_value = conn
        retriever.connection_pool = pool
        retriever.escape_query = MagicMock(return_value="q")

        query = "original\nRead again:\noriginal"
        retriever._get_relevant_documents_rrf(query, np.array([0.1]), [])
        # escape_query should receive stripped query
        retriever.escape_query.assert_called_with("original")


class TestHasDataAndClose:
    def test_has_data_true(self, retriever):
        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchone.return_value = (3,)
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        pool = MagicMock()
        pool.getconn.return_value = conn
        retriever.connection_pool = pool
        assert retriever.has_data() is True

    def test_has_data_false_on_error(self, retriever):
        pool = MagicMock()
        pool.getconn.side_effect = RuntimeError("boom")
        retriever.connection_pool = pool
        assert retriever.has_data() is False

    def test_close_closes_pool(self, retriever):
        pool = MagicMock()
        retriever.connection_pool = pool
        retriever.close()
        pool.closeall.assert_called_once()


class TestGetDocumentNames:
    def test_strips_data_directory_prefix(self, retriever, monkeypatch):
        monkeypatch.setenv("data_directory", "/data")
        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchall.return_value = [("/data/geo.txt", "default")]
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        pool = MagicMock()
        pool.getconn.return_value = conn
        retriever.connection_pool = pool

        names = retriever.get_all_document_names()
        assert names == [{"filename": "geo.txt", "dataset": "default"}]
