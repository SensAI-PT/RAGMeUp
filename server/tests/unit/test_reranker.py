"""Unit tests for Reranker wrapper."""
from unittest.mock import MagicMock, patch

import pytest


class TestReranker:
    def test_empty_documents_short_circuits(self, monkeypatch):
        monkeypatch.setenv("rerank_model", "ms-marco-MiniLM-L-12-v2")
        with patch("Reranker.Ranker") as Ranker:
            Ranker.return_value = MagicMock()
            from Reranker import Reranker

            reranker = Reranker()
            assert reranker.rerank_documents([], "query") == []
            reranker.reranker.rerank.assert_not_called()

    def test_maps_flashrank_results_to_content_and_float_score(self, monkeypatch):
        monkeypatch.setenv("rerank_model", "ms-marco-MiniLM-L-12-v2")
        flash = MagicMock()
        flash.rerank.return_value = [
            {"id": 1, "text": "second", "score": 0.2, "metadata": {"source": "b"}},
            {"id": 0, "text": "first", "score": 0.9, "metadata": {"source": "a"}},
        ]
        with patch("Reranker.Ranker", return_value=flash), patch("Reranker.RerankRequest") as Req:
            from Reranker import Reranker

            reranker = Reranker()
            docs = [
                {"content": "first", "metadata": {"source": "a"}},
                {"content": "second", "metadata": {"source": "b"}},
            ]
            results = reranker.rerank_documents(docs, "prompt")

        Req.assert_called_once()
        assert results[0]["content"] == "first"
        assert results[0]["score"] == pytest.approx(0.9)
        assert "text" not in results[0]
        assert results[1]["content"] == "second"
