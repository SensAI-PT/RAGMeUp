"""Unit tests for provenance helpers."""
from unittest.mock import MagicMock

import numpy as np
import pytest

from provenance import compute_llm_provenance, compute_rerank_provenance


class TestComputeRerankProvenance:
    def test_uses_answer_only_by_default(self, monkeypatch, sample_documents):
        monkeypatch.delenv("attribute_include_query", raising=False)
        reranker = MagicMock()
        reranker.rerank_documents.return_value = sample_documents

        result = compute_rerank_provenance(reranker, "query?", sample_documents, "answer")

        reranker.rerank_documents.assert_called_once_with(sample_documents, "answer")
        assert result is sample_documents

    def test_prepends_query_when_configured(self, monkeypatch, sample_documents):
        monkeypatch.setenv("attribute_include_query", "True")
        reranker = MagicMock()
        reranker.rerank_documents.return_value = sample_documents

        compute_rerank_provenance(reranker, "query?", sample_documents, "answer")

        reranker.rerank_documents.assert_called_once_with(
            sample_documents, "query?\nanswer"
        )


class TestComputeLlmProvenance:
    def test_scores_each_document(self, monkeypatch, sample_documents):
        monkeypatch.setenv(
            "provenance_llm_prompt",
            "q={query} c={context} a={answer}",
        )
        monkeypatch.setenv("attribute_include_query", "True")
        llm = MagicMock()
        llm.generate_response.side_effect = [("0.9", []), ("0.1", [])]

        scores = compute_llm_provenance(llm, "Q", sample_documents, "A")

        assert scores == [{"score": "0.9"}, {"score": "0.1"}]
        assert llm.generate_response.call_count == 2

    def test_escapes_braces_in_document_content(self, monkeypatch):
        monkeypatch.setenv(
            "provenance_llm_prompt",
            "{query}{context}{answer}",
        )
        monkeypatch.setenv("attribute_include_query", "True")
        docs = [{"content": "has {braces}", "metadata": {}}]
        llm = MagicMock()
        llm.generate_response.return_value = ("1", [])

        compute_llm_provenance(llm, "q", docs, "a")

        # Content braces doubled for str.format safety; prompt still formats
        assert docs[0]["content"] == "has {{braces}}"


class TestDocumentSimilarityAttribution:
    def test_compute_similarity_returns_one_score_per_doc(self, monkeypatch, sample_documents):
        monkeypatch.setenv("embedding_cpu", "True")
        monkeypatch.setenv("provenance_similarity_llm", "fake-model")
        monkeypatch.setenv("attribute_include_query", "False")

        # Import after stubs/env are ready
        from provenance import DocumentSimilarityAttribution
        import sklearn.metrics.pairwise as pairwise

        attr = DocumentSimilarityAttribution.__new__(DocumentSimilarityAttribution)
        attr.model = MagicMock()
        # encode called for answer then for context texts
        attr.model.encode.side_effect = [
            np.array([[0.1, 0.2]]),  # answer
            np.array([[0.1, 0.2], [0.3, 0.4]]),  # contexts
        ]
        pairwise.cosine_similarity = MagicMock(return_value=np.array([[0.95]]))

        scores = attr.compute_similarity("q", sample_documents, "answer text")
        assert len(scores) == 2
        assert all("score" in s for s in scores)
