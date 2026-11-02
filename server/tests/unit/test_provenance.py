"""Unit tests for provenance helpers."""
from unittest.mock import MagicMock

import numpy as np
import pytest

import provenance
from provenance import (
    DocumentSimilarityAttribution,
    compute_llm_provenance,
    compute_rerank_provenance,
)


@pytest.fixture(autouse=True)
def clean_provenance_env(monkeypatch):
    """Keep the documented and legacy query-inclusion flags out of each test."""
    monkeypatch.delenv("provenance_include_query", raising=False)
    monkeypatch.delenv("attribute_include_query", raising=False)


class TestComputeRerankProvenance:
    def test_uses_answer_only_by_default(self, sample_documents):
        reranker = MagicMock()
        reranker.rerank_documents.return_value = sample_documents

        result = compute_rerank_provenance(reranker, "query?", sample_documents, "answer")

        reranker.rerank_documents.assert_called_once_with(sample_documents, "answer")
        assert result is sample_documents

    def test_prepends_query_when_documented_setting_is_true(
        self, monkeypatch, sample_documents
    ):
        monkeypatch.setenv("provenance_include_query", "True")
        reranker = MagicMock()
        reranker.rerank_documents.return_value = sample_documents

        compute_rerank_provenance(reranker, "query?", sample_documents, "answer")

        reranker.rerank_documents.assert_called_once_with(
            sample_documents, "query?\nanswer"
        )

    def test_legacy_attribute_include_query_still_works(
        self, monkeypatch, sample_documents
    ):
        monkeypatch.setenv("attribute_include_query", "True")
        reranker = MagicMock()
        reranker.rerank_documents.return_value = sample_documents

        compute_rerank_provenance(reranker, "query?", sample_documents, "answer")

        reranker.rerank_documents.assert_called_once_with(
            sample_documents, "query?\nanswer"
        )

    def test_documented_setting_takes_precedence_over_legacy(
        self, monkeypatch, sample_documents
    ):
        monkeypatch.setenv("provenance_include_query", "False")
        monkeypatch.setenv("attribute_include_query", "True")
        reranker = MagicMock()
        reranker.rerank_documents.return_value = sample_documents

        compute_rerank_provenance(reranker, "query?", sample_documents, "answer")

        reranker.rerank_documents.assert_called_once_with(sample_documents, "answer")


class TestComputeLlmProvenance:
    def test_scores_each_document_with_query(self, monkeypatch, sample_documents):
        monkeypatch.setenv(
            "provenance_llm_prompt",
            "q={query} c={context} a={answer}",
        )
        monkeypatch.setenv("provenance_include_query", "True")
        llm = MagicMock()
        llm.generate_response.side_effect = [("0.9", []), ("0.1", [])]

        scores = compute_llm_provenance(llm, "Q", sample_documents, "A")

        assert scores == [{"score": "0.9"}, {"score": "0.1"}]
        assert llm.generate_response.call_count == 2
        prompts = [call.args[1] for call in llm.generate_response.call_args_list]
        assert all("The user asked Q" in prompt for prompt in prompts)

    def test_query_is_omitted_when_disabled(self, monkeypatch, sample_documents):
        monkeypatch.setenv(
            "provenance_llm_prompt",
            "q={query} c={context} a={answer}",
        )
        monkeypatch.setenv("provenance_include_query", "False")
        llm = MagicMock()
        llm.generate_response.return_value = ("1", [])

        compute_llm_provenance(llm, "Q", sample_documents, "A")

        prompts = [call.args[1] for call in llm.generate_response.call_args_list]
        assert all("The user asked Q" not in prompt for prompt in prompts)

    def test_does_not_mutate_document_content(self, monkeypatch):
        monkeypatch.setenv(
            "provenance_llm_prompt",
            "{query}{context}{answer}",
        )
        monkeypatch.setenv("provenance_include_query", "False")
        docs = [{"content": "has {braces}", "metadata": {}}]
        llm = MagicMock()
        llm.generate_response.return_value = ("1", [])

        compute_llm_provenance(llm, "q", docs, "a")

        # The caller still owns the original document and can safely return it.
        assert docs[0]["content"] == "has {braces}"
        prompt = llm.generate_response.call_args.args[1]
        assert "has {braces}" in prompt
        assert "has {{braces}}" not in prompt


class TestDocumentSimilarityAttribution:
    def test_compute_similarity_returns_one_score_per_doc(
        self, monkeypatch, sample_documents
    ):
        monkeypatch.setenv("embedding_cpu", "True")
        monkeypatch.setenv("provenance_similarity_llm", "fake-model")
        monkeypatch.setenv("provenance_include_query", "False")

        attr = DocumentSimilarityAttribution.__new__(DocumentSimilarityAttribution)
        attr.model = MagicMock()
        # encode called for answer then for context texts
        attr.model.encode.side_effect = [
            np.array([[0.1, 0.2]]),  # answer
            np.array([[0.1, 0.2], [0.3, 0.4]]),  # contexts
        ]
        monkeypatch.setattr(
            provenance, "cosine_similarity", MagicMock(return_value=np.array([[0.95]]))
        )

        scores = attr.compute_similarity("q", sample_documents, "answer text")
        assert len(scores) == 2
        assert all("score" in s for s in scores)

    def test_compares_each_document_against_answer(self, monkeypatch):
        monkeypatch.setenv("provenance_include_query", "False")
        calls = []

        def fake_cosine(left, right):
            calls.append((left.copy(), right.copy()))
            return np.array([[1.0]])

        monkeypatch.setattr(provenance, "cosine_similarity", fake_cosine)

        attr = DocumentSimilarityAttribution.__new__(DocumentSimilarityAttribution)
        attr.model = MagicMock()
        attr.model.encode.side_effect = [
            np.array([[2.0]]),  # answer
            np.array([[3.0], [5.0]]),  # two documents
        ]
        docs = [
            {"content": "doc one", "metadata": {}},
            {"content": "doc two", "metadata": {}},
        ]

        attr.compute_similarity("q", docs, "answer")

        assert len(calls) == 2
        assert calls[0][0].tolist() == [[3.0]]
        assert calls[0][1].tolist() == [[2.0]]
        assert calls[1][0].tolist() == [[5.0]]
        assert calls[1][1].tolist() == [[2.0]]
