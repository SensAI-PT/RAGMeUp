"""Unit tests for RAGHelper pure helpers and chat pipeline branches."""
import hashlib
import json
from unittest.mock import MagicMock

import numpy as np
import pytest


class TestDeduplicateAndFormat:
    def test_deduplicate_chunks_keeps_last_id_wins_dict_order(self, rag_helper):
        docs = [
            {"id": "a", "content": "first"},
            {"id": "b", "content": "keep"},
            {"id": "a", "content": "second"},
        ]
        result = rag_helper._deduplicate_chunks(docs)
        by_id = {d["id"]: d for d in result}
        assert len(result) == 2
        assert by_id["a"]["content"] == "second"

    def test_format_documents(self, rag_helper, sample_documents):
        formatted = rag_helper.format_documents(sample_documents)
        assert "[Document]" in formatted
        assert "Paris" in formatted
        assert "geo.txt" in formatted
        assert "Metadata" in formatted


class TestHandleDocuments:
    def test_without_rerank_copies_distance_to_score(
        self, rag_helper, monkeypatch, sample_documents
    ):
        monkeypatch.setenv("rerank", "False")
        rag_helper.retriever.get_relevant_documents.return_value = sample_documents
        docs = rag_helper.handle_documents("q", np.array([0.1]), [])
        assert docs[0]["score"] == 0.1
        assert docs[1]["score"] == 0.2

    def test_with_rerank_truncates_to_rerank_k(self, rag_helper, monkeypatch, sample_documents):
        monkeypatch.setenv("rerank", "True")
        monkeypatch.setenv("rerank_k", "1")
        rag_helper.retriever.get_relevant_documents.return_value = sample_documents
        rag_helper.reranker.rerank_documents.return_value = [
            {**sample_documents[0], "score": 0.99},
            {**sample_documents[1], "score": 0.1},
        ]
        docs = rag_helper.handle_documents("q", np.array([0.1]), ["default"])
        assert len(docs) == 1
        assert docs[0]["score"] == 0.99


class TestTextSplitterInit:
    def test_paragraph_chunker(self, rag_helper, monkeypatch):
        monkeypatch.setenv("splitter", "ParagraphChunker")
        monkeypatch.setenv("paragraph_chunker_max_chunk_size", "128")
        monkeypatch.setenv("paragraph_chunker_paragraph_separator", r"\n\s*\n")
        splitter = rag_helper._initialize_text_splitter()
        from ParagraphChunker import ParagraphChunker

        assert isinstance(splitter, ParagraphChunker)
        assert splitter.max_chunk_size == 128

    def test_recursive_splitter(self, rag_helper, monkeypatch):
        monkeypatch.setenv("splitter", "RecursiveCharacterTextSplitter")
        monkeypatch.setenv("recursive_splitter_chunk_size", "500")
        monkeypatch.setenv("recursive_splitter_chunk_overlap", "50")
        splitter = rag_helper._initialize_text_splitter()
        assert splitter is not None


class TestHandleUserInteraction:
    def _base_env(self, monkeypatch):
        monkeypatch.setenv("rerank", "False")
        monkeypatch.setenv("use_hyde", "False")
        monkeypatch.setenv("use_rewrite_loop", "False")
        monkeypatch.setenv("use_re2", "False")
        monkeypatch.setenv("use_summarization", "False")
        monkeypatch.setenv("provenance_method", "none")
        monkeypatch.setenv("rag_instruction", "Context:\n{context}")
        monkeypatch.setenv("rag_question_initial", "Q: {question}")
        monkeypatch.setenv("rag_question_followup", "Follow: {question}")
        monkeypatch.setenv("rag_fetch_new_question", "Need docs for {question}?")

    def test_initial_turn_fetches_and_answers(
        self, rag_helper, monkeypatch, sample_documents
    ):
        self._base_env(monkeypatch)
        rag_helper.retriever.get_relevant_documents.return_value = sample_documents
        rag_helper.llm.generate_response.return_value = (
            "Paris is in France.",
            [{"role": "user", "content": "Where is Paris?"}],
        )

        (
            response,
            documents,
            fetched,
            rewritten,
            history,
            provenance,
        ) = rag_helper.handle_user_interaction("Where is Paris?", [], [])

        assert response == "Paris is in France."
        assert fetched is True
        assert rewritten is None
        assert provenance is None
        assert documents[0]["content"] == sample_documents[0]["content"]
        assert documents[0]["score"] == 0.1
        assert history[-1]["role"] == "assistant"
        rag_helper.embeddings.encode.assert_called()

    def test_followup_skips_retrieval_when_llm_says_no(
        self, rag_helper, monkeypatch, sample_documents
    ):
        self._base_env(monkeypatch)
        history = [
            {"role": "system", "content": "ctx"},
            {"role": "user", "content": "Where is Paris?"},
            {"role": "assistant", "content": "France"},
        ]
        rag_helper.llm.generate_response.side_effect = [
            ("No", history),  # fetch decision
            ("Still France", history),  # final answer
        ]

        response, documents, fetched, *_ = rag_helper.handle_user_interaction(
            "Can you repeat that?", history, []
        )

        assert fetched is False
        assert documents is None
        assert response == "Still France"
        rag_helper.retriever.get_relevant_documents.assert_not_called()

    def test_hyde_replaces_prompt_before_retrieval(self, rag_helper, monkeypatch):
        self._base_env(monkeypatch)
        monkeypatch.setenv("use_hyde", "True")
        monkeypatch.setenv("hyde_query", "Write a passage about: {question}")
        rag_helper.retriever.get_relevant_documents.return_value = []
        rag_helper.llm.generate_response.side_effect = [
            ("Hypothetical passage about cats.", []),
            ("Cats are mammals.", [{"role": "user", "content": "x"}]),
        ]

        rag_helper.handle_user_interaction("What are cats?", [], [])

        encoded_arg = rag_helper.embeddings.encode.call_args[0][0]
        assert "Hypothetical" in encoded_arg

    def test_rewrite_loop_re_retrieves(self, rag_helper, monkeypatch, sample_documents):
        self._base_env(monkeypatch)
        monkeypatch.setenv("use_rewrite_loop", "True")
        monkeypatch.setenv("rewrite_query_instruction", "Docs: {context}")
        monkeypatch.setenv("rewrite_query_question", "Enough for {question}?")
        monkeypatch.setenv(
            "rewrite_query_prompt", "Rewrite {question} because {motivation}"
        )
        rag_helper.retriever.get_relevant_documents.return_value = sample_documents
        rag_helper.llm.generate_response.side_effect = [
            ("No, missing info", []),  # rewrite check
            ("better query about Paris", []),  # rewritten query
            ("Final answer", [{"role": "user", "content": "x"}]),
        ]

        _, _, _, rewritten, _, _ = rag_helper.handle_user_interaction(
            "Paris?", [], []
        )

        assert rewritten == "better query about Paris"
        assert rag_helper.retriever.get_relevant_documents.call_count == 2

    def test_re2_appends_prompt(self, rag_helper, monkeypatch, sample_documents):
        self._base_env(monkeypatch)
        monkeypatch.setenv("use_re2", "True")
        monkeypatch.setenv("re2_prompt", "READ AGAIN")
        rag_helper.retriever.get_relevant_documents.return_value = sample_documents

        captured = {}

        def capture_generate(system, prompt, history):
            captured["prompt"] = prompt
            return ("ans", [{"role": "user", "content": prompt}])

        rag_helper.llm.generate_response.side_effect = capture_generate
        rag_helper.handle_user_interaction("hello", [], [])
        assert "READ AGAIN" in captured["prompt"]
        assert "hello" in captured["prompt"]


class TestHandleUserInteractionStream:
    def test_stream_yields_tokens_and_done(self, rag_helper, monkeypatch, sample_documents):
        monkeypatch.setenv("rerank", "False")
        monkeypatch.setenv("use_hyde", "False")
        monkeypatch.setenv("use_rewrite_loop", "False")
        monkeypatch.setenv("use_re2", "False")
        monkeypatch.setenv("use_summarization", "False")
        monkeypatch.setenv("provenance_method", "none")
        monkeypatch.setenv("rag_instruction", "{context}")
        monkeypatch.setenv("rag_question_initial", "{question}")

        rag_helper.retriever.get_relevant_documents.return_value = sample_documents
        rag_helper.llm.generate_response_stream.return_value = (
            iter(["Hi", "!"]),
            [{"role": "user", "content": "q"}],
        )

        events = list(rag_helper.handle_user_interaction_stream("q", [], []))
        types = [e[0] for e in events]
        assert "step" in types
        assert "documents" in types
        assert types.count("token") == 2
        assert types[-1] == "done"
        done = events[-1][1]
        assert done["reply"] == "Hi!"
        assert done["fetched_new_documents"] is True


class TestAddDocumentTxt:
    def test_add_txt_chunks_and_upserts(self, rag_helper, tmp_path, monkeypatch):
        monkeypatch.setenv("file_types", "txt")
        path = tmp_path / "note.txt"
        path.write_text("hello world", encoding="utf-8")
        rag_helper.splitter.split_text.return_value = ["hello world"]
        rag_helper.embeddings.encode.return_value = np.array([0.1, 0.2])

        rag_helper.add_document(str(path), "demo")

        rag_helper.retriever.add_documents.assert_called_once()
        docs = rag_helper.retriever.add_documents.call_args[0][0]
        assert len(docs) == 1
        assert docs[0]["id"] == hashlib.md5(b"hello world").hexdigest()
        meta = json.loads(docs[0]["metadata"])
        assert meta["dataset"] == "demo"
        assert meta["source"] == str(path)


class TestComputeProvenanceScores:
    def test_none_method_returns_none(self, rag_helper, monkeypatch, sample_documents):
        monkeypatch.setenv("provenance_method", "none")
        assert (
            rag_helper.compute_provenance_scores("q", sample_documents, "a") is None
        )

    def test_llm_method_delegates(self, rag_helper, monkeypatch, sample_documents):
        monkeypatch.setenv("provenance_method", "llm")
        monkeypatch.setenv("provenance_llm_prompt", "{query}{context}{answer}")
        monkeypatch.setenv("attribute_include_query", "True")
        rag_helper.llm.generate_response.return_value = ("0.5", [])
        scores = rag_helper.compute_provenance_scores("q", sample_documents, "a")
        assert len(scores) == 2
