"""End-to-end pipeline tests with mocked LLM / embeddings / retriever."""
import pytest


@pytest.mark.integration
class TestRagPipelineBranches:
    def test_summarization_triggers_when_history_large(
        self, rag_helper, monkeypatch, sample_documents
    ):
        monkeypatch.setenv("rerank", "False")
        monkeypatch.setenv("use_hyde", "False")
        monkeypatch.setenv("use_rewrite_loop", "False")
        monkeypatch.setenv("use_re2", "False")
        monkeypatch.setenv("use_summarization", "True")
        monkeypatch.setenv("summarization_threshold", "10")
        monkeypatch.setenv("summarization_query", "Summarize: {history}")
        monkeypatch.setenv("provenance_method", "none")
        monkeypatch.setenv("rag_instruction", "{context}")
        monkeypatch.setenv("rag_question_followup", "{question}")
        monkeypatch.setenv("rag_fetch_new_question", "Need docs for {question}?")

        rag_helper.tiktoken_encoder = type(
            "Enc",
            (),
            {"encode": staticmethod(lambda s: list(range(50)))},
        )()
        rag_helper.retriever.get_relevant_documents.return_value = sample_documents
        rag_helper.llm.generate_response.side_effect = [
            ("summary of chat", []),  # summarization
            ("Yes", []),  # fetch decision
            ("final", [{"role": "user", "content": "q"}]),
        ]

        history = [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
        ]
        response, *_ = rag_helper.handle_user_interaction("new q", history, [])
        assert response == "final"
        assert rag_helper.llm.generate_response.call_count == 3

    def test_followup_with_new_docs_strips_system_from_history(
        self, rag_helper, monkeypatch, sample_documents
    ):
        monkeypatch.setenv("rerank", "False")
        monkeypatch.setenv("use_hyde", "False")
        monkeypatch.setenv("use_rewrite_loop", "False")
        monkeypatch.setenv("use_re2", "False")
        monkeypatch.setenv("use_summarization", "False")
        monkeypatch.setenv("provenance_method", "none")
        monkeypatch.setenv("rag_instruction", "CTX {context}")
        monkeypatch.setenv("rag_question_followup", "Q {question}")
        monkeypatch.setenv("rag_fetch_new_question", "Need {question}?")

        history = [
            {"role": "system", "content": "old system"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        rag_helper.retriever.get_relevant_documents.return_value = sample_documents
        calls = []

        def gen(system, prompt, hist):
            calls.append({"system": system, "hist": hist})
            return ("ans", list(hist) + [{"role": "user", "content": prompt}])

        rag_helper.llm.generate_response.side_effect = gen

        # First call: fetch decision → "Yes..."; subsequent: final answer
        responses = iter(
            [
                ("Yes fetch", []),
            ]
        )

        def side_effect(system, prompt, hist):
            try:
                return next(responses)
            except StopIteration:
                return gen(system, prompt, hist)

        rag_helper.llm.generate_response.side_effect = side_effect

        rag_helper.handle_user_interaction("follow", history, ["default"])
        assert calls, "expected final generate_response call"
        assert calls[0]["system"] is not None
        assert all(m["role"] != "system" for m in calls[0]["hist"])
