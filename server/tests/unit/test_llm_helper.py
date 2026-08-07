"""Unit tests for LLMHelper (no live API calls)."""
from unittest.mock import MagicMock, patch

import pytest

from LLMHelper import LLMHelper, OllamaClient


@pytest.fixture
def llm_env(monkeypatch):
    """Clear LLM provider flags so each test can opt in explicitly."""
    for key in (
        "use_openai",
        "use_azure",
        "use_gemini",
        "use_anthropic",
        "use_ollama",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME",
        "AZURE_OPENAI_API_VERSION",
        "ANTHROPIC_API_KEY",
        "ollama_model_name",
        "openai_model_name",
        "temperature",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("temperature", "0.0")


class TestCleanReply:
    def test_passthrough_plain_text(self, logger, llm_env, monkeypatch):
        monkeypatch.setenv("use_ollama", "True")
        monkeypatch.setenv("ollama_model_name", "test")
        helper = LLMHelper(logger)
        assert helper.clean_reply("hello") == "hello"

    def test_strips_fenced_code_block(self, logger, llm_env, monkeypatch):
        monkeypatch.setenv("use_ollama", "True")
        monkeypatch.setenv("ollama_model_name", "test")
        helper = LLMHelper(logger)
        reply = "```json\n{\"a\": 1}\n```"
        assert helper.clean_reply(reply) == '{"a": 1}'

    def test_does_not_strip_partial_fences(self, logger, llm_env, monkeypatch):
        monkeypatch.setenv("use_ollama", "True")
        monkeypatch.setenv("ollama_model_name", "test")
        helper = LLMHelper(logger)
        reply = "```not closed"
        assert helper.clean_reply(reply) == reply


class TestInitializeClient:
    def test_raises_when_no_backend(self, logger, llm_env):
        with pytest.raises(ValueError, match="No LLM backend"):
            LLMHelper(logger)

    def test_openai_missing_key_returns_none(self, logger, llm_env, monkeypatch):
        monkeypatch.setenv("use_openai", "True")
        helper = LLMHelper(logger)
        assert helper.client is None

    def test_openai_with_key(self, logger, llm_env, monkeypatch):
        monkeypatch.setenv("use_openai", "True")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with patch("LLMHelper.openai.OpenAI") as OpenAI:
            OpenAI.return_value = MagicMock(name="openai-client")
            helper = LLMHelper(logger)
            OpenAI.assert_called_once_with(api_key="sk-test")
            assert helper.client is OpenAI.return_value

    def test_azure_requires_all_settings(self, logger, llm_env, monkeypatch):
        monkeypatch.setenv("use_azure", "True")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
        # missing endpoint / deployment / version
        helper = LLMHelper(logger)
        assert helper.client is None

    def test_azure_with_full_config(self, logger, llm_env, monkeypatch):
        monkeypatch.setenv("use_azure", "True")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
        monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o")
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        with patch("LLMHelper.openai.AzureOpenAI") as Azure:
            Azure.return_value = MagicMock()
            helper = LLMHelper(logger)
            assert helper.client is Azure.return_value

    def test_gemini_missing_key(self, logger, llm_env, monkeypatch):
        monkeypatch.setenv("use_gemini", "True")
        helper = LLMHelper(logger)
        assert helper.client is None

    def test_anthropic_missing_key(self, logger, llm_env, monkeypatch):
        monkeypatch.setenv("use_anthropic", "True")
        helper = LLMHelper(logger)
        assert helper.client is None

    def test_ollama_client(self, logger, llm_env, monkeypatch):
        monkeypatch.setenv("use_ollama", "True")
        monkeypatch.setenv("ollama_model_name", "llama3")
        helper = LLMHelper(logger)
        assert isinstance(helper.client, OllamaClient)
        assert helper.client.model == "llama3"


class TestGenerateResponse:
    def test_openai_injects_system_prompt_and_returns_content(
        self, logger, llm_env, monkeypatch
    ):
        monkeypatch.setenv("use_openai", "True")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("openai_model_name", "gpt-4o-mini")

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="answer"))]
        )
        with patch("LLMHelper.openai.OpenAI", return_value=mock_client):
            helper = LLMHelper(logger)
            response, thread = helper.generate_response("sys", "user q", [])

        assert response == "answer"
        assert thread[0] == {"role": "system", "content": "sys"}
        assert thread[-1] == {"role": "user", "content": "user q"}
        mock_client.chat.completions.create.assert_called_once()

    def test_overwrites_existing_system_message(self, logger, llm_env, monkeypatch):
        monkeypatch.setenv("use_openai", "True")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("openai_model_name", "gpt-4o-mini")

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="ok"))]
        )
        history = [{"role": "system", "content": "old"}, {"role": "user", "content": "hi"}]
        with patch("LLMHelper.openai.OpenAI", return_value=mock_client):
            helper = LLMHelper(logger)
            _, thread = helper.generate_response("new sys", "follow up", history)

        assert thread[0]["content"] == "new sys"

    def test_ollama_path(self, logger, llm_env, monkeypatch):
        monkeypatch.setenv("use_ollama", "True")
        monkeypatch.setenv("ollama_model_name", "llama3")
        helper = LLMHelper(logger)
        helper.client = MagicMock()
        helper.client.chat.return_value = "ollama says hi"
        response, thread = helper.generate_response(None, "hello", [])
        assert response == "ollama says hi"
        assert thread == [{"role": "user", "content": "hello"}]


class TestGenerateResponseStream:
    def test_openai_stream_yields_deltas(self, logger, llm_env, monkeypatch):
        monkeypatch.setenv("use_openai", "True")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("openai_model_name", "gpt-4o-mini")

        chunk1 = MagicMock()
        chunk1.choices = [MagicMock(delta=MagicMock(content="Hel"))]
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock(delta=MagicMock(content="lo"))]
        empty = MagicMock()
        empty.choices = [MagicMock(delta=MagicMock(content=None))]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = [chunk1, chunk2, empty]
        with patch("LLMHelper.openai.OpenAI", return_value=mock_client):
            helper = LLMHelper(logger)
            gen, thread = helper.generate_response_stream(None, "q", [])
            assert "".join(gen) == "Hello"
            assert thread[-1]["content"] == "q"

    def test_stream_raises_without_backend(self, logger, llm_env, monkeypatch):
        monkeypatch.setenv("use_ollama", "True")
        monkeypatch.setenv("ollama_model_name", "x")
        helper = LLMHelper(logger)
        # Clear all backends after init
        for key in ("use_openai", "use_azure", "use_gemini", "use_anthropic", "use_ollama"):
            monkeypatch.setenv(key, "False")
        with pytest.raises(ValueError, match="No LLM backend"):
            helper.generate_response_stream(None, "q", [])
