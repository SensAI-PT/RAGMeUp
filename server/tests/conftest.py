"""
Shared fixtures for RAGMeUp server tests.

Sets RAGMEUP_TESTING before any server import so bootstrap is skipped, and
stubs optional heavy ML packages when they are not installed so pure unit
tests can still import production modules.
"""
from __future__ import annotations

import logging
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure server/ is on sys.path (pytest.ini also sets pythonpath)
SERVER_ROOT = Path(__file__).resolve().parents[1]
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

# Must be set before importing server.py
os.environ["RAGMEUP_TESTING"] = "1"


def _ensure_module(name: str, attrs: dict | None = None) -> types.ModuleType:
    """Register a lightweight stub module if the real package is missing."""
    if name in sys.modules:
        return sys.modules[name]
    try:
        __import__(name)
        return sys.modules[name]
    except Exception:
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs or {})
        # Mark as package when dotted parents need children
        if "." not in name:
            mod.__path__ = []
        sys.modules[name] = mod
        return mod


def _stub_heavy_imports() -> None:
    """Allow importing RAGHelper / provenance / Reranker without full ML stack."""
    # google.genai is imported as `from google import genai` and `from google.genai import types`
    google = _ensure_module("google")
    genai = _ensure_module("google.genai")
    genai_types = _ensure_module("google.genai.types")
    if not hasattr(google, "genai"):
        google.genai = genai
    genai.Client = MagicMock
    genai.types = genai_types
    genai_types.UserContent = MagicMock
    genai_types.ModelContent = MagicMock
    genai_types.GenerateContentConfig = MagicMock

    for name in (
        "openai",
        "anthropic",
        "ollama",
        "sentence_transformers",
        "flashrank",
        "tiktoken",
        "jq",
        "pptx",
        "docling",
        "docling.document_converter",
        "langchain_text_splitters",
        "langchain_experimental",
        "langchain_experimental.text_splitter",
        "sklearn",
        "sklearn.metrics",
        "sklearn.metrics.pairwise",
    ):
        _ensure_module(name)

    # Concrete attributes used at import / call time
    sys.modules["ollama"].chat = MagicMock()
    openai_mod = sys.modules["openai"]
    if not hasattr(openai_mod, "OpenAI") or isinstance(getattr(openai_mod, "OpenAI", None), MagicMock):
        openai_mod.OpenAI = MagicMock
        openai_mod.AzureOpenAI = MagicMock
    # Always ensure attributes exist on stubs
    for attr, value in (("OpenAI", MagicMock), ("AzureOpenAI", MagicMock)):
        if not hasattr(openai_mod, attr):
            setattr(openai_mod, attr, value)
    anthropic_mod = sys.modules["anthropic"]
    if not hasattr(anthropic_mod, "Anthropic"):
        anthropic_mod.Anthropic = MagicMock
    sys.modules["sentence_transformers"].SentenceTransformer = MagicMock
    sys.modules["flashrank"].Ranker = MagicMock
    sys.modules["flashrank"].RerankRequest = MagicMock
    sys.modules["docling.document_converter"].DocumentConverter = MagicMock
    sys.modules["langchain_text_splitters"].RecursiveCharacterTextSplitter = MagicMock
    sys.modules["langchain_experimental.text_splitter"].SemanticChunker = MagicMock
    sys.modules["sklearn.metrics.pairwise"].cosine_similarity = MagicMock(
        return_value=[[1.0]]
    )
    sys.modules["pptx"].Presentation = MagicMock


_stub_heavy_imports()


@pytest.fixture
def logger():
    log = logging.getLogger("ragmeup-test")
    log.setLevel(logging.DEBUG)
    return log


@pytest.fixture
def sample_documents():
    return [
        {
            "content": "Paris is the capital of France.",
            "metadata": {"source": "geo.txt", "dataset": "default", "distance": 0.1},
        },
        {
            "content": "Berlin is the capital of Germany.",
            "metadata": {"source": "geo.txt", "dataset": "default", "distance": 0.2},
        },
    ]


@pytest.fixture
def mock_llm(logger):
    """LLMHelper stand-in that returns controllable canned replies."""
    llm = MagicMock()
    llm.generate_response.return_value = ("Yes", [{"role": "user", "content": "q"}])
    llm.generate_response_stream.return_value = (
        iter(["Hello", " ", "world"]),
        [{"role": "user", "content": "q"}],
    )
    return llm


@pytest.fixture
def mock_embeddings():
    emb = MagicMock()
    emb.encode.return_value = MagicMock(tolist=lambda: [0.1, 0.2, 0.3])
    emb.get_sentence_embedding_dimension.return_value = 3
    return emb


@pytest.fixture
def mock_retriever(sample_documents):
    retriever = MagicMock()
    retriever.get_relevant_documents.return_value = sample_documents
    retriever.has_data.return_value = True
    retriever.get_all_document_names.return_value = [
        {"filename": "geo.txt", "dataset": "default"}
    ]
    retriever.get_datasets.return_value = ["default"]
    retriever.delete.return_value = 2
    return retriever


@pytest.fixture
def rag_helper(logger, mock_llm, mock_embeddings, mock_retriever):
    """RAGHelper without running the real __init__ (no models / DB)."""
    from RAGHelper import RAGHelper

    helper = RAGHelper.__new__(RAGHelper)
    helper.logger = logger
    helper.db_pool = MagicMock()
    helper.llm = mock_llm
    helper.embeddings = mock_embeddings
    helper.retriever = mock_retriever
    helper.converter = MagicMock()
    helper.splitter = MagicMock()
    helper.splitter.split_text.return_value = ["chunk one", "chunk two"]
    helper.reranker = MagicMock()
    helper.similarity_attribution = MagicMock()
    return helper


@pytest.fixture
def mock_raghelper(sample_documents):
    """Fully mocked RAGHelper for Flask route contract tests."""
    helper = MagicMock()
    helper.llm.generate_response.return_value = ("📘 Title 📗", [])
    helper.retriever.get_all_document_names.return_value = [
        {"filename": "geo.txt", "dataset": "default"}
    ]
    helper.retriever.get_datasets.return_value = ["default"]
    helper.retriever.delete.return_value = 2
    helper.handle_user_interaction.return_value = (
        "reply",
        sample_documents,
        True,
        None,
        [{"role": "assistant", "content": "reply"}],
        None,
    )
    return helper


@pytest.fixture
def flask_app(mock_raghelper, tmp_path, monkeypatch):
    """Flask app with mocked raghelper and a temp data directory."""
    monkeypatch.setenv("RAGMEUP_TESTING", "1")
    monkeypatch.setenv("data_directory", str(tmp_path))

    import server as server_module

    server_module.raghelper = mock_raghelper
    server_module.app.config.update(TESTING=True)
    return server_module


@pytest.fixture
def client(flask_app):
    return flask_app.app.test_client()

@pytest.fixture
def postgres_uri():
    """URI for ParadeDB integration tests; None if not configured."""
    uri = os.getenv("TEST_POSTGRES_URI") or os.getenv("postgres_uri")
    if uri:
        return uri
    # Fall back to server/.env without overriding the process broadly
    env_path = SERVER_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("postgres_uri="):
                return stripped.split("=", 1)[1].strip().strip('"').strip("'")
    return None



@pytest.fixture
def db_pool(postgres_uri):
    """Live connection pool; skips when ParadeDB is unreachable."""
    if not postgres_uri:
        pytest.skip("Set TEST_POSTGRES_URI (or postgres_uri) for ParadeDB integration tests")

    import psycopg2
    from psycopg2 import pool

    try:
        pool_obj = pool.SimpleConnectionPool(minconn=1, maxconn=3, dsn=postgres_uri)
        conn = pool_obj.getconn()
        conn.cursor().execute("SELECT 1")
        pool_obj.putconn(conn)
    except Exception as exc:
        pytest.skip(f"ParadeDB not reachable at {postgres_uri}: {exc}")

    yield pool_obj
    pool_obj.closeall()
