"""Flask API integration tests with a mocked RAGHelper."""
import io
from pathlib import Path

import pytest


@pytest.mark.integration
class TestCreateTitle:
    def test_returns_title(self, client, mock_raghelper):
        mock_raghelper.llm.generate_response.return_value = ("📘 Paris 📗", [])
        resp = client.post("/create_title", json={"question": "Where is Paris?"})
        assert resp.status_code == 200
        assert resp.get_json()["title"] == "📘 Paris 📗"


@pytest.mark.integration
class TestChat:
    def test_chat_success(self, client, mock_raghelper, sample_documents):
        mock_raghelper.handle_user_interaction.return_value = (
            "Paris is in France.",
            sample_documents,
            True,
            None,
            [{"role": "assistant", "content": "Paris is in France."}],
            None,
        )
        resp = client.post(
            "/chat",
            json={"prompt": "Where is Paris?", "history": [], "docs": [], "datasets": []},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["reply"] == "Paris is in France."
        assert body["fetched_new_documents"] is True
        assert body["question"] == "Where is Paris?"

    def test_chat_keeps_prior_docs_when_not_fetched(self, client, mock_raghelper):
        prior = [{"content": "old", "metadata": {"source": "x"}}]
        mock_raghelper.handle_user_interaction.return_value = (
            "ok",
            None,
            False,
            None,
            [],
            None,
        )
        resp = client.post(
            "/chat",
            json={"prompt": "again", "history": [], "docs": prior, "datasets": []},
        )
        assert resp.get_json()["documents"] == prior

    def test_chat_attaches_provenance(self, client, mock_raghelper, sample_documents):
        mock_raghelper.handle_user_interaction.return_value = (
            "ans",
            [dict(d) for d in sample_documents],
            True,
            None,
            [],
            [{"score": 0.8}, {"score": 0.2}],
        )
        resp = client.post("/chat", json={"prompt": "q", "history": [], "docs": []})
        docs = resp.get_json()["documents"]
        assert docs[0]["provenance"] == 0.8
        assert docs[1]["provenance"] == 0.2


@pytest.mark.integration
class TestChatStream:
    def test_sse_events(self, client, mock_raghelper, sample_documents):
        def fake_stream(prompt, history, datasets):
            yield ("step", "Retrieving...")
            yield ("token", "Hi")
            yield ("token", "!")
            yield (
                "done",
                {
                    "reply": "Hi!",
                    "history": [{"role": "assistant", "content": "Hi!"}],
                    "documents": sample_documents,
                    "rewritten": None,
                    "fetched_new_documents": True,
                    "provenance_scores": None,
                },
            )

        mock_raghelper.handle_user_interaction_stream.side_effect = fake_stream
        resp = client.post(
            "/chat_stream",
            json={"prompt": "q", "history": [], "docs": [], "datasets": []},
        )
        assert resp.status_code == 200
        assert resp.mimetype == "text/event-stream"
        payload = resp.data.decode("utf-8")
        assert "event: step" in payload
        assert "event: token" in payload
        assert "event: done" in payload
        assert "Hi!" in payload


@pytest.mark.integration
class TestDocumentsEndpoints:
    def test_get_documents(self, client, mock_raghelper):
        resp = client.get("/get_documents")
        assert resp.status_code == 200
        assert resp.get_json() == [{"filename": "geo.txt", "dataset": "default"}]

    def test_get_datasets(self, client, mock_raghelper):
        resp = client.get("/get_datasets")
        assert resp.get_json() == ["default"]

    def test_get_document_not_found(self, client):
        resp = client.post("/get_document", json={"filename": "missing.txt"})
        assert resp.status_code == 404

    def test_get_document_sends_file(self, client, tmp_path, monkeypatch):
        monkeypatch.setenv("data_directory", str(tmp_path))
        f = tmp_path / "hello.txt"
        f.write_text("content", encoding="utf-8")
        resp = client.post("/get_document", json={"filename": "hello.txt"})
        assert resp.status_code == 200
        assert resp.data == b"content"

    def test_delete_missing_file(self, client):
        resp = client.post("/delete", json={"filename": "nope.txt"})
        assert resp.status_code == 404

    def test_delete_removes_file_and_db_rows(self, client, mock_raghelper, tmp_path, monkeypatch):
        monkeypatch.setenv("data_directory", str(tmp_path))
        f = tmp_path / "bye.txt"
        f.write_text("x", encoding="utf-8")
        mock_raghelper.retriever.delete.return_value = 1
        resp = client.post("/delete", json={"filename": "bye.txt"})
        assert resp.status_code == 200
        assert resp.get_json()["count"] == 1
        assert not f.exists()
        mock_raghelper.retriever.delete.assert_called_once()

    def test_add_document_requires_file(self, client):
        resp = client.post("/add_document", data={"dataset": "d"})
        assert resp.status_code == 400

    def test_add_document_requires_dataset(self, client):
        data = {"file": (io.BytesIO(b"hi"), "a.txt")}
        resp = client.post("/add_document", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400
        assert "dataset" in resp.get_json()["error"].lower()

    def test_add_document_saves_and_indexes(self, client, mock_raghelper, tmp_path, monkeypatch):
        monkeypatch.setenv("data_directory", str(tmp_path))
        data = {
            "file": (io.BytesIO(b"hello"), "note.txt"),
            "dataset": "demo",
        }
        resp = client.post(
            "/add_document", data=data, content_type="multipart/form-data"
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["dataset"] == "demo"
        assert Path(body["file"]).exists()
        mock_raghelper.add_document.assert_called_once()


@pytest.mark.integration
class TestConfigEndpoints:
    def test_get_config_empty_when_missing(self, client, flask_app, tmp_path, monkeypatch):
        monkeypatch.setattr(flask_app, "_env_file_path", lambda: str(tmp_path / "nope.env"))
        resp = client.get("/config")
        assert resp.status_code == 200
        assert resp.get_json() == {}

    def test_get_config_reads_env(self, client, flask_app, tmp_path, monkeypatch):
        env_path = tmp_path / ".env"
        env_path.write_text("foo=bar\n# comment\nbaz=1\n", encoding="utf-8")
        monkeypatch.setattr(flask_app, "_env_file_path", lambda: str(env_path))
        resp = client.get("/config")
        assert resp.get_json() == {"foo": "bar", "baz": "1"}

    def test_put_config_requires_values(self, client):
        resp = client.put("/config", json={"config": {}})
        assert resp.status_code == 400

    def test_put_config_updates_and_optionally_reloads(
        self, client, flask_app, mock_raghelper, tmp_path, monkeypatch
    ):
        env_path = tmp_path / ".env"
        env_path.write_text("existing=1\nkeep=yes\n", encoding="utf-8")
        monkeypatch.setattr(flask_app, "_env_file_path", lambda: str(env_path))

        resp = client.put(
            "/config",
            json={"config": {"existing": "2", "new_key": "x"}, "reinitialize": True},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["status"] == "ok"
        text = env_path.read_text(encoding="utf-8")
        assert "existing=2" in text
        assert "new_key=x" in text
        assert "keep=yes" in text
        mock_raghelper.reload_llm.assert_called_once()
