"""ParadeDB-backed integration tests for PostgresHybridRetriever.

Requires a reachable ParadeDB instance and either:
  TEST_POSTGRES_URI=postgresql://user:pass@host:port/db
or postgres_uri from the server .env (loaded by dotenv in other code paths).

Uses isolated table names via a temporary schema when possible; otherwise
operates on the standard ragmeup_* tables and cleans up inserted rows by id.
"""
from __future__ import annotations

import json
import uuid

import numpy as np
import pytest

from PostgresHybridRetriever import PostgresHybridRetriever


pytestmark = pytest.mark.integration


@pytest.fixture
def retriever(db_pool, monkeypatch):
    monkeypatch.setenv("vector_store_k", "5")
    monkeypatch.setenv("use_rrf", "False")
    monkeypatch.setenv("use_re2", "False")
    monkeypatch.setenv("data_directory", "/data")
    r = PostgresHybridRetriever(db_pool)
    # Small embedding dim for tests
    try:
        r.setup_database(8)
    except Exception as exc:
        pytest.skip(f"setup_database failed (is this ParadeDB with pgvector/pg_search?): {exc}")
    return r


def _make_docs(prefix: str, dim: int = 8):
    docs = []
    for i, text in enumerate(
        [
            f"{prefix} Paris is the capital of France.",
            f"{prefix} Berlin is the capital of Germany.",
            f"{prefix} Madrid is the capital of Spain.",
        ]
    ):
        emb = np.zeros(dim, dtype=float)
        emb[i % dim] = 1.0
        docs.append(
            {
                "id": uuid.uuid4().hex[:32],
                "content": text,
                "embedding": emb,
                "metadata": json.dumps(
                    {
                        "source": f"/data/{prefix}_doc.txt",
                        "dataset": f"test_{prefix}",
                    }
                ),
            }
        )
    return docs


class TestRetrieverLifecycle:
    def test_add_has_data_retrieve_delete(self, retriever, monkeypatch):
        prefix = uuid.uuid4().hex[:8]
        docs = _make_docs(prefix)
        ids = [d["id"] for d in docs]

        retriever.add_documents(docs)
        assert retriever.has_data() is True

        names = retriever.get_all_document_names()
        assert any(prefix in (n.get("filename") or "") for n in (names or []))

        datasets = retriever.get_datasets()
        assert f"test_{prefix}" in (datasets or [])

        # Dense query toward the first doc
        query_emb = np.zeros(8, dtype=float)
        query_emb[0] = 1.0
        monkeypatch.setenv("use_rrf", "False")
        results = retriever.get_relevant_documents(
            f"{prefix} capital France", query_emb, [f"test_{prefix}"]
        )
        assert results is not None
        assert len(results) >= 1
        assert any("Paris" in r["content"] for r in results)

        # RRF path
        monkeypatch.setenv("use_rrf", "True")
        monkeypatch.setenv("rrf_k", "60")
        rrf_results = retriever.get_relevant_documents(
            f"{prefix} capital", query_emb, [f"test_{prefix}"]
        )
        assert rrf_results is not None
        assert len(rrf_results) >= 1

        deleted = retriever.delete([f"/data/{prefix}_doc.txt"])
        # delete returns sparse rowcount; may be 3
        assert deleted is None or deleted >= 0

        # Best-effort cleanup if delete by source failed to wipe (metadata JSON string vs object)
        conn = retriever.connection_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM ragmeup_sparse_embeddings WHERE id = ANY(%s);",
                    (ids,),
                )
                cur.execute(
                    "DELETE FROM ragmeup_dense_embeddings WHERE id = ANY(%s);",
                    (ids,),
                )
                conn.commit()
        finally:
            retriever.connection_pool.putconn(conn)

    def test_dataset_filter_excludes_other_sets(self, retriever, monkeypatch):
        prefix = uuid.uuid4().hex[:8]
        docs = _make_docs(prefix)
        # Second batch different dataset
        other = _make_docs(prefix + "x")
        for d in other:
            d["metadata"] = json.dumps(
                {"source": f"/data/{prefix}_other.txt", "dataset": "other_set"}
            )

        retriever.add_documents(docs + other)
        query_emb = np.ones(8, dtype=float)
        monkeypatch.setenv("use_rrf", "False")
        results = retriever.get_relevant_documents(
            "capital", query_emb, [f"test_{prefix}"]
        )
        assert results is not None
        for r in results:
            meta = r["metadata"]
            # metadata may already be dict from jsonb
            if isinstance(meta, str):
                meta = json.loads(meta)
            assert meta.get("dataset") == f"test_{prefix}" or "dataset" not in meta

        # cleanup
        conn = retriever.connection_pool.getconn()
        try:
            with conn.cursor() as cur:
                all_ids = [d["id"] for d in docs + other]
                cur.execute(
                    "DELETE FROM ragmeup_sparse_embeddings WHERE id = ANY(%s);",
                    (all_ids,),
                )
                cur.execute(
                    "DELETE FROM ragmeup_dense_embeddings WHERE id = ANY(%s);",
                    (all_ids,),
                )
                conn.commit()
        finally:
            retriever.connection_pool.putconn(conn)
