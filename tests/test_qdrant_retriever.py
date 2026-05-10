import json

import numpy as np

from medical_rag_reranker.retrieval.qdrant import QdrantRetriever


class FakeModel:
    def encode(self, texts, **kwargs):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


def test_qdrant_retriever_parses_query_points() -> None:
    retriever = QdrantRetriever(
        url="https://qdrant.test",
        collection_name="medical",
        model_name="fake",
        model=FakeModel(),
    )
    captured = {}

    def fake_request(method, path, payload=None):
        captured["method"] = method
        captured["path"] = path
        captured["payload"] = payload
        return {
            "result": {
                "points": [
                    {
                        "score": 0.95,
                        "payload": {
                            "doc_id": "doc-1",
                            "text": "remote chunk",
                            "source": "QDRANT",
                        },
                    }
                ]
            }
        }

    retriever._request_json = fake_request

    hits = retriever.retrieve("question", top_k=1)

    assert hits[0].doc_id == "doc-1"
    assert hits[0].score == 0.95
    assert retriever.last_payloads["doc-1"]["text"] == "remote chunk"
    assert captured["method"] == "POST"
    assert captured["path"] == "/collections/medical/points/query"
    assert captured["payload"]["with_payload"] is True


def test_qdrant_manifest_does_not_persist_api_key(tmp_path) -> None:
    retriever = QdrantRetriever(
        url="https://qdrant.test",
        collection_name="medical",
        model_name="fake",
        api_key="secret",
        model=FakeModel(),
    )
    manifest_path = tmp_path / "qdrant_index.json"

    retriever.save(str(manifest_path))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["collection_name"] == "medical"
    assert manifest["api_key_env"] == "QDRANT_API_KEY"
    assert "api_key" not in manifest


def test_qdrant_fetch_payloads_by_doc_ids() -> None:
    retriever = QdrantRetriever(
        url="https://qdrant.test",
        collection_name="medical",
        model_name="fake",
        model=FakeModel(),
    )
    captured = {}

    def fake_request(method, path, payload=None):
        captured["method"] = method
        captured["path"] = path
        captured["payload"] = payload
        return {
            "result": [
                {
                    "payload": {
                        "doc_id": "doc-1",
                        "text": "remote chunk",
                    }
                }
            ]
        }

    retriever._request_json = fake_request

    payloads = retriever.fetch_payloads(["doc-1"])

    assert payloads["doc-1"]["text"] == "remote chunk"
    assert captured["method"] == "POST"
    assert captured["path"] == "/collections/medical/points"
    assert captured["payload"]["with_payload"] is True
