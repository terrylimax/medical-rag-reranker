from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import numpy as np

from medical_rag_reranker.utils.progress import count_text_lines, progress

from . import Retriever, ScoredDoc


QDRANT_FORMAT = "medical-rag-reranker.qdrant-index"


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "None", "null"):
        return None
    return text


def _point_id(doc_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"medical-rag-reranker:{doc_id}"))


def _json_safe_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in row.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            payload[str(key)] = value
        elif isinstance(value, (list, tuple, set)):
            payload[str(key)] = [str(item) for item in value if item is not None]
        else:
            payload[str(key)] = str(value)
    return payload


@dataclass
class QdrantRetriever(Retriever):
    """Remote Qdrant-backed dense retriever.

    Embeddings are computed locally with SentenceTransformer and stored remotely
    in Qdrant. The saved index is a lightweight manifest, not a local vector dump.
    """

    url: str = "http://localhost:6333"
    collection_name: str = "medical_rag_docs"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    max_seq_length: int | None = None
    api_key: str | None = None
    api_key_env: str = "QDRANT_API_KEY"
    timeout_seconds: float = 60.0
    vector_size: int | None = None
    model: Any | None = None
    last_payloads: dict[str, dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        self.url = str(self.url).rstrip("/")
        self.collection_name = str(self.collection_name)
        self.api_key = _as_optional_str(self.api_key) or _as_optional_str(
            os.getenv(str(self.api_key_env or "QDRANT_API_KEY"))
        )
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as e:
                raise RuntimeError(
                    "Qdrant retriever requires `sentence-transformers` to encode "
                    "documents and queries."
                ) from e
            self.model = SentenceTransformer(self.model_name)
        if self.max_seq_length is not None:
            self.model.max_seq_length = int(self.max_seq_length)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["api-key"] = self.api_key
        return headers

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.url}{path}",
            data=data,
            headers=self._headers(),
            method=method,
        )
        try:
            with urllib.request.urlopen(
                request, timeout=float(self.timeout_seconds)
            ) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Qdrant request failed: {method} {path} -> {exc.code}. {body}"
            ) from exc
        if not body:
            return {}
        return json.loads(body.decode("utf-8"))

    def _collection_exists(self) -> bool:
        request = urllib.request.Request(
            f"{self.url}/collections/{self.collection_name}",
            headers=self._headers(),
            method="GET",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=float(self.timeout_seconds)
            ) as response:
                response.read()
            return True
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return False
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Qdrant collection check failed: {exc.code}. {body}"
            ) from exc

    def _ensure_collection(self, vector_size: int) -> None:
        if self._collection_exists():
            return
        self._request_json(
            "PUT",
            f"/collections/{self.collection_name}",
            {
                "vectors": {
                    "size": int(vector_size),
                    "distance": "Cosine",
                }
            },
        )

    def index(self, corpus_path: str) -> None:
        rows: list[dict[str, Any]] = []
        texts: list[str] = []
        total = count_text_lines(corpus_path)
        with open(corpus_path, "r", encoding="utf-8") as f:
            stream = progress(
                f,
                desc="Reading Qdrant index corpus",
                total=total,
                unit="doc",
            )
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                doc_id = str(row.get("doc_id") or "").strip()
                text = str(row.get("text") or "").strip()
                if not doc_id or not text:
                    continue
                rows.append(row)
                texts.append(text)

        if not rows:
            raise ValueError("Cannot build Qdrant index from an empty corpus.")

        embeddings = self.model.encode(
            texts,
            batch_size=int(self.batch_size),
            show_progress_bar=True,
        )
        vectors = _l2_normalize(np.asarray(embeddings, dtype=np.float32))
        self.vector_size = int(vectors.shape[1])
        self._ensure_collection(self.vector_size)

        batch_size = max(1, int(self.batch_size))
        starts = range(0, len(rows), batch_size)
        total_batches = (len(rows) + batch_size - 1) // batch_size
        for start in progress(
            starts,
            desc="Uploading Qdrant points",
            total=total_batches,
            unit="batch",
        ):
            points: list[dict[str, Any]] = []
            for row, vector in zip(
                rows[start : start + batch_size],
                vectors[start : start + batch_size],
            ):
                doc_id = str(row["doc_id"])
                payload = _json_safe_payload(row)
                payload["doc_id"] = doc_id
                points.append(
                    {
                        "id": _point_id(doc_id),
                        "vector": vector.astype(float).tolist(),
                        "payload": payload,
                    }
                )
            self._request_json(
                "PUT",
                f"/collections/{self.collection_name}/points?wait=true",
                {"points": points},
            )

    def retrieve(self, query: str, top_k: int) -> List[ScoredDoc]:
        if int(top_k) <= 0:
            return []
        q = self.model.encode([query])
        q = _l2_normalize(np.asarray(q, dtype=np.float32))[0].astype(float).tolist()

        payload = {
            "query": q,
            "limit": int(top_k),
            "with_payload": True,
            "with_vector": False,
        }
        try:
            response = self._request_json(
                "POST",
                f"/collections/{self.collection_name}/points/query",
                payload,
            )
            points_obj = response.get("result", {})
            points = (
                points_obj.get("points", [])
                if isinstance(points_obj, dict)
                else points_obj
            )
        except RuntimeError:
            response = self._request_json(
                "POST",
                f"/collections/{self.collection_name}/points/search",
                {
                    "vector": q,
                    "limit": int(top_k),
                    "with_payload": True,
                    "with_vector": False,
                },
            )
            points = response.get("result", [])

        hits: list[ScoredDoc] = []
        self.last_payloads = {}
        for point in points:
            payload_obj = point.get("payload") or {}
            doc_id = str(payload_obj.get("doc_id") or point.get("id"))
            self.last_payloads[doc_id] = dict(payload_obj)
            hits.append(ScoredDoc(doc_id=doc_id, score=float(point.get("score", 0.0))))
        return hits

    def fetch_payloads(self, doc_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not doc_ids:
            return {}
        response = self._request_json(
            "POST",
            f"/collections/{self.collection_name}/points",
            {
                "ids": [_point_id(str(doc_id)) for doc_id in doc_ids],
                "with_payload": True,
                "with_vector": False,
            },
        )
        payloads: dict[str, dict[str, Any]] = {}
        for point in response.get("result", []):
            payload_obj = point.get("payload") or {}
            doc_id = str(payload_obj.get("doc_id") or point.get("id"))
            payloads[doc_id] = dict(payload_obj)
        return payloads

    def save(self, path: str) -> None:
        manifest = {
            "format": QDRANT_FORMAT,
            "version": 1,
            "url": self.url,
            "collection_name": self.collection_name,
            "model_name": self.model_name,
            "batch_size": int(self.batch_size),
            "max_seq_length": self.max_seq_length,
            "api_key_env": str(self.api_key_env or "QDRANT_API_KEY"),
            "timeout_seconds": float(self.timeout_seconds),
            "vector_size": self.vector_size,
        }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str):
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        if obj.get("format") != QDRANT_FORMAT:
            raise ValueError("Unsupported Qdrant index manifest format.")
        api_key_env = str(obj.get("api_key_env") or "QDRANT_API_KEY")
        return cls(
            url=str(
                os.getenv("QDRANT_URL") or obj.get("url") or "http://localhost:6333"
            ),
            collection_name=str(
                os.getenv("QDRANT_COLLECTION")
                or obj.get("collection_name")
                or "medical_rag_docs"
            ),
            model_name=str(obj.get("model_name")),
            batch_size=int(obj.get("batch_size", 64)),
            max_seq_length=(
                int(obj["max_seq_length"])
                if obj.get("max_seq_length") is not None
                else None
            ),
            api_key=os.getenv(api_key_env),
            api_key_env=api_key_env,
            timeout_seconds=float(obj.get("timeout_seconds", 60.0)),
            vector_size=(
                int(obj["vector_size"]) if obj.get("vector_size") is not None else None
            ),
        )
