from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from omegaconf import DictConfig
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    set_seed,
)

from medical_rag_reranker.retrieval.loading import load_retriever

CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")


@dataclass
class RetrievedChunk:
    doc_id: str
    score: float
    text: str
    source: str | None = None


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "null", "None"):
        return None
    return text


def _resolve_query_text(query_obj: dict) -> str:
    text = query_obj.get("text")
    if text:
        return str(text)

    question = query_obj.get("question")
    if question:
        return str(question)

    raise ValueError("Query row must contain either `text` or `question`.")


def _resolve_query_id(query_obj: dict, fallback_idx: int) -> str:
    qid = query_obj.get("query_id")
    if qid is None:
        qid = query_obj.get("question_id")
    if qid is None:
        qid = f"query-{fallback_idx}"
    return str(qid)


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off", ""):
        return False
    raise ValueError(f"Cannot parse boolean value from: {value!r}")


def _load_docstore(corpus_path: str) -> dict[str, dict[str, Any]]:
    """Load corpus rows into a `{doc_id: row}` map used during generation."""
    path = Path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Corpus file does not exist: {path}. Set generation.corpus_path correctly."
        )

    docstore: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = row.get("doc_id")
            if not doc_id:
                continue
            docstore[str(doc_id)] = row
    return docstore


def _truncate_text(text: str, max_chars: int = 240) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3] + "..."


def _build_prompt(question: str, docs: list[RetrievedChunk]) -> str:
    if docs:
        context_lines = []
        for d in docs:
            context_lines.append(f"[{d.doc_id}] {d.text}")
        context = "\n\n".join(context_lines)
    else:
        context = "(no retrieved documents)"

    return (
        "You are a medical QA assistant.\n"
        "Rules:\n"
        "1) Answer strictly using only the provided context.\n"
        "2) If context is insufficient, say so explicitly.\n"
        "3) Cite supporting sources using [doc_id] format.\n"
        "4) Do not invent citations.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )


class LocalGenerator:
    """Thin wrapper over local Hugging Face generation models."""

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        max_input_tokens: int,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = int(max_new_tokens)
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)
        self.max_input_tokens = int(max_input_tokens)
        self.local_files_only = bool(local_files_only)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=self.local_files_only,
            )
        except Exception as e:  # pragma: no cover - environment-specific
            raise RuntimeError(
                f"Failed to load tokenizer for `{model_name}`. "
                "If this is a sentencepiece model, ensure sentencepiece is installed. "
                "If running offline, ensure the model is already cached locally or set "
                "`generation.local_files_only=false`."
            ) from e

        self.is_seq2seq = True
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                local_files_only=self.local_files_only,
            )
        except Exception:
            self.is_seq2seq = False
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                local_files_only=self.local_files_only,
            )

        if (
            self.tokenizer.pad_token_id is None
            and self.tokenizer.eos_token_id is not None
        ):
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.device)

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.do_sample:
            generate_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            output = self.model.generate(**inputs, **generate_kwargs)

        if self.is_seq2seq:
            text = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
            return text

        prompt_len = inputs["input_ids"].shape[1]
        generated_tokens = output[0][prompt_len:]
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        if not text:
            text = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return text


class RemoteOpenAICompatibleGenerator:
    """Thin client for vLLM and other OpenAI-compatible HTTP servers."""

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        api_key: str | None = None,
        api_type: str = "chat",
        timeout_seconds: float = 120.0,
    ) -> None:
        cleaned_base_url = str(base_url or "").strip().rstrip("/")
        if not cleaned_base_url:
            raise ValueError(
                "Remote generation requires `generation.remote_base_url` "
                "or `VLLM_BASE_URL`."
            )

        self.base_url = cleaned_base_url
        self.model_name = str(model_name)
        self.max_new_tokens = int(max_new_tokens)
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)
        self.api_key = _as_optional_str(api_key)
        self.api_type = str(api_type or "chat").strip().lower()
        self.timeout_seconds = float(timeout_seconds)

    def _request(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(
            f"{self.base_url}{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=float(self.timeout_seconds)
            ) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Remote generation request failed: {exc.code}. {body}"
            ) from exc
        return json.loads(body.decode("utf-8"))

    def generate(self, prompt: str) -> str:
        temperature = self.temperature if self.do_sample else 0.0
        if self.api_type in ("completion", "completions"):
            response = self._request(
                "/completions",
                {
                    "model": self.model_name,
                    "prompt": prompt,
                    "max_tokens": self.max_new_tokens,
                    "temperature": temperature,
                },
            )
            choices = response.get("choices") or []
            if not choices:
                return ""
            return str(choices[0].get("text") or "").strip()

        response = self._request(
            "/chat/completions",
            {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_new_tokens,
                "temperature": temperature,
            },
        )
        choices = response.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return str(message.get("content") or "").strip()


def _build_generator_from_cfg(cfg: DictConfig):
    backend = str(getattr(cfg.generation, "backend", "local")).strip().lower()
    model_name = str(cfg.generation.llm_model_name)
    if backend in ("local", "transformers"):
        return LocalGenerator(
            model_name=model_name,
            max_new_tokens=int(cfg.generation.max_new_tokens),
            do_sample=bool(cfg.generation.do_sample),
            temperature=float(cfg.generation.temperature),
            max_input_tokens=int(getattr(cfg.generation, "max_input_tokens", 1024)),
            local_files_only=_as_bool(
                getattr(cfg.generation, "local_files_only", False)
            ),
        )

    if backend in ("openai_compatible", "vllm", "remote"):
        return RemoteOpenAICompatibleGenerator(
            base_url=str(getattr(cfg.generation, "remote_base_url", "") or ""),
            model_name=model_name,
            max_new_tokens=int(cfg.generation.max_new_tokens),
            do_sample=bool(cfg.generation.do_sample),
            temperature=float(cfg.generation.temperature),
            api_key=_as_optional_str(getattr(cfg.generation, "remote_api_key", None)),
            api_type=str(getattr(cfg.generation, "remote_api_type", "chat")),
            timeout_seconds=float(
                getattr(cfg.generation, "remote_timeout_seconds", 120.0)
            ),
        )

    raise ValueError(
        f"Unsupported generation backend: {backend!r}. "
        "Use `local` or `openai_compatible`."
    )


def _retrieve_docs(
    retriever,
    docstore: dict[str, dict[str, Any]],
    question: str,
    top_k: int,
) -> list[RetrievedChunk]:
    """Resolve retrieved doc ids into full text chunks from the docstore."""
    hits = retriever.retrieve(question, top_k=int(top_k))
    rows: list[RetrievedChunk] = []
    retriever_payloads = getattr(retriever, "last_payloads", {}) or {}
    for h in hits:
        doc = docstore.get(h.doc_id, retriever_payloads.get(h.doc_id, {}))
        rows.append(
            RetrievedChunk(
                doc_id=h.doc_id,
                score=float(h.score),
                text=str(doc.get("text", "")),
                source=str(doc["source"]) if "source" in doc else None,
            )
        )
    return rows


def _detect_citations(answer: str) -> list[str]:
    found = CITATION_PATTERN.findall(answer)
    unique: list[str] = []
    seen: set[str] = set()
    for item in found:
        key = item.strip()
        if key and key not in seen:
            unique.append(key)
            seen.add(key)
    return unique


def _partition_citations(
    citations: list[str],
    retrieved_doc_ids: list[str],
) -> tuple[list[str], list[str]]:
    retrieved_set = set(retrieved_doc_ids)
    supported: list[str] = []
    unsupported: list[str] = []
    for citation in citations:
        if citation in retrieved_set:
            supported.append(citation)
        else:
            unsupported.append(citation)
    return supported, unsupported


def _serialize_doc(
    *,
    doc_id: str,
    score: float,
    text: str,
    source: str | None = None,
    retrieval_score: float | None = None,
    reranker_score: float | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "doc_id": doc_id,
        "score": float(score),
        "text": text,
        "source": source,
    }
    if retrieval_score is not None:
        row["retrieval_score"] = float(retrieval_score)
    if reranker_score is not None:
        row["reranker_score"] = float(reranker_score)
    return row


def _run_one_question(
    llm,
    retriever,
    docstore: dict[str, dict[str, Any]],
    question: str,
    top_k: int,
    retrieve_top_k: int,
    reranker=None,
    query_id: str | None = None,
) -> dict[str, Any]:
    """Run retrieval, optional reranking, and generation for one question."""
    total_start = perf_counter()
    retrieval_start = perf_counter()
    retrieved_candidates = _retrieve_docs(
        retriever=retriever,
        docstore=docstore,
        question=question,
        top_k=retrieve_top_k if reranker is not None else top_k,
    )
    retrieval_latency_ms = (perf_counter() - retrieval_start) * 1000.0
    retrieved = retrieved_candidates
    rerank_latency_ms = 0.0
    retrieved_before_rerank: list[dict[str, Any]] | None = None

    if reranker is not None and retrieved_candidates:
        from medical_rag_reranker.inference.rerank import CandidateDoc

        retrieved_before_rerank = [
            _serialize_doc(
                doc_id=d.doc_id,
                score=d.score,
                retrieval_score=d.score,
                text=d.text,
                source=d.source,
            )
            for d in retrieved_candidates
        ]
        reranked_docs, rerank_latency_ms = reranker.rerank(
            question=question,
            candidates=[
                CandidateDoc(
                    doc_id=d.doc_id,
                    text=d.text,
                    retrieval_score=d.score,
                    source=d.source,
                )
                for d in retrieved_candidates
            ],
            top_k=top_k,
        )
        retrieved = [
            RetrievedChunk(
                doc_id=d.doc_id,
                score=d.reranker_score,
                text=d.text,
                source=d.source,
            )
            for d in reranked_docs
        ]

    prompt = _build_prompt(question=question, docs=retrieved)
    generation_start = perf_counter()
    answer = llm.generate(prompt)
    generation_latency_ms = (perf_counter() - generation_start) * 1000.0
    citations_detected = _detect_citations(answer)
    supported_citations, unsupported_citations = _partition_citations(
        citations=citations_detected,
        retrieved_doc_ids=[doc.doc_id for doc in retrieved],
    )

    out: dict[str, Any] = {
        "question": question,
        "retrieved": [
            _serialize_doc(
                doc_id=d.doc_id,
                score=d.score,
                text=d.text,
                source=d.source,
            )
            for d in retrieved
        ],
        "answer": answer,
        "citations_detected": citations_detected,
        "supported_citations_detected": supported_citations,
        "unsupported_citations_detected": unsupported_citations,
        "reranker_enabled": reranker is not None,
        "retrieval_latency_ms": float(retrieval_latency_ms),
        "generation_latency_ms": float(generation_latency_ms),
        "end_to_end_latency_ms": float((perf_counter() - total_start) * 1000.0),
    }
    if retrieved_before_rerank is not None:
        retrieval_scores_by_doc_id = {
            str(doc["doc_id"]): float(doc["retrieval_score"])
            for doc in retrieved_before_rerank
        }
        out["retrieved_before_rerank"] = retrieved_before_rerank
        out["retrieved"] = [
            _serialize_doc(
                doc_id=d.doc_id,
                score=d.score,
                retrieval_score=retrieval_scores_by_doc_id.get(d.doc_id, d.score),
                reranker_score=d.score,
                text=d.text,
                source=d.source,
            )
            for d in retrieved
        ]
        out["rerank_latency_ms"] = float(rerank_latency_ms)

    if query_id is not None:
        out["query_id"] = query_id
    return out


def _write_examples_report(
    report_path: Path,
    results: list[dict[str, Any]],
    retriever_name: str,
    llm_model_name: str,
    top_k: int,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    reranker_enabled = any(bool(item.get("reranker_enabled")) for item in results)
    title = (
        "# RAG Generation Examples (With Reranker)"
        if reranker_enabled
        else "# Baseline Generation Examples (No Reranker)"
    )
    lines.append(title)
    lines.append("")
    lines.append(f"- retriever: `{retriever_name}`")
    lines.append(f"- llm_model: `{llm_model_name}`")
    lines.append(f"- top_k: `{top_k}`")
    lines.append(f"- reranker_enabled: `{reranker_enabled}`")
    lines.append(f"- num_examples: `{len(results)}`")
    lines.append("")

    for i, item in enumerate(results, start=1):
        qid = item.get("query_id", f"query-{i}")
        lines.append(f"## Example {i} (`{qid}`)")
        lines.append("")
        lines.append(f"**Question**: {item['question']}")
        lines.append("")
        before_rerank = item.get("retrieved_before_rerank", [])
        if before_rerank:
            lines.append("**Top docs before rerank**:")
            for rank, doc in enumerate(before_rerank, start=1):
                lines.append(
                    f"{rank}. `{doc['doc_id']}` "
                    f"(score={doc['score']:.4f}) "
                    f"- {_truncate_text(str(doc.get('text', '')))}"
                )
            lines.append("")

        header = "**Top docs after rerank**:" if before_rerank else "**Top docs**:"
        lines.append(header)
        for rank, doc in enumerate(item.get("retrieved", []), start=1):
            lines.append(
                f"{rank}. `{doc['doc_id']}` "
                f"(score={doc['score']:.4f}) "
                f"- {_truncate_text(str(doc.get('text', '')))}"
            )
            if "retrieval_score" in doc and "reranker_score" in doc:
                lines.append(
                    f"   retrieval_score={doc['retrieval_score']:.4f}, "
                    f"reranker_score={doc['reranker_score']:.4f}"
                )
        lines.append("")
        lines.append("**Answer**:")
        lines.append("")
        lines.append(str(item["answer"]))
        lines.append("")
        cited = ", ".join(f"`{c}`" for c in item.get("citations_detected", []))
        lines.append(f"**Citations detected**: {cited if cited else '_none_'}")
        supported = ", ".join(
            f"`{c}`" for c in item.get("supported_citations_detected", [])
        )
        unsupported = ", ".join(
            f"`{c}`" for c in item.get("unsupported_citations_detected", [])
        )
        lines.append(f"**Supported citations**: {supported if supported else '_none_'}")
        lines.append(
            f"**Unsupported citations**: {unsupported if unsupported else '_none_'}"
        )
        if "rerank_latency_ms" in item:
            lines.append(f"**Rerank latency (ms)**: {item['rerank_latency_ms']:.2f}")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_examples_report(
    report_path: Path,
    results: list[dict[str, Any]],
    retriever_name: str,
    llm_model_name: str,
    top_k: int,
) -> None:
    """Persist a human-readable markdown report for generated examples."""
    _write_examples_report(
        report_path=report_path,
        results=results,
        retriever_name=retriever_name,
        llm_model_name=llm_model_name,
        top_k=top_k,
    )


def write_results_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Persist raw generation/evaluation rows as JSONL."""
    _write_jsonl(path=path, rows=rows)


def _load_queries(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Queries file does not exist: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def generate_from_cfg(
    cfg: DictConfig,
    question: str | None,
    queries_path: str | None,
    output_path: str | None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Generate answers using retriever context and a local transformers LLM."""
    if not bool(getattr(cfg.generation, "enabled", True)):
        raise RuntimeError(
            "Generation is disabled in config (`generation.enabled=false`)."
        )

    generation_seed = int(getattr(cfg.generation, "seed", 42))
    set_seed(generation_seed)

    retriever = load_retriever(
        retriever_name=str(cfg.retrieval.name),
        index_path=str(cfg.generation.index),
        retrieval_cfg=cfg.retrieval,
    )
    corpus_path = _as_optional_str(getattr(cfg.generation, "corpus_path", None))
    docstore = _load_docstore(corpus_path) if corpus_path is not None else {}
    use_reranker = _as_bool(getattr(cfg.generation, "use_reranker", False))
    retrieve_top_k = int(
        getattr(cfg.generation, "retrieve_top_k", cfg.generation.top_k)
    )
    retrieve_top_k = max(retrieve_top_k, int(cfg.generation.top_k))

    reranker = None
    if use_reranker:
        from medical_rag_reranker.inference.rerank import CrossEncoderBatchReranker

        checkpoint_path = _as_optional_str(
            getattr(cfg.generation, "reranker_checkpoint_path", None)
        )
        if checkpoint_path is None:
            raise RuntimeError(
                "generation.use_reranker=true requires "
                "`generation.reranker_checkpoint_path`."
            )

        reranker = CrossEncoderBatchReranker.from_cfg(
            cfg=cfg,
            checkpoint_path=checkpoint_path,
            batch_size=int(getattr(cfg.generation, "reranker_batch_size", 16)),
        )

    llm = _build_generator_from_cfg(cfg)

    if question:
        return _run_one_question(
            llm=llm,
            retriever=retriever,
            docstore=docstore,
            question=question,
            top_k=int(cfg.generation.top_k),
            retrieve_top_k=retrieve_top_k,
            reranker=reranker,
            query_id=None,
        )

    mode = str(getattr(cfg.generation, "mode", "single")).lower()
    if mode == "single":
        raise ValueError(
            "Single mode requires `question` argument. "
            "Pass --question or set generation.mode=batch."
        )

    effective_queries = _as_optional_str(queries_path) or str(
        cfg.generation.queries_path
    )
    queries_rows = _load_queries(
        path=Path(effective_queries),
        limit=int(getattr(cfg.generation, "examples_limit", 20)),
    )

    results: list[dict[str, Any]] = []
    for idx, row in enumerate(queries_rows, start=1):
        qid = _resolve_query_id(row, fallback_idx=idx)
        qtext = _resolve_query_text(row)
        results.append(
            _run_one_question(
                llm=llm,
                retriever=retriever,
                docstore=docstore,
                question=qtext,
                top_k=int(cfg.generation.top_k),
                retrieve_top_k=retrieve_top_k,
                reranker=reranker,
                query_id=qid,
            )
        )

    report_file = Path(_as_optional_str(output_path) or str(cfg.generation.report_path))
    write_examples_report(
        report_path=report_file,
        results=results,
        retriever_name=str(cfg.retrieval.name),
        llm_model_name=str(cfg.generation.llm_model_name),
        top_k=int(cfg.generation.top_k),
    )

    results_jsonl = _as_optional_str(
        getattr(cfg.generation, "results_jsonl_path", None)
    )
    if results_jsonl:
        write_results_jsonl(Path(results_jsonl), results)

    return results
