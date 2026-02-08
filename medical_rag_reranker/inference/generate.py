from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    set_seed,
)

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


def _resolve_manifest_path(index_arg: str) -> Path:
    p = Path(index_arg)
    if p.exists() and p.is_dir():
        candidate = p / "hybrid_index.json"
        if not candidate.exists():
            raise FileNotFoundError(f"Expected hybrid_index.json inside directory: {p}")
        return candidate
    return p


def _load_hybrid_from_manifest(manifest_path: Path):
    from medical_rag_reranker.retrieval.bm25 import BM25Retriever
    from medical_rag_reranker.retrieval.dense import DenseRetriever
    from medical_rag_reranker.retrieval.hybrid import HybridRetriever

    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    if m.get("format") != "medical-rag-reranker.hybrid-index":
        raise ValueError("Unsupported hybrid manifest format.")

    base_dir = manifest_path.parent
    bm25_index = Path(m["bm25_index"])
    dense_index = Path(m["dense_index"])

    if not bm25_index.is_absolute():
        bm25_index = base_dir / bm25_index
    if not dense_index.is_absolute():
        dense_index = base_dir / dense_index

    bm25 = BM25Retriever.load(str(bm25_index))
    dense = DenseRetriever.load(str(dense_index))

    return HybridRetriever(
        bm25=bm25,
        dense=dense,
        alpha=float(m.get("alpha", 0.5)),
        cand_k=int(m.get("cand_k", 50)),
    )


def _load_retriever(retriever_name: str, index_path: str):
    if retriever_name == "bm25":
        from medical_rag_reranker.retrieval.bm25 import BM25Retriever

        return BM25Retriever.load(index_path)
    if retriever_name == "dense":
        try:
            from medical_rag_reranker.retrieval.dense import DenseRetriever
        except Exception as e:
            raise RuntimeError(
                "Dense retriever dependencies are missing. "
                "Install `sentence-transformers` to use retrieval=dense."
            ) from e
        return DenseRetriever.load(index_path)
    if retriever_name == "hybrid":
        manifest_path = _resolve_manifest_path(index_path)
        return _load_hybrid_from_manifest(manifest_path)
    raise ValueError(f"Unsupported retriever: {retriever_name}")


def _load_docstore(corpus_path: str) -> dict[str, dict[str, Any]]:
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
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        max_input_tokens: int,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = int(max_new_tokens)
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)
        self.max_input_tokens = int(max_input_tokens)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:  # pragma: no cover - environment-specific
            raise RuntimeError(
                f"Failed to load tokenizer for `{model_name}`. "
                "If this is a sentencepiece model, ensure sentencepiece is installed."
            ) from e

        self.is_seq2seq = True
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except Exception:
            self.is_seq2seq = False
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

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


def _retrieve_docs(
    retriever,
    docstore: dict[str, dict[str, Any]],
    question: str,
    top_k: int,
) -> list[RetrievedChunk]:
    hits = retriever.retrieve(question, top_k=int(top_k))
    rows: list[RetrievedChunk] = []
    for h in hits:
        doc = docstore.get(h.doc_id, {})
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


def _run_one_question(
    llm: LocalGenerator,
    retriever,
    docstore: dict[str, dict[str, Any]],
    question: str,
    top_k: int,
    query_id: str | None = None,
) -> dict[str, Any]:
    retrieved = _retrieve_docs(
        retriever=retriever,
        docstore=docstore,
        question=question,
        top_k=top_k,
    )
    prompt = _build_prompt(question=question, docs=retrieved)
    answer = llm.generate(prompt)
    citations_detected = _detect_citations(answer)

    out: dict[str, Any] = {
        "question": question,
        "retrieved": [
            {
                "doc_id": d.doc_id,
                "score": d.score,
                "text": d.text,
                "source": d.source,
            }
            for d in retrieved
        ],
        "answer": answer,
        "citations_detected": citations_detected,
    }
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
    lines.append("# Baseline Generation Examples (No Reranker)")
    lines.append("")
    lines.append(f"- retriever: `{retriever_name}`")
    lines.append(f"- llm_model: `{llm_model_name}`")
    lines.append(f"- top_k: `{top_k}`")
    lines.append(f"- num_examples: `{len(results)}`")
    lines.append("")

    for i, item in enumerate(results, start=1):
        qid = item.get("query_id", f"query-{i}")
        lines.append(f"## Example {i} (`{qid}`)")
        lines.append("")
        lines.append(f"**Question**: {item['question']}")
        lines.append("")
        lines.append("**Top docs**:")
        for rank, doc in enumerate(item.get("retrieved", []), start=1):
            lines.append(
                f"{rank}. `{doc['doc_id']}` "
                f"(score={doc['score']:.4f}) "
                f"- {_truncate_text(str(doc.get('text', '')))}"
            )
        lines.append("")
        lines.append("**Answer**:")
        lines.append("")
        lines.append(str(item["answer"]))
        lines.append("")
        cited = ", ".join(f"`{c}`" for c in item.get("citations_detected", []))
        lines.append(f"**Citations detected**: {cited if cited else '_none_'}")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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

    retriever = _load_retriever(
        retriever_name=str(cfg.retrieval.name),
        index_path=str(cfg.generation.index),
    )
    docstore = _load_docstore(str(cfg.generation.corpus_path))

    llm = LocalGenerator(
        model_name=str(cfg.generation.llm_model_name),
        max_new_tokens=int(cfg.generation.max_new_tokens),
        do_sample=bool(cfg.generation.do_sample),
        temperature=float(cfg.generation.temperature),
        max_input_tokens=int(getattr(cfg.generation, "max_input_tokens", 1024)),
    )

    if question:
        return _run_one_question(
            llm=llm,
            retriever=retriever,
            docstore=docstore,
            question=question,
            top_k=int(cfg.generation.top_k),
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
                query_id=qid,
            )
        )

    report_file = Path(_as_optional_str(output_path) or str(cfg.generation.report_path))
    _write_examples_report(
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
        _write_jsonl(Path(results_jsonl), results)

    return results
