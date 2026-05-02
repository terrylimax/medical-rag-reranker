# RRF Retrieval Comparison

Evaluation split: `data/processed/eval_queries.jsonl`, `300` queries.

Important: the table below is based only on saved metric artifacts from actual
runs under `runs/*.trec.metrics.json`. Older manually written benchmark reports
may contain stale numbers and should not be treated as the source of truth.

## What Was Evaluated

There are two different RRF use cases in the project:

1. Hybrid RRF: merge rankings from two retrievers for the same query.

   - `hybrid`: BM25 + Dense MiniLM.
   - `hybrid_medcpt_pilot`: BM25 + MedCPT pilot bi-encoder.

2. Query-side RAG Fusion RRF: build several query variants, retrieve for each
   variant, then merge the rankings with RRF.
   - `rag_fusion_bm25`: query fusion over BM25.
   - `rag_fusion_dense`: query fusion over Dense MiniLM.
   - `rag_fusion_medcpt_pilot`: query fusion over MedCPT pilot.

## Metrics From Run Artifacts

| Method                           | Source                                           |   R@10 |   R@20 |   R@50 | MRR@10 | NDCG@10 |  Hit@1 | p50 ms | p95 ms |
| -------------------------------- | ------------------------------------------------ | -----: | -----: | -----: | -----: | ------: | -----: | -----: | -----: |
| BM25 baseline                    | `runs/bm25.trec.metrics.json`                    | 0.7133 | 0.7600 | 0.8067 | 0.3972 |  0.4731 | 0.2567 |   37.9 |   56.3 |
| Dense MiniLM baseline            | `runs/dense.trec.metrics.json`                   | 0.8400 | 0.8567 | 0.8833 | 0.5895 |  0.6505 | 0.4533 |   13.2 |   17.0 |
| MedCPT pilot baseline            | `runs/medcpt_pilot.trec.metrics.json`            | 0.8167 | 0.8400 | 0.8767 | 0.5522 |  0.6159 | 0.4300 |   67.0 |   71.4 |
| Hybrid RRF: BM25 + Dense MiniLM  | `runs/hybrid.trec.metrics.json`                  | 0.8133 | 0.8533 | 0.8900 | 0.5364 |  0.6038 | 0.3967 |   51.2 |   70.1 |
| Hybrid RRF: BM25 + MedCPT pilot  | `runs/hybrid_medcpt_pilot.trec.metrics.json`     | 0.8200 | 0.8667 | 0.8967 | 0.6080 |  0.6597 | 0.4967 |  113.1 |  136.8 |
| RAG Fusion RRF over BM25         | `runs/rag_fusion_bm25.trec.metrics.json`         | 0.7233 | 0.7867 | 0.8300 | 0.3884 |  0.4685 | 0.2500 |  157.0 |  284.3 |
| RAG Fusion RRF over Dense MiniLM | `runs/rag_fusion_dense.trec.metrics.json`        | 0.8233 | 0.8700 | 0.8933 | 0.5169 |  0.5910 | 0.3633 |  105.6 |  155.4 |
| RAG Fusion RRF over MedCPT pilot | `runs/rag_fusion_medcpt_pilot.trec.metrics.json` | 0.7033 | 0.7767 | 0.8767 | 0.4606 |  0.5183 | 0.3600 |  352.7 |  590.0 |

## Interpretation

Hybrid RRF is useful here. The best current retrieval setup is
`hybrid_medcpt_pilot`: it gives the strongest early ranking and candidate recall
among the tested methods.

Query-side RAG Fusion RRF is not a good production default in the current
implementation. It can improve deeper recall for BM25/Dense, but it worsens
early ranking quality and adds substantial latency. It is better suited as a
fallback mode when the primary retriever has low confidence or returns too few
usable candidates.

## Saved Artifacts

- `runs/hybrid.trec.metrics.json`
- `runs/hybrid_medcpt_pilot.trec.metrics.json`
- `runs/rag_fusion_bm25.trec.metrics.json`
- `runs/rag_fusion_dense.trec.metrics.json`
- `runs/rag_fusion_medcpt_pilot.trec.metrics.json`
- `runs/rag_fusion_bm25.trec.queries.jsonl`
- `runs/rag_fusion_dense.trec.queries.jsonl`
- `runs/rag_fusion_medcpt_pilot.trec.queries.jsonl`
