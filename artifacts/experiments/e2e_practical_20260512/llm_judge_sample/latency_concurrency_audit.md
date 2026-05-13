# Latency Concurrency Audit

- source: `e2e_summary_with_llm_judge_n50.csv`
- raw latency columns are recomputed from `generation/*.raw.jsonl`.
- throughput-normalized latency divides raw latency by inferred remote concurrency.

- runs with `generation_remote_concurrency=4` metadata: `25`
- runs without metadata, assumed `1`: `7`

## Runs Without Concurrency Metadata

- `dense_qdrant_8026f7536bfe_top5_ret20_reranktrue_heuristic`
- `hybrid_qdrant_6c41cc2303a0_top5_ret20_rerankfalse_heuristic`
- `hybrid_qdrant_cdeefb6d16f8_top5_ret20_rerankfalse_heuristic`
- `bm25_779137bf3e9c_top5_ret20_reranktrue_heuristic`
- `bm25_779137bf3e9c_top5_ret20_rerankfalse_heuristic`
- `hybrid_qdrant_cdeefb6d16f8_top5_ret20_reranktrue_heuristic`
- `dense_qdrant_8026f7536bfe_top5_ret20_rerankfalse_heuristic`

## Top By Raw E2E p50

| Method                        | Rerank | Concurrency | Raw E2E p50 ms | Norm E2E p50 ms | LLM n50 Pass |
| ----------------------------- | -----: | ----------: | -------------: | --------------: | -----------: |
| `bm25`                        |  false |         1.0 |           2501 |            2501 |         0.70 |
| `bm25`                        |   true |         1.0 |           5395 |            5395 |         0.68 |
| `dense_qdrant`                |  false |         1.0 |           5505 |            5505 |         0.66 |
| `graph_bm25`                  |  false |         4.0 |           5619 |            1405 |         0.72 |
| `rag_fusion_bm25`             |  false |         4.0 |           5785 |            1446 |         0.76 |
| `medcpt_zero_shot`            |  false |         4.0 |           5849 |            1462 |         0.66 |
| `rag_fusion_bm25`             |  false |         4.0 |           5850 |            1463 |         0.74 |
| `hybrid_qdrant`               |  false |         1.0 |           6156 |            6156 |         0.72 |
| `medcpt_trained`              |  false |         4.0 |           6169 |            1542 |         0.62 |
| `graph_qdrant`                |  false |         4.0 |           6176 |            1544 |         0.68 |
| `graph_hybrid_qdrant`         |  false |         4.0 |           6274 |            1568 |         0.68 |
| `hybrid_medcpt_trained`       |  false |         4.0 |           6408 |            1602 |         0.62 |
| `hybrid_medcpt_trained`       |  false |         4.0 |           6485 |            1621 |         0.64 |
| `hybrid_qdrant`               |  false |         1.0 |           6495 |            6495 |         0.70 |
| `graph_hybrid_medcpt_trained` |  false |         4.0 |           6668 |            1667 |         0.64 |
