# E2E Summary With LLM-as-a-Judge n50

- source e2e: `artifacts/experiments/e2e_practical_20260512/e2e_summary.csv`
- source judge: `artifacts/experiments/e2e_practical_20260512/llm_judge_sample/llm_judge_sample_n50_summary.csv`
- join key: `generation_run_name == run_name`

## Top Runs

| Rank | Method                        | Rerank | LLM Pass | Faith |  Rel | Comp | Hit@5 | MRR@10 | Gen p50 ms | E2E p50 ms |
| ---: | ----------------------------- | -----: | -------: | ----: | ---: | ---: | ----: | -----: | ---------: | ---------: |
|    1 | `rag_fusion_qdrant`           |  false |     0.86 |  4.64 | 4.90 | 4.28 | 0.570 |  0.415 |       5779 |       7040 |
|    2 | `medcpt_zero_shot`            |   true |     0.80 |  4.32 | 4.88 | 4.26 | 0.513 |  0.329 |       5290 |       7496 |
|    3 | `rag_fusion_qdrant`           |  false |     0.78 |  4.38 | 4.98 | 4.20 | 0.607 |  0.469 |       5970 |       6702 |
|    4 | `rag_fusion_bm25`             |  false |     0.76 |  4.34 | 4.88 | 4.10 | 0.590 |  0.388 |       5500 |       5785 |
|    5 | `rag_fusion_bm25`             |   true |     0.76 |  4.30 | 4.94 | 4.18 | 0.587 |  0.387 |       5387 |       8731 |
|    6 | `rag_fusion_bm25`             |   true |     0.76 |  4.30 | 4.92 | 4.16 | 0.590 |  0.388 |       5537 |       8992 |
|    7 | `rag_fusion_qdrant`           |   true |     0.76 |  4.16 | 4.74 | 4.08 | 0.570 |  0.415 |       5258 |       8442 |
|    8 | `rag_fusion_bm25`             |  false |     0.74 |  4.30 | 4.72 | 3.96 | 0.587 |  0.387 |       5662 |       5850 |
|    9 | `graph_bm25`                  |   true |     0.72 |  4.34 | 4.86 | 4.02 | 0.607 |  0.395 |       6071 |       8324 |
|   10 | `graph_bm25`                  |  false |     0.72 |  4.28 | 4.72 | 3.92 | 0.607 |  0.395 |       5580 |       5619 |
|   11 | `hybrid_qdrant`               |  false |     0.72 |  4.18 | 4.96 | 4.10 | 0.673 |  0.490 |       5874 |       6156 |
|   12 | `bm25`                        |  false |     0.70 |  4.28 | 4.74 | 3.96 | 0.607 |  0.397 |       2448 |       2501 |
|   13 | `hybrid_qdrant`               |  false |     0.70 |  4.14 | 4.96 | 4.06 | 0.673 |  0.490 |       6185 |       6495 |
|   14 | `graph_hybrid_qdrant`         |  false |     0.68 |  4.10 | 4.88 | 3.94 | 0.670 |  0.491 |       5956 |       6274 |
|   15 | `bm25`                        |   true |     0.68 |  4.06 | 4.78 | 3.94 | 0.607 |  0.397 |       2695 |       5395 |
|   16 | `hybrid_qdrant`               |   true |     0.68 |  4.00 | 4.86 | 3.96 | 0.673 |  0.490 |       6162 |       8825 |
|   17 | `graph_qdrant`                |  false |     0.68 |  4.00 | 4.80 | 3.76 | 0.647 |  0.508 |       5973 |       6176 |
|   18 | `rag_fusion_qdrant`           |   true |     0.68 |  4.00 | 4.64 | 3.80 | 0.607 |  0.469 |       5478 |       8346 |
|   19 | `medcpt_zero_shot`            |  false |     0.66 |  4.00 | 4.74 | 3.98 | 0.513 |  0.329 |       5785 |       5849 |
|   20 | `graph_hybrid_medcpt_trained` |   true |     0.66 |  3.98 | 4.80 | 3.76 | 0.757 |  0.609 |       6226 |       8903 |

## Merge Check

- e2e rows: `32`
- judge rows: `32`
- merged rows: `32`
- missing judge rows: `0`
