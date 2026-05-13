# LLM-as-a-Judge Sample Ranking

- run_id: `e2e_practical_20260512`
- sample size: `50` examples per generation run
- sort: `pass_rate`, then `avg_faithfulness`, `avg_relevance`, `avg_completeness`

## Overall Ranking

| Rank | Method                                     | Rerank | Pass | Faithfulness | Relevance | Completeness | Safety |
| ---: | ------------------------------------------ | -----: | ---: | -----------: | --------: | -----------: | -----: |
|    1 | `rag_fusion_qdrant_555d1daa83b4`           |  false | 0.86 |         4.64 |      4.90 |         4.28 |   4.96 |
|    2 | `medcpt_zero_shot_12724ffc67c5`            |   true | 0.80 |         4.32 |      4.88 |         4.26 |   4.96 |
|    3 | `rag_fusion_qdrant_eee7957793f5`           |  false | 0.78 |         4.38 |      4.98 |         4.20 |   4.94 |
|    4 | `rag_fusion_bm25_955998117418`             |  false | 0.76 |         4.34 |      4.88 |         4.10 |   5.00 |
|    5 | `rag_fusion_bm25_dd4607175604`             |   true | 0.76 |         4.30 |      4.94 |         4.18 |   4.94 |
|    6 | `rag_fusion_bm25_955998117418`             |   true | 0.76 |         4.30 |      4.92 |         4.16 |   4.94 |
|    7 | `rag_fusion_qdrant_555d1daa83b4`           |   true | 0.76 |         4.16 |      4.74 |         4.08 |   4.96 |
|    8 | `rag_fusion_bm25_dd4607175604`             |  false | 0.74 |         4.30 |      4.72 |         3.96 |   5.00 |
|    9 | `graph_bm25_8958ada059b3`                  |   true | 0.72 |         4.34 |      4.86 |         4.02 |   4.94 |
|   10 | `graph_bm25_8958ada059b3`                  |  false | 0.72 |         4.28 |      4.72 |         3.92 |   5.00 |
|   11 | `hybrid_qdrant_6c41cc2303a0`               |  false | 0.72 |         4.18 |      4.96 |         4.10 |   4.96 |
|   12 | `bm25_779137bf3e9c`                        |  false | 0.70 |         4.28 |      4.74 |         3.96 |   5.00 |
|   13 | `hybrid_qdrant_cdeefb6d16f8`               |  false | 0.70 |         4.14 |      4.96 |         4.06 |   4.96 |
|   14 | `graph_hybrid_qdrant_29dba24853cd`         |  false | 0.68 |         4.10 |      4.88 |         3.94 |   4.94 |
|   15 | `bm25_779137bf3e9c`                        |   true | 0.68 |         4.06 |      4.78 |         3.94 |   4.94 |
|   16 | `hybrid_qdrant_cdeefb6d16f8`               |   true | 0.68 |         4.00 |      4.86 |         3.96 |   4.96 |
|   17 | `graph_qdrant_eb939b866742`                |  false | 0.68 |         4.00 |      4.80 |         3.76 |   4.98 |
|   18 | `rag_fusion_qdrant_eee7957793f5`           |   true | 0.68 |         4.00 |      4.64 |         3.80 |   4.96 |
|   19 | `medcpt_zero_shot_12724ffc67c5`            |  false | 0.66 |         4.00 |      4.74 |         3.98 |   4.94 |
|   20 | `graph_hybrid_medcpt_trained_da2564769ef9` |   true | 0.66 |         3.98 |      4.80 |         3.76 |   4.94 |
|   21 | `dense_qdrant_8026f7536bfe`                |  false | 0.66 |         3.96 |      4.76 |         3.68 |   4.90 |
|   22 | `graph_hybrid_qdrant_29dba24853cd`         |   true | 0.66 |         3.94 |      4.88 |         3.88 |   4.96 |
|   23 | `graph_qdrant_eb939b866742`                |   true | 0.66 |         3.92 |      4.74 |         3.82 |   4.96 |
|   24 | `hybrid_qdrant_6c41cc2303a0`               |   true | 0.66 |         3.90 |      4.78 |         3.88 |   4.96 |
|   25 | `hybrid_medcpt_trained_61d332ff06a8`       |  false | 0.64 |         4.04 |      4.88 |         3.92 |   4.90 |
|   26 | `graph_hybrid_medcpt_trained_da2564769ef9` |  false | 0.64 |         4.02 |      4.84 |         3.96 |   4.90 |
|   27 | `hybrid_medcpt_trained_6a0f43c9408c`       |   true | 0.64 |         3.88 |      4.80 |         3.70 |   4.94 |
|   28 | `medcpt_trained_98e67905a9d4`              |   true | 0.64 |         3.88 |      4.76 |         3.86 |   4.92 |
|   29 | `hybrid_medcpt_trained_61d332ff06a8`       |   true | 0.64 |         3.86 |      4.80 |         3.70 |   4.94 |
|   30 | `dense_qdrant_8026f7536bfe`                |   true | 0.64 |         3.86 |      4.54 |         3.74 |   4.96 |
|   31 | `hybrid_medcpt_trained_6a0f43c9408c`       |  false | 0.62 |         4.00 |      4.88 |         3.88 |   4.90 |
|   32 | `medcpt_trained_98e67905a9d4`              |  false | 0.62 |         3.76 |      4.76 |         3.68 |   4.96 |

## Best Variant Per Method Family

| Family                                     | Best Run                                                                   | Pass | Faithfulness | Relevance | Completeness | Safety |
| ------------------------------------------ | -------------------------------------------------------------------------- | ---: | -----------: | --------: | -----------: | -----: |
| `rag_fusion_qdrant_555d1daa83b4`           | `rag_fusion_qdrant_555d1daa83b4_top5_ret20_rerankfalse_heuristic`          | 0.86 |         4.64 |      4.90 |         4.28 |   4.96 |
| `medcpt_zero_shot_12724ffc67c5`            | `medcpt_zero_shot_12724ffc67c5_top5_ret20_reranktrue_heuristic`            | 0.80 |         4.32 |      4.88 |         4.26 |   4.96 |
| `rag_fusion_qdrant_eee7957793f5`           | `rag_fusion_qdrant_eee7957793f5_top5_ret20_rerankfalse_heuristic`          | 0.78 |         4.38 |      4.98 |         4.20 |   4.94 |
| `rag_fusion_bm25_955998117418`             | `rag_fusion_bm25_955998117418_top5_ret20_rerankfalse_heuristic`            | 0.76 |         4.34 |      4.88 |         4.10 |   5.00 |
| `rag_fusion_bm25_dd4607175604`             | `rag_fusion_bm25_dd4607175604_top5_ret20_reranktrue_heuristic`             | 0.76 |         4.30 |      4.94 |         4.18 |   4.94 |
| `graph_bm25_8958ada059b3`                  | `graph_bm25_8958ada059b3_top5_ret20_reranktrue_heuristic`                  | 0.72 |         4.34 |      4.86 |         4.02 |   4.94 |
| `hybrid_qdrant_6c41cc2303a0`               | `hybrid_qdrant_6c41cc2303a0_top5_ret20_rerankfalse_heuristic`              | 0.72 |         4.18 |      4.96 |         4.10 |   4.96 |
| `bm25_779137bf3e9c`                        | `bm25_779137bf3e9c_top5_ret20_rerankfalse_heuristic`                       | 0.70 |         4.28 |      4.74 |         3.96 |   5.00 |
| `hybrid_qdrant_cdeefb6d16f8`               | `hybrid_qdrant_cdeefb6d16f8_top5_ret20_rerankfalse_heuristic`              | 0.70 |         4.14 |      4.96 |         4.06 |   4.96 |
| `graph_hybrid_qdrant_29dba24853cd`         | `graph_hybrid_qdrant_29dba24853cd_top5_ret20_rerankfalse_heuristic`        | 0.68 |         4.10 |      4.88 |         3.94 |   4.94 |
| `graph_qdrant_eb939b866742`                | `graph_qdrant_eb939b866742_top5_ret20_rerankfalse_heuristic`               | 0.68 |         4.00 |      4.80 |         3.76 |   4.98 |
| `graph_hybrid_medcpt_trained_da2564769ef9` | `graph_hybrid_medcpt_trained_da2564769ef9_top5_ret20_reranktrue_heuristic` | 0.66 |         3.98 |      4.80 |         3.76 |   4.94 |
| `dense_qdrant_8026f7536bfe`                | `dense_qdrant_8026f7536bfe_top5_ret20_rerankfalse_heuristic`               | 0.66 |         3.96 |      4.76 |         3.68 |   4.90 |
| `hybrid_medcpt_trained_61d332ff06a8`       | `hybrid_medcpt_trained_61d332ff06a8_top5_ret20_rerankfalse_heuristic`      | 0.64 |         4.04 |      4.88 |         3.92 |   4.90 |
| `hybrid_medcpt_trained_6a0f43c9408c`       | `hybrid_medcpt_trained_6a0f43c9408c_top5_ret20_reranktrue_heuristic`       | 0.64 |         3.88 |      4.80 |         3.70 |   4.94 |
| `medcpt_trained_98e67905a9d4`              | `medcpt_trained_98e67905a9d4_top5_ret20_reranktrue_heuristic`              | 0.64 |         3.88 |      4.76 |         3.86 |   4.92 |

## Reranker Delta By Base Method

| Base Method                                | No Rerank Pass | Rerank Pass | Delta | No Rerank Faith | Rerank Faith |
| ------------------------------------------ | -------------: | ----------: | ----: | --------------: | -----------: |
| `bm25_779137bf3e9c`                        |           0.70 |        0.68 | -0.02 |            4.28 |         4.06 |
| `dense_qdrant_8026f7536bfe`                |           0.66 |        0.64 | -0.02 |            3.96 |         3.86 |
| `graph_bm25_8958ada059b3`                  |           0.72 |        0.72 | +0.00 |            4.28 |         4.34 |
| `graph_hybrid_medcpt_trained_da2564769ef9` |           0.64 |        0.66 | +0.02 |            4.02 |         3.98 |
| `graph_hybrid_qdrant_29dba24853cd`         |           0.68 |        0.66 | -0.02 |            4.10 |         3.94 |
| `graph_qdrant_eb939b866742`                |           0.68 |        0.66 | -0.02 |            4.00 |         3.92 |
| `hybrid_medcpt_trained_61d332ff06a8`       |           0.64 |        0.64 | +0.00 |            4.04 |         3.86 |
| `hybrid_medcpt_trained_6a0f43c9408c`       |           0.62 |        0.64 | +0.02 |            4.00 |         3.88 |
| `hybrid_qdrant_6c41cc2303a0`               |           0.72 |        0.66 | -0.06 |            4.18 |         3.90 |
| `hybrid_qdrant_cdeefb6d16f8`               |           0.70 |        0.68 | -0.02 |            4.14 |         4.00 |
| `medcpt_trained_98e67905a9d4`              |           0.62 |        0.64 | +0.02 |            3.76 |         3.88 |
| `medcpt_zero_shot_12724ffc67c5`            |           0.66 |        0.80 | +0.14 |            4.00 |         4.32 |
| `rag_fusion_bm25_955998117418`             |           0.76 |        0.76 | +0.00 |            4.34 |         4.30 |
| `rag_fusion_bm25_dd4607175604`             |           0.74 |        0.76 | +0.02 |            4.30 |         4.30 |
| `rag_fusion_qdrant_555d1daa83b4`           |           0.86 |        0.76 | -0.10 |            4.64 |         4.16 |
| `rag_fusion_qdrant_eee7957793f5`           |           0.78 |        0.68 | -0.10 |            4.38 |         4.00 |
