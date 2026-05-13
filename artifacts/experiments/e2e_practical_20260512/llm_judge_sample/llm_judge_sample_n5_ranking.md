# LLM-as-a-Judge Sample Ranking

- run_id: `e2e_practical_20260512`
- sample size: `5` examples per generation run
- sort: `pass_rate`, then `avg_faithfulness`, `avg_relevance`, `avg_completeness`

## Overall Ranking

| Rank | Method                                     | Rerank | Pass | Faithfulness | Relevance | Completeness | Safety |
| ---: | ------------------------------------------ | -----: | ---: | -----------: | --------: | -----------: | -----: |
|    1 | `bm25_779137bf3e9c`                        |  false | 0.80 |         4.60 |      5.00 |         4.40 |   5.00 |
|    2 | `graph_bm25_8958ada059b3`                  |  false | 0.80 |         4.60 |      5.00 |         4.40 |   5.00 |
|    3 | `graph_qdrant_eb939b866742`                |   true | 0.80 |         4.60 |      5.00 |         4.20 |   5.00 |
|    4 | `rag_fusion_bm25_955998117418`             |  false | 0.80 |         4.40 |      5.00 |         4.20 |   5.00 |
|    5 | `rag_fusion_bm25_dd4607175604`             |  false | 0.80 |         4.40 |      5.00 |         4.20 |   5.00 |
|    6 | `dense_qdrant_8026f7536bfe`                |   true | 0.80 |         4.40 |      5.00 |         4.00 |   5.00 |
|    7 | `graph_hybrid_qdrant_29dba24853cd`         |   true | 0.80 |         4.40 |      5.00 |         4.00 |   5.00 |
|    8 | `hybrid_qdrant_6c41cc2303a0`               |   true | 0.80 |         4.40 |      5.00 |         4.00 |   5.00 |
|    9 | `hybrid_qdrant_cdeefb6d16f8`               |   true | 0.80 |         4.40 |      5.00 |         4.00 |   5.00 |
|   10 | `rag_fusion_qdrant_555d1daa83b4`           |   true | 0.80 |         4.40 |      5.00 |         4.00 |   5.00 |
|   11 | `rag_fusion_qdrant_eee7957793f5`           |   true | 0.80 |         4.40 |      5.00 |         4.00 |   5.00 |
|   12 | `rag_fusion_qdrant_555d1daa83b4`           |  false | 0.60 |         4.40 |      4.20 |         3.00 |   5.00 |
|   13 | `graph_hybrid_medcpt_trained_da2564769ef9` |  false | 0.60 |         4.20 |      5.00 |         4.00 |   5.00 |
|   14 | `rag_fusion_bm25_955998117418`             |   true | 0.60 |         4.00 |      5.00 |         4.00 |   5.00 |
|   15 | `dense_qdrant_8026f7536bfe`                |  false | 0.60 |         4.00 |      5.00 |         3.80 |   5.00 |
|   16 | `graph_qdrant_eb939b866742`                |  false | 0.60 |         4.00 |      5.00 |         3.80 |   5.00 |
|   17 | `medcpt_zero_shot_12724ffc67c5`            |   true | 0.60 |         4.00 |      5.00 |         3.80 |   5.00 |
|   18 | `rag_fusion_bm25_dd4607175604`             |   true | 0.60 |         4.00 |      5.00 |         3.80 |   5.00 |
|   19 | `rag_fusion_qdrant_eee7957793f5`           |  false | 0.60 |         4.00 |      5.00 |         3.60 |   5.00 |
|   20 | `medcpt_trained_98e67905a9d4`              |   true | 0.60 |         3.80 |      5.00 |         3.40 |   5.00 |
|   21 | `graph_hybrid_medcpt_trained_da2564769ef9` |   true | 0.60 |         3.80 |      4.60 |         3.40 |   5.00 |
|   22 | `hybrid_medcpt_trained_6a0f43c9408c`       |   true | 0.60 |         3.80 |      4.60 |         3.40 |   5.00 |
|   23 | `hybrid_medcpt_trained_61d332ff06a8`       |   true | 0.60 |         3.60 |      4.60 |         3.40 |   5.00 |
|   24 | `hybrid_medcpt_trained_6a0f43c9408c`       |  false | 0.40 |         3.80 |      5.00 |         3.80 |   5.00 |
|   25 | `graph_hybrid_qdrant_29dba24853cd`         |  false | 0.40 |         3.80 |      5.00 |         3.60 |   5.00 |
|   26 | `hybrid_qdrant_6c41cc2303a0`               |  false | 0.40 |         3.80 |      5.00 |         3.60 |   5.00 |
|   27 | `hybrid_qdrant_cdeefb6d16f8`               |  false | 0.40 |         3.80 |      5.00 |         3.60 |   5.00 |
|   28 | `hybrid_medcpt_trained_61d332ff06a8`       |  false | 0.40 |         3.40 |      5.00 |         3.60 |   5.00 |
|   29 | `graph_bm25_8958ada059b3`                  |   true | 0.40 |         3.40 |      4.20 |         3.40 |   5.00 |
|   30 | `bm25_779137bf3e9c`                        |   true | 0.40 |         3.40 |      4.20 |         3.20 |   5.00 |
|   31 | `medcpt_trained_98e67905a9d4`              |  false | 0.40 |         3.20 |      5.00 |         3.00 |   5.00 |
|   32 | `medcpt_zero_shot_12724ffc67c5`            |  false | 0.40 |         2.60 |      4.20 |         3.40 |   5.00 |

## Best Variant Per Method Family

| Family                                     | Best Run                                                                    | Pass | Faithfulness | Relevance | Completeness | Safety |
| ------------------------------------------ | --------------------------------------------------------------------------- | ---: | -----------: | --------: | -----------: | -----: |
| `bm25_779137bf3e9c`                        | `bm25_779137bf3e9c_top5_ret20_rerankfalse_heuristic`                        | 0.80 |         4.60 |      5.00 |         4.40 |   5.00 |
| `graph_bm25_8958ada059b3`                  | `graph_bm25_8958ada059b3_top5_ret20_rerankfalse_heuristic`                  | 0.80 |         4.60 |      5.00 |         4.40 |   5.00 |
| `graph_qdrant_eb939b866742`                | `graph_qdrant_eb939b866742_top5_ret20_reranktrue_heuristic`                 | 0.80 |         4.60 |      5.00 |         4.20 |   5.00 |
| `rag_fusion_bm25_955998117418`             | `rag_fusion_bm25_955998117418_top5_ret20_rerankfalse_heuristic`             | 0.80 |         4.40 |      5.00 |         4.20 |   5.00 |
| `rag_fusion_bm25_dd4607175604`             | `rag_fusion_bm25_dd4607175604_top5_ret20_rerankfalse_heuristic`             | 0.80 |         4.40 |      5.00 |         4.20 |   5.00 |
| `dense_qdrant_8026f7536bfe`                | `dense_qdrant_8026f7536bfe_top5_ret20_reranktrue_heuristic`                 | 0.80 |         4.40 |      5.00 |         4.00 |   5.00 |
| `graph_hybrid_qdrant_29dba24853cd`         | `graph_hybrid_qdrant_29dba24853cd_top5_ret20_reranktrue_heuristic`          | 0.80 |         4.40 |      5.00 |         4.00 |   5.00 |
| `hybrid_qdrant_6c41cc2303a0`               | `hybrid_qdrant_6c41cc2303a0_top5_ret20_reranktrue_heuristic`                | 0.80 |         4.40 |      5.00 |         4.00 |   5.00 |
| `hybrid_qdrant_cdeefb6d16f8`               | `hybrid_qdrant_cdeefb6d16f8_top5_ret20_reranktrue_heuristic`                | 0.80 |         4.40 |      5.00 |         4.00 |   5.00 |
| `rag_fusion_qdrant_555d1daa83b4`           | `rag_fusion_qdrant_555d1daa83b4_top5_ret20_reranktrue_heuristic`            | 0.80 |         4.40 |      5.00 |         4.00 |   5.00 |
| `rag_fusion_qdrant_eee7957793f5`           | `rag_fusion_qdrant_eee7957793f5_top5_ret20_reranktrue_heuristic`            | 0.80 |         4.40 |      5.00 |         4.00 |   5.00 |
| `graph_hybrid_medcpt_trained_da2564769ef9` | `graph_hybrid_medcpt_trained_da2564769ef9_top5_ret20_rerankfalse_heuristic` | 0.60 |         4.20 |      5.00 |         4.00 |   5.00 |
| `medcpt_zero_shot_12724ffc67c5`            | `medcpt_zero_shot_12724ffc67c5_top5_ret20_reranktrue_heuristic`             | 0.60 |         4.00 |      5.00 |         3.80 |   5.00 |
| `medcpt_trained_98e67905a9d4`              | `medcpt_trained_98e67905a9d4_top5_ret20_reranktrue_heuristic`               | 0.60 |         3.80 |      5.00 |         3.40 |   5.00 |
| `hybrid_medcpt_trained_6a0f43c9408c`       | `hybrid_medcpt_trained_6a0f43c9408c_top5_ret20_reranktrue_heuristic`        | 0.60 |         3.80 |      4.60 |         3.40 |   5.00 |
| `hybrid_medcpt_trained_61d332ff06a8`       | `hybrid_medcpt_trained_61d332ff06a8_top5_ret20_reranktrue_heuristic`        | 0.60 |         3.60 |      4.60 |         3.40 |   5.00 |
