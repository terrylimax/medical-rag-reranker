# Colab Experiment Plan

Этот план рассчитан на три Colab-запуска: обучение/индексация, retrieval benchmark, generation quality benchmark.

## 1. Где хранить артефакты

Источник истины после Colab:

```text
/content/drive/MyDrive/medical-rag-reranker-colab/<RUN_ID>/
├── manifest.json
├── data/
│   └── processed/
│       ├── qa.jsonl
│       ├── corpus.jsonl
│       ├── splits.json
│       ├── eval_queries.jsonl
│       └── qrels.tsv
├── artifacts/
│   ├── retriever_training/
│   │   ├── train_retriever.jsonl
│   │   ├── val_retriever.jsonl
│   │   └── retriever_training_data_summary.json
│   ├── models/
│   │   └── retriever/trained_medcpt_biencoder/
│   │       ├── query_encoder/
│   │       ├── doc_encoder/
│   │       └── training_summary.json
│   └── indices/
│       ├── bm25/bm25_index.json.gz
│       ├── dense_minilm/dense_index.pkl
│       ├── medcpt_zero_shot/medcpt_index.pkl
│       ├── trained_medcpt/medcpt_index.pkl
│       ├── hybrid_minilm/hybrid_index.json
│       ├── hybrid_medcpt_zero_shot/hybrid_index.json
│       ├── hybrid_trained_medcpt/hybrid_index.json
│       └── graph_metadata/graph_index.json
└── runs/
    ├── retrieval/
    ├── reranked/
    ├── generation/
    └── summaries/
```

В локальный проект после Colab стоит переносить только нужную финальную версию:

```text
artifacts/experiments/<RUN_ID>/
├── manifest.json
├── indices/
├── models/
└── summaries/
```

Если файлы крупные, не коммить `.pkl` и encoder directories напрямую в Git. Хранить их через Drive/DVC, а в репозитории держать `manifest.json`, summary tables и отчёты.

## 2. Порядок запуска

1. Открыть `colab/01_train_retriever_and_build_indices.ipynb`.
2. Включить GPU: `Runtime -> Change runtime type -> T4/A100`.
3. Задать `RUN_ID`, например `medquad_full_v1`.
4. Запустить notebook сверху вниз.
5. Проверить, что появился `manifest.json` и все индексы в `artifacts/indices`.
6. Открыть `colab/02_benchmark_retrieval_reranker_graph.ipynb` с тем же `RUN_ID`.
7. Если есть checkpoint cross-encoder reranker, задать env `RERANKER_CHECKPOINT_PATH` или переменную в notebook.
8. Запустить benchmark. Итоговая таблица будет в `runs/summaries/retrieval_benchmark_summary.csv`.
9. Открыть `colab/03_evaluate_generation_quality.ipynb`.
10. Запустить generation benchmark. Итоговая таблица будет в `runs/summaries/generation_quality_summary.csv`.

## 3. Что измерять для retrieval

Основные метрики:

- `NDCG@10`, `NDCG@20`, `NDCG@50`
- `R@10`, `R@20`, `R@50`
- `P@5`, `P@10`
- `Hit@10`, `MRR@10`
- `latency_mean_ms`, `latency_p95_ms`
- `index_size_mb`

Методы:

- `bm25`
- `dense_minilm`
- `medcpt_zero_shot`
- `trained_medcpt`
- `hybrid_minilm`
- `hybrid_medcpt_zero_shot`
- `hybrid_trained_medcpt`
- `rag_fusion_bm25`
- `rag_fusion_dense_minilm`
- `rag_fusion_medcpt_zero_shot`
- `rag_fusion_trained_medcpt`
- `graph_bm25`
- `graph_hybrid_minilm`
- `graph_hybrid_trained_medcpt`
- `<base_method>__reranked` для методов, где есть reranker checkpoint

## 4. Как трактовать graph methods

В текущих Colab-файлах graph baseline делается поверх metadata graph:

- вершины документов: `doc_id`
- связи: `diagnosis_or_topic`, `question_intent`, `group_id`
- seed candidates берутся из обычного retrieval run
- соседние документы добавляются через graph expansion
- итоговый score агрегируется через RRF-подобный boost

Это честный и воспроизводимый baseline для итоговой работы: графовая часть не использует LLM и не подглядывает в qrels. В тексте работы её лучше описывать как graph-aware candidate expansion, а не как полноразмерный медицинский knowledge graph.

## 5. Что измерять для generation

Ноутбук 03 генерирует ответы из сохранённых TREC run-файлов, поэтому можно сравнить обычные, graph-aware и reranked retrieval outputs одним генератором. Для MedQuAD важно оценивать не только reference-free groundedness, но и связь с единственным gold answer-документом.

Метрики:

- `avg_gold_in_context`
- `avg_top1_is_gold`
- `avg_gold_in_top_1`, `avg_gold_in_top_3`, `avg_gold_in_top_5`, `avg_gold_in_top_10`
- `avg_reciprocal_gold_rank`
- `gold_rank_mean`, `gold_rank_median`, `gold_rank_not_found_rate`
- `avg_rouge_l_f1_to_gold`
- `avg_lexical_cosine_to_gold`
- `avg_citation_points_to_gold`
- `avg_context_relevance`
- `avg_groundedness`
- `avg_answer_relevance`
- `avg_supported_citation_rate`
- `avg_unsupported_citation_rate`
- `generation_latency_mean_ms`

Ноутбук 03 также делает sweep по `CONTEXT_TOP_K = 1, 3, 5`. Это нужно, чтобы отделить качество поиска gold-документа от устойчивости генератора к дополнительным distractor-документам.

LLM-as-a-Judge включается опционально через переменные окружения:

```text
USE_LLM_JUDGE=true
LLM_JUDGE_BASE_URL=http://localhost:8000/v1
LLM_JUDGE_MODEL=mistralai/Mistral-7B-Instruct-v0.3
LLM_JUDGE_API_KEY=EMPTY
```

Judge получает вопрос, retrieved context, generated answer и reference answer. Он возвращает `faithfulness`, `relevance`, `completeness`, `safety`, `verdict`, `rationale`.

Для финальной таблицы стоит взять 3-5 методов:

- сильный sparse baseline: `bm25`
- лучший dense/hybrid retrieval по `NDCG@10`
- лучший graph-aware метод
- лучший reranked метод
- при необходимости RAG Fusion как отдельный ablation

## 6. Минимальный критерий готовности эксперимента

Эксперимент можно считать готовым, если есть:

- `manifest.json` с путями ко всем индексам и моделям
- retrieval summary CSV/JSON
- generation summary CSV/JSON
- markdown examples для generation
- сохранённые TREC run-файлы для воспроизводимости
- зафиксированный `RUN_ID`, branch и дата запуска
