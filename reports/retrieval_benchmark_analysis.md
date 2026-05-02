# Retrieval Benchmark Analysis

> Historical report. Do not use this table as the canonical source for current
> retrieval metrics. Current metrics must be read from the generated
> `runs/*.trec.metrics.json` artifacts and the canonical RRF report
> `reports/rrf_retrieval_comparison.md`.

## Setup

- eval queries: `data/processed/eval_queries.jsonl`
- qrels: `data/processed/qrels.tsv`
- queries: `300`
- corpus: `data/processed/corpus.jsonl`
- top_k: `10`
- hybrid candidate pool: `50`
- hybrid score fusion: `alpha=0.6`
- hybrid RRF: `rrf_k=60`

## Results

| Retriever    |    P@5 |    R@5 | NDCG@5 |   P@10 |   R@10 | NDCG@10 |  Hit@1 | Hit@10 | Avg latency ms |
| ------------ | -----: | -----: | -----: | -----: | -----: | ------: | -----: | -----: | -------------: |
| BM25         | 0.1220 | 0.6100 | 0.4379 | 0.0713 | 0.7133 |  0.4716 | 0.2533 | 0.7133 |          41.40 |
| Dense        | 0.1513 | 0.7567 | 0.6239 | 0.0840 | 0.8400 |  0.6516 | 0.4533 | 0.8400 |          41.51 |
| Hybrid score | 0.1467 | 0.7333 | 0.5755 | 0.0807 | 0.8067 |  0.6002 | 0.4000 | 0.8067 |          46.03 |
| Hybrid RRF   | 0.1467 | 0.7333 | 0.5815 | 0.0813 | 0.8133 |  0.6082 | 0.3967 | 0.8133 |          45.59 |

Additional rank diagnostics:

- `BM25`: mean rank of found relevant doc `3.00`, not found in top-10 `86`
- `Dense`: mean rank of found relevant doc `2.23`, not found in top-10 `48`
- `Hybrid score`: mean rank of found relevant doc `2.38`, not found in top-10 `58`
- `Hybrid RRF`: mean rank of found relevant doc `2.39`, not found in top-10 `56`

## Interpretation

### Best baseline

On the current MedQuAD-derived corpus, the best standalone retriever is `dense`.

- It has the strongest `R@5`, `R@10`, `NDCG@5`, `NDCG@10`, `Hit@1`, and `Hit@10`.
- It is essentially tied with BM25 on average latency in this local setup.
- It finds the relevant answer snippet more often and places it earlier in the ranking.

### Why dense wins here

The current corpus is answer-centric:

- one document is one answer snippet
- many relevant cases are paraphrastic or semantically close
- lexical overlap is useful, but exact token matching is not enough

That favors semantic retrieval. A pretrained sentence embedding model can connect medical paraphrases better than BM25, which only sees token overlap and document-frequency weighting.

### Why hybrid does not beat dense here

The hybrid variants help BM25 a lot, but they do not surpass dense on this corpus.

Likely reasons:

- BM25 is the weaker signal here, so adding it can dilute a stronger dense ranking
- qrels contain only one gold document per query, so extra lexical candidates do not get much credit
- the corpus already consists of compact answer snippets rather than long noisy documents, so dense retrieval gets a clean semantic matching problem

### Score fusion vs RRF

`hybrid_rrf` is slightly better than `hybrid_score`.

- `NDCG@10`: `0.6082` vs `0.6002`
- `R@10`: `0.8133` vs `0.8067`
- `Hit@10`: `0.8133` vs `0.8067`

This is consistent with the usual expectation that `RRF` is more robust when different retrievers produce scores on different scales.

## Methodological justification

### Why these data

`MedQuAD` is appropriate for the current baseline because it provides:

- domain-specific medical questions
- answer texts that can be used directly as retrieval documents
- reproducible `query -> relevant answer` mapping for offline evaluation

This makes it a good first-stage benchmark for comparing retrieval methods before moving to harder corpora with long documents and chunking.

### Why this validation format

The project uses standard IR validation:

- `eval_queries.jsonl` for held-out questions
- `qrels.tsv` in TREC format
- `P@k`, `R@k`, `NDCG@k`

This is the right choice for retriever comparison because it evaluates ranking quality directly and is reproducible across runs.

### Why these indexing engines

- `BM25` is the lexical baseline every retrieval setup should beat or at least match.
- `Dense` tests whether semantic embeddings help on medical paraphrases.
- `Hybrid` tests whether lexical and semantic signals complement each other.
- `RRF` is worth testing because it fuses rankings without assuming the score scales are compatible.

### Why these experiments

This benchmark sequence is justified because it answers the minimum defensible research questions:

1. Does semantic retrieval beat lexical retrieval on this corpus?
2. Does hybrid fusion add value over pure dense retrieval?
3. If hybrid is used, is `score fusion` or `RRF` more stable?

## Training decision

You do **not** need to train a retriever or reranker before benchmarking `BM25`, `dense`, `hybrid(score)`, and `hybrid(rrf)`.

- `BM25` requires no training.
- `Dense` here uses a pretrained sentence embedding model and only needs indexing.
- `Hybrid` only combines BM25 and dense outputs.

You **do** need reranker training if the next experiment is:

- `BM25 -> reranker`
- `dense -> reranker`
- `hybrid -> reranker`

That is a separate second-stage experiment, not a prerequisite for retriever benchmarking.

## Recommendation

For the current corpus and validation setup:

- use `dense` as the strongest baseline
- keep `BM25` as the lexical reference point
- keep `hybrid_rrf` as the main hybrid variant if you want one hybrid baseline
- postpone reranker training until after the retrieval-only comparison is fixed and documented
