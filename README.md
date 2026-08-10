# RAG Document Intelligence

A compact, production-minded retrieval-augmented generation workbench: ingest documents, chunk them, build a searchable vector index, retrieve evidence and evaluate retrieval quality independently of generation.

## Why this repository exists

A credible RAG project should be more than a chat wrapper. This repo focuses on the engineering pieces recruiters can inspect: deterministic ingestion, chunk metadata, pluggable embeddings, retrieval, citations, offline evaluation, tests and an API boundary.

## Initial public baseline

The first implementation intentionally runs locally and does not require paid APIs. It uses TF-IDF as a deterministic sparse retrieval baseline so the complete ingestion/retrieval/evaluation loop is testable before adding embedding providers or vector databases.

```bash
pip install -r requirements.txt
python -m src.demo
```

## Planned architecture

```text
PDF / Markdown / text
        |
        v
 ingestion + normalization
        |
        v
 metadata-aware chunking
        |
        +----> sparse baseline (TF-IDF)
        +----> dense embeddings / vector DB
                         |
                         v
                 hybrid retrieval
                         |
                         v
                  reranking layer
                         |
                         v
              evidence-grounded answer
                         |
                         v
             retrieval + answer evals
```

## Evaluation targets

- Recall@k
- MRR
- context precision
- citation coverage
- latency
- answer faithfulness when a generator is enabled

The repository uses public/synthetic examples only and contains no employer documents.
