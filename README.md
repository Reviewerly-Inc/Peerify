# Peerify : Claim Verification Benchmark


Benchmark generation and evaluation pipeline for verifying claims in academic peer reviews against paper content and author responses.

## What It Does

1. **Crawls** NeurIPS/ICLR submissions, reviews, and author responses from OpenReview
2. **Pairs** reviewer comments with author responses
3. **Extracts** atomic claims from reviews using LLMs
4. **Generates** six benchmarks (B1–B6) with different verification perspectives
5. **Evaluates** retrieval and LLM-based claim verification

## Benchmarks

| ID | Name | Description |
|----|------|-------------|
| B1 | Golden Supported | Claims extracted from the paper itself (should be supported) |
| B2 | Reviewer vs Author | Reviewer claims verified against author responses |
| B3 | Reviewer vs Paper | Reviewer claims verified against the full paper |
| B4 | Agreement | Claims where B2 and B3 labels agree |
| B5 | Verifiable | Filtered subset of B4 — only verifiable claims |
| B6 | Human | Human-annotated subset |

## Project Structure

```
purify/
├── claim_verification/
│   ├── config.py                    # Paths, credentials (env vars), constants
│   ├── pipeline/
│   │   ├── orchestrator.py          # Main CLI — runs the full pipeline
│   │   ├── crawler.py               # OpenReview API crawling
│   │   ├── pairer.py                # Review ↔ response pairing
│   │   └── claim_extractor.py       # LLM-based claim extraction + filtering
│   ├── benchmarks/
│   │   ├── golden_benchmark.py      # B1
│   │   ├── reviewer_benchmark.py    # B2, B3
│   │   ├── agreement_benchmark.py   # B4
│   │   └── verifiable_benchmark.py  # B5
│   ├── evaluation/
│   │   ├── retrieval_evaluation.py         # Retrieve + verify per claim
│   │   ├── full_paper_evaluation.py        # Verify with full paper text (no retrieval)
│   │   ├── evaluate_retrieval_metrics.py   # Retrieval & classification metrics
│   │   ├── claim_extraction_evaluator.py   # Extraction quality evaluation
│   │   ├── run_all_evaluations.py          # Batch runner for all eval configs
│   │   └── run_all_extraction_evals.py     # Batch runner for extraction evals
│   ├── retrieval/
│   │   └── pipeline_wrapper.py      # TF-IDF, BM25, SBERT, FAISS, RRF, bi+cross-encoder
│   ├── preprocessing/
│   │   ├── chunking.py              # Markdown → token-aware chunks
│   │   └── pdf_processing/
│   │       ├── parse_pdf.py         # Docling & Nougat PDF parsers
│   │       └── clean_markdown.py    # Markdown cleanup utilities
│   └── utils/
│       ├── rephrase_b1_claims.py
│       ├── recalculate_retrieval_metrics.py
│       ├── aggregate_experiment_metrics.py
│       ├── calculate_benchmark_statistics.py
│       └── count_benchmark_labels.py
├── data/
│   └── benchmark/                   # Pre-built benchmark JSONL files (B1–B6)
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
export OPENREVIEW_USERNAME="your-email"
export OPENREVIEW_PASSWORD="your-password"
export OPENAI_API_KEY="sk-..."          # optional, for GPT-based verification
```

## Quick Start

### Generate benchmarks from scratch

```bash
# Full pipeline (crawl → pair → extract → B1–B5)
python -m claim_verification.pipeline.orchestrator --step all

# Individual steps
python -m claim_verification.pipeline.orchestrator --step crawl --max-papers 50
python -m claim_verification.pipeline.orchestrator --step pair
python -m claim_verification.pipeline.orchestrator --step extract
python -m claim_verification.pipeline.orchestrator --step b1
python -m claim_verification.pipeline.orchestrator --step b2b3 --model gpt-4o-mini
```

### Evaluate

```bash
# Run all retrieval + verification evaluations
python -m claim_verification.evaluation.run_all_evaluations --llm-model gpt-4o-mini --top-k 10

# Full-paper evaluation (no retrieval step)
python -m claim_verification.evaluation.full_paper_evaluation --benchmark B3 --model gpt-4o-mini

# Claim extraction quality
python -m claim_verification.evaluation.claim_extraction_evaluator --benchmark B2
```

## Configuration

All credentials come from environment variables — nothing is hardcoded.

| Variable | Purpose |
|----------|---------|
| `OPENREVIEW_USERNAME` | OpenReview login |
| `OPENREVIEW_PASSWORD` | OpenReview password |
| `OPENAI_API_KEY` | OpenAI API (for GPT models) |
| `BENCHMARK_DATA_ROOT` | Override default data directory |

## Retrieval Methods

| Method | Module function |
|--------|----------------|
| TF-IDF | `retrieve_top_k_evidences_tfidf` |
| BM25 | `retrieve_top_k_evidences_bm25` |
| SBERT | `retrieve_top_k_evidences_sbert` |
| FAISS | `retrieve_top_k_evidences_faiss` |
| RRF (BM25 + SBERT) | `retrieve_top_k_evidences_rrf` |
| Bi-encoder + Cross-encoder | `retrieve_top_k_evidences_biencoder_crossencoder` |
| BM25 + Cross-encoder | `retrieve_top_k_evidences_bm25_crossencoder` |

## Verification Labels

Every claim is classified as one of:
- **Supported** — evidence fully backs the claim
- **Partially Supported** — some alignment, some gaps
- **Contradicted** — evidence conflicts with the claim
- **Not Determined** — insufficient evidence
