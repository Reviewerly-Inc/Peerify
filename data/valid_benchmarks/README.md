# valid_benchmarks/

Canonical claim-verification benchmark suite. Built 2026-05-03.

## Provenance

| Benchmark | Source | Labeling |
|---|---|---|
| B1 (golden supported) | original `data/benchmark/B1_golden_supported.jsonl`, copied unchanged | gpt-5-mini extraction (claims are paraphrased from paper paragraphs; label is `Supported` by construction) |
| B2 (claim vs author response) | re-labeled by claude-sonnet-4-6, then disagreements arbitrated by claude-opus-4-7 | canonical `label` = arbitrated truth |
| B3 (claim vs paper content) | same process as B2 | canonical `label` = arbitrated truth |
| B4 (agreement) | derived from canonical B2/B3 labels | rows where B2.label == B3.label |
| B5 (verifiable) | derived from B4 | filtered by verifiable claim_type |
| B6 (human) | original `data/benchmark/B6_human.jsonl`, copied unchanged | human-annotated 150-claim subset of B2 |

## Schema additions vs the original benchmark

For B2 and B3, the canonical `label` is now the arbitrated truth. Each row also includes:

- `label_gpt5mini` — what gpt-5-mini originally said
- `label_sonnet46` — what claude-sonnet-4-6 said
- `judge_winner` ∈ {`agreement`, `gpt5mini`, `sonnet46`, `neither`}
- `judge_reasoning` (under `justification`) — explanation of the choice when arbitration happened

For B4, both B2 and B3 audit trails are preserved with `b2_label_gpt5mini`, `b2_label_sonnet46`, etc.

## Sizes

- B1: 500 rows
- B2: 800 rows
- B3: 800 rows
- B4: 175 rows (agreement subset of B2/B3)
- B5: 131 rows (verifiable filter on B4)
- B6: 150 rows (human-annotated subset of B2)

See `disagreement_analysis.md` for the labeler-disagreement statistics that motivated this revision.
