# Labeler Disagreement Analysis

## Setup

The B2 (claim vs author response) and B3 (claim vs paper content) benchmarks each contain 800 atomic reviewer claims drawn from 84 NeurIPS 2024 / ICLR 2024 papers. Every claim was labeled twice, independently:

- **Labeler A**: `gpt-5-mini` — used in the original benchmark generation pass
- **Labeler B**: `claude-sonnet-4-6` — re-labeling pass on the same claims and same source material, using byte-identical prompts

For every record where the two labelers disagreed, an **independent third-model arbitrator** — `claude-opus-4-7`, distinct from both labelers — was given the claim, the relevant source material, and both candidate labels (with model identities anonymised and order randomised per record) and asked to either pick the correct label or produce a new one if both candidates were wrong.

> **Methodology note.** This document refers to the arbitrator as "the judge". The judge here is **an LLM (`claude-opus-4-7`), not a human annotator.** A subsequent human-annotation pass would be valuable as additional ground truth; it has not yet been performed for B2/B3 outside the 150-claim B6 subset.

## Inter-labeler agreement before arbitration

| benchmark | n | gpt-5-mini agrees with sonnet-4-6 | rate |
|---|---:|---:|---:|
| B2 (claim vs author response) | 800 | 480 | **60.0%** |
| B3 (claim vs paper content) | 800 | 459 | **57.4%** |

The two labelers disagreed on 320 B2 records and 341 B3 records.

### Where the labelers diverged

`claude-sonnet-4-6` was systematically **more conservative** than `gpt-5-mini`:

- On B2, gpt-5-mini called 286 claims `Supported`; sonnet-4-6 called only 158 the same way. 107 of those moved to `Partially Supported` and 42 moved to `Contradicted`.
- On B3, gpt-5-mini gave a definite `Supported` or `Contradicted` label to 447 claims; sonnet-4-6 gave only 333. The ~70 record gap shifted into `Not Determined` — sonnet-4-6 was more willing to say the paper does not address the claim at all.

The two models agreed almost completely on `Not Determined` itself (52/67 on B2, 193/251 on B3), but diverged on every confident label.

## Judge verdicts on the disagreements

| | n disagreements | gpt-5-mini correct | sonnet-4-6 correct | neither (judge gave a third label) |
|---|---:|---:|---:|---:|
| **B2** | 320 | **158 (49.4%)** | 136 (42.5%) | 26 (8.1%) |
| **B3** | 341 | **168 (49.3%)** | 137 (40.2%) | 35 (10.3%) |

One B3 row failed JSON parsing during arbitration and is preserved with `judge_winner = "error"`.

**Headline reading**: when the two labelers disagreed, gpt-5-mini was correct slightly more often (~49%) than sonnet-4-6 (~41%), but **neither label was correct in roughly 1 of every 10 disagreements**. The judge volunteered a third option in those cases.

## Where each labeler tends to be right

These breakdowns slice the disagreements by the (gpt-5-mini, sonnet-4-6) label pair to show *what kind of disagreement* tends to favour which model.

### B2 disagreements — top patterns

| gpt-5-mini said | sonnet-4-6 said | n | gpt-5-mini wins | sonnet-4-6 wins | neither |
|---|---|---:|---:|---:|---:|
| Supported | Partially Supported | 107 | 58 | 39 | 10 |
| Contradicted | Partially Supported | 58 | 45 | 12 | 1 |
| Supported | Contradicted | 42 | 11 | **27** | 4 |
| Partially Supported | Contradicted | 34 | 11 | **20** | 3 |
| Partially Supported | Supported | 23 | 9 | 13 | 1 |
| Partially Supported | Not Determined | 22 | 7 | **15** | 0 |

Reading: when sonnet-4-6 disagreed with gpt-5-mini's `Supported` and instead chose `Contradicted`, it was right about twice as often as gpt-5-mini (27 vs 11). When sonnet-4-6 instead pushed `Supported` claims down to `Partially Supported`, it was right less often than gpt-5-mini was. So sonnet-4-6's instinct to be more cautious is well-calibrated when it goes all the way to `Contradicted`, but somewhat over-corrects when it merely steps down to `Partially Supported`.

### B3 disagreements — top patterns

| gpt-5-mini said | sonnet-4-6 said | n | gpt-5-mini wins | sonnet-4-6 wins | neither |
|---|---|---:|---:|---:|---:|
| Supported | Not Determined | 65 | 18 | **43** | 3 |
| Supported | Partially Supported | 43 | 33 | 5 | 5 |
| Contradicted | Partially Supported | 43 | 23 | 13 | 7 |
| Contradicted | Not Determined | 43 | 10 | **31** | 2 |
| Not Determined | Supported | 27 | 25 | 2 | 0 |
| Partially Supported | Not Determined | 22 | 3 | **17** | 2 |

Reading: on B3, when sonnet-4-6 said `Not Determined` against any confident gpt-5-mini label, sonnet-4-6 was right by a wide margin — `Supported→Not Determined` (43 vs 18), `Contradicted→Not Determined` (31 vs 10), `Partially Supported→Not Determined` (17 vs 3). gpt-5-mini was over-claiming evidence in the paper that wasn't actually there. Conversely, when gpt-5-mini said `Not Determined` and sonnet-4-6 disagreed and called `Supported`, gpt-5-mini was nearly always right (25 vs 2) — sonnet-4-6 went looking for support and over-found it.

## Canonical-label distribution after arbitration

| label | B2 | B3 |
|---|---:|---:|
| Supported | 217 | 207 |
| Partially Supported | 310 | 93 |
| Contradicted | 189 | 141 |
| Not Determined | 84 | 359 |

The B3 distribution is heavily weighted toward `Not Determined` after arbitration — a meaningful share of reviewer claims simply do not have a verifiable answer in the paper text alone.

## Effect on derived benchmarks

- **B4 (agreement)**: rows where the canonical B2 label equals the canonical B3 label. Original (gpt-5-mini-only): 204. After arbitration: **175**.
- **B5 (verifiable filter on B4)**: original: 609. After arbitration: **131**.

The dramatic drop in B5 is mostly explained by the migration of B3 labels into `Not Determined`, which forces `Not Determined`/`Not Determined` to dominate B4 and removes most of the `Supported`/`Supported` and `Contradicted`/`Contradicted` rows that were the bulk of the original B5.

## Files

- `B1_golden_supported.jsonl`, `B6_human.jsonl` — copied unchanged from the original benchmark suite
- `B2_reviewer_author.jsonl`, `B3_reviewer_paper.jsonl` — canonical `label` is the arbitrated truth; original gpt-5-mini and sonnet-4-6 labels preserved as `label_gpt5mini` and `label_sonnet46`; `judge_winner` records how the canonical label was decided
- `B4_agreement.jsonl`, `B5_verifiable.jsonl` — re-derived from the canonical B2/B3 labels

## Anti-bias controls used during arbitration

- The judge model (`claude-opus-4-7`) was different from both labeler models.
- Candidate labels were presented to the judge as `Candidate 1` and `Candidate 2`, never named.
- The (gpt-5-mini-first vs sonnet-4-6-first) ordering was randomised per record using a seeded RNG.
- Each judge call also produced a 2–3 sentence reasoning string referencing concrete evidence from the source; these reasonings are stored in the `justification` field of the canonical B2/B3 files for audit.

## Open follow-ups

1. A real human-annotation pass on a stratified sample of the 661 disagreements would tell us how often the LLM judge itself is wrong. The 150-row B6 set is the only human-labelled subset right now.
2. The single B3 record that failed JSON parsing (`paper_id = BOrut7M2X7`) can be re-judged in isolation if exact accounting matters.
