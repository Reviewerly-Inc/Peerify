"""Build data/valid_benchmarks/ as the canonical, judge-arbitrated benchmark
suite. Sources:

  B1 ← data/benchmark/B1_golden_supported.jsonl   (unchanged — labels are
                                                    Supported by construction)
  B6 ← data/benchmark/B6_human.jsonl              (unchanged — already human-
                                                    annotated)
  B2 ← data/benchmark_claude/B2_reviewer_author_judged.jsonl
       — `label` = judge_label (canonical), preserves gpt-5-mini and
         sonnet-4-6 labels in `label_gpt5mini` / `label_sonnet46`.
  B3 ← data/benchmark_claude/B3_reviewer_paper_judged.jsonl  (same pattern)
  B4 ← derived from B2/B3 valid: keep rows where the canonical labels match
  B5 ← B4 filtered by verifiable claim_type
"""
from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

VERIFIABLE_TYPES = {"experimental", "methodology", "quantitative", "baseline"}
NON_VERIFIABLE_TYPES = {"novelty", "presentation", "clarity", "other"}
VERIFIABLE_KEYWORDS = [
    "accuracy", "performance", "result", "experiment", "dataset",
    "baseline", "benchmark", "metric", "evaluation", "ablation",
    "table", "training", "parameter", "loss", "convergence",
    "runtime", "method", "algorithm", "implementation",
    "statistical", "significant", "%", "improve",
]
NON_VERIFIABLE_KEYWORDS = [
    "novel", "originality", "contribution", "figure quality",
    "picture", "visualization", "writing", "presentation",
    "clarity", "grammar", "typo", "interesting", "exciting",
    "related work", "motivation",
]


def _read_jsonl(p: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _write_jsonl(p: Path, rows: List[Dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _is_verifiable(record: Dict[str, Any]) -> bool:
    claim_type = (record.get("claim_type") or "other").lower()
    claim_text = (record.get("claim") or "").lower()
    if claim_type in VERIFIABLE_TYPES:
        return True
    if claim_type in NON_VERIFIABLE_TYPES:
        return False
    has_v = any(kw in claim_text for kw in VERIFIABLE_KEYWORDS)
    has_n = any(kw in claim_text for kw in NON_VERIFIABLE_KEYWORDS)
    if has_v and not has_n:
        return True
    if has_n and not has_v:
        return False
    return claim_type in VERIFIABLE_TYPES


def _promote_judged(rec: Dict[str, Any], extra_label_field: str = "") -> Dict[str, Any]:
    """Take a judged record and produce a 'valid' record:
    - `label` = judge_label (the canonical truth)
    - `label_gpt5mini` = original gpt-5-mini label
    - `label_sonnet46` = sonnet-4-6 label
    - `justification` = judge_reasoning if disagreement was resolved else original
    - keep the audit fields (judge_winner, judge_model)
    """
    label_gpt5mini = rec.get("label")
    label_sonnet46 = rec.get("claude_label")
    judge_label = rec.get("judge_label")
    judge_winner = rec.get("judge_winner", "agreement")

    if judge_winner == "agreement":
        justification = rec.get("justification") or rec.get("claude_justification") or ""
        evidence = rec.get("evidence") or rec.get("claude_evidence") or ""
    else:
        justification = rec.get("judge_reasoning") or ""
        # Use whichever evidence is from the winning candidate; otherwise gpt-5-mini's
        if judge_winner == "sonnet46":
            evidence = rec.get("claude_evidence") or ""
        elif judge_winner == "gpt5mini":
            evidence = rec.get("evidence") or ""
        else:
            evidence = rec.get("evidence") or rec.get("claude_evidence") or ""

    out = dict(rec)
    # Replace the canonical label with the judge's verdict
    out["label"] = judge_label
    out["label_gpt5mini"] = label_gpt5mini
    out["label_sonnet46"] = label_sonnet46
    out["justification"] = justification
    out["evidence"] = evidence
    if extra_label_field:
        # B3 has an additional 'section' field on the original
        out["section"] = rec.get("claude_section") if judge_winner == "sonnet46" \
                         else rec.get("section", "")
    # Drop the prefixed 'claude_*' fields to keep the schema clean — but keep the
    # judge fields as the audit trail.
    for k in ["claude_label", "claude_justification", "claude_evidence",
              "claude_section", "claude_model", "claude_labeling_timestamp"]:
        out.pop(k, None)
    return out


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    src_orig = repo / "data" / "benchmark"
    src_claude = repo / "data" / "benchmark_claude"
    out = repo / "data" / "valid_benchmarks"
    out.mkdir(parents=True, exist_ok=True)

    # ---- B1: unchanged copy ----
    b1_src = src_orig / "B1_golden_supported.jsonl"
    b1_dst = out / "B1_golden_supported.jsonl"
    shutil.copy2(b1_src, b1_dst)
    print(f"B1: copied {b1_src.name} -> {b1_dst}")

    # ---- B6: unchanged copy ----
    b6_src = src_orig / "B6_human.jsonl"
    b6_dst = out / "B6_human.jsonl"
    shutil.copy2(b6_src, b6_dst)
    print(f"B6: copied {b6_src.name} -> {b6_dst}")

    # ---- B2: promote judged file ----
    b2_judged = _read_jsonl(src_claude / "B2_reviewer_author_judged.jsonl")
    b2_valid = [_promote_judged(r) for r in b2_judged]
    _write_jsonl(out / "B2_reviewer_author.jsonl", b2_valid)
    print(f"B2: wrote {len(b2_valid)} rows to {out / 'B2_reviewer_author.jsonl'}")

    # ---- B3: promote judged file ----
    b3_judged = _read_jsonl(src_claude / "B3_reviewer_paper_judged.jsonl")
    b3_valid = [_promote_judged(r, extra_label_field="section") for r in b3_judged]
    _write_jsonl(out / "B3_reviewer_paper.jsonl", b3_valid)
    print(f"B3: wrote {len(b3_valid)} rows to {out / 'B3_reviewer_paper.jsonl'}")

    # ---- B4: derive from B2/B3 valid (use canonical `label` field) ----
    b3_index = {(r.get("paper_id"), r.get("claim")): r for r in b3_valid}
    b4_rows = []
    for b2_row in b2_valid:
        key = (b2_row.get("paper_id"), b2_row.get("claim"))
        b3_row = b3_index.get(key)
        if b3_row is None:
            continue
        b2_label = b2_row["label"]
        b3_label = b3_row["label"]
        merged = {
            "claim": b2_row["claim"],
            "claim_type": b2_row.get("claim_type"),
            "paper_id": b2_row.get("paper_id"),
            "paper_title": b2_row.get("paper_title"),
            "paper_venue": b2_row.get("paper_venue"),
            "paper_decision": b2_row.get("paper_decision"),
            "decision": b2_row.get("decision", "Unknown"),
            "review_id": b2_row.get("review_id"),
            "reviewer": b2_row.get("reviewer"),
            "review_text": b2_row.get("review_text"),
            "b2_label": b2_label,
            "b3_label": b3_label,
            "b2_justification": b2_row.get("justification"),
            "b3_justification": b3_row.get("justification"),
            "b2_evidence": b2_row.get("evidence"),
            "b3_evidence": b3_row.get("evidence"),
            "b3_section": b3_row.get("section"),
            "agreement": b2_label == b3_label,
            "agreement_type": "exact" if b2_label == b3_label else (
                "disagree" if {b2_label, b3_label} == {"Supported", "Contradicted"}
                else "partial"
            ),
            # Audit trail of how the canonical labels were obtained
            "b2_label_gpt5mini": b2_row.get("label_gpt5mini"),
            "b2_label_sonnet46": b2_row.get("label_sonnet46"),
            "b3_label_gpt5mini": b3_row.get("label_gpt5mini"),
            "b3_label_sonnet46": b3_row.get("label_sonnet46"),
            "b2_judge_winner": b2_row.get("judge_winner"),
            "b3_judge_winner": b3_row.get("judge_winner"),
        }
        b4_rows.append(merged)

    b4_exact = [r for r in b4_rows if r["agreement"]]
    _write_jsonl(out / "B4_agreement.jsonl", b4_exact)
    print(f"B4: {len(b4_exact)} agreement rows out of {len(b4_rows)} pairs "
          f"({100*len(b4_exact)/max(len(b4_rows),1):.1f}%)")

    # ---- B5: verifiable filter on B4 ----
    b5 = [dict(r, is_verifiable=True) for r in b4_exact if _is_verifiable(r)]
    _write_jsonl(out / "B5_verifiable.jsonl", b5)
    print(f"B5: {len(b5)} verifiable rows from B4 ({100*len(b5)/max(len(b4_exact),1):.1f}%)")

    # ---- README in the new folder ----
    readme = out / "README.md"
    readme.write_text(_build_readme(b2_valid, b3_valid, b4_exact, b5))
    print(f"wrote {readme}")

    # ---- summary stats ----
    print("\n=== B2/B3 canonical label distribution ===")
    print(f"B2: {dict(Counter(r['label'] for r in b2_valid))}")
    print(f"B3: {dict(Counter(r['label'] for r in b3_valid))}")
    print("\n=== judge winner (B2/B3) — applies only to disagreements ===")
    b2w = Counter(r.get("judge_winner") for r in b2_valid)
    b3w = Counter(r.get("judge_winner") for r in b3_valid)
    print(f"B2: {dict(b2w)}")
    print(f"B3: {dict(b3w)}")


def _build_readme(b2, b3, b4, b5) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    return f"""# valid_benchmarks/

Canonical claim-verification benchmark suite. Built {today}.

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
- `judge_winner` ∈ {{`agreement`, `gpt5mini`, `sonnet46`, `neither`}}
- `judge_reasoning` (under `justification`) — explanation of the choice when arbitration happened

For B4, both B2 and B3 audit trails are preserved with `b2_label_gpt5mini`, `b2_label_sonnet46`, etc.

## Sizes

- B1: 500 rows
- B2: {len(b2)} rows
- B3: {len(b3)} rows
- B4: {len(b4)} rows (agreement subset of B2/B3)
- B5: {len(b5)} rows (verifiable filter on B4)
- B6: 150 rows (human-annotated subset of B2)

See `disagreement_analysis.md` for the labeler-disagreement statistics that motivated this revision.
"""


if __name__ == "__main__":
    main()
