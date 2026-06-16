"""Re-derive B4 (agreement) and B5 (verifiable filter) from the Claude-relabeled
B2 + B3 outputs, using the same logic as
claim_verification/benchmarks/agreement_benchmark.py and verifiable_benchmark.py
but driven by `claude_label` instead of the original `label`.

Outputs:
  data/benchmark/B4_agreement_claude.jsonl
  data/benchmark/B5_verifiable_claude.jsonl
"""
from __future__ import annotations

import json
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
    rows = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(p: Path, rows: List[Dict[str, Any]]) -> None:
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _is_verifiable(record: Dict[str, Any]) -> bool:
    """Mirror VerifiableBenchmarkGenerator._is_verifiable."""
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


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    bench = repo / "data" / "benchmark"

    b2 = _read_jsonl(bench / "B2_reviewer_author_claude.jsonl")
    b3 = _read_jsonl(bench / "B3_reviewer_paper_claude.jsonl")
    if len(b2) != len(b3):
        raise SystemExit(f"Row-count mismatch: B2={len(b2)} B3={len(b3)}")

    # Original B2/B3 use claim order; align on (paper_id, claim) just in case.
    b3_index = {(r.get("paper_id"), r.get("claim")): r for r in b3}

    b4_rows: List[Dict[str, Any]] = []
    for b2_row in b2:
        key = (b2_row.get("paper_id"), b2_row.get("claim"))
        b3_row = b3_index.get(key)
        if b3_row is None:
            continue
        b2_label = b2_row["claude_label"]
        b3_label = b3_row["claude_label"]
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

            # Original (gpt-5-mini) labels for reference
            "b2_label_gpt5mini": b2_row.get("label"),
            "b3_label_gpt5mini": b3_row.get("label"),

            # Claude (sonnet-4-6) labels
            "b2_label": b2_label,
            "b3_label": b3_label,
            "b2_justification": b2_row.get("claude_justification"),
            "b3_justification": b3_row.get("claude_justification"),
            "b2_evidence": b2_row.get("claude_evidence"),
            "b3_evidence": b3_row.get("claude_evidence"),
            "b3_section": b3_row.get("claude_section"),

            "agreement": b2_label == b3_label,
            "agreement_type": "exact" if b2_label == b3_label else (
                "disagree" if {b2_label, b3_label} == {"Supported", "Contradicted"}
                else "partial"
            ),

            "claude_model": "claude-sonnet-4-6",
            "labeling_timestamp": datetime.now().isoformat(),
        }
        b4_rows.append(merged)

    exact = [r for r in b4_rows if r["agreement"]]
    print(f"B4: {len(exact)} exact-agreement rows out of {len(b4_rows)} "
          f"({100*len(exact)/len(b4_rows):.1f}%)")

    _write_jsonl(bench / "B4_agreement_claude.jsonl", exact)
    print(f"  wrote {bench / 'B4_agreement_claude.jsonl'}")

    b5 = [dict(r, is_verifiable=True) for r in exact if _is_verifiable(r)]
    print(f"B5: {len(b5)} verifiable rows from B4 ({100*len(b5)/max(len(exact),1):.1f}%)")
    _write_jsonl(bench / "B5_verifiable_claude.jsonl", b5)
    print(f"  wrote {bench / 'B5_verifiable_claude.jsonl'}")

    # Summary
    type_dist_b4 = Counter(r.get("claim_type", "other") for r in exact)
    type_dist_b5 = Counter(r.get("claim_type", "other") for r in b5)
    label_pairs = Counter((r["b2_label"], r["b3_label"]) for r in b4_rows)
    print("\nLabel pair distribution (Claude B2 vs B3):")
    for (a, b), n in sorted(label_pairs.items(), key=lambda kv: -kv[1]):
        print(f"  {a:25s} | {b:25s} | {n}")
    print(f"\nB4 claim_type distribution: {dict(type_dist_b4)}")
    print(f"B5 claim_type distribution: {dict(type_dist_b5)}")


if __name__ == "__main__":
    main()
