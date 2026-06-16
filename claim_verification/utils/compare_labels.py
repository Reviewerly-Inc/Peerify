"""Compare gpt-5-mini labels (original) with claude-sonnet-4-6 labels
(re-labeled) on B2 and B3.

Reads:
  data/benchmark/B2_reviewer_author_claude.jsonl
  data/benchmark/B3_reviewer_paper_claude.jsonl

Writes:
  data/benchmark/label_comparison_report.json   (machine-readable)
  data/benchmark/label_comparison_report.md     (human-readable)

For each benchmark, computes:
  - Exact agreement count + rate
  - Label distribution per model
  - Confusion matrix (rows = gpt-5-mini, cols = claude)
  - Per-decision agreement (Reject/Poster/Oral/Spotlight)
  - Per-claim-type agreement
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

LABELS = ["Supported", "Partially Supported", "Contradicted", "Not Determined"]


def _read_jsonl(p: Path) -> List[Dict[str, Any]]:
    rows = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _confusion(rows: List[Dict[str, Any]]) -> List[List[int]]:
    """rows = gpt5mini, cols = claude."""
    m = [[0 for _ in LABELS] for _ in LABELS]
    for r in rows:
        try:
            i = LABELS.index(r["label"])
            j = LABELS.index(r["claude_label"])
            m[i][j] += 1
        except ValueError:
            continue
    return m


def _agreement(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    agree = sum(1 for r in rows if r["label"] == r["claude_label"])
    gpt = Counter(r["label"] for r in rows)
    cla = Counter(r["claude_label"] for r in rows)
    by_decision = defaultdict(lambda: [0, 0])  # [agree, total]
    for r in rows:
        d = r.get("decision", "Unknown")
        by_decision[d][1] += 1
        if r["label"] == r["claude_label"]:
            by_decision[d][0] += 1
    by_type = defaultdict(lambda: [0, 0])
    for r in rows:
        t = r.get("claim_type", "other")
        by_type[t][1] += 1
        if r["label"] == r["claude_label"]:
            by_type[t][0] += 1
    return {
        "n": n,
        "agree": agree,
        "rate": agree / n if n else 0.0,
        "dist_gpt5mini": dict(gpt),
        "dist_sonnet46": dict(cla),
        "confusion": {  # nested dict for readability
            gpt_label: {claude_label: _confusion(rows)[i][j]
                        for j, claude_label in enumerate(LABELS)}
            for i, gpt_label in enumerate(LABELS)
        },
        "agreement_by_decision": {d: {"agree": v[0], "total": v[1],
                                      "rate": v[0]/v[1] if v[1] else 0.0}
                                  for d, v in by_decision.items()},
        "agreement_by_type": {t: {"agree": v[0], "total": v[1],
                                  "rate": v[0]/v[1] if v[1] else 0.0}
                              for t, v in by_type.items()},
    }


def _format_md(b2: Dict[str, Any], b3: Dict[str, Any]) -> str:
    lines = [
        "# gpt-5-mini vs claude-sonnet-4-6 — label comparison",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "Same 800 reviewer claims, same 84 papers, same prompts. Only the labeling model changed.",
        "",
    ]

    for name, r in [("B2 (claim vs author response)", b2),
                    ("B3 (claim vs paper content)", b3)]:
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"- N = {r['n']}")
        lines.append(f"- Exact agreement: **{r['agree']}/{r['n']} ({100*r['rate']:.1f}%)**")
        lines.append("")
        lines.append("### Label distribution")
        lines.append("| label | gpt-5-mini | claude-sonnet-4-6 |")
        lines.append("|---|---:|---:|")
        for L in LABELS:
            lines.append(f"| {L} | {r['dist_gpt5mini'].get(L, 0)} | {r['dist_sonnet46'].get(L, 0)} |")
        lines.append("")
        lines.append("### Confusion matrix (rows = gpt-5-mini, cols = claude-sonnet-4-6)")
        lines.append("| gpt5 \\ claude | " + " | ".join(LABELS) + " | total |")
        lines.append("|---" * (len(LABELS) + 2) + "|")
        for L_row in LABELS:
            row = r["confusion"][L_row]
            tot = sum(row.values())
            cells = [str(row[L_col]) for L_col in LABELS]
            lines.append(f"| **{L_row}** | " + " | ".join(cells) + f" | {tot} |")
        lines.append("")
        lines.append("### Agreement by decision")
        lines.append("| decision | agree | total | rate |")
        lines.append("|---|---:|---:|---:|")
        for d, v in sorted(r["agreement_by_decision"].items()):
            lines.append(f"| {d} | {v['agree']} | {v['total']} | {100*v['rate']:.1f}% |")
        lines.append("")
        lines.append("### Agreement by claim_type")
        lines.append("| claim_type | agree | total | rate |")
        lines.append("|---|---:|---:|---:|")
        for t, v in sorted(r["agreement_by_type"].items(), key=lambda kv: -kv[1]["total"]):
            lines.append(f"| {t} | {v['agree']} | {v['total']} | {100*v['rate']:.1f}% |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    bench = repo / "data" / "benchmark"

    b2 = _read_jsonl(bench / "B2_reviewer_author_claude.jsonl")
    b3_path = bench / "B3_reviewer_paper_claude.jsonl"
    b3 = _read_jsonl(b3_path) if b3_path.exists() else []

    out = {
        "generated": datetime.now().isoformat(),
        "B2": _agreement(b2),
        "B3": _agreement(b3) if b3 else None,
    }
    (bench / "label_comparison_report.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    (bench / "label_comparison_report.md").write_text(
        _format_md(out["B2"], out["B3"] or {"n": 0, "agree": 0, "rate": 0.0,
                                            "dist_gpt5mini": {}, "dist_sonnet46": {},
                                            "confusion": {L: {L2: 0 for L2 in LABELS} for L in LABELS},
                                            "agreement_by_decision": {},
                                            "agreement_by_type": {}}))
    print(f"Wrote {bench / 'label_comparison_report.md'}")
    print(f"Wrote {bench / 'label_comparison_report.json'}")
    print()
    print(f"B2: {out['B2']['agree']}/{out['B2']['n']} agree ({100*out['B2']['rate']:.1f}%)")
    if out["B3"]:
        print(f"B3: {out['B3']['agree']}/{out['B3']['n']} agree ({100*out['B3']['rate']:.1f}%)")
    else:
        print("B3: not yet relabeled")


if __name__ == "__main__":
    main()
