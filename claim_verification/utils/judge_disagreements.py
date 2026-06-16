"""For every B2/B3 record where gpt-5-mini and claude-sonnet-4-6 disagree,
ask claude-opus-4-7 to act as a judge and pick the correct label
(or assign a new one if neither candidate is right).

Reads:
  data/benchmark_claude/B2_reviewer_author_claude.jsonl
  data/benchmark_claude/B3_reviewer_paper_claude.jsonl
  data/markdown/<paper_id>.md  (B3 only)

Writes:
  data/benchmark_claude/B2_reviewer_author_judged.jsonl
  data/benchmark_claude/B3_reviewer_paper_judged.jsonl
  data/benchmark_claude/judge_report.md
  data/benchmark_claude/judge_report.json

Each output row keeps every original field plus:
  judge_label              the final correct label per Opus
  judge_winner             'gpt5mini' | 'sonnet46' | 'neither'
  judge_reasoning          one-paragraph explanation
  judge_model              'claude-opus-4-7'
  judge_timestamp          ISO timestamp

For agreement rows (no disagreement to resolve) judge_label == both labels and
winner is 'agreement' (no Opus call made).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import anthropic

JUDGE_MODEL = "claude-opus-4-7"
VALID_LABELS = ["Supported", "Partially Supported", "Contradicted", "Not Determined"]
PAPER_CONTENT_TRUNCATE = 15000
MARKDOWN_TRUNCATE = 30000


# ---------- prompt builders ----------

B2_DEFINITIONS = """\
Valid labels:
- "Supported": Authors clearly AGREE with the claim, acknowledge it as valid, or confirm what the reviewer stated
- "Partially Supported": Authors partially address the claim, acknowledge some validity but not fully
- "Contradicted": Authors explicitly DISAGREE with the claim, provide counter-evidence, or refute it
- "Not Determined": Authors don't address this specific claim at all, or it's unclear"""

B3_DEFINITIONS = """\
Valid labels:
- "Supported": The claim is TRUE according to the paper. The paper confirms what the claim states.
- "Partially Supported": The claim is partially true, some aspects are correct but others are not.
- "Contradicted": The claim is FALSE according to the paper. The paper contradicts the claim.
- "Not Determined": The paper doesn't address this topic, or there's insufficient information."""


def build_judge_prompt(mode: str, claim: str, source_text: str,
                       cand_a_label: str, cand_a_just: str,
                       cand_b_label: str, cand_b_just: str,
                       source_label: str) -> str:
    defs = B2_DEFINITIONS if mode == "b2" else B3_DEFINITIONS
    if mode == "b2":
        task = ("Determine how the authors address this reviewer's claim "
                "in their response.")
    else:
        task = ("Determine whether this reviewer's claim is TRUE according "
                "to the paper content.")
    return f"""\
You are an expert tie-breaker for an academic peer-review claim verification task.

{task}

Two AI assistants have already produced candidate labels for this record. They
disagree. Read the source material yourself and decide which (if either) is
correct. The right answer may match Candidate 1, Candidate 2, or NEITHER (in
which case give the new correct label).

{defs}

Reviewer Claim:
{claim}

{source_label}:
{source_text}

Candidate 1 said: "{cand_a_label}"
  Justification: {cand_a_just}

Candidate 2 said: "{cand_b_label}"
  Justification: {cand_b_just}

Return ONLY a JSON object with these keys:
- "judge_label": one of "Supported", "Partially Supported", "Contradicted", "Not Determined"
- "winner": "candidate_1" if you agree with Candidate 1, "candidate_2" if you agree with Candidate 2, or "neither" if both are wrong
- "reasoning": 2-3 sentences explaining your choice, citing specific evidence from the source

Reply with the JSON object only — no preamble, no markdown fences.
"""


JUDGE_SYSTEM = ("You are an expert academic peer-review analyst acting as a "
                "neutral judge between two AI candidates. Return only valid JSON.")


# ---------- LLM call ----------

def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    return s


def _parse_json(s: str) -> Optional[Dict[str, Any]]:
    s = _strip_fences(s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def call_judge(client: anthropic.Anthropic, user_prompt: str,
               max_retries: int = 4) -> Optional[Dict[str, Any]]:
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=1024,
                system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = resp.content[0].text if resp.content else ""
            return _parse_json(text)
        except (anthropic.APIStatusError, anthropic.APIConnectionError,
                anthropic.RateLimitError) as e:
            if attempt == max_retries - 1:
                raise
            wait = min(2 ** attempt, 30)
            print(f"    {type(e).__name__}; retry in {wait}s",
                  file=sys.stderr, flush=True)
            time.sleep(wait)
    return None


# ---------- file I/O ----------

def read_jsonl(p: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def write_jsonl(p: Path, rows: List[Dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_paper_md(md_dir: Path, paper_id: str) -> Optional[str]:
    p = md_dir / f"{paper_id}.md"
    if not p.exists():
        return None
    text = p.read_text(encoding="utf-8")
    if len(text) > MARKDOWN_TRUNCATE:
        text = text[:MARKDOWN_TRUNCATE] + "\n\n[Content truncated...]"
    return text


# ---------- judging core ----------

def _build_inputs_for_b2(rec: Dict[str, Any]) -> Tuple[str, str]:
    claim = rec.get("claim", "")
    src = (rec.get("combined_author_response") or rec.get("author_response") or "").strip()
    return claim, src


def _build_inputs_for_b3(rec: Dict[str, Any], md_dir: Path) -> Tuple[str, str]:
    claim = rec.get("claim", "")
    md = load_paper_md(md_dir, rec.get("paper_id", "")) or ""
    src = md[:PAPER_CONTENT_TRUNCATE]
    return claim, src


def _candidates_for(rec: Dict[str, Any], rng: random.Random) -> Tuple[
        Tuple[str, str, str], Tuple[str, str, str], str, str]:
    """Return ((labelA, justA, owner_a), (labelB, justB, owner_b), owner_for_cand1, owner_for_cand2).

    owner is 'gpt5mini' or 'sonnet46'. Order is randomised per record to avoid
    position bias in the judge.
    """
    gpt = (rec.get("label", ""), rec.get("justification", "") or "", "gpt5mini")
    cla = (rec.get("claude_label", ""), rec.get("claude_justification", "") or "", "sonnet46")
    pair = [gpt, cla]
    rng.shuffle(pair)
    return pair[0], pair[1], pair[0][2], pair[1][2]


def judge_record(client: anthropic.Anthropic, mode: str, rec: Dict[str, Any],
                 source_text: str, source_label: str,
                 rng: random.Random) -> Dict[str, Any]:
    cand1, cand2, owner1, owner2 = _candidates_for(rec, rng)
    prompt = build_judge_prompt(
        mode=mode,
        claim=rec.get("claim", ""),
        source_text=source_text,
        cand_a_label=cand1[0], cand_a_just=cand1[1],
        cand_b_label=cand2[0], cand_b_just=cand2[1],
        source_label=source_label,
    )
    result = call_judge(client, prompt)
    if not result:
        return {"judge_label": "Not Determined",
                "judge_winner": "error",
                "judge_reasoning": "JSON parse failed"}

    label = result.get("judge_label", "")
    if label not in VALID_LABELS:
        label = "Not Determined"
    winner_field = (result.get("winner") or "").strip().lower()
    if winner_field == "candidate_1":
        winner = owner1
    elif winner_field == "candidate_2":
        winner = owner2
    elif winner_field == "neither":
        winner = "neither"
    else:
        # Best-effort: infer from label match
        if label == rec.get("label") and label != rec.get("claude_label"):
            winner = "gpt5mini"
        elif label == rec.get("claude_label") and label != rec.get("label"):
            winner = "sonnet46"
        else:
            winner = "neither"
    return {"judge_label": label,
            "judge_winner": winner,
            "judge_reasoning": result.get("reasoning", "")}


# ---------- runners ----------

def run(mode: str, in_path: Path, out_path: Path, md_dir: Optional[Path],
        max_workers: int, limit: Optional[int]) -> Dict[str, Any]:
    rows = read_jsonl(in_path)
    if limit:
        rows = rows[:limit]
    print(f"[{mode.upper()}] {len(rows)} total records")

    disagreements = [(i, r) for i, r in enumerate(rows)
                     if r.get("label") != r.get("claude_label")]
    print(f"[{mode.upper()}] {len(disagreements)} disagreements to judge")

    client = anthropic.Anthropic()
    out: List[Dict[str, Any]] = [dict(r) for r in rows]
    lock = Lock()
    counter = {"done": 0, "errors": 0}
    base_rng = random.Random(42)

    def worker(i: int, rec: Dict[str, Any]) -> None:
        # Per-thread RNG seeded from base+i for reproducibility
        rng = random.Random(base_rng.randrange(0, 2**31) ^ i)
        if mode == "b2":
            claim, src = _build_inputs_for_b2(rec)
            source_label = "Author's Response"
        else:
            claim, src = _build_inputs_for_b3(rec, md_dir)
            source_label = "Paper Content"
        if not src or len(src.strip()) < 20:
            judge_out = {"judge_label": "Not Determined",
                         "judge_winner": "error",
                         "judge_reasoning": "Empty source material"}
        else:
            try:
                judge_out = judge_record(client, mode, rec, src, source_label, rng)
            except Exception as e:
                judge_out = {"judge_label": "Not Determined",
                             "judge_winner": "error",
                             "judge_reasoning": f"{type(e).__name__}: {e}"}
        with lock:
            for k, v in judge_out.items():
                out[i][k] = v
            out[i]["judge_model"] = JUDGE_MODEL
            out[i]["judge_timestamp"] = datetime.now().isoformat()
            counter["done"] += 1
            if judge_out["judge_winner"] == "error":
                counter["errors"] += 1
            if counter["done"] % 25 == 0 or counter["done"] == len(disagreements):
                print(f"  [{mode.upper()}] {counter['done']}/{len(disagreements)} judged"
                      f" ({counter['errors']} errors)", flush=True)

    # Mark agreement rows with judge_winner='agreement', judge_label = the agreed label
    for i, r in enumerate(rows):
        if r.get("label") == r.get("claude_label"):
            out[i]["judge_label"] = r.get("label")
            out[i]["judge_winner"] = "agreement"
            out[i]["judge_reasoning"] = "Both models agreed; no Opus call made."
            out[i]["judge_model"] = JUDGE_MODEL
            out[i]["judge_timestamp"] = datetime.now().isoformat()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(worker, i, rec) for i, rec in disagreements]
        for _ in as_completed(futs):
            pass

    write_jsonl(out_path, out)
    print(f"[{mode.upper()}] wrote {out_path}")

    # Stats over disagreement rows only
    judged = [out[i] for i, _ in disagreements]
    from collections import Counter
    win_counts = Counter(r["judge_winner"] for r in judged)
    return {
        "mode": mode,
        "total_records": len(rows),
        "disagreements_judged": len(disagreements),
        "agreement_rows": len(rows) - len(disagreements),
        "winner_counts": dict(win_counts),
        "errors": counter["errors"],
    }


# ---------- report ----------

def write_report(b2: Dict[str, Any], b3: Optional[Dict[str, Any]],
                 out_dir: Path) -> None:
    lines = [
        "# Opus 4.7 judge — gpt-5-mini vs claude-sonnet-4-6 disagreements",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "For every record where the two labelers disagreed, claude-opus-4-7",
        "was given the claim, the source material, and both candidate labels",
        "(anonymised + order-randomised) and asked to pick the correct label",
        "or assign a new one.",
        "",
    ]
    for name, s in [("B2 (claim vs author response)", b2),
                    ("B3 (claim vs paper content)", b3)]:
        if s is None:
            continue
        wc = s["winner_counts"]
        n = s["disagreements_judged"]
        lines += [
            f"## {name}",
            "",
            f"- Total records: {s['total_records']}",
            f"- Both models agreed (no judge needed): {s['agreement_rows']}",
            f"- Disagreements judged by Opus: **{n}**",
            f"- Errors: {s['errors']}",
            "",
            "### Judge verdicts on the disagreements",
            "",
            "| verdict | count | rate |",
            "|---|---:|---:|",
        ]
        for label, cnt in sorted(wc.items(), key=lambda kv: -kv[1]):
            pct = 100 * cnt / n if n else 0
            display = {"gpt5mini": "gpt-5-mini was correct",
                       "sonnet46": "claude-sonnet-4-6 was correct",
                       "neither": "neither — judge gave a different label",
                       "error": "error / no result"}.get(label, label)
            lines.append(f"| {display} | {cnt} | {pct:.1f}% |")
        lines.append("")
    out_md = out_dir / "judge_report.md"
    out_md.write_text("\n".join(lines))
    print(f"  wrote {out_md}")
    out_json = out_dir / "judge_report.json"
    out_json.write_text(json.dumps({
        "generated": datetime.now().isoformat(),
        "judge_model": JUDGE_MODEL,
        "B2": b2,
        "B3": b3,
    }, indent=2))
    print(f"  wrote {out_json}")


# ---------- CLI ----------

def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    bench_claude = repo / "data" / "benchmark_claude"
    md_dir = repo / "data" / "markdown"

    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["b2", "b3", "both"], default="both")
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N rows (smoke test)")
    args = p.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    b2_summary = b3_summary = None

    if args.mode in ("b2", "both"):
        b2_summary = run(
            "b2",
            bench_claude / "B2_reviewer_author_claude.jsonl",
            bench_claude / "B2_reviewer_author_judged.jsonl",
            md_dir=None,
            max_workers=args.max_workers,
            limit=args.limit,
        )

    if args.mode in ("b3", "both"):
        b3_summary = run(
            "b3",
            bench_claude / "B3_reviewer_paper_claude.jsonl",
            bench_claude / "B3_reviewer_paper_judged.jsonl",
            md_dir=md_dir,
            max_workers=args.max_workers,
            limit=args.limit,
        )

    write_report(b2_summary, b3_summary, bench_claude)


if __name__ == "__main__":
    main()
