"""Re-label B2 / B3 records with Claude Sonnet 4.6 using the exact same prompts.

Inputs (read-only):
  data/benchmark/B2_reviewer_author.jsonl  (800 rows, has combined_author_response)
  data/benchmark/B3_reviewer_paper.jsonl   (800 rows, paper content reloaded from markdown)
  data/markdown/<paper_id>.md              (only needed for B3)

Outputs:
  data/benchmark/B2_reviewer_author_claude.jsonl
  data/benchmark/B3_reviewer_paper_claude.jsonl

Each output row preserves every original field (so the gpt-5-mini label stays
in `label`, `justification`, `evidence`) and adds:
  claude_label, claude_justification, claude_evidence, claude_section (B3 only),
  claude_model, claude_labeling_timestamp
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import anthropic

CLAUDE_MODEL = "claude-sonnet-4-6"
VALID_LABELS = ["Supported", "Partially Supported", "Contradicted", "Not Determined"]
PAPER_CONTENT_TRUNCATE = 15000  # matches reviewer_benchmark.py:219
MARKDOWN_TRUNCATE_FOR_B3 = 30000  # matches reviewer_benchmark.py:84


# ---------- prompt builders (verbatim from reviewer_benchmark.py) ----------

def build_b2_user_prompt(claim: str, author_response: str) -> str:
    return f"""
You are an expert at analyzing academic discourse between reviewers and authors.

Given a reviewer's claim and the author's response, determine how the authors address this claim.

Guidelines:
- "Supported": Authors clearly AGREE with the claim, acknowledge it as valid, or confirm what the reviewer stated
- "Partially Supported": Authors partially address the claim, acknowledge some validity but not fully
- "Contradicted": Authors explicitly DISAGREE with the claim, provide counter-evidence, or refute it
- "Not Determined": Authors don't address this specific claim at all, or it's unclear

Reviewer Claim: {claim}

Author's Response:
{author_response}

Return a JSON object:
{{"label": "Supported" or "Partially Supported" or "Contradicted" or "Not Determined", "justification": "brief explanation", "evidence": "exact quote from response supporting this label"}}
"""


def build_b3_user_prompt(claim: str, paper_content: str) -> str:
    return f"""
You are an expert at analyzing academic papers and determining if reviewer claims are true.

Given a reviewer's claim about a paper and the paper content, determine if the claim is TRUE according to the paper.

Important: You are evaluating whether the claim accurately describes what is in the paper.

Guidelines:
- "Supported": The claim is TRUE according to the paper. The paper confirms what the claim states.
- "Partially Supported": The claim is partially true, some aspects are correct but others are not.
- "Contradicted": The claim is FALSE according to the paper. The paper contradicts the claim.
- "Not Determined": The paper doesn't address this topic, or there's insufficient information.

Example: If claim says "the paper lacks comparison with baseline X" and paper indeed doesn't compare with X, the label is "Supported" (the claim is true).

Reviewer Claim: {claim}

Paper Content:
{paper_content[:PAPER_CONTENT_TRUNCATE]}

Return a JSON object:
{{"label": "Supported" or "Partially Supported" or "Contradicted" or "Not Determined", "justification": "brief explanation referencing specific sections", "evidence": "exact quote from paper", "section": "section name where evidence found"}}
"""


B2_SYSTEM = "You are an expert at analyzing academic discourse. Return only valid JSON."
B3_SYSTEM = "You are an expert at analyzing academic papers. Return only valid JSON."


# ---------- LLM call ----------

def _strip_code_fences(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```\s*$", "", content)
    return content


def _extract_json_object(content: str) -> Optional[Dict[str, Any]]:
    content = _strip_code_fences(content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Last-resort: grab the first {...} block.
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        return None


def call_claude(client: anthropic.Anthropic, system: str, user: str,
                max_retries: int = 4) -> Optional[Dict[str, Any]]:
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1024,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = resp.content[0].text if resp.content else ""
            return _extract_json_object(text)
        except anthropic.APIStatusError as e:
            if attempt == max_retries - 1:
                raise
            wait = min(2 ** attempt, 30)
            print(f"    APIStatusError {e.status_code}; retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
        except (anthropic.APIConnectionError, anthropic.RateLimitError) as e:
            if attempt == max_retries - 1:
                raise
            wait = min(2 ** attempt, 30)
            print(f"    {type(e).__name__}; retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
    return None


# ---------- per-row labeling ----------

def label_b2(client: anthropic.Anthropic, record: Dict[str, Any]) -> Dict[str, Any]:
    claim = record.get("claim", "")
    author_response = record.get("combined_author_response") or record.get("author_response") or ""
    if not author_response or len(author_response.strip()) < 20:
        return {"label": "Not Determined",
                "justification": "No author response available",
                "evidence": ""}
    result = call_claude(client, B2_SYSTEM, build_b2_user_prompt(claim, author_response))
    if not result:
        return {"label": "Not Determined", "justification": "JSON parse failed", "evidence": ""}
    label = result.get("label", "Not Determined")
    if label not in VALID_LABELS:
        label = "Not Determined"
    return {"label": label,
            "justification": result.get("justification", ""),
            "evidence": result.get("evidence", "")}


def label_b3(client: anthropic.Anthropic, record: Dict[str, Any],
             paper_content: str) -> Dict[str, Any]:
    claim = record.get("claim", "")
    if not paper_content or len(paper_content.strip()) < 100:
        return {"label": "Not Determined",
                "justification": "No paper content available",
                "evidence": "", "section": ""}
    result = call_claude(client, B3_SYSTEM, build_b3_user_prompt(claim, paper_content))
    if not result:
        return {"label": "Not Determined", "justification": "JSON parse failed",
                "evidence": "", "section": ""}
    label = result.get("label", "Not Determined")
    if label not in VALID_LABELS:
        label = "Not Determined"
    return {"label": label,
            "justification": result.get("justification", ""),
            "evidence": result.get("evidence", ""),
            "section": result.get("section", "")}


# ---------- file I/O ----------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_paper_markdown(markdown_dir: Path, paper_id: str) -> Optional[str]:
    md = markdown_dir / f"{paper_id}.md"
    if not md.exists():
        return None
    text = md.read_text(encoding="utf-8")
    if len(text) > MARKDOWN_TRUNCATE_FOR_B3:
        text = text[:MARKDOWN_TRUNCATE_FOR_B3] + "\n\n[Content truncated...]"
    return text


# ---------- runners ----------

def run_b2(in_path: Path, out_path: Path, max_workers: int, limit: Optional[int]) -> None:
    rows = read_jsonl(in_path)
    if limit:
        rows = rows[:limit]
    print(f"[B2] {len(rows)} records to relabel")

    client = anthropic.Anthropic()
    out: List[Optional[Dict[str, Any]]] = [None] * len(rows)
    lock = Lock()
    done = {"n": 0}

    def worker(i: int, rec: Dict[str, Any]) -> None:
        try:
            res = label_b2(client, rec)
        except Exception as e:
            res = {"label": "Not Determined",
                   "justification": f"Error: {e}", "evidence": ""}
        merged = dict(rec)
        merged["claude_label"] = res["label"]
        merged["claude_justification"] = res["justification"]
        merged["claude_evidence"] = res["evidence"]
        merged["claude_model"] = CLAUDE_MODEL
        merged["claude_labeling_timestamp"] = datetime.now().isoformat()
        out[i] = merged
        with lock:
            done["n"] += 1
            if done["n"] % 25 == 0 or done["n"] == len(rows):
                print(f"  [B2] {done['n']}/{len(rows)}")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(worker, i, r) for i, r in enumerate(rows)]
        for _ in as_completed(futs):
            pass

    write_jsonl(out_path, [r for r in out if r is not None])
    print(f"[B2] wrote {out_path}")


def run_b3(in_path: Path, out_path: Path, markdown_dir: Path,
           max_workers: int, limit: Optional[int]) -> None:
    rows = read_jsonl(in_path)
    if limit:
        rows = rows[:limit]
    print(f"[B3] {len(rows)} records to relabel")

    # Cache markdown per paper_id
    paper_cache: Dict[str, Optional[str]] = {}
    cache_lock = Lock()

    def get_paper(pid: str) -> Optional[str]:
        with cache_lock:
            if pid in paper_cache:
                return paper_cache[pid]
        text = load_paper_markdown(markdown_dir, pid)
        with cache_lock:
            paper_cache[pid] = text
        return text

    missing = []
    for r in rows:
        pid = r.get("paper_id", "")
        if not (markdown_dir / f"{pid}.md").exists():
            missing.append(pid)
    if missing:
        uniq = sorted(set(missing))
        print(f"[B3] WARNING: {len(uniq)} unique paper_ids have no markdown "
              f"({len(missing)} rows will fall through to Not Determined)")
        print(f"  example missing: {uniq[:5]}")

    client = anthropic.Anthropic()
    out: List[Optional[Dict[str, Any]]] = [None] * len(rows)
    lock = Lock()
    done = {"n": 0}

    def worker(i: int, rec: Dict[str, Any]) -> None:
        pid = rec.get("paper_id", "")
        paper_md = get_paper(pid) or ""
        try:
            res = label_b3(client, rec, paper_md)
        except Exception as e:
            res = {"label": "Not Determined",
                   "justification": f"Error: {e}",
                   "evidence": "", "section": ""}
        merged = dict(rec)
        merged["claude_label"] = res["label"]
        merged["claude_justification"] = res["justification"]
        merged["claude_evidence"] = res["evidence"]
        merged["claude_section"] = res["section"]
        merged["claude_model"] = CLAUDE_MODEL
        merged["claude_labeling_timestamp"] = datetime.now().isoformat()
        out[i] = merged
        with lock:
            done["n"] += 1
            if done["n"] % 25 == 0 or done["n"] == len(rows):
                print(f"  [B3] {done['n']}/{len(rows)}")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(worker, i, r) for i, r in enumerate(rows)]
        for _ in as_completed(futs):
            pass

    write_jsonl(out_path, [r for r in out if r is not None])
    print(f"[B3] wrote {out_path}")


# ---------- CLI ----------

def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    bench = repo / "data" / "benchmark"
    md_dir = repo / "data" / "markdown"

    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["b2", "b3"], required=True)
    p.add_argument("--in-file", type=Path, default=None)
    p.add_argument("--out-file", type=Path, default=None)
    p.add_argument("--markdown-dir", type=Path, default=md_dir)
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None,
                   help="Process only first N rows (for smoke test)")
    args = p.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set in environment", file=sys.stderr)
        sys.exit(1)

    if args.mode == "b2":
        in_path = args.in_file or bench / "B2_reviewer_author.jsonl"
        out_path = args.out_file or bench / "B2_reviewer_author_claude.jsonl"
        run_b2(in_path, out_path, args.max_workers, args.limit)
    else:
        in_path = args.in_file or bench / "B3_reviewer_paper.jsonl"
        out_path = args.out_file or bench / "B3_reviewer_paper_claude.jsonl"
        run_b3(in_path, out_path, args.markdown_dir, args.max_workers, args.limit)


if __name__ == "__main__":
    main()
