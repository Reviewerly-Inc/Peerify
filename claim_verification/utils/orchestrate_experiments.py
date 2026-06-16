"""Run one experiment cell.

Usage:
  python -m claim_verification.utils.orchestrate_experiments \\
      --mode {full_paper,rag} \\
      --benchmark B1|B2|B3|B4|B5|B6 \\
      --model <model_id> \\
      [--retriever bm25|bm25-cross|faiss|biencoder-crossencoder] \\
      [--top-k 10] [--max-entries N] \\
      --output-root data/experiments_v2

Wraps the existing eval scripts so we have a single entrypoint with consistent
output naming. Output goes to:
  data/experiments_v2/<mode>/<benchmark>/<model_safe>/<retriever_or_full>/{results.jsonl,metrics.json}
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _safe_name(s: str) -> str:
    return s.replace("/", "__").replace(":", "_")


def run_full_paper(benchmark: str, model: str, output_dir: Path,
                   max_entries: int | None, num_workers: int) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect provider prefix
    if model.startswith("claude-"):
        spec = f"claude:{model}"
    elif model.startswith("gemini-"):
        spec = f"gemini:{model}"
    elif "/" in model:
        spec = f"openrouter:{model}"
    else:
        spec = f"openai:{model}"

    cmd = [sys.executable, "-m", "claim_verification.evaluation.full_paper_evaluation",
           "--benchmark", benchmark,
           "--models", spec,
           "--output-dir", str(output_dir),
           "--num-workers", str(num_workers)]
    if max_entries:
        cmd += ["--max-entries", str(max_entries)]
    return subprocess.call(cmd, cwd=str(REPO))


def run_rag(benchmark: str, model: str, retriever: str, top_k: int,
            output_dir: Path, max_entries: int | None) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "claim_verification.evaluation.evaluate_retrieval_metrics",
           "--benchmark", benchmark,
           "--retriever", retriever,
           "--llm-model", model,
           "--top-k", str(top_k),
           "--output-dir", str(output_dir)]
    if max_entries:
        cmd += ["--max-entries", str(max_entries)]
    return subprocess.call(cmd, cwd=str(REPO))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["full_paper", "rag"], required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--retriever", default=None)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--max-entries", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--output-root", default=str(REPO / "data" / "experiments_v2"))
    args = p.parse_args()

    out = Path(args.output_root) / args.mode / args.benchmark / _safe_name(args.model)
    if args.mode == "rag":
        if not args.retriever:
            print("--retriever required for RAG mode", file=sys.stderr)
            sys.exit(2)
        out = out / args.retriever
        rc = run_rag(args.benchmark, args.model, args.retriever, args.top_k,
                     out, args.max_entries)
    else:
        out = out / "full"
        rc = run_full_paper(args.benchmark, args.model, out,
                            args.max_entries, args.num_workers)
    sys.exit(rc)


if __name__ == "__main__":
    main()
