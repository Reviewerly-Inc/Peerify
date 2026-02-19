

import os
import json
import argparse
import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import subprocess


from claim_verification.config import DataStructure


# All available retrievers
ALL_RETRIEVERS = [
    'bm25',
    'faiss',
    'biencoder-crossencoder',
    'bm25-cross',
]

# All benchmarks
ALL_BENCHMARKS = ['B1', 'B2', 'B3', 'B4', 'B5', "B6"]


def run_evaluation(
    benchmark: str,
    retriever: str,
    llm_model: str,
    top_k: int,
    max_entries: Optional[int] = None,
    script_type: str = 'retrieval_metrics',
    dry_run: bool = False
) -> dict:
    """
    Run evaluation for a single benchmark + retriever combination.
    
    Args:
        benchmark: Benchmark name (B1, B2, etc.)
        retriever: Retrieval method name
        llm_model: LLM model name
        top_k: Number of chunks to retrieve
        max_entries: Maximum number of entries to process (None = all)
        script_type: Type of evaluation script ('retrieval_metrics' or 'real_evaluation')
        dry_run: If True, only count tokens without sending requests to LLM
        
    Returns:
        Dictionary with execution result
    """
    if script_type == 'retrieval_metrics':
        script = 'evaluate_retrieval_metrics'
    else:
        script = 'retrieval_evaluation'
    
    cmd = [
        sys.executable, '-m', f'claim_verification.evaluation.{script}',
        '--benchmark', benchmark,
        '--retriever', retriever,
        '--llm-model', llm_model,
        '--top-k', str(top_k),
    ]
    
    if max_entries is not None:
        cmd.extend(['--max-entries', str(max_entries)])
    
    if dry_run:
        cmd.extend(['--dry-run'])
    
    print(f"\n{'='*80}")
    print(f"Running: {benchmark} with {retriever} (top-{top_k})")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        
        if result.returncode == 0:
            print(f"✓ Success: {benchmark} + {retriever}")
            return {
                'benchmark': benchmark,
                'retriever': retriever,
                'status': 'success',
                'stdout': result.stdout,
                'stderr': result.stderr,
            }
        else:
            print(f"✗ Failed: {benchmark} + {retriever}")
            print(f"Error: {result.stderr[:500]}")
            return {
                'benchmark': benchmark,
                'retriever': retriever,
                'status': 'failed',
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
            }
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout: {benchmark} + {retriever}")
        return {
            'benchmark': benchmark,
            'retriever': retriever,
            'status': 'timeout',
        }
    except Exception as e:
        print(f"✗ Error: {benchmark} + {retriever}: {e}")
        return {
            'benchmark': benchmark,
            'retriever': retriever,
            'status': 'error',
            'error': str(e),
        }


def run_all_evaluations(
    benchmarks: List[str],
    retrievers: List[str],
    llm_model: str,
    top_k: int,
    max_entries: Optional[int] = None,
    script_type: str = 'retrieval_metrics',
    output_summary: Optional[str] = None,
    dry_run: bool = False
) -> dict:
    """
    Run evaluation for all benchmark + retriever combinations.
    
    Args:
        benchmarks: List of benchmark names (B1, B2, etc.)
        retrievers: List of retrieval method names
        llm_model: LLM model name
        top_k: Number of chunks to retrieve
        max_entries: Maximum number of entries to process (None = all)
        script_type: Type of evaluation script ('retrieval_metrics' or 'real_evaluation')
        output_summary: Path to save summary JSON (optional)
        dry_run: If True, only count tokens without sending requests to LLM
        
    Returns:
        Dictionary with summary of all runs
    """
    data_structure = DataStructure()
    results = []
    successful = []
    failed = []
    
    total_combinations = len(benchmarks) * len(retrievers)
    current = 0
    
    print(f"\n{'='*80}")
    print(f"RUNNING ALL EVALUATIONS" + (" (DRY RUN - Token Counting Only)" if dry_run else ""))
    print(f"{'='*80}")
    print(f"Benchmarks: {', '.join(benchmarks)}")
    print(f"Retrievers: {', '.join(retrievers)}")
    print(f"LLM Model: {llm_model}")
    print(f"Top-K: {top_k}")
    print(f"Script Type: {script_type}")
    print(f"Total Combinations: {total_combinations}")
    if max_entries:
        print(f"Max Entries per Benchmark: {max_entries}")
    if dry_run:
        print(f"Dry Run Mode: ENABLED (only counting tokens, not sending LLM requests)")
    print(f"{'='*80}\n")
    
    start_time = datetime.now()
    
    for benchmark in benchmarks:
        for retriever in retrievers:
            current += 1
            print(f"\n[{current}/{total_combinations}] Processing {benchmark} + {retriever}...")
            
            result = run_evaluation(
                benchmark=benchmark,
                retriever=retriever,
                llm_model=llm_model,
                top_k=top_k,
                max_entries=max_entries,
                script_type=script_type,
                dry_run=dry_run
            )
            
            results.append(result)
            
            if result['status'] == 'success':
                successful.append(f"{benchmark} + {retriever}")
            else:
                failed.append(f"{benchmark} + {retriever}")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Create summary
    summary = {
        'timestamp': start_time.isoformat(),
        'duration_seconds': duration,
        'configuration': {
            'benchmarks': benchmarks,
            'retrievers': retrievers,
            'llm_model': llm_model,
            'top_k': top_k,
            'max_entries': max_entries,
            'script_type': script_type,
            'dry_run': dry_run,
        },
        'total_combinations': total_combinations,
        'successful': len(successful),
        'failed': len(failed),
        'successful_combinations': successful,
        'failed_combinations': failed,
        'results': results,
    }
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total Combinations: {total_combinations}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Duration: {duration/60:.2f} minutes")
    print(f"\nSuccessful Combinations:")
    for combo in successful:
        print(f"  ✓ {combo}")
    
    if failed:
        print(f"\nFailed Combinations:")
        for combo in failed:
            print(f"  ✗ {combo}")
    
    # Save summary if requested
    if output_summary:
        summary_file = Path(output_summary)
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSummary saved to: {summary_file}")
    else:
        # Save to default location
        data_structure = DataStructure()
        summary_file = data_structure.logs_dir / f"evaluation_summary_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSummary saved to: {summary_file}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Run evaluation with all retrievers across all benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks with all retrievers
  python3 -m claim_verification.evaluation.run_all_evaluations --llm-model gpt-4o-mini --top-k 10

  # Run specific benchmarks with specific retrievers
  python3 -m claim_verification.evaluation.run_all_evaluations --benchmarks B1 B2 --retrievers bm25 sbert

  # Test with limited entries
  python3 -m claim_verification.evaluation.run_all_evaluations --llm-model gpt-4o-mini --top-k 5 --max-entries 50

  # Use real_benchmark_evaluation instead of evaluate_retrieval_metrics
  python3 -m claim_verification.evaluation.run_all_evaluations --script-type real_evaluation
        """
    )
    
    parser.add_argument(
        '--benchmarks',
        nargs='+',
        default=ALL_BENCHMARKS,
        choices=ALL_BENCHMARKS,
        help=f'Benchmarks to evaluate (default: all: {", ".join(ALL_BENCHMARKS)})'
    )
    
    parser.add_argument(
        '--retrievers',
        nargs='+',
        default=ALL_RETRIEVERS,
        choices=ALL_RETRIEVERS,
        help=f'Retrievers to use (default: all: {", ".join(ALL_RETRIEVERS)})'
    )
    
    parser.add_argument(
        '--llm-model',
        default='gpt-4o-mini',
        help='LLM model name (default: gpt-4o-mini)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of chunks to retrieve (default: 10)'
    )
    
    parser.add_argument(
        '--max-entries',
        type=int,
        help='Maximum number of entries to process per benchmark (default: all)'
    )
    
    parser.add_argument(
        '--script-type',
        choices=['retrieval_metrics', 'real_evaluation'],
        default='retrieval_metrics',
        help='Type of evaluation script to use (default: retrieval_metrics)'
    )
    
    parser.add_argument(
        '--output-summary',
        help='Path to save summary JSON (default: auto-generated in data/logs/)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only count tokens without sending requests to LLM (for cost estimation)'
    )
    
    args = parser.parse_args()
    
    summary = run_all_evaluations(
        benchmarks=args.benchmarks,
        retrievers=args.retrievers,
        llm_model=args.llm_model,
        top_k=args.top_k,
        max_entries=args.max_entries,
        script_type=args.script_type,
        output_summary=args.output_summary,
        dry_run=args.dry_run
    )
    
    # Exit with error code if any failed
    if summary['failed'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
