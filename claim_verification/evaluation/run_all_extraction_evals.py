

import os
import json
import argparse
import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import subprocess


from claim_verification.config import DataStructure


# All supported benchmarks for claim extraction evaluation
ALL_BENCHMARKS = ['B2', 'B3']


def run_claim_extraction_evaluation(
    benchmark: str,
    llm_model: str,
    max_reviews: Optional[int] = None,
    num_workers: int = 10,
    skip_fenice: bool = True,
    skip_gemma: bool = True,
    skip_reference_free: bool = True,
    use_llm_matching: bool = True,
    cosine_threshold: float = 0.3
) -> dict:
    """
    Run claim extraction evaluation for a single benchmark.
    
    Args:
        benchmark: Benchmark name (B2, B3)
        llm_model: LLM model name
        max_reviews: Maximum number of reviews to process (None = all)
        num_workers: Number of parallel workers for API calls
        skip_fenice: Skip FENICE extraction
        skip_gemma: Skip Gemma extraction
        skip_reference_free: Skip reference-free metrics
        use_llm_matching: Use LLM for matching (if False, only cosine similarity)
        cosine_threshold: Threshold for cosine similarity matching
        
    Returns:
        Dictionary with execution result
    """
    cmd = [
        sys.executable, '-m', 'claim_verification.claim_extraction_evaluator',
        '--benchmark', benchmark,
        '--llm-model', llm_model,
        '--num-workers', str(num_workers),
        '--cosine-threshold', str(cosine_threshold),
    ]
    
    if max_reviews is not None:
        cmd.extend(['--max-reviews', str(max_reviews)])
    
    if skip_fenice:
        cmd.append('--skip-fenice')
    
    if skip_gemma:
        cmd.append('--skip-gemma')
    
    if skip_reference_free:
        cmd.append('--skip-reference-free')
    
    if not use_llm_matching:
        cmd.append('--no-vllm')
    
    print(f"\n{'='*80}")
    print(f"Running claim extraction evaluation: {benchmark}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout
        )
        
        if result.returncode == 0:
            print(f"✓ Success: {benchmark}")
            return {
                'benchmark': benchmark,
                'status': 'success',
                'stdout': result.stdout,
                'stderr': result.stderr,
            }
        else:
            print(f"✗ Failed: {benchmark}")
            print(f"Error: {result.stderr[:500]}")
            return {
                'benchmark': benchmark,
                'status': 'failed',
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
            }
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout: {benchmark}")
        return {
            'benchmark': benchmark,
            'status': 'timeout',
        }
    except Exception as e:
        print(f"✗ Error: {benchmark}: {e}")
        return {
            'benchmark': benchmark,
            'status': 'error',
            'error': str(e),
        }


def run_all_claim_extraction_evaluations(
    benchmarks: List[str],
    llm_model: str,
    max_reviews: Optional[int] = None,
    num_workers: int = 10,
    skip_fenice: bool = True,
    skip_gemma: bool = True,
    skip_reference_free: bool = True,
    use_llm_matching: bool = True,
    cosine_threshold: float = 0.3,
    output_summary: Optional[str] = None
) -> dict:
    """
    Run claim extraction evaluation for all specified benchmarks.
    
    Args:
        benchmarks: List of benchmark names (B2, B3)
        llm_model: LLM model name
        max_reviews: Maximum number of reviews to process per benchmark (None = all)
        num_workers: Number of parallel workers for API calls
        skip_fenice: Skip FENICE extraction
        skip_gemma: Skip Gemma extraction
        skip_reference_free: Skip reference-free metrics
        use_llm_matching: Use LLM for matching
        cosine_threshold: Threshold for cosine similarity matching
        output_summary: Path to save summary JSON (optional)
        
    Returns:
        Dictionary with summary of all runs
    """
    data_structure = DataStructure()
    results = []
    successful = []
    failed = []
    
    total_benchmarks = len(benchmarks)
    
    print(f"\n{'='*80}")
    print(f"RUNNING ALL CLAIM EXTRACTION EVALUATIONS")
    print(f"{'='*80}")
    print(f"Benchmarks: {', '.join(benchmarks)}")
    print(f"LLM Model: {llm_model}")
    print(f"Num Workers: {num_workers}")
    print(f"Skip FENICE: {skip_fenice}")
    print(f"Skip Gemma: {skip_gemma}")
    print(f"Skip Reference-Free: {skip_reference_free}")
    print(f"Use LLM Matching: {use_llm_matching}")
    print(f"Total Benchmarks: {total_benchmarks}")
    if max_reviews:
        print(f"Max Reviews per Benchmark: {max_reviews}")
    print(f"{'='*80}\n")
    
    start_time = datetime.now()
    
    for idx, benchmark in enumerate(benchmarks, 1):
        print(f"\n[{idx}/{total_benchmarks}] Processing {benchmark}...")
        
        result = run_claim_extraction_evaluation(
            benchmark=benchmark,
            llm_model=llm_model,
            max_reviews=max_reviews,
            num_workers=num_workers,
            skip_fenice=skip_fenice,
            skip_gemma=skip_gemma,
            skip_reference_free=skip_reference_free,
            use_llm_matching=use_llm_matching,
            cosine_threshold=cosine_threshold
        )
        
        results.append(result)
        
        if result['status'] == 'success':
            successful.append(benchmark)
        else:
            failed.append(benchmark)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Create summary
    summary = {
        'timestamp': start_time.isoformat(),
        'duration_seconds': duration,
        'configuration': {
            'benchmarks': benchmarks,
            'llm_model': llm_model,
            'num_workers': num_workers,
            'max_reviews': max_reviews,
            'skip_fenice': skip_fenice,
            'skip_gemma': skip_gemma,
            'skip_reference_free': skip_reference_free,
            'use_llm_matching': use_llm_matching,
            'cosine_threshold': cosine_threshold,
        },
        'total_benchmarks': total_benchmarks,
        'successful': len(successful),
        'failed': len(failed),
        'successful_benchmarks': successful,
        'failed_benchmarks': failed,
        'results': results,
    }
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total Benchmarks: {total_benchmarks}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Duration: {duration/60:.2f} minutes")
    print(f"\nSuccessful Benchmarks:")
    for benchmark in successful:
        print(f"  ✓ {benchmark}")
    
    if failed:
        print(f"\nFailed Benchmarks:")
        for benchmark in failed:
            print(f"  ✗ {benchmark}")
    
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
        summary_file = data_structure.logs_dir / f"claim_extraction_evaluation_summary_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSummary saved to: {summary_file}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Run claim extraction evaluation across all benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks with default settings
  python3 -m claim_verification.evaluation.run_all_extraction_evals --llm-model gpt-5-mini

  # Run specific benchmarks with more workers
  python3 -m claim_verification.evaluation.run_all_extraction_evals --benchmarks B2 B3 --llm-model gpt-5-mini --num-workers 20

  # Test with limited reviews
  python3 -m claim_verification.evaluation.run_all_extraction_evals --llm-model gpt-5-mini --max-reviews 10

  # Fastest mode (skip everything except LLM extraction + matching)
  python3 -m claim_verification.evaluation.run_all_extraction_evals --llm-model gpt-5-mini --num-workers 20
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
        '--llm-model',
        default='gpt-5-mini',
        help='LLM model name (default: gpt-5-mini)'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=10,
        help='Number of parallel workers for API calls (default: 10)'
    )
    
    parser.add_argument(
        '--max-reviews',
        type=int,
        help='Maximum number of reviews to process per benchmark (default: all)'
    )
    
    parser.add_argument(
        '--skip-fenice',
        action='store_true',
        default=True,
        help='Skip FENICE extraction (default: True)'
    )
    
    parser.add_argument(
        '--no-skip-fenice',
        dest='skip_fenice',
        action='store_false',
        help='Do NOT skip FENICE extraction'
    )
    
    parser.add_argument(
        '--skip-gemma',
        action='store_true',
        default=True,
        help='Skip Gemma extraction (default: True)'
    )
    
    parser.add_argument(
        '--no-skip-gemma',
        dest='skip_gemma',
        action='store_false',
        help='Do NOT skip Gemma extraction'
    )
    
    parser.add_argument(
        '--skip-reference-free',
        action='store_true',
        default=True,
        help='Skip reference-free metrics (default: True)'
    )
    
    parser.add_argument(
        '--no-skip-reference-free',
        dest='skip_reference_free',
        action='store_false',
        help='Do NOT skip reference-free metrics'
    )
    
    parser.add_argument(
        '--no-llm-matching',
        dest='use_llm_matching',
        action='store_false',
        help='Skip LLM matching (only use cosine similarity)'
    )
    
    parser.add_argument(
        '--cosine-threshold',
        type=float,
        default=0.3,
        help='Cosine similarity threshold for matching (default: 0.3)'
    )
    
    parser.add_argument(
        '--output-summary',
        help='Path to save summary JSON (default: auto-generated in data/logs/)'
    )
    
    args = parser.parse_args()
    
    summary = run_all_claim_extraction_evaluations(
        benchmarks=args.benchmarks,
        llm_model=args.llm_model,
        max_reviews=args.max_reviews,
        num_workers=args.num_workers,
        skip_fenice=args.skip_fenice,
        skip_gemma=args.skip_gemma,
        skip_reference_free=args.skip_reference_free,
        use_llm_matching=args.use_llm_matching,
        cosine_threshold=args.cosine_threshold,
        output_summary=args.output_summary
    )
    
    # Exit with error code if any failed
    if summary['failed'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
