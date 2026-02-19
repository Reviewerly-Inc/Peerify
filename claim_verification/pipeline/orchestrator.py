

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from claim_verification.config import (
    DataStructure, DEFAULT_MODEL, BENCHMARK_TARGETS, NUM_PAPERS_TO_SELECT, as_str
)


def print_header(title: str):
    """Print a section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def run_step_crawl(max_papers: Optional[int] = None):
    """Step 1: Crawl OpenReview."""
    print_header("STEP 1: CRAWLING OPENREVIEW")
    
    from claim_verification.pipeline.crawler import OpenReviewCrawler
    
    crawler = OpenReviewCrawler(max_papers=max_papers)
    result = crawler.crawl_all_venues()
    
    return result


def run_step_pair(num_papers: int = NUM_PAPERS_TO_SELECT, seed: int = 42):
    """Step 2: Pair reviews with responses."""
    print_header("STEP 2: PAIRING REVIEWS AND RESPONSES")
    
    from claim_verification.pipeline.pairer import ReviewResponsePairer
    
    pairer = ReviewResponsePairer(random_seed=seed)
    pairs = pairer.run(num_papers=num_papers)
    
    return pairs


def run_step_extract(target: int = BENCHMARK_TARGETS['B2_reviewer_author'], model: str = DEFAULT_MODEL):
    """Step 3: Extract claims from reviews."""
    print_header("STEP 3: EXTRACTING CLAIMS")
    
    from claim_verification.pipeline.claim_extractor import ClaimExtractor
    
    extractor = ClaimExtractor(model=model)
    all_claims, verifiable_claims = extractor.run(target_claims=target, filter_verifiable=True)
    
    return all_claims, verifiable_claims


def run_step_b1(target: int = BENCHMARK_TARGETS['B1_golden_supported'], model: str = DEFAULT_MODEL):
    """Step 4a: Generate B1 Golden Supported benchmark."""
    print_header("STEP 4a: GENERATING B1 (GOLDEN SUPPORTED)")
    
    from claim_verification.benchmarks.golden_benchmark import GoldenBenchmarkGenerator
    
    generator = GoldenBenchmarkGenerator(model=model)
    claims = generator.run(target_claims=target)
    
    return claims


def run_step_b2b3(target: int = BENCHMARK_TARGETS['B2_reviewer_author'], model: str = DEFAULT_MODEL):
    """Step 4b: Generate B2 and B3 benchmarks."""
    print_header("STEP 4b: GENERATING B2 & B3 (REVIEWER CLAIMS)")
    
    from claim_verification.benchmarks.reviewer_benchmark import ReviewerBenchmarkGenerator
    
    generator = ReviewerBenchmarkGenerator(model=model)
    b2, b3 = generator.run(target=target)
    
    return b2, b3


def run_step_b4():
    """Step 4c: Generate B4 Agreement benchmark."""
    print_header("STEP 4c: GENERATING B4 (AGREEMENT)")
    
    from claim_verification.benchmarks.agreement_benchmark import AgreementBenchmarkGenerator
    
    generator = AgreementBenchmarkGenerator()
    results, stats = generator.run()
    
    return results, stats


def run_step_b5():
    """Step 4d: Generate B5 Verifiable benchmark."""
    print_header("STEP 4d: GENERATING B5 (VERIFIABLE ONLY)")
    
    from claim_verification.benchmarks.verifiable_benchmark import VerifiableBenchmarkGenerator
    
    generator = VerifiableBenchmarkGenerator()
    results, stats = generator.run()
    
    return results, stats


def run_full_pipeline(
    max_papers: Optional[int] = None,
    num_papers: int = NUM_PAPERS_TO_SELECT,
    model: str = DEFAULT_MODEL,
    seed: int = 42,
    skip_crawl: bool = False
):
    """
    Run the complete pipeline.
    
    Args:
        max_papers: Max total papers to crawl across all venues (None uses default from config: 1000)
        num_papers: Number of papers to select for benchmark
        model: OpenAI model to use
        seed: Random seed for reproducibility
        skip_crawl: Skip crawling if data already exists
    """
    data_structure = DataStructure()
    results = {}
    
    print_header("BENCHMARK GENERATION PIPELINE")
    print(f"Configuration:")
    from claim_verification.config import MAX_PAPERS_TO_CRAWL
    max_crawl = max_papers if max_papers is not None else MAX_PAPERS_TO_CRAWL
    print(f"  Max total papers to crawl: {max_crawl} (distributed across venues)")
    print(f"  Papers to select: {num_papers}")
    print(f"  Model: {model}")
    print(f"  Random seed: {seed}")
    print(f"  Data directory: {as_str(data_structure.base_dir)}")
    print()
    
    # Step 1: Crawl
    if skip_crawl:
        combined_file = data_structure.raw_dir / "all_papers.json"
        if combined_file.exists():
            print("Skipping crawl - data already exists")
        else:
            print("WARNING: No crawl data found, running crawl...")
            results['crawl'] = run_step_crawl(max_papers)
    else:
        results['crawl'] = run_step_crawl(max_papers)
    
    # Step 2: Pair
    results['pairs'] = run_step_pair(num_papers, seed)
    
    if not results['pairs']:
        print("ERROR: No pairs created. Cannot continue.")
        return results
    
    # Step 3: Extract claims
    all_claims, verifiable_claims = run_step_extract(
        target=BENCHMARK_TARGETS['B2_reviewer_author'],
        model=model
    )
    results['all_claims'] = all_claims
    results['verifiable_claims'] = verifiable_claims
    
    if not all_claims:
        print("ERROR: No claims extracted. Cannot continue.")
        return results
    
    # Step 4a: B1 Golden Supported
    results['b1'] = run_step_b1(
        target=BENCHMARK_TARGETS['B1_golden_supported'],
        model=model
    )
    
    # Step 4b: B2 & B3
    b2, b3 = run_step_b2b3(
        target=BENCHMARK_TARGETS['B2_reviewer_author'],
        model=model
    )
    results['b2'] = b2
    results['b3'] = b3
    
    # Step 4c: B4 Agreement
    b4, b4_stats = run_step_b4()
    results['b4'] = b4
    results['b4_stats'] = b4_stats
    
    # Step 4d: B5 Verifiable
    b5, b5_stats = run_step_b5()
    results['b5'] = b5
    results['b5_stats'] = b5_stats
    
    # Final summary
    print_header("PIPELINE COMPLETE")
    print("Benchmark Summary:")
    print(f"  B1 (Golden Supported): {len(results.get('b1', []))} claims")
    print(f"  B2 (Reviewer vs Author): {len(results.get('b2', []))} claims")
    print(f"  B3 (Reviewer vs Paper): {len(results.get('b3', []))} claims")
    print(f"  B4 (Agreement): {len(results.get('b4', []))} records")
    print(f"  B5 (Verifiable Only): {len(results.get('b5', []))} claims")
    print()
    print(f"Output directory: {as_str(data_structure.benchmark_dir)}")
    
    # Save pipeline summary
    summary = {
        'pipeline_timestamp': datetime.now().isoformat(),
        'configuration': {
            'max_papers_per_venue': max_papers,
            'num_papers_selected': num_papers,
            'model': model,
            'random_seed': seed,
        },
        'results': {
            'b1_count': len(results.get('b1', [])),
            'b2_count': len(results.get('b2', [])),
            'b3_count': len(results.get('b3', [])),
            'b4_count': len(results.get('b4', [])),
            'b5_count': len(results.get('b5', [])),
            'b4_agreement_rate': results.get('b4_stats', {}).get('exact_agreement_rate'),
            'b5_filter_rate': results.get('b5_stats', {}).get('filter_rate'),
        }
    }
    
    summary_file = data_structure.logs_dir / "pipeline_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nPipeline summary saved to {as_str(summary_file)}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark Generation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  all       Run complete pipeline (default)
  crawl     Step 1: Crawl OpenReview
  pair      Step 2: Pair reviews and responses
  extract   Step 3: Extract claims
  b1        Step 4a: Generate B1 (Golden Supported)
  b2b3      Step 4b: Generate B2 & B3 (Reviewer Claims)
  b4        Step 4c: Generate B4 (Agreement)
  b5        Step 4d: Generate B5 (Verifiable Only)

Example:
  python -m claim_verification.pipeline.orchestrator --step all
  python -m claim_verification.pipeline.orchestrator --step crawl --max-papers 50
  python -m claim_verification.pipeline.orchestrator --step b2b3 --model gpt-4o-mini
        """
    )
    
    parser.add_argument('--step', default='all',
                       choices=['all', 'crawl', 'pair', 'extract', 'b1', 'b2b3', 'b4', 'b5'],
                       help='Pipeline step to run (default: all)')
    parser.add_argument('--max-papers', type=int, default=None,
                       help='Max total papers to crawl across all venues (default: 1000 from config)')
    parser.add_argument('--num-papers', type=int, default=NUM_PAPERS_TO_SELECT,
                       help=f'Number of papers to select (default: {NUM_PAPERS_TO_SELECT})')
    parser.add_argument('--model', default=DEFAULT_MODEL,
                       help=f'OpenAI model to use (default: {DEFAULT_MODEL})')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--skip-crawl', action='store_true',
                       help='Skip crawling if data exists')
    
    args = parser.parse_args()
    
    try:
        if args.step == 'all':
            run_full_pipeline(
                max_papers=args.max_papers,
                num_papers=args.num_papers,
                model=args.model,
                seed=args.seed,
                skip_crawl=args.skip_crawl
            )
        elif args.step == 'crawl':
            run_step_crawl(args.max_papers)
        elif args.step == 'pair':
            run_step_pair(args.num_papers, args.seed)
        elif args.step == 'extract':
            run_step_extract(model=args.model)
        elif args.step == 'b1':
            run_step_b1(model=args.model)
        elif args.step == 'b2b3':
            run_step_b2b3(model=args.model)
        elif args.step == 'b4':
            run_step_b4()
        elif args.step == 'b5':
            run_step_b5()
        
        print("\nDone!")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

