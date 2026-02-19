
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Set
from collections import Counter, defaultdict
from datetime import datetime
import sys

from claim_verification.config import DataStructure


def load_benchmark(benchmark_path: Path) -> List[Dict[str, Any]]:
    """Load benchmark JSONL file."""
    entries = []
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def calculate_statistics(entries: List[Dict[str, Any]], benchmark_name: str) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for a benchmark.
    
    Args:
        entries: List of benchmark entries
        benchmark_name: Name of the benchmark (B1, B2, etc.)
        
    Returns:
        Dictionary with all statistics
    """
    stats = {
        'benchmark_name': benchmark_name,
        'total_claims': len(entries),
    }
    
    # Basic counts
    unique_papers = set()
    unique_reviews = set()
    unique_reviewers = set()
    labels = []
    decisions = []
    venues = []
    claim_types = []
    claim_lengths = []
    timestamps = []
    
    # For B4, track both B2 and B3 labels
    b2_labels = []
    b3_labels = []
    
    # Group entries by venue and decision for breakdowns
    entries_by_venue = defaultdict(list)
    entries_by_decision = defaultdict(list)
    
    for entry in entries:
        # Paper IDs
        if 'paper_id' in entry:
            unique_papers.add(entry['paper_id'])
        
        # Review IDs
        if 'review_id' in entry:
            unique_reviews.add(entry['review_id'])
        
        # Reviewers
        if 'reviewer' in entry:
            unique_reviewers.add(entry['reviewer'])
        
        # Labels
        if 'label' in entry:
            labels.append(entry['label'])
        
        # B4 has both b2_label and b3_label
        if 'b2_label' in entry:
            b2_labels.append(entry['b2_label'])
        if 'b3_label' in entry:
            b3_labels.append(entry['b3_label'])
        
        # Decisions
        decision = None
        if 'decision' in entry:
            decision = entry['decision']
            decisions.append(decision)
        elif 'paper_decision' in entry:
            decision = entry['paper_decision']
            decisions.append(decision)
        
        # Venues
        venue = None
        if 'paper_venue' in entry:
            venue = entry['paper_venue']
            venues.append(venue)
        
        # Group by venue
        if venue:
            entries_by_venue[venue].append(entry)
        
        # Group by decision
        if decision:
            entries_by_decision[decision].append(entry)
        
        # Claim types
        if 'claim_type' in entry:
            claim_types.append(entry['claim_type'])
        
        # Claim lengths
        if 'claim' in entry:
            claim_lengths.append(len(entry['claim']))
        
        # Timestamps
        if 'extraction_timestamp' in entry:
            timestamps.append(entry['extraction_timestamp'])
        elif 'labeling_timestamp' in entry:
            timestamps.append(entry['labeling_timestamp'])
    
    # Unique counts
    stats['unique_papers'] = len(unique_papers)
    stats['unique_reviews'] = len(unique_reviews)
    stats['unique_reviewers'] = len(unique_reviewers)
    
    # Label distribution
    if labels:
        label_dist = Counter(labels)
        stats['label_distribution'] = dict(label_dist)
        stats['label_counts'] = {
            label: count for label, count in label_dist.items()
        }
    
    # B2 and B3 label distributions (for B4)
    if b2_labels:
        b2_label_dist = Counter(b2_labels)
        stats['b2_label_distribution'] = dict(b2_label_dist)
    if b3_labels:
        b3_label_dist = Counter(b3_labels)
        stats['b3_label_distribution'] = dict(b3_label_dist)
    
    # Decision distribution
    if decisions:
        decision_dist = Counter(decisions)
        stats['decision_distribution'] = dict(decision_dist)
        stats['decision_counts'] = {
            decision: count for decision, count in decision_dist.items()
        }
    
    # Venue distribution
    if venues:
        venue_dist = Counter(venues)
        stats['venue_distribution'] = dict(venue_dist)
        stats['venue_counts'] = {
            venue: count for venue, count in venue_dist.items()
        }
    
    # Claim type distribution
    if claim_types:
        claim_type_dist = Counter(claim_types)
        stats['claim_type_distribution'] = dict(claim_type_dist)
        stats['claim_type_counts'] = {
            claim_type: count for claim_type, count in claim_type_dist.items()
        }
    
    # Claim length statistics
    if claim_lengths:
        stats['claim_length'] = {
            'mean': sum(claim_lengths) / len(claim_lengths),
            'min': min(claim_lengths),
            'max': max(claim_lengths),
            'median': sorted(claim_lengths)[len(claim_lengths) // 2],
        }
    
    # Timestamp range
    if timestamps:
        try:
            parsed_timestamps = [datetime.fromisoformat(ts.replace('Z', '+00:00')) for ts in timestamps]
            stats['timestamp_range'] = {
                'earliest': min(parsed_timestamps).isoformat(),
                'latest': max(parsed_timestamps).isoformat(),
            }
        except:
            pass
    
    # Additional statistics
    stats['papers_per_claim'] = stats['unique_papers'] / stats['total_claims'] if stats['total_claims'] > 0 else 0
    stats['reviews_per_claim'] = stats['unique_reviews'] / stats['total_claims'] if stats['total_claims'] > 0 else 0
    stats['claims_per_paper'] = stats['total_claims'] / stats['unique_papers'] if stats['unique_papers'] > 0 else 0
    
    # Statistics by venue
    stats['by_venue'] = {}
    for venue, venue_entries in entries_by_venue.items():
        venue_stats = calculate_subset_statistics(venue_entries, venue)
        stats['by_venue'][venue] = venue_stats
    
    # Statistics by decision
    stats['by_decision'] = {}
    for decision, decision_entries in entries_by_decision.items():
        decision_stats = calculate_subset_statistics(decision_entries, decision)
        stats['by_decision'][decision] = decision_stats
    
    return stats


def calculate_subset_statistics(entries: List[Dict[str, Any]], subset_name: str) -> Dict[str, Any]:
    """
    Calculate statistics for a subset of entries (by venue or decision).
    
    Args:
        entries: List of benchmark entries for this subset
        subset_name: Name of the subset (venue or decision)
        
    Returns:
        Dictionary with statistics for this subset
    """
    stats = {
        'name': subset_name,
        'total_claims': len(entries),
    }
    
    unique_papers = set()
    unique_reviews = set()
    unique_reviewers = set()
    labels = []
    claim_types = []
    claim_lengths = []
    b2_labels = []
    b3_labels = []
    
    for entry in entries:
        if 'paper_id' in entry:
            unique_papers.add(entry['paper_id'])
        if 'review_id' in entry:
            unique_reviews.add(entry['review_id'])
        if 'reviewer' in entry:
            unique_reviewers.add(entry['reviewer'])
        if 'label' in entry:
            labels.append(entry['label'])
        if 'b2_label' in entry:
            b2_labels.append(entry['b2_label'])
        if 'b3_label' in entry:
            b3_labels.append(entry['b3_label'])
        if 'claim_type' in entry:
            claim_types.append(entry['claim_type'])
        if 'claim' in entry:
            claim_lengths.append(len(entry['claim']))
    
    stats['unique_papers'] = len(unique_papers)
    stats['unique_reviews'] = len(unique_reviews)
    stats['unique_reviewers'] = len(unique_reviewers)
    
    if labels:
        stats['label_distribution'] = dict(Counter(labels))
    if b2_labels:
        stats['b2_label_distribution'] = dict(Counter(b2_labels))
    if b3_labels:
        stats['b3_label_distribution'] = dict(Counter(b3_labels))
    if claim_types:
        stats['claim_type_distribution'] = dict(Counter(claim_types))
    if claim_lengths:
        stats['claim_length'] = {
            'mean': sum(claim_lengths) / len(claim_lengths),
            'min': min(claim_lengths),
            'max': max(claim_lengths),
            'median': sorted(claim_lengths)[len(claim_lengths) // 2],
        }
    
    stats['claims_per_paper'] = stats['total_claims'] / stats['unique_papers'] if stats['unique_papers'] > 0 else 0
    
    return stats


def print_statistics(stats: Dict[str, Any]):
    """Print statistics in a readable format."""
    print(f"\n{'='*80}")
    print(f"BENCHMARK STATISTICS: {stats['benchmark_name']}")
    print(f"{'='*80}\n")
    
    print(f"Total Claims: {stats['total_claims']}")
    print(f"Unique Papers: {stats['unique_papers']}")
    print(f"Unique Reviews: {stats['unique_reviews']}")
    print(f"Unique Reviewers: {stats['unique_reviewers']}")
    print(f"Claims per Paper: {stats.get('claims_per_paper', 0):.2f}")
    print(f"Papers per Claim: {stats.get('papers_per_claim', 0):.2f}")
    
    if 'label_distribution' in stats:
        print(f"\nLabel Distribution:")
        for label, count in sorted(stats['label_distribution'].items()):
            percentage = (count / stats['total_claims']) * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")
    
    if 'b2_label_distribution' in stats:
        print(f"\nB2 Label Distribution:")
        total_b2 = sum(stats['b2_label_distribution'].values())
        for label, count in sorted(stats['b2_label_distribution'].items()):
            percentage = (count / total_b2) * 100 if total_b2 > 0 else 0
            print(f"  {label}: {count} ({percentage:.1f}%)")
    
    if 'b3_label_distribution' in stats:
        print(f"\nB3 Label Distribution:")
        total_b3 = sum(stats['b3_label_distribution'].values())
        for label, count in sorted(stats['b3_label_distribution'].items()):
            percentage = (count / total_b3) * 100 if total_b3 > 0 else 0
            print(f"  {label}: {count} ({percentage:.1f}%)")
    
    if 'decision_distribution' in stats:
        print(f"\nDecision Distribution:")
        for decision, count in sorted(stats['decision_distribution'].items()):
            percentage = (count / stats['total_claims']) * 100
            print(f"  {decision}: {count} ({percentage:.1f}%)")
    
    if 'venue_distribution' in stats:
        print(f"\nVenue Distribution:")
        for venue, count in sorted(stats['venue_distribution'].items()):
            percentage = (count / stats['total_claims']) * 100
            print(f"  {venue}: {count} ({percentage:.1f}%)")
    
    if 'claim_type_distribution' in stats:
        print(f"\nClaim Type Distribution:")
        for claim_type, count in sorted(stats['claim_type_distribution'].items()):
            percentage = (count / stats['total_claims']) * 100
            print(f"  {claim_type}: {count} ({percentage:.1f}%)")
    
    if 'claim_length' in stats:
        print(f"\nClaim Length Statistics:")
        print(f"  Mean: {stats['claim_length']['mean']:.1f} characters")
        print(f"  Min: {stats['claim_length']['min']} characters")
        print(f"  Max: {stats['claim_length']['max']} characters")
        print(f"  Median: {stats['claim_length']['median']:.1f} characters")
    
    if 'timestamp_range' in stats:
        print(f"\nTimestamp Range:")
        print(f"  Earliest: {stats['timestamp_range']['earliest']}")
        print(f"  Latest: {stats['timestamp_range']['latest']}")
    
    print()


def calculate_all_benchmark_statistics(
    benchmarks: List[str] = None,
    output_file: str = None
) -> Dict[str, Any]:
    """
    Calculate statistics for all benchmarks.
    
    Args:
        benchmarks: List of benchmark names (B1, B2, etc.) or None for all
        output_file: Path to save statistics JSON (optional)
        
    Returns:
        Dictionary with statistics for all benchmarks
    """
    data_structure = DataStructure()
    
    if benchmarks is None:
        benchmarks = ['B1', 'B2', 'B3', 'B4', 'B5']
    
    benchmark_map = {
        'B1': 'B1_golden_supported.jsonl',
        'B2': 'B2_reviewer_author.jsonl',
        'B3': 'B3_reviewer_paper.jsonl',
        'B4': 'B4_agreement.jsonl',
        'B5': 'B5_verifiable.jsonl',
    }
    
    all_stats = {}
    
    for benchmark_name in benchmarks:
        benchmark_file = benchmark_map.get(benchmark_name.upper())
        if not benchmark_file:
            print(f"Warning: Unknown benchmark {benchmark_name}, skipping...")
            continue
        
        benchmark_path = data_structure.benchmark_dir / benchmark_file
        
        if not benchmark_path.exists():
            print(f"Warning: Benchmark file not found: {benchmark_path}, skipping...")
            continue
        
        print(f"\nLoading {benchmark_name} from {benchmark_path}...")
        entries = load_benchmark(benchmark_path)
        
        print(f"Calculating statistics for {benchmark_name}...")
        stats = calculate_statistics(entries, benchmark_name)
        all_stats[benchmark_name] = stats
        
        print_statistics(stats)
    
    # Summary across all benchmarks
    print(f"\n{'='*80}")
    print("SUMMARY ACROSS ALL BENCHMARKS")
    print(f"{'='*80}\n")
    
    total_claims = sum(s['total_claims'] for s in all_stats.values())
    total_papers = len(set().union(*[set() for s in all_stats.values()]))  # This won't work, need to track papers
    
    print(f"Total Claims (all benchmarks): {total_claims}")
    print(f"Number of Benchmarks: {len(all_stats)}")
    
    for benchmark_name, stats in all_stats.items():
        print(f"\n{benchmark_name}:")
        print(f"  Claims: {stats['total_claims']}")
        print(f"  Papers: {stats['unique_papers']}")
        print(f"  Reviews: {stats['unique_reviews']}")
    
    # Save to file
    if output_file:
        output_path = Path(output_file)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = data_structure.logs_dir / f"benchmark_statistics_{timestamp}.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    
    print(f"\nStatistics saved to: {output_path}")
    
    return all_stats


def main():
    parser = argparse.ArgumentParser(
        description='Calculate statistics for benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate statistics for all benchmarks
  python3 -m claim_verification.calculate_benchmark_statistics

  # Calculate statistics for specific benchmarks
  python3 -m claim_verification.calculate_benchmark_statistics --benchmarks B1 B2

  # Save to specific file
  python3 -m claim_verification.calculate_benchmark_statistics --output data/logs/my_stats.json
        """
    )
    
    parser.add_argument(
        '--benchmarks',
        nargs='+',
        choices=['B1', 'B2', 'B3', 'B4', 'B5'],
        help='Benchmarks to analyze (default: all)'
    )
    
    parser.add_argument(
        '--output',
        help='Output JSON file path (default: auto-generated in data/logs/)'
    )
    
    args = parser.parse_args()
    
    calculate_all_benchmark_statistics(
        benchmarks=args.benchmarks,
        output_file=args.output
    )


if __name__ == '__main__':
    main()
