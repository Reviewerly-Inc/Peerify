

import json
from pathlib import Path
import sys
from typing import List, Dict, Any, Tuple
from datetime import datetime
from collections import Counter

from claim_verification.config import DataStructure, as_str


# Verifiable claim types
VERIFIABLE_TYPES = ['experimental', 'methodology', 'quantitative', 'baseline']

# Non-verifiable types
NON_VERIFIABLE_TYPES = ['novelty', 'presentation', 'clarity', 'other']


class VerifiableBenchmarkGenerator:
    """
    Generates the B5 Verifiable Claims benchmark by filtering B4 (agreement).
    
    B5 contains only verifiable claims from the agreement benchmark (B4).
    This is a static filter - no LLM calls needed.
    """
    
    def __init__(self):
        self.data_structure = DataStructure()
    
    def load_agreement_benchmark(self) -> List[Dict[str, Any]]:
        """Load the B4 agreement benchmark."""
        agreement_file = self.data_structure.get_benchmark_file("B4_agreement")
        
        if not agreement_file.exists():
            raise FileNotFoundError(
                f"Agreement benchmark (B4) not found: {agreement_file}\n"
                "Please run agreement_benchmark.py first to generate B4."
            )
        
        records = []
        with open(agreement_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        
        print(f"Loaded {len(records)} agreement records from B4")
        return records
    
    def load_verifiable_claims(self) -> List[str]:
        """Load the list of verifiable claims from claim_extractor output."""
        verifiable_file = self.data_structure.get_verifiable_claims_file()
        
        if verifiable_file.exists():
            with open(verifiable_file, 'r', encoding='utf-8') as f:
                claims = json.load(f)
            return [c.get('claim', '') for c in claims]
        
        return []
    
    def _is_verifiable(self, record: Dict[str, Any], verifiable_claims: List[str]) -> bool:
        """
        Determine if a claim is verifiable.
        
        A claim is verifiable if:
        1. Its claim_type is in VERIFIABLE_TYPES, OR
        2. It was marked as verifiable in the extraction phase, OR
        3. It contains keywords indicating verifiability
        """
        claim_type = record.get('claim_type', 'other').lower()
        claim_text = record.get('claim', '').lower()
        
        # Check claim type
        if claim_type in VERIFIABLE_TYPES:
            return True
        
        if claim_type in NON_VERIFIABLE_TYPES:
            return False
        
        # Check if in verifiable claims list
        if record.get('claim', '') in verifiable_claims:
            return True
        
        # Keyword-based check
        verifiable_keywords = [
            'accuracy', 'performance', 'result', 'experiment', 'dataset',
            'baseline', 'benchmark', 'metric', 'evaluation', 'ablation',
            'table', 'training', 'parameter', 'loss', 'convergence',
            'runtime', 'method', 'algorithm', 'implementation', 
            'statistical', 'significant', '%', 'improve'
        ]
        
        non_verifiable_keywords = [
            'novel', 'originality', 'contribution', 'figure quality',
            'picture', 'visualization', 'writing', 'presentation',
            'clarity', 'grammar', 'typo', 'interesting', 'exciting',
            'related work', 'motivation'
        ]
        
        has_verifiable = any(kw in claim_text for kw in verifiable_keywords)
        has_non_verifiable = any(kw in claim_text for kw in non_verifiable_keywords)
        
        # Prefer verifiable if both present
        if has_verifiable and not has_non_verifiable:
            return True
        
        if has_non_verifiable and not has_verifiable:
            return False
        
        # Default: include experimental and methodology types
        return claim_type in ['experimental', 'methodology', 'quantitative', 'baseline']
    
    def filter_verifiable(
        self, 
        records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter records to keep only verifiable claims.
        """
        # Load verifiable claims list from extraction phase
        verifiable_claims = self.load_verifiable_claims()
        print(f"Loaded {len(verifiable_claims)} pre-classified verifiable claims")
        
        filtered = []
        
        for record in records:
            if self._is_verifiable(record, verifiable_claims):
                # Add is_verifiable flag
                record_copy = record.copy()
                record_copy['is_verifiable'] = True
                filtered.append(record_copy)
        
        print(f"Filtered to {len(filtered)} verifiable claims (from {len(records)})")
        
        return filtered
    
    def compute_statistics(self, original: List[Dict[str, Any]], filtered: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute filtering statistics."""
        original_count = len(original)
        filtered_count = len(filtered)
        
        if original_count == 0:
            return {'error': 'No records to analyze'}
        
        # Claim type distribution before/after
        original_types = Counter(r.get('claim_type', 'other') for r in original)
        filtered_types = Counter(r.get('claim_type', 'other') for r in filtered)
        
        # Label distribution in filtered set
        b2_labels = Counter(r.get('b2_label', 'Not Determined') for r in filtered)
        b3_labels = Counter(r.get('b3_label', 'Not Determined') for r in filtered)
        
        # Venue distribution
        venue_dist = Counter(r.get('paper_venue', 'unknown') for r in filtered)
        
        stats = {
            'original_count': original_count,
            'filtered_count': filtered_count,
            'filter_rate': filtered_count / original_count if original_count > 0 else 0,
            'removed_count': original_count - filtered_count,
            'claim_type_original': dict(original_types),
            'claim_type_filtered': dict(filtered_types),
            'b2_label_distribution': dict(b2_labels),
            'b3_label_distribution': dict(b3_labels),
            'venue_distribution': dict(venue_dist),
        }
        
        return stats
    
    def save_benchmark(self, results: List[Dict[str, Any]], stats: Dict[str, Any], split_by_decision: bool = True):
        """Save the B5 benchmark, optionally split by decision type."""
        from collections import defaultdict
        
        # Group by decision
        results_by_decision = defaultdict(list)
        for record in results:
            decision = record.get('decision', 'Unknown')
            results_by_decision[decision].append(record)
        
        # Save all results
        b5_file = self.data_structure.get_benchmark_file("B5_verifiable")
        with open(b5_file, 'w', encoding='utf-8') as f:
            for record in results:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"Saved B5 (all) to {as_str(b5_file)}: {len(results)} records")
        
        # Save split by decision
        if split_by_decision:
            for decision, decision_results in results_by_decision.items():
                decision_file = self.data_structure.get_benchmark_file("B5_verifiable", decision)
                with open(decision_file, 'w', encoding='utf-8') as f:
                    for record in decision_results:
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
                print(f"  Saved B5 ({decision}) to {as_str(decision_file)}: {len(decision_results)} records")
        
        # Save summary
        summary = {
            'benchmark': 'B5_verifiable',
            'generation_timestamp': datetime.now().isoformat(),
            'statistics': stats,
        }
        
        summary_file = self.data_structure.logs_dir / "B5_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved B5 summary to {as_str(summary_file)}")
    
    def run(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Run the B5 verifiable claims benchmark generation.
        """
        print("="*60)
        print("B5: VERIFIABLE CLAIMS BENCHMARK (Filter)")
        print("="*60)
        
        # Load B4 agreement benchmark
        records = self.load_agreement_benchmark()
        
        # Filter to verifiable claims
        filtered = self.filter_verifiable(records)
        
        # Compute statistics
        stats = self.compute_statistics(records, filtered)
        
        # Print statistics
        print(f"\nFilter Statistics:")
        print(f"  Original: {stats['original_count']} claims")
        print(f"  Filtered: {stats['filtered_count']} claims ({stats['filter_rate']:.1%})")
        print(f"  Removed: {stats['removed_count']} claims")
        print(f"\nClaim types (filtered):")
        for ct, count in stats['claim_type_filtered'].items():
            print(f"    {ct}: {count}")
        
        # Save results
        self.save_benchmark(filtered, stats)
        
        print("="*60)
        print(f"B5 COMPLETE: {len(filtered)} verifiable claims")
        print("="*60)
        
        return filtered, stats


def main():
    try:
        generator = VerifiableBenchmarkGenerator()
        results, stats = generator.run()
        
        print(f"\nB5 benchmark generated: {len(results)} verifiable claims")
        print(f"Filter rate: {stats['filter_rate']:.1%}")
        
    except Exception as e:
        print(f"Error generating B5 benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

