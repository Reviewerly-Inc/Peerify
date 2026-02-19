
import json
from pathlib import Path
import sys
from typing import List, Dict, Any, Tuple
from datetime import datetime
from collections import Counter

from claim_verification.config import DataStructure, as_str


class AgreementBenchmarkGenerator:
    """
    Generates the B4 Agreement benchmark by comparing B2 and B3 labels.
    
    This is a static comparison - no LLM calls needed.
    """
    
    def __init__(self):
        self.data_structure = DataStructure()
    
    def load_combined_benchmark(self) -> List[Dict[str, Any]]:
        """Load the combined B2+B3 benchmark."""
        combined_file = self.data_structure.get_benchmark_file("B2_B3_combined")
        
        if not combined_file.exists():
            raise FileNotFoundError(
                f"Combined benchmark not found: {combined_file}\n"
                "Please run reviewer_benchmark.py first."
            )
        
        records = []
        with open(combined_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        
        print(f"Loaded {len(records)} combined records")
        return records
    
    def compute_agreement(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Compute agreement between B2 and B3 labels.
        
        Agreement types:
        - exact_agreement: B2 and B3 have exactly the same label
        - partial_agreement: Labels are compatible (e.g., Supported vs Partially Supported)
        - disagreement: Labels conflict (e.g., Supported vs Contradicted)
        """
        results = []
        
        # Define agreement matrix
        # Full agreement: same label
        # Partial agreement: one level apart
        # Disagreement: opposite labels
        agreement_map = {
            ('Supported', 'Supported'): 'exact',
            ('Supported', 'Partially Supported'): 'partial',
            ('Supported', 'Contradicted'): 'disagree',
            ('Supported', 'Not Determined'): 'partial',
            
            ('Partially Supported', 'Supported'): 'partial',
            ('Partially Supported', 'Partially Supported'): 'exact',
            ('Partially Supported', 'Contradicted'): 'partial',
            ('Partially Supported', 'Not Determined'): 'partial',
            
            ('Contradicted', 'Supported'): 'disagree',
            ('Contradicted', 'Partially Supported'): 'partial',
            ('Contradicted', 'Contradicted'): 'exact',
            ('Contradicted', 'Not Determined'): 'partial',
            
            ('Not Determined', 'Supported'): 'partial',
            ('Not Determined', 'Partially Supported'): 'partial',
            ('Not Determined', 'Contradicted'): 'partial',
            ('Not Determined', 'Not Determined'): 'exact',
        }
        
        for record in records:
            b2_label = record.get('b2_label', 'Not Determined')
            b3_label = record.get('b3_label', 'Not Determined')
            
            # Determine agreement type
            agreement_type = agreement_map.get((b2_label, b3_label), 'partial')
            exact_match = b2_label == b3_label
            
            # Create agreement record
            agreement_record = {
                'claim': record.get('claim'),
                'claim_type': record.get('claim_type'),
                'paper_id': record.get('paper_id'),
                'paper_title': record.get('paper_title'),
                'paper_venue': record.get('paper_venue'),
                'paper_decision': record.get('paper_decision'),  # Add decision metadata
                'decision': record.get('decision', 'Unknown'),  # Add normalized decision
                'review_id': record.get('review_id'),
                
                # Labels
                'b2_label': b2_label,
                'b3_label': b3_label,
                
                # Agreement metrics
                'agreement': exact_match,
                'agreement_type': agreement_type,
                
                # Evidence for reference
                'b2_evidence': record.get('b2_evidence', ''),
                'b3_evidence': record.get('b3_evidence', ''),
                'b2_justification': record.get('b2_justification', ''),
                'b3_justification': record.get('b3_justification', ''),
            }
            
            results.append(agreement_record)
        
        return results
    
    def compute_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute agreement statistics."""
        total = len(results)
        
        if total == 0:
            return {'error': 'No results to analyze'}
        
        # Basic agreement
        exact_agreements = sum(1 for r in results if r.get('agreement', False))
        exact_rate = exact_agreements / total
        
        # Agreement type distribution
        type_counts = Counter(r.get('agreement_type', 'unknown') for r in results)
        
        # Label pair distribution (confusion matrix style)
        label_pairs = Counter((r.get('b2_label'), r.get('b3_label')) for r in results)
        
        # Per-label agreement rates
        label_agreement = {}
        for label in ['Supported', 'Partially Supported', 'Contradicted', 'Not Determined']:
            b2_with_label = [r for r in results if r.get('b2_label') == label]
            if b2_with_label:
                matching = sum(1 for r in b2_with_label if r.get('b3_label') == label)
                label_agreement[label] = {
                    'count': len(b2_with_label),
                    'matching': matching,
                    'rate': matching / len(b2_with_label)
                }
        
        # Interesting disagreements (B2 says Supported but B3 says Contradicted, or vice versa)
        strong_disagreements = [
            r for r in results 
            if (r.get('b2_label') == 'Supported' and r.get('b3_label') == 'Contradicted') or
               (r.get('b2_label') == 'Contradicted' and r.get('b3_label') == 'Supported')
        ]
        
        stats = {
            'total_claims': total,
            'exact_agreement_count': exact_agreements,
            'exact_agreement_rate': exact_rate,
            'agreement_type_distribution': dict(type_counts),
            'label_pair_distribution': {f"{k[0]}_vs_{k[1]}": v for k, v in label_pairs.items()},
            'per_label_agreement': label_agreement,
            'strong_disagreement_count': len(strong_disagreements),
            'strong_disagreement_rate': len(strong_disagreements) / total if total > 0 else 0,
        }
        
        return stats
    
    def save_benchmark(self, results: List[Dict[str, Any]], stats: Dict[str, Any], split_by_decision: bool = True):
        """Save the B4 benchmark and statistics, optionally split by decision type."""
        from collections import defaultdict
        
        # Group by decision
        results_by_decision = defaultdict(list)
        for record in results:
            decision = record.get('decision', 'Unknown')
            results_by_decision[decision].append(record)
        
        # Save all results
        b4_file = self.data_structure.get_benchmark_file("B4_agreement")
        with open(b4_file, 'w', encoding='utf-8') as f:
            for record in results:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"Saved B4 (all) to {as_str(b4_file)}: {len(results)} records")
        
        # Save split by decision
        if split_by_decision:
            for decision, decision_results in results_by_decision.items():
                decision_file = self.data_structure.get_benchmark_file("B4_agreement", decision)
                with open(decision_file, 'w', encoding='utf-8') as f:
                    for record in decision_results:
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
                print(f"  Saved B4 ({decision}) to {as_str(decision_file)}: {len(decision_results)} records")
        
        # Save summary with statistics
        summary = {
            'benchmark': 'B4_agreement',
            'generation_timestamp': datetime.now().isoformat(),
            'statistics': stats,
        }
        
        summary_file = self.data_structure.logs_dir / "B4_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved B4 summary to {as_str(summary_file)}")
        
        # Save disagreements separately for analysis
        disagreements = [r for r in results if not r.get('agreement', False)]
        disagree_file = self.data_structure.benchmark_dir / "B4_disagreements.jsonl"
        with open(disagree_file, 'w', encoding='utf-8') as f:
            for record in disagreements:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"Saved {len(disagreements)} disagreements to {as_str(disagree_file)}")
    
    def run(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Run the B4 agreement benchmark generation.
        
        B4 contains only records with exact agreement (b2_label == b3_label).
        """
        print("="*60)
        print("B4: AGREEMENT BENCHMARK (Exact Agreements Only)")
        print("="*60)
        
        # Load combined benchmark
        records = self.load_combined_benchmark()
        
        # Compute agreement for all records (for statistics)
        all_results = self.compute_agreement(records)
        
        # Compute statistics on all records
        all_stats = self.compute_statistics(all_results)
        
        # Filter to only exact agreements (b2_label == b3_label)
        exact_results = [r for r in all_results if r.get('agreement', False)]
        
        # Compute statistics on exact agreements only
        stats = self.compute_statistics(exact_results)
        stats['original_total'] = len(all_results)
        stats['exact_agreement_count'] = len(exact_results)
        stats['exact_agreement_rate'] = len(exact_results) / len(all_results) if all_results else 0
        
        # Print key statistics
        print(f"\nAgreement Statistics (All Records):")
        print(f"  Total claims: {all_stats['total_claims']}")
        print(f"  Exact agreement: {all_stats['exact_agreement_count']} ({all_stats['exact_agreement_rate']:.1%})")
        print(f"  Agreement types: {all_stats['agreement_type_distribution']}")
        print(f"  Strong disagreements: {all_stats['strong_disagreement_count']} ({all_stats['strong_disagreement_rate']:.1%})")
        
        print(f"\nB4 Filtered Results (Exact Agreements Only):")
        print(f"  Exact agreements: {len(exact_results)}")
        print(f"  Percentage of total: {len(exact_results) / len(all_results) * 100:.1f}%")
        
        # Save only exact agreements
        self.save_benchmark(exact_results, stats)
        
        print("="*60)
        print(f"B4 COMPLETE: {len(exact_results)} exact agreement records")
        print("="*60)
        
        return exact_results, stats


def main():
    try:
        generator = AgreementBenchmarkGenerator()
        results, stats = generator.run()
        
        print(f"\nB4 benchmark generated: {len(results)} records")
        print(f"Exact agreement rate: {stats['exact_agreement_rate']:.1%}")
        
    except Exception as e:
        print(f"Error generating B4 benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

