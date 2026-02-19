
import json
from pathlib import Path
from collections import Counter
from typing import Dict, Any
import sys

from claim_verification.config import DataStructure


def count_labels_in_file(file_path: Path) -> Dict[str, Any]:
    """
    Count labels in a benchmark JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        Dictionary with label counts and file info
    """
    label_counts = Counter()
    total_instances = 0
    missing_labels = 0
    
    # Determine which label fields to check based on filename
    filename = file_path.stem
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
                total_instances += 1
                
                # Check for different label fields
                label = None
                if 'label' in entry:
                    label = entry['label']
                elif 'b2_label' in entry:
                    label = entry['b2_label']
                elif 'b3_label' in entry:
                    label = entry['b3_label']
                
                if label:
                    # Normalize label (handle case variations)
                    label_normalized = label.strip()
                    label_counts[label_normalized] += 1
                else:
                    missing_labels += 1
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line in {file_path.name}: {e}")
                continue
    
    return {
        'file': file_path.name,
        'total_instances': total_instances,
        'missing_labels': missing_labels,
        'label_counts': dict(label_counts),
        'unique_labels': sorted(label_counts.keys())
    }


def main():
    """Count labels in all benchmark files."""
    data_structure = DataStructure()
    benchmark_dir = data_structure.benchmark_dir
    
    # Find all JSONL files (excluding extracted_claims.json and verifiable_claims.json)
    jsonl_files = sorted([
        f for f in benchmark_dir.glob('*.jsonl')
        if f.name not in ['extracted_claims.json', 'verifiable_claims.json']
    ])
    
    if not jsonl_files:
        print(f"No JSONL files found in {benchmark_dir}")
        return
    
    print(f"Found {len(jsonl_files)} benchmark files\n")
    print("=" * 80)
    
    all_results = {}
    
    for file_path in jsonl_files:
        result = count_labels_in_file(file_path)
        all_results[result['file']] = result
        
        print(f"\nFile: {result['file']}")
        print(f"Total Instances: {result['total_instances']}")
        print(f"Missing Labels: {result['missing_labels']}")
        print(f"Unique Labels: {len(result['unique_labels'])}")
        print("\nLabel Distribution:")
        print("-" * 80)
        
        # Sort by count (descending)
        sorted_labels = sorted(
            result['label_counts'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for label, count in sorted_labels:
            percentage = (count / result['total_instances'] * 100) if result['total_instances'] > 0 else 0
            print(f"  {label:30s}: {count:5d} ({percentage:5.1f}%)")
        
        print("=" * 80)
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    
    # Get all unique labels across all files
    all_labels = set()
    for result in all_results.values():
        all_labels.update(result['unique_labels'])
    all_labels = sorted(all_labels)
    
    # Print header
    header = f"{'File':<30s} {'Total':>8s}"
    for label in all_labels:
        header += f" {label[:15]:>15s}"
    print(header)
    print("-" * len(header))
    
    # Print rows
    for filename in sorted(all_results.keys()):
        result = all_results[filename]
        row = f"{filename:<30s} {result['total_instances']:>8d}"
        for label in all_labels:
            count = result['label_counts'].get(label, 0)
            row += f" {count:>15d}"
        print(row)
    
    print("=" * 80)
    
    # Save to JSON
    output_file = benchmark_dir / 'label_counts.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
