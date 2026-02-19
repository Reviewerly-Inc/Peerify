
import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
import sys


from claim_verification.config import DataStructure


def load_metrics(metrics_path: Path) -> Optional[Dict[str, Any]]:
    """Load metrics from a JSON file."""
    try:
        with open(metrics_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {metrics_path}: {e}")
        return None


def extract_experiment_info(metrics_path: Path) -> Dict[str, str]:
    """
    Extract benchmark, technique, model, and top_k from metrics file path.
    
    Path format: experiments/{benchmark}/{technique}/{model}/{top_k}/metrics.json
    """
    parts = metrics_path.parts
    info = {
        'benchmark': 'unknown',
        'technique': 'unknown',
        'model': 'unknown',
        'top_k': 'unknown',
        'file_path': str(metrics_path)
    }
    
    # Find benchmark (B1, B2, etc.)
    for part in parts:
        if part.startswith('B') and ('_' in part or part in ['B1', 'B2', 'B3', 'B4', 'B5']):
            info['benchmark'] = part
            break
    
    # Find technique and model
    for i, part in enumerate(parts):
        if part in ['full_paper', 'bm25', 'tfidf', 'sbert', 'faiss', 'rrf', 
                    'biencoder_crossencoder', 'bm25_cross', 'bm25_crossencoder']:
            info['technique'] = part
            # Model is usually the next part
            if i + 1 < len(parts):
                info['model'] = parts[i + 1]
            # Top-k is usually after model (for retrieval techniques)
            if i + 2 < len(parts):
                top_k_part = parts[i + 2]
                if top_k_part.startswith('top'):
                    info['top_k'] = top_k_part.replace('top', '')
                elif top_k_part == 'metrics.json':
                    # For full_paper, there's no top_k
                    info['top_k'] = 'N/A'
            break
    
    return info


def flatten_metrics(metrics: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
    """
    Flatten nested metrics dictionary.
    
    Example:
    {
        'classification_metrics': {
            'accuracy': 0.85,
            'precision': 0.82
        }
    }
    becomes:
    {
        'classification_metrics.accuracy': 0.85,
        'classification_metrics.precision': 0.82
    }
    """
    flattened = {}
    for key, value in metrics.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_metrics(value, new_key))
        elif isinstance(value, list):
            # Convert lists to strings or handle special cases
            if len(value) > 0 and isinstance(value[0], (int, float, str)):
                flattened[new_key] = ', '.join(map(str, value))
            else:
                flattened[new_key] = str(value)
        else:
            flattened[new_key] = value
    return flattened


def collect_all_metrics(experiments_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Collect all metrics files and group them.
    
    Returns:
        Dictionary with keys:
        - 'full_paper': List of full_paper metrics
        - '{technique}': List of metrics for each technique
    """
    all_metrics = defaultdict(list)
    
    # Find all metrics.json files
    for metrics_file in experiments_dir.rglob('metrics.json'):
        metrics = load_metrics(metrics_file)
        if not metrics:
            continue
        
        info = extract_experiment_info(metrics_file)
        technique = info['technique']
        
        # Flatten metrics
        flattened = flatten_metrics(metrics)
        
        # Combine info and metrics
        row = {
            'benchmark': info['benchmark'],
            'technique': technique,
            'model': info['model'],
            'top_k': info['top_k'],
            **flattened
        }
        
        all_metrics[technique].append(row)
    
    return dict(all_metrics)


def write_csv(data: List[Dict[str, Any]], output_path: Path, technique: str):
    """Write metrics data to CSV file."""
    if not data:
        print(f"No data for {technique}, skipping CSV creation")
        return
    
    # Get all unique keys from all rows
    all_keys = set()
    for row in data:
        all_keys.update(row.keys())
    
    # Define column order (important fields first)
    priority_keys = ['benchmark', 'technique', 'model', 'top_k']
    other_keys = sorted([k for k in all_keys if k not in priority_keys])
    fieldnames = [k for k in priority_keys if k in all_keys] + other_keys
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"✓ Written {len(data)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate experiment metrics into CSV files'
    )
    parser.add_argument(
        '--experiments-dir',
        type=str,
        default='data/experiments',
        help='Base experiments directory (default: data/experiments)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/aggregated_metrics',
        help='Output directory for CSV files (default: data/aggregated_metrics)'
    )
    parser.add_argument(
        '--benchmark',
        type=str,
        help='Process only a specific benchmark (e.g., B1_golden_supported)'
    )
    
    args = parser.parse_args()
    
    experiments_dir = Path(args.experiments_dir)
    if not experiments_dir.exists():
        print(f"Error: Experiments directory not found: {experiments_dir}")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Collecting metrics from: {experiments_dir}")
    print(f"Output directory: {output_dir}")
    
    # Collect all metrics
    all_metrics = collect_all_metrics(experiments_dir)
    
    # Filter by benchmark if specified
    if args.benchmark:
        for technique in all_metrics:
            all_metrics[technique] = [
                row for row in all_metrics[technique]
                if args.benchmark in row.get('benchmark', '')
            ]
    
    # Separate full_paper from other techniques
    full_paper_data = all_metrics.pop('full_paper', [])
    technique_data = all_metrics
    
    # Write full_paper CSV
    if full_paper_data:
        full_paper_path = output_dir / 'full_paper_metrics.csv'
        write_csv(full_paper_data, full_paper_path, 'full_paper')
        print(f"\nFull paper metrics: {len(full_paper_data)} experiments")
    else:
        print("\nNo full_paper metrics found")
    
    # Write CSV for each technique
    print(f"\nTechnique metrics:")
    for technique, data in sorted(technique_data.items()):
        if data:
            technique_path = output_dir / f'{technique}_metrics.csv'
            write_csv(data, technique_path, technique)
            print(f"  {technique}: {len(data)} experiments")
    
    print(f"\n✓ All metrics aggregated to: {output_dir}")


if __name__ == '__main__':
    main()
