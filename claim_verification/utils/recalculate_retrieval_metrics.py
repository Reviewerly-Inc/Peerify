

import json
import argparse
import csv
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Import the fixed matching functions
try:
    from claim_verification.evaluate_retrieval_metrics import (
        match_chunk_to_evidence,
        calculate_recall_at_k,
        calculate_ndcg_at_k,
        calculate_mrr,
        normalize_text
    )
except ImportError:
    from evaluate_retrieval_metrics import (
        match_chunk_to_evidence,
        calculate_recall_at_k,
        calculate_ndcg_at_k,
        calculate_mrr,
        normalize_text
    )


def recalculate_metrics_for_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recalculate retrieval metrics for a single entry.
    
    Args:
        entry: Entry from results.jsonl with evaluation_result
        
    Returns:
        Updated entry with recalculated metrics
    """
    if 'evaluation_result' not in entry:
        return entry
    
    eval_result = entry['evaluation_result']
    if 'retrieval' not in eval_result:
        return entry
    
    retrieval = eval_result['retrieval']
    retrieved_chunks = retrieval.get('retrieved_chunks', [])
    ground_truth_evidence = entry.get('ground_truth_evidence', [])
    
    if not ground_truth_evidence:
        # No ground truth evidence, can't calculate metrics
        return entry
    
    # Recalculate matched chunks
    matched_chunks = []
    matched_evidence_indices = set()
    
    for i, chunk in enumerate(retrieved_chunks):
        for evidence_idx, evidence in enumerate(ground_truth_evidence):
            if match_chunk_to_evidence(chunk, evidence):
                if evidence_idx not in matched_evidence_indices:
                    matched_chunks.append({
                        'rank': i + 1,
                        'chunk_idx': chunk.get('chunk_idx', i + 1),
                        'section': chunk.get('section', 'unknown'),
                        'matched_evidence': {
                            'section': evidence.get('section', 'unknown'),
                            'subsection': evidence.get('subsection', ''),
                        }
                    })
                    matched_evidence_indices.add(evidence_idx)
                break
    
    # Update retrieval results
    retrieval['num_matched'] = len(matched_chunks)
    retrieval['matched_chunks'] = matched_chunks
    
    # Recalculate metrics
    k_values = [1, 3, 5, 10, 20]
    metrics = {}
    
    # Get top_k from retrieved chunks (use length if not specified)
    num_retrieved = len(retrieved_chunks)
    
    for k in k_values:
        if k <= num_retrieved:  # Only calculate if we have enough retrieved chunks
            recall = calculate_recall_at_k(retrieved_chunks, ground_truth_evidence, k)
            ndcg = calculate_ndcg_at_k(retrieved_chunks, ground_truth_evidence, k)
            metrics[f'recall@{k}'] = recall
            metrics[f'ndcg@{k}'] = ndcg
        else:
            # If not enough chunks, set to 0 or None
            metrics[f'recall@{k}'] = 0.0
            metrics[f'ndcg@{k}'] = 0.0
    
    mrr = calculate_mrr(retrieved_chunks, ground_truth_evidence)
    metrics['mrr'] = mrr
    
    # Update metrics in evaluation_result
    eval_result['metrics'] = metrics
    
    return entry


def process_results_file(results_path: Path, output_path: Path = None) -> None:
    """
    Process a results.jsonl file and recalculate metrics.
    
    Args:
        results_path: Path to input results.jsonl file
        output_path: Path to output file (if None, overwrites input)
    """
    if output_path is None:
        output_path = results_path
    
    print(f"Processing: {results_path}")
    
    entries = []
    with open(results_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    entry = json.loads(line)
                    updated_entry = recalculate_metrics_for_entry(entry)
                    entries.append(updated_entry)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping line {line_num} due to JSON error: {e}")
                    continue
    
    print(f"Processed {len(entries)} entries")
    
    # Write updated results
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Updated results written to: {output_path}")
    
    # Print summary statistics
    if entries:
        total = len(entries)
        total_matched = sum(1 for e in entries 
                          if e.get('evaluation_result', {}).get('retrieval', {}).get('num_matched', 0) > 0)
        
        # Calculate average metrics
        avg_recall_1 = np.mean([e.get('evaluation_result', {}).get('metrics', {}).get('recall@1', 0.0) 
                                for e in entries])
        avg_recall_10 = np.mean([e.get('evaluation_result', {}).get('metrics', {}).get('recall@10', 0.0) 
                                 for e in entries])
        avg_mrr = np.mean([e.get('evaluation_result', {}).get('metrics', {}).get('mrr', 0.0) 
                          for e in entries])
        
        print(f"\nSummary:")
        print(f"  Total entries: {total}")
        print(f"  Entries with matches: {total_matched} ({100*total_matched/total:.1f}%)")
        print(f"  Average Recall@1: {avg_recall_1:.3f}")
        print(f"  Average Recall@10: {avg_recall_10:.3f}")
        print(f"  Average MRR: {avg_mrr:.3f}")


def find_all_results_files(base_dir: Path) -> List[Path]:
    """
    Find all results.jsonl files in the experiments directory.
    
    Args:
        base_dir: Base directory (e.g., data/experiments)
        
    Returns:
        List of paths to results.jsonl files
    """
    results_files = []
    for results_file in base_dir.rglob('results.jsonl'):
        results_files.append(results_file)
    return sorted(results_files)


def extract_experiment_info(results_path: Path) -> Dict[str, str]:
    """
    Extract benchmark, retriever, model, and top-k from results file path.
    
    Args:
        results_path: Path to results.jsonl file
        
    Returns:
        Dictionary with benchmark, retriever, model, top_k
    """
    parts = results_path.parts
    info = {
        'benchmark': 'unknown',
        'retriever': 'unknown',
        'model': 'unknown',
        'top_k': 'unknown',
        'file_path': str(results_path)
    }
    
    # Find benchmark (B1, B2, etc.)
    for part in parts:
        if part.startswith('B') and ('_' in part or part in ['B1', 'B2', 'B3', 'B4', 'B5']):
            info['benchmark'] = part
            break
    
    # Find retriever (parent of model directory)
    for i, part in enumerate(parts):
        if part in ['bm25', 'tfidf', 'sbert', 'faiss', 'rrf', 'biencoder_crossencoder', 
                    'bm25_cross', 'bm25_crossencoder', 'full_paper']:
            info['retriever'] = part
            # Model is usually the next part
            if i + 1 < len(parts):
                info['model'] = parts[i + 1]
            # Top-k is usually after model
            if i + 2 < len(parts):
                top_k_part = parts[i + 2]
                if top_k_part.startswith('top'):
                    info['top_k'] = top_k_part.replace('top', '')
            break
    
    return info


def generate_csv_summary(results_files: List[Path], output_csv: Path) -> None:
    """
    Generate a CSV summary of all results.
    
    Args:
        results_files: List of paths to results.jsonl files
        output_csv: Path to output CSV file
    """
    rows = []
    
    for results_file in results_files:
        try:
            entries = []
            with open(results_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))
            
            if not entries:
                continue
            
            # Calculate aggregate metrics
            total = len(entries)
            matched = sum(1 for e in entries 
                        if e.get('evaluation_result', {}).get('retrieval', {}).get('num_matched', 0) > 0)
            
            avg_recall_1 = np.mean([e.get('evaluation_result', {}).get('metrics', {}).get('recall@1', 0.0) 
                                   for e in entries])
            avg_recall_3 = np.mean([e.get('evaluation_result', {}).get('metrics', {}).get('recall@3', 0.0) 
                                  for e in entries])
            avg_recall_5 = np.mean([e.get('evaluation_result', {}).get('metrics', {}).get('recall@5', 0.0) 
                                  for e in entries])
            avg_recall_10 = np.mean([e.get('evaluation_result', {}).get('metrics', {}).get('recall@10', 0.0) 
                                   for e in entries])
            avg_recall_20 = np.mean([e.get('evaluation_result', {}).get('metrics', {}).get('recall@20', 0.0) 
                                   for e in entries])
            
            avg_ndcg_1 = np.mean([e.get('evaluation_result', {}).get('metrics', {}).get('ndcg@1', 0.0) 
                                for e in entries])
            avg_ndcg_3 = np.mean([e.get('evaluation_result', {}).get('metrics', {}).get('ndcg@3', 0.0) 
                                 for e in entries])
            avg_ndcg_5 = np.mean([e.get('evaluation_result', {}).get('metrics', {}).get('ndcg@5', 0.0) 
                                 for e in entries])
            avg_ndcg_10 = np.mean([e.get('evaluation_result', {}).get('metrics', {}).get('ndcg@10', 0.0) 
                                 for e in entries])
            avg_ndcg_20 = np.mean([e.get('evaluation_result', {}).get('metrics', {}).get('ndcg@20', 0.0) 
                                 for e in entries])
            
            avg_mrr = np.mean([e.get('evaluation_result', {}).get('metrics', {}).get('mrr', 0.0) 
                             for e in entries])
            
            # Extract experiment info
            info = extract_experiment_info(results_file)
            
            # Classification metrics (if available)
            verification_results = [e.get('evaluation_result', {}).get('verification', {}) for e in entries]
            if verification_results and verification_results[0]:
                correct = sum(1 for v in verification_results if v.get('is_correct', False))
                accuracy = correct / total if total > 0 else 0.0
            else:
                accuracy = None
            
            row = {
                'benchmark': info['benchmark'],
                'retriever': info['retriever'],
                'model': info['model'],
                'top_k': info['top_k'],
                'total_entries': total,
                'matched_entries': matched,
                'match_rate': f"{100*matched/total:.2f}%" if total > 0 else "0.00%",
                'recall@1': f"{avg_recall_1:.4f}",
                'recall@3': f"{avg_recall_3:.4f}",
                'recall@5': f"{avg_recall_5:.4f}",
                'recall@10': f"{avg_recall_10:.4f}",
                'recall@20': f"{avg_recall_20:.4f}",
                'ndcg@1': f"{avg_ndcg_1:.4f}",
                'ndcg@3': f"{avg_ndcg_3:.4f}",
                'ndcg@5': f"{avg_ndcg_5:.4f}",
                'ndcg@10': f"{avg_ndcg_10:.4f}",
                'ndcg@20': f"{avg_ndcg_20:.4f}",
                'mrr': f"{avg_mrr:.4f}",
                'accuracy': f"{accuracy:.4f}" if accuracy is not None else "N/A",
                'file_path': info['file_path']
            }
            rows.append(row)
        except Exception as e:
            print(f"Error processing {results_file}: {e}")
            continue
    
    # Write CSV
    if rows:
        fieldnames = ['benchmark', 'retriever', 'model', 'top_k', 'total_entries', 'matched_entries', 
                     'match_rate', 'recall@1', 'recall@3', 'recall@5', 'recall@10', 'recall@20',
                     'ndcg@1', 'ndcg@3', 'ndcg@5', 'ndcg@10', 'ndcg@20', 'mrr', 'accuracy', 'file_path']
        
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"\nCSV summary written to: {output_csv}")
        print(f"Total experiments: {len(rows)}")
    else:
        print("No results to write to CSV")


def main():
    parser = argparse.ArgumentParser(
        description='Recalculate retrieval metrics from existing results.jsonl files'
    )
    parser.add_argument(
        '--results-file',
        type=str,
        help='Path to a specific results.jsonl file to process'
    )
    parser.add_argument(
        '--experiments-dir',
        type=str,
        default='data/experiments',
        help='Base experiments directory to process all results.jsonl files (default: data/experiments)'
    )
    parser.add_argument(
        '--benchmark',
        type=str,
        help='Process only results for a specific benchmark (e.g., B1_golden_supported)'
    )
    parser.add_argument(
        '--retriever',
        type=str,
        help='Process only results for a specific retriever'
    )
    parser.add_argument(
        '--output-suffix',
        type=str,
        default='',
        help='Suffix to add to output filename (default: overwrites original)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without making changes'
    )
    parser.add_argument(
        '--csv-output',
        type=str,
        default='retrieval_metrics_summary.csv',
        help='Path to output CSV file with summary (default: retrieval_metrics_summary.csv)'
    )
    
    args = parser.parse_args()
    
    if args.results_file:
        # Process single file
        results_path = Path(args.results_file)
        if not results_path.exists():
            print(f"Error: File not found: {results_path}")
            return
        
        if args.dry_run:
            print(f"Would process: {results_path}")
            return
        
        output_path = results_path
        if args.output_suffix:
            output_path = results_path.parent / f"{results_path.stem}{args.output_suffix}{results_path.suffix}"
        
        process_results_file(results_path, output_path)
    else:
        # Process all files in experiments directory
        experiments_dir = Path(args.experiments_dir)
        if not experiments_dir.exists():
            print(f"Error: Directory not found: {experiments_dir}")
            return
        
        results_files = find_all_results_files(experiments_dir)
        
        # Filter by benchmark if specified
        if args.benchmark:
            results_files = [f for f in results_files if args.benchmark in str(f)]
        
        # Filter by retriever if specified
        if args.retriever:
            results_files = [f for f in results_files if args.retriever in str(f)]
        
        if not results_files:
            print("No results.jsonl files found matching criteria")
            return
        
        print(f"Found {len(results_files)} results.jsonl files to process")
        
        if args.dry_run:
            for f in results_files:
                print(f"Would process: {f}")
            return
        
        # Process all files
        for results_file in results_files:
            output_path = results_file
            if args.output_suffix:
                output_path = results_file.parent / f"{results_file.stem}{args.output_suffix}{results_file.suffix}"
            
            process_results_file(results_file, output_path)
            print()
        
        # Generate CSV summary
        csv_path = Path(args.csv_output)
        if not csv_path.is_absolute():
            csv_path = Path.cwd() / csv_path
        generate_csv_summary(results_files, csv_path)
        
        # Generate CSV summary
        csv_path = Path(args.csv_output)
        if not csv_path.is_absolute():
            csv_path = Path.cwd() / csv_path
        generate_csv_summary(results_files, csv_path)


if __name__ == '__main__':
    main()
