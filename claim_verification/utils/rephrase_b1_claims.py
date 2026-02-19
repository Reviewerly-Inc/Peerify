
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from claim_verification.config import DataStructure, DEFAULT_MODEL
from openai import OpenAI


def create_rephrase_prompt(claim: str) -> str:
    """Create a prompt for rephrasing a claim."""
    prompt = f"""Rephrase the following claim completely using different wording. 
The rephrased version should have the same meaning but use entirely different words and sentence structure.
Do not use the same key phrases or terminology if possible.

Original claim:
{claim}

Rephrased claim:"""
    return prompt


def rephrase_claim(claim: str, model: str, client: OpenAI) -> str:
    """
    Rephrase a single claim using the LLM.
    
    Args:
        claim: Original claim text
        model: LLM model name
        client: OpenAI client
        
    Returns:
        Rephrased claim text
    """
    prompt = create_rephrase_prompt(claim)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that rephrases claims while preserving their meaning."},
                {"role": "user", "content": prompt}
            ],
          
        )
        
        rephrased = response.choices[0].message.content.strip()
        return rephrased
    except Exception as e:
        print(f"Error rephrasing claim: {e}")
        return claim  # Return original if rephrasing fails


def rephrase_claim_entry(entry: Dict[str, Any], model: str, client: OpenAI) -> Dict[str, Any]:
    """
    Rephrase a single benchmark entry's claim.
    
    Args:
        entry: Benchmark entry dictionary
        model: LLM model name
        client: OpenAI client
        
    Returns:
        Entry with rephrased claim
    """
    original_claim = entry.get('claim', '')
    if not original_claim:
        return entry
    
    rephrased_claim = rephrase_claim(original_claim, model, client)
    
    # Create new entry with rephrased claim
    new_entry = entry.copy()
    new_entry['claim'] = rephrased_claim
    new_entry['original_claim'] = original_claim  # Keep original for reference
    new_entry['rephrasing_timestamp'] = datetime.now().isoformat()
    new_entry['rephrasing_model'] = model
    
    return new_entry


def rephrase_b1_benchmark(
    model: str = DEFAULT_MODEL,
    max_workers: int = 5,
    max_entries: int = None,
    openai_api_key: str = None
) -> List[Dict[str, Any]]:
    """
    Rephrase all claims in B1 benchmark.
    
    Args:
        model: LLM model name
        max_workers: Number of parallel workers
        max_entries: Maximum number of entries to process (None = all)
        openai_api_key: OpenAI API key (None = use env var)
        
    Returns:
        List of entries with rephrased claims
    """
    data_structure = DataStructure()
    b1_file = data_structure.get_benchmark_file("B1_golden_supported")
    
    if not b1_file.exists():
        raise FileNotFoundError(f"B1 benchmark not found: {b1_file}")
    
    # Load entries
    print(f"Loading B1 benchmark from {b1_file}...")
    entries = []
    with open(b1_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    
    if max_entries:
        entries = entries[:max_entries]
    
    print(f"Loaded {len(entries)} entries")
    print(f"Rephrasing claims using model: {model}")
    print(f"Using {max_workers} parallel workers")
    
    # Initialize OpenAI client
    if openai_api_key:
        client = OpenAI(api_key=openai_api_key)
    else:
        client = OpenAI()  # Uses OPENAI_API_KEY env var
    
    # Rephrase claims in parallel
    rephrased_entries = []
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(rephrase_claim_entry, entry, model, client): i
            for i, entry in enumerate(entries)
        }
        
        for future in as_completed(futures):
            idx = futures[future]
            try:
                rephrased_entry = future.result()
                rephrased_entries.append((idx, rephrased_entry))
            except Exception as e:
                print(f"Error processing entry {idx}: {e}")
                failed_count += 1
                # Keep original entry if rephrasing fails
                rephrased_entries.append((idx, entries[idx]))
    
    # Sort by original index to maintain order
    rephrased_entries.sort(key=lambda x: x[0])
    final_entries = [entry for _, entry in rephrased_entries]
    
    print(f"\nRephrasing complete!")
    print(f"  Successfully rephrased: {len(final_entries) - failed_count}")
    print(f"  Failed: {failed_count}")
    
    return final_entries


def save_rephrased_benchmark(entries: List[Dict[str, Any]], backup: bool = True):
    """
    Save rephrased entries back to B1 benchmark file.
    
    Args:
        entries: List of entries with rephrased claims
        backup: Whether to backup the original file
    """
    data_structure = DataStructure()
    b1_file = data_structure.get_benchmark_file("B1_golden_supported")
    
    # Backup original file
    if backup and b1_file.exists():
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = b1_file.parent / f"B1_golden_supported_backup_{timestamp}.jsonl"
        print(f"\nCreating backup: {backup_file}")
        import shutil
        shutil.copy2(b1_file, backup_file)
    
    # Save rephrased entries
    print(f"\nSaving rephrased B1 benchmark to {b1_file}...")
    with open(b1_file, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(entries)} entries to {b1_file}")
    
    # Save summary
    summary = {
        'benchmark': 'B1_golden_supported',
        'rephrasing_timestamp': datetime.now().isoformat(),
        'total_claims': len(entries),
        'model': entries[0].get('rephrasing_model') if entries else None,
    }
    
    summary_file = data_structure.logs_dir / "B1_rephrasing_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary to {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Rephrase all claims in B1 benchmark using LLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Rephrase all claims in B1
  python3 -m claim_verification.rephrase_b1_claims

  # Use specific model
  python3 -m claim_verification.rephrase_b1_claims --model gpt-4o

  # Test with limited entries
  python3 -m claim_verification.rephrase_b1_claims --max-entries 10

  # More parallel workers
  python3 -m claim_verification.rephrase_b1_claims --max-workers 10
        """
    )
    
    parser.add_argument(
        '--model',
        default=DEFAULT_MODEL,
        help=f'LLM model name (default: {DEFAULT_MODEL})'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=5,
        help='Number of parallel workers (default: 5)'
    )
    
    parser.add_argument(
        '--max-entries',
        type=int,
        help='Maximum number of entries to process (default: all)'
    )
    
    parser.add_argument(
        '--openai-api-key',
        help='OpenAI API key (default: uses OPENAI_API_KEY env var)'
    )
    
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create backup of original file'
    )
    
    args = parser.parse_args()
    
    # Rephrase claims
    rephrased_entries = rephrase_b1_benchmark(
        model=args.model,
        max_workers=args.max_workers,
        max_entries=args.max_entries,
        openai_api_key=args.openai_api_key
    )
    
    # Save rephrased benchmark
    save_rephrased_benchmark(rephrased_entries, backup=not args.no_backup)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
