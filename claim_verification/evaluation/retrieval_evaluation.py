
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import sys

# Token counting support
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not available for token counting. Install with: pip install tiktoken")


# Import from claim_extract
from claim_verification.retrieval.pipeline_wrapper import (
    retrieve_top_k_evidences_bm25,
    retrieve_top_k_evidences_tfidf,
    retrieve_top_k_evidences_sbert,
    retrieve_top_k_evidences_faiss,
    retrieve_top_k_evidences_rrf,
    retrieve_top_k_evidences_biencoder_crossencoder,
    retrieve_top_k_evidences_bm25_crossencoder,
    entailment_for_claim,
)

from claim_verification.preprocessing import chunking as _chunking


from claim_verification.config import DataStructure, as_str


RETRIEVER_FUNCS = {
    'bm25': retrieve_top_k_evidences_bm25,
    'tfidf': retrieve_top_k_evidences_tfidf,
    'sbert': retrieve_top_k_evidences_sbert,
    'faiss': retrieve_top_k_evidences_faiss,
    'rrf': retrieve_top_k_evidences_rrf,
    'biencoder-crossencoder': retrieve_top_k_evidences_biencoder_crossencoder,
    'bm25-cross': retrieve_top_k_evidences_bm25_crossencoder,
    'bm25-crossencoder': retrieve_top_k_evidences_bm25_crossencoder,
}


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Count the number of tokens in a text using tiktoken.
    
    Args:
        text: Text to count tokens for
        model: Model name to get the tokenizer for
        
    Returns:
        Number of tokens
    """
    if not TIKTOKEN_AVAILABLE:
        # Fallback: rough estimate based on character count
        return len(text) // 4
    
    try:
        # Try to get encoding for the specific model
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base (used by GPT-4, GPT-3.5-turbo)
            encoding = tiktoken.get_encoding("cl100k_base")
        
        return len(encoding.encode(text))
    except Exception as e:
        # Fallback: rough estimate
        return len(text) // 4


def create_verification_prompt_for_counting(claim: str, evidences: List[Dict[str, Any]]) -> str:
    """
    Create a verification prompt for token counting purposes.
    This mirrors the prompt structure used in entailment_for_claim.
    
    Args:
        claim: The claim to verify
        evidences: List of evidence chunks
        
    Returns:
        The full prompt text
    """
    # Combine evidences
    evidence_texts = []
    for i, ev in enumerate(evidences):
        text = ev.get('text', '') if isinstance(ev, dict) else str(ev)
        evidence_texts.append(f"Evidence {i+1}: {text}")
    
    evidence_combined = "\n\n".join(evidence_texts)
    
    prompt = f"""You are a factual‑verification API. Respond **ONLY** with a valid JSON object—no other text.

Label definitions you must use:
  • Supported: The paper's content fully backs the claim with no gaps or contradictions.
  • Partially Supported: Some parts of the claim align with the paper, but other details are missing or unclear.
  • Contradicted: The claim directly conflicts with the paper's content or established facts.
  • Not Determined: The paper's content is insufficient to confirm or deny the claim.

Task: Classify the claim using **exactly one** of the four labels above based on the evidence.

CLAIM:
{claim}

EVIDENCE:
{evidence_combined}

Return only valid JSON with "result" and "justification" keys."""
    
    return prompt


def find_chunks_for_paper(paper_id: str, data_structure: DataStructure) -> List[Dict[str, Any]]:
    """
    Find chunks for a paper, creating from markdown/PDF if needed.
    
    Args:
        paper_id: The paper ID
        data_structure: DataStructure instance for paths
        
    Returns:
        List of chunk dictionaries
    """
    # Check if chunks already exist
    chunk_file = data_structure.chunks_dir / f"{paper_id}_chunks.jsonl"
    if chunk_file.exists():
        return _chunking.load_chunks_from_file(str(chunk_file))
    
    # Try to find markdown and create chunks
    md_file = data_structure.markdown_dir / f"{paper_id}.md"
    if md_file.exists():
        print(f"Found markdown: {md_file}, creating chunks...")
        data_structure.chunks_dir.mkdir(parents=True, exist_ok=True)
        if _chunking.chunk_single_file(str(md_file), str(chunk_file)):
            return _chunking.load_chunks_from_file(str(chunk_file))
    
    # Try to find PDF and convert
    pdf_file = data_structure.pdfs_dir / f"{paper_id}.pdf"
    if pdf_file.exists():
        print(f"Found PDF: {pdf_file}, converting to markdown and creating chunks...")
        data_structure.markdown_dir.mkdir(parents=True, exist_ok=True)
        data_structure.chunks_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert PDF to markdown
        if _chunking.convert_pdf_to_markdown(str(pdf_file), str(md_file)):
            # Create chunks
            if _chunking.chunk_single_file(str(md_file), str(chunk_file)):
                return _chunking.load_chunks_from_file(str(chunk_file))
        else:
            print(f"Warning: Failed to convert PDF {pdf_file} to markdown")
    else:
        print(f"PDF not found at expected path: {pdf_file}")
    
    return []


def load_benchmark(path: str) -> List[Dict[str, Any]]:
    """
    Load benchmark JSONL file.
    """
    data: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                data.append(entry)
    return data


def get_claim_from_entry(entry: Dict[str, Any]) -> str:
    """Extract claim from entry."""
    return entry.get('claim', '')


def get_paper_id(entry: Dict[str, Any]) -> str:
    """Extract paper_id from entry."""
    return entry.get('paper_id', '')


def get_label_from_entry(entry: Dict[str, Any], benchmark_name: str = '') -> str:
    """
    Extract label from entry based on benchmark type.
    
    B1: label (always "Supported")
    B2: label
    B3: label
    B4: b2_label and/or b3_label (evaluate both separately)
    B5: same as B4
    """
    # For B4 and B5, we can evaluate both labels
    # Default to b2_label if available, otherwise label
    if 'b2_label' in entry:
        return entry.get('b2_label', 'Unknown')
    if 'b3_label' in entry:
        return entry.get('b3_label', 'Unknown')
    return entry.get('label', 'Unknown')


def get_benchmark_name(benchmark_path: str) -> str:
    """Extract benchmark name from file path."""
    path = Path(benchmark_path)
    stem = path.stem
    # Remove suffixes like _poster, _oral, etc.
    if '_' in stem and stem.startswith('B'):
        # Keep only the base name (e.g., B1_golden_supported)
        parts = stem.split('_')
        if len(parts) >= 3:
            return '_'.join(parts[:3])
    return stem


def evaluate_benchmark(
    benchmark_path: str,
    output_jsonl: str,
    retriever: str,
    llm_model: str,
    top_k: int,
    max_entries: int = None,
    benchmark_name: str = '',
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate benchmark with retrieval + verification.
    
    Args:
        benchmark_path: Path to benchmark JSONL file
        output_jsonl: Output path for results JSONL
        retriever: Retrieval method name
        llm_model: LLM model name
        top_k: Number of chunks to retrieve
        max_entries: Maximum number of entries to process
        benchmark_name: Benchmark name (auto-detected if empty)
        dry_run: If True, only count tokens without sending LLM requests
    """
    entries = load_benchmark(benchmark_path)
    if max_entries is not None:
        entries = entries[:max_entries]
    
    if not benchmark_name:
        benchmark_name = get_benchmark_name(benchmark_path)
    
    retrieve_fn = RETRIEVER_FUNCS.get(retriever)
    if retrieve_fn is None:
        raise ValueError(f"Unknown retriever: {retriever}")
    
    data_structure = DataStructure()
    results: List[Dict[str, Any]] = []
    skipped: List[str] = []
    
    # Token counting for dry run mode
    total_input_tokens = 0
    total_output_tokens_estimate = 0  # Estimate output tokens (~100 for JSON response)
    
    if dry_run:
        print(f"\n{'='*60}")
        print("DRY RUN MODE - Only counting tokens, not sending LLM requests")
        print(f"{'='*60}\n")
    
    for entry in entries:
        paper_id = get_paper_id(entry)
        claim = get_claim_from_entry(entry)
        label = get_label_from_entry(entry, benchmark_name)
        
        if not paper_id or not claim:
            skipped.append(f"{paper_id or 'unknown'}: missing paper_id or claim")
            continue
        
        # Ensure chunks exist (create from markdown/PDF if needed)
        chunks = find_chunks_for_paper(paper_id, data_structure)
        if not chunks:
            skipped.append(f"{paper_id}: no markdown/pdf to chunk")
            continue
        
        # Retrieve evidences
        evidences = retrieve_fn(claim, chunks, top_k=top_k)
        
        if dry_run:
            # Only count tokens, don't send LLM request
            prompt = create_verification_prompt_for_counting(claim, evidences)
            input_tokens = count_tokens(prompt, llm_model)
            total_input_tokens += input_tokens
            total_output_tokens_estimate += 100  # Rough estimate for JSON response
            
            result = {
                'paper_id': paper_id,
                'claim': claim,
                'ground_truth': label,
                'predicted': 'DRY_RUN',
                'justification': f'Dry run - counted {input_tokens} input tokens',
                'retriever': retriever,
                'llm_model': llm_model,
                'benchmark': benchmark_name,
                'input_tokens': input_tokens,
            }
        else:
            # Verify claim
            verification = entailment_for_claim(claim, evidences, llm_model)
            predicted = verification.get('result', 'Not Determined')
            
            result = {
                'paper_id': paper_id,
                'claim': claim,
                'ground_truth': label,
                'predicted': predicted,
                'justification': verification.get('justification', ''),
                'retriever': retriever,
                'llm_model': llm_model,
                'benchmark': benchmark_name,
            }
        
        # Include original entry fields
        for key in ['paper_title', 'paper_venue', 'paper_decision', 'decision']:
            if key in entry:
                result[key] = entry[key]
        
        results.append(result)
    
    # Save per-instance JSONL
    output_dir = os.path.dirname(output_jsonl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    # Compute overall accuracy (not meaningful for dry run)
    if results and not dry_run:
        correct = sum(1 for r in results if r['ground_truth'].lower() == r['predicted'].lower())
        accuracy = correct / len(results)
    else:
        accuracy = 0.0
    
    summary = {
        'benchmark': benchmark_name,
        'total_processed': len(results),
        'total_skipped': len(skipped),
        'accuracy': accuracy,
        'retriever': retriever,
        'llm_model': llm_model,
        'top_k': top_k,
        'dry_run': dry_run,
    }
    
    if dry_run:
        num_instances = len(results)
        avg_input_tokens = total_input_tokens / num_instances if num_instances > 0 else 0
        avg_output_tokens = total_output_tokens_estimate / num_instances if num_instances > 0 else 0
        summary['num_instances'] = num_instances
        summary['avg_input_tokens_per_instance'] = avg_input_tokens
        summary['avg_output_tokens_per_instance'] = avg_output_tokens
        summary['avg_tokens_per_instance'] = avg_input_tokens + avg_output_tokens
        print(f"\n{'='*60}")
        print("TOKEN COUNT SUMMARY (DRY RUN)")
        print(f"{'='*60}")
        print(f"Number of Instances: {num_instances:,}")
        print(f"Avg Input Tokens per Instance: {avg_input_tokens:,.1f}")
        print(f"Avg Output Tokens per Instance (estimate): {avg_output_tokens:,.1f}")
        print(f"Avg Tokens per Instance: {avg_input_tokens + avg_output_tokens:,.1f}")
        print(f"{'='*60}\n")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate benchmark with retrieval+verification')
    parser.add_argument('--benchmark', required=True, help='Path to benchmark JSONL file (or benchmark name: B1, B2, B3, B4, B5)')
    parser.add_argument('--output', help='Output JSONL path (default: auto-generated in data/experiments/)')
    parser.add_argument('--retriever', default='bm25-cross', choices=list(RETRIEVER_FUNCS.keys()))
    parser.add_argument('--llm-model', default='gpt-4o-mini', help='LLM model name')
    parser.add_argument('--top-k', type=int, default=3, help='Number of chunks to retrieve')
    parser.add_argument('--max-entries', type=int, help='Maximum number of entries to process')
    parser.add_argument('--dry-run', action='store_true', help='Only count tokens without sending LLM requests')
    
    args = parser.parse_args()
    
    # Determine benchmark path
    data_structure = DataStructure()
    benchmark_path = args.benchmark
    
    # If it's a benchmark name (B1, B2, etc.), resolve to file path
    if benchmark_path.upper() in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']:
        benchmark_map = {
            'B1': 'B1_golden_supported.jsonl',
            'B2': 'B2_reviewer_author.jsonl',
            'B3': 'B3_reviewer_paper.jsonl',
            'B4': 'B4_agreement.jsonl',
            'B5': 'B5_verifiable.jsonl',
            'B6': 'B6_human.jsonl'
        }
        benchmark_path = str(data_structure.benchmark_dir / benchmark_map[benchmark_path.upper()])
    
    if not Path(benchmark_path).exists():
        print(f"Error: Benchmark file not found: {benchmark_path}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_jsonl = args.output
    else:
        benchmark_name = get_benchmark_name(benchmark_path)
        model_name = args.llm_model.replace('/', '_').replace('-', '_')
        retriever_name = args.retriever.replace('-', '_')
        output_dir = data_structure.base_dir / 'experiments' / benchmark_name / retriever_name / model_name / f"top{args.top_k}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_jsonl = str(output_dir / 'results.jsonl')
    
    summary = evaluate_benchmark(
        benchmark_path=benchmark_path,
        output_jsonl=output_jsonl,
        retriever=args.retriever,
        llm_model=args.llm_model,
        top_k=args.top_k,
        max_entries=args.max_entries,
        dry_run=args.dry_run,
    )
    
    print(json.dumps(summary, indent=2))
    print(f"\nResults saved to: {output_jsonl}")


if __name__ == '__main__':
    main()
