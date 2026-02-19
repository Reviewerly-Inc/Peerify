

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
import sys

# Token counting support
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not available for token counting. Install with: pip install tiktoken")

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Classification metrics will not be calculated.")


# Import from claim_extract
from claim_verification.retrieval.pipeline_wrapper import (
    retrieve_top_k_evidences_bm25,
    entailment_for_claim,
)

# Import from real_benchmark_evaluation
try:
    from claim_verification.evaluation.retrieval_evaluation import (
        find_chunks_for_paper,
        RETRIEVER_FUNCS,
        load_benchmark,
        get_claim_from_entry,
        get_paper_id,
        get_label_from_entry,
        get_benchmark_name,
    )
except ImportError:
    # Fallback for when running directly from this directory
    from claim_verification.evaluation.retrieval_evaluation import (
        find_chunks_for_paper,
        RETRIEVER_FUNCS,
        load_benchmark,
        get_claim_from_entry,
        get_paper_id,
        get_label_from_entry,
        get_benchmark_name,
    )
from claim_verification.config import DataStructure


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


def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, strip whitespace)."""
    return text.lower().strip()


def extract_paragraph_text(chunk: Dict[str, Any]) -> str:
    """Extract paragraph text from a chunk."""
    text = chunk.get('text', '')
    # Take first 500 chars to match evidence source format
    return normalize_text(text[:500])


def extract_parent_section(section_name: str) -> str:
    """
    Extract parent section identifier from section name.
    
    Examples:
    - "4.2 Transfer on 3D Object Detection" -> "4.2"
    - "Introduction" -> "Introduction"
    - "A.3 Experimental Setup" -> "A.3"
    - "## 4.2 Transfer" -> "4.2"
    """
    import re
    # Remove markdown headers
    section_name = re.sub(r'^#+\s*', '', section_name.strip())
    
    # Try to extract numbered section (e.g., "4.2", "A.3", "1.1")
    match = re.match(r'^([A-Z]?\.?\d+(?:\.\d+)*)', section_name)
    if match:
        return match.group(1)
    
    # If no number, return the first significant word (for named sections like "Introduction")
    words = section_name.split()
    if words:
        # Return first word if it's a common section name, otherwise first 2 words
        first_word = words[0].lower()
        if first_word in ['introduction', 'abstract', 'conclusion', 'related', 'methodology', 'methods', 'experiments', 'results', 'discussion', 'appendix']:
            return first_word
        # For other cases, return first 2 words as parent
        return ' '.join(words[:2]).lower() if len(words) >= 2 else first_word
    
    return section_name.lower()


def match_chunk_to_evidence(
    chunk: Dict[str, Any],
    evidence: Dict[str, Any]
) -> bool:
    """
    Check if a chunk matches a ground truth evidence.
    
    Matches based on:
    - Parent section name (e.g., "4.2", "Introduction") - checks both chunk.section field AND chunk text
    - Paragraph text overlap (first 500 chars)
    
    Returns True if chunk matches the evidence.
    """
    chunk_section = normalize_text(chunk.get('section', 'unknown'))
    evidence_section = normalize_text(evidence.get('section', 'unknown'))
    evidence_section_original = evidence.get('section', 'unknown')  # Keep original for text matching
    chunk_text = normalize_text(chunk.get('text', ''))
    chunk_text_original = chunk.get('text', '')  # Keep original for case-sensitive matching
    
    # Extract parent sections for matching
    evidence_parent = extract_parent_section(evidence_section_original)
    chunk_parent_from_field = extract_parent_section(chunk_section)
    
    # Section matching: check if parent sections match OR if evidence section appears in chunk text
    section_matches = False
    if evidence_section != 'unknown' and evidence_parent:
        # First check if parent sections match
        if chunk_parent_from_field == evidence_parent:
            section_matches = True
        else:
            # Check if evidence section name appears in chunk text (as markdown header or plain text)
            # Look for markdown headers: ## Section Name or ### Section Name
            # Try both normalized and original case
            section_patterns = [
                f"## {evidence_section_original}",  # Original case with ##
                f"### {evidence_section_original}",  # Original case with ###
                f"# {evidence_section_original}",    # Original case with #
                evidence_section_original,            # Original case plain
                f"## {evidence_section}",            # Normalized with ##
                f"### {evidence_section}",           # Normalized with ###
                f"# {evidence_section}",             # Normalized with #
                evidence_section,                     # Normalized plain
                f"## {evidence_parent}",              # Parent section with ##
                f"### {evidence_parent}",             # Parent section with ###
                f"# {evidence_parent}",               # Parent section with #
                evidence_parent                       # Parent section plain
            ]
            for pattern in section_patterns:
                # Check in both normalized and original text
                if pattern in chunk_text or pattern in chunk_text_original:
                    section_matches = True
                    break
            
            # Also check if parent section appears in chunk text
            if not section_matches and evidence_parent:
                if evidence_parent in chunk_text or evidence_parent in chunk_text_original:
                    section_matches = True
    else:
        # If evidence section is unknown, don't require section match
        section_matches = True
    
    if not section_matches:
        return False
    
    # If parent sections match, we consider it a match (no need for exact paragraph match)
    # This is more lenient - if both are from the same section (e.g., "4.2"), it's a match
    if evidence_parent and chunk_parent_from_field == evidence_parent:
        return True
    
    # If sections match via text (evidence section found in chunk text), also consider it a match
    # We've already verified section_matches is True, so if we got here, the section was found in text
    # In this case, we can be lenient and return True if parent sections are close
    if section_matches:
        # Check if there's any paragraph overlap as additional confirmation
        evidence_para = normalize_text(evidence.get('paragraph', ''))
        chunk_para = extract_paragraph_text(chunk)
        
        if evidence_para:
            # If evidence paragraph is substantial, check for overlap
            if len(evidence_para) > 30:  # Only check if evidence para is substantial
                # First try exact substring match
                if evidence_para in chunk_para:
                    return True
                
                # Calculate word overlap for fuzzy matching
                evidence_words = set(evidence_para.split())
                chunk_words = set(chunk_para.split())
                
                # Remove very short words (likely noise)
                evidence_words = {w for w in evidence_words if len(w) > 2}
                chunk_words = {w for w in chunk_words if len(w) > 2}
                
                if len(evidence_words) > 0:
                    overlap = len(evidence_words & chunk_words) / len(evidence_words)
                    # Require at least 30% word overlap (more lenient since we're matching by parent section)
                    if overlap >= 0.3:
                        return True
            else:
                # For short paragraphs, require exact match
                if evidence_para in chunk_para:
                    return True
        
        # If section matches but paragraph doesn't, still return True if parent sections match
        # This handles cases where chunks are from the same section but different paragraphs
        return True
    
    return True


def get_ground_truth_evidence(entry: Dict[str, Any], benchmark_name: str = '') -> List[Dict[str, Any]]:
    """
    Extract ground truth evidence from a benchmark entry.
    
    Handles different benchmark formats:
    - B1: section_name, subsection_name, paragraph fields directly in entry
    - B2/B3/B4/B5: evidence field contains text (not structured), may not be calculable
    """
    evidence_list = []
    
    # B1 has structured evidence: section_name, subsection_name, paragraph
    if 'section_name' in entry or 'paragraph' in entry:
        evidence = {}
        if 'section_name' in entry:
            evidence['section'] = entry['section_name']
        elif 'section' in entry:
            evidence['section'] = entry['section']
        
        if 'subsection_name' in entry:
            evidence['subsection'] = entry['subsection_name']
        elif 'subsection' in entry:
            evidence['subsection'] = entry['subsection']
        
        if 'paragraph' in entry:
            evidence['paragraph'] = entry['paragraph']
        
        if evidence:
            evidence_list = [evidence]
    
    # B2/B3/B4/B5 have evidence field (text, not structured)
    # For these, we can't calculate structured retrieval metrics
    # But we can still calculate classification metrics
    
    return evidence_list


def calculate_recall_at_k(
    retrieved_chunks: List[Dict[str, Any]],
    ground_truth_evidence: List[Dict[str, Any]],
    k: int
) -> float:
    """
    Calculate Recall@K: fraction of relevant chunks found in top-K.
    
    Args:
        retrieved_chunks: List of retrieved chunks (ordered by relevance)
        ground_truth_evidence: List of ground truth evidence dicts
        k: Top K chunks to consider
        
    Returns:
        Recall@K value (0.0 to 1.0)
    """
    if not ground_truth_evidence:
        return 0.0
    
    top_k_chunks = retrieved_chunks[:k]
    
    # Count how many ground truth evidence items are matched
    matched_evidence = set()
    for evidence in ground_truth_evidence:
        for chunk in top_k_chunks:
            if match_chunk_to_evidence(chunk, evidence):
                matched_evidence.add(id(evidence))  # Use id to track unique evidence
                break
    
    return len(matched_evidence) / len(ground_truth_evidence)


def calculate_ndcg_at_k(
    retrieved_chunks: List[Dict[str, Any]],
    ground_truth_evidence: List[Dict[str, Any]],
    k: int
) -> float:
    """
    Calculate NDCG@K: Normalized Discounted Cumulative Gain at K.
    
    Args:
        retrieved_chunks: List of retrieved chunks (ordered by relevance)
        ground_truth_evidence: List of ground truth evidence dicts
        k: Top K chunks to consider
        
    Returns:
        NDCG@K value (0.0 to 1.0)
    """
    if not ground_truth_evidence:
        return 0.0
    
    top_k_chunks = retrieved_chunks[:k]
    
    # Calculate DCG: relevance is 1 if chunk matches any evidence, 0 otherwise
    dcg = 0.0
    for i, chunk in enumerate(top_k_chunks):
        relevance = 0
        for evidence in ground_truth_evidence:
            if match_chunk_to_evidence(chunk, evidence):
                relevance = 1
                break
        
        # DCG: relevance / log2(rank + 1)
        rank = i + 1
        dcg += relevance / np.log2(rank + 1)
    
    # Calculate IDCG: ideal DCG (all relevant chunks at top)
    num_relevant = len(ground_truth_evidence)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(num_relevant, k)))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def calculate_mrr(
    retrieved_chunks: List[Dict[str, Any]],
    ground_truth_evidence: List[Dict[str, Any]]
) -> float:
    """
    Calculate MRR: Mean Reciprocal Rank.
    
    Returns the reciprocal of the rank of the first relevant chunk.
    If no relevant chunk is found, returns 0.0.
    """
    if not ground_truth_evidence:
        return 0.0
    
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        for evidence in ground_truth_evidence:
            if match_chunk_to_evidence(chunk, evidence):
                return 1.0 / rank
    
    return 0.0


def calculate_classification_metrics(
    ground_truth_labels: List[str],
    predicted_labels: List[str]
) -> Dict[str, Any]:
    """
    Calculate classification metrics: accuracy, precision, recall, F1 per label.
    
    Labels: Supported, Partially Supported, Contradicted, Not Determined
    """
    if not SKLEARN_AVAILABLE:
        return {
            'error': 'sklearn not available',
            'accuracy': 0.0,
        }
    
    # Normalize label names (handle variations)
    label_mapping = {
        'supported': 'Supported',
        'partially supported': 'Partially Supported',
        'contradicted': 'Contradicted',
        'not determined': 'Not Determined',
        'undetermined': 'Not Determined',
        'not-determined': 'Not Determined',
    }
    
    normalized_gt = [label_mapping.get(gt.lower(), gt) for gt in ground_truth_labels]
    normalized_pred = [label_mapping.get(pred.lower(), pred) for pred in predicted_labels]
    
    # Get unique labels
    all_labels = sorted(set(normalized_gt + normalized_pred))
    
    # Overall accuracy
    accuracy = accuracy_score(normalized_gt, normalized_pred)
    
    # Per-label metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        normalized_gt, normalized_pred, labels=all_labels, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(normalized_gt, normalized_pred, labels=all_labels)
    
    # Per-label metrics as dict
    per_label_metrics = {}
    for i, label in enumerate(all_labels):
        per_label_metrics[label] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }
    
    # Macro averages
    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))
    
    # Weighted averages
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        normalized_gt, normalized_pred, labels=all_labels, average='weighted', zero_division=0
    )
    
    return {
        'accuracy': float(accuracy),
        'per_label': per_label_metrics,
        'macro_avg': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1,
        },
        'weighted_avg': {
            'precision': float(weighted_precision),
            'recall': float(weighted_recall),
            'f1': float(weighted_f1),
        },
        'confusion_matrix': {
            'labels': all_labels,
            'matrix': cm.tolist(),
        }
    }


def evaluate_retrieval_and_verification_for_claim(
    claim: str,
    paper_id: str,
    ground_truth_evidence: List[Dict[str, Any]],
    ground_truth_label: str,
    chunks: List[Dict[str, Any]],
    retrieve_fn,
    llm_model: str,
    top_k: int = 10,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Evaluate both retrieval and verification for a single claim.
    
    Args:
        dry_run: If True, only count tokens without sending LLM requests
    
    Returns:
        Dictionary with retrieval results, verification results, and all metrics
    """
    # Step 1: Retrieve top-K chunks
    retrieved_chunks = retrieve_fn(claim, chunks, top_k=top_k)
    
    # Step 2: Calculate retrieval metrics (only if we have structured ground truth)
    k_values = [1, 3, 5, 10, 20]
    retrieval_metrics = {}
    
    if ground_truth_evidence:
        for k in k_values:
            if k <= top_k:
                retrieval_metrics[f'recall@{k}'] = calculate_recall_at_k(retrieved_chunks, ground_truth_evidence, k)
                retrieval_metrics[f'ndcg@{k}'] = calculate_ndcg_at_k(retrieved_chunks, ground_truth_evidence, k)
        
        retrieval_metrics['mrr'] = calculate_mrr(retrieved_chunks, ground_truth_evidence)
        
        # Find which retrieved chunks match ground truth
        matched_chunks = []
        for i, chunk in enumerate(retrieved_chunks):
            for evidence in ground_truth_evidence:
                if match_chunk_to_evidence(chunk, evidence):
                    matched_chunks.append({
                        'rank': i + 1,
                        'chunk_idx': chunk.get('idx', i + 1),
                        'section': chunk.get('section', 'unknown'),
                        'matched_evidence': {
                            'section': evidence.get('section', 'unknown'),
                            'subsection': evidence.get('subsection', ''),
                        }
                    })
                    break
    else:
        matched_chunks = []
    
    # Prepare retrieved chunks for output (include text, section, idx)
    retrieved_chunks_output = []
    for i, chunk in enumerate(retrieved_chunks):
        chunk_info = {
            'rank': i + 1,
            'chunk_idx': chunk.get('idx', i + 1),
            'section': chunk.get('section', 'unknown'),
            'subsection': chunk.get('subsection', ''),
            'text': chunk.get('text', '')[:500],  # First 500 chars for readability
        }
        retrieved_chunks_output.append(chunk_info)
    
    # Step 3: Verify claim (predict label) or count tokens in dry run mode
    input_tokens = 0
    if dry_run:
        # Only count tokens, don't send LLM request
        prompt = create_verification_prompt_for_counting(claim, retrieved_chunks)
        input_tokens = count_tokens(prompt, llm_model)
        predicted_label = 'DRY_RUN'
        justification = f'Dry run - counted {input_tokens} input tokens'
    else:
        verification_result = entailment_for_claim(claim, retrieved_chunks, llm_model)
        predicted_label = verification_result.get('result', 'Not Determined')
        justification = verification_result.get('justification', '')
    
    # Step 4: Check if prediction matches ground truth
    is_correct = (predicted_label.lower() == ground_truth_label.lower()) if not dry_run else False
    
    result = {
        'retrieval': {
            'num_retrieved': len(retrieved_chunks),
            'num_ground_truth': len(ground_truth_evidence),
            'num_matched': len(matched_chunks),
            'matched_chunks': matched_chunks,
            'retrieved_chunks': retrieved_chunks_output,
            'metrics': retrieval_metrics,
        },
        'verification': {
            'ground_truth_label': ground_truth_label,
            'predicted_label': predicted_label,
            'justification': justification,
            'is_correct': is_correct,
        }
    }
    
    if dry_run:
        result['input_tokens'] = input_tokens
    
    return result


def evaluate_benchmark_retrieval_and_verification(
    benchmark_path: str,
    retriever: str,
    llm_model: str,
    top_k: int = 10,
    max_entries: Optional[int] = None,
    output_dir: Optional[str] = None,
    benchmark_name: str = '',
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate both retrieval and verification metrics for a benchmark.
    
    Args:
        benchmark_path: Path to benchmark JSONL file
        retriever: Retriever name (bm25, tfidf, sbert, etc.)
        llm_model: LLM model for verification
        top_k: Number of chunks to retrieve
        max_entries: Maximum number of entries to process (None = all)
        output_dir: Output directory for results (auto-generated if None)
        benchmark_name: Benchmark name (auto-detected if empty)
        dry_run: If True, only count tokens without sending LLM requests
        
    Returns:
        Dictionary with aggregated metrics (retrieval + classification)
    """
    entries = load_benchmark(benchmark_path)
    
    if max_entries is not None:
        entries = entries[:max_entries]
    
    if not benchmark_name:
        benchmark_name = get_benchmark_name(benchmark_path)
    
    # Get retriever function
    retrieve_fn = RETRIEVER_FUNCS.get(retriever)
    if retrieve_fn is None:
        raise ValueError(f"Unknown retriever: {retriever}")
    
    data_structure = DataStructure()
    
    # Determine output directory
    if output_dir is None:
        model_name = llm_model.replace('/', '_').replace('-', '_')
        retriever_name = retriever.replace('-', '_')
        output_dir = data_structure.base_dir / 'experiments' / benchmark_name / retriever_name / model_name / f'top{top_k}'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if dry_run:
        print(f"\n{'='*60}")
        print("DRY RUN MODE - Only counting tokens, not sending LLM requests")
        print(f"{'='*60}\n")
    
    # Results storage
    results: List[Dict[str, Any]] = []
    all_retrieval_metrics: Dict[str, List[float]] = defaultdict(list)
    ground_truth_labels: List[str] = []
    predicted_labels: List[str] = []
    skipped: List[str] = []
    
    # Token counting for dry run mode
    total_input_tokens = 0
    total_output_tokens_estimate = 0
    
    # Process each entry
    for entry_idx, entry in enumerate(entries, 1):
        paper_id = get_paper_id(entry)
        claim = get_claim_from_entry(entry)
        ground_truth_evidence = get_ground_truth_evidence(entry, benchmark_name)
        ground_truth_label = get_label_from_entry(entry, benchmark_name)
        
        if not paper_id or not claim:
            skipped.append(f"{paper_id or 'unknown'}: missing paper_id or claim")
            continue
        
        # Load chunks
        chunks = find_chunks_for_paper(paper_id, data_structure)
        if not chunks:
            skipped.append(f"{paper_id}: no chunks available")
            continue
        
        # Evaluate retrieval and verification
        try:
            eval_result = evaluate_retrieval_and_verification_for_claim(
                claim=claim,
                paper_id=paper_id,
                ground_truth_evidence=ground_truth_evidence,
                ground_truth_label=ground_truth_label,
                chunks=chunks,
                retrieve_fn=retrieve_fn,
                llm_model=llm_model,
                top_k=top_k,
                dry_run=dry_run
            )
            
            # Create result entry
            result_entry = {
                'paper_id': paper_id,
                'claim': claim,
                'ground_truth_label': ground_truth_label,
                'ground_truth_evidence_count': len(ground_truth_evidence),
                'ground_truth_evidence': ground_truth_evidence,
                'evaluation_result': eval_result,
            }
            
            # Track tokens in dry run mode
            if dry_run:
                input_tokens = eval_result.get('input_tokens', 0)
                total_input_tokens += input_tokens
                total_output_tokens_estimate += 100  # Rough estimate for JSON response
            
            # Add retrieval metrics to aggregated list (if we have evidence)
            if ground_truth_evidence:
                for metric_name, metric_value in eval_result['retrieval']['metrics'].items():
                    all_retrieval_metrics[metric_name].append(metric_value)
            
            # Add labels for classification metrics (not meaningful in dry run)
            ground_truth_labels.append(ground_truth_label)
            predicted_labels.append(eval_result['verification']['predicted_label'])
            
            results.append(result_entry)
            
            if entry_idx % 10 == 0:
                if dry_run:
                    avg_so_far = total_input_tokens / entry_idx if entry_idx > 0 else 0
                    print(f"Counted {entry_idx}/{len(entries)} claims... (Avg input tokens: {avg_so_far:,.1f})")
                else:
                    print(f"Processed {entry_idx}/{len(entries)} claims...")
                
        except Exception as e:
            skipped.append(f"{paper_id}: error - {e}")
            print(f"Error processing {paper_id}: {e}")
    
    # Calculate aggregated retrieval metrics
    aggregated_retrieval_metrics = {}
    for metric_name, values in all_retrieval_metrics.items():
        if values:
            aggregated_retrieval_metrics[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }
    
    # Calculate classification metrics (not meaningful in dry run)
    classification_metrics = {}
    if ground_truth_labels and predicted_labels and not dry_run:
        classification_metrics = calculate_classification_metrics(ground_truth_labels, predicted_labels)
    
    # Save per-instance results
    results_file = output_dir / 'results.jsonl'
    with open(results_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # Save aggregated metrics
    metrics_summary = {
        'benchmark_path': benchmark_path,
        'benchmark_name': benchmark_name,
        'retriever': retriever,
        'llm_model': llm_model,
        'top_k': top_k,
        'total_processed': len(results),
        'total_skipped': len(skipped),
        'retrieval_metrics': aggregated_retrieval_metrics,
        'classification_metrics': classification_metrics,
        'dry_run': dry_run,
    }
    
    if dry_run:
        num_instances = len(results)
        avg_input_tokens = total_input_tokens / num_instances if num_instances > 0 else 0
        avg_output_tokens = total_output_tokens_estimate / num_instances if num_instances > 0 else 0
        metrics_summary['num_instances'] = num_instances
        metrics_summary['avg_input_tokens_per_instance'] = avg_input_tokens
        metrics_summary['avg_output_tokens_per_instance'] = avg_output_tokens
        metrics_summary['avg_tokens_per_instance'] = avg_input_tokens + avg_output_tokens
    
    metrics_file = output_dir / 'metrics.json'
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Metrics saved to: {metrics_file}")
    
    if dry_run:
        print(f"\n{'='*60}")
        print("TOKEN COUNT SUMMARY (DRY RUN)")
        print(f"{'='*60}")
        print(f"Number of Instances: {num_instances:,}")
        print(f"Avg Input Tokens per Instance: {avg_input_tokens:,.1f}")
        print(f"Avg Output Tokens per Instance (estimate): {avg_output_tokens:,.1f}")
        print(f"Avg Tokens per Instance: {avg_input_tokens + avg_output_tokens:,.1f}")
        print(f"{'='*60}\n")
    
    return metrics_summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate benchmark with retrieval and classification metrics')
    parser.add_argument('--benchmark', required=True, help='Path to benchmark JSONL file (or benchmark name: B1, B2, B3, B4, B5)')
    parser.add_argument('--output-dir', help='Output directory (default: auto-generated in data/experiments/)')
    parser.add_argument('--retriever', default='bm25-cross', choices=list(RETRIEVER_FUNCS.keys()))
    parser.add_argument('--llm-model', default='gpt-4o-mini', help='LLM model name')
    parser.add_argument('--top-k', type=int, default=10, help='Number of chunks to retrieve')
    parser.add_argument('--max-entries', type=int, help='Maximum number of entries to process')
    parser.add_argument('--dry-run', action='store_true', help='Only count tokens without sending LLM requests')
    
    args = parser.parse_args()
    
    # Determine benchmark path
    data_structure = DataStructure()
    benchmark_path = args.benchmark
    
    # If it's a benchmark name (B1, B2, etc.), resolve to file path
    if benchmark_path.upper() in ['B1', 'B2', 'B3', 'B4', 'B5']:
        benchmark_map = {
            'B1': 'B1_golden_supported.jsonl',
            'B2': 'B2_reviewer_author.jsonl',
            'B3': 'B3_reviewer_paper.jsonl',
            'B4': 'B4_agreement.jsonl',
            'B5': 'B5_verifiable.jsonl',
        }
        benchmark_path = str(data_structure.benchmark_dir / benchmark_map[benchmark_path.upper()])
    
    if not Path(benchmark_path).exists():
        print(f"Error: Benchmark file not found: {benchmark_path}")
        sys.exit(1)
    
    metrics_summary = evaluate_benchmark_retrieval_and_verification(
        benchmark_path=benchmark_path,
        retriever=args.retriever,
        llm_model=args.llm_model,
        top_k=args.top_k,
        max_entries=args.max_entries,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )
    
    print(json.dumps(metrics_summary, indent=2))


if __name__ == '__main__':
    main()
