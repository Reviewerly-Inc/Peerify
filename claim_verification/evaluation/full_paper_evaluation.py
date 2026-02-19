

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import sys

# Token counting support
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not available for token counting. Install with: pip install tiktoken")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai library not available. Install with: pip install openai")

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: anthropic library not available. Install with: pip install anthropic")

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google.genai library not available. Install with: pip install google-genai")

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


# Import from real_benchmark_evaluation
try:
    from claim_verification.evaluation.retrieval_evaluation import (
        load_benchmark,
        get_claim_from_entry,
        get_paper_id,
        get_label_from_entry,
        get_benchmark_name,
    )
except ImportError:
    # Fallback for when running directly from this directory
    from claim_verification.evaluation.retrieval_evaluation import (
        load_benchmark,
        get_claim_from_entry,
        get_paper_id,
        get_label_from_entry,
        get_benchmark_name,
    )
from claim_verification.config import DataStructure

# API keys - use environment variables
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', '')


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


def load_full_paper(paper_id: str, data_structure: DataStructure) -> str:
    """
    Load the full paper markdown for a paper.
    
    NOTE: This loads the COMPLETE paper text - no chunking, no retrieval.
    The entire paper is returned as-is for direct use in LLM prompts.
    
    Args:
        paper_id: The paper ID
        data_structure: DataStructure instance for paths
        
    Returns:
        Full paper text as string, or empty string if not found
    """
    md_file = data_structure.markdown_dir / f"{paper_id}.md"
    
    if md_file.exists():
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading {md_file}: {e}")
    
    print(f"Warning: Paper not found for {paper_id}")
    return ""


def create_verification_prompt(claim: str, paper_text: str) -> str:
    """
    Create a prompt for claim verification using the full paper.
    
    Args:
        claim: The claim to verify
        paper_text: The full paper text
        
    Returns:
        Formatted prompt string
    """
    prompt = (
        # === SYSTEM ===
        "You are a factual‑verification API. Respond **ONLY** with a valid JSON object—no other text.\n\n"
        
        # === LABEL DEFINITIONS ===
        "Label definitions you must use:\n"
        "  • Supported: The paper's content fully backs the claim with no gaps or contradictions.\n"
        "  • Partially Supported: Some parts of the claim align with the paper, but other details are missing or unclear.\n"
        "  • Contradicted: The claim directly conflicts with the paper's content or established facts.\n"
        "  • Not Determined: The paper's content is insufficient to confirm or deny the claim.\n\n"
        
        # === TASK ===
        "Task: Classify the claim using **exactly one** of the four labels above based on the entire paper.\n\n"
        
        # === INPUTS ===
        f"CLAIM:\n{claim}\n\n"
        f"FULL PAPER:\n{paper_text}\n\n"
        
        # === OUTPUT RULES ===
        "CRITICAL OUTPUT RULES:\n"
        "  • Output ONLY the JSON object—no thinking notes, no markdown, no extra text.\n"
        "  • The very first character you emit must be '{' and the very last one must be '}'.\n"
        "  • Provide exactly two keys, lowercase, in this order: result, justification.\n"
        "  • For \"result\", choose one of: Supported, Partially Supported, Contradicted, Not Determined.\n"
        "  • \"justification\" must be a single concise sentence (≤ 30 words) explaining why the label was chosen; escape any internal quotes.\n"
        "  • Do NOT include additional keys, formatting, or commentary.\n\n"
        
        # === JSON TEMPLATE (for reference only) ===
        "Return exactly:\n"
        "{\n"
        '  "result": "<Supported|Partially Supported|Contradicted|Not Determined>",\n'
        '  "justification": "<brief explanation>"\n'
        "}"
    )
    return prompt


def verify_with_openai(
    claim: str,
    paper_text: str,
    model: str,
    client: OpenAI,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Verify a claim using OpenAI API.
    
    Args:
        claim: The claim to verify
        paper_text: The full paper text
        model: OpenAI model name (e.g., 'gpt-4o', 'gpt-4o-mini')
        client: OpenAI client instance
        max_retries: Maximum number of retry attempts
        
    Returns:
        Dictionary with 'result' and 'justification'
    """
    prompt = create_verification_prompt(claim, paper_text)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a factual verification API that responds only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                result = json.loads(response_text)
                return result
            except json.JSONDecodeError:
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    return result
                else:
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    return {
                        'result': 'Not Determined',
                        'justification': f'Failed to parse JSON response: {response_text[:100]}...'
                    }
                    
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            return {
                'result': 'Not Determined',
                'justification': f'Error during OpenAI API call: {str(e)}'
            }
    
    return {
        'result': 'Not Determined',
        'justification': 'Failed after all retry attempts'
    }


def verify_with_claude(
    claim: str,
    paper_text: str,
    model: str,
    client: Anthropic,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Verify a claim using Anthropic Claude API.
    
    Args:
        claim: The claim to verify
        paper_text: The full paper text
        model: Claude model name (e.g., 'claude-3-5-sonnet-20241022')
        client: Anthropic client instance
        max_retries: Maximum number of retry attempts
        
    Returns:
        Dictionary with 'result' and 'justification'
    """
    prompt = create_verification_prompt(claim, paper_text)
    
    # Determine max_tokens based on model
    # Claude 3.5 Sonnet supports up to 8192, others typically 4096
    if 'sonnet' in model.lower() or '3.5' in model:
        max_tokens = 8192
    else:
        max_tokens = 4096
    
    for attempt in range(max_retries):
        try:
            # Use streaming for long requests (required for operations >10 minutes)
            # Anthropic models have max_tokens limits
            message = client.messages.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                stream=True,  # Required for long requests
            )
            
            # Collect streaming response
            # Anthropic streaming returns chunks with type 'content_block_delta' containing delta.text
            response_text = ""
            for chunk in message:
                try:
                    # Check chunk type attribute
                    chunk_type = getattr(chunk, 'type', None)
                    if chunk_type == 'content_block_delta':
                        # Extract text from delta
                        delta = getattr(chunk, 'delta', None)
                        if delta:
                            text = getattr(delta, 'text', '')
                            if text:
                                response_text += text
                    # Ignore other chunk types (content_block_start, content_block_stop, message_delta, message_stop)
                except Exception as e:
                    # If there's an error processing a chunk, log and continue
                    print(f"Warning: Error processing chunk: {e}")
                    continue
            
            response_text = response_text.strip()
            
            # Try to parse JSON response
            try:
                result = json.loads(response_text)
                print(result)
                return result
            except json.JSONDecodeError:
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    return result
                else:
                    print(response_text)
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    return {
                        'result': 'Not Determined',
                        'justification': f'Failed to parse JSON response: {response_text[:100]}...'
                    }
                    
        except Exception as e:
            error_msg = str(e)
            # Log the full error for debugging
            print(f"Claude API error (attempt {attempt + 1}/{max_retries}): {error_msg}")
            
            # Check if it's a 400 error with specific message
            if "400" in error_msg or "Bad Request" in error_msg:
                # Try with lower max_tokens or check if prompt is too long
                if attempt < max_retries - 1:
                    # Try with even lower max_tokens on retry
                    time.sleep(2 ** attempt)
                    continue
                return {
                    'result': 'Not Determined',
                    'justification': f'Claude API 400 error (likely max_tokens or prompt length issue): {error_msg[:200]}'
                }
            
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            return {
                'result': 'Not Determined',
                'justification': f'Error during Claude API call: {error_msg[:200]}'
            }
    
    return {
        'result': 'Not Determined',
        'justification': 'Failed after all retry attempts'
    }


def verify_with_gemini(
    claim: str,
    paper_text: str,
    model: str,
    api_key: Optional[str] = None,
    max_retries: int = 10,
    initial_sleep: float = 1.0
) -> Dict[str, Any]:
    """
    Verify a claim using Google Gemini API.
    
    Args:
        claim: The claim to verify
        paper_text: The full paper text
        model: Gemini model name (e.g., 'gemini-2.5-flash')
        api_key: Optional API key (if None, uses GOOGLE_API_KEY env var)
        max_retries: Maximum number of retry attempts
        initial_sleep: Initial sleep time in seconds
        
    Returns:
        Dictionary with 'result' and 'justification'
    """
    prompt = create_verification_prompt(claim, paper_text)
    
    # Initialize client
    client = None
    if GEMINI_AVAILABLE:
        try:
            if api_key:
                client = genai.Client(api_key=api_key)
            else:
                client = genai.Client()
        except Exception as e:
            return {
                'result': 'Not Determined',
                'justification': f'Error initializing Gemini client: {str(e)}'
            }
    
    attempt = 0
    while attempt < max_retries:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )
            
            # Extract response text
            if hasattr(response, 'text'):
                response_text = response.text.strip()
            elif hasattr(response, 'content'):
                if isinstance(response.content, str):
                    response_text = response.content.strip()
                elif isinstance(response.content, list) and len(response.content) > 0:
                    response_text = str(response.content[0]).strip()
                else:
                    response_text = str(response.content).strip()
            else:
                response_text = str(response).strip()
            
            # Try to parse JSON response
            try:
                result = json.loads(response_text)
                return result
            except json.JSONDecodeError:
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    return result
                else:
                    attempt += 1
                    if attempt < max_retries:
                        sleep_time = initial_sleep * (2 ** (attempt - 1))
                        time.sleep(sleep_time)
                        continue
                    return {
                        'result': 'Not Determined',
                        'justification': f'Failed to parse JSON response after {max_retries} attempts'
                    }
                    
        except Exception as e:
            attempt += 1
            if attempt < max_retries:
                sleep_time = initial_sleep * (2 ** (attempt - 1))
                time.sleep(sleep_time)
                continue
            return {
                'result': 'Not Determined',
                'justification': f'Error during Gemini API call: {str(e)}'
            }
    
    return {
        'result': 'Not Determined',
        'justification': f'Failed after {max_retries} retry attempts'
    }


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


def evaluate_full_paper_models(
    benchmark_path: str,
    models: List[str],
    max_entries: Optional[int] = None,
    output_dir: Optional[str] = None,
    benchmark_name: str = '',
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
    num_workers: int = 4,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate claim verification by feeding full papers to multiple LLM models.
    
    Args:
        benchmark_path: Path to benchmark JSONL file
        models: List of model identifiers (e.g., ['openai:gpt-4o', 'claude:claude-3-5-sonnet-20241022'])
        max_entries: Maximum number of entries to process (None = all)
        output_dir: Output directory for results (auto-generated if None)
        benchmark_name: Benchmark name (auto-detected if empty)
        openai_api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
        anthropic_api_key: Anthropic API key (if None, uses ANTHROPIC_API_KEY env var)
        google_api_key: Google API key (if None, uses GOOGLE_API_KEY env var)
        num_workers: Number of parallel workers for processing
        dry_run: If True, only count tokens without sending LLM requests
        
    Returns:
        Dictionary with aggregated metrics per model
    """
    entries = load_benchmark(benchmark_path)
    
    if max_entries is not None:
        entries = entries[:max_entries]
    
    if not benchmark_name:
        benchmark_name = get_benchmark_name(benchmark_path)
    
    data_structure = DataStructure()
    
    if dry_run:
        print(f"\n{'='*60}")
        print("DRY RUN MODE - Only counting tokens, not sending LLM requests")
        print(f"{'='*60}\n")
    
    # Initialize API clients (skip in dry run mode)
    openai_client = None
    anthropic_client = None
    
    if not dry_run:
        if OPENAI_AVAILABLE:
            api_key = openai_api_key or OPENAI_API_KEY or os.environ.get('OPENAI_API_KEY', '')
            if api_key:
                openai_client = OpenAI(api_key=api_key)
                print("✓ OpenAI client initialized")
            else:
                print("⚠ OpenAI API key not found")
        
        if ANTHROPIC_AVAILABLE:
            api_key = anthropic_api_key or ANTHROPIC_API_KEY or os.environ.get('ANTHROPIC_API_KEY', '')
            if api_key:
                anthropic_client = Anthropic(api_key=api_key)
                print("✓ Anthropic client initialized")
            else:
                print("⚠ Anthropic API key not found")
        
        if GEMINI_AVAILABLE:
            api_key = google_api_key or GOOGLE_API_KEY or os.environ.get('GOOGLE_API_KEY', '')
            if api_key:
                os.environ['GOOGLE_API_KEY'] = api_key
                print("✓ Google Gemini client will be initialized on first use")
            else:
                print("⚠ Google API key not found")
    
    # Determine output directory
    if output_dir is None:
        output_dir = data_structure.base_dir / 'experiments' / benchmark_name / 'full_paper'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # Process each model
    for model_spec in models:
        # Parse model specification: "provider:model_name" or just "model_name" (defaults to openai)
        if ':' in model_spec:
            provider, model_name = model_spec.split(':', 1)
        else:
            provider = 'openai'
            model_name = model_spec
        
        print(f"\n{'='*60}")
        print(f"Processing model: {provider}:{model_name}")
        print(f"{'='*60}")
        
        # Create model-specific output directory
        model_output_dir = output_dir / model_name.replace('/', '_').replace('-', '_')
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        results: List[Dict[str, Any]] = []
        ground_truth_labels: List[str] = []
        predicted_labels: List[str] = []
        skipped: List[str] = []
        
        # Token counting for dry run mode
        total_input_tokens = 0
        total_output_tokens_estimate = 0
        
        # Process entries
        for entry_idx, entry in enumerate(entries, 1):
            paper_id = get_paper_id(entry)
            claim = get_claim_from_entry(entry)
            ground_truth_label = get_label_from_entry(entry, benchmark_name)
            
            if not paper_id or not claim:
                skipped.append(f"{paper_id or 'unknown'}: missing paper_id or claim")
                continue
            
            if not ground_truth_label or ground_truth_label == 'Unknown':
                skipped.append(f"{paper_id}: no ground truth label")
                continue
            
            # Load full paper
            paper_text = load_full_paper(paper_id, data_structure)
            if not paper_text:
                skipped.append(f"{paper_id}: paper not found")
                continue
            
            if dry_run:
                # Only count tokens, don't send LLM request
                prompt = create_verification_prompt(claim, paper_text)
                input_tokens = count_tokens(prompt, model_name)
                total_input_tokens += input_tokens
                total_output_tokens_estimate += 100  # Rough estimate for JSON response
                
                result_entry = {
                    'paper_id': paper_id,
                    'claim': claim,
                    'ground_truth_label': ground_truth_label,
                    'predicted_label': 'DRY_RUN',
                    'justification': f'Dry run - counted {input_tokens} input tokens',
                    'is_correct': False,
                    'input_tokens': input_tokens,
                }
                
                results.append(result_entry)
                
                if entry_idx % 10 == 0:
                    avg_so_far = total_input_tokens / entry_idx if entry_idx > 0 else 0
                    print(f"Counted {entry_idx}/{len(entries)} claims... (Avg input tokens: {avg_so_far:,.1f})")
            else:
                # Verify claim with the appropriate model
                try:
                    if provider == 'openai' or provider == 'gpt':
                        if not openai_client:
                            skipped.append(f"{paper_id}: OpenAI client not available")
                            continue
                        verification_result = verify_with_openai(claim, paper_text, model_name, openai_client)
                    elif provider == 'claude' or provider == 'anthropic':
                        if not anthropic_client:
                            skipped.append(f"{paper_id}: Anthropic client not available")
                            continue
                        verification_result = verify_with_claude(claim, paper_text, model_name, anthropic_client)
                    elif provider == 'gemini' or provider == 'google':
                        if not GEMINI_AVAILABLE:
                            skipped.append(f"{paper_id}: Gemini client not available")
                            continue
                        api_key = google_api_key or GOOGLE_API_KEY or os.environ.get('GOOGLE_API_KEY', '')
                        verification_result = verify_with_gemini(claim, paper_text, model_name, api_key=api_key)
                    else:
                        skipped.append(f"{paper_id}: unknown provider: {provider}")
                        continue
                    
                    predicted_label = verification_result.get('result', 'Not Determined')
                    justification = verification_result.get('justification', '')
                    
                    # Check if prediction matches ground truth
                    is_correct = (predicted_label.lower() == ground_truth_label.lower())
                    
                    # Create result entry
                    result_entry = {
                        'paper_id': paper_id,
                        'claim': claim,
                        'ground_truth_label': ground_truth_label,
                        'predicted_label': predicted_label,
                        'justification': justification,
                        'is_correct': is_correct,
                    }
                    
                    results.append(result_entry)
                    ground_truth_labels.append(ground_truth_label)
                    predicted_labels.append(predicted_label)
                    
                    if entry_idx % 10 == 0:
                        print(f"Processed {entry_idx}/{len(entries)} claims...")
                        
                except Exception as e:
                    skipped.append(f"{paper_id}: error - {e}")
                    print(f"Error processing {paper_id}: {e}")
        
        # Calculate classification metrics (skip in dry run mode)
        classification_metrics = {}
        if ground_truth_labels and predicted_labels and not dry_run:
            classification_metrics = calculate_classification_metrics(ground_truth_labels, predicted_labels)
        
        # Save per-instance results
        results_file = model_output_dir / 'results.jsonl'
        with open(results_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        # Save aggregated metrics
        metrics_summary = {
            'benchmark_path': benchmark_path,
            'benchmark_name': benchmark_name,
            'model_provider': provider,
            'model_name': model_name,
            'total_processed': len(results),
            'total_skipped': len(skipped),
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
        
        metrics_file = model_output_dir / 'metrics.json'
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_summary, f, indent=2, ensure_ascii=False)
        
        all_results[f"{provider}:{model_name}"] = metrics_summary
        
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
            print(f"{'='*60}")
        elif classification_metrics:
            print(f"Accuracy: {classification_metrics.get('accuracy', 0.0):.4f}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate benchmark with full paper (no retrieval)')
    parser.add_argument('--benchmark', required=True, help='Path to benchmark JSONL file (or benchmark name: B1, B2, B3, B4, B5)')
    parser.add_argument('--models', nargs='+', required=True, help='Model identifiers (e.g., openai:gpt-4o claude:claude-3-5-sonnet-20241022)')
    parser.add_argument('--output-dir', help='Output directory (default: auto-generated in data/experiments/)')
    parser.add_argument('--max-entries', type=int, help='Maximum number of entries to process')
    parser.add_argument('--openai-api-key', help='OpenAI API key (if not set, uses OPENAI_API_KEY env var)')
    parser.add_argument('--anthropic-api-key', help='Anthropic API key (if not set, uses ANTHROPIC_API_KEY env var)')
    parser.add_argument('--google-api-key', help='Google API key (if not set, uses GOOGLE_API_KEY env var)')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of parallel workers (default: 4)')
    parser.add_argument('--dry-run', action='store_true', help='Only count tokens without sending LLM requests')
    
    args = parser.parse_args()
    
    # Determine benchmark path
    data_structure = DataStructure()
    benchmark_path = args.benchmark
    
    # If it's a benchmark name (B1, B2, etc.), resolve to file path
    if benchmark_path.upper() in ['B1', 'B2', 'B3', 'B4', 'B5', "B6"]:
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
    
    all_results = evaluate_full_paper_models(
        benchmark_path=benchmark_path,
        models=args.models,
        max_entries=args.max_entries,
        output_dir=args.output_dir,
        openai_api_key=args.openai_api_key,
        anthropic_api_key=args.anthropic_api_key,
        google_api_key=args.google_api_key,
        num_workers=args.num_workers,
        dry_run=args.dry_run,
    )
    
    print("\n" + "="*60)
    print("Summary" + (" (DRY RUN)" if args.dry_run else ""))
    print("="*60)
    for model_spec, metrics in all_results.items():
        if args.dry_run:
            avg_tokens = metrics.get('avg_tokens_per_instance', 0)
            num_instances = metrics.get('num_instances', 0)
            print(f"{model_spec}: Avg Tokens/Instance = {avg_tokens:,.1f} ({num_instances} instances)")
        else:
            accuracy = metrics.get('classification_metrics', {}).get('accuracy', 0.0)
            print(f"{model_spec}: Accuracy = {accuracy:.4f}")


if __name__ == '__main__':
    main()
