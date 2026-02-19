

import json
import argparse
import os
from pathlib import Path
import sys
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Progress bar support
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available. Install with: pip install tqdm for progress bars")

# Token counting support
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not available for token counting. Install with: pip install tiktoken")


def _get_fenice_extractor():
    from claim_verification.retrieval.pipeline_wrapper import extract_claims_from_review_fenice
    return extract_claims_from_review_fenice

def _get_gemma_extractor():
    from claim_verification.retrieval.pipeline_wrapper import extract_claims_from_review_gemma
    return extract_claims_from_review_gemma

# Import OpenAI for LLM as a judge
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai library not available. Install with: pip install openai")

# Import config
from claim_verification.config import DataStructure, DEFAULT_MODEL

# OpenAI API key from environment
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')

# Default LLM model for judging (gpt-5-mini)
LLM_JUDGE_MODEL = "gpt-5-mini"


def get_openai_client() -> Optional[OpenAI]:
    """Get OpenAI client, initializing if needed."""
    if not OPENAI_AVAILABLE:
        return None
    api_key = OPENAI_API_KEY or os.environ.get('OPENAI_API_KEY', '')
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def call_openai_llm(prompt: str, system_prompt: str = "", model: str = None) -> Optional[str]:
    """
    Call OpenAI LLM with given prompt.
    
    Args:
        prompt: User prompt
        system_prompt: System prompt
        model: Model name (defaults to LLM_JUDGE_MODEL)
        
    Returns:
        Response text or None on error
    """
    if model is None:
        model = LLM_JUDGE_MODEL
    
    client = get_openai_client()
    if not client:
        return None
    
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling OpenAI: {e}")
        return None


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


def extract_claims_with_llm(review_text: str, model: str = None) -> List[str]:
    """
    Extract claims from review text using specified LLM model.
    
    Args:
        review_text: The review text to extract claims from
        model: LLM model to use (defaults to LLM_JUDGE_MODEL)
        
    Returns:
        List of extracted claim strings
    """
    if not OPENAI_AVAILABLE:
        return []
    
    if model is None:
        model = LLM_JUDGE_MODEL
    
    try:
        prompt = f"""Extract factual claims from the following review text. 
A claim is a specific, verifiable statement about the paper.

Review Text:
{review_text}

Extract all factual claims. Each claim should:
1. Be a single, atomic statement
2. Be verifiable (can be checked against the paper or author response)
3. Be specific and factual (not opinions or general comments)

Return the claims as a JSON array of strings. Example format:
["claim 1", "claim 2", "claim 3"]

Return only valid JSON."""

        result_text = call_openai_llm(
            prompt=prompt,
            system_prompt="You are an expert at extracting factual claims from academic reviews. Return only valid JSON arrays.",
            model=model
        )
        
        if not result_text:
            return []
        
        # Try to parse JSON
        try:
            claims = json.loads(result_text)
            if isinstance(claims, list):
                return [str(c).strip() for c in claims if c and str(c).strip()]
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if json_match:
                claims = json.loads(json_match.group())
                if isinstance(claims, list):
                    return [str(c).strip() for c in claims if c and str(c).strip()]
        
        return []
        
    except Exception as e:
        print(f"Error in LLM extraction: {e}")
        return []


class ClaimExtractionEvaluator:
    """Evaluates claim extraction methods against gold claims from benchmark."""
    
    def __init__(self, llm_model: str = None):
        self.data_structure = DataStructure()
        self.vectorizer = TfidfVectorizer()
        self.llm_model = llm_model or LLM_JUDGE_MODEL
    
    def load_benchmark_data(self, benchmark_file: Path) -> List[Dict[str, Any]]:
        """Load benchmark data from JSONL file."""
        data = []
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
        return data
    
    def extract_claims_fenice(self, review_text: str) -> List[str]:
        """Extract claims using FENICE method."""
        try:
            extract_fn = _get_fenice_extractor()
            claims = extract_fn(review_text)
            return [c.strip() for c in claims if c.strip()]
        except Exception as e:
            print(f"Error in FENICE extraction: {e}")
            return []
    
    def extract_claims_gemma(self, review_text: str) -> List[str]:
        """Extract claims using Gemma method."""
        try:
            extract_fn = _get_gemma_extractor()
            claims = extract_fn(review_text)
            return [c.strip() for c in claims if c.strip()]
        except Exception as e:
            print(f"Error in Gemma extraction: {e}")
            return []
    
    def extract_claims_llm(self, review_text: str) -> List[str]:
        """Extract claims using LLM method."""
        return extract_claims_with_llm(review_text, model=self.llm_model)
    
    def match_with_cosine_similarity(self, extracted_claims: List[str], gold_claims: List[str], 
                                     threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Match extracted claims with gold claims using cosine similarity."""
        if not extracted_claims or not gold_claims:
            return []
        
        # Fit vectorizer on all claims
        all_texts = extracted_claims + gold_claims
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        
        extracted_vecs = tfidf_matrix[:len(extracted_claims)]
        gold_vecs = tfidf_matrix[len(extracted_claims):]
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(extracted_vecs, gold_vecs)
        
        matches = []
        for i, ext_claim in enumerate(extracted_claims):
            # Find best match
            best_idx = np.argmax(similarity_matrix[i])
            best_score = similarity_matrix[i][best_idx]
            
            if best_score >= threshold:
                matches.append({
                    'extracted_claim': ext_claim,
                    'gold_claim': gold_claims[best_idx],
                    'similarity_score': float(best_score),
                    'gold_index': int(best_idx),
                    'matched': True
                })
            else:
                matches.append({
                    'extracted_claim': ext_claim,
                    'gold_claim': None,
                    'similarity_score': float(best_score),
                    'gold_index': None,
                    'matched': False
                })
        
        return matches
    
    def evaluate_claim_atomicity(self, claim: str) -> Dict[str, Any]:
        """Evaluate if claim contains only one fact or relation (Atomicity)."""
        if not OPENAI_AVAILABLE:
            return {'atomicity': None, 'reasoning': 'OpenAI not available'}
        
        try:
            prompt = f"""Evaluate if the following claim is atomic (contains only one fact or relation):

Claim: {claim}

Is this claim atomic? An atomic claim should:
1. Contain only one main fact or relation
2. Not combine multiple independent facts
3. Be self-contained without multiple connected ideas

Return a JSON object with:
- "atomic": true/false
- "reasoning": brief explanation

Return only valid JSON."""

            result_text = call_openai_llm(
                prompt=prompt,
                system_prompt="You are an expert at evaluating claim atomicity. Return only valid JSON.",
                model=self.llm_model
            )
            
            if not result_text:
                return {'atomicity': None, 'reasoning': 'Empty response from LLM'}
            
            # Try to parse JSON
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks or text
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    return {'atomicity': None, 'reasoning': f'Failed to parse JSON: {result_text[:100]}'}
            
            return {
                'atomicity': result.get('atomic', False),
                'reasoning': result.get('reasoning', '')
            }
            
        except Exception as e:
            return {'atomicity': None, 'reasoning': f'Error: {str(e)}'}
    
    def evaluate_claim_fluency(self, claim: str) -> Dict[str, Any]:
        """Evaluate if claim is grammatical and readable (Fluency)."""
        if not OPENAI_AVAILABLE:
            return {'fluency': None, 'reasoning': 'OpenAI not available'}
        
        try:
            prompt = f"""Evaluate the fluency of the following claim (grammatical correctness and readability):

Claim: {claim}

Is this claim fluent? A fluent claim should:
1. Be grammatically correct
2. Be clear and readable
3. Use proper sentence structure
4. Be natural and coherent

Return a JSON object with:
- "fluent": true/false
- "reasoning": brief explanation

Return only valid JSON."""

            result_text = call_openai_llm(
                prompt=prompt,
                system_prompt="You are an expert at evaluating text fluency. Return only valid JSON.",
                model=self.llm_model
            )
            
            if not result_text:
                return {'fluency': None, 'reasoning': 'Empty response from LLM'}
            
            # Try to parse JSON
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks or text
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    return {'fluency': None, 'reasoning': f'Failed to parse JSON: {result_text[:100]}'}
            
            return {
                'fluency': result.get('fluent', False),
                'reasoning': result.get('reasoning', '')
            }
            
        except Exception as e:
            return {'fluency': None, 'reasoning': f'Error: {str(e)}'}
    
    def evaluate_claim_decontextualization(self, claim: str) -> Dict[str, Any]:
        """Evaluate if claim can stand alone without missing references (Decontextualization)."""
        if not OPENAI_AVAILABLE:
            return {'decontextualized': None, 'reasoning': 'OpenAI not available'}
        
        try:
            prompt = f"""Evaluate if the following claim is decontextualized (can stand alone without pronouns or missing references):

Claim: {claim}

Is this claim decontextualized? A decontextualized claim should:
1. Not contain vague pronouns (it, they, this, etc.) without clear referents
2. Not rely on external context to be understood
3. Be self-contained and complete
4. Not have missing information

Return a JSON object with:
- "decontextualized": true/false
- "reasoning": brief explanation

Return only valid JSON."""

            result_text = call_openai_llm(
                prompt=prompt,
                system_prompt="You are an expert at evaluating text decontextualization. Return only valid JSON.",
                model=self.llm_model
            )
            
            if not result_text:
                return {'decontextualized': None, 'reasoning': 'Empty response from LLM'}
            
            # Try to parse JSON
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks or text
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    return {'decontextualized': None, 'reasoning': f'Failed to parse JSON: {result_text[:100]}'}
            
            return {
                'decontextualized': result.get('decontextualized', False),
                'reasoning': result.get('reasoning', '')
            }
            
        except Exception as e:
            return {'decontextualized': None, 'reasoning': f'Error: {str(e)}'}
    
    def evaluate_claim_faithfulness(self, claim: str, source_text: str) -> Dict[str, Any]:
        """Evaluate if claim is factually supported by the source text (Faithfulness)."""
        if not OPENAI_AVAILABLE:
            return {'faithful': None, 'reasoning': 'OpenAI not available'}
        
        try:
            prompt = f"""Evaluate if the claim is factually supported by the source text:

Claim: {claim}

Source Text: {source_text}

Is this claim faithful to the source text? A faithful claim should:
1. Be factually supported by the source
2. Not introduce new information not in the source
3. Not contradict the source
4. Accurately represent the source content

Return a JSON object with:
- "faithful": true/false
- "reasoning": brief explanation

Return only valid JSON."""

            result_text = call_openai_llm(
                prompt=prompt,
                system_prompt="You are an expert at evaluating factual faithfulness. Return only valid JSON.",
                model=self.llm_model
            )
            
            if not result_text:
                return {'faithful': None, 'reasoning': 'Empty response from LLM'}
            
            # Try to parse JSON
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks or text
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    return {'faithful': None, 'reasoning': f'Failed to parse JSON: {result_text[:100]}'}
            
            return {
                'faithful': result.get('faithful', False),
                'reasoning': result.get('reasoning', '')
            }
            
        except Exception as e:
            return {'faithful': None, 'reasoning': f'Error: {str(e)}'}
    
    def match_with_llm(self, extracted_claim: str, gold_claims: List[str], 
                       review_text: str) -> Dict[str, Any]:
        """Use GPT-5-mini to find best matching gold claim."""
        if not OPENAI_AVAILABLE or not gold_claims:
            return {
                'extracted_claim': extracted_claim,
                'gold_claim': None,
                'matched': False,
                'confidence': 0.0,
                'reasoning': 'OpenAI not available'
            }
        
        # Format gold claims for prompt
        gold_claims_text = '\n'.join([f"{i+1}. {claim}" for i, claim in enumerate(gold_claims)])
        
        prompt = f"""Given a claim extracted from a review and a list of gold standard claims, 
determine if the extracted claim matches any of the gold claims semantically.

Extracted Claim: {extracted_claim}

Gold Claims (from the same review):
{gold_claims_text}

Review Context:
{review_text[:1000]}...

Please provide a JSON response with:
- "matched": true/false
- "gold_index": the index (starting from 0) of the matched gold claim, or null if no match
- "confidence": a float between 0 and 1 indicating confidence in the match
- "reasoning": brief explanation

Return only valid JSON."""

        try:
            result_text = call_openai_llm(
                prompt=prompt,
                system_prompt="You are an expert at semantic matching of academic claims. Return only valid JSON.",
                model=self.llm_model
            )
            
            if not result_text:
                return {
                    'extracted_claim': extracted_claim,
                    'gold_claim': None,
                    'matched': False,
                    'confidence': 0.0,
                    'reasoning': 'Empty response from LLM'
                }
            
            # Try to parse JSON
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    return {
                        'extracted_claim': extracted_claim,
                        'gold_claim': None,
                        'matched': False,
                        'confidence': 0.0,
                        'reasoning': f'Failed to parse response: {result_text[:100]}'
                    }
            
            # Get the matched gold claim
            gold_idx = result.get('gold_index')
            if gold_idx is not None and 0 <= gold_idx < len(gold_claims):
                gold_claim = gold_claims[gold_idx]
            else:
                gold_claim = None
            
            return {
                'extracted_claim': extracted_claim,
                'gold_claim': gold_claim,
                'matched': result.get('matched', False),
                'confidence': float(result.get('confidence', 0.0)),
                'gold_index': gold_idx,
                'reasoning': result.get('reasoning', '')
            }
            
        except Exception as e:
            return {
                'extracted_claim': extracted_claim,
                'gold_claim': None,
                'matched': False,
                'confidence': 0.0,
                'reasoning': f'Error: {str(e)}'
            }
    
    # Keep backward compatibility alias
    def match_with_vllm(self, extracted_claim: str, gold_claims: List[str], 
                       review_text: str) -> Dict[str, Any]:
        """Alias for match_with_llm for backward compatibility."""
        return self.match_with_llm(extracted_claim, gold_claims, review_text)
    
    def match_claims_parallel(self, extracted_claims: List[str], gold_claims: List[str],
                             review_text: str, num_workers: int = 10,
                             desc: str = "Matching claims") -> List[Dict[str, Any]]:
        """Match multiple claims in parallel using LLM."""
        if not extracted_claims or not OPENAI_AVAILABLE:
            return []
        
        matches = []
        iterable = tqdm(extracted_claims, desc=desc, disable=not TQDM_AVAILABLE) if TQDM_AVAILABLE else extracted_claims
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self.match_with_llm, claim, gold_claims, review_text): claim
                for claim in extracted_claims
            }
            
            for future in as_completed(futures):
                try:
                    match = future.result()
                    matches.append(match)
                except Exception as e:
                    claim = futures[future]
                    matches.append({
                        'extracted_claim': claim,
                        'gold_claim': None,
                        'matched': False,
                        'confidence': 0.0,
                        'reasoning': f'Error: {str(e)}'
                    })
        
        # Sort to maintain original order
        claim_to_match = {m['extracted_claim']: m for m in matches}
        return [claim_to_match.get(claim, {
            'extracted_claim': claim,
            'gold_claim': None,
            'matched': False,
            'confidence': 0.0,
            'reasoning': 'Not processed'
        }) for claim in extracted_claims]
    
    def evaluate_reference_free_parallel(self, claims: List[str], review_text: str,
                                        num_workers: int = 10,
                                        desc: str = "Evaluating claims") -> List[Dict[str, Any]]:
        """Evaluate reference-free metrics for multiple claims in parallel."""
        if not claims or not OPENAI_AVAILABLE:
            return []
        
        def evaluate_single_claim(claim: str) -> Dict[str, Any]:
            atomicity = self.evaluate_claim_atomicity(claim)
            fluency = self.evaluate_claim_fluency(claim)
            decontext = self.evaluate_claim_decontextualization(claim)
            faithfulness = self.evaluate_claim_faithfulness(claim, review_text)
            return {
                'claim': claim,
                'atomicity': atomicity,
                'fluency': fluency,
                'decontextualization': decontext,
                'faithfulness': faithfulness
            }
        
        results = []
        iterable = tqdm(claims, desc=desc, disable=not TQDM_AVAILABLE) if TQDM_AVAILABLE else claims
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(evaluate_single_claim, claim): claim
                for claim in claims
            }
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    claim = futures[future]
                    results.append({
                        'claim': claim,
                        'atomicity': {'atomicity': None, 'reasoning': f'Error: {str(e)}'},
                        'fluency': {'fluency': None, 'reasoning': f'Error: {str(e)}'},
                        'decontextualization': {'decontextualized': None, 'reasoning': f'Error: {str(e)}'},
                        'faithfulness': {'faithful': None, 'reasoning': f'Error: {str(e)}'}
                    })
        
        # Sort to maintain original order
        claim_to_result = {r['claim']: r for r in results}
        return [claim_to_result.get(claim, {
            'claim': claim,
            'atomicity': {'atomicity': None, 'reasoning': 'Not processed'},
            'fluency': {'fluency': None, 'reasoning': 'Not processed'},
            'decontextualization': {'decontextualized': None, 'reasoning': 'Not processed'},
            'faithfulness': {'faithful': None, 'reasoning': 'Not processed'}
        }) for claim in claims]
    
    def calculate_metrics(self, matches: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate precision, recall, and F1 from matches."""
        if not matches:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'matched_count': 0, 'total_extracted': 0}
        
        true_positives = sum(1 for m in matches if m.get('matched', False))
        total_extracted = len(matches)
        
        precision = true_positives / total_extracted if total_extracted > 0 else 0.0
        
        return {
            'precision': precision,
            'matched_count': true_positives,
            'total_extracted': total_extracted
        }
    
    def evaluate(self, benchmark_name: str = 'B2', output_dir: Optional[Path] = None,
                 use_vllm: bool = True, cosine_threshold: float = 0.3, 
                 max_reviews: Optional[int] = None,
                 skip_fenice: bool = False, skip_gemma: bool = False,
                 skip_reference_free: bool = False,
                 num_workers: int = 10):
        """
        Main evaluation function.
        
        Args:
            benchmark_name: Benchmark to use (B2, B3, etc.)
            output_dir: Output directory for results
            use_vllm: Whether to use LLM for matching
            cosine_threshold: Threshold for cosine similarity matching
            max_reviews: Maximum number of reviews to process (None = all)
            skip_fenice: Skip FENICE extraction (faster, no GPU needed)
            skip_gemma: Skip Gemma extraction (faster, no GPU needed)
            skip_reference_free: Skip reference-free metrics evaluation (much faster)
            num_workers: Number of parallel workers for API calls (default: 10)
        """
        print(f"Evaluating claim extraction against {benchmark_name} benchmark...")
        
        # Load benchmark data
        benchmark_map = {
            'B2': 'B2_reviewer_author.jsonl',
            'B3': 'B3_reviewer_paper.jsonl',
        }
        
        if benchmark_name not in benchmark_map:
            raise ValueError(f"Unknown benchmark: {benchmark_name}. Use B2 or B3.")
        
        benchmark_file = self.data_structure.benchmark_dir / benchmark_map[benchmark_name]
        
        if not benchmark_file.exists():
            raise FileNotFoundError(f"Benchmark file not found: {benchmark_file}")
        
        print(f"Loading benchmark data from {benchmark_file}...")
        benchmark_data = self.load_benchmark_data(benchmark_file)
        print(f"Loaded {len(benchmark_data)} benchmark entries")
        
        # Group by review text to get unique reviews with their gold claims
        review_data = {}
        for item in benchmark_data:
            review_text = item.get('review_text', '')
            if not review_text:
                continue
            
            # Use review_id as key to group by review
            review_id = item.get('review_id', '')
            if not review_id:
                review_id = f"review_{hash(review_text) % 1000000}"
            
            if review_id not in review_data:
                review_data[review_id] = {
                    'review_id': review_id,
                    'paper_id': item.get('paper_id'),
                    'review_text': review_text,
                    'gold_claims': [],
                    'items': []
                }
            
            # Add claim to gold claims
            claim = item.get('claim', '')
            if claim and claim not in review_data[review_id]['gold_claims']:
                review_data[review_id]['gold_claims'].append(claim)
            
            review_data[review_id]['items'].append(item)
        
        print(f"Found {len(review_data)} unique reviews")
        
        if max_reviews:
            review_data = dict(list(review_data.items())[:max_reviews])
            print(f"Processing first {len(review_data)} reviews")
        
        # Process each review
        results = []
        total_reviews = len(review_data)
        
        # Create progress bar for reviews
        review_iter = tqdm(
            enumerate(review_data.items(), 1),
            total=total_reviews,
            desc="Processing reviews",
            disable=not TQDM_AVAILABLE
        ) if TQDM_AVAILABLE else enumerate(review_data.items(), 1)
        
        for idx, (review_id, data) in review_iter:
            if TQDM_AVAILABLE:
                tqdm.write(f"\nReview {idx}/{total_reviews}: {review_id} ({len(data['gold_claims'])} gold claims)")
            else:
                print(f"\nProcessing review {idx}/{total_reviews}")
                print(f"  Review ID: {review_id}")
                print(f"  Paper ID: {data['paper_id']}")
                print(f"  Gold claims: {len(data['gold_claims'])}")
            
            review_text = data['review_text']
            
            # Extract claims using methods (skip if flags set)
            fenice_claims = []
            if not skip_fenice:
                print("  Extracting with FENICE...")
                fenice_claims = self.extract_claims_fenice(review_text)
                print(f"    Extracted {len(fenice_claims)} claims")
            else:
                print("  Skipping FENICE extraction")
            
            gemma_claims = []
            if not skip_gemma:
                print("  Extracting with Gemma...")
                gemma_claims = self.extract_claims_gemma(review_text)
                print(f"    Extracted {len(gemma_claims)} claims")
            else:
                print("  Skipping Gemma extraction")
            
            print("  Extracting with LLM...")
            llm_claims = self.extract_claims_llm(review_text)
            print(f"    Extracted {len(llm_claims)} claims")
            
            gold_claims = data['gold_claims']
            
            # Match FENICE claims
            print("  Matching FENICE claims (cosine similarity)...")
            fenice_cosine_matches = self.match_with_cosine_similarity(
                fenice_claims, gold_claims, cosine_threshold
            )
            
            fenice_vllm_matches = []
            if use_vllm and OPENAI_AVAILABLE and fenice_claims:
                fenice_vllm_matches = self.match_claims_parallel(
                    fenice_claims, gold_claims, review_text, num_workers,
                    desc=f"  Review {idx}: Matching FENICE claims (LLM)"
                )
            
            # Match Gemma claims
            print("  Matching Gemma claims (cosine similarity)...")
            gemma_cosine_matches = self.match_with_cosine_similarity(
                gemma_claims, gold_claims, cosine_threshold
            )
            
            gemma_vllm_matches = []
            if use_vllm and OPENAI_AVAILABLE and gemma_claims:
                gemma_vllm_matches = self.match_claims_parallel(
                    gemma_claims, gold_claims, review_text, num_workers,
                    desc=f"  Review {idx}: Matching Gemma claims (LLM)"
                )
            
            # Match LLM claims
            print("  Matching LLM claims (cosine similarity)...")
            llm_cosine_matches = self.match_with_cosine_similarity(
                llm_claims, gold_claims, cosine_threshold
            )
            
            llm_vllm_matches = []
            if use_vllm and OPENAI_AVAILABLE and llm_claims:
                llm_vllm_matches = self.match_claims_parallel(
                    llm_claims, gold_claims, review_text, num_workers,
                    desc=f"  Review {idx}: Matching LLM claims (LLM)"
                )
            
            # Calculate metrics
            fenice_cosine_metrics = self.calculate_metrics(fenice_cosine_matches)
            fenice_vllm_metrics = self.calculate_metrics(fenice_vllm_matches) if fenice_vllm_matches else {'precision': 0.0, 'matched_count': 0, 'total_extracted': 0}
            
            gemma_cosine_metrics = self.calculate_metrics(gemma_cosine_matches)
            gemma_vllm_metrics = self.calculate_metrics(gemma_vllm_matches) if gemma_vllm_matches else {'precision': 0.0, 'matched_count': 0, 'total_extracted': 0}
            
            llm_cosine_metrics = self.calculate_metrics(llm_cosine_matches)
            llm_vllm_metrics = self.calculate_metrics(llm_vllm_matches) if llm_vllm_matches else {'precision': 0.0, 'matched_count': 0, 'total_extracted': 0}
            
            # Evaluate claims with reference-free metrics (AIDA properties)
            fenice_reference_free = []
            gemma_reference_free = []
            llm_reference_free = []
            
            if not skip_reference_free and use_vllm and OPENAI_AVAILABLE:
                if fenice_claims:
                    fenice_reference_free = self.evaluate_reference_free_parallel(
                        fenice_claims, review_text, num_workers,
                        desc=f"  Review {idx}: FENICE reference-free metrics"
                    )
                
                if gemma_claims:
                    gemma_reference_free = self.evaluate_reference_free_parallel(
                        gemma_claims, review_text, num_workers,
                        desc=f"  Review {idx}: Gemma reference-free metrics"
                    )
                
                if llm_claims:
                    llm_reference_free = self.evaluate_reference_free_parallel(
                        llm_claims, review_text, num_workers,
                        desc=f"  Review {idx}: LLM reference-free metrics"
                    )
            else:
                if skip_reference_free:
                    print("  Skipping reference-free metrics evaluation")
            
            result = {
                'review_id': review_id,
                'paper_id': data['paper_id'],
                'review_index': idx,
                'review_text_length': len(review_text),
                'gold_claims_count': len(gold_claims),
                'gold_claims': gold_claims,
                
                # FENICE results
                'fenice_claims_count': len(fenice_claims),
                'fenice_claims': fenice_claims,
                'fenice_cosine_matches': fenice_cosine_matches,
                'fenice_cosine_precision': fenice_cosine_metrics['precision'],
                'fenice_cosine_matched': fenice_cosine_metrics['matched_count'],
                
                # Gemma results
                'gemma_claims_count': len(gemma_claims),
                'gemma_claims': gemma_claims,
                'gemma_cosine_matches': gemma_cosine_matches,
                'gemma_cosine_precision': gemma_cosine_metrics['precision'],
                'gemma_cosine_matched': gemma_cosine_metrics['matched_count'],
                
                # LLM results
                'llm_claims_count': len(llm_claims),
                'llm_claims': llm_claims,
                'llm_cosine_matches': llm_cosine_matches,
                'llm_cosine_precision': llm_cosine_metrics['precision'],
                'llm_cosine_matched': llm_cosine_metrics['matched_count'],
            }
            
            # Add LLM matching results if available
            if use_vllm and OPENAI_AVAILABLE:
                result.update({
                    'fenice_vllm_matches': fenice_vllm_matches,
                    'fenice_vllm_precision': fenice_vllm_metrics['precision'],
                    'fenice_vllm_matched': fenice_vllm_metrics['matched_count'],
                    'gemma_vllm_matches': gemma_vllm_matches,
                    'gemma_vllm_precision': gemma_vllm_metrics['precision'],
                    'gemma_vllm_matched': gemma_vllm_metrics['matched_count'],
                    'llm_vllm_matches': llm_vllm_matches,
                    'llm_vllm_precision': llm_vllm_metrics['precision'],
                    'llm_vllm_matched': llm_vllm_metrics['matched_count'],
                    'fenice_reference_free': fenice_reference_free,
                    'gemma_reference_free': gemma_reference_free,
                    'llm_reference_free': llm_reference_free,
                })
            
            results.append(result)
        
        # Save results
        if output_dir is None:
            output_dir = self.data_structure.logs_dir / 'claim_extraction_evaluation'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{benchmark_name}_claim_extraction_evaluation.jsonl"
        
        print(f"\nSaving results to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        # Calculate overall metrics
        self._calculate_overall_metrics(results, output_dir, benchmark_name)
        
        print(f"\nEvaluation completed!")
        print(f"Results saved to: {output_file}")
        
        return output_file
    
    def _calculate_reference_free_metrics(self, reference_free_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate metrics for reference-free evaluation."""
        if not reference_free_list:
            return {
                'atomicity_rate': 0.0,
                'fluency_rate': 0.0,
                'decontextualization_rate': 0.0,
                'faithfulness_rate': 0.0,
                'total_claims': 0
            }
        
        total = len(reference_free_list)
        atomicity_count = sum(1 for item in reference_free_list 
                             if item.get('atomicity', {}).get('atomicity') is True)
        fluency_count = sum(1 for item in reference_free_list 
                          if item.get('fluency', {}).get('fluency') is True)
        decontext_count = sum(1 for item in reference_free_list 
                            if item.get('decontextualization', {}).get('decontextualized') is True)
        faithfulness_count = sum(1 for item in reference_free_list 
                               if item.get('faithfulness', {}).get('faithful') is True)
        
        return {
            'atomicity_rate': atomicity_count / total if total > 0 else 0.0,
            'fluency_rate': fluency_count / total if total > 0 else 0.0,
            'decontextualization_rate': decontext_count / total if total > 0 else 0.0,
            'faithfulness_rate': faithfulness_count / total if total > 0 else 0.0,
            'total_claims': total,
            'atomicity_count': atomicity_count,
            'fluency_count': fluency_count,
            'decontextualization_count': decontext_count,
            'faithfulness_count': faithfulness_count
        }
    
    def _calculate_overall_metrics(self, results: List[Dict[str, Any]], output_dir: Path, benchmark_name: str):
        """Calculate and save overall metrics."""
        total_gold = sum(r['gold_claims_count'] for r in results)
        
        # Calculate metrics for each method
        methods = ['fenice', 'gemma', 'llm']
        overall_metrics = {
            'benchmark': benchmark_name,
            'evaluation_timestamp': datetime.now().isoformat(),
            'total_reviews': len(results),
            'total_gold_claims': total_gold,
        }
        
        for method in methods:
            # Cosine similarity metrics
            cosine_tp = sum(r[f'{method}_cosine_matched'] for r in results)
            cosine_total = sum(r[f'{method}_claims_count'] for r in results)
            cosine_precision = cosine_tp / cosine_total if cosine_total > 0 else 0.0
            
            # Count unique gold claims matched for recall
            cosine_unique_matched = 0
            for r in results:
                matches = r.get(f'{method}_cosine_matches', [])
                matched_indices = set()
                for match in matches:
                    if match.get('matched', False):
                        gold_idx = match.get('gold_index')
                        if gold_idx is not None:
                            matched_indices.add((r['review_index'], gold_idx))
                cosine_unique_matched += len(matched_indices)
            
            cosine_recall = cosine_unique_matched / total_gold if total_gold > 0 else 0.0
            cosine_f1 = 2 * cosine_precision * cosine_recall / (cosine_precision + cosine_recall) if (cosine_precision + cosine_recall) > 0 else 0.0
            
            overall_metrics[method] = {
                'total_extracted': cosine_total,
                'cosine_similarity': {
                    'precision': cosine_precision,
                    'recall': cosine_recall,
                    'f1': cosine_f1,
                    'true_positives': cosine_tp
                }
            }
            
            # vLLM metrics if available
            if any(f'{method}_vllm_matched' in r for r in results):
                vllm_tp = sum(r[f'{method}_vllm_matched'] for r in results)
                vllm_precision = vllm_tp / cosine_total if cosine_total > 0 else 0.0
                
                vllm_unique_matched = 0
                for r in results:
                    matches = r.get(f'{method}_vllm_matches', [])
                    matched_indices = set()
                    for match in matches:
                        if match.get('matched', False):
                            gold_idx = match.get('gold_index')
                            if gold_idx is not None:
                                matched_indices.add((r['review_index'], gold_idx))
                    vllm_unique_matched += len(matched_indices)
                
                vllm_recall = vllm_unique_matched / total_gold if total_gold > 0 else 0.0
                vllm_f1 = 2 * vllm_precision * vllm_recall / (vllm_precision + vllm_recall) if (vllm_precision + vllm_recall) > 0 else 0.0
                
                overall_metrics[method]['vllm'] = {
                    'precision': vllm_precision,
                    'recall': vllm_recall,
                    'f1': vllm_f1,
                    'true_positives': vllm_tp
                }
            
            # Reference-free metrics if available
            if any(f'{method}_reference_free' in r for r in results):
                all_reference_free = []
                for r in results:
                    ref_free = r.get(f'{method}_reference_free', [])
                    all_reference_free.extend(ref_free)
                
                ref_free_metrics = self._calculate_reference_free_metrics(all_reference_free)
                overall_metrics[method]['reference_free'] = ref_free_metrics
        
        # Save metrics
        metrics_file = output_dir / f"{benchmark_name}_claim_extraction_metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(overall_metrics, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "="*80)
        print("OVERALL METRICS")
        print("="*80)
        print(f"Total Reviews: {len(results)}")
        print(f"Total Gold Claims: {total_gold}")
        print()
        
        for method in methods:
            method_name = method.upper()
            metrics = overall_metrics[method]
            print(f"{method_name} Results:")
            print(f"  Cosine Similarity:")
            print(f"    Precision: {metrics['cosine_similarity']['precision']:.4f}")
            print(f"    Recall: {metrics['cosine_similarity']['recall']:.4f}")
            print(f"    F1: {metrics['cosine_similarity']['f1']:.4f}")
            print(f"    True Positives: {metrics['cosine_similarity']['true_positives']}/{metrics['total_extracted']}")
            
            if 'vllm' in metrics:
                print(f"  vLLM:")
                print(f"    Precision: {metrics['vllm']['precision']:.4f}")
                print(f"    Recall: {metrics['vllm']['recall']:.4f}")
                print(f"    F1: {metrics['vllm']['f1']:.4f}")
                print(f"    True Positives: {metrics['vllm']['true_positives']}/{metrics['total_extracted']}")
            
            if 'reference_free' in metrics:
                ref_free = metrics['reference_free']
                print(f"  Reference-Free Metrics:")
                print(f"    Atomicity Rate: {ref_free['atomicity_rate']:.4f} ({ref_free['atomicity_count']}/{ref_free['total_claims']})")
                print(f"    Fluency Rate: {ref_free['fluency_rate']:.4f} ({ref_free['fluency_count']}/{ref_free['total_claims']})")
                print(f"    Decontextualization Rate: {ref_free['decontextualization_rate']:.4f} ({ref_free['decontextualization_count']}/{ref_free['total_claims']})")
                print(f"    Faithfulness Rate: {ref_free['faithfulness_rate']:.4f} ({ref_free['faithfulness_count']}/{ref_free['total_claims']})")
            print()
        
        print("="*80)
        print(f"Metrics saved to: {metrics_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate claim extraction methods against benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate against B2 benchmark
  python3 -m claim_verification.evaluation.claim_extraction_evaluator --benchmark B2

  # Evaluate against B3 with custom settings
  python3 -m claim_verification.evaluation.claim_extraction_evaluator --benchmark B3 --cosine-threshold 0.5

  # Test with limited reviews
  python3 -m claim_verification.evaluation.claim_extraction_evaluator --benchmark B2 --max-reviews 10

  # Skip LLM matching (faster)
  python3 -m claim_verification.evaluation.claim_extraction_evaluator --benchmark B2 --no-vllm

  # Use a different LLM model as judge
  python3 -m claim_verification.evaluation.claim_extraction_evaluator --benchmark B2 --llm-model gpt-4o-mini

  # Fast mode: skip FENICE, Gemma, and reference-free metrics (only LLM extraction + matching)
  python3 -m claim_verification.evaluation.claim_extraction_evaluator --benchmark B2 --skip-fenice --skip-gemma --skip-reference-free

  # Use parallel processing with more workers (faster API calls)
  python3 -m claim_verification.evaluation.claim_extraction_evaluator --benchmark B2 --llm-model gpt-5-mini --num-workers 20

  # Only evaluate LLM extraction (fastest)
  python3 -m claim_verification.evaluation.claim_extraction_evaluator --benchmark B2 --skip-fenice --skip-gemma --skip-reference-free --no-vllm
        """
    )
    
    parser.add_argument('--benchmark', default='B2', choices=['B2', 'B3'],
                        help='Benchmark to use (default: B2)')
    parser.add_argument('--output-dir', type=Path,
                        help='Output directory (default: data/logs/claim_extraction_evaluation/)')
    parser.add_argument('--no-vllm', action='store_true',
                        help='Skip vLLM evaluation (faster)')
    parser.add_argument('--cosine-threshold', type=float, default=0.3,
                        help='Cosine similarity threshold for matching (default: 0.3)')
    parser.add_argument('--max-reviews', type=int,
                        help='Maximum number of reviews to process (default: all)')
    parser.add_argument('--llm-model', type=str, default=LLM_JUDGE_MODEL,
                        help=f'LLM model to use as judge (default: {LLM_JUDGE_MODEL})')
    parser.add_argument('--skip-fenice', action='store_true',
                        help='Skip FENICE extraction (faster, no GPU needed)')
    parser.add_argument('--skip-gemma', action='store_true',
                        help='Skip Gemma extraction (faster, no GPU needed)')
    parser.add_argument('--skip-reference-free', action='store_true',
                        help='Skip reference-free metrics evaluation (much faster, saves many API calls)')
    parser.add_argument('--num-workers', type=int, default=10,
                        help='Number of parallel workers for API calls (default: 10)')
    
    args = parser.parse_args()
    
    try:
        evaluator = ClaimExtractionEvaluator(llm_model=args.llm_model)
        
        output_file = evaluator.evaluate(
            benchmark_name=args.benchmark,
            output_dir=args.output_dir,
            use_vllm=not args.no_vllm,
            cosine_threshold=args.cosine_threshold,
            max_reviews=args.max_reviews,
            skip_fenice=args.skip_fenice,
            skip_gemma=args.skip_gemma,
            skip_reference_free=args.skip_reference_free,
            num_workers=args.num_workers
        )
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
