

import json
import re
import time
from pathlib import Path
import sys
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import os

from claim_verification.config import (
    DataStructure, DEFAULT_MODEL, BENCHMARK_TARGETS, as_str
)

# Import OpenAI
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class ClaimExtractor:
    """
    Extracts and filters atomic claims from reviewer comments.
    """
    
    def __init__(self, model: str = DEFAULT_MODEL, max_workers: int = 5):
        self.model = model
        self.max_workers = max_workers
        self.data_structure = DataStructure()
        self.client = None
    
    def _setup_client(self):
        """Setup OpenAI client."""
        if OpenAI is None:
            raise RuntimeError("openai package required. Install with: pip install openai")
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable must be set")
        
        self.client = OpenAI(api_key=api_key)
    
    def load_pairs(self) -> List[Dict[str, Any]]:
        """Load review-response pairs."""
        pairs_file = self.data_structure.get_paired_file()
        
        if not pairs_file.exists():
            raise FileNotFoundError(
                f"Pairs file not found: {pairs_file}\n"
                "Please run pairer.py first."
            )
        
        with open(pairs_file, 'r', encoding='utf-8') as f:
            pairs = json.load(f)
        
        print(f"Loaded {len(pairs)} review-response pairs")
        return pairs
    
    def _extract_atomic_claims(self, review_text: str) -> List[Dict[str, Any]]:
        """
        Extract atomic claims from a review text using LLM.
        
        Returns list of claims with metadata.
        """
        if not review_text or len(review_text.strip()) < 50:
            return []
        
        prompt = f"""
You are an expert at analyzing academic peer reviews and extracting specific, atomic claims.

Given the following review text, extract ALL atomic claims that represent:
1. **Weaknesses**: Specific criticisms, concerns, or issues identified by the reviewer
2. **Questions that indicate problems**: Questions that suggest gaps or concerns
3. **Concerns about methodology, results, or evaluation**
4. **Claims about missing elements**: Missing baselines, experiments, comparisons
5. **Technical issues**: Incorrect statements, bugs, or errors mentioned

CRITICAL REQUIREMENTS:
- Each claim must be a SINGLE, ATOMIC statement (one specific point)
- Each claim should be self-contained and independently verifiable
- Claims should be 10-50 words
- Preserve the original meaning and specificity

DO NOT include:
- General praise or positive statements
- Summary statements
- Vague or overly broad statements
- Claims that combine multiple unrelated points

Review text:
{review_text}

Return ONLY a JSON array of objects, where each object has:
- "claim": the atomic claim text
- "claim_type": one of "experimental", "methodology", "quantitative", "baseline", "clarity", "novelty", "presentation", "other"

Example format:
[
  {{"claim": "The paper does not compare with the SOTA method X", "claim_type": "baseline"}},
  {{"claim": "The accuracy results on dataset Y are not statistically significant", "claim_type": "quantitative"}}
]
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting atomic claims from academic reviews. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
            )
            
            content = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith('```'):
                content = re.sub(r'```(?:json)?\s*', '', content)
                content = re.sub(r'```\s*$', '', content)
            
            claims = json.loads(content)
            
            if isinstance(claims, list):
                # Validate and clean claims
                valid_claims = []
                for claim in claims:
                    if isinstance(claim, dict) and 'claim' in claim:
                        claim_text = claim['claim'].strip()
                        if claim_text and len(claim_text) > 15:
                            valid_claims.append({
                                'claim': claim_text,
                                'claim_type': claim.get('claim_type', 'other'),
                                'claim_length': len(claim_text),
                            })
                return valid_claims
            else:
                return []
                
        except json.JSONDecodeError as e:
            print(f"    JSON parse error: {e}")
            return []
        except Exception as e:
            print(f"    Error extracting claims: {e}")
            return []
    
    def _is_verifiable_claim_llm(self, claim: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Use LLM to check if a claim is verifiable (can be checked against the paper).
        
        Verifiable claims:
        - Experimental results claims
        - Methodology claims
        - Dataset/baseline claims
        - Quantitative claims (numbers, metrics)
        - Technical accuracy claims
        
        Non-verifiable claims:
        - Novelty/originality claims
        - Picture/figure quality claims
        - Writing quality claims
        - Subjective opinions
        - Related work coverage claims
        
        Returns:
            Tuple of (is_verifiable: bool, reason: str)
        """
        claim_text = claim.get('claim', '')
        
        if not claim_text:
            return False, "Empty claim"
        
        prompt = f"""
You are an expert at classifying academic review claims into verifiable vs non-verifiable.

A claim is VERIFIABLE if it can be objectively checked against the paper content:
- Experimental results (accuracy, performance, metrics, numbers)
- Methodology details (algorithms, techniques, implementations)
- Dataset information (which datasets used, dataset sizes)
- Baseline/comparison claims (what methods were compared)
- Quantitative claims (any specific numbers, statistics)
- Technical implementation details (code, reproducibility)
- Ablation study claims

A claim is NOT VERIFIABLE if it's subjective or cannot be objectively verified:
- Novelty or originality claims ("not novel", "limited contribution")
- Figure/picture quality claims ("figures are unclear")
- Writing quality claims ("poorly written", "typos")
- Presentation claims ("hard to follow", "not well organized")
- Subjective opinions ("not interesting", "incremental")
- Related work completeness ("missing citations", "should cite X")
- Motivation claims ("weak motivation")

Claim: "{claim_text}"

Is this claim verifiable? Return ONLY a JSON object:
{{"is_verifiable": true or false, "reason": "brief explanation", "claim_type": "experimental" or "methodology" or "quantitative" or "baseline" or "novelty" or "presentation" or "subjective" or "other"}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at classifying claims. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
            )
            
            content = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks
            if content.startswith('```'):
                content = re.sub(r'```(?:json)?\s*', '', content)
                content = re.sub(r'```\s*$', '', content)
            
            result = json.loads(content)
            
            is_verifiable = result.get('is_verifiable', False)
            reason = result.get('reason', '')
            claim_type = result.get('claim_type', 'other')
            
            # Update claim with the determined type
            claim['claim_type'] = claim_type
            claim['verifiability_reason'] = reason
            
            return is_verifiable, reason
            
        except Exception as e:
            # Fallback to simple heuristic on error
            return self._is_verifiable_claim_heuristic(claim)
    
    def _is_verifiable_claim_heuristic(self, claim: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Fallback heuristic-based check for verifiability.
        Used when LLM call fails.
        """
        claim_type = claim.get('claim_type', 'other').lower()
        claim_text = claim.get('claim', '').lower()
        
        # Verifiable types
        verifiable_types = ['experimental', 'methodology', 'quantitative', 'baseline']
        
        if claim_type in verifiable_types:
            return True, f"Claim type '{claim_type}' is verifiable"
        
        # Keyword checks
        verifiable_keywords = [
            'accuracy', 'performance', 'result', 'experiment', 'dataset',
            'baseline', 'benchmark', 'metric', 'evaluation', 'ablation',
            'table', 'training', 'parameter', 'loss', 'convergence',
            'runtime', 'method', 'algorithm', 'implementation', 'statistical',
        ]
        
        non_verifiable_keywords = [
            'novel', 'originality', 'contribution', 'figure quality',
            'picture', 'visualization', 'writing', 'presentation',
            'clarity', 'grammar', 'typo', 'related work', 'citation',
            'motivation', 'interesting', 'boring',
        ]
        
        has_verifiable = any(kw in claim_text for kw in verifiable_keywords)
        has_non_verifiable = any(kw in claim_text for kw in non_verifiable_keywords)
        
        if has_non_verifiable and not has_verifiable:
            return False, "Contains non-verifiable keywords"
        
        if has_verifiable:
            return True, "Contains verifiable keywords"
        
        return claim_type in verifiable_types, "Default based on claim type"
    
    def _filter_verifiable_claims_llm(self, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Use LLM to filter claims and keep only verifiable ones.
        Processes each claim individually for accurate classification.
        """
        if not claims:
            return []
        
        filtered = []
        
        for i, claim in enumerate(claims):
            try:
                is_verifiable, reason = self._is_verifiable_claim_llm(claim)
                
                claim['is_verifiable'] = is_verifiable
                claim['verifiability_reason'] = reason
                
                if is_verifiable:
                    filtered.append(claim)
                    
            except Exception as e:
                print(f"    Error checking claim {i+1}: {e}")
                # On error, use heuristic fallback
                is_verifiable, reason = self._is_verifiable_claim_heuristic(claim)
                claim['is_verifiable'] = is_verifiable
                if is_verifiable:
                    filtered.append(claim)
            
            # Brief pause between API calls
            time.sleep(0.1)
        
        return filtered
    
    def extract_claims_from_pairs(
        self, 
        pairs: List[Dict[str, Any]], 
        target_claims: int = BENCHMARK_TARGETS['B2_reviewer_author'],
        filter_verifiable: bool = False
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract claims from all review-response pairs.
        
        Returns:
        - all_claims: All extracted claims
        - verifiable_claims: Filtered verifiable claims only
        """
        if self.client is None:
            self._setup_client()
        
        all_claims = []
        total_extracted = 0
        lock = Lock()
        
        def process_pair(pair_idx: int, pair: Dict[str, Any]) -> List[Dict[str, Any]]:
            """Process a single pair and extract claims."""
            nonlocal total_extracted
            
            review_text = pair.get('review_text', '')
            
            if not review_text or len(review_text.strip()) < 50:
                return []
            
            # Extract atomic claims
            claims = self._extract_atomic_claims(review_text)
            
            if not claims:
                return []
            
            # Add metadata to each claim
            pair_claims = []
            for claim in claims:
                claim_record = {
                    **claim,
                    'paper_id': pair.get('paper_id'),
                    'paper_title': pair.get('paper_title'),
                    'paper_venue': pair.get('paper_venue'),
                    'paper_decision': pair.get('paper_decision'),  # Add decision
                    'review_id': pair.get('review_id'),
                    'reviewer': pair.get('reviewer'),
                    'review_text': review_text,
                    'author_response': pair.get('response_text'),  # Individual paired response
                    'combined_author_response': pair.get('combined_author_response', ''),  # ALL responses for paper
                    'is_direct_reply': pair.get('is_direct_reply', False),
                    'extraction_timestamp': datetime.now().isoformat(),
                }
                pair_claims.append(claim_record)
            
            with lock:
                total_extracted += len(pair_claims)
                print(f"  [{pair_idx+1}/{len(pairs)}] Extracted {len(pair_claims)} claims (total: {total_extracted})")
            
            return pair_claims
        
        print(f"Extracting claims from {len(pairs)} pairs...")
        print(f"Target: {target_claims} claims")
        
        # Process pairs in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(process_pair, i, pair): i 
                for i, pair in enumerate(pairs)
            }
            
            for future in as_completed(futures):
                try:
                    pair_claims = future.result()
                    all_claims.extend(pair_claims)
                    
                    # Check if we've reached target
                    if len(all_claims) >= target_claims:
                        print(f"\nReached target of {target_claims} claims")
                        # Cancel remaining futures
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        break
                        
                except Exception as e:
                    pair_idx = futures[future]
                    print(f"  Error processing pair {pair_idx}: {e}")
                
                # Brief pause between batches
                time.sleep(0.1)
        
        # Limit to target
        all_claims = all_claims[:target_claims]
        
        print(f"\nExtracted {len(all_claims)} total claims")
        
        # Filter verifiable claims
        print("\nFiltering for verifiable claims...")
        verifiable_claims = []
        
        if filter_verifiable:
            # Process in batches for LLM filtering
            batch_size = 20
            for i in range(0, len(all_claims), batch_size):
                batch = all_claims[i:i+batch_size]
                filtered_batch = self._filter_verifiable_claims_llm(batch)
                verifiable_claims.extend(filtered_batch)
                print(f"  Processed batch {i//batch_size + 1}: {len(filtered_batch)}/{len(batch)} verifiable")
                time.sleep(0.2)
        else:
            # Heuristic-based filtering (fallback)
            for claim in all_claims:
                is_verifiable, reason = self._is_verifiable_claim_heuristic(claim)
                claim['is_verifiable'] = is_verifiable
                claim['verifiability_reason'] = reason
                if is_verifiable:
                    verifiable_claims.append(claim)
        
        print(f"Verifiable claims: {len(verifiable_claims)} out of {len(all_claims)}")
        
        # Print claim type distribution
        type_counts = {}
        for claim in all_claims:
            ct = claim.get('claim_type', 'other')
            type_counts[ct] = type_counts.get(ct, 0) + 1
        print(f"Claim type distribution: {type_counts}")
        
        return all_claims, verifiable_claims
    
    def save_claims(
        self, 
        all_claims: List[Dict[str, Any]], 
        verifiable_claims: List[Dict[str, Any]]
    ):
        """Save extracted claims to files."""
        # Save all claims
        claims_file = self.data_structure.get_claims_file()
        with open(claims_file, 'w', encoding='utf-8') as f:
            json.dump(all_claims, f, indent=2, ensure_ascii=False)
        print(f"Saved all claims to {as_str(claims_file)}")
        
        # Save verifiable claims
        verifiable_file = self.data_structure.get_verifiable_claims_file()
        with open(verifiable_file, 'w', encoding='utf-8') as f:
            json.dump(verifiable_claims, f, indent=2, ensure_ascii=False)
        print(f"Saved verifiable claims to {as_str(verifiable_file)}")
        
        # Save summary
        summary = {
            'extraction_timestamp': datetime.now().isoformat(),
            'model': self.model,
            'total_claims': len(all_claims),
            'verifiable_claims': len(verifiable_claims),
            'claim_type_distribution': {},
            'venue_distribution': {},
        }
        
        for claim in all_claims:
            ct = claim.get('claim_type', 'other')
            summary['claim_type_distribution'][ct] = summary['claim_type_distribution'].get(ct, 0) + 1
            
            venue = claim.get('paper_venue', 'unknown')
            summary['venue_distribution'][venue] = summary['venue_distribution'].get(venue, 0) + 1
        
        summary_file = self.data_structure.logs_dir / "claim_extraction_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved summary to {as_str(summary_file)}")
    
    def run(
        self, 
        target_claims: int = BENCHMARK_TARGETS['B2_reviewer_author'],
        filter_verifiable: bool = True
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Run the full claim extraction pipeline.
        
        Returns:
        - all_claims: All extracted claims
        - verifiable_claims: Filtered verifiable claims only
        """
        print("="*60)
        print("CLAIM EXTRACTION")
        print("="*60)
        
        # Load pairs
        pairs = self.load_pairs()
        
        # Extract and filter claims
        all_claims, verifiable_claims = self.extract_claims_from_pairs(
            pairs, 
            target_claims=target_claims,
            filter_verifiable=filter_verifiable
        )
        
        # Save results
        self.save_claims(all_claims, verifiable_claims)
        
        print("="*60)
        print(f"EXTRACTION COMPLETE")
        print(f"  Total claims: {len(all_claims)}")
        print(f"  Verifiable claims: {len(verifiable_claims)}")
        print("="*60)
        
        return all_claims, verifiable_claims


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract and filter claims from reviews')
    parser.add_argument('--model', default=DEFAULT_MODEL,
                       help=f'OpenAI model to use (default: {DEFAULT_MODEL})')
    parser.add_argument('--target', type=int, default=BENCHMARK_TARGETS['B2_reviewer_author'],
                       help=f'Target number of claims (default: {BENCHMARK_TARGETS["B2_reviewer_author"]})')
    parser.add_argument('--max-workers', type=int, default=5,
                       help='Number of parallel workers (default: 5)')
    parser.add_argument('--no-filter', action='store_true',
                       help='Skip verifiable claim filtering')
    
    args = parser.parse_args()
    
    try:
        extractor = ClaimExtractor(model=args.model, max_workers=args.max_workers)
        all_claims, verifiable_claims = extractor.run(
            target_claims=args.target,
            filter_verifiable=not args.no_filter
        )
        
        print(f"\nExtraction completed!")
        print(f"Total claims: {len(all_claims)}")
        print(f"Verifiable claims: {len(verifiable_claims)}")
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

