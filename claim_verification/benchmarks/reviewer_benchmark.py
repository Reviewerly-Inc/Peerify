

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
    DataStructure, DEFAULT_MODEL, BENCHMARK_TARGETS, as_str, normalize_decision
)

try:
    from claim_verification.preprocessing import chunking as chunking_module
except ImportError:
    chunking_module = None

# Import OpenAI
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# Valid labels for B2 and B3
VALID_LABELS = ["Supported", "Partially Supported", "Contradicted", "Not Determined"]


class ReviewerBenchmarkGenerator:
    """
    Generates B2 and B3 benchmarks from the same reviewer claims.
    
    B2: Labels claims against author response
    B3: Labels same claims against paper content
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
    
    def load_claims(self) -> List[Dict[str, Any]]:
        """Load extracted claims from claim_extractor."""
        claims_file = self.data_structure.get_claims_file()
        
        if not claims_file.exists():
            raise FileNotFoundError(
                f"Claims file not found: {claims_file}\n"
                "Please run claim_extractor.py first."
            )
        
        with open(claims_file, 'r', encoding='utf-8') as f:
            claims = json.load(f)
        
        print(f"Loaded {len(claims)} claims")
        return claims
    
    def _get_paper_content(self, submission_id: str) -> Optional[str]:
        """Get paper content (markdown) for a submission."""
        md_path = self.data_structure.markdown_dir / f"{submission_id}.md"
        
        if md_path.exists():
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Truncate if too long (for LLM context)
            if len(content) > 30000:
                content = content[:30000] + "\n\n[Content truncated...]"
            return content
        
        # Try to convert from PDF
        pdf_path = self.data_structure.pdfs_dir / f"{submission_id}.pdf"
        if pdf_path.exists() and chunking_module is not None:
            try:
                if chunking_module.convert_pdf_to_markdown(str(pdf_path), str(md_path)):
                    with open(md_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if len(content) > 30000:
                        content = content[:30000] + "\n\n[Content truncated...]"
                    return content
            except Exception as e:
                print(f"    Error converting PDF: {e}")
        
        return None
    
    def _label_claim_against_author_response(
        self, 
        claim: str, 
        author_response: str
    ) -> Dict[str, Any]:
        """
        Label a claim against the author's response (B2).
        
        Labels:
        - Supported: Authors clearly agree, acknowledge as valid
        - Partially Supported: Authors partially address, acknowledge some validity
        - Contradicted: Authors explicitly disagree, provide counter-evidence
        - Not Determined: Authors don't address this claim
        """
        if not author_response or len(author_response.strip()) < 20:
            return {
                'label': 'Not Determined',
                'justification': 'No author response available',
                'evidence': ''
            }
        
        prompt = f"""
You are an expert at analyzing academic discourse between reviewers and authors.

Given a reviewer's claim and the author's response, determine how the authors address this claim.

Guidelines:
- "Supported": Authors clearly AGREE with the claim, acknowledge it as valid, or confirm what the reviewer stated
- "Partially Supported": Authors partially address the claim, acknowledge some validity but not fully
- "Contradicted": Authors explicitly DISAGREE with the claim, provide counter-evidence, or refute it
- "Not Determined": Authors don't address this specific claim at all, or it's unclear

Reviewer Claim: {claim}

Author's Response:
{author_response}

Return a JSON object:
{{"label": "Supported" or "Partially Supported" or "Contradicted" or "Not Determined", "justification": "brief explanation", "evidence": "exact quote from response supporting this label"}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing academic discourse. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
            )
            
            content = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks
            if content.startswith('```'):
                content = re.sub(r'```(?:json)?\s*', '', content)
                content = re.sub(r'```\s*$', '', content)
            
            result = json.loads(content)
            
            # Validate label
            label = result.get('label', 'Not Determined')
            if label not in VALID_LABELS:
                label = 'Not Determined'
            
            return {
                'label': label,
                'justification': result.get('justification', ''),
                'evidence': result.get('evidence', '')
            }
            
        except Exception as e:
            return {
                'label': 'Not Determined',
                'justification': f'Error during labeling: {e}',
                'evidence': ''
            }
    
    def _label_claim_against_paper(
        self, 
        claim: str, 
        paper_content: str
    ) -> Dict[str, Any]:
        """
        Label a claim against the paper content (B3).
        
        This determines if the claim is TRUE according to the paper:
        - Supported: Claim is TRUE according to the paper
        - Partially Supported: Claim is partially true
        - Contradicted: Claim is FALSE according to the paper
        - Not Determined: Cannot determine from paper content
        """
        if not paper_content or len(paper_content.strip()) < 100:
            return {
                'label': 'Not Determined',
                'justification': 'No paper content available',
                'evidence': '',
                'section': ''
            }
        
        prompt = f"""
You are an expert at analyzing academic papers and determining if reviewer claims are true.

Given a reviewer's claim about a paper and the paper content, determine if the claim is TRUE according to the paper.

Important: You are evaluating whether the claim accurately describes what is in the paper.

Guidelines:
- "Supported": The claim is TRUE according to the paper. The paper confirms what the claim states.
- "Partially Supported": The claim is partially true, some aspects are correct but others are not.
- "Contradicted": The claim is FALSE according to the paper. The paper contradicts the claim.
- "Not Determined": The paper doesn't address this topic, or there's insufficient information.

Example: If claim says "the paper lacks comparison with baseline X" and paper indeed doesn't compare with X, the label is "Supported" (the claim is true).

Reviewer Claim: {claim}

Paper Content:
{paper_content[:15000]}

Return a JSON object:
{{"label": "Supported" or "Partially Supported" or "Contradicted" or "Not Determined", "justification": "brief explanation referencing specific sections", "evidence": "exact quote from paper", "section": "section name where evidence found"}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing academic papers. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
            )
            
            content = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks
            if content.startswith('```'):
                content = re.sub(r'```(?:json)?\s*', '', content)
                content = re.sub(r'```\s*$', '', content)
            
            result = json.loads(content)
            
            # Validate label
            label = result.get('label', 'Not Determined')
            if label not in VALID_LABELS:
                label = 'Not Determined'
            
            return {
                'label': label,
                'justification': result.get('justification', ''),
                'evidence': result.get('evidence', ''),
                'section': result.get('section', '')
            }
            
        except Exception as e:
            return {
                'label': 'Not Determined',
                'justification': f'Error during labeling: {e}',
                'evidence': '',
                'section': ''
            }
    
    def generate_b2_b3(
        self, 
        claims: List[Dict[str, Any]],
        target: int = BENCHMARK_TARGETS['B2_reviewer_author']
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Generate B2 and B3 benchmarks from the same claims.
        """
        if self.client is None:
            self._setup_client()
        
        # Limit claims to target
        claims = claims[:target]
        
        print(f"Processing {len(claims)} claims for B2 and B3...")
        
        b2_results = []
        b3_results = []
        
        # Cache paper content to avoid repeated loading
        paper_cache = {}
        
        lock = Lock()
        processed = 0
        
        def process_claim(claim_idx: int, claim: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            """Process a single claim for both B2 and B3."""
            nonlocal processed
            
            claim_text = claim.get('claim', '')
            paper_id = claim.get('paper_id', '')
            
            # B2: Label against COMBINED author response (all responses for this paper)
            # Use combined_author_response if available, otherwise fall back to individual response
            combined_author_response = claim.get('combined_author_response', '')
            if not combined_author_response:
                # Fallback to individual response if combined not available
                combined_author_response = claim.get('author_response', '')
            
            b2_result = self._label_claim_against_author_response(claim_text, combined_author_response)
            
            # B3: Label against paper content
            # Get paper content (with caching)
            if paper_id not in paper_cache:
                paper_cache[paper_id] = self._get_paper_content(paper_id)
            paper_content = paper_cache.get(paper_id)
            
            b3_result = self._label_claim_against_paper(claim_text, paper_content or '')
            
            # Build result records
            paper_decision = claim.get('paper_decision')
            decision_normalized = normalize_decision(paper_decision) if paper_decision else 'Unknown'
            
            base_record = {
                'claim': claim_text,
                'claim_type': claim.get('claim_type', 'other'),
                'paper_id': paper_id,
                'paper_title': claim.get('paper_title'),
                'paper_venue': claim.get('paper_venue'),
                'paper_decision': paper_decision,  # Original decision value
                'decision': decision_normalized,  # Normalized decision category
                'review_id': claim.get('review_id'),
                'reviewer': claim.get('reviewer'),
                'review_text': claim.get('review_text'),
                'labeling_timestamp': datetime.now().isoformat(),
                'model': self.model,
            }
            
            b2_record = {
                **base_record,
                'label': b2_result['label'],
                'justification': b2_result['justification'],
                'evidence': b2_result['evidence'],
                'combined_author_response': combined_author_response,  # Combined response used for labeling
                'author_response': claim.get('author_response', ''),  # Individual paired response (for reference)
            }
            
            b3_record = {
                **base_record,
                'label': b3_result['label'],
                'justification': b3_result['justification'],
                'evidence': b3_result['evidence'],
                'section': b3_result.get('section', ''),
            }
            
            with lock:
                processed += 1
                if processed % 10 == 0:
                    print(f"  Processed {processed}/{len(claims)} claims")
            
            return b2_record, b3_record
        
        # Process claims in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(process_claim, i, claim): i 
                for i, claim in enumerate(claims)
            }
            
            results = {}
            for future in as_completed(futures):
                claim_idx = futures[future]
                try:
                    b2_record, b3_record = future.result()
                    results[claim_idx] = (b2_record, b3_record)
                except Exception as e:
                    print(f"  Error processing claim {claim_idx}: {e}")
                
                time.sleep(0.1)  # Rate limiting
        
        # Sort by index to maintain order
        for idx in sorted(results.keys()):
            b2_record, b3_record = results[idx]
            b2_results.append(b2_record)
            b3_results.append(b3_record)
        
        print(f"\nGenerated {len(b2_results)} B2 records and {len(b3_results)} B3 records")
        
        # Print label distributions
        b2_dist = {}
        b3_dist = {}
        for r in b2_results:
            b2_dist[r['label']] = b2_dist.get(r['label'], 0) + 1
        for r in b3_results:
            b3_dist[r['label']] = b3_dist.get(r['label'], 0) + 1
        
        print(f"B2 label distribution: {b2_dist}")
        print(f"B3 label distribution: {b3_dist}")
        
        return b2_results, b3_results
    
    def save_benchmarks(
        self, 
        b2_results: List[Dict[str, Any]], 
        b3_results: List[Dict[str, Any]],
        split_by_decision: bool = True
    ):
        """Save B2 and B3 benchmarks, optionally split by decision type."""
        from collections import defaultdict
        
        # Group by decision
        b2_by_decision = defaultdict(list)
        b3_by_decision = defaultdict(list)
        
        for record in b2_results:
            decision = record.get('decision', 'Unknown')
            b2_by_decision[decision].append(record)
        
        for record in b3_results:
            decision = record.get('decision', 'Unknown')
            b3_by_decision[decision].append(record)
        
        # Save B2 (all + by decision)
        b2_file = self.data_structure.get_benchmark_file("B2_reviewer_author")
        with open(b2_file, 'w', encoding='utf-8') as f:
            for record in b2_results:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"Saved B2 (all) to {as_str(b2_file)}: {len(b2_results)} records")
        
        if split_by_decision:
            for decision, records in b2_by_decision.items():
                b2_decision_file = self.data_structure.get_benchmark_file("B2_reviewer_author", decision)
                with open(b2_decision_file, 'w', encoding='utf-8') as f:
                    for record in records:
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
                print(f"  Saved B2 ({decision}) to {as_str(b2_decision_file)}: {len(records)} records")
        
        # Save B3 (all + by decision)
        b3_file = self.data_structure.get_benchmark_file("B3_reviewer_paper")
        with open(b3_file, 'w', encoding='utf-8') as f:
            for record in b3_results:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"Saved B3 (all) to {as_str(b3_file)}: {len(b3_results)} records")
        
        if split_by_decision:
            for decision, records in b3_by_decision.items():
                b3_decision_file = self.data_structure.get_benchmark_file("B3_reviewer_paper", decision)
                with open(b3_decision_file, 'w', encoding='utf-8') as f:
                    for record in records:
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
                print(f"  Saved B3 ({decision}) to {as_str(b3_decision_file)}: {len(records)} records")
        
        # Save combined (same claims with both labels)
        combined = []
        for b2, b3 in zip(b2_results, b3_results):
            combined_record = {
                'claim': b2['claim'],
                'claim_type': b2.get('claim_type'),
                'paper_id': b2['paper_id'],
                'paper_title': b2.get('paper_title'),
                'paper_venue': b2.get('paper_venue'),
                'paper_decision': b2.get('paper_decision'),
                'decision': b2.get('decision', 'Unknown'),  # Add normalized decision
                'review_id': b2.get('review_id'),
                'reviewer': b2.get('reviewer'),
                'review_text': b2.get('review_text'),
                'author_response': b2.get('author_response'),
                
                # B2 fields
                'b2_label': b2['label'],
                'b2_justification': b2['justification'],
                'b2_evidence': b2['evidence'],
                
                # B3 fields
                'b3_label': b3['label'],
                'b3_justification': b3['justification'],
                'b3_evidence': b3['evidence'],
                'b3_section': b3.get('section', ''),
                
                'labeling_timestamp': datetime.now().isoformat(),
                'model': self.model,
            }
            combined.append(combined_record)
        
        combined_file = self.data_structure.get_benchmark_file("B2_B3_combined")
        with open(combined_file, 'w', encoding='utf-8') as f:
            for record in combined:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"Saved combined B2+B3 to {as_str(combined_file)}: {len(combined)} records")
        
        # Save summaries
        for name, results in [('B2', b2_results), ('B3', b3_results)]:
            label_dist = {}
            venue_dist = {}
            for r in results:
                label_dist[r['label']] = label_dist.get(r['label'], 0) + 1
                venue = r.get('paper_venue', 'unknown')
                venue_dist[venue] = venue_dist.get(venue, 0) + 1
            
            summary = {
                'benchmark': name,
                'generation_timestamp': datetime.now().isoformat(),
                'model': self.model,
                'total_claims': len(results),
                'label_distribution': label_dist,
                'venue_distribution': venue_dist,
            }
            
            summary_file = self.data_structure.logs_dir / f"{name}_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"Saved {name} summary to {as_str(summary_file)}")
    
    def run(self, target: int = BENCHMARK_TARGETS['B2_reviewer_author']) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Run the full B2/B3 benchmark generation pipeline.
        """
        print("="*60)
        print("B2 & B3: REVIEWER CLAIMS BENCHMARKS")
        print("="*60)
        
        # Load claims
        claims = self.load_claims()
        
        # Generate B2 and B3
        b2_results, b3_results = self.generate_b2_b3(claims, target)
        
        if not b2_results:
            print("ERROR: No results generated!")
            return [], []
        
        # Save benchmarks
        self.save_benchmarks(b2_results, b3_results)
        
        print("="*60)
        print(f"B2 & B3 COMPLETE")
        print(f"  B2 (vs Author): {len(b2_results)} claims")
        print(f"  B3 (vs Paper): {len(b3_results)} claims")
        print("="*60)
        
        return b2_results, b3_results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate B2 and B3 benchmarks')
    parser.add_argument('--model', default=DEFAULT_MODEL,
                       help=f'OpenAI model to use (default: {DEFAULT_MODEL})')
    parser.add_argument('--target', type=int, default=BENCHMARK_TARGETS['B2_reviewer_author'],
                       help=f'Target number of claims (default: {BENCHMARK_TARGETS["B2_reviewer_author"]})')
    parser.add_argument('--max-workers', type=int, default=5,
                       help='Number of parallel workers (default: 5)')
    
    args = parser.parse_args()
    
    try:
        generator = ReviewerBenchmarkGenerator(model=args.model, max_workers=args.max_workers)
        b2, b3 = generator.run(target=args.target)
        
        print(f"\nBenchmarks generated:")
        print(f"  B2: {len(b2)} claims")
        print(f"  B3: {len(b3)} claims")
        
    except Exception as e:
        print(f"Error generating benchmarks: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

