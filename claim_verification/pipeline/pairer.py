#!/usr/bin/env python3

import json
import random
import re
from pathlib import Path
import sys
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from claim_verification.config import (
    DataStructure, NUM_PAPERS_TO_SELECT, as_str
)


class ReviewResponsePairer:
    """
    Pairs reviewer comments with specific author responses using replyto threading.
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.data_structure = DataStructure()
        random.seed(random_seed)
    
    def load_papers(self) -> List[Dict[str, Any]]:
        """Load crawled papers from raw data."""
        combined_file = self.data_structure.raw_dir / "all_papers.json"
        
        if not combined_file.exists():
            raise FileNotFoundError(
                f"Combined papers file not found: {combined_file}\n"
                "Please run crawler.py first."
            )
        
        with open(combined_file, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        
        print(f"Loaded {len(papers)} papers from {as_str(combined_file)}")
        return papers
    
    def _is_promise_response(self, response_text: str) -> bool:
        """
        Check if a response is mostly about future updates/promises rather than
        substantive addressing of the review point.
        
        Returns True if the response is primarily a promise, False if substantive.
        """
        response_lower = response_text.lower()
        
        # Promise indicators
        promise_patterns = [
            r'\bwe will\b',
            r'\bwe plan to\b',
            r'\bwe intend to\b',
            r'\bwe are going to\b',
            r'\bin the revision\b',
            r'\bin the camera[- ]?ready\b',
            r'\bfuture work\b',
            r'\bfuture version\b',
            r'\bupdated version\b',
            r'\bwe have updated\b',
            r'\bplease see the updated\b',
            r'\bwe have revised\b',
        ]
        
        # Substantive response indicators
        substantive_patterns = [
            r'\bwe agree\b',
            r'\bwe disagree\b',
            r'\bactually\b',
            r'\bin fact\b',
            r'\bthe reason is\b',
            r'\bthis is because\b',
            r'\bour method\b',
            r'\bour approach\b',
            r'\bthe results show\b',
            r'\bas shown in\b',
            r'\bwe clarify\b',
            r'\bto clarify\b',
            r'\bthe confusion\b',
            r'\bwe address\b',
            r'\bregarding\b',
            r'\bconcerning\b',
        ]
        
        promise_count = sum(1 for p in promise_patterns if re.search(p, response_lower))
        substantive_count = sum(1 for p in substantive_patterns if re.search(p, response_lower))
        
        # If mostly promises and few substantive indicators, it's a promise
        if promise_count > 2 and substantive_count < 2:
            return True
        
        # If very short, likely not substantive
        if len(response_text) < 100 and promise_count > 0:
            return True
        
        return False
    
    def _pair_reviews_and_responses(self, paper: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Pair reviews with their specific author responses using replyto.
        
        Returns list of pairs: [{review: ..., author_response: ...}, ...]
        """
        reviews = paper.get('reviews', [])
        author_responses = paper.get('author_responses', [])
        
        if not reviews or not author_responses:
            return []
        
        pairs = []
        
        # Build a map of review_id -> review
        review_map = {r['review_id']: r for r in reviews}
        
        # For each author response, find the review it replies to
        for response in author_responses:
            replyto = response.get('replyto')
            
            if not replyto:
                continue
            
            # Check if this response replies to a review
            if replyto in review_map:
                review = review_map[replyto]
                response_text = response.get('response_text', '')
                
                # Filter out promise-only responses
                if self._is_promise_response(response_text):
                    continue
                
                # Valid pair found
                pairs.append({
                    'review': review,
                    'author_response': response,
                    'is_direct_reply': True  # Response directly replies to this review
                })
        
        # If no direct pairs found, try to match general responses to reviews
        if not pairs:
            # Look for responses that reply to the submission itself
            submission_id = paper.get('submission_id')
            general_responses = [
                r for r in author_responses 
                if r.get('replyto') == submission_id or r.get('replyto') is None
            ]
            
            for response in general_responses:
                response_text = response.get('response_text', '')
                
                if self._is_promise_response(response_text):
                    continue
                
                # For general responses, pair with the first review that has substantive content
                for review in reviews:
                    if review.get('review_text') and len(review['review_text']) > 100:
                        pairs.append({
                            'review': review,
                            'author_response': response,
                            'is_direct_reply': False  # General response, not direct reply
                        })
                        break
                
                if pairs:
                    break
        
        return pairs
    
    def select_papers_with_pairs(
        self, 
        papers: List[Dict[str, Any]], 
        num_papers: int = NUM_PAPERS_TO_SELECT
    ) -> List[Dict[str, Any]]:
        """
        Select papers that have valid review-response pairs.
        
        Selection criteria:
        - Must have at least one review
        - Must have at least one substantive author response
        - Balanced across venues if possible
        """
        valid_papers = []
        
        for paper in papers:
            pairs = self._pair_reviews_and_responses(paper)
            
            if pairs:
                # Add pairs to paper
                paper_with_pairs = {
                    **paper,
                    'review_response_pairs': pairs,
                    'num_valid_pairs': len(pairs)
                }
                valid_papers.append(paper_with_pairs)
        
        print(f"Found {len(valid_papers)} papers with valid review-response pairs")
        
        if len(valid_papers) == 0:
            return []
        
        # Balance across venues
        by_venue = {}
        for paper in valid_papers:
            venue = paper.get('venue', 'unknown')
            if venue not in by_venue:
                by_venue[venue] = []
            by_venue[venue].append(paper)
        
        # Shuffle each venue's papers
        for venue in by_venue:
            random.shuffle(by_venue[venue])
        
        # Select equal numbers from each venue
        selected = []
        venues = list(by_venue.keys())
        per_venue = max(1, num_papers // len(venues)) if venues else 0
        
        for venue in venues:
            take = min(per_venue, len(by_venue[venue]))
            selected.extend(by_venue[venue][:take])
        
        # If we need more, take from remaining
        if len(selected) < num_papers:
            remaining = []
            for venue in venues:
                remaining.extend(by_venue[venue][per_venue:])
            random.shuffle(remaining)
            needed = num_papers - len(selected)
            selected.extend(remaining[:needed])
        
        # Final shuffle
        random.shuffle(selected)
        selected = selected[:num_papers]
        
        print(f"Selected {len(selected)} papers (target: {num_papers})")
        
        # Print venue distribution
        venue_counts = {}
        for paper in selected:
            venue = paper.get('venue', 'unknown')
            venue_counts[venue] = venue_counts.get(venue, 0) + 1
        print(f"Venue distribution: {venue_counts}")
        
        return selected
    
    def create_final_pairs(self, selected_papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create the final list of review-response pairs.
        
        For each paper, selects ONE pair (the best one based on response quality).
        """
        final_pairs = []
        
        for paper in selected_papers:
            pairs = paper.get('review_response_pairs', [])
            
            if not pairs:
                continue
            
            # Select the best pair (prefer direct replies, then longest response)
            direct_pairs = [p for p in pairs if p.get('is_direct_reply', False)]
            
            if direct_pairs:
                # Among direct pairs, select one with longest response
                best_pair = max(
                    direct_pairs, 
                    key=lambda p: len(p['author_response'].get('response_text', ''))
                )
            else:
                # Fall back to any pair with longest response
                best_pair = max(
                    pairs, 
                    key=lambda p: len(p['author_response'].get('response_text', ''))
                )
            
            # Get ALL author responses for this paper (for combined response)
            all_author_responses = paper.get('author_responses', [])
            combined_response_texts = []
            for resp in all_author_responses:
                resp_text = resp.get('response_text', '').strip()
                if resp_text and len(resp_text) > 20:
                    combined_response_texts.append(resp_text)
            
            combined_author_response = '\n\n---\n\n'.join(combined_response_texts)
            
            # Create final pair record
            pair_record = {
                'paper_id': paper.get('submission_id'),
                'paper_title': paper.get('title'),
                'paper_venue': paper.get('venue'),
                'paper_decision': paper.get('decision'),
                
                # Review information (the specific review paired)
                'review_id': best_pair['review'].get('review_id'),
                'reviewer': best_pair['review'].get('reviewer'),
                'review_text': best_pair['review'].get('review_text'),
                'review_rating': best_pair['review'].get('rating'),
                'review_confidence': best_pair['review'].get('confidence'),
                
                # Specific author response (the one paired with this review)
                'response_id': best_pair['author_response'].get('response_id'),
                'response_text': best_pair['author_response'].get('response_text'),
                'is_direct_reply': best_pair.get('is_direct_reply', False),
                
                # Combined author response (ALL responses for this paper)
                'combined_author_response': combined_author_response,
                'num_author_responses': len(all_author_responses),
                
                # Metadata
                'pair_timestamp': datetime.now().isoformat(),
            }
            
            final_pairs.append(pair_record)
        
        print(f"Created {len(final_pairs)} final review-response pairs")
        
        # Statistics
        direct_count = sum(1 for p in final_pairs if p.get('is_direct_reply', False))
        print(f"  Direct replies: {direct_count}")
        print(f"  General responses: {len(final_pairs) - direct_count}")
        
        return final_pairs
    
    def save_pairs(self, pairs: List[Dict[str, Any]], selected_papers: List[Dict[str, Any]]):
        """Save the pairs and selected papers to files."""
        # Save pairs
        pairs_file = self.data_structure.get_paired_file()
        with open(pairs_file, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)
        print(f"Saved pairs to {as_str(pairs_file)}")
        
        # Save selected papers (for reference)
        papers_file = self.data_structure.get_selected_papers_file()
        # Slim down the paper data for storage
        slim_papers = []
        for paper in selected_papers:
            slim_papers.append({
                'submission_id': paper.get('submission_id'),
                'title': paper.get('title'),
                'venue': paper.get('venue'),
                'decision': paper.get('decision'),
                'pdf_local_path': paper.get('pdf_local_path'),
                'num_reviews': paper.get('num_reviews'),
                'num_author_responses': paper.get('num_author_responses'),
                'num_valid_pairs': paper.get('num_valid_pairs'),
            })
        
        with open(papers_file, 'w', encoding='utf-8') as f:
            json.dump(slim_papers, f, indent=2, ensure_ascii=False)
        print(f"Saved selected papers to {as_str(papers_file)}")
        
        # Save summary
        summary = {
            'pairing_timestamp': datetime.now().isoformat(),
            'random_seed': self.random_seed,
            'total_pairs': len(pairs),
            'direct_replies': sum(1 for p in pairs if p.get('is_direct_reply', False)),
            'general_responses': sum(1 for p in pairs if not p.get('is_direct_reply', False)),
            'venue_distribution': {},
        }
        
        for pair in pairs:
            venue = pair.get('paper_venue', 'unknown')
            summary['venue_distribution'][venue] = summary['venue_distribution'].get(venue, 0) + 1
        
        summary_file = self.data_structure.logs_dir / "pairing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved summary to {as_str(summary_file)}")
    
    def run(self, num_papers: int = NUM_PAPERS_TO_SELECT) -> List[Dict[str, Any]]:
        """
        Run the full pairing pipeline.
        
        Returns the list of review-response pairs.
        """
        print("="*60)
        print("REVIEW-RESPONSE PAIRING")
        print("="*60)
        
        # Load papers
        papers = self.load_papers()
        
        # Select papers with valid pairs
        selected_papers = self.select_papers_with_pairs(papers, num_papers)
        
        if not selected_papers:
            print("ERROR: No papers with valid review-response pairs found!")
            return []
        
        # Create final pairs (one per paper)
        pairs = self.create_final_pairs(selected_papers)
        
        # Save results
        self.save_pairs(pairs, selected_papers)
        
        print("="*60)
        print(f"PAIRING COMPLETE: {len(pairs)} pairs created")
        print("="*60)
        
        return pairs


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Pair reviews with author responses')
    parser.add_argument('--num-papers', type=int, default=NUM_PAPERS_TO_SELECT,
                       help=f'Number of papers to select (default: {NUM_PAPERS_TO_SELECT})')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    try:
        pairer = ReviewResponsePairer(random_seed=args.seed)
        pairs = pairer.run(num_papers=args.num_papers)
        
        print(f"\nPairing completed! Created {len(pairs)} pairs")
        
    except Exception as e:
        print(f"Error during pairing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

