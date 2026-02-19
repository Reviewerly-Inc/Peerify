

import importlib
import os
import pickle
import logging
import time
import json
import requests
from pathlib import Path
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime
import random

from claim_verification.config import (
    DataStructure, openreview_credentials, VENUES, as_str, ensure_dir, MAX_PAPERS_TO_CRAWL
)


class OpenReviewCrawler:
    """
    Crawler that fetches OpenReview submissions with full thread information,
    including the replyto field for proper response pairing.
    """
    
    def __init__(self, max_papers: Optional[int] = None):
        # Default to config value if not specified
        self.max_papers = max_papers if max_papers is not None else MAX_PAPERS_TO_CRAWL
        self.data_structure = DataStructure()
        self.client = None
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        log_file = self.data_structure.logs_dir / f"crawl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _import_external_openreview(self):
        """Import external OpenReview library."""
        proj_root = Path(__file__).resolve().parents[1]
        original_sys_path = list(sys.path)
        try:
            sys.path = [p for p in sys.path if p != str(proj_root)]
            return importlib.import_module('openreview')
        finally:
            sys.path = original_sys_path
    
    def _make_client(self):
        """Create OpenReview client."""
        creds = openreview_credentials()
        if not creds["username"] or not creds["password"]:
            raise RuntimeError(
                "OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD must be set in environment variables."
            )
        
        ext_openreview = self._import_external_openreview()
        return ext_openreview.api.OpenReviewClient(
            baseurl=creds["baseurl"],
            username=creds["username"],
            password=creds["password"],
        )
    
    def _get_submissions(self, venue_id: str, retries: int = 5) -> List[Any]:
        """Fetch submissions for a venue with directReplies."""
        for attempt in range(retries):
            try:
                self.logger.info(f"Fetching submissions for venue: {venue_id} (Attempt {attempt + 1}/{retries})")
                venue_group = self.client.get_group(venue_id)
                submission_name = venue_group.content['submission_name']['value']
                submissions = self.client.get_all_notes(
                    invitation=f'{venue_id}/-/{submission_name}',
                    details='directReplies'
                )
                return submissions
            except Exception as e:
                if e.__class__.__name__ == 'OpenReviewException':
                    retry_after = 30
                    self.logger.warning(f"Rate limit or API error. Retrying in {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue
                self.logger.error(f"Error fetching submissions for {venue_id}: {e}")
                raise
        
        self.logger.error(f"Failed to fetch submissions for {venue_id} after {retries} attempts.")
        return []
    
    def _get_full_thread(self, forum_id: str, retries: int = 3) -> List[Dict[str, Any]]:
        """
        Fetch ALL notes in a forum thread, including nested replies.
        This captures the replyto field for proper threading.
        """
        for attempt in range(retries):
            try:
                self.logger.debug(f"Fetching full thread for forum: {forum_id}")
                notes = self.client.get_all_notes(forum=forum_id)
                
                if notes is None:
                    return []
                
                # Convert notes to dicts with essential fields including replyto
                thread_notes = []
                for note in notes:
                    note_dict = {
                        'id': note.id,
                        'forum': getattr(note, 'forum', forum_id),
                        'replyto': getattr(note, 'replyto', None),  # KEY: This enables proper pairing
                        'invitation': getattr(note, 'invitation', ''),
                        'signatures': getattr(note, 'signatures', []),
                        'cdate': getattr(note, 'cdate', None),
                        'tcdate': getattr(note, 'tcdate', None),
                        'content': {}
                    }
                    
                    # Extract content
                    if hasattr(note, 'content') and note.content:
                        for key, value in note.content.items():
                            if isinstance(value, dict):
                                note_dict['content'][key] = value.get('value', value)
                            else:
                                note_dict['content'][key] = value
                    
                    thread_notes.append(note_dict)
                
                return thread_notes
                
            except Exception as e:
                if e.__class__.__name__ == 'OpenReviewException':
                    time.sleep(10)
                    continue
                self.logger.warning(f"Error fetching thread {forum_id}: {e}")
                return []
        
        return []
    
    def _get_value(self, content: Dict, field: str) -> Any:
        """Helper to extract value from OpenReview content."""
        if content is None:
            return None
        v = content.get(field)
        if isinstance(v, dict):
            return v.get('value', v)
        return v
    
    def _download_pdf(self, pdf_url: str, pdf_path: Path, retries: int = 3) -> bool:
        """Download a PDF file from URL."""
        for attempt in range(retries):
            try:
                self.logger.debug(f"Downloading PDF: {pdf_url}")
                response = requests.get(pdf_url, timeout=30)
                response.raise_for_status()
                
                with open(pdf_path, 'wb') as f:
                    f.write(response.content)
                return True
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Failed to download PDF (attempt {attempt + 1}): {e}")
                if attempt < retries - 1:
                    time.sleep(5)
        return False
    
    def _extract_paper_info(self, submission: Any, venue: str) -> Dict[str, Any]:
        """Extract paper information from a submission."""
        try:
            content = getattr(submission, 'content', {}) or {}
            details = getattr(submission, 'details', {}) or {}
            direct_replies = details.get('directReplies', [])
            
            # Extract decision
            decision = None
            decision_comment = None
            for reply in direct_replies:
                reply_content = reply.get('content', {})
                if 'decision' in reply_content:
                    dec = reply_content['decision']
                    decision = dec.get('value') if isinstance(dec, dict) else dec
                    comment = reply_content.get('comment', {})
                    decision_comment = comment.get('value') if isinstance(comment, dict) else comment
                    break
            
            # Build PDF URL
            pdf_url = None
            pdf_val = self._get_value(content, 'pdf')
            if pdf_val:
                if pdf_val.startswith('/') or not pdf_val.startswith('http'):
                    pdf_url = f"https://openreview.net/pdf?id={submission.id}"
                else:
                    pdf_url = pdf_val
            
            paper_info = {
                'submission_id': submission.id,
                'forum_id': getattr(submission, 'forum', submission.id),
                'submission_number': getattr(submission, 'number', None),
                'venue': venue,
                'title': self._get_value(content, 'title'),
                'abstract': self._get_value(content, 'abstract'),
                'authors': self._get_value(content, 'authors'),
                'keywords': self._get_value(content, 'keywords'),
                'pdf_url': pdf_url,
                'decision': decision,
                'decision_comment': decision_comment,
                'crawl_timestamp': datetime.now().isoformat(),
            }
            
            return paper_info
            
        except Exception as e:
            self.logger.error(f"Error extracting paper info for {submission.id}: {e}")
            return {
                'submission_id': submission.id,
                'forum_id': getattr(submission, 'forum', submission.id),
                'venue': venue,
                'error': str(e)
            }
    
    def _classify_note(self, note: Dict[str, Any]) -> str:
        """
        Classify a note as review, author_response, meta_review, decision, or other.
        """
        invitation = note.get('invitation', '').lower()
        signatures = note.get('signatures', [])
        
        # Check signatures for author
        is_author = any('Authors' in sig for sig in signatures)
        
        # Check invitation type
        if 'official_review' in invitation:
            return 'review'
        elif 'official_comment' in invitation:
            if is_author:
                return 'author_response'
            else:
                return 'comment'
        elif 'meta_review' in invitation:
            return 'meta_review'
        elif 'decision' in invitation:
            return 'decision'
        elif 'revision' in invitation or 'submission' in invitation:
            return 'submission'
        else:
            return 'other'
    
    def _process_thread(self, paper_info: Dict[str, Any], thread_notes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a thread to extract reviews, author responses, and their relationships.
        
        This follows the working approach from claim_extract/2_extract_structured_reviews.py:
        - Process ALL direct replies
        - Classify by writer field (not invitation type)
        - Extract text from content fields
        
        The key insight is using the `replyto` field:
        - Reviews have replyto = submission_id (the paper) or None
        - Author responses to specific reviews have replyto = review_id
        """
        submission_id = paper_info['submission_id']
        
        reviews = []
        author_responses = []
        
        for note in thread_notes:
            note_id = note.get('id')
            replyto = note.get('replyto')
            content = note.get('content', {})
            signatures = note.get('signatures', [])
            invitation = note.get('invitation', '').lower()
            
            # Extract writer from signatures (this is the key classification method)
            writer = None
            if signatures:
                sig = signatures[0]
                if isinstance(sig, str):
                    # Check for Authors signature
                    if 'Authors' in sig:
                        writer = 'Authors'
                    elif 'Reviewer_' in sig:
                        writer = sig.split('/')[-1]
                    elif 'Area_Chair' in sig or 'AC' in sig:
                        writer = 'Area_Chair'
                    else:
                        writer = sig.split('/')[-1]
            
            # Skip if we can't determine writer
            if not writer:
                continue
            
            # Process reviewer comments (including official reviews)
            if writer.startswith('Reviewer_'):
                # Extract review text (combine fields if structured)
                review_text = ""
                text_fields = ['summary', 'strengths', 'weaknesses', 'questions', 'review', 'comment']
                for field in text_fields:
                    field_val = content.get(field)
                    if field_val:
                        # Handle dict with 'value' key
                        if isinstance(field_val, dict):
                            text = field_val.get('value', '')
                        else:
                            text = str(field_val)
                        
                        if text and text.strip():
                            review_text += f"\n\n{field.capitalize()}: {text.strip()}"
                
                if not review_text:
                    # Try generic comment field
                    comment_val = content.get('comment', '')
                    if isinstance(comment_val, dict):
                        review_text = comment_val.get('value', '') or ''
                    else:
                        review_text = str(comment_val) if comment_val else ''
                
                review_text = review_text.strip()
                
                if review_text and len(review_text) > 50:
                    reviews.append({
                        'review_id': note_id,
                        'replyto': replyto,
                        'reviewer': writer,
                        'rating': self._get_value(content, 'rating'),
                        'confidence': self._get_value(content, 'confidence'),
                        'review_text': review_text,
                        'timestamp': note.get('cdate') or note.get('tcdate'),
                    })
            
            # Process author responses (writer == 'Authors')
            elif writer == 'Authors':
                # Extract response text from comment field
                comment_val = content.get('comment')
                if isinstance(comment_val, dict):
                    response_text = comment_val.get('value', '') or ''
                else:
                    response_text = str(comment_val) if comment_val else ''
                
                response_text = response_text.strip()
                
                if response_text and len(response_text) > 20:
                    author_responses.append({
                        'response_id': note_id,
                        'replyto': replyto,  # KEY: This tells us which review it replies to
                        'writer': writer,
                        'response_text': response_text,
                        'timestamp': note.get('cdate') or note.get('tcdate'),
                    })
        
        # Build the complete paper thread data
        paper_thread = {
            **paper_info,
            'reviews': reviews,
            'author_responses': author_responses,
            'num_reviews': len(reviews),
            'num_author_responses': len(author_responses),
        }
        
        return paper_thread
    
    def crawl_venue(self, venue_name: str, max_papers: Optional[int] = None) -> List[Dict[str, Any]]:
        """Crawl a single venue and return paper threads."""
        venue_config = VENUES.get(venue_name)
        if not venue_config:
            raise ValueError(f"Unknown venue: {venue_name}. Available: {list(VENUES.keys())}")
        
        venue_id = venue_config['venue_id']
        self.logger.info(f"Crawling venue: {venue_name} ({venue_id})")
        
        # Use provided max_papers or fall back to instance default
        venue_max = max_papers if max_papers is not None else self.max_papers
        
        # Initialize client
        if self.client is None:
            self.client = self._make_client()
        
        # Fetch submissions
        submissions = self._get_submissions(venue_id)
        if not submissions:
            self.logger.error(f"No submissions found for {venue_name}")
            return []
        
        self.logger.info(f"Found {len(submissions)} submissions for {venue_name}")
        
        # Apply venue-specific limit if specified
        if venue_max and venue_max > 0 and len(submissions) > venue_max:
            submissions = random.sample(submissions, venue_max)
            self.logger.info(f"Limited to {venue_max} papers for {venue_name}")
        
        # Process each submission
        paper_threads = []
        for i, submission in enumerate(submissions, 1):
            self.logger.info(f"Processing [{i}/{len(submissions)}] {submission.id}")
            
            # Extract basic paper info
            paper_info = self._extract_paper_info(submission, venue_name)
            
            # Get full thread with replyto information
            thread_notes = self._get_full_thread(submission.id)
            
            # Process thread to extract reviews and responses
            paper_thread = self._process_thread(paper_info, thread_notes)
            
            # Download PDF if available
            if paper_info.get('pdf_url'):
                pdf_path = self.data_structure.pdfs_dir / f"{submission.id}.pdf"
                if not pdf_path.exists():
                    if self._download_pdf(paper_info['pdf_url'], pdf_path):
                        paper_thread['pdf_local_path'] = str(pdf_path)
                    else:
                        paper_thread['pdf_local_path'] = None
                else:
                    paper_thread['pdf_local_path'] = str(pdf_path)
            
            paper_threads.append(paper_thread)
            
            # Brief pause to avoid rate limiting
            time.sleep(0.5)
        
        return paper_threads
    
    def crawl_all_venues(self) -> Dict[str, Any]:
        """Crawl all configured venues and save results."""
        self.logger.info("Starting crawl for all venues...")
        
        # Calculate papers per venue (distribute max_papers across venues)
        venue_names = list(VENUES.keys())
        num_venues = len(venue_names)
        papers_per_venue = self.max_papers // num_venues if num_venues > 0 else self.max_papers
        remaining_papers = self.max_papers % num_venues
        
        self.logger.info(f"Total crawl limit: {self.max_papers} papers")
        self.logger.info(f"Papers per venue: {papers_per_venue} (+ {remaining_papers} extra for first venue)")
        
        all_papers = []
        venue_stats = {}
        
        for venue_idx, venue_name in enumerate(venue_names):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Crawling {venue_name}")
            self.logger.info(f"{'='*60}")
            
            # Add extra papers to first venue if remainder exists
            venue_limit = papers_per_venue + (remaining_papers if venue_idx == 0 else 0)
            self.logger.info(f"Limit for {venue_name}: {venue_limit} papers")
            
            papers = self.crawl_venue(venue_name, max_papers=venue_limit)
            all_papers.extend(papers)
            
            # Calculate stats
            papers_with_reviews = len([p for p in papers if p.get('reviews')])
            papers_with_responses = len([p for p in papers if p.get('author_responses')])
            papers_with_both = len([p for p in papers if p.get('reviews') and p.get('author_responses')])
            
            venue_stats[venue_name] = {
                'total_papers': len(papers),
                'papers_with_reviews': papers_with_reviews,
                'papers_with_responses': papers_with_responses,
                'papers_with_both': papers_with_both,
            }
            
            # Save venue-specific data
            venue_file = self.data_structure.get_raw_threads_file(venue_name)
            with open(venue_file, 'w', encoding='utf-8') as f:
                json.dump(papers, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved {len(papers)} papers to {as_str(venue_file)}")
        
        # Save combined data
        combined_file = self.data_structure.raw_dir / "all_papers.json"
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(all_papers, f, indent=2, ensure_ascii=False)
        
        # Save summary
        summary = {
            'crawl_timestamp': datetime.now().isoformat(),
            'total_papers': len(all_papers),
            'venue_stats': venue_stats,
            'papers_with_reviews': len([p for p in all_papers if p.get('reviews')]),
            'papers_with_responses': len([p for p in all_papers if p.get('author_responses')]),
            'papers_with_both': len([p for p in all_papers if p.get('reviews') and p.get('author_responses')]),
        }
        
        summary_file = self.data_structure.logs_dir / "crawl_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("CRAWL COMPLETE")
        self.logger.info(f"Total papers: {len(all_papers)}")
        self.logger.info(f"Papers with reviews: {summary['papers_with_reviews']}")
        self.logger.info(f"Papers with author responses: {summary['papers_with_responses']}")
        self.logger.info(f"Papers with both: {summary['papers_with_both']}")
        self.logger.info(f"{'='*60}")
        
        return {
            'papers': all_papers,
            'summary': summary
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Crawl OpenReview with threading support')
    parser.add_argument('--venue', choices=list(VENUES.keys()) + ['all'], default='all',
                       help='Venue to crawl (default: all)')
    parser.add_argument('--max-papers', type=int, default=None,
                       help='Maximum total papers to crawl across all venues (default: 1000 from config)')
    
    args = parser.parse_args()
    
    try:
        crawler = OpenReviewCrawler(max_papers=args.max_papers)
        
        if args.venue == 'all':
            result = crawler.crawl_all_venues()
        else:
            papers = crawler.crawl_venue(args.venue)
            result = {'papers': papers, 'venue': args.venue}
        
        print(f"\nCrawl completed! Total papers: {len(result['papers'])}")
        
    except Exception as e:
        print(f"Error during crawl: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

