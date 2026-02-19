
import json
import re
import time
import random
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


class GoldenBenchmarkGenerator:
    """
    Generates the B1 Golden Supported benchmark.
    
    Extracts factual claims from paper content that are definitively true
    (supported by the paper itself).
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
    
    def load_selected_papers(self) -> List[Dict[str, Any]]:
        """Load selected papers from pairing step."""
        papers_file = self.data_structure.get_selected_papers_file()
        
        if not papers_file.exists():
            raise FileNotFoundError(
                f"Selected papers file not found: {papers_file}\n"
                "Please run pairer.py first."
            )
        
        with open(papers_file, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        
        print(f"Loaded {len(papers)} selected papers")
        return papers
    
    def _convert_pdf_to_markdown(self, submission_id: str) -> Optional[str]:
        """Convert PDF to markdown if not already done."""
        pdf_path = self.data_structure.pdfs_dir / f"{submission_id}.pdf"
        md_path = self.data_structure.markdown_dir / f"{submission_id}.md"
        
        # Return existing markdown
        if md_path.exists():
            with open(md_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        # Convert PDF if available
        if pdf_path.exists() and chunking_module is not None:
            print(f"    Converting PDF to markdown: {submission_id}")
            try:
                if chunking_module.convert_pdf_to_markdown(str(pdf_path), str(md_path)):
                    with open(md_path, 'r', encoding='utf-8') as f:
                        return f.read()
            except Exception as e:
                print(f"    Error converting PDF: {e}")
        
        return None
    
    def _parse_markdown_sections(self, markdown_text: str) -> List[Dict[str, Any]]:
        """
        Parse markdown into sections and subsections with paragraphs.
        """
        sections = []
        lines = markdown_text.split('\n')
        
        current_section = None
        current_subsection = None
        current_content = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check for main section header (##)
            if line_stripped.startswith('##') and not line_stripped.startswith('###'):
                # Save previous section
                if current_content:
                    content_text = '\n'.join(current_content)
                    paragraphs = self._split_into_paragraphs(content_text)
                    if paragraphs:
                        sections.append({
                            'section_name': current_section or 'Unknown',
                            'subsection_name': current_subsection or '',
                            'paragraphs': paragraphs
                        })
                
                current_section = line_stripped.replace('##', '').strip()
                current_subsection = None
                current_content = []
                continue
            
            # Check for subsection header (###)
            if line_stripped.startswith('###'):
                if current_content:
                    content_text = '\n'.join(current_content)
                    paragraphs = self._split_into_paragraphs(content_text)
                    if paragraphs:
                        sections.append({
                            'section_name': current_section or 'Unknown',
                            'subsection_name': current_subsection or '',
                            'paragraphs': paragraphs
                        })
                
                current_subsection = line_stripped.replace('###', '').strip()
                current_content = []
                continue
            
            # Regular content
            if line_stripped and not line_stripped.startswith('#'):
                current_content.append(line_stripped)
        
        # Save last section
        if current_content:
            content_text = '\n'.join(current_content)
            paragraphs = self._split_into_paragraphs(content_text)
            if paragraphs:
                sections.append({
                    'section_name': current_section or 'Unknown',
                    'subsection_name': current_subsection or '',
                    'paragraphs': paragraphs
                })
        
        return sections
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into meaningful paragraphs."""
        paragraphs = []
        
        # Split on double newlines or significant breaks
        raw_paragraphs = re.split(r'\n\s*\n', text)
        
        for para in raw_paragraphs:
            para = para.strip()
            if para and len(para) > 50:
                # If too long, split on sentences
                if len(para) > 500:
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    current_para = []
                    for sent in sentences:
                        sent = sent.strip()
                        if sent:
                            current_para.append(sent)
                            if len(' '.join(current_para)) > 200:
                                paragraphs.append(' '.join(current_para))
                                current_para = []
                    if current_para:
                        paragraphs.append(' '.join(current_para))
                else:
                    paragraphs.append(para)
        
        return paragraphs
    
    def _extract_claims_from_paragraph(
        self, 
        paragraph: str, 
        num_claims: int = 2
    ) -> List[Dict[str, str]]:
        """
        Extract factual claims from a paragraph that are clearly supported.
        """
        if not paragraph or len(paragraph) < 100:
            return []
        
        prompt = f"""
You are an expert at extracting atomic, factual claims from academic text.

Extract {num_claims} atomic claim(s) from the following paragraph. Each claim must:
1. Be a SINGLE, self-contained fact clearly stated in the text
2. Be specific and verifiable (numbers, methods, results)
3. Be 15-50 words long
4. Be DEFINITELY TRUE according to the paragraph

IMPORTANT: 
- Rephrase the claim to be semantically equivalent but with DIFFERENT wording
- Do NOT copy text verbatim from the paragraph
- Focus on experimental results, methodology, or quantitative claims

Paragraph:
{paragraph}

Return ONLY a JSON array of claim strings:
["claim 1", "claim 2"]
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting verifiable factual claims. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
            )
            
            content = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks
            if content.startswith('```'):
                content = re.sub(r'```(?:json)?\s*', '', content)
                content = re.sub(r'```\s*$', '', content)
            
            claims = json.loads(content)
            
            if isinstance(claims, list):
                return [
                    {'claim': c.strip(), 'source_paragraph': paragraph}
                    for c in claims if c.strip() and len(c.strip()) > 15
                ]
            return []
            
        except Exception as e:
            print(f"    Error extracting claims: {e}")
            return []
    
    def extract_golden_claims(
        self, 
        papers: List[Dict[str, Any]], 
        target_claims: int = BENCHMARK_TARGETS['B1_golden_supported']
    ) -> List[Dict[str, Any]]:
        """
        Extract golden supported claims from paper content.
        """
        if self.client is None:
            self._setup_client()
        
        all_claims = []
        claims_per_paper = max(5, target_claims // len(papers) + 1)
        
        print(f"Extracting ~{claims_per_paper} claims per paper...")
        print(f"Target: {target_claims} claims")
        
        for paper_idx, paper in enumerate(papers):
            if len(all_claims) >= target_claims:
                break
            
            submission_id = paper.get('submission_id')
            title = paper.get('title', '')
            
            print(f"\n[{paper_idx+1}/{len(papers)}] Processing {submission_id}")
            print(f"  Title: {title[:60]}...")
            
            # Get markdown content
            markdown = self._convert_pdf_to_markdown(submission_id)
            
            if not markdown:
                print(f"  ⚠ No markdown available, skipping")
                continue
            
            # Parse into sections
            sections = self._parse_markdown_sections(markdown)
            
            if not sections:
                print(f"  ⚠ No sections found, skipping")
                continue
            
            print(f"  Found {len(sections)} sections")
            
            # Shuffle sections for diversity
            random.shuffle(sections)
            
            paper_claims = []
            
            for section_data in sections:
                if len(paper_claims) >= claims_per_paper:
                    break
                
                section_name = section_data['section_name']
                subsection_name = section_data['subsection_name']
                paragraphs = section_data['paragraphs']
                
                # Skip certain sections
                skip_sections = ['abstract', 'introduction', 'conclusion', 'related work', 'references']
                if any(skip in section_name.lower() for skip in skip_sections):
                    continue
                
                for para in paragraphs:
                    if len(paper_claims) >= claims_per_paper:
                        break
                    
                    # Extract 1-2 claims from this paragraph
                    remaining = min(2, claims_per_paper - len(paper_claims))
                    extracted = self._extract_claims_from_paragraph(para, remaining)
                    
                    for claim_data in extracted:
                        paper_decision = paper.get('decision')
                        decision_normalized = normalize_decision(paper_decision) if paper_decision else 'Unknown'
                        
                        claim_record = {
                            'claim': claim_data['claim'],
                            'label': 'Supported',  # Golden truth
                            'paragraph': claim_data['source_paragraph'],
                            'section_name': section_name,
                            'subsection_name': subsection_name,
                            'paper_id': submission_id,
                            'paper_title': title,
                            'paper_venue': paper.get('venue'),
                            'paper_decision': paper_decision,  # Original decision value
                            'decision': decision_normalized,  # Normalized decision category
                            'extraction_timestamp': datetime.now().isoformat(),
                            'model': self.model,
                        }
                        paper_claims.append(claim_record)
                    
                    time.sleep(0.3)  # Rate limiting
            
            all_claims.extend(paper_claims)
            print(f"  ✓ Extracted {len(paper_claims)} claims (total: {len(all_claims)})")
        
        # Limit to target
        all_claims = all_claims[:target_claims]
        
        print(f"\n{'='*40}")
        print(f"Total golden claims: {len(all_claims)}")
        
        return all_claims
    
    def save_benchmark(self, claims: List[Dict[str, Any]], split_by_decision: bool = True):
        """Save the B1 benchmark, optionally split by decision type."""
        from collections import defaultdict
        
        # Group by decision
        claims_by_decision = defaultdict(list)
        for claim in claims:
            decision = claim.get('decision', 'Unknown')
            claims_by_decision[decision].append(claim)
        
        # Save all claims
        benchmark_file = self.data_structure.get_benchmark_file("B1_golden_supported")
        with open(benchmark_file, 'w', encoding='utf-8') as f:
            for claim in claims:
                f.write(json.dumps(claim, ensure_ascii=False) + '\n')
        
        print(f"Saved B1 benchmark (all) to {as_str(benchmark_file)}: {len(claims)} records")
        
        # Save split by decision
        if split_by_decision:
            for decision, decision_claims in claims_by_decision.items():
                decision_file = self.data_structure.get_benchmark_file("B1_golden_supported", decision)
                with open(decision_file, 'w', encoding='utf-8') as f:
                    for claim in decision_claims:
                        f.write(json.dumps(claim, ensure_ascii=False) + '\n')
                print(f"  Saved B1 ({decision}) to {as_str(decision_file)}: {len(decision_claims)} records")
        
        # Save summary
        summary = {
            'benchmark': 'B1_golden_supported',
            'generation_timestamp': datetime.now().isoformat(),
            'model': self.model,
            'total_claims': len(claims),
            'label_distribution': {'Supported': len(claims)},
            'venue_distribution': {},
            'section_distribution': {},
            'decision_distribution': {},
        }
        
        for claim in claims:
            venue = claim.get('paper_venue', 'unknown')
            summary['venue_distribution'][venue] = summary['venue_distribution'].get(venue, 0) + 1
            
            decision = claim.get('decision', 'Unknown')
            summary['decision_distribution'][decision] = summary['decision_distribution'].get(decision, 0) + 1
            
            section = claim.get('section_name', 'unknown')
            summary['section_distribution'][section] = summary['section_distribution'].get(section, 0) + 1
        
        summary_file = self.data_structure.logs_dir / "B1_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved summary to {as_str(summary_file)}")
    
    def run(self, target_claims: int = BENCHMARK_TARGETS['B1_golden_supported']) -> List[Dict[str, Any]]:
        """
        Run the full B1 benchmark generation pipeline.
        """
        print("="*60)
        print("B1: GOLDEN SUPPORTED BENCHMARK")
        print("="*60)
        
        # Load selected papers
        papers = self.load_selected_papers()
        
        # Extract golden claims
        claims = self.extract_golden_claims(papers, target_claims)
        
        if not claims:
            print("ERROR: No claims extracted!")
            return []
        
        # Save benchmark
        self.save_benchmark(claims)
        
        print("="*60)
        print(f"B1 COMPLETE: {len(claims)} golden supported claims")
        print("="*60)
        
        return claims


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate B1 Golden Supported benchmark')
    parser.add_argument('--model', default=DEFAULT_MODEL,
                       help=f'OpenAI model to use (default: {DEFAULT_MODEL})')
    parser.add_argument('--target', type=int, default=BENCHMARK_TARGETS['B1_golden_supported'],
                       help=f'Target number of claims (default: {BENCHMARK_TARGETS["B1_golden_supported"]})')
    parser.add_argument('--max-workers', type=int, default=5,
                       help='Number of parallel workers (default: 5)')
    
    args = parser.parse_args()
    
    try:
        generator = GoldenBenchmarkGenerator(model=args.model, max_workers=args.max_workers)
        claims = generator.run(target_claims=args.target)
        
        print(f"\nB1 benchmark generated: {len(claims)} claims")
        
    except Exception as e:
        print(f"Error generating B1 benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

