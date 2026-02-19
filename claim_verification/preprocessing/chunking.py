"""Markdown chunking with token-aware splitting and PDF conversion."""

import json
import re
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    from claim_verification.preprocessing.pdf_processing.parse_pdf import (
        parse_single_pdf_with_docling_standard,
    )
    from claim_verification.preprocessing.pdf_processing.clean_markdown import (
        remove_long_repeats,
        remove_neurips_checklist,
        clean_text_remove_long_repeats,
    )
    PDF_PARSING_AVAILABLE = True
except ImportError:
    PDF_PARSING_AVAILABLE = False

ENCODING_NAME = "cl100k_base"
CHUNK_TOKEN_SIZE = 256

if TIKTOKEN_AVAILABLE:
    try:
        _encoding = tiktoken.get_encoding(ENCODING_NAME)
    except Exception:
        TIKTOKEN_AVAILABLE = False


def num_tokens(text: str) -> int:
    if TIKTOKEN_AVAILABLE:
        return len(_encoding.encode(text))
    return len(text) // 4


def detect_section(text: str) -> str:
    match = re.match(r'^##\s+(.+?)(?:\n|$)', text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return 'unknown'


def chunk_document(text: str, max_tokens: int = CHUNK_TOKEN_SIZE) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for p in paras:
        p_tokens = num_tokens(p)
        if p_tokens > max_tokens:
            if current:
                chunks.append("\n\n".join(current))
                current, current_tokens = [], 0

            sents = re.split(r'(?<=[\.\?\!])\s+', p)
            buf: List[str] = []
            buf_tokens = 0
            for s in sents:
                s_tokens = num_tokens(s)
                if buf_tokens + s_tokens <= max_tokens:
                    buf.append(s)
                    buf_tokens += s_tokens
                else:
                    if buf:
                        chunks.append(" ".join(buf))
                    buf = [s]
                    buf_tokens = s_tokens
            if buf:
                chunks.append(" ".join(buf))
            continue

        if current_tokens + p_tokens <= max_tokens:
            current.append(p)
            current_tokens += p_tokens
        else:
            chunks.append("\n\n".join(current))
            current = [p]
            current_tokens = p_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def chunk_single_file(markdown_path: str, output_path: str, max_tokens: int = CHUNK_TOKEN_SIZE) -> bool:
    try:
        with open(markdown_path, 'r', encoding='utf-8') as f:
            text = f.read()

        chunks = chunk_document(text, max_tokens)

        current_section = "unknown"
        chunks_with_sections = []
        for i, chunk_text in enumerate(chunks):
            detected_section = detect_section(chunk_text)
            if detected_section != "unknown":
                current_section = detected_section
            chunks_with_sections.append({"idx": i + 1, "text": chunk_text, "section": current_section})

        record = {"file_id": Path(markdown_path).stem, "chunks": chunks_with_sections}

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return True
    except Exception as e:
        print(f"Error chunking {markdown_path}: {e}")
        return False


def chunk_directory(input_dir: str, output_dir: str, max_tokens: int = CHUNK_TOKEN_SIZE):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for md_path in input_path.glob("*.md"):
        stem = md_path.stem
        text = md_path.read_text(encoding="utf-8")
        chunks = chunk_document(text, max_tokens)

        record = {
            "file_id": stem,
            "chunks": [{"idx": i + 1, "text": chunks[i]} for i in range(len(chunks))],
        }

        out_path = output_path / f"{stem}_chunks.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_chunks_from_file(chunks_path: str) -> List[Dict[str, Any]]:
    try:
        with open(chunks_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if 'chunks' in data:
                        return data['chunks']
        return []
    except Exception as e:
        print(f"Error loading chunks from {chunks_path}: {e}")
        return []


def convert_pdf_to_markdown(pdf_path: str, output_path: str) -> bool:
    if not PDF_PARSING_AVAILABLE:
        print("PDF parsing not available")
        return False

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        parse_single_pdf_with_docling_standard(
            file_path=pdf_path,
            code_enrichment=False,
            formula_enrichment=False,
            output_path=output_path,
        )

        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()

        content = clean_text_remove_long_repeats(content, char_thresh=10, punct_seq_thresh=10)
        content = remove_neurips_checklist(content)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return True
    except Exception as e:
        print(f"Error converting PDF {pdf_path}: {e}")
        return False


def chunk_submission_if_needed(submission_id: str, base_data_path: str, groups: List[str] = None) -> List[Dict[str, Any]]:
    if groups is None:
        groups = ['oral', 'poster', 'spotlight', 'reject']

    for group in groups:
        chunks_path = os.path.join(base_data_path, group, "4_chunked", "docling-standard", f"{submission_id}_chunks.jsonl")
        if os.path.exists(chunks_path):
            return load_chunks_from_file(chunks_path)

    for group in groups:
        markdown_path = os.path.join(base_data_path, group, "3_cleaned", "docling-standard", f"{submission_id}.md")
        if os.path.exists(markdown_path):
            chunks_path = os.path.join(base_data_path, group, "4_chunked", "docling-standard", f"{submission_id}_chunks.jsonl")
            if chunk_single_file(markdown_path, chunks_path):
                return load_chunks_from_file(chunks_path)

    for group in groups:
        pdf_path = os.path.join(base_data_path, "pdfs", f"{submission_id}.pdf")
        if os.path.exists(pdf_path):
            parsed_dir = os.path.join(base_data_path, group, "2_parsed", "docling-standard")
            cleaned_dir = os.path.join(base_data_path, group, "3_cleaned", "docling-standard")
            chunked_dir = os.path.join(base_data_path, group, "4_chunked", "docling-standard")

            os.makedirs(parsed_dir, exist_ok=True)
            os.makedirs(cleaned_dir, exist_ok=True)
            os.makedirs(chunked_dir, exist_ok=True)

            parsed_md_path = os.path.join(parsed_dir, f"{submission_id}.md")
            cleaned_md_path = os.path.join(cleaned_dir, f"{submission_id}.md")

            if convert_pdf_to_markdown(pdf_path, parsed_md_path):
                shutil.copy2(parsed_md_path, cleaned_md_path)
                chunks_path = os.path.join(chunked_dir, f"{submission_id}_chunks.jsonl")
                if chunk_single_file(cleaned_md_path, chunks_path):
                    return load_chunks_from_file(chunks_path)

    return []
