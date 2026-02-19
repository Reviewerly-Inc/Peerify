from __future__ import annotations

import os
import sys
import time
import json
from typing import List
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import DocItemLabel, ImageRefMode
from docling_core.types.doc.document import DEFAULT_EXPORT_LABELS
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import InferenceFramework
from docling.datamodel import vlm_model_specs
from docling.pipeline.vlm_pipeline import VlmPipeline

import torch
from transformers import AutoProcessor, VisionEncoderDecoderModel
from pdf2image import convert_from_path
from tqdm import tqdm

sys.modules['flash_attn'] = None

__all__ = [
    "parse_single_pdf_with_docling_standard",
    "parse_single_pdf_with_docling_vllm",
    "parse_all_pdfs_in_directory_docling",
    "parse_single_pdf_with_nougat",
    "parse_all_pdfs_in_directory_nougat",
]


def parse_single_pdf_with_docling_standard(file_path, code_enrichment, formula_enrichment, output_path):
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_code_enrichment = code_enrichment
    pipeline_options.do_formula_enrichment = formula_enrichment

    converter = DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    })
    result = converter.convert(file_path)
    with open(output_path, 'w') as markdown_file:
        markdown_file.write(result.document.export_to_markdown())


def parse_single_pdf_with_docling_vllm(file_path, output_path, vlm_model_name: str, save_mode='markdown'):
    file_path = Path(file_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        vlm_model = getattr(vlm_model_specs, vlm_model_name)
    except AttributeError:
        available = [name for name in dir(vlm_model_specs) if not name.startswith("__")]
        raise ValueError(f"Model '{vlm_model_name}' not found. Available models: {available}")

    pipeline_options = VlmPipelineOptions()
    pipeline_options.generate_page_images = True
    pipeline_options.vlm_options = vlm_model

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_cls=VlmPipeline, pipeline_options=pipeline_options),
            InputFormat.IMAGE: PdfFormatOption(pipeline_cls=VlmPipeline, pipeline_options=pipeline_options),
        }
    )

    model_id = pipeline_options.vlm_options.repo_id.replace("/", "_")
    framework = pipeline_options.vlm_options.inference_framework

    start_time = time.time()
    res = converter.convert(file_path)
    inference_time = sum(page.predictions.vlm_response.generation_time for page in res.pages)
    total_time = time.time() - start_time

    base_name = f"{file_path.stem}-{model_id}-{framework}"

    with (output_path / f"{base_name}.json").open("w") as fp:
        json.dump(res.document.export_to_dict(), fp)

    if save_mode == 'json':
        res.document.save_as_json(output_path / f"{base_name}.json", image_mode=ImageRefMode.PLACEHOLDER)
    elif save_mode == 'markdown':
        res.document.save_as_markdown(output_path / f"{base_name}.md", image_mode=ImageRefMode.PLACEHOLDER)
    elif save_mode == 'html':
        res.document.save_as_html(
            output_path / f"{base_name}.html",
            image_mode=ImageRefMode.EMBEDDED,
            labels=[*DEFAULT_EXPORT_LABELS, DocItemLabel.FOOTNOTE],
            split_page_view=True,
        )

    num_pages = res.document.num_pages()
    return {
        "source": str(file_path),
        "model_id": model_id,
        "framework": str(framework),
        "num_pages": num_pages,
        "inference_time": inference_time,
        "total_time": total_time,
    }


def parse_all_pdfs_in_directory_docling(directory, output_directory, method, **kwargs):
    directory = Path(directory)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    pdf_files = [f for f in directory.iterdir() if f.suffix.lower() == '.pdf']

    for file_path in pdf_files:
        output_path = output_directory / f"{file_path.stem}.md"
        if method == 'docling_standard':
            parse_single_pdf_with_docling_standard(
                file_path=file_path,
                code_enrichment=kwargs.get('code_enrichment', False),
                formula_enrichment=kwargs.get('formula_enrichment', False),
                output_path=output_path,
            )
        elif method == 'docling_vllm':
            parse_single_pdf_with_docling_vllm(
                file_path=file_path,
                output_path=output_directory,
                vlm_model_name=kwargs.get('vlm_model_name', ''),
                save_mode=kwargs.get('save_mode', 'markdown'),
            )
        else:
            raise ValueError("Invalid method. Use 'docling_standard' or 'docling_vllm'.")


def _resolve_nougat_tag(model_tag: str) -> str:
    mapping = {
        "0.1.0-small": "facebook/nougat-small",
        "0.1.0-base": "facebook/nougat-base",
        "nougat-small": "facebook/nougat-small",
        "nougat-base": "facebook/nougat-base",
    }
    if model_tag not in mapping:
        raise ValueError(f"Unsupported model '{model_tag}'. Choose one of: {list(mapping)}")
    return mapping[model_tag]


def _load_nougat(model_tag: str):
    name = _resolve_nougat_tag(model_tag)
    processor = AutoProcessor.from_pretrained(name)
    model = VisionEncoderDecoderModel.from_pretrained(name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return processor, model, device


def _markdown_from_images(images: List, processor, model, device, batch_size: int = 1) -> str:
    markdown_parts: List[str] = []
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i : i + batch_size]
        pix = processor(images=batch_imgs, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            outputs = model.generate(
                pix,
                min_length=1,
                max_new_tokens=4096,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
            )
        batch_text = processor.batch_decode(outputs, skip_special_tokens=True)
        batch_text = [processor.post_process_generation(t, fix_markdown=True) for t in batch_text]
        markdown_parts.extend(batch_text)
    markdown_clean = "\n".join([line for line in markdown_parts if not line.strip().startswith("![")])
    return markdown_clean


def parse_single_pdf_with_nougat(file_path, output_path, model="0.1.0-small"):
    file_path = Path(file_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processor, model_obj, device = _load_nougat(model)
    images = convert_from_path(str(file_path), dpi=300)
    markdown = _markdown_from_images(images, processor, model_obj, device)
    output_path.write_text(markdown, encoding="utf-8")


def parse_all_pdfs_in_directory_nougat(directory, output_directory, model="0.1.0-small", batch_size=2):
    directory = Path(directory)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    processor, model_obj, device = _load_nougat(model)

    pdf_files = sorted(p for p in directory.iterdir() if p.suffix.lower() == ".pdf")
    for pdf in tqdm(pdf_files, desc="Nougat PDFs"):
        images = convert_from_path(str(pdf), dpi=300)
        md = _markdown_from_images(images, processor, model_obj, device, batch_size=batch_size)
        (output_directory / f"{pdf.stem}.md").write_text(md, encoding="utf-8")
