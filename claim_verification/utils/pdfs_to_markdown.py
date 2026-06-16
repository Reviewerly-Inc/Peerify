"""Convert all PDFs in data/pdfs to markdown via docling, mirroring the
clean step the original B3 pipeline used (clean_text_remove_long_repeats +
remove_neurips_checklist).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def _convert_one(pdf_path_str: str, md_path_str: str) -> tuple[str, bool, str]:
    pdf_path = Path(pdf_path_str)
    md_path = Path(md_path_str)
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat

        opts = PdfPipelineOptions()
        opts.do_code_enrichment = False
        opts.do_formula_enrichment = False
        conv = DocumentConverter(format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=opts)
        })
        result = conv.convert(str(pdf_path))
        md = result.document.export_to_markdown()

        try:
            from claim_verification.preprocessing.pdf_processing.clean_markdown import (
                clean_text_remove_long_repeats, remove_neurips_checklist,
            )
            md = clean_text_remove_long_repeats(md, char_thresh=10, punct_seq_thresh=10)
            md = remove_neurips_checklist(md)
        except ImportError:
            pass

        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md, encoding="utf-8")
        return (pdf_path.stem, True, "")
    except Exception as e:
        return (pdf_path.stem, False, f"{type(e).__name__}: {e}")


def main() -> None:
    repo = Path(__file__).resolve().parents[2]

    p = argparse.ArgumentParser()
    p.add_argument("--pdf-dir", type=Path, default=repo / "data" / "pdfs")
    p.add_argument("--md-dir", type=Path, default=repo / "data" / "markdown")
    p.add_argument("--workers", type=int, default=2,
                   help="Process-level concurrency. Docling holds models in memory; "
                        "2 is safe on a 16GB machine.")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    args.md_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(args.pdf_dir.glob("*.pdf"))
    todo = [pdf for pdf in pdfs if not (args.md_dir / f"{pdf.stem}.md").exists()]
    if args.limit:
        todo = todo[: args.limit]

    print(f"PDFs: {len(pdfs)} total, {len(pdfs) - len(todo)} already converted, "
          f"{len(todo)} to do")
    if not todo:
        return

    t0 = time.time()
    failed: list[tuple[str, str]] = []
    if args.workers <= 1:
        for i, pdf in enumerate(todo, 1):
            stem, ok, err = _convert_one(str(pdf), str(args.md_dir / f"{pdf.stem}.md"))
            elapsed = time.time() - t0
            tag = "OK" if ok else "FAIL"
            print(f"  [{i}/{len(todo)}] {tag} {stem} ({elapsed:.0f}s elapsed)")
            if not ok:
                failed.append((stem, err))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {
                pool.submit(_convert_one, str(pdf), str(args.md_dir / f"{pdf.stem}.md")): pdf
                for pdf in todo
            }
            done = 0
            for fut in as_completed(futs):
                stem, ok, err = fut.result()
                done += 1
                elapsed = time.time() - t0
                tag = "OK" if ok else "FAIL"
                print(f"  [{done}/{len(todo)}] {tag} {stem} ({elapsed:.0f}s elapsed)")
                if not ok:
                    failed.append((stem, err))
                sys.stdout.flush()

    print(f"\nDone in {time.time()-t0:.0f}s. {len(failed)} failures.")
    for stem, err in failed:
        print(f"  FAIL {stem}: {err}")


if __name__ == "__main__":
    main()
