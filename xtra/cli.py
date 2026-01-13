#!/usr/bin/env python
"""CLI script to extract text from PDF and image files."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from .extractors import (
    AzureDocumentIntelligenceExtractor,
    EasyOcrExtractor,
    GoogleDocumentAIExtractor,
    PaddleOcrExtractor,
    PdfExtractor,
    TesseractOcrExtractor,
)
from .models import ExtractorType


def _create_extractor(args: argparse.Namespace, languages: list[str]) -> Any:
    """Create extractor based on CLI arguments."""
    extractor_type = ExtractorType(args.extractor)

    if extractor_type == ExtractorType.PDF:
        return PdfExtractor(args.input)
    if extractor_type == ExtractorType.EASYOCR:
        return EasyOcrExtractor(args.input, languages=languages)
    if extractor_type == ExtractorType.TESSERACT:
        return TesseractOcrExtractor(args.input, languages=languages)
    if extractor_type == ExtractorType.PADDLE:
        lang = languages[0] if languages else "en"
        return PaddleOcrExtractor(args.input, lang=lang)
    if extractor_type == ExtractorType.AZURE_DI:
        return _create_azure_extractor(args)
    if extractor_type == ExtractorType.GOOGLE_DOCAI:
        return _create_google_extractor(args)

    print(f"Error: Unknown extractor type: {args.extractor}", file=sys.stderr)
    sys.exit(1)


def _create_azure_extractor(args: argparse.Namespace) -> AzureDocumentIntelligenceExtractor:
    """Create Azure Document Intelligence extractor."""
    endpoint = args.azure_endpoint or os.environ.get("XTRA_AZURE_DI_ENDPOINT")
    key = args.azure_key or os.environ.get("XTRA_AZURE_DI_KEY")
    model = args.azure_model or os.environ.get("XTRA_AZURE_DI_MODEL", "prebuilt-read")
    if not endpoint or not key:
        print(
            "Error: Azure credentials required. Use --azure-endpoint/--azure-key "
            "or set XTRA_AZURE_DI_ENDPOINT/XTRA_AZURE_DI_KEY environment variables.",
            file=sys.stderr,
        )
        sys.exit(1)
    assert endpoint is not None and key is not None and model is not None
    return AzureDocumentIntelligenceExtractor(
        args.input, endpoint=endpoint, key=key, model_id=model
    )


def _create_google_extractor(args: argparse.Namespace) -> GoogleDocumentAIExtractor:
    """Create Google Document AI extractor."""
    processor = args.google_processor_name or os.environ.get("XTRA_GOOGLE_DOCAI_PROCESSOR_NAME")
    creds = args.google_credentials_path or os.environ.get("XTRA_GOOGLE_DOCAI_CREDENTIALS_PATH")
    if not processor or not creds:
        print(
            "Error: Google credentials required. Use --google-processor-name/"
            "--google-credentials-path or set XTRA_GOOGLE_DOCAI_PROCESSOR_NAME/"
            "XTRA_GOOGLE_DOCAI_CREDENTIALS_PATH environment variables.",
            file=sys.stderr,
        )
        sys.exit(1)
    assert processor is not None and creds is not None
    return GoogleDocumentAIExtractor(args.input, processor_name=processor, credentials_path=creds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract text from PDF/image files")
    parser.add_argument("input", type=Path, help="Input file path")
    parser.add_argument(
        "--extractor",
        type=str,
        choices=[e.value for e in ExtractorType],
        required=True,
        help="Extractor type: pdf, easyocr, tesseract, paddle, azure-di, google-docai",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="OCR languages, comma-separated (default: en)",
    )
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Page numbers to extract, comma-separated (default: all). Example: 0,1,2",
    )
    parser.add_argument(
        "--azure-endpoint",
        type=str,
        default=None,
        help="Azure endpoint URL (or XTRA_AZURE_DI_ENDPOINT env var)",
    )
    parser.add_argument(
        "--azure-key",
        type=str,
        default=None,
        help="Azure API key (or XTRA_AZURE_DI_KEY env var)",
    )
    parser.add_argument(
        "--azure-model",
        type=str,
        default=None,
        help="Azure model ID (or XTRA_AZURE_DI_MODEL env var, default: prebuilt-read)",
    )
    parser.add_argument(
        "--google-processor-name",
        type=str,
        default=None,
        help="Google processor name (or XTRA_GOOGLE_DOCAI_PROCESSOR_NAME env var)",
    )
    parser.add_argument(
        "--google-credentials-path",
        type=str,
        default=None,
        help="Google service account JSON path (or XTRA_GOOGLE_DOCAI_CREDENTIALS_PATH)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    languages = [lang.strip() for lang in args.lang.split(",")]
    pages: Optional[Sequence[int]] = None
    if args.pages:
        pages = [int(p.strip()) for p in args.pages.split(",")]

    extractor = _create_extractor(args, languages)

    with extractor:
        doc = extractor.extract(pages=pages)

    if args.json:
        # pydantic v2 uses model_dump_json, v1 uses json
        if hasattr(doc, "model_dump_json"):
            print(doc.model_dump_json(indent=2))
        else:
            print(doc.json(indent=2))
    else:
        for page in doc.pages:
            print(f"=== Page {page.page + 1} ===")
            for text in page.texts:
                bbox = text.bbox
                conf = f" ({text.confidence:.2f})" if text.confidence else ""
                print(
                    f"[{bbox.x0:.1f},{bbox.y0:.1f},{bbox.x1:.1f},{bbox.y1:.1f}]{conf} {text.text}"
                )
            print()


if __name__ == "__main__":
    main()
