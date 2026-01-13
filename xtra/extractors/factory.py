"""Factory for creating extractors by type."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

from ..models import SourceType
from .base import BaseExtractor


def _get_credential(key: str, credentials: Optional[Dict[str, str]]) -> Optional[str]:
    """Get credential from dict or environment variable."""
    if credentials and key in credentials:
        return credentials[key]
    return os.environ.get(key)


def create_extractor(
    path: Path,
    extractor_type: SourceType,
    *,
    languages: Optional[List[str]] = None,
    dpi: int = 200,
    use_gpu: bool = False,
    credentials: Optional[Dict[str, str]] = None,
) -> BaseExtractor:
    """Create an extractor by type with unified parameters.

    Args:
        path: Path to document/image file.
        extractor_type: SourceType enum value specifying which extractor to use:
            - SourceType.PDF - Native PDF extraction
            - SourceType.EASYOCR - EasyOCR for images and PDFs (auto-detects)
            - SourceType.TESSERACT - Tesseract for images and PDFs (auto-detects)
            - SourceType.PADDLE - PaddleOCR for images and PDFs (auto-detects)
            - SourceType.AZURE_DI - Azure Document Intelligence
            - SourceType.GOOGLE_DOCAI - Google Document AI
        languages: Language codes for OCR (default: ["en"]).
            EasyOCR/Tesseract use full list, PaddleOCR uses first language.
        dpi: DPI for PDF-to-image conversion (default: 200).
        use_gpu: Enable GPU acceleration where supported (default: False).
        credentials: Override credentials dict. If None, reads from env vars:
            - AZURE_DI_ENDPOINT, AZURE_DI_KEY for Azure
            - GOOGLE_DOCAI_PROCESSOR_NAME, GOOGLE_DOCAI_CREDENTIALS_PATH for Google

    Returns:
        Configured extractor instance.

    Raises:
        ValueError: If extractor_type is invalid or required credentials are missing.

    Example:
        >>> from xtra import create_extractor, SourceType
        >>> with create_extractor(Path("doc.pdf"), SourceType.PDF) as ext:
        ...     doc = ext.extract()
    """
    languages = languages or ["en"]

    if extractor_type == SourceType.PDF:
        from .pdf import PdfExtractor

        return PdfExtractor(path)

    elif extractor_type == SourceType.EASYOCR:
        from .ocr import EasyOcrExtractor

        return EasyOcrExtractor(path, languages=languages, gpu=use_gpu, dpi=dpi)

    elif extractor_type == SourceType.TESSERACT:
        from .tesseract_ocr import TesseractOcrExtractor

        return TesseractOcrExtractor(path, languages=languages, dpi=dpi)

    elif extractor_type == SourceType.PADDLE:
        from .paddle_ocr import PaddleOcrExtractor

        # PaddleOCR uses single language string
        lang = languages[0] if languages else "en"
        return PaddleOcrExtractor(path, lang=lang, use_gpu=use_gpu, dpi=dpi)

    elif extractor_type == SourceType.AZURE_DI:
        from .azure_di import AzureDocumentIntelligenceExtractor

        endpoint = _get_credential("AZURE_DI_ENDPOINT", credentials)
        key = _get_credential("AZURE_DI_KEY", credentials)

        if not endpoint or not key:
            raise ValueError(
                "Azure credentials required. Set AZURE_DI_ENDPOINT and AZURE_DI_KEY "
                "environment variables or pass credentials dict."
            )

        return AzureDocumentIntelligenceExtractor(path, endpoint=endpoint, key=key)

    elif extractor_type == SourceType.GOOGLE_DOCAI:
        from .google_docai import GoogleDocumentAIExtractor

        processor_name = _get_credential("GOOGLE_DOCAI_PROCESSOR_NAME", credentials)
        credentials_path = _get_credential("GOOGLE_DOCAI_CREDENTIALS_PATH", credentials)

        if not processor_name:
            raise ValueError(
                "Google Document AI processor name required. Set GOOGLE_DOCAI_PROCESSOR_NAME "
                "environment variable or pass credentials dict."
            )

        if not credentials_path:
            raise ValueError(
                "Google Document AI credentials path required. Set GOOGLE_DOCAI_CREDENTIALS_PATH "
                "environment variable or pass credentials dict."
            )

        return GoogleDocumentAIExtractor(
            path, processor_name=processor_name, credentials_path=credentials_path
        )

    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}")
