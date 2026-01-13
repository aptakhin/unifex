"""Shared image loading utilities for OCR extractors."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pypdfium2 as pdfium
from PIL import Image


def load_images_from_path(path: Path, dpi: int = 200) -> List[Image.Image]:
    """Load image(s) from a file path.

    Automatically detects whether the file is a PDF or image and handles
    accordingly. For PDFs, each page is converted to an image at the specified DPI.

    Args:
        path: Path to the image or PDF file.
        dpi: DPI for PDF-to-image conversion. Default 200.

    Returns:
        List of PIL Image objects (one per page for PDFs, single for images).
    """
    if path.suffix.lower() == ".pdf":
        return _load_pdf_as_images(path, dpi)
    else:
        return [Image.open(path)]


def _load_pdf_as_images(path: Path, dpi: int) -> List[Image.Image]:
    """Convert PDF pages to PIL Images at the specified DPI."""
    images: List[Image.Image] = []
    pdf = pdfium.PdfDocument(path)
    scale = dpi / 72.0

    for page in pdf:
        bitmap = page.render(scale=scale)
        images.append(bitmap.to_pil())

    pdf.close()
    return images


def is_pdf(path: Path) -> bool:
    """Check if a path refers to a PDF file."""
    return path.suffix.lower() == ".pdf"
