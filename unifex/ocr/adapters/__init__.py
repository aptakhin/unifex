"""Adapters for converting OCR results to internal models."""

from unifex.ocr.adapters.azure_di import AzureDocumentIntelligenceAdapter
from unifex.ocr.adapters.easy_ocr import EasyOCRAdapter
from unifex.ocr.adapters.google_docai import GoogleDocumentAIAdapter
from unifex.ocr.adapters.paddle_ocr import PaddleOCRAdapter
from unifex.ocr.adapters.tesseract_ocr import TesseractAdapter

__all__ = [
    "AzureDocumentIntelligenceAdapter",
    "EasyOCRAdapter",
    "GoogleDocumentAIAdapter",
    "PaddleOCRAdapter",
    "TesseractAdapter",
]
