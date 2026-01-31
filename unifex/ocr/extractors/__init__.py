"""OCR extractors."""

from unifex.ocr.extractors.azure_di import AzureDocumentIntelligenceExtractor
from unifex.ocr.extractors.easy_ocr import EasyOcrExtractor
from unifex.ocr.extractors.google_docai import GoogleDocumentAIExtractor
from unifex.ocr.extractors.paddle_ocr import PaddleOcrExtractor
from unifex.ocr.extractors.tesseract_ocr import TesseractOcrExtractor

__all__ = [
    "AzureDocumentIntelligenceExtractor",
    "EasyOcrExtractor",
    "GoogleDocumentAIExtractor",
    "PaddleOcrExtractor",
    "TesseractOcrExtractor",
]
