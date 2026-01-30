"""OCR and cloud document extractors."""

from xtra.extractors.ocr.azure_di import AzureDocumentIntelligenceExtractor
from xtra.extractors.ocr.easy_ocr import EasyOcrExtractor
from xtra.extractors.ocr.google_docai import GoogleDocumentAIExtractor
from xtra.extractors.ocr.paddle_ocr import PaddleOcrExtractor
from xtra.extractors.ocr.tesseract_ocr import TesseractOcrExtractor

__all__ = [
    "AzureDocumentIntelligenceExtractor",
    "EasyOcrExtractor",
    "GoogleDocumentAIExtractor",
    "PaddleOcrExtractor",
    "TesseractOcrExtractor",
]
