from .azure_di import AzureDocumentIntelligenceExtractor
from .base import BaseExtractor, ExtractionResult
from .factory import create_extractor
from .google_docai import GoogleDocumentAIExtractor
from .ocr import EasyOcrExtractor
from .paddle_ocr import PaddleOcrExtractor
from .pdf import PdfExtractor
from .tesseract_ocr import TesseractOcrExtractor

# Backward compatibility alias (deprecated)
OcrExtractor = EasyOcrExtractor

__all__ = [
    "AzureDocumentIntelligenceExtractor",
    "BaseExtractor",
    "ExtractionResult",
    "GoogleDocumentAIExtractor",
    "PdfExtractor",
    "EasyOcrExtractor",
    "TesseractOcrExtractor",
    "PaddleOcrExtractor",
    "create_extractor",
    # Deprecated alias
    "OcrExtractor",
]
