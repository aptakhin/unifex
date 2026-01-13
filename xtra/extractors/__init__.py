from .azure_di import AzureDocumentIntelligenceExtractor
from .base import BaseExtractor, ExtractionResult
from .factory import create_extractor
from .ocr import EasyOcrExtractor, PdfToImageEasyOcrExtractor
from .pdf import PdfExtractor

# Backward compatibility aliases (deprecated)
OcrExtractor = EasyOcrExtractor
PdfToImageOcrExtractor = PdfToImageEasyOcrExtractor

__all__ = [
    "AzureDocumentIntelligenceExtractor",
    "BaseExtractor",
    "ExtractionResult",
    "PdfExtractor",
    "EasyOcrExtractor",
    "PdfToImageEasyOcrExtractor",
    "create_extractor",
    # Deprecated aliases
    "OcrExtractor",
    "PdfToImageOcrExtractor",
]
