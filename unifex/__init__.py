from unifex.base import (
    BaseExtractor,
    BBox,
    CoordinateUnit,
    ExtractionResult,
    ExtractorMetadata,
    ExtractorType,
    FontInfo,
    Page,
    TextBlock,
)
from unifex.ocr.adapters import AzureDocumentIntelligenceAdapter
from unifex.pdf import PdfExtractor
from unifex.text_factory import create_extractor

__all__ = [
    # Adapters
    "AzureDocumentIntelligenceAdapter",
    # Core extractors (always available)
    "BaseExtractor",
    "ExtractionResult",
    "PdfExtractor",
    "create_extractor",
    # Models
    "BBox",
    "CoordinateUnit",
    "FontInfo",
    "TextBlock",
    "Page",
    "ExtractorMetadata",
    "ExtractorType",
    # Optional extractors (lazy loaded)
    "AzureDocumentIntelligenceExtractor",
    "GoogleDocumentAIExtractor",
    "EasyOcrExtractor",
    "TesseractOcrExtractor",
    "PaddleOcrExtractor",
]

# Lazy loading for optional dependencies
_LAZY_IMPORTS = {
    "AzureDocumentIntelligenceExtractor": "unifex.ocr.extractors.azure_di",
    "GoogleDocumentAIExtractor": "unifex.ocr.extractors.google_docai",
    "EasyOcrExtractor": "unifex.ocr.extractors.easy_ocr",
    "TesseractOcrExtractor": "unifex.ocr.extractors.tesseract_ocr",
    "PaddleOcrExtractor": "unifex.ocr.extractors.paddle_ocr",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
