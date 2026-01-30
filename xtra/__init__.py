from xtra.adapters import AzureDocumentIntelligenceAdapter
from xtra.extractors import (
    BaseExtractor,
    ExtractionResult,
    PdfExtractor,
    create_extractor,
)
from xtra.models import (
    BBox,
    CoordinateUnit,
    ExtractorMetadata,
    ExtractorType,
    FontInfo,
    Page,
    TextBlock,
)

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
    "AzureDocumentIntelligenceExtractor": "xtra.extractors.azure_di",
    "GoogleDocumentAIExtractor": "xtra.extractors.google_docai",
    "EasyOcrExtractor": "xtra.extractors.easy_ocr",
    "TesseractOcrExtractor": "xtra.extractors.tesseract_ocr",
    "PaddleOcrExtractor": "xtra.extractors.paddle_ocr",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
