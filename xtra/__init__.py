from .adapters import AzureDocumentIntelligenceAdapter
from .extractors import (
    AzureDocumentIntelligenceExtractor,
    BaseExtractor,
    EasyOcrExtractor,
    ExtractionResult,
    OcrExtractor,
    PaddleOcrExtractor,
    PdfExtractor,
    TesseractOcrExtractor,
    create_extractor,
)
from .models import (
    BBox,
    Document,
    DocumentMetadata,
    FontInfo,
    Page,
    PdfObjectInfo,
    SourceType,
    TextBlock,
)

__all__ = [
    # Adapters
    "AzureDocumentIntelligenceAdapter",
    # Extractors
    "AzureDocumentIntelligenceExtractor",
    "BaseExtractor",
    "ExtractionResult",
    "PdfExtractor",
    "EasyOcrExtractor",
    "TesseractOcrExtractor",
    "PaddleOcrExtractor",
    "create_extractor",
    # Deprecated alias
    "OcrExtractor",
    # Models
    "BBox",
    "FontInfo",
    "TextBlock",
    "Page",
    "PdfObjectInfo",
    "DocumentMetadata",
    "Document",
    "SourceType",
]
