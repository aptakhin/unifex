from xtra.extractors.azure_di import AzureDocumentIntelligenceExtractor
from xtra.extractors.base import BaseExtractor, ExtractionResult
from xtra.extractors.character_mergers import (
    BasicLineMerger,
    CharacterMerger,
    CharInfo,
    KeepCharacterMerger,
)
from xtra.extractors.factory import create_extractor
from xtra.extractors.google_docai import GoogleDocumentAIExtractor
from xtra.extractors.easy_ocr import EasyOcrExtractor
from xtra.extractors.paddle_ocr import PaddleOcrExtractor
from xtra.extractors.pdf import PdfExtractor
from xtra.extractors.tesseract_ocr import TesseractOcrExtractor

__all__ = [
    "AzureDocumentIntelligenceExtractor",
    "BaseExtractor",
    "BasicLineMerger",
    "CharacterMerger",
    "CharInfo",
    "ExtractionResult",
    "GoogleDocumentAIExtractor",
    "KeepCharacterMerger",
    "PdfExtractor",
    "EasyOcrExtractor",
    "TesseractOcrExtractor",
    "PaddleOcrExtractor",
    "create_extractor",
]
