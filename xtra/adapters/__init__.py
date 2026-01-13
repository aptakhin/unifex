"""Adapters for converting external schemas to internal models."""

from xtra.adapters.azure_di import AzureDocumentIntelligenceAdapter
from xtra.adapters.paddle_ocr import PaddleOCRAdapter

__all__ = [
    "AzureDocumentIntelligenceAdapter",
    "PaddleOCRAdapter",
]
