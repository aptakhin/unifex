"""Base classes, models, and utilities for unifex extractors."""

from unifex.base.base import (
    BaseExtractor,
    ExecutorType,
    ExtractionResult,
    PageExtractionResult,
)
from unifex.base.coordinates import CoordinateConverter
from unifex.base.geometry import polygon_to_bbox_and_rotation
from unifex.base.image_loader import ImageLoader
from unifex.base.models import (
    BBox,
    CoordinateInfo,
    CoordinateUnit,
    Document,
    ExtractorMetadata,
    ExtractorType,
    FontInfo,
    Page,
    SearchResult,
    Table,
    TableCell,
    TextBlock,
)

__all__ = [
    # Base extractor classes
    "BaseExtractor",
    "ExecutorType",
    "ExtractionResult",
    "PageExtractionResult",
    # Coordinate utilities
    "CoordinateConverter",
    # Geometry utilities
    "polygon_to_bbox_and_rotation",
    # Image utilities
    "ImageLoader",
    # Models
    "BBox",
    "CoordinateInfo",
    "CoordinateUnit",
    "Document",
    "ExtractorMetadata",
    "ExtractorType",
    "FontInfo",
    "Page",
    "SearchResult",
    "Table",
    "TableCell",
    "TextBlock",
]
