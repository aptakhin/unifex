"""Base classes, models, and utilities for xtra extractors."""

from xtra.base.base import (
    BaseExtractor,
    ExecutorType,
    ExtractionResult,
    PageExtractionResult,
)
from xtra.base.coordinates import CoordinateConverter
from xtra.base.geometry import polygon_to_bbox_and_rotation
from xtra.base.image_loader import ImageLoader
from xtra.base.models import (
    BBox,
    CoordinateInfo,
    CoordinateUnit,
    Document,
    ExtractorMetadata,
    ExtractorType,
    FontInfo,
    Page,
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
    "Table",
    "TableCell",
    "TextBlock",
]
