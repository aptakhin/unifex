"""Benchmarks for model creation and manipulation."""

import pytest
from pathlib import Path

from xtra.models import (
    BBox,
    TextBlock,
    Page,
    Document,
    CoordinateInfo,
    CoordinateUnit,
    ExtractorMetadata,
    ExtractorType,
    FontInfo,
)


@pytest.mark.benchmark
def test_bbox_creation():
    """Benchmark BBox instantiation."""
    bbox = BBox(x0=10.0, y0=20.0, x1=100.0, y1=200.0)
    assert bbox.x0 == 10.0


@pytest.mark.benchmark
def test_textblock_creation():
    """Benchmark TextBlock creation with bbox and metadata."""
    bbox = BBox(x0=48.3, y0=57.8, x1=205.4, y1=74.6)
    font = FontInfo(name="Arial", size=12.0, weight=400)
    block = TextBlock(
        text="Sample document text",
        bbox=bbox,
        rotation=0.0,
        confidence=0.98,
        font_info=font,
    )
    assert block.text == "Sample document text"


@pytest.mark.benchmark
def test_page_with_multiple_textblocks():
    """Benchmark Page creation with multiple text blocks."""
    texts = []
    for i in range(10):
        bbox = BBox(x0=48.0 + i, y0=57.0 + i * 20, x1=200.0 + i, y1=74.0 + i * 20)
        texts.append(
            TextBlock(
                text=f"Text block {i}",
                bbox=bbox,
                confidence=0.95,
            )
        )
    
    coord_info = CoordinateInfo(unit=CoordinateUnit.POINTS)
    page = Page(
        page=0,
        width=595.0,
        height=842.0,
        texts=texts,
        coordinate_info=coord_info,
    )
    assert len(page.texts) == 10


@pytest.mark.benchmark
def test_document_creation():
    """Benchmark full Document creation with pages."""
    pages = []
    for page_num in range(3):
        texts = []
        for i in range(5):
            bbox = BBox(x0=48.0, y0=57.0 + i * 20, x1=200.0, y1=74.0 + i * 20)
            texts.append(
                TextBlock(text=f"Page {page_num} text {i}", bbox=bbox, confidence=0.95)
            )
        
        pages.append(
            Page(
                page=page_num,
                width=595.0,
                height=842.0,
                texts=texts,
                coordinate_info=CoordinateInfo(unit=CoordinateUnit.POINTS),
            )
        )
    
    metadata = ExtractorMetadata(
        extractor_type=ExtractorType.PDF,
        creator="Test Creator",
        title="Test Document",
    )
    
    doc = Document(
        path=Path("/tmp/test.pdf"),
        pages=pages,
        metadata=metadata,
    )
    assert len(doc.pages) == 3


@pytest.mark.benchmark
def test_model_dict_conversion():
    """Benchmark model serialization to dict."""
    bbox = BBox(x0=10.0, y0=20.0, x1=100.0, y1=200.0)
    text_block = TextBlock(text="Test", bbox=bbox, confidence=0.99)
    result = text_block.model_dump() if hasattr(text_block, 'model_dump') else text_block.dict()
    assert isinstance(result, dict)
