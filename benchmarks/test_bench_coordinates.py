"""Benchmarks for coordinate conversion operations."""

import pytest

from xtra.coordinates import CoordinateConverter
from xtra.models import BBox, CoordinateUnit, TextBlock, Page, CoordinateInfo


@pytest.mark.benchmark
def test_points_to_pixels_conversion():
    """Benchmark conversion from points to pixels."""
    converter = CoordinateConverter(
        source_unit=CoordinateUnit.POINTS,
        page_width=595.0,
        page_height=842.0,
        dpi=300.0,
    )
    result = converter.convert_value(100.0, CoordinateUnit.PIXELS, is_x=True)
    assert result > 0


@pytest.mark.benchmark
def test_bbox_conversion():
    """Benchmark BBox conversion between units."""
    converter = CoordinateConverter(
        source_unit=CoordinateUnit.POINTS,
        page_width=595.0,
        page_height=842.0,
        dpi=300.0,
    )
    bbox = BBox(x0=48.0, y0=57.0, x1=200.0, y1=74.0)
    converted = converter.convert_bbox(bbox, CoordinateUnit.PIXELS)
    assert converted.x0 != bbox.x0


@pytest.mark.benchmark
def test_textblock_conversion():
    """Benchmark TextBlock coordinate conversion."""
    converter = CoordinateConverter(
        source_unit=CoordinateUnit.POINTS,
        page_width=595.0,
        page_height=842.0,
        dpi=200.0,
    )
    bbox = BBox(x0=48.0, y0=57.0, x1=200.0, y1=74.0)
    block = TextBlock(text="Sample text", bbox=bbox, confidence=0.98)
    converted = converter.convert_text_block(block, CoordinateUnit.INCHES)
    assert converted.text == block.text


@pytest.mark.benchmark
def test_page_conversion_with_multiple_blocks():
    """Benchmark full Page conversion with multiple text blocks."""
    texts = []
    for i in range(20):
        bbox = BBox(x0=48.0 + i, y0=57.0 + i * 10, x1=200.0 + i, y1=74.0 + i * 10)
        texts.append(TextBlock(text=f"Block {i}", bbox=bbox, confidence=0.95))
    
    page = Page(
        page=0,
        width=595.0,
        height=842.0,
        texts=texts,
        coordinate_info=CoordinateInfo(unit=CoordinateUnit.POINTS),
    )
    
    converter = CoordinateConverter(
        source_unit=CoordinateUnit.POINTS,
        page_width=595.0,
        page_height=842.0,
        dpi=150.0,
    )
    
    converted_page = converter.convert_page(page, CoordinateUnit.PIXELS, target_dpi=150.0)
    assert len(converted_page.texts) == 20


@pytest.mark.benchmark
def test_normalized_to_points():
    """Benchmark conversion from normalized coordinates to points."""
    converter = CoordinateConverter(
        source_unit=CoordinateUnit.NORMALIZED,
        page_width=1.0,
        page_height=1.0,
    )
    result = converter.convert_value(0.5, CoordinateUnit.POINTS, is_x=True)
    assert isinstance(result, float)


@pytest.mark.benchmark
def test_multiple_coordinate_conversions():
    """Benchmark multiple sequential coordinate conversions."""
    converter = CoordinateConverter(
        source_unit=CoordinateUnit.POINTS,
        page_width=595.0,
        page_height=842.0,
        dpi=200.0,
    )
    
    bbox = BBox(x0=48.0, y0=57.0, x1=200.0, y1=74.0)
    
    # Chain multiple conversions
    pixels = converter.convert_bbox(bbox, CoordinateUnit.PIXELS)
    
    converter_pixels = CoordinateConverter(
        source_unit=CoordinateUnit.PIXELS,
        page_width=595.0 * (200.0 / 72.0),
        page_height=842.0 * (200.0 / 72.0),
        dpi=200.0,
    )
    inches = converter_pixels.convert_bbox(pixels, CoordinateUnit.INCHES)
    
    assert inches.x0 != bbox.x0
