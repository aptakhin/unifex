from pathlib import Path

from xtra.extractors import (
    BasicLineMerger,
    KeepCharacterMerger,
    PdfExtractor,
)


TEST_DATA_DIR = Path(__file__).parent / "data"


def test_basic_line_merger_is_default() -> None:
    """BasicLineMerger should be used by default (merges into lines)."""
    with PdfExtractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf") as extractor:
        doc = extractor.extract()
    # Default behavior groups characters into lines
    page1 = doc.pages[0]
    assert len(page1.texts) == 3  # 3 lines, not hundreds of characters


def test_basic_line_merger_groups_text_into_lines() -> None:
    """BasicLineMerger should group characters into text lines."""
    with PdfExtractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf") as extractor:
        doc = extractor.extract()
    page1 = doc.pages[0]
    # Should have 3 text blocks (lines)
    assert len(page1.texts) == 3
    texts = [t.text for t in page1.texts]
    assert "First page. First text" in texts


def test_keep_character_merger_preserves_individual_chars() -> None:
    """KeepCharacterMerger should create one TextBlock per character."""
    with PdfExtractor(
        TEST_DATA_DIR / "test_pdf_2p_text.pdf",
        character_merger=KeepCharacterMerger(),
    ) as extractor:
        doc = extractor.extract()
    page1 = doc.pages[0]
    # Should have many more text blocks (one per character)
    assert len(page1.texts) > 20  # More than the 3 lines
    # Each text block should be a single character
    for text_block in page1.texts:
        assert len(text_block.text) == 1


def test_keep_character_merger_includes_whitespace() -> None:
    """KeepCharacterMerger should include whitespace characters."""
    with PdfExtractor(
        TEST_DATA_DIR / "test_pdf_2p_text.pdf",
        character_merger=KeepCharacterMerger(),
    ) as extractor:
        doc = extractor.extract()
    page1 = doc.pages[0]
    # Should have some whitespace characters
    whitespace_chars = [t for t in page1.texts if t.text.isspace()]
    assert len(whitespace_chars) > 0


def test_keep_character_merger_has_font_info() -> None:
    """KeepCharacterMerger should preserve font info for each character."""
    with PdfExtractor(
        TEST_DATA_DIR / "test_pdf_2p_text.pdf",
        character_merger=KeepCharacterMerger(),
    ) as extractor:
        doc = extractor.extract()
    page1 = doc.pages[0]
    # At least some characters should have font info
    has_font = any(t.font_info is not None for t in page1.texts)
    assert has_font, "At least one character should have font info"


def test_keep_character_merger_has_bbox_for_each_char() -> None:
    """Each character should have its own bounding box."""
    with PdfExtractor(
        TEST_DATA_DIR / "test_pdf_2p_text.pdf",
        character_merger=KeepCharacterMerger(),
    ) as extractor:
        doc = extractor.extract()
    page1 = doc.pages[0]
    for text_block in page1.texts:
        assert text_block.bbox is not None
        # Bbox should be valid (non-negative dimensions)
        assert text_block.bbox.x1 >= text_block.bbox.x0
        assert text_block.bbox.y1 >= text_block.bbox.y0


def test_basic_line_merger_custom_threshold() -> None:
    """BasicLineMerger should accept custom line_gap_threshold."""
    merger = BasicLineMerger(line_gap_threshold=1.0)
    with PdfExtractor(
        TEST_DATA_DIR / "test_pdf_2p_text.pdf",
        character_merger=merger,
    ) as extractor:
        doc = extractor.extract()
    # With a smaller threshold, we might get more blocks
    page1 = doc.pages[0]
    assert len(page1.texts) >= 3


def test_character_merger_works_with_coordinate_units() -> None:
    """Character mergers should work with different coordinate units."""
    from xtra.models import CoordinateUnit

    with PdfExtractor(
        TEST_DATA_DIR / "test_pdf_2p_text.pdf",
        output_unit=CoordinateUnit.NORMALIZED,
        character_merger=KeepCharacterMerger(),
    ) as extractor:
        doc = extractor.extract()
    page1 = doc.pages[0]
    # All coordinates should be normalized (0-1)
    for text_block in page1.texts:
        assert 0 <= text_block.bbox.x0 <= 1
        assert 0 <= text_block.bbox.y0 <= 1
        assert 0 <= text_block.bbox.x1 <= 1
        assert 0 <= text_block.bbox.y1 <= 1
