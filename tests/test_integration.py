"""Integration tests for all extractors.

These tests use real files and services (no mocking).
All extractors are tested via create_extractor factory with parametrization.

Guidelines:
- Local OCR tests (EasyOCR, Tesseract, PaddleOCR) run unconditionally.
- Cloud tests (Azure, Google) require credentials in environment variables.
- Use 2-letter ISO 639-1 language codes (e.g., "en", "fr", "de") for all extractors.
"""

from pathlib import Path

import pytest

from xtra.extractors.factory import create_extractor
from xtra.models import ExtractorType

TEST_DATA_DIR = Path(__file__).parent / "data"

# All OCR extractors with their expected ocr_engine metadata value
OCR_EXTRACTORS = [
    pytest.param(ExtractorType.EASYOCR, "easyocr", id="easyocr"),
    pytest.param(ExtractorType.TESSERACT, "tesseract", id="tesseract"),
    pytest.param(ExtractorType.PADDLE, "paddleocr", id="paddle"),
    pytest.param(ExtractorType.AZURE_DI, "azure_document_intelligence", id="azure"),
    pytest.param(ExtractorType.GOOGLE_DOCAI, "google_document_ai", id="google"),
]

# Local OCR extractors only (for image tests - cloud APIs don't support raw images)
LOCAL_OCR_EXTRACTORS = [
    pytest.param(ExtractorType.EASYOCR, "easyocr", id="easyocr"),
    pytest.param(ExtractorType.TESSERACT, "tesseract", id="tesseract"),
    pytest.param(ExtractorType.PADDLE, "paddleocr", id="paddle"),
]


def test_pdf_extractor() -> None:
    """Test PDF extractor with real PDF file."""
    with create_extractor(TEST_DATA_DIR / "test_pdf_2p_text.pdf", ExtractorType.PDF) as extractor:
        doc = extractor.extract()

    assert doc.path == TEST_DATA_DIR / "test_pdf_2p_text.pdf"
    assert len(doc.pages) == 2
    assert doc.metadata is not None
    assert doc.metadata.source_type == ExtractorType.PDF

    # Verify page 1 content
    page1 = doc.pages[0]
    page1_texts = [t.text for t in page1.texts]
    assert "First page. First text" in page1_texts
    assert "First page. Second text" in page1_texts
    assert "First page. Fourth text" in page1_texts
    assert len(page1.texts) == 3

    # Verify page 2 content
    page2 = doc.pages[1]
    assert len(page2.texts) == 1
    assert page2.texts[0].text == "Second page. Third text"

    # Verify page structure
    for page in doc.pages:
        assert page.width > 0
        assert page.height > 0
        for text in page.texts:
            assert text.bbox is not None
            assert text.bbox.x0 < text.bbox.x1
            assert text.bbox.y0 < text.bbox.y1


@pytest.mark.parametrize("extractor_type,ocr_engine", LOCAL_OCR_EXTRACTORS)
def test_ocr_extract_image(extractor_type: ExtractorType, ocr_engine: str) -> None:
    """Test OCR extraction from image file."""
    with create_extractor(
        TEST_DATA_DIR / "test_image.png",
        extractor_type,
        languages=["en"],
    ) as extractor:
        doc = extractor.extract()

    assert doc.path == TEST_DATA_DIR / "test_image.png"
    assert len(doc.pages) == 1
    assert doc.metadata is not None
    assert doc.metadata.source_type == extractor_type
    assert doc.metadata.extra["ocr_engine"] == ocr_engine

    # Verify OCR detected text
    page = doc.pages[0]
    assert page.width > 0
    assert page.height > 0
    assert len(page.texts) > 0

    all_text = " ".join(t.text for t in page.texts).lower()
    assert len(all_text) > 0, "Expected OCR to extract some text from image"

    # Verify confidence scores
    for text in page.texts:
        assert text.confidence is not None
        assert 0.0 <= text.confidence <= 1.0


@pytest.mark.parametrize("extractor_type,ocr_engine", OCR_EXTRACTORS)
def test_ocr_extract_pdf(extractor_type: ExtractorType, ocr_engine: str) -> None:
    """Test OCR extraction from PDF file."""
    with create_extractor(
        TEST_DATA_DIR / "test_pdf_2p_text.pdf",
        extractor_type,
        languages=["en"],
        dpi=150,
    ) as extractor:
        doc = extractor.extract()

    assert doc.path == TEST_DATA_DIR / "test_pdf_2p_text.pdf"
    assert len(doc.pages) == 2
    assert doc.metadata is not None
    assert doc.metadata.source_type == extractor_type
    assert doc.metadata.extra["ocr_engine"] == ocr_engine

    # Verify pages have content
    for page in doc.pages:
        assert page.width > 0
        assert page.height > 0
        assert len(page.texts) > 0

    # Verify text was extracted
    page1_text = " ".join(t.text for t in doc.pages[0].texts).lower()
    assert len(page1_text) > 0, "Expected OCR to extract some text from page 1"

    page2_text = " ".join(t.text for t in doc.pages[1].texts).lower()
    assert len(page2_text) > 0, "Expected OCR to extract some text from page 2"

    # Verify confidence scores
    for page in doc.pages:
        for text in page.texts:
            assert text.confidence is not None
            assert 0.0 <= text.confidence <= 1.0
