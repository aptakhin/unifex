"""Unit tests for PaddleOCR adapter and Pydantic models."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from xtra.adapters.paddle_ocr import (
    PaddleOCRAdapter,
    PaddleOCRDetection,
    PaddleOCRResult,
)
from xtra.models import ExtractorType


class TestPaddleOCRDetection:
    """Tests for PaddleOCRDetection Pydantic model."""

    def test_valid_detection(self) -> None:
        detection = PaddleOCRDetection(
            polygon=[[10, 20], [90, 20], [90, 50], [10, 50]],
            text="Hello",
            confidence=0.95,
        )
        assert detection.text == "Hello"
        assert detection.confidence == 0.95
        assert len(detection.polygon) == 4

    def test_from_paddle_format(self) -> None:
        item = ([[10, 20], [90, 20], [90, 50], [10, 50]], ("Hello", 0.95))
        detection = PaddleOCRDetection.from_paddle_format(item)
        assert detection.text == "Hello"
        assert detection.confidence == 0.95

    def test_invalid_polygon_wrong_point_count(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            PaddleOCRDetection(
                polygon=[[10, 20], [90, 20], [90, 50]],  # Only 3 points
                text="Hello",
                confidence=0.95,
            )
        assert "points" in str(exc_info.value)

    def test_invalid_polygon_wrong_coordinate_count(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            PaddleOCRDetection(
                polygon=[[10], [90, 20], [90, 50], [10, 50]],  # First point has 1 coord
                text="Hello",
                confidence=0.95,
            )
        assert "coordinates" in str(exc_info.value)


class TestPaddleOCRResult:
    """Tests for PaddleOCRResult Pydantic model."""

    def test_from_paddle_output_success(self) -> None:
        paddle_output = [
            [
                [[[10, 20], [90, 20], [90, 50], [10, 50]], ("Hello", 0.955)],
                [[[100, 20], [190, 20], [190, 50], [100, 50]], ("World", 0.872)],
            ]
        ]
        result = PaddleOCRResult.from_paddle_output(paddle_output)
        assert len(result.detections) == 2
        assert result.detections[0].text == "Hello"
        assert result.detections[1].text == "World"

    def test_from_paddle_output_none(self) -> None:
        result = PaddleOCRResult.from_paddle_output(None)
        assert len(result.detections) == 0

    def test_from_paddle_output_empty_outer(self) -> None:
        result = PaddleOCRResult.from_paddle_output([])
        assert len(result.detections) == 0

    def test_from_paddle_output_empty_inner(self) -> None:
        result = PaddleOCRResult.from_paddle_output([[]])
        assert len(result.detections) == 0

    def test_from_paddle_output_none_items(self) -> None:
        result = PaddleOCRResult.from_paddle_output([[None]])
        assert len(result.detections) == 0

    def test_from_paddle_output_mixed_none_items(self) -> None:
        paddle_output = [
            [
                None,
                [[[10, 20], [90, 20], [90, 50], [10, 50]], ("Hello", 0.95)],
                None,
            ]
        ]
        result = PaddleOCRResult.from_paddle_output(paddle_output)
        assert len(result.detections) == 1
        assert result.detections[0].text == "Hello"


class TestPaddleOCRAdapter:
    """Tests for PaddleOCRAdapter conversion logic."""

    def test_convert_result_success(self) -> None:
        adapter = PaddleOCRAdapter()
        paddle_output = [
            [
                [[[10, 20], [90, 20], [90, 50], [10, 50]], ("Hello", 0.955)],
                [[[100, 20], [190, 20], [190, 50], [100, 50]], ("World", 0.872)],
            ]
        ]

        blocks = adapter.convert_result(paddle_output)

        assert len(blocks) == 2
        assert blocks[0].text == "Hello"
        assert blocks[0].confidence == pytest.approx(0.955, rel=0.01)
        assert blocks[0].bbox.x0 == pytest.approx(10.0, rel=0.01)
        assert blocks[0].bbox.y0 == pytest.approx(20.0, rel=0.01)
        assert blocks[0].bbox.x1 == pytest.approx(90.0, rel=0.01)
        assert blocks[0].bbox.y1 == pytest.approx(50.0, rel=0.01)

        assert blocks[1].text == "World"
        assert blocks[1].confidence == pytest.approx(0.872, rel=0.01)

    def test_convert_result_empty(self) -> None:
        adapter = PaddleOCRAdapter()
        assert adapter.convert_result(None) == []
        assert adapter.convert_result([]) == []
        assert adapter.convert_result([[]]) == []
        assert adapter.convert_result([[None]]) == []

    def test_convert_result_filters_empty_text(self) -> None:
        adapter = PaddleOCRAdapter()
        paddle_output = [
            [
                [[[10, 20], [90, 20], [90, 50], [10, 50]], ("", 0.95)],
                [[[100, 20], [190, 20], [190, 50], [100, 50]], ("  ", 0.95)],
                [[[200, 20], [290, 20], [290, 50], [200, 50]], ("Valid", 0.95)],
            ]
        ]

        blocks = adapter.convert_result(paddle_output)

        assert len(blocks) == 1
        assert blocks[0].text == "Valid"

    def test_convert_result_with_rotation(self) -> None:
        adapter = PaddleOCRAdapter()
        # Rotated text (not axis-aligned)
        paddle_output = [
            [
                [[[10, 30], [90, 20], [95, 50], [15, 60]], ("Rotated", 0.9)],
            ]
        ]

        blocks = adapter.convert_result(paddle_output)

        assert len(blocks) == 1
        assert blocks[0].text == "Rotated"
        # Rotation should be detected (non-zero for rotated text)
        assert blocks[0].rotation is not None


class TestPaddleOcrExtractorWithPdf:
    """Unit tests for PaddleOcrExtractor with PDF files."""

    def test_get_metadata_with_pdf(self) -> None:
        with (
            patch("xtra.extractors.paddle_ocr.PaddleOCR"),
            patch("xtra.extractors._image_loader.pdfium") as mock_pdfium,
        ):
            mock_pdf = MagicMock()
            mock_page = MagicMock()
            mock_bitmap = MagicMock()
            mock_pil_img = MagicMock()
            mock_pil_img.size = (800, 600)

            mock_bitmap.to_pil.return_value = mock_pil_img
            mock_page.render.return_value = mock_bitmap
            mock_pdf.__iter__ = lambda self: iter([mock_page])
            mock_pdfium.PdfDocument.return_value = mock_pdf

            from xtra.extractors.paddle_ocr import PaddleOcrExtractor

            extractor = PaddleOcrExtractor(Path("/fake/document.pdf"))
            metadata = extractor.get_metadata()

            assert metadata.source_type == ExtractorType.PADDLE
            assert metadata.extra["ocr_engine"] == "paddleocr"
            assert metadata.extra["dpi"] == 200

    def test_get_page_count_pdf(self) -> None:
        with (
            patch("xtra.extractors.paddle_ocr.PaddleOCR"),
            patch("xtra.extractors._image_loader.pdfium") as mock_pdfium,
        ):
            mock_pdf = MagicMock()
            mock_pages = [MagicMock(), MagicMock(), MagicMock()]
            for page in mock_pages:
                mock_bitmap = MagicMock()
                mock_pil_img = MagicMock()
                mock_pil_img.size = (800, 600)
                mock_bitmap.to_pil.return_value = mock_pil_img
                page.render.return_value = mock_bitmap

            mock_pdf.__iter__ = lambda self: iter(mock_pages)
            mock_pdfium.PdfDocument.return_value = mock_pdf

            from xtra.extractors.paddle_ocr import PaddleOcrExtractor

            extractor = PaddleOcrExtractor(Path("/fake/document.pdf"))
            assert extractor.get_page_count() == 3

    def test_extract_page_success_pdf(self) -> None:
        with (
            patch("xtra.extractors.paddle_ocr.PaddleOCR") as mock_paddle_class,
            patch("xtra.extractors._image_loader.pdfium") as mock_pdfium,
        ):
            mock_pdf = MagicMock()
            mock_page = MagicMock()
            mock_bitmap = MagicMock()
            mock_pil_img = MagicMock()
            mock_pil_img.size = (1200, 900)

            mock_bitmap.to_pil.return_value = mock_pil_img
            mock_page.render.return_value = mock_bitmap
            mock_pdf.__iter__ = lambda self: iter([mock_page])
            mock_pdfium.PdfDocument.return_value = mock_pdf

            mock_paddle = MagicMock()
            mock_paddle.ocr.return_value = [
                [
                    [[[50, 100], [250, 100], [250, 150], [50, 150]], ("PDF", 0.925)],
                ]
            ]
            mock_paddle_class.return_value = mock_paddle

            from xtra.extractors.paddle_ocr import PaddleOcrExtractor

            extractor = PaddleOcrExtractor(Path("/fake/document.pdf"))
            result = extractor.extract_page(0)

            assert result.success is True
            assert result.page.page == 0
            # Dimensions converted from pixels to points (default) at 200 DPI
            assert result.page.width == 432.0  # 1200 * (72/200)
            assert result.page.height == 324.0  # 900 * (72/200)
            assert len(result.page.texts) == 1
            assert result.page.texts[0].text == "PDF"

    def test_custom_dpi_pdf(self) -> None:
        with (
            patch("xtra.extractors.paddle_ocr.PaddleOCR"),
            patch("xtra.extractors._image_loader.pdfium") as mock_pdfium,
        ):
            mock_pdf = MagicMock()
            mock_page = MagicMock()
            mock_bitmap = MagicMock()
            mock_pil_img = MagicMock()
            mock_pil_img.size = (800, 600)

            mock_bitmap.to_pil.return_value = mock_pil_img
            mock_page.render.return_value = mock_bitmap
            mock_pdf.__iter__ = lambda self: iter([mock_page])
            mock_pdfium.PdfDocument.return_value = mock_pdf

            from xtra.extractors.paddle_ocr import PaddleOcrExtractor

            extractor = PaddleOcrExtractor(Path("/fake/document.pdf"), dpi=300)
            metadata = extractor.get_metadata()

            assert metadata.extra["dpi"] == 300
            # Check render was called with correct scale (300/72)
            mock_page.render.assert_called_once()
            call_kwargs = mock_page.render.call_args[1]
            assert call_kwargs["scale"] == pytest.approx(300 / 72, rel=0.01)
