import argparse
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from xtra.cli import _build_table_options, main
from xtra.models import CoordinateUnit, ExtractorType

TEST_DATA_DIR = Path(__file__).parent.parent / "data"


def test_extractor_type_used_in_cli() -> None:
    # ExtractorType is used as the extractor choices in CLI
    assert ExtractorType.PDF == "pdf"
    assert ExtractorType.EASYOCR == "easyocr"
    assert ExtractorType.TESSERACT == "tesseract"
    assert ExtractorType.PADDLE == "paddle"
    assert ExtractorType.AZURE_DI == "azure-di"
    assert ExtractorType.GOOGLE_DOCAI == "google-docai"


def test_cli_pdf_extractor(capsys: pytest.CaptureFixture) -> None:
    test_args = ["cli", str(TEST_DATA_DIR / "test_pdf_2p_text.pdf"), "--extractor", "pdf"]
    with patch.object(sys, "argv", test_args):
        main()
    captured = capsys.readouterr()
    assert "=== Page 1 ===" in captured.out
    assert "First page. First text" in captured.out


def test_cli_pdf_extractor_json(capsys: pytest.CaptureFixture) -> None:
    test_args = [
        "cli",
        str(TEST_DATA_DIR / "test_pdf_2p_text.pdf"),
        "--extractor",
        "pdf",
        "--json",
    ]
    with patch.object(sys, "argv", test_args):
        main()
    captured = capsys.readouterr()
    assert '"text": "First page. First text"' in captured.out


def test_cli_pdf_extractor_specific_pages(capsys: pytest.CaptureFixture) -> None:
    test_args = [
        "cli",
        str(TEST_DATA_DIR / "test_pdf_2p_text.pdf"),
        "--extractor",
        "pdf",
        "--pages",
        "0",
    ]
    with patch.object(sys, "argv", test_args):
        main()
    captured = capsys.readouterr()
    assert "=== Page 1 ===" in captured.out
    assert "=== Page 2 ===" not in captured.out


def test_cli_file_not_found() -> None:
    test_args = ["cli", "/nonexistent/file.pdf", "--extractor", "pdf"]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


def test_cli_languages_parsing(capsys: pytest.CaptureFixture) -> None:
    test_args = [
        "cli",
        str(TEST_DATA_DIR / "test_pdf_2p_text.pdf"),
        "--extractor",
        "pdf",
        "--lang",
        "en,it,de",
    ]
    with patch.object(sys, "argv", test_args):
        main()
    # Should not raise, languages are parsed but only used for OCR
    captured = capsys.readouterr()
    assert "=== Page 1 ===" in captured.out


def test_cli_unit_option(capsys: pytest.CaptureFixture) -> None:
    """--unit option should work (renamed from --output-unit)."""
    test_args = [
        "cli",
        str(TEST_DATA_DIR / "test_pdf_2p_text.pdf"),
        "--extractor",
        "pdf",
        "--unit",
        "inches",
    ]
    with patch.object(sys, "argv", test_args):
        main()
    captured = capsys.readouterr()
    assert "=== Page 1 ===" in captured.out


def test_cli_tables_unsupported_extractor() -> None:
    """--tables should fail with unsupported extractors (e.g., easyocr)."""
    test_args = [
        "cli",
        str(TEST_DATA_DIR / "test_pdf_2p_text.pdf"),
        "--extractor",
        "easyocr",
        "--tables",
    ]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


@pytest.fixture
def make_table_args():
    """Factory fixture for creating table option args."""

    def _make(**kwargs) -> argparse.Namespace:
        defaults = {
            "unit": CoordinateUnit.POINTS.value,
            "dpi": 72,
            "pdf_table_area": None,
            "pdf_table_columns": None,
            "pdf_table_lattice": False,
            "pdf_table_stream": False,
        }
        if "unit" in kwargs and isinstance(kwargs["unit"], CoordinateUnit):
            kwargs["unit"] = kwargs["unit"].value
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    return _make


class TestBuildTableOptions:
    """Tests for _build_table_options coordinate conversion."""

    def test_no_area_or_columns(self, make_table_args) -> None:
        """When no area/columns specified, return empty dict with flags only."""
        args = make_table_args(pdf_table_lattice=True)
        result = _build_table_options(args, page_width=612, page_height=792)
        assert result == {"lattice": True}

    def test_area_in_points_no_conversion(self, make_table_args) -> None:
        """Area in points should be passed through unchanged."""
        args = make_table_args(unit=CoordinateUnit.POINTS, pdf_table_area="0,0,500,400")
        result = _build_table_options(args, page_width=612, page_height=792)
        assert result["area"] == [0.0, 0.0, 500.0, 400.0]

    def test_columns_in_points_no_conversion(self, make_table_args) -> None:
        """Columns in points should be passed through unchanged."""
        args = make_table_args(unit=CoordinateUnit.POINTS, pdf_table_columns="100,200,300")
        result = _build_table_options(args, page_width=612, page_height=792)
        assert result["columns"] == [100.0, 200.0, 300.0]

    def test_area_in_inches_converted_to_points(self, make_table_args) -> None:
        """Area in inches should be converted to points (1 inch = 72 points)."""
        args = make_table_args(unit=CoordinateUnit.INCHES, pdf_table_area="0,0,5,4")
        result = _build_table_options(args, page_width=612, page_height=792)
        assert result["area"] == pytest.approx([0.0, 0.0, 360.0, 288.0])

    def test_columns_in_inches_converted_to_points(self, make_table_args) -> None:
        """Columns in inches should be converted to points."""
        args = make_table_args(unit=CoordinateUnit.INCHES, pdf_table_columns="1,2,3")
        result = _build_table_options(args, page_width=612, page_height=792)
        assert result["columns"] == pytest.approx([72.0, 144.0, 216.0])

    def test_area_in_pixels_converted_to_points(self, make_table_args) -> None:
        """Area in pixels should be converted to points using DPI."""
        args = make_table_args(unit=CoordinateUnit.PIXELS, dpi=144, pdf_table_area="0,0,288,216")
        result = _build_table_options(args, page_width=612, page_height=792)
        assert result["area"] == pytest.approx([0.0, 0.0, 144.0, 108.0])

    def test_columns_in_pixels_converted_to_points(self, make_table_args) -> None:
        """Columns in pixels should be converted to points using DPI."""
        args = make_table_args(unit=CoordinateUnit.PIXELS, dpi=144, pdf_table_columns="144,288,432")
        result = _build_table_options(args, page_width=612, page_height=792)
        assert result["columns"] == pytest.approx([72.0, 144.0, 216.0])

    def test_area_in_normalized_converted_to_points(self, make_table_args) -> None:
        """Area in normalized (0-1) coords should be converted to points."""
        args = make_table_args(unit=CoordinateUnit.NORMALIZED, pdf_table_area="0,0,0.5,0.5")
        result = _build_table_options(args, page_width=612, page_height=792)
        assert result["area"] == pytest.approx([0.0, 0.0, 396.0, 306.0])

    def test_columns_in_normalized_converted_to_points(self, make_table_args) -> None:
        """Columns in normalized coords should be converted to points."""
        args = make_table_args(unit=CoordinateUnit.NORMALIZED, pdf_table_columns="0.25,0.5,0.75")
        result = _build_table_options(args, page_width=612, page_height=792)
        assert result["columns"] == pytest.approx([153.0, 306.0, 459.0])

    def test_lattice_and_stream_flags(self, make_table_args) -> None:
        """Lattice and stream flags should be included when set."""
        args = make_table_args(pdf_table_lattice=True, pdf_table_stream=True)
        result = _build_table_options(args, page_width=612, page_height=792)
        assert result["lattice"] is True
        assert result["stream"] is True
