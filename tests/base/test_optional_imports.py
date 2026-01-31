"""Tests for optional dependency handling and lazy imports."""

import builtins
import sys
from unittest.mock import patch

import pytest


class TestLazyImports:
    """Test lazy import functionality in __init__.py modules."""

    def test_xtra_getattr_returns_extractor(self):
        """Test that __getattr__ returns the correct extractor class."""
        from xtra import EasyOcrExtractor

        assert EasyOcrExtractor is not None
        assert EasyOcrExtractor.__name__ == "EasyOcrExtractor"

    def test_xtra_getattr_raises_attribute_error_for_unknown(self):
        """Test that __getattr__ raises AttributeError for unknown names."""
        import xtra

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = xtra.NonExistentClass

    def test_ocr_extractors_import(self):
        """Test that OCR extractors can be imported."""
        from xtra.ocr.extractors.paddle_ocr import PaddleOcrExtractor

        assert PaddleOcrExtractor is not None
        assert PaddleOcrExtractor.__name__ == "PaddleOcrExtractor"


class TestCheckFunctions:
    """Test the _check_xxx_installed functions with mocked imports."""

    @pytest.mark.parametrize(
        "module_path,func_name,blocked_import,expected_extra",
        [
            ("xtra.ocr.extractors.easy_ocr", "_check_easyocr_installed", "easyocr", "easyocr"),
            (
                "xtra.ocr.extractors.tesseract_ocr",
                "_check_pytesseract_installed",
                "pytesseract",
                "tesseract",
            ),
            ("xtra.ocr.extractors.paddle_ocr", "_check_paddleocr_installed", "paddleocr", "paddle"),
            ("xtra.ocr.extractors.azure_di", "_check_azure_installed", "azure", "azure"),
            (
                "xtra.ocr.extractors.google_docai",
                "_check_google_docai_installed",
                "google",
                "google",
            ),
        ],
        ids=["easyocr", "tesseract", "paddleocr", "azure", "google"],
    )
    def test_check_installed_raises_when_missing(
        self, module_path, func_name, blocked_import, expected_extra
    ):
        """Test that _check_xxx_installed raises ImportError when dependency is missing."""
        import importlib

        module = importlib.import_module(module_path)
        check_func = getattr(module, func_name)

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if blocked_import in name:
                raise ImportError(f"No module named {blocked_import}")
            return original_import(name, *args, **kwargs)

        saved = {k: v for k, v in sys.modules.items() if blocked_import in k}
        for k in saved:
            del sys.modules[k]

        with patch.object(builtins, "__import__", mock_import):
            with pytest.raises(ImportError, match=f"pip install xtra\\[{expected_extra}\\]"):
                check_func()

        sys.modules.update(saved)
