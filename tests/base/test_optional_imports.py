"""Tests for optional dependency handling and lazy imports."""

import builtins
import sys
from unittest.mock import patch

import pytest


class TestLazyImports:
    """Test lazy import functionality in __init__.py modules."""

    def test_unifex_getattr_returns_extractor(self):
        """Test that __getattr__ returns the correct extractor class."""
        from unifex import EasyOcrExtractor

        assert EasyOcrExtractor is not None
        assert EasyOcrExtractor.__name__ == "EasyOcrExtractor"

    def test_unifex_getattr_raises_attribute_error_for_unknown(self):
        """Test that __getattr__ raises AttributeError for unknown names."""
        import unifex

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = unifex.NonExistentClass

    def test_ocr_extractors_import(self):
        """Test that OCR extractors can be imported."""
        from unifex.ocr.extractors.paddle_ocr import PaddleOcrExtractor

        assert PaddleOcrExtractor is not None
        assert PaddleOcrExtractor.__name__ == "PaddleOcrExtractor"


class TestCheckFunctions:
    """Test the _check_xxx_installed functions with mocked imports."""

    @pytest.mark.parametrize(
        "module_path,func_name,blocked_import,expected_extra",
        [
            ("unifex.ocr.extractors.easy_ocr", "_check_easyocr_installed", "easyocr", "easyocr"),
            (
                "unifex.ocr.extractors.tesseract_ocr",
                "_check_pytesseract_installed",
                "pytesseract",
                "tesseract",
            ),
            (
                "unifex.ocr.extractors.paddle_ocr",
                "_check_paddleocr_installed",
                "paddleocr",
                "paddle",
            ),
            ("unifex.ocr.extractors.azure_di", "_check_azure_installed", "azure", "azure"),
            (
                "unifex.ocr.extractors.google_docai",
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
            with pytest.raises(ImportError, match=f"pip install unifex\\[{expected_extra}\\]"):
                check_func()

        sys.modules.update(saved)
