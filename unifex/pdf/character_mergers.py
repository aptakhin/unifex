"""Character mergers for PDF text extraction.

This module provides different strategies for merging individual characters
into TextBlocks during PDF extraction.
"""

from __future__ import annotations

import ctypes
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from unifex.base import BBox, FontInfo, TextBlock

if TYPE_CHECKING:
    import pypdfium2 as pdfium

# Import raw pdfium for optimized font extraction
try:
    import pypdfium2.raw as pdfium_c

    _HAS_PDFIUM_RAW = True
except ImportError:
    pdfium_c = None  # type: ignore[assignment]
    _HAS_PDFIUM_RAW = False

logger = logging.getLogger(__name__)

# Font cache type: (font_name, rounded_size) -> FontInfo
FontCacheKey = tuple[str | None, float]
FontCache = dict[FontCacheKey, FontInfo]


@dataclass
class CharInfo:
    """Information about a single character extracted from PDF."""

    char: str
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    rotation: float
    index: int


class CharacterMerger(ABC):
    """Abstract base class for character merging strategies."""

    @abstractmethod
    def merge(
        self,
        chars: list[CharInfo],
        textpage: pdfium.PdfTextPage | None,
        page_height: float,
    ) -> list[TextBlock]:
        """Merge characters into TextBlocks.

        Args:
            chars: List of character information extracted from the PDF page.
            textpage: The pdfium text page object for font info extraction.
                Can be None for testing without PDF dependencies.
            page_height: Height of the page for coordinate conversion.

        Returns:
            List of TextBlocks created from the characters.
        """
        ...

    def _extract_font_info(
        self,
        textpage: pdfium.PdfTextPage | None,
        char_index: int,
    ) -> FontInfo | None:
        """Extract font information for a character."""
        if textpage is None:
            return None
        try:
            text_obj = textpage.get_textobj(char_index)
            if text_obj is None:
                return None

            font = text_obj.get_font()
            font_size = text_obj.get_font_size()
            name = font.get_base_name() or font.get_family_name() or None
            weight = font.get_weight()

            return FontInfo(name=name, size=font_size, weight=weight)
        except (AttributeError, IndexError) as e:
            logger.debug("Failed to extract font info for char %d: %s", char_index, e)
            return None
        except Exception as e:
            # Catch pdfium errors without importing the module at top level
            if "PdfiumError" in type(e).__name__:
                logger.debug("Failed to extract font info for char %d: %s", char_index, e)
                return None
            raise


class BasicLineMerger(CharacterMerger):
    """Merges characters into text blocks based on line breaks.

    This is the default merger that groups characters on the same line
    into a single TextBlock. A new block is started when there's a
    significant vertical gap between characters.
    """

    def __init__(self, line_gap_threshold: float = 5.0) -> None:
        """Initialize the merger.

        Args:
            line_gap_threshold: Vertical gap (in points) that triggers a new block.
        """
        self.line_gap_threshold = line_gap_threshold

    def merge(
        self,
        chars: list[CharInfo],
        textpage: pdfium.PdfTextPage | None,
        page_height: float,
    ) -> list[TextBlock]:
        if not chars:
            return []

        blocks: list[TextBlock] = []
        current_chars: list[CharInfo] = []
        prev_char: CharInfo | None = None

        for char_info in chars:
            if self._is_new_block(prev_char, char_info):
                if current_chars:
                    block = self._create_text_block(current_chars, textpage, page_height)
                    if block and block.text.strip():
                        blocks.append(block)
                current_chars = []

            current_chars.append(char_info)
            prev_char = char_info

        if current_chars:
            block = self._create_text_block(current_chars, textpage, page_height)
            if block and block.text.strip():
                blocks.append(block)

        return blocks

    def _is_new_block(self, prev: CharInfo | None, curr: CharInfo) -> bool:
        if prev is None:
            return False
        vertical_gap = abs(curr.bbox[1] - prev.bbox[1])
        return vertical_gap > self.line_gap_threshold

    def _create_text_block(
        self,
        chars: list[CharInfo],
        textpage: pdfium.PdfTextPage | None,
        page_height: float,
    ) -> TextBlock | None:
        if not chars:
            return None

        text = "".join(c.char for c in chars).strip()
        if not text:
            return None

        x0 = min(c.bbox[0] for c in chars)
        y0 = min(c.bbox[1] for c in chars)
        x1 = max(c.bbox[2] for c in chars)
        y1 = max(c.bbox[3] for c in chars)

        bbox = BBox(x0=x0, y0=page_height - y1, x1=x1, y1=page_height - y0)
        rotation = chars[0].rotation if chars else 0
        font_info = self._extract_font_info(textpage, chars[0].index)

        return TextBlock(
            text=text,
            bbox=bbox,
            rotation=float(rotation),
            font_info=font_info,
        )


class KeepCharacterMerger(CharacterMerger):
    """Preserves each character as a separate TextBlock.

    This merger creates individual TextBlocks for each character,
    preserving the exact position and font information for every glyph.
    Useful for antifraud analysis and advanced text processing algorithms.

    Uses raw pdfium API with caching for 3.7x faster font extraction.
    """

    def merge(
        self,
        chars: list[CharInfo],
        textpage: pdfium.PdfTextPage | None,
        page_height: float,
    ) -> list[TextBlock]:
        blocks: list[TextBlock] = []
        font_cache: FontCache = {}

        for char_info in chars:
            x0, y0, x1, y1 = char_info.bbox
            bbox = BBox(x0=x0, y0=page_height - y1, x1=x1, y1=page_height - y0)
            font_info = self._extract_font_info_cached(textpage, char_info.index, font_cache)

            block = TextBlock(
                text=char_info.char,
                bbox=bbox,
                rotation=float(char_info.rotation),
                font_info=font_info,
            )
            blocks.append(block)

        return blocks

    def _extract_font_info_cached(
        self,
        textpage: pdfium.PdfTextPage | None,
        char_index: int,
        cache: FontCache,
    ) -> FontInfo | None:
        """Extract font info using raw pdfium API with caching.

        Uses FPDFText_GetFontInfo/GetFontSize/GetFontWeight directly,
        avoiding expensive Python object creation for repeated fonts.
        """
        if textpage is None or not _HAS_PDFIUM_RAW:
            return self._extract_font_info(textpage, char_index)

        try:
            # Get font name and size for cache key
            buf = ctypes.create_string_buffer(256)
            flags = ctypes.c_int()
            result = pdfium_c.FPDFText_GetFontInfo(
                textpage.raw, char_index, buf, 256, ctypes.byref(flags)
            )
            font_name = buf.value.decode("utf-8", errors="ignore") if result > 0 else None
            font_size = pdfium_c.FPDFText_GetFontSize(textpage.raw, char_index)

            # Round size for better cache hits
            cache_key: FontCacheKey = (font_name, round(font_size, 1))

            if cache_key in cache:
                return cache[cache_key]

            # Cache miss - get weight and create FontInfo
            weight = pdfium_c.FPDFText_GetFontWeight(textpage.raw, char_index)
            font_info = FontInfo(name=font_name, size=font_size, weight=weight)
            cache[cache_key] = font_info
            return font_info

        except Exception as e:
            logger.debug("Failed to extract font info for char %d: %s", char_index, e)
            return None
