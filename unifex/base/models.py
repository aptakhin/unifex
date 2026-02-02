from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

try:
    from pydantic import BaseModel, ConfigDict, Field

    PYDANTIC_V2 = True
except ImportError:
    from pydantic import BaseModel, Field

    PYDANTIC_V2 = False


class ExtractorType(StrEnum):
    PDF = "pdf"
    EASYOCR = "easyocr"
    TESSERACT = "tesseract"
    PADDLE = "paddle"
    AZURE_DI = "azure-di"
    GOOGLE_DOCAI = "google-docai"


class CoordinateUnit(StrEnum):
    """Units for coordinate output."""

    PIXELS = "pixels"  # Image pixels at a given DPI
    POINTS = "points"  # 1/72 inch (PDF native, default)
    INCHES = "inches"  # Imperial inches
    NORMALIZED = "normalized"  # 0-1 relative to page dimensions


class BBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class FontInfo(BaseModel):
    name: str | None = None
    size: float | None = None
    flags: int | None = None
    weight: int | None = None


class TextBlock(BaseModel):
    text: str
    bbox: BBox
    rotation: float = 0.0
    confidence: float | None = None
    font_info: FontInfo | None = None


class CoordinateInfo(BaseModel):
    """Information about the coordinate system used."""

    unit: CoordinateUnit
    dpi: float | None = None  # Only meaningful for pixel-based coords


class TableCell(BaseModel):
    """A cell within a table."""

    text: str
    row: int
    col: int
    bbox: BBox | None = None  # Cell bbox if available


class Table(BaseModel):
    """A table extracted from a document page."""

    page: int  # Page number (0-indexed)
    cells: list[TableCell] = Field(default_factory=list)
    row_count: int = 0
    col_count: int = 0
    bbox: BBox | None = None  # Table bbox if available

    def to_dataframe(self) -> pd.DataFrame:
        """Convert table to pandas DataFrame.

        Returns:
            DataFrame with all rows as data. Columns are numbered 0, 1, 2, ...
            Missing cells are filled with empty strings.

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). Install with: pip install unifex[tables]"
            ) from None

        # Build 2D grid from sparse cell list
        grid: dict[tuple[int, int], str] = {(cell.row, cell.col): cell.text for cell in self.cells}

        # Create rows as lists
        data = []
        for row_idx in range(self.row_count):
            row = [grid.get((row_idx, col_idx), "") for col_idx in range(self.col_count)]
            data.append(row)

        return pd.DataFrame(data)


class Page(BaseModel):
    page: int
    width: float
    height: float
    texts: list[TextBlock] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)
    coordinate_info: CoordinateInfo | None = None

    def search(
        self,
        pattern: str | re.Pattern[str],
        *,
        case_sensitive: bool = True,
    ) -> list[TextBlock]:
        """Search for text blocks matching a pattern.

        Args:
            pattern: String for substring search, or compiled regex pattern.
            case_sensitive: Whether search is case-sensitive (default True).
                           Ignored if pattern is already a compiled regex.

        Returns:
            List of matching TextBlock objects.
        """
        if isinstance(pattern, re.Pattern):
            compiled = pattern
        else:
            flags = 0 if case_sensitive else re.IGNORECASE
            compiled = re.compile(re.escape(pattern), flags)

        return [block for block in self.texts if compiled.search(block.text)]


class ExtractorMetadata(BaseModel):
    extractor_type: ExtractorType
    creator: str | None = None
    producer: str | None = None
    title: str | None = None
    author: str | None = None
    creation_date: str | None = None
    modification_date: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """A search result containing matched text and its location."""

    page: int
    block: TextBlock
    original_blocks: list[TextBlock] = Field(default_factory=list)


def _merge_blocks_by_gap(blocks: list[TextBlock], gap: float) -> list[TextBlock]:
    """Merge horizontally adjacent TextBlocks within gap threshold.

    Args:
        blocks: TextBlocks sorted by x0 position (assumed same line).
        gap: Maximum horizontal gap between blocks to merge.

    Returns:
        List of merged TextBlocks.
    """
    if not blocks:
        return []

    # Sort by x0 to ensure left-to-right order
    sorted_blocks = sorted(blocks, key=lambda b: b.bbox.x0)
    merged: list[TextBlock] = []
    current_group: list[TextBlock] = [sorted_blocks[0]]

    for block in sorted_blocks[1:]:
        prev = current_group[-1]
        # Check horizontal gap (block.x0 - prev.x1)
        if block.bbox.x0 - prev.bbox.x1 <= gap:
            current_group.append(block)
        else:
            # Finalize current group
            merged.append(_create_merged_block(current_group))
            current_group = [block]

    # Finalize last group
    merged.append(_create_merged_block(current_group))
    return merged


def _create_merged_block(blocks: list[TextBlock]) -> TextBlock:
    """Create a single TextBlock from a list of adjacent blocks."""
    if len(blocks) == 1:
        return blocks[0]

    text = " ".join(b.text for b in blocks)
    x0 = min(b.bbox.x0 for b in blocks)
    y0 = min(b.bbox.y0 for b in blocks)
    x1 = max(b.bbox.x1 for b in blocks)
    y1 = max(b.bbox.y1 for b in blocks)

    return TextBlock(
        text=text,
        bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
        rotation=blocks[0].rotation,
        confidence=None,  # Can't combine confidences meaningfully
    )


def _search_blocks(
    blocks: list[TextBlock],
    pattern: str | re.Pattern[str],
    case_sensitive: bool,
    merge_gap: float | None,
    line_gap: float,
) -> list[tuple[TextBlock, list[TextBlock]]]:
    """Search blocks for pattern, optionally merging adjacent blocks first.

    Returns:
        List of (matched_block, original_blocks) tuples.
    """
    if isinstance(pattern, re.Pattern):
        compiled = pattern
    else:
        flags = 0 if case_sensitive else re.IGNORECASE
        compiled = re.compile(re.escape(pattern), flags)

    if merge_gap is not None:
        # Group blocks by approximate y position (same line)
        line_groups: dict[int, list[TextBlock]] = {}
        for block in blocks:
            # Round y0 using line_gap tolerance
            line_key = int(block.bbox.y0 / line_gap) if line_gap > 0 else 0
            if line_key not in line_groups:
                line_groups[line_key] = []
            line_groups[line_key].append(block)

        # Merge each line and search
        results: list[tuple[TextBlock, list[TextBlock]]] = []
        for line_blocks in line_groups.values():
            sorted_line = sorted(line_blocks, key=lambda b: b.bbox.x0)
            merged = _merge_blocks_by_gap(sorted_line, merge_gap)

            for merged_block in merged:
                if compiled.search(merged_block.text):
                    # Find which original blocks contributed
                    originals = [b for b in sorted_line if _blocks_overlap(b, merged_block)]
                    results.append((merged_block, originals))
        return results
    else:
        # No merging, search individual blocks
        return [(block, [block]) for block in blocks if compiled.search(block.text)]


def _blocks_overlap(a: TextBlock, b: TextBlock) -> bool:
    """Check if block a overlaps with block b horizontally."""
    return not (a.bbox.x1 < b.bbox.x0 or a.bbox.x0 > b.bbox.x1)


class _DocumentSearchMixin:
    """Mixin providing search functionality for Document."""

    pages: list[Page]

    def search(
        self,
        pattern: str | re.Pattern[str],
        *,
        case_sensitive: bool = True,
        pages: int | list[int] | None = None,
        merge_gap: float | None = None,
        line_gap: float = 5.0,
    ) -> list[SearchResult]:
        """Search for text blocks matching a pattern.

        Args:
            pattern: String for substring search, or compiled regex pattern.
            case_sensitive: Whether search is case-sensitive (default True).
                           Ignored if pattern is already a compiled regex.
            pages: Page number(s) to search. None searches all pages.
                  Can be a single int or list of ints (0-indexed).
            merge_gap: If set, merge horizontally adjacent blocks within this
                      gap (in coordinate units) before searching. Useful for
                      word-level OCR output when searching for phrases.
            line_gap: Vertical tolerance for grouping blocks into lines
                     when merge_gap is used (default 5.0 points).

        Returns:
            List of SearchResult objects.
        """
        # Normalize pages parameter
        if pages is None:
            target_pages = None
        elif isinstance(pages, int):
            target_pages = {pages}
        else:
            target_pages = set(pages)

        results: list[SearchResult] = []
        for page in self.pages:
            if target_pages is not None and page.page not in target_pages:
                continue

            matches = _search_blocks(page.texts, pattern, case_sensitive, merge_gap, line_gap)
            for matched_block, originals in matches:
                results.append(
                    SearchResult(
                        page=page.page,
                        block=matched_block,
                        original_blocks=originals,
                    )
                )

        return results


if PYDANTIC_V2:

    class Document(_DocumentSearchMixin, BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        path: Path
        pages: list[Page] = Field(default_factory=list)
        metadata: ExtractorMetadata | None = None
else:

    class Document(_DocumentSearchMixin, BaseModel):
        path: Path
        pages: list[Page] = Field(default_factory=list)
        metadata: ExtractorMetadata | None = None

        class Config:
            arbitrary_types_allowed = True
