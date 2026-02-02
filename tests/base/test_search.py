import re
from pathlib import Path

from unifex.base import BBox, Document, Page, TextBlock


def make_text_block(
    text: str, x0: float = 0, y0: float = 0, x1: float = 100, y1: float = 20
) -> TextBlock:
    """Helper to create a TextBlock with minimal boilerplate."""
    return TextBlock(text=text, bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1))


def make_page_with_blocks(page_num: int, blocks: list[TextBlock]) -> Page:
    """Helper to create a Page with specific TextBlocks."""
    return Page(page=page_num, width=595.0, height=842.0, texts=blocks)


def make_page(page_num: int, texts: list[str]) -> Page:
    """Helper to create a Page with text blocks (auto-positioned horizontally)."""
    blocks = [make_text_block(t, x0=i * 50, x1=i * 50 + 40) for i, t in enumerate(texts)]
    return make_page_with_blocks(page_num, blocks)


def make_document(pages_texts: list[list[str]]) -> Document:
    """Helper to create a Document with pages and texts."""
    pages = [make_page(i, texts) for i, texts in enumerate(pages_texts)]
    return Document(path=Path("/tmp/test.pdf"), pages=pages)


class TestDocumentSearch:
    """Tests for Document.search() method."""

    def test_search_substring_single_match(self) -> None:
        """Search finds a single matching text block by substring."""
        doc = make_document([["First page.", "Second text"]])
        results = doc.search("First")
        assert len(results) == 1
        assert results[0].page == 0
        assert results[0].block.text == "First page."

    def test_search_substring_multiple_matches(self) -> None:
        """Search finds multiple matching text blocks."""
        doc = make_document([["First page.", "First text", "Other"]])
        results = doc.search("First")
        assert len(results) == 2
        assert results[0].block.text == "First page."
        assert results[1].block.text == "First text"

    def test_search_substring_across_pages(self) -> None:
        """Search finds matches across multiple pages."""
        doc = make_document([["First page."], ["First on page 2"]])
        results = doc.search("First")
        assert len(results) == 2
        assert results[0].page == 0
        assert results[1].page == 1

    def test_search_substring_no_match(self) -> None:
        """Search returns empty list when no matches found."""
        doc = make_document([["Hello world"]])
        results = doc.search("Goodbye")
        assert results == []

    def test_search_substring_case_sensitive(self) -> None:
        """Search is case-sensitive by default."""
        doc = make_document([["First page.", "first text"]])
        results = doc.search("First")
        assert len(results) == 1
        assert results[0].block.text == "First page."

    def test_search_substring_case_insensitive(self) -> None:
        """Search can be case-insensitive."""
        doc = make_document([["First page.", "first text"]])
        results = doc.search("First", case_sensitive=False)
        assert len(results) == 2

    def test_search_compiled_regex(self) -> None:
        """Search with compiled regex pattern."""
        doc = make_document([["First page.", "Second page."]])
        results = doc.search(re.compile(r"\w+ page\."))
        assert len(results) == 2

    def test_search_compiled_regex_case_insensitive(self) -> None:
        """Search with compiled regex that has its own flags."""
        doc = make_document([["First page.", "FIRST text"]])
        results = doc.search(re.compile(r"first", re.IGNORECASE))
        assert len(results) == 2

    def test_search_empty_document(self) -> None:
        """Search on empty document returns empty list."""
        doc = Document(path=Path("/tmp/test.pdf"), pages=[])
        results = doc.search("test")
        assert results == []

    def test_search_empty_page(self) -> None:
        """Search on document with empty pages returns empty list."""
        doc = make_document([[]])
        results = doc.search("test")
        assert results == []

    def test_search_specific_page(self) -> None:
        """Search on specific page only."""
        doc = make_document([["First match"], ["Second match"], ["Third match"]])
        results = doc.search("match", pages=1)
        assert len(results) == 1
        assert results[0].page == 1

    def test_search_multiple_pages(self) -> None:
        """Search on multiple specific pages."""
        doc = make_document([["First match"], ["Second match"], ["Third match"]])
        results = doc.search("match", pages=[0, 2])
        assert len(results) == 2
        assert results[0].page == 0
        assert results[1].page == 2


class TestMergedTextSearch:
    """Tests for searching text that spans multiple TextBlocks (word-level OCR)."""

    def test_search_merged_adjacent_blocks(self) -> None:
        """Search finds pattern spanning adjacent text blocks."""
        # Simulate word-level OCR output: "First" at x=0-40, "page" at x=50-90
        blocks = [
            TextBlock(text="First", bbox=BBox(x0=0, y0=0, x1=40, y1=20)),
            TextBlock(text="page", bbox=BBox(x0=50, y0=0, x1=90, y1=20)),
        ]
        doc = Document(
            path=Path("/tmp/test.pdf"),
            pages=[make_page_with_blocks(0, blocks)],
        )
        # With merge_gap=15, blocks 10 units apart should merge
        results = doc.search("First page", merge_gap=15.0)
        assert len(results) == 1
        # block is the merged result
        assert results[0].block.text == "First page"
        assert results[0].block.bbox.x0 == 0
        assert results[0].block.bbox.x1 == 90
        # original_blocks preserves the originals
        assert len(results[0].original_blocks) == 2
        assert results[0].original_blocks[0].text == "First"
        assert results[0].original_blocks[1].text == "page"

    def test_search_merged_regex(self) -> None:
        """Search with regex spanning merged blocks."""
        blocks = [
            TextBlock(text="First", bbox=BBox(x0=0, y0=0, x1=40, y1=20)),
            TextBlock(text="page.", bbox=BBox(x0=50, y0=0, x1=100, y1=20)),
        ]
        doc = Document(
            path=Path("/tmp/test.pdf"),
            pages=[make_page_with_blocks(0, blocks)],
        )
        results = doc.search(re.compile(r"\w+ page\."), merge_gap=15.0)
        assert len(results) == 1
        assert results[0].block.text == "First page."
        assert len(results[0].original_blocks) == 2

    def test_search_no_merge_without_gap(self) -> None:
        """Without merge_gap, multi-word search doesn't match separate blocks."""
        blocks = [
            TextBlock(text="First", bbox=BBox(x0=0, y0=0, x1=40, y1=20)),
            TextBlock(text="page", bbox=BBox(x0=50, y0=0, x1=90, y1=20)),
        ]
        doc = Document(
            path=Path("/tmp/test.pdf"),
            pages=[make_page_with_blocks(0, blocks)],
        )
        # Without merge_gap, "First page" won't match individual blocks
        results = doc.search("First page")
        assert len(results) == 0

    def test_search_merged_three_blocks(self) -> None:
        """Search spans three adjacent text blocks."""
        blocks = [
            TextBlock(text="one", bbox=BBox(x0=0, y0=0, x1=30, y1=20)),
            TextBlock(text="two", bbox=BBox(x0=40, y0=0, x1=70, y1=20)),
            TextBlock(text="three", bbox=BBox(x0=80, y0=0, x1=120, y1=20)),
        ]
        doc = Document(
            path=Path("/tmp/test.pdf"),
            pages=[make_page_with_blocks(0, blocks)],
        )
        results = doc.search("one two three", merge_gap=15.0)
        assert len(results) == 1
        assert results[0].block.text == "one two three"
        assert len(results[0].original_blocks) == 3

    def test_search_partial_block_no_merge(self) -> None:
        """Search matches partial text from single block (no merge needed)."""
        doc = make_document([["Hello world"]])
        results = doc.search("Hello")
        assert len(results) == 1
        # Single block: original_blocks contains just that one block
        assert len(results[0].original_blocks) == 1
        assert results[0].original_blocks[0].text == "Hello world"

    def test_search_blocks_too_far_apart(self) -> None:
        """Blocks beyond merge_gap threshold are not merged."""
        blocks = [
            TextBlock(text="First", bbox=BBox(x0=0, y0=0, x1=40, y1=20)),
            TextBlock(text="page", bbox=BBox(x0=100, y0=0, x1=140, y1=20)),  # 60 units gap
        ]
        doc = Document(
            path=Path("/tmp/test.pdf"),
            pages=[make_page_with_blocks(0, blocks)],
        )
        # With merge_gap=15, blocks 60 units apart should NOT merge
        results = doc.search("First page", merge_gap=15.0)
        assert len(results) == 0


class TestPageSearch:
    """Tests for Page.search() method."""

    def test_page_search_substring(self) -> None:
        """Page search finds matching text blocks."""
        page = make_page(0, ["First text", "Second text", "Other"])
        results = page.search("text")
        assert len(results) == 2

    def test_page_search_compiled_regex(self) -> None:
        """Page search with compiled regex."""
        page = make_page(0, ["Test123", "Test456", "Hello"])
        results = page.search(re.compile(r"Test\d+"))
        assert len(results) == 2

    def test_page_search_no_match(self) -> None:
        """Page search returns empty when no matches."""
        page = make_page(0, ["Hello", "World"])
        results = page.search("Goodbye")
        assert results == []


class TestSearchResultModel:
    """Tests for SearchResult model structure."""

    def test_search_result_has_page(self) -> None:
        """SearchResult contains page number."""
        doc = make_document([["Test"]])
        results = doc.search("Test")
        assert results[0].page == 0

    def test_search_result_has_block(self) -> None:
        """SearchResult contains merged block."""
        doc = make_document([["Test"]])
        results = doc.search("Test")
        assert isinstance(results[0].block, TextBlock)
        assert results[0].block.text == "Test"

    def test_search_result_has_original_blocks(self) -> None:
        """SearchResult contains list of original blocks."""
        doc = make_document([["Test"]])
        results = doc.search("Test")
        assert isinstance(results[0].original_blocks, list)
        assert len(results[0].original_blocks) == 1

    def test_search_preserves_bbox(self) -> None:
        """Search results preserve bounding box information."""
        doc = Document(
            path=Path("/tmp/test.pdf"),
            pages=[
                Page(
                    page=0,
                    width=595.0,
                    height=842.0,
                    texts=[
                        TextBlock(
                            text="First page.",
                            bbox=BBox(x0=46.7, y0=55.6, x1=130.3, y1=74.5),
                            confidence=0.99,
                        )
                    ],
                )
            ],
        )
        results = doc.search("First")
        assert len(results) == 1
        # block has same data as the original
        assert results[0].block.bbox.x0 == 46.7
        assert results[0].block.bbox.y0 == 55.6
        assert results[0].block.confidence == 0.99
        # original_blocks[0] is the actual original
        assert results[0].original_blocks[0].confidence == 0.99
