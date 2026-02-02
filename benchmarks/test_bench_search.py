"""Search performance benchmarks."""

import re
from pathlib import Path

import pytest

from unifex.pdf import PdfExtractor

BENCH_DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def extracted_document():
    """Pre-extracted 40-page document for search benchmarks."""
    with PdfExtractor(BENCH_DATA / "benchmark_large.pdf") as ext:
        return ext.extract().document


class TestSearchBenchmarks:
    """Document search performance on 40-page document."""

    @pytest.mark.benchmark
    def test_search_substring(self, extracted_document) -> None:
        """Benchmark substring search."""
        results = extracted_document.search("Lorem ipsum")
        assert len(results) > 0

    @pytest.mark.benchmark
    def test_search_regex(self, extracted_document) -> None:
        """Benchmark regex search."""
        results = extracted_document.search(re.compile(r"Page \d+"))
        assert len(results) > 0

    @pytest.mark.benchmark
    def test_search_case_insensitive(self, extracted_document) -> None:
        """Benchmark case-insensitive search."""
        results = extracted_document.search("lorem", case_sensitive=False)
        assert len(results) > 0

    @pytest.mark.benchmark
    def test_search_specific_pages(self, extracted_document) -> None:
        """Benchmark search on subset of pages."""
        results = extracted_document.search("paragraph", pages=[0, 5, 10])
        assert len(results) > 0
