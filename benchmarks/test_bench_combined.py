"""Combined extraction + search benchmarks (main usage pattern)."""

import re
from pathlib import Path

import pytest

from unifex.pdf import PdfExtractor

BENCH_DATA = Path(__file__).parent / "data"


class TestCombinedWorkflowBenchmarks:
    """The primary usage pattern: extract then search."""

    @pytest.mark.benchmark
    def test_extract_and_search(self) -> None:
        """Full workflow: extraction + substring search."""
        with PdfExtractor(BENCH_DATA / "benchmark_large.pdf") as ext:
            doc = ext.extract().document
            results = doc.search("Lorem ipsum")
        assert isinstance(results, list)

    @pytest.mark.benchmark
    def test_extract_and_multiple_searches(self) -> None:
        """Extract once, search multiple times (typical usage)."""
        with PdfExtractor(BENCH_DATA / "benchmark_large.pdf") as ext:
            doc = ext.extract().document
            r1 = doc.search("Lorem ipsum")
            r2 = doc.search("paragraph")
            r3 = doc.search(re.compile(r"Page \d+"))
        assert all(isinstance(r, list) for r in [r1, r2, r3])

    @pytest.mark.benchmark
    def test_extract_search_with_merge(self) -> None:
        """Extract and search with block merging."""
        with PdfExtractor(BENCH_DATA / "benchmark_large.pdf") as ext:
            doc = ext.extract().document
            results = doc.search("Lorem", merge_gap=10.0)
        assert isinstance(results, list)
