"""Benchmarks for PDF extraction performance."""

from pathlib import Path

import pytest

from unifex.pdf import PdfExtractor

BENCH_DATA = Path(__file__).parent / "data"


class TestPdfExtractionBenchmarks:
    """PDF extraction performance on 40-page document."""

    @pytest.mark.benchmark
    def test_extract_full_document(self) -> None:
        """Benchmark extracting entire 40-page PDF."""
        with PdfExtractor(BENCH_DATA / "benchmark_large.pdf") as ext:
            result = ext.extract()
        assert result.success

    @pytest.mark.benchmark
    def test_extract_single_page(self) -> None:
        """Benchmark single page extraction."""
        with PdfExtractor(BENCH_DATA / "benchmark_large.pdf") as ext:
            result = ext.extract(pages=[0])
        assert result.success

    @pytest.mark.benchmark
    def test_extract_ten_pages(self) -> None:
        """Benchmark extracting 10 pages."""
        with PdfExtractor(BENCH_DATA / "benchmark_large.pdf") as ext:
            result = ext.extract(pages=list(range(10)))
        assert result.success
