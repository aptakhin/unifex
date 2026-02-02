"""Tests for parallel LLM extraction functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from unifex.base import ExecutorType
from unifex.llm.models import (
    LLMBatchExtractionResult,
    LLMExtractionResult,
    LLMProvider,
    PageExtractionResult,
)

TEST_DATA_DIR = Path(__file__).parent.parent / "data"


def make_fake_extractor(
    responses: dict[tuple[int, ...], dict[str, Any]] | None = None,
    error_pages: set[int] | None = None,
    usage: dict[str, int] | None = None,
) -> Any:
    """Create a fake extractor for testing.

    Args:
        responses: Dict mapping page tuples to response data.
        error_pages: Set of pages that should raise errors.
        usage: Usage dict to return with each response.
    """
    error_pages = error_pages or set()
    usage = usage or {"prompt_tokens": 100, "completion_tokens": 50}

    def fake_extractor(  # noqa: PLR0913
        path: Path,
        model: str,
        schema: Any,
        prompt: str | None,
        pages: list[int] | None,
        dpi: int,
        max_retries: int,
        temperature: float,
        credentials: dict[str, str] | None,
        base_url: str | None,
        headers: dict[str, str] | None,
    ) -> LLMExtractionResult[dict[str, Any]]:
        page_key = tuple(pages) if pages else ()

        # Check if any page should error
        if pages:
            for page in pages:
                if page in error_pages:
                    raise ValueError(f"API error for page {page}")

        # Get response data
        if responses and page_key in responses:
            data = responses[page_key]
        else:
            data = {"pages": pages}

        return LLMExtractionResult(
            data=data,
            model="gpt-4o",
            provider=LLMProvider.OPENAI,
            usage=usage,
        )

    return fake_extractor


def make_fake_async_extractor(
    responses: dict[tuple[int, ...], dict[str, Any]] | None = None,
    error_pages: set[int] | None = None,
    usage: dict[str, int] | None = None,
) -> Any:
    """Create an async fake extractor for testing."""
    sync_extractor = make_fake_extractor(responses, error_pages, usage)

    async def fake_async_extractor(  # noqa: PLR0913
        path: Path,
        model: str,
        schema: Any,
        prompt: str | None,
        pages: list[int] | None,
        dpi: int,
        max_retries: int,
        temperature: float,
        credentials: dict[str, str] | None,
        base_url: str | None,
        headers: dict[str, str] | None,
    ) -> LLMExtractionResult[dict[str, Any]]:
        return sync_extractor(
            path,
            model,
            schema,
            prompt,
            pages,
            dpi,
            max_retries,
            temperature,
            credentials,
            base_url,
            headers,
        )

    return fake_async_extractor


class TestExtractStructuredSingle:
    """Tests for extract_structured (single extraction, no max_workers)."""

    def test_returns_single_result(self) -> None:
        """Test that extract_structured returns LLMExtractionResult."""
        from unifex.llm_factory import extract_structured

        fake = make_fake_extractor()

        result = extract_structured(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            model="openai/gpt-4o",
            pages=[0, 1, 2],
            _extractor=fake,
        )

        # Returns LLMExtractionResult, not batch result
        assert isinstance(result, LLMExtractionResult)
        assert result.data == {"pages": [0, 1, 2]}

    def test_all_pages_in_one_request(self) -> None:
        """Test that all pages are sent in one request."""
        from unifex.llm_factory import extract_structured

        call_log: list[list[int] | None] = []

        def tracking_extractor(
            path: Path,
            model: str,
            schema: Any,
            prompt: str | None,
            pages: list[int] | None,
            *args: Any,
            **kwargs: Any,
        ) -> LLMExtractionResult[dict[str, Any]]:
            call_log.append(pages)
            return LLMExtractionResult(
                data={"pages": pages}, model="gpt-4o", provider=LLMProvider.OPENAI
            )

        extract_structured(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            model="openai/gpt-4o",
            pages=[0, 1, 2],
            _extractor=tracking_extractor,
        )

        assert len(call_log) == 1
        assert call_log[0] == [0, 1, 2]


class TestExtractStructuredParallel:
    """Tests for extract_structured_parallel."""

    def test_returns_batch_result(self) -> None:
        """Test that extract_structured_parallel returns LLMBatchExtractionResult."""
        from unifex.llm_factory import extract_structured_parallel

        fake = make_fake_extractor()

        result = extract_structured_parallel(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            model="openai/gpt-4o",
            pages=[0, 1, 2],
            max_workers=2,
            _extractor=fake,
        )

        # Returns LLMBatchExtractionResult
        assert isinstance(result, LLMBatchExtractionResult)
        assert len(result.results) == 3

    def test_page_results_have_correct_structure(self) -> None:
        """Test that each page result has correct structure."""
        from unifex.llm_factory import extract_structured_parallel

        fake = make_fake_extractor()

        result = extract_structured_parallel(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            model="openai/gpt-4o",
            pages=[0, 1],
            max_workers=2,
            _extractor=fake,
        )

        # Each result is PageExtractionResult
        for page_result in result.results:
            assert isinstance(page_result, PageExtractionResult)
            assert page_result.page is not None
            assert page_result.data is not None

    def test_preserves_page_order(self) -> None:
        """Test that results maintain original page order."""
        from unifex.llm_factory import extract_structured_parallel

        fake = make_fake_extractor()

        result = extract_structured_parallel(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            model="openai/gpt-4o",
            pages=[0, 1, 2, 3],
            max_workers=4,
            _extractor=fake,
        )

        # Results should be in original page order
        assert result.results[0].page == 0
        assert result.results[0].data is not None
        assert result.results[0].data["pages"] == [0]
        assert result.results[1].page == 1
        assert result.results[1].data is not None
        assert result.results[1].data["pages"] == [1]
        assert result.results[2].page == 2
        assert result.results[3].page == 3

    def test_aggregates_usage(self) -> None:
        """Test that usage is aggregated across all pages."""
        from unifex.llm_factory import extract_structured_parallel

        fake = make_fake_extractor(usage={"prompt_tokens": 100, "completion_tokens": 50})

        result = extract_structured_parallel(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            model="openai/gpt-4o",
            pages=[0, 1],
            max_workers=2,
            _extractor=fake,
        )

        # Total usage should be aggregated
        assert result.total_usage == {"prompt_tokens": 200, "completion_tokens": 100}

        # Individual page usage should also be present
        assert result.results[0].usage == {"prompt_tokens": 100, "completion_tokens": 50}
        assert result.results[1].usage == {"prompt_tokens": 100, "completion_tokens": 50}

    def test_captures_page_errors(self) -> None:
        """Test that errors on individual pages are captured, not raised."""
        from unifex.llm_factory import extract_structured_parallel

        fake = make_fake_extractor(error_pages={1})

        result = extract_structured_parallel(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            model="openai/gpt-4o",
            pages=[0, 1, 2],
            max_workers=2,
            _extractor=fake,
        )

        # Page 0 succeeded
        assert result.results[0].data is not None
        assert result.results[0].error is None

        # Page 1 failed - error captured
        assert result.results[1].data is None
        assert result.results[1].error is not None
        assert "API error for page 1" in result.results[1].error

        # Page 2 succeeded
        assert result.results[2].data is not None
        assert result.results[2].error is None

    def test_thread_executor(self) -> None:
        """Test with thread executor type."""
        from unifex.llm_factory import extract_structured_parallel

        fake = make_fake_extractor()

        result = extract_structured_parallel(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            model="openai/gpt-4o",
            pages=[0, 1],
            max_workers=2,
            executor=ExecutorType.THREAD,
            _extractor=fake,
        )

        assert len(result.results) == 2


class TestExtractStructuredParallelAsync:
    """Tests for extract_structured_parallel_async."""

    @pytest.mark.asyncio
    async def test_returns_batch_result(self) -> None:
        """Test that async parallel returns LLMBatchExtractionResult."""
        from unifex.llm_factory import extract_structured_parallel_async

        fake_async = make_fake_async_extractor()

        result = await extract_structured_parallel_async(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            model="openai/gpt-4o",
            pages=[0, 1, 2],
            max_workers=2,
            _extractor=fake_async,
        )

        assert isinstance(result, LLMBatchExtractionResult)
        assert len(result.results) == 3

    @pytest.mark.asyncio
    async def test_preserves_order(self) -> None:
        """Test that async parallel preserves page order."""
        from unifex.llm_factory import extract_structured_parallel_async

        fake_async = make_fake_async_extractor()

        result = await extract_structured_parallel_async(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            model="openai/gpt-4o",
            pages=[0, 1, 2, 3],
            max_workers=4,
            _extractor=fake_async,
        )

        assert result.results[0].page == 0
        assert result.results[1].page == 1
        assert result.results[2].page == 2
        assert result.results[3].page == 3

    @pytest.mark.asyncio
    async def test_captures_errors(self) -> None:
        """Test that async parallel captures errors."""
        from unifex.llm_factory import extract_structured_parallel_async

        fake_async = make_fake_async_extractor(error_pages={1})

        result = await extract_structured_parallel_async(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            model="openai/gpt-4o",
            pages=[0, 1, 2],
            max_workers=2,
            _extractor=fake_async,
        )

        assert result.results[0].error is None
        assert result.results[1].error is not None
        assert result.results[2].error is None

    @pytest.mark.asyncio
    async def test_aggregates_usage(self) -> None:
        """Test that async parallel aggregates usage."""
        from unifex.llm_factory import extract_structured_parallel_async

        fake_async = make_fake_async_extractor(
            usage={"prompt_tokens": 100, "completion_tokens": 50}
        )

        result = await extract_structured_parallel_async(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            model="openai/gpt-4o",
            pages=[0, 1],
            max_workers=2,
            _extractor=fake_async,
        )

        assert result.total_usage == {"prompt_tokens": 200, "completion_tokens": 100}


class TestExtractStructuredAsyncSingle:
    """Tests for extract_structured_async (single extraction, no max_workers)."""

    @pytest.mark.asyncio
    async def test_returns_single_result(self) -> None:
        """Test that extract_structured_async returns LLMExtractionResult."""
        from unifex.llm_factory import extract_structured_async

        fake_async = make_fake_async_extractor()

        result = await extract_structured_async(
            TEST_DATA_DIR / "test_pdf_2p_text.pdf",
            model="openai/gpt-4o",
            pages=[0, 1],
            _extractor=fake_async,
        )

        assert isinstance(result, LLMExtractionResult)
        assert result.data == {"pages": [0, 1]}
