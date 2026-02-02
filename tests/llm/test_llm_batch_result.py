"""Tests for LLMBatchExtractionResult model."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from unifex.llm.models import (
    LLMBatchExtractionResult,
    LLMExtractionResult,
    LLMProvider,
    PageExtractionResult,
)


class Invoice(BaseModel):
    """Test schema for extraction."""

    total: float
    vendor: str


class TestPageExtractionResult:
    """Tests for PageExtractionResult model."""

    def test_page_result_with_typed_data(self) -> None:
        """Test PageExtractionResult with typed Pydantic model data."""
        invoice = Invoice(total=100.0, vendor="Acme")
        page_result: PageExtractionResult[Invoice] = PageExtractionResult(
            page=0,
            data=invoice,
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )

        assert page_result.page == 0
        assert page_result.data is not None
        assert page_result.data.total == 100.0
        assert page_result.data.vendor == "Acme"
        assert page_result.usage == {"prompt_tokens": 100, "completion_tokens": 50}
        assert page_result.error is None

    def test_page_result_with_dict_data(self) -> None:
        """Test PageExtractionResult with dict data (no schema)."""
        page_result: PageExtractionResult[dict[str, Any]] = PageExtractionResult(
            page=0,
            data={"total": 100.0, "vendor": "Acme"},
        )

        assert page_result.page == 0
        assert page_result.data == {"total": 100.0, "vendor": "Acme"}

    def test_page_result_with_error(self) -> None:
        """Test PageExtractionResult with error (data is None)."""
        page_result: PageExtractionResult[Invoice] = PageExtractionResult(
            page=0,
            data=None,
            error="API rate limit exceeded",
        )

        assert page_result.page == 0
        assert page_result.data is None
        assert page_result.error == "API rate limit exceeded"


class TestLLMBatchExtractionResult:
    """Tests for LLMBatchExtractionResult model."""

    def test_batch_result_with_typed_data(self) -> None:
        """Test batch result with typed Pydantic model data."""
        page_results = [
            PageExtractionResult(
                page=0,
                data=Invoice(total=100.0, vendor="Acme"),
                usage={"prompt_tokens": 100, "completion_tokens": 50},
            ),
            PageExtractionResult(
                page=1,
                data=Invoice(total=200.0, vendor="Corp"),
                usage={"prompt_tokens": 100, "completion_tokens": 50},
            ),
        ]
        result: LLMBatchExtractionResult[Invoice] = LLMBatchExtractionResult(
            results=page_results,
            model="gpt-4o",
            provider=LLMProvider.OPENAI,
            total_usage={"prompt_tokens": 200, "completion_tokens": 100},
        )

        assert len(result.results) == 2
        assert result.results[0].data is not None
        assert result.results[0].data.total == 100.0
        assert result.results[1].data is not None
        assert result.results[1].data.vendor == "Corp"
        assert result.model == "gpt-4o"
        assert result.provider == LLMProvider.OPENAI
        assert result.total_usage == {"prompt_tokens": 200, "completion_tokens": 100}

    def test_batch_result_with_dict_data(self) -> None:
        """Test batch result with dict data (no schema)."""
        page_results = [
            PageExtractionResult(page=0, data={"value": "a"}),
            PageExtractionResult(page=1, data={"value": "b"}),
        ]
        result: LLMBatchExtractionResult[dict[str, Any]] = LLMBatchExtractionResult(
            results=page_results,
            model="gpt-4o",
            provider=LLMProvider.OPENAI,
        )

        assert result.results[0].data == {"value": "a"}
        assert result.results[1].data == {"value": "b"}

    def test_batch_result_iteration(self) -> None:
        """Test that batch result can be iterated via .results."""
        page_results = [
            PageExtractionResult(page=i, data=Invoice(total=float(i), vendor=f"V{i}"))
            for i in range(3)
        ]
        result: LLMBatchExtractionResult[Invoice] = LLMBatchExtractionResult(
            results=page_results,
            model="gpt-4o",
            provider=LLMProvider.OPENAI,
        )

        pages = list(result.results)
        assert len(pages) == 3
        assert pages[0].page == 0
        assert pages[1].data is not None
        assert pages[1].data.total == 1.0
        assert pages[2].data is not None
        assert pages[2].data.vendor == "V2"

    def test_batch_result_indexing(self) -> None:
        """Test that batch result supports indexing via .results."""
        page_results = [
            PageExtractionResult(page=0, data=Invoice(total=100.0, vendor="A")),
            PageExtractionResult(page=1, data=Invoice(total=200.0, vendor="B")),
        ]
        result: LLMBatchExtractionResult[Invoice] = LLMBatchExtractionResult(
            results=page_results,
            model="gpt-4o",
            provider=LLMProvider.OPENAI,
        )

        assert result.results[0].data is not None
        assert result.results[0].data.total == 100.0
        assert result.results[1].data is not None
        assert result.results[1].data.vendor == "B"

    def test_batch_result_different_from_single_result(self) -> None:
        """Test that batch and single results are clearly different types."""
        single: LLMExtractionResult[Invoice] = LLMExtractionResult(
            data=Invoice(total=100.0, vendor="Acme"),
            model="gpt-4o",
            provider=LLMProvider.OPENAI,
        )
        batch: LLMBatchExtractionResult[Invoice] = LLMBatchExtractionResult(
            results=[PageExtractionResult(page=0, data=Invoice(total=100.0, vendor="Acme"))],
            model="gpt-4o",
            provider=LLMProvider.OPENAI,
        )

        # Different types
        assert type(single) is not type(batch)

        # Single has .data, batch has .results
        assert hasattr(single, "data")
        assert hasattr(batch, "results")
        assert not hasattr(single, "results")

    def test_batch_result_mixed_success_and_error(self) -> None:
        """Test batch result with some successful and some failed pages."""
        page_results: list[PageExtractionResult[Invoice]] = [
            PageExtractionResult(
                page=0,
                data=Invoice(total=100.0, vendor="Acme"),
                usage={"prompt_tokens": 100, "completion_tokens": 50},
            ),
            PageExtractionResult(
                page=1,
                data=None,
                error="API error on page 1",
            ),
            PageExtractionResult(
                page=2,
                data=Invoice(total=300.0, vendor="Corp"),
                usage={"prompt_tokens": 100, "completion_tokens": 50},
            ),
        ]
        result: LLMBatchExtractionResult[Invoice] = LLMBatchExtractionResult(
            results=page_results,
            model="gpt-4o",
            provider=LLMProvider.OPENAI,
            total_usage={"prompt_tokens": 200, "completion_tokens": 100},
        )

        # Page 0 succeeded
        assert result.results[0].data is not None
        assert result.results[0].error is None

        # Page 1 failed
        assert result.results[1].data is None
        assert result.results[1].error == "API error on page 1"

        # Page 2 succeeded
        assert result.results[2].data is not None
        data2 = result.results[2].data
        assert data2.total == 300.0
