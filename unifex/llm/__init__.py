"""LLM-based structured data extraction."""

from unifex.llm.models import (
    LLMBatchExtractionResult,
    LLMExtractionResult,
    LLMProvider,
    PageExtractionConfig,
    PageExtractionResult,
)

__all__ = [
    "LLMBatchExtractionResult",
    "LLMExtractionResult",
    "LLMProvider",
    "PageExtractionConfig",
    "PageExtractionResult",
    "extract_structured",
    "extract_structured_async",
    "extract_structured_parallel",
    "extract_structured_parallel_async",
]


def __getattr__(name: str):
    """Lazy load factory functions to avoid circular imports."""
    if name == "extract_structured":
        from unifex.llm_factory import extract_structured

        return extract_structured
    if name == "extract_structured_async":
        from unifex.llm_factory import extract_structured_async

        return extract_structured_async
    if name == "extract_structured_parallel":
        from unifex.llm_factory import extract_structured_parallel

        return extract_structured_parallel
    if name == "extract_structured_parallel_async":
        from unifex.llm_factory import extract_structured_parallel_async

        return extract_structured_parallel_async
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
