"""LLM-based structured data extraction."""

from unifex.llm.models import (
    LLMExtractionResult,
    LLMProvider,
    PageExtractionConfig,
)

__all__ = [
    "LLMExtractionResult",
    "LLMProvider",
    "PageExtractionConfig",
    "extract_structured",
    "extract_structured_async",
]


def __getattr__(name: str):
    """Lazy load factory functions to avoid circular imports."""
    if name == "extract_structured":
        from unifex.llm_factory import extract_structured

        return extract_structured
    if name == "extract_structured_async":
        from unifex.llm_factory import extract_structured_async

        return extract_structured_async
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
