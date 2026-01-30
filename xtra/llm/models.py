"""Models for LLM-based extraction."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel


T = TypeVar("T")


class LLMProvider(StrEnum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE_OPENAI = "azure-openai"


class LLMExtractionResult(BaseModel, Generic[T]):
    """Result of LLM extraction."""

    model_config = {"arbitrary_types_allowed": True}

    data: T
    model: str
    provider: LLMProvider
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Any] = None


class PageExtractionConfig(BaseModel):
    """Configuration for page selection."""

    page_numbers: Optional[List[int]] = None  # None = all pages
    combine_pages: bool = True  # Combine all pages into single extraction
