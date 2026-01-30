"""Tests for Anthropic LLM extractor."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import BaseModel

from xtra.llm.models import LLMProvider

TEST_DATA_DIR = Path(__file__).parent.parent / "data"


class SimpleSchema(BaseModel):
    """Simple schema for testing."""

    text: str


class TestExtractAnthropic:
    """Tests for extract_anthropic function."""

    def test_with_schema(self) -> None:
        """Test extraction with a schema."""
        mock_anthropic_class = MagicMock()
        mock_anthropic_instance = MagicMock()
        mock_anthropic_class.return_value = mock_anthropic_instance

        mock_instructor = MagicMock()
        mock_instructor_client = MagicMock()
        mock_instructor.from_anthropic.return_value = mock_instructor_client

        mock_response = SimpleSchema(text="extracted text")
        mock_instructor_client.messages.create.return_value = mock_response

        with (
            patch.dict(
                sys.modules,
                {"anthropic": MagicMock(Anthropic=mock_anthropic_class)},
            ),
            patch.dict(sys.modules, {"instructor": mock_instructor}),
        ):
            from xtra.llm.extractors.anthropic import extract_anthropic

            result = extract_anthropic(
                TEST_DATA_DIR / "test_image.png",
                model="claude-3-5-sonnet",
                schema=SimpleSchema,
                api_key="test-key",
            )

        assert result.data == mock_response
        assert result.model == "claude-3-5-sonnet"
        assert result.provider == LLMProvider.ANTHROPIC

    def test_without_schema_json_mode(self) -> None:
        """Test extraction without schema returns dict."""
        mock_anthropic_class = MagicMock()
        mock_anthropic_instance = MagicMock()
        mock_anthropic_class.return_value = mock_anthropic_instance

        # Mock raw response for JSON mode
        mock_content = MagicMock()
        mock_content.text = '{"key": "value"}'
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_anthropic_instance.messages.create.return_value = mock_response

        mock_instructor = MagicMock()

        with (
            patch.dict(
                sys.modules,
                {"anthropic": MagicMock(Anthropic=mock_anthropic_class)},
            ),
            patch.dict(sys.modules, {"instructor": mock_instructor}),
        ):
            from xtra.llm.extractors.anthropic import extract_anthropic

            result = extract_anthropic(
                TEST_DATA_DIR / "test_image.png",
                model="claude-3-5-sonnet",
                schema=None,
                api_key="test-key",
            )

        assert result.data == {"key": "value"}
        assert result.provider == LLMProvider.ANTHROPIC


class TestExtractAnthropicAsync:
    """Tests for extract_anthropic_async function."""

    async def test_async_with_schema(self) -> None:
        """Test async extraction with a schema."""
        mock_async_client_class = MagicMock()
        mock_async_client_instance = MagicMock()
        mock_async_client_class.return_value = mock_async_client_instance

        mock_instructor = MagicMock()
        mock_instructor_client = MagicMock()
        mock_instructor.from_anthropic.return_value = mock_instructor_client

        # Mock async response - must be awaitable
        mock_response = SimpleSchema(text="async extracted")
        mock_instructor_client.messages.create = AsyncMock(return_value=mock_response)

        with (
            patch.dict(
                sys.modules,
                {"anthropic": MagicMock(AsyncAnthropic=mock_async_client_class)},
            ),
            patch.dict(sys.modules, {"instructor": mock_instructor}),
        ):
            from xtra.llm.extractors.anthropic import extract_anthropic_async

            result = await extract_anthropic_async(
                TEST_DATA_DIR / "test_image.png",
                model="claude-3-5-sonnet",
                schema=SimpleSchema,
                api_key="test-key",
            )

        assert result.data == mock_response
        assert result.provider == LLMProvider.ANTHROPIC

    async def test_async_without_schema(self) -> None:
        """Test async extraction without schema."""
        mock_async_client_class = MagicMock()
        mock_async_client_instance = MagicMock()
        mock_async_client_class.return_value = mock_async_client_instance

        # Mock async raw response - must be awaitable
        mock_content = MagicMock()
        mock_content.text = '{"async_key": "async_value"}'
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_async_client_instance.messages.create = AsyncMock(return_value=mock_response)

        mock_instructor = MagicMock()

        with (
            patch.dict(
                sys.modules,
                {"anthropic": MagicMock(AsyncAnthropic=mock_async_client_class)},
            ),
            patch.dict(sys.modules, {"instructor": mock_instructor}),
        ):
            from xtra.llm.extractors.anthropic import extract_anthropic_async

            result = await extract_anthropic_async(
                TEST_DATA_DIR / "test_image.png",
                model="claude-3-5-sonnet",
                schema=None,
                api_key="test-key",
            )

        assert result.data == {"async_key": "async_value"}
