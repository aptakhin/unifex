"""Tests for OpenAI LLM extractor."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import BaseModel

from xtra.llm.models import LLMProvider

TEST_DATA_DIR = Path(__file__).parent.parent / "data"


class SimpleSchema(BaseModel):
    """Simple schema for testing."""

    text: str


class TestExtractOpenAI:
    """Tests for extract_openai function."""

    def test_with_schema(self) -> None:
        """Test extraction with a schema."""
        mock_openai_class = MagicMock()
        mock_openai_instance = MagicMock()
        mock_openai_class.return_value = mock_openai_instance

        mock_instructor = MagicMock()
        mock_instructor_client = MagicMock()
        mock_instructor.from_openai.return_value = mock_instructor_client

        mock_response = SimpleSchema(text="extracted text")
        mock_instructor_client.chat.completions.create.return_value = mock_response

        with (
            patch.dict(
                sys.modules,
                {"openai": MagicMock(OpenAI=mock_openai_class)},
            ),
            patch.dict(sys.modules, {"instructor": mock_instructor}),
        ):
            from xtra.llm.extractors.openai import extract_openai

            result = extract_openai(
                TEST_DATA_DIR / "test_image.png",
                model="gpt-4o",
                schema=SimpleSchema,
                api_key="test-key",
            )

        assert result.data == mock_response
        assert result.model == "gpt-4o"
        assert result.provider == LLMProvider.OPENAI

    def test_without_schema_json_mode(self) -> None:
        """Test extraction without schema returns dict."""
        mock_openai_class = MagicMock()
        mock_openai_instance = MagicMock()
        mock_openai_class.return_value = mock_openai_instance

        # Mock raw response for JSON mode
        mock_message = MagicMock()
        mock_message.content = '{"key": "value"}'
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_openai_instance.chat.completions.create.return_value = mock_response

        mock_instructor = MagicMock()

        with (
            patch.dict(
                sys.modules,
                {"openai": MagicMock(OpenAI=mock_openai_class)},
            ),
            patch.dict(sys.modules, {"instructor": mock_instructor}),
        ):
            from xtra.llm.extractors.openai import extract_openai

            result = extract_openai(
                TEST_DATA_DIR / "test_image.png",
                model="gpt-4o",
                schema=None,
                api_key="test-key",
            )

        assert result.data == {"key": "value"}
        assert result.provider == LLMProvider.OPENAI

    def test_with_custom_base_url(self) -> None:
        """Test extraction with custom base_url."""
        mock_openai_class = MagicMock()
        mock_openai_instance = MagicMock()
        mock_openai_class.return_value = mock_openai_instance

        mock_instructor = MagicMock()
        mock_instructor_client = MagicMock()
        mock_instructor.from_openai.return_value = mock_instructor_client

        mock_response = SimpleSchema(text="custom endpoint")
        mock_instructor_client.chat.completions.create.return_value = mock_response

        with (
            patch.dict(
                sys.modules,
                {"openai": MagicMock(OpenAI=mock_openai_class)},
            ),
            patch.dict(sys.modules, {"instructor": mock_instructor}),
        ):
            from xtra.llm.extractors.openai import extract_openai

            result = extract_openai(
                TEST_DATA_DIR / "test_image.png",
                model="gpt-4o",
                schema=SimpleSchema,
                base_url="https://custom.api.com",
                headers={"X-Custom": "header"},
            )

        assert result.data == mock_response
        # Verify OpenAI was called with custom params
        call_kwargs = mock_openai_class.call_args.kwargs
        assert call_kwargs["base_url"] == "https://custom.api.com"
        assert call_kwargs["default_headers"] == {"X-Custom": "header"}


class TestExtractOpenAIAsync:
    """Tests for extract_openai_async function."""

    async def test_async_with_schema(self) -> None:
        """Test async extraction with a schema."""
        mock_async_client_class = MagicMock()
        mock_async_client_instance = MagicMock()
        mock_async_client_class.return_value = mock_async_client_instance

        mock_instructor = MagicMock()
        mock_instructor_client = MagicMock()
        mock_instructor.from_openai.return_value = mock_instructor_client

        # Mock async response - must be awaitable
        mock_response = SimpleSchema(text="async extracted")
        mock_instructor_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with (
            patch.dict(
                sys.modules,
                {"openai": MagicMock(AsyncOpenAI=mock_async_client_class)},
            ),
            patch.dict(sys.modules, {"instructor": mock_instructor}),
        ):
            from xtra.llm.extractors.openai import extract_openai_async

            result = await extract_openai_async(
                TEST_DATA_DIR / "test_image.png",
                model="gpt-4o",
                schema=SimpleSchema,
                api_key="test-key",
            )

        assert result.data == mock_response
        assert result.provider == LLMProvider.OPENAI

    async def test_async_without_schema(self) -> None:
        """Test async extraction without schema."""
        mock_async_client_class = MagicMock()
        mock_async_client_instance = MagicMock()
        mock_async_client_class.return_value = mock_async_client_instance

        # Mock async raw response - must be awaitable
        mock_message = MagicMock()
        mock_message.content = '{"async_key": "async_value"}'
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_async_client_instance.chat.completions.create = AsyncMock(return_value=mock_response)

        mock_instructor = MagicMock()

        with (
            patch.dict(
                sys.modules,
                {"openai": MagicMock(AsyncOpenAI=mock_async_client_class)},
            ),
            patch.dict(sys.modules, {"instructor": mock_instructor}),
        ):
            from xtra.llm.extractors.openai import extract_openai_async

            result = await extract_openai_async(
                TEST_DATA_DIR / "test_image.png",
                model="gpt-4o",
                schema=None,
                api_key="test-key",
            )

        assert result.data == {"async_key": "async_value"}
