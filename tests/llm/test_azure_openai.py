"""Tests for Azure OpenAI LLM extractor."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from xtra.llm.models import LLMProvider

TEST_DATA_DIR = Path(__file__).parent.parent / "data"


class SimpleSchema(BaseModel):
    """Simple schema for testing."""

    text: str


class TestExtractAzureOpenAI:
    """Tests for extract_azure_openai function."""

    def test_raises_without_endpoint(self) -> None:
        """Test that missing endpoint raises ValueError."""
        from xtra.llm.extractors.azure_openai import extract_azure_openai

        with pytest.raises(ValueError, match="endpoint required"):
            extract_azure_openai(
                TEST_DATA_DIR / "test_image.png",
                model="my-deployment",
                endpoint=None,
            )

    def test_with_schema(self) -> None:
        """Test extraction with a schema."""
        # Create mocks
        mock_azure_client_class = MagicMock()
        mock_azure_client_instance = MagicMock()
        mock_azure_client_class.return_value = mock_azure_client_instance

        mock_instructor = MagicMock()
        mock_instructor_client = MagicMock()
        mock_instructor.from_openai.return_value = mock_instructor_client

        # Mock response
        mock_response = SimpleSchema(text="extracted text")
        mock_instructor_client.chat.completions.create.return_value = mock_response

        with (
            patch.dict(
                sys.modules,
                {"openai": MagicMock(AzureOpenAI=mock_azure_client_class)},
            ),
            patch.dict(sys.modules, {"instructor": mock_instructor}),
        ):
            # Re-import to get the patched version
            from xtra.llm.extractors.azure_openai import extract_azure_openai

            result = extract_azure_openai(
                TEST_DATA_DIR / "test_image.png",
                model="my-deployment",
                schema=SimpleSchema,
                endpoint="https://test.openai.azure.com",
                api_key="test-key",
            )

        assert result.data == mock_response
        assert result.model == "my-deployment"
        assert result.provider == LLMProvider.AZURE_OPENAI

    def test_without_schema_json_mode(self) -> None:
        """Test extraction without schema returns dict."""
        mock_azure_client_class = MagicMock()
        mock_azure_client_instance = MagicMock()
        mock_azure_client_class.return_value = mock_azure_client_instance

        # Mock raw response for JSON mode
        mock_message = MagicMock()
        mock_message.content = '{"key": "value"}'
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_azure_client_instance.chat.completions.create.return_value = mock_response

        mock_instructor = MagicMock()

        with (
            patch.dict(
                sys.modules,
                {"openai": MagicMock(AzureOpenAI=mock_azure_client_class)},
            ),
            patch.dict(sys.modules, {"instructor": mock_instructor}),
        ):
            from xtra.llm.extractors.azure_openai import extract_azure_openai

            result = extract_azure_openai(
                TEST_DATA_DIR / "test_image.png",
                model="my-deployment",
                schema=None,
                endpoint="https://test.openai.azure.com",
            )

        assert result.data == {"key": "value"}
        assert result.provider == LLMProvider.AZURE_OPENAI


class TestExtractAzureOpenAIAsync:
    """Tests for extract_azure_openai_async function."""

    def test_raises_without_endpoint(self) -> None:
        """Test that missing endpoint raises ValueError."""
        import asyncio

        from xtra.llm.extractors.azure_openai import extract_azure_openai_async

        with pytest.raises(ValueError, match="endpoint required"):
            asyncio.run(
                extract_azure_openai_async(
                    TEST_DATA_DIR / "test_image.png",
                    model="my-deployment",
                    endpoint=None,
                )
            )

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
                {"openai": MagicMock(AsyncAzureOpenAI=mock_async_client_class)},
            ),
            patch.dict(sys.modules, {"instructor": mock_instructor}),
        ):
            from xtra.llm.extractors.azure_openai import extract_azure_openai_async

            result = await extract_azure_openai_async(
                TEST_DATA_DIR / "test_image.png",
                model="my-deployment",
                schema=SimpleSchema,
                endpoint="https://test.openai.azure.com",
            )

        assert result.data == mock_response
        assert result.provider == LLMProvider.AZURE_OPENAI

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
                {"openai": MagicMock(AsyncAzureOpenAI=mock_async_client_class)},
            ),
            patch.dict(sys.modules, {"instructor": mock_instructor}),
        ):
            from xtra.llm.extractors.azure_openai import extract_azure_openai_async

            result = await extract_azure_openai_async(
                TEST_DATA_DIR / "test_image.png",
                model="my-deployment",
                schema=None,
                endpoint="https://test.openai.azure.com",
            )

        assert result.data == {"async_key": "async_value"}
