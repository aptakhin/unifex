"""Tests for Google Document AI extractor and adapter."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from xtra.adapters.google_docai import GoogleDocumentAIAdapter
from xtra.extractors.google_docai import GoogleDocumentAIExtractor
from xtra.models import SourceType


class TestGoogleDocumentAIExtractor:
    def test_get_metadata(self) -> None:
        with (
            patch(
                "xtra.extractors.google_docai.documentai.DocumentProcessorServiceClient"
            ) as mock_client_class,
            patch(
                "xtra.extractors.google_docai.service_account.Credentials.from_service_account_file"
            ) as mock_creds,
            patch("xtra.extractors.google_docai.documentai.RawDocument"),
            patch("xtra.extractors.google_docai.documentai.ProcessRequest"),
        ):
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_creds.return_value = MagicMock()

            mock_document = MagicMock()
            mock_document.pages = []

            mock_result = MagicMock()
            mock_result.document = mock_document

            mock_client.process_document.return_value = mock_result

            with patch("builtins.open", MagicMock()):
                extractor = GoogleDocumentAIExtractor(
                    Path("/tmp/test.pdf"),
                    processor_name="projects/test-project/locations/us/processors/abc123",
                    credentials_path="/tmp/creds.json",
                )

            metadata = extractor.get_metadata()

            assert metadata.source_type == SourceType.GOOGLE_DOCAI
            assert metadata.extra["ocr_engine"] == "google_document_ai"
            assert "processor_name" in metadata.extra

    def test_get_page_count_empty(self) -> None:
        with (
            patch(
                "xtra.extractors.google_docai.documentai.DocumentProcessorServiceClient"
            ) as mock_client_class,
            patch(
                "xtra.extractors.google_docai.service_account.Credentials.from_service_account_file"
            ) as mock_creds,
            patch("xtra.extractors.google_docai.documentai.RawDocument"),
            patch("xtra.extractors.google_docai.documentai.ProcessRequest"),
        ):
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_creds.return_value = MagicMock()

            mock_document = MagicMock()
            mock_document.pages = []

            mock_result = MagicMock()
            mock_result.document = mock_document

            mock_client.process_document.return_value = mock_result

            with patch("builtins.open", MagicMock()):
                extractor = GoogleDocumentAIExtractor(
                    Path("/tmp/test.pdf"),
                    processor_name="projects/test-project/locations/us/processors/abc123",
                    credentials_path="/tmp/creds.json",
                )

            assert extractor.get_page_count() == 0

    def test_get_page_count_with_pages(self) -> None:
        with (
            patch(
                "xtra.extractors.google_docai.documentai.DocumentProcessorServiceClient"
            ) as mock_client_class,
            patch(
                "xtra.extractors.google_docai.service_account.Credentials.from_service_account_file"
            ) as mock_creds,
            patch("xtra.extractors.google_docai.documentai.RawDocument"),
            patch("xtra.extractors.google_docai.documentai.ProcessRequest"),
        ):
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_creds.return_value = MagicMock()

            mock_page1 = MagicMock()
            mock_page2 = MagicMock()
            mock_document = MagicMock()
            mock_document.pages = [mock_page1, mock_page2]

            mock_result = MagicMock()
            mock_result.document = mock_document

            mock_client.process_document.return_value = mock_result

            with patch("builtins.open", MagicMock()):
                extractor = GoogleDocumentAIExtractor(
                    Path("/tmp/test.pdf"),
                    processor_name="projects/test-project/locations/us/processors/abc123",
                    credentials_path="/tmp/creds.json",
                )

            assert extractor.get_page_count() == 2

    def test_extract_page_success(self) -> None:
        with (
            patch(
                "xtra.extractors.google_docai.documentai.DocumentProcessorServiceClient"
            ) as mock_client_class,
            patch(
                "xtra.extractors.google_docai.service_account.Credentials.from_service_account_file"
            ) as mock_creds,
            patch("xtra.extractors.google_docai.documentai.RawDocument"),
            patch("xtra.extractors.google_docai.documentai.ProcessRequest"),
        ):
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_creds.return_value = MagicMock()

            # Create mock token (word)
            mock_vertex1 = MagicMock()
            mock_vertex1.x = 10.0
            mock_vertex1.y = 20.0
            mock_vertex2 = MagicMock()
            mock_vertex2.x = 60.0
            mock_vertex2.y = 20.0
            mock_vertex3 = MagicMock()
            mock_vertex3.x = 60.0
            mock_vertex3.y = 40.0
            mock_vertex4 = MagicMock()
            mock_vertex4.x = 10.0
            mock_vertex4.y = 40.0

            mock_bounding_poly = MagicMock()
            mock_bounding_poly.normalized_vertices = [
                mock_vertex1,
                mock_vertex2,
                mock_vertex3,
                mock_vertex4,
            ]

            mock_layout = MagicMock()
            mock_layout.bounding_poly = mock_bounding_poly
            mock_layout.confidence = 0.95
            mock_layout.text_anchor = MagicMock()
            mock_layout.text_anchor.text_segments = [MagicMock(start_index=0, end_index=5)]

            mock_token = MagicMock()
            mock_token.layout = mock_layout

            # Create mock page
            mock_dimension = MagicMock()
            mock_dimension.width = 612.0
            mock_dimension.height = 792.0

            mock_page = MagicMock()
            mock_page.dimension = mock_dimension
            mock_page.tokens = [mock_token]

            mock_document = MagicMock()
            mock_document.pages = [mock_page]
            mock_document.text = "Hello"

            mock_result = MagicMock()
            mock_result.document = mock_document

            mock_client.process_document.return_value = mock_result

            with patch("builtins.open", MagicMock()):
                extractor = GoogleDocumentAIExtractor(
                    Path("/tmp/test.pdf"),
                    processor_name="projects/test-project/locations/us/processors/abc123",
                    credentials_path="/tmp/creds.json",
                )

            result = extractor.extract_page(0)

            assert result.success
            assert result.page.page == 0
            assert result.page.width == 612.0
            assert result.page.height == 792.0
            assert len(result.page.texts) == 1
            assert result.page.texts[0].text == "Hello"
            assert result.page.texts[0].confidence == 0.95

    def test_extract_page_out_of_range(self) -> None:
        with (
            patch(
                "xtra.extractors.google_docai.documentai.DocumentProcessorServiceClient"
            ) as mock_client_class,
            patch(
                "xtra.extractors.google_docai.service_account.Credentials.from_service_account_file"
            ) as mock_creds,
            patch("xtra.extractors.google_docai.documentai.RawDocument"),
            patch("xtra.extractors.google_docai.documentai.ProcessRequest"),
        ):
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_creds.return_value = MagicMock()

            mock_document = MagicMock()
            mock_document.pages = []

            mock_result = MagicMock()
            mock_result.document = mock_document

            mock_client.process_document.return_value = mock_result

            with patch("builtins.open", MagicMock()):
                extractor = GoogleDocumentAIExtractor(
                    Path("/tmp/test.pdf"),
                    processor_name="projects/test-project/locations/us/processors/abc123",
                    credentials_path="/tmp/creds.json",
                )

            result = extractor.extract_page(5)

            assert not result.success
            assert result.error is not None
            assert "out of range" in result.error.lower()

    def test_extract_page_no_result(self) -> None:
        with (
            patch(
                "xtra.extractors.google_docai.documentai.DocumentProcessorServiceClient"
            ) as mock_client_class,
            patch(
                "xtra.extractors.google_docai.service_account.Credentials.from_service_account_file"
            ) as mock_creds,
            patch("xtra.extractors.google_docai.documentai.RawDocument"),
            patch("xtra.extractors.google_docai.documentai.ProcessRequest"),
        ):
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_creds.return_value = MagicMock()

            mock_client.process_document.side_effect = ValueError("Failed")

            with patch("builtins.open", MagicMock()):
                extractor = GoogleDocumentAIExtractor(
                    Path("/tmp/test.pdf"),
                    processor_name="projects/test-project/locations/us/processors/abc123",
                    credentials_path="/tmp/creds.json",
                )

            result = extractor.extract_page(0)

            assert not result.success
            assert result.error is not None

    def test_close(self) -> None:
        with (
            patch(
                "xtra.extractors.google_docai.documentai.DocumentProcessorServiceClient"
            ) as mock_client_class,
            patch(
                "xtra.extractors.google_docai.service_account.Credentials.from_service_account_file"
            ) as mock_creds,
            patch("xtra.extractors.google_docai.documentai.RawDocument"),
            patch("xtra.extractors.google_docai.documentai.ProcessRequest"),
        ):
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_creds.return_value = MagicMock()

            mock_document = MagicMock()
            mock_document.pages = []

            mock_result = MagicMock()
            mock_result.document = mock_document

            mock_client.process_document.return_value = mock_result

            with patch("builtins.open", MagicMock()):
                extractor = GoogleDocumentAIExtractor(
                    Path("/tmp/test.pdf"),
                    processor_name="projects/test-project/locations/us/processors/abc123",
                    credentials_path="/tmp/creds.json",
                )

            extractor.close()
            # Client transport should be closed
            mock_client.transport.close.assert_called_once()


class TestGoogleDocumentAIAdapter:
    def test_normalized_vertices_to_bbox_and_rotation_horizontal(self) -> None:
        # Horizontal text with normalized vertices
        mock_vertex1 = MagicMock(x=0.1, y=0.2)
        mock_vertex2 = MagicMock(x=0.5, y=0.2)
        mock_vertex3 = MagicMock(x=0.5, y=0.3)
        mock_vertex4 = MagicMock(x=0.1, y=0.3)
        vertices = [mock_vertex1, mock_vertex2, mock_vertex3, mock_vertex4]

        bbox, rotation = GoogleDocumentAIAdapter._vertices_to_bbox_and_rotation(
            vertices, page_width=612.0, page_height=792.0
        )

        # Check denormalized coordinates
        assert abs(bbox.x0 - 61.2) < 0.1  # 0.1 * 612
        assert abs(bbox.y0 - 158.4) < 0.1  # 0.2 * 792
        assert abs(bbox.x1 - 306.0) < 0.1  # 0.5 * 612
        assert abs(bbox.y1 - 237.6) < 0.1  # 0.3 * 792
        assert rotation == 0.0

    def test_vertices_to_bbox_short_vertices(self) -> None:
        # Too few vertices
        vertices = [MagicMock(x=0.1, y=0.1)]
        bbox, rotation = GoogleDocumentAIAdapter._vertices_to_bbox_and_rotation(
            vertices, page_width=612.0, page_height=792.0
        )

        assert bbox.x0 == 0
        assert bbox.y0 == 0
        assert bbox.x1 == 0
        assert bbox.y1 == 0
        assert rotation == 0.0

    def test_page_count_with_none_result(self) -> None:
        adapter = GoogleDocumentAIAdapter(None, "test-processor")
        assert adapter.page_count == 0

    def test_page_count_with_result(self) -> None:
        mock_document = MagicMock()
        mock_document.pages = [MagicMock(), MagicMock()]
        adapter = GoogleDocumentAIAdapter(mock_document, "test-processor")
        assert adapter.page_count == 2

    def test_get_metadata_with_none_result(self) -> None:
        adapter = GoogleDocumentAIAdapter(None, "test-processor")
        metadata = adapter.get_metadata()

        assert metadata.source_type == SourceType.GOOGLE_DOCAI
        assert metadata.extra["processor_name"] == "test-processor"
        assert metadata.extra["ocr_engine"] == "google_document_ai"

    def test_convert_page_raises_on_none_result(self) -> None:
        adapter = GoogleDocumentAIAdapter(None, "test-processor")
        try:
            adapter.convert_page(0)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "No analysis result" in str(e)

    def test_convert_page_raises_on_out_of_range(self) -> None:
        mock_document = MagicMock()
        mock_document.pages = []
        adapter = GoogleDocumentAIAdapter(mock_document, "test-processor")
        try:
            adapter.convert_page(0)
            assert False, "Expected IndexError"
        except IndexError as e:
            assert "out of range" in str(e)

    def test_convert_page_to_blocks_empty_tokens(self) -> None:
        mock_document = MagicMock()
        mock_document.text = ""

        adapter = GoogleDocumentAIAdapter(mock_document, "test-processor")

        mock_page = MagicMock()
        mock_page.tokens = []

        blocks = adapter._convert_page_to_blocks(mock_page)
        assert blocks == []

    def test_convert_page_to_blocks_skip_invalid_tokens(self) -> None:
        mock_document = MagicMock()
        mock_document.text = "Hello World"

        adapter = GoogleDocumentAIAdapter(mock_document, "test-processor")

        # Token with no layout
        mock_token1 = MagicMock()
        mock_token1.layout = None

        # Token with no bounding_poly
        mock_token2 = MagicMock()
        mock_token2.layout = MagicMock()
        mock_token2.layout.bounding_poly = None

        # Valid token
        mock_vertex1 = MagicMock(x=0.0, y=0.0)
        mock_vertex2 = MagicMock(x=0.1, y=0.0)
        mock_vertex3 = MagicMock(x=0.1, y=0.1)
        mock_vertex4 = MagicMock(x=0.0, y=0.1)

        mock_token3 = MagicMock()
        mock_token3.layout = MagicMock()
        mock_token3.layout.bounding_poly = MagicMock()
        mock_token3.layout.bounding_poly.normalized_vertices = [
            mock_vertex1,
            mock_vertex2,
            mock_vertex3,
            mock_vertex4,
        ]
        mock_token3.layout.confidence = 0.9
        mock_token3.layout.text_anchor = MagicMock()
        mock_token3.layout.text_anchor.text_segments = [MagicMock(start_index=6, end_index=11)]

        mock_page = MagicMock()
        mock_page.tokens = [mock_token1, mock_token2, mock_token3]
        mock_page.dimension = MagicMock(width=612.0, height=792.0)

        blocks = adapter._convert_page_to_blocks(mock_page)
        assert len(blocks) == 1
        assert blocks[0].text == "World"
