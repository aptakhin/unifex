# unifex

A Python library for document text extraction with local and cloud OCR solutions.

**Focus:** Built for tasks like fraud detection where precision matters. We needed a universal tool for both PDF and image processing with best-in-class OCR support through local engines (EasyOCR, Tesseract, PaddleOCR) and cloud services (Azure Document Intelligence, Google Document AI).

📖 **[Documentation](https://aptakhin.name/unifex/)**

## Features

- **Multiple OCR Backends**: Local (EasyOCR, Tesseract, PaddleOCR) and cloud (Azure Document Intelligence, Google Document AI) OCR support
- **PDF Text Extraction**: Native PDF text extraction using pypdfium2
- **LLM Extraction**: Extract structured data using GPT-4o, Claude, Gemini, or OpenAI-compatible APIs
- **Unified Coordinates**: Seamless conversion between POINTS, PIXELS, INCHES, and NORMALIZED coordinate systems
- **Table Extraction**: PDF (tabula), PaddleOCR (PPStructure), and cloud OCR (Azure DI, Google DocAI)
- **Parallel Extraction**: Process multiple pages concurrently with thread or process executors
- **Async Support**: Native async/await API for integration with async applications
- **Unified Extractors**: Each OCR extractor auto-detects file type (PDF vs image) and handles conversion internally
- **Schema Adapters**: Clean separation of external API schemas from internal models
- **Pydantic Models**: Type-safe document representation with pydantic v1/v2 compatibility

## Alternatives

For broader document processing, check out [Docling](https://docling-project.github.io) and [Kreuzberg](https://kreuzberg.dev/).

## Installation

```bash
pip install unifex
```

Or with optional dependencies:

```bash
pip install unifex[pdf]       # PDF text extraction
pip install unifex[easyocr]   # EasyOCR support
pip install unifex[tesseract] # Tesseract OCR support
pip install unifex[azure]     # Azure Document Intelligence
pip install unifex[google]    # Google Document AI
pip install unifex[llm-openai]     # OpenAI/GPT-4 extraction
pip install unifex[llm-anthropic]  # Anthropic/Claude extraction
pip install unifex[all]       # All dependencies
```

## Quick Start

### Factory Interface (Recommended)

The simplest way to use unifex is via the factory interface. Both string paths and `Path` objects are accepted:

```python
from unifex import create_extractor, ExtractorType

# PDF extraction (native text) - string path
with create_extractor("document.pdf", ExtractorType.PDF) as extractor:
    result = extractor.extract()
    doc = result.document  # Access the Document

# EasyOCR for images
with create_extractor("image.png", ExtractorType.EASYOCR, languages=["en"]) as extractor:
    result = extractor.extract()

# EasyOCR for PDFs (auto-converts to images internally)
with create_extractor("scanned.pdf", ExtractorType.EASYOCR, dpi=200) as extractor:
    result = extractor.extract()

# Azure Document Intelligence (credentials from env vars)
with create_extractor("document.pdf", ExtractorType.AZURE_DI) as extractor:
    result = extractor.extract()

# Path objects also work
from pathlib import Path
with create_extractor(Path("document.pdf"), ExtractorType.PDF) as extractor:
    result = extractor.extract()
```

### Example Output

The `extract()` method returns an `ExtractionResult` containing the `Document` and per-page results:

```python
from unifex import create_extractor, ExtractorType

with create_extractor("document.pdf", ExtractorType.PDF) as extractor:
    result = extractor.extract()

# Check extraction status
print(f"Success: {result.success}")  # True if all pages extracted

# Access extracted document
doc = result.document
print(f"Pages: {len(doc.pages)}")  # Pages: 2

for page in doc.pages:
    print(f"Page {page.page + 1} ({page.width:.0f}x{page.height:.0f}):")
    for text in page.texts:
        print(f"  - \"{text.text}\"")
        print(f"    bbox: ({text.bbox.x0:.1f}, {text.bbox.y0:.1f}, {text.bbox.x1:.1f}, {text.bbox.y1:.1f})")

# Handle errors if any
if not result.success:
    for page_num, error in result.errors:
        print(f"Page {page_num} failed: {error}")
```

Output:
```
Pages: 2
Page 1 (595x842):
  - "First page. First text"
    bbox: (48.3, 57.8, 205.4, 74.6)
  - "First page. Second text"
    bbox: (48.0, 81.4, 231.2, 98.6)
  - "First page. Fourth text"
    bbox: (47.8, 120.5, 221.9, 137.4)
Page 2 (595x842):
  - "Second page. Third text"
    bbox: (47.4, 81.1, 236.9, 98.3)
```

For more detailed examples, see the [documentation](https://aptakhin.name/unifex/).

### PDF Text Extraction

```python
from unifex import PdfExtractor

# String paths work directly
with PdfExtractor("document.pdf") as extractor:
    result = extractor.extract()
    for page in result.document.pages:
        for text in page.texts:
            print(text.text)
```

### Language Codes

All OCR extractors use **2-letter ISO 639-1 language codes** (e.g., `"en"`, `"fr"`, `"de"`, `"it"`).
Extractors that require different formats (like Tesseract) convert internally.

### Parallel Extraction

Extract multiple pages concurrently for faster processing:

```python
from unifex import create_extractor, ExtractorType, ExecutorType

# Thread-based parallelism (recommended for most cases)
with create_extractor("large_document.pdf", ExtractorType.EASYOCR) as extractor:
    result = extractor.extract(max_workers=4)  # 4 parallel workers

# Process-based parallelism (for CPU-bound pure Python workloads)
with create_extractor("large_document.pdf", ExtractorType.EASYOCR) as extractor:
    result = extractor.extract(max_workers=4, executor=ExecutorType.PROCESS)

# Extract specific pages in parallel
with create_extractor("document.pdf", ExtractorType.PDF) as extractor:
    result = extractor.extract(pages=[0, 2, 5, 8], max_workers=4)
```

**Executor Types:**

| Executor | Best For | Notes |
|----------|----------|-------|
| `THREAD` (default) | Most OCR use cases | Shared model cache, low overhead, C libraries release GIL |
| `PROCESS` | CPU-bound pure Python | Models duplicated per worker, higher memory usage |

### Async Extraction

For async applications, use the async API:

```python
import asyncio
from unifex import create_extractor, ExtractorType

async def extract_document():
    with create_extractor("document.pdf", ExtractorType.EASYOCR) as extractor:
        result = await extractor.extract_async(max_workers=4)
        return result.document

doc = asyncio.run(extract_document())
```

### OCR Extraction

Local OCR engines (EasyOCR, Tesseract, PaddleOCR) and cloud services (Azure Document Intelligence, Google Document AI). All extractors auto-detect file type (PDF vs image) and handle conversion internally.

```python
from unifex import (
    EasyOcrExtractor, TesseractOcrExtractor, PaddleOcrExtractor,
    AzureDocumentIntelligenceExtractor, GoogleDocumentAIExtractor,
)

# Local OCR (works for both images and PDFs)
with EasyOcrExtractor("scanned.pdf", languages=["en"], dpi=200) as extractor:
    result = extractor.extract()

# Tesseract (requires system install: brew install tesseract)
with TesseractOcrExtractor("image.png", languages=["en"]) as extractor:
    result = extractor.extract()

# PaddleOCR (excellent for Chinese)
with PaddleOcrExtractor("chinese_doc.png", lang="ch") as extractor:
    result = extractor.extract()

# Cloud: Azure Document Intelligence
with AzureDocumentIntelligenceExtractor(
    "document.pdf",
    endpoint="https://your-resource.cognitiveservices.azure.com",
    key="your-api-key",
) as extractor:
    result = extractor.extract()

# Cloud: Google Document AI
with GoogleDocumentAIExtractor(
    "document.pdf",
    processor_name="projects/your-project/locations/us/processors/id",
    credentials_path="/path/to/credentials.json",
) as extractor:
    result = extractor.extract()
```

### LLM Extraction

Extract structured data using vision-capable LLMs (OpenAI, Anthropic, Google, Azure OpenAI). Supports custom prompts, Pydantic schemas, parallel extraction, async API, and OpenAI-compatible endpoints (vLLM, Ollama).

```python
from pydantic import BaseModel
from unifex.llm import extract_structured, extract_structured_async

class Invoice(BaseModel):
    invoice_number: str
    date: str
    total: float

# Basic extraction with Pydantic schema
result = extract_structured("invoice.pdf", model="openai/gpt-4o", schema=Invoice)
invoice: Invoice = result.data

# With custom prompt and parallel workers
result = extract_structured(
    "large_doc.pdf",
    model="anthropic/claude-sonnet-4-20250514",
    prompt="Extract invoice details",
    max_workers=4,  # Process pages in parallel
)

# OpenAI-compatible APIs (vLLM, Ollama) with custom headers
result = extract_structured(
    "document.pdf",
    model="openai/llava",
    base_url="http://localhost:11434/v1",
    headers={"X-Custom-Auth": "token"},
)

# Async API
result = await extract_structured_async("doc.pdf", model="openai/gpt-4o", max_workers=4)
```

## CLI Usage

```bash
# OCR extractors: pdf, easyocr, tesseract, paddle, azure-di, google-docai
uv run python -m unifex.cli document.pdf --extractor easyocr --lang en

# Parallel extraction with process executor
uv run python -m unifex.cli large_doc.pdf --extractor easyocr --max-workers 4 --executor process

# Cloud OCR (credentials via CLI or env vars)
uv run python -m unifex.cli document.pdf --extractor azure-di \
    --azure-endpoint https://your-resource.cognitiveservices.azure.com --azure-key your-key

# LLM extraction with parallel workers and custom endpoint
uv run python -m unifex.cli document.pdf --llm openai/gpt-4o --max-workers 4 \
    --llm-base-url https://your-proxy.com/v1 --llm-header "X-Auth=token"

# JSON output, specific pages
uv run python -m unifex.cli document.pdf --extractor pdf --pages 0,1,2 --json
```

## Environment Variables

Cloud extractors and LLM providers support configuration via environment variables:

**OCR Extractors:**

| Variable | Description |
|----------|-------------|
| `UNIFEX_AZURE_DI_ENDPOINT` | Azure Document Intelligence endpoint URL |
| `UNIFEX_AZURE_DI_KEY` | Azure Document Intelligence API key |
| `UNIFEX_AZURE_DI_MODEL` | Azure model ID (default: `prebuilt-read`) |
| `UNIFEX_GOOGLE_DOCAI_PROCESSOR_NAME` | Google Document AI processor name |
| `UNIFEX_GOOGLE_DOCAI_CREDENTIALS_PATH` | Path to Google service account JSON |

**LLM Providers:**

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `GOOGLE_API_KEY` | Google AI API key |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_VERSION` | Azure OpenAI API version (default: `2024-02-15-preview`) |

## Development

### Setup

```bash
# Install dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run fast tests only (unit tests, <0.5s per test)
uv run pytest tests/base tests/ocr

# Run integration tests only (slow, load ML models)
uv run pytest tests/integration

# Run with coverage
uv run pytest --cov=unifex --cov-report=term-missing
```

### Test Structure

```
tests/
├── base/           # Fast unit tests (<0.5s each) - run in pre-commit
├── ocr/            # OCR adapter unit tests (mocked) - run in pre-commit
├── llm/            # LLM unit tests (mocked) - run in pre-commit
└── integration/    # Slow tests - NOT in pre-commit
    ├── ocr/        # OCR integration tests (load real ML models)
    └── llm/        # LLM integration tests (call real APIs)
```

**Pre-commit runs:** `tests/base`, `tests/ocr`, and `tests/llm` with 0.5s timeout per test.

**CI runs:** All tests including integration tests.

### Integration Tests

Integration tests load real ML models and call real services. They are in `tests/integration/`.

**Local extractors** (no credentials required):
- `PdfExtractor` - Tests PDF text extraction
- `EasyOcrExtractor` - Tests image and PDF OCR with EasyOCR
- `TesseractOcrExtractor` - Tests image and PDF OCR with Tesseract (requires Tesseract installed)
- `PaddleOcrExtractor` - Tests image and PDF OCR with PaddleOCR

**Cloud extractors** (require credentials):
- `AzureDocumentIntelligenceExtractor` - Tests Azure Document Intelligence
- `GoogleDocumentAIExtractor` - Tests Google Document AI

#### Azure Credentials Setup

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your Azure Document Intelligence credentials:
   ```
   UNIFEX_AZURE_DI_ENDPOINT=https://your-resource.cognitiveservices.azure.com
   UNIFEX_AZURE_DI_KEY=your-api-key
   ```

3. Load environment variables and run tests:
   ```bash
   set -a; source .env; set +a;
   uv run pytest tests/test_integration.py -v
   ```

Azure integration tests are automatically skipped if credentials are not configured.

#### Google Document AI Credentials Setup

1. Create a Google Cloud project and enable the Document AI API
2. Create a Document AI processor in the Google Cloud Console
3. Create a service account with Document AI permissions
4. Download the service account JSON key file

5. Edit `.env` with your Google Document AI credentials:
   ```
   UNIFEX_GOOGLE_DOCAI_PROCESSOR_NAME=projects/your-project/locations/us/processors/your-processor-id
   UNIFEX_GOOGLE_DOCAI_CREDENTIALS_PATH=/path/to/your/service-account.json
   ```

Google Document AI integration tests are automatically skipped if credentials are not configured.

### Documentation

Build and serve the documentation locally:

```bash
# Serve docs with live reload
uv run mkdocs serve

# Build static site
uv run mkdocs build
```

Open http://localhost:8000 to view the documentation.

### Pre-commit Checks

The pre-commit hook runs automatically on `git commit`. To run manually:

```bash
uv run pre-commit run --all-files
```

This runs:
- `ruff format` - Code formatting
- `ruff check --fix` - Linting with auto-fix
- `ty check` - Type checking
- `pytest` - Test suite

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.

## Future plans

- Detecting language helper
- Performance measurement
