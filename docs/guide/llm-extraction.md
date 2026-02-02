# LLM Extraction

Extract structured data from documents using vision-capable LLMs.

## Supported Providers

- **OpenAI**: GPT-4o, GPT-4o-mini
- **Anthropic**: Claude Sonnet, Claude Opus
- **Google**: Gemini Pro, Gemini Flash
- **Azure OpenAI**: GPT-4o via Azure
- **OpenAI-Compatible**: vLLM, Ollama, and other compatible APIs

## Basic Usage

### Free-form Extraction

<!-- skip: next -->
```python
from unifex.llm import extract_structured

result = extract_structured(
    "document.pdf",
    model="openai/gpt-4o",
)
print(result.data)
```

### With Custom Prompt

<!-- skip: next -->
```python
from unifex.llm import extract_structured

result = extract_structured(
    "image.png",
    model="anthropic/claude-sonnet-4-20250514",
    prompt="Extract all visible text from this image",
)
```

## Structured Extraction with Pydantic

Define a Pydantic model for type-safe structured output:

<!-- skip: next -->
```python
from pydantic import BaseModel
from unifex.llm import extract_structured

class DocumentContent(BaseModel):
    title: str | None
    paragraphs: list[str]

result = extract_structured(
    "document.pdf",
    model="openai/gpt-4o",
    schema=DocumentContent,
)
content: DocumentContent = result.data
print(f"Found {len(content.paragraphs)} paragraphs")
```

## OpenAI-Compatible APIs

Use custom base URLs for self-hosted or alternative APIs:

<!-- skip: next -->
```python
from unifex.llm import extract_structured

# vLLM server
result = extract_structured(
    "document.pdf",
    model="openai/meta-llama/Llama-3.2-90B-Vision-Instruct",
    base_url="http://localhost:8000/v1",
)

# Ollama
result = extract_structured(
    "image.png",
    model="openai/llava",
    base_url="http://localhost:11434/v1",
)

# With custom headers
result = extract_structured(
    "document.pdf",
    model="openai/gpt-4o",
    base_url="https://your-proxy.com/v1",
    headers={"X-Custom-Auth": "your-token"},
)
```

## Parallel Extraction

Process multiple pages in parallel for faster extraction using
[`extract_structured_parallel`][unifex.llm_factory.extract_structured_parallel]:

<!-- skip: next -->
```python
from unifex.llm import extract_structured, extract_structured_parallel

# Sequential: all pages sent in one request
result = extract_structured("document.pdf", model="openai/gpt-4o")
# result.data is the extracted data

# Parallel: each page processed separately with 4 concurrent workers
batch_result = extract_structured_parallel(
    "document.pdf",
    model="openai/gpt-4o",
    max_workers=4,
)
# batch_result.results is a list of PageExtractionResult
# batch_result.total_usage contains aggregated token usage

for page_result in batch_result.results:
    if page_result.error:
        print(f"Page {page_result.page} failed: {page_result.error}")
    else:
        print(f"Page {page_result.page}: {page_result.data}")
```

Results are guaranteed to be in the same order as input pages.

## Async API

<!-- skip: next -->
```python
import asyncio
from unifex.llm import extract_structured_async, extract_structured_parallel_async

async def extract():
    # Single request
    result = await extract_structured_async(
        "document.pdf",
        model="openai/gpt-4o",
    )
    return result.data

async def extract_parallel():
    # Parallel requests
    batch_result = await extract_structured_parallel_async(
        "document.pdf",
        model="openai/gpt-4o",
        max_workers=4,
    )
    return batch_result.results

data = asyncio.run(extract())
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `GOOGLE_API_KEY` | Google AI API key |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_VERSION` | Azure OpenAI API version |
