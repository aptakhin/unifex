# API Reference

## Factory Function

### create_extractor

The main entry point for creating extractors.

::: unifex.create_extractor

## Extractor Types

### ExtractorType

Enum for available extractor types.

::: unifex.ExtractorType

## Coordinate Units

### CoordinateUnit

Enum for coordinate output units.

::: unifex.CoordinateUnit

## Executor Types

### ExecutorType

Enum for parallel execution modes.

::: unifex.base.ExecutorType

## LLM Extraction

### extract_structured

Extract structured data from a document using an LLM (single request).

::: unifex.llm_factory.extract_structured

### extract_structured_async

Async version of extract_structured.

::: unifex.llm_factory.extract_structured_async

### extract_structured_parallel

Extract structured data in parallel (one page per request).

::: unifex.llm_factory.extract_structured_parallel

### extract_structured_parallel_async

Async parallel extraction.

::: unifex.llm_factory.extract_structured_parallel_async
