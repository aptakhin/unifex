# Models Reference

## Core Models

### Document

The top-level container for extracted content.

::: unifex.base.Document

### Page

Represents a single page with text blocks and tables.

::: unifex.base.Page

### TextBlock

A text element with bounding box and confidence.

::: unifex.base.TextBlock

### BBox

Bounding box coordinates.

::: unifex.base.BBox

### Table

Extracted table with rows and cells.

::: unifex.base.Table

### TableCell

Individual cell within a table.

::: unifex.base.TableCell

## Result Models

### ExtractionResult

Result of document extraction.

::: unifex.base.ExtractionResult

### PageExtractionResult

Result of single page extraction.

::: unifex.base.PageExtractionResult

## Metadata

### ExtractorMetadata

Metadata about the extractor used.

::: unifex.base.ExtractorMetadata

## LLM Models

### LLMExtractionResult

Result of single LLM extraction request.

::: unifex.llm.models.LLMExtractionResult

### LLMBatchExtractionResult

Result of parallel LLM extraction across multiple pages.

::: unifex.llm.models.LLMBatchExtractionResult

### PageExtractionResult (LLM)

Result for a single page in batch LLM processing.

::: unifex.llm.models.PageExtractionResult

### LLMProvider

Supported LLM providers.

::: unifex.llm.models.LLMProvider
