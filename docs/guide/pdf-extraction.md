# PDF Extraction

Native PDF text extraction using pypdfium2, with optional table extraction via tabula-py.

## Basic Usage

```python
from xtra import PdfExtractor

with PdfExtractor("document.pdf") as extractor:
    result = extractor.extract()
    for page in result.document.pages:
        for text in page.texts[:2]:  # Show first 2 texts
            print(text.text)
```

## Using the Factory

```python
from xtra import create_extractor, ExtractorType

with create_extractor("document.pdf", ExtractorType.PDF) as extractor:
    result = extractor.extract()
    print(f"Extracted {len(result.document.pages)} pages")
```

## Extracting Specific Pages

```python
from xtra import PdfExtractor

with PdfExtractor("document.pdf") as extractor:
    # Extract only page 0
    result = extractor.extract(pages=[0])
    print(f"Extracted page: {result.document.pages[0].page}")
```

## Parallel Extraction

Extract multiple pages concurrently for faster processing:

```python
from xtra import PdfExtractor

with PdfExtractor("document.pdf") as extractor:
    # Use 2 parallel workers
    result = extractor.extract(max_workers=2)
    print(f"Extracted {len(result.document.pages)} pages in parallel")
```

See [Parallel Processing](parallel-processing.md) for more details.

## Coordinate Units

Control the output coordinate system for bounding boxes:

```python
from xtra import PdfExtractor, CoordinateUnit

# Points (default) - 1/72 inch, PDF native
with PdfExtractor("document.pdf", output_unit=CoordinateUnit.POINTS) as extractor:
    result = extractor.extract()
    print(f"Points: {result.document.pages[0].texts[0].bbox}")

# Inches
with PdfExtractor("document.pdf", output_unit=CoordinateUnit.INCHES) as extractor:
    result = extractor.extract()
    print(f"Inches: {result.document.pages[0].texts[0].bbox}")

# Normalized (0-1 range relative to page dimensions)
with PdfExtractor("document.pdf", output_unit=CoordinateUnit.NORMALIZED) as extractor:
    result = extractor.extract()
    print(f"Normalized: {result.document.pages[0].texts[0].bbox}")
```

Available units:

| Unit | Description |
|------|-------------|
| `POINTS` | 1/72 inch (PDF native, resolution-independent) |
| `PIXELS` | Pixels at specified DPI (not supported for PDF) |
| `INCHES` | Imperial inches |
| `NORMALIZED` | 0-1 range relative to page dimensions |

!!! note
    PDF extractor doesn't support `PIXELS` output because PDFs don't have inherent DPI.

## Table Extraction

Extract tables from PDFs using tabula-py. Requires the `tables` extra:

```bash
uv sync --extra tables
```

### Basic Table Extraction

<!-- skip: next -->
```python
from xtra import PdfExtractor

with PdfExtractor("table.pdf") as extractor:
    result = extractor.extract(table_options={})
    for page in result.document.pages:
        for table in page.tables:
            print(f"Table with {len(table.rows)} rows")
            for row in table.rows:
                print([cell.text for cell in row])
```

### Table Options

Pass tabula options to control extraction behavior. Coordinates in `area` and `columns`
follow the extractor's `output_unit` setting (default: `POINTS`).

<!-- skip: next -->
```python
from xtra import PdfExtractor, CoordinateUnit

with PdfExtractor("table.pdf") as extractor:
    # Lattice mode: for tables with visible borders
    result = extractor.extract(table_options={"lattice": True})

    # Stream mode: for tables without visible borders
    result = extractor.extract(table_options={"stream": True})

    # Extract from specific area (top, left, bottom, right in points - the default)
    result = extractor.extract(table_options={
        "area": (100, 50, 400, 500),
        "columns": [100, 200, 350],  # column boundaries in points
    })

    # Multiple tables per page
    result = extractor.extract(table_options={"multiple_tables": True})

# Using inches instead of points
with PdfExtractor("table.pdf", output_unit=CoordinateUnit.INCHES) as extractor:
    result = extractor.extract(table_options={
        "area": (1.0, 0.5, 5.0, 7.0),   # in inches
        "columns": [1.5, 3.0, 5.5],      # in inches
    })
```

## Path Objects

Both string paths and `Path` objects are supported:

```python
from pathlib import Path
from xtra import PdfExtractor

with PdfExtractor(Path("document.pdf")) as extractor:
    result = extractor.extract()
    print(f"Extracted {len(result.document.pages)} pages")
```

## When to Use PDF Extraction

PDF extraction is ideal for:

- Documents with embedded text (not scanned images)
- High-quality PDFs where text is selectable
- When you need precise character positioning
- Extracting structured tables from PDFs

For scanned PDFs or images, use [OCR Extraction](ocr-extraction.md) instead.
