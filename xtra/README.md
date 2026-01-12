# xtra

Library for unifying document text extraction from PDF and OCR sources.

## Features

- PDF text extraction with pypdfium2
- OCR text extraction with EasyOCR
- PDF-to-image OCR for scanned PDFs
- Bounding box coordinates for all text blocks
- Rotation angle extraction
- Font metadata (PDF only)
- PDF object structure for fraud detection
- Pydantic models (v1 and v2 compatible)

## Architecture

Extractors are in `extractors/` folder. User must explicitly choose extractor type.

```
xtra/
├── extractors/
│   ├── base.py      # BaseExtractor abstract class
│   ├── pdf.py       # PdfExtractor (pypdfium2)
│   └── ocr.py       # OcrExtractor, PdfToImageOcrExtractor (EasyOCR)
├── models.py        # Pydantic models
└── cli.py           # CLI tool
```

## Usage

### Python API

```python
from pathlib import Path
from xtra import PdfExtractor, OcrExtractor, PdfToImageOcrExtractor

# PDF extraction
with PdfExtractor(Path("document.pdf")) as extractor:
    doc = extractor.extract()

# Specific pages only (for parallel processing)
with PdfExtractor(Path("document.pdf")) as extractor:
    doc = extractor.extract(page_numbers=[0, 2, 4])

# Single page extraction (can be parallelized)
with PdfExtractor(Path("document.pdf")) as extractor:
    result = extractor.extract_page(0)
    if result.success:
        print(result.page.texts)

# OCR for images
with OcrExtractor(Path("image.png"), languages=["en", "it"]) as extractor:
    doc = extractor.extract()

# PDF to image OCR (for scanned PDFs)
with PdfToImageOcrExtractor(Path("scanned.pdf"), languages=["en"]) as extractor:
    doc = extractor.extract()
```

### CLI

```bash
# PDF extraction (explicit)
poetry run python -m xtra.cli document.pdf --extractor pdf

# OCR extraction (explicit)
poetry run python -m xtra.cli image.png --extractor ocr --lang en,it

# PDF to image OCR
poetry run python -m xtra.cli scanned.pdf --extractor pdf-ocr

# Specific pages
poetry run python -m xtra.cli document.pdf --extractor pdf --pages 0,1,2

# JSON output
poetry run python -m xtra.cli document.pdf --extractor pdf --json
```

## Models

- `TextBlock`: text, bbox, rotation, confidence (OCR), font_info (PDF)
- `BBox`: x0, y0, x1, y1
- `FontInfo`: name, size, weight
- `Page`: page_number, width, height, texts
- `Document`: path, pages, metadata
- `DocumentMetadata`: source_type, fonts, pdf_objects, etc.

## Alternatives

- kreuzberg: wide local OCR support, plans for cloud support