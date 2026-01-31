# Extractors Reference

## PDF Extractor

### PdfExtractor

Native PDF text extraction using pypdfium2.

::: unifex.pdf.PdfExtractor

## Local OCR Extractors

### EasyOcrExtractor

OCR using EasyOCR library.

::: unifex.ocr.extractors.easy_ocr.EasyOcrExtractor

### TesseractOcrExtractor

OCR using Tesseract.

::: unifex.ocr.extractors.tesseract_ocr.TesseractOcrExtractor

### PaddleOcrExtractor

OCR using PaddleOCR.

::: unifex.ocr.extractors.paddle_ocr.PaddleOcrExtractor

## Cloud OCR Extractors

### AzureDocumentIntelligenceExtractor

Azure Document Intelligence OCR.

::: unifex.ocr.extractors.azure_di.AzureDocumentIntelligenceExtractor

### GoogleDocumentAIExtractor

Google Document AI OCR.

::: unifex.ocr.extractors.google_docai.GoogleDocumentAIExtractor

## LLM Extractors

### extract_structured

Synchronous LLM extraction function.

::: unifex.llm_factory.extract_structured

### extract_structured_async

Asynchronous LLM extraction function.

::: unifex.llm_factory.extract_structured_async
