# Installation

## Basic Installation

Install unifex using pip:

```bash
pip install unifex
```

Or using uv:

```bash
uv add unifex
```

## Optional Dependencies

unifex uses optional dependencies to keep the base installation lightweight. Install only what you need:

### PDF Extraction

```bash
pip install unifex[pdf]
```

### Local OCR Engines

```bash
# EasyOCR
pip install unifex[easyocr]

# Tesseract (requires system Tesseract installation)
pip install unifex[tesseract]

# PaddleOCR
pip install unifex[paddle]
```

### Cloud OCR Services

```bash
# Azure Document Intelligence
pip install unifex[azure]

# Google Document AI
pip install unifex[google]
```

### LLM Providers

```bash
# OpenAI
pip install unifex[llm-openai]

# Anthropic
pip install unifex[llm-anthropic]

# Google Gemini
pip install unifex[llm-google]

# All LLM providers
pip install unifex[llm-all]
```

### Everything

```bash
pip install unifex[all]
```

## System Requirements

### Tesseract OCR

Tesseract requires system installation:

- **macOS**: `brew install tesseract`
- **Ubuntu**: `apt-get install tesseract-ocr`
- **Windows**: Download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)

### PaddleOCR

PaddleOCR works out of the box but may require additional setup for GPU acceleration.
