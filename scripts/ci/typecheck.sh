#!/usr/bin/env bash
set -e

echo "=== Running type checker (ty) via pre-commit ==="
uv run pre-commit run ty-check --all-files

echo "=== Type check passed ==="
