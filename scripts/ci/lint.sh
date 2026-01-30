#!/usr/bin/env bash
set -e

echo "=== Checking code formatting with ruff ==="
poetry run ruff format --check

echo "=== Running ruff linter ==="
poetry run ruff check

echo "=== Lint passed ==="
