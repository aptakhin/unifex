#!/usr/bin/env bash
set -e

echo "=== Running type checker (ty) ==="
poetry run ty check

echo "=== Type check passed ==="
