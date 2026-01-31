#!/usr/bin/env bash
set -e

# Publish to PyPI
# Requires PYPI_TOKEN environment variable

if [ -z "$PYPI_TOKEN" ]; then
    echo "Error: PYPI_TOKEN environment variable is not set"
    exit 1
fi

echo "=== Publishing to PyPI ==="

# Publish using uv with token
uv publish --token "$PYPI_TOKEN"

echo "=== Published successfully ==="
