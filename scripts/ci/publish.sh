#!/usr/bin/env bash
set -e

# Publish to PyPI
# Requires PYPI_TOKEN environment variable

if [ -z "$PYPI_TOKEN" ]; then
    echo "Error: PYPI_TOKEN environment variable is not set"
    exit 1
fi

echo "=== Publishing to PyPI ==="

# Configure poetry to use PyPI token
poetry config pypi-token.pypi "$PYPI_TOKEN"

# Publish
poetry publish

echo "=== Published successfully ==="
