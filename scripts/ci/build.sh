#!/usr/bin/env bash
set -e

echo "=== Building package ==="

# Clean previous builds
rm -rf dist/

# Build the package
poetry build

echo "=== Build artifacts ==="
ls -la dist/

echo "=== Build complete ==="
