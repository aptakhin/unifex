#!/usr/bin/env bash
set -e

echo "=== Building package ==="

# Clean previous builds
rm -rf dist/

# Build the package
uv build

echo "=== Build artifacts ==="
ls -la dist/

echo "=== Build complete ==="
