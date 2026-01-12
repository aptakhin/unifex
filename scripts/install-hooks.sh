#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
HOOKS_DIR="$REPO_ROOT/.git/hooks"

echo "Installing git hooks..."

# Create hooks directory if it doesn't exist
mkdir -p "$HOOKS_DIR"

# Create symlink for pre-commit hook
ln -sf "$SCRIPT_DIR/pre-commit.sh" "$HOOKS_DIR/pre-commit"

echo "Git hooks installed successfully."
echo "  pre-commit -> scripts/pre-commit.sh"
