#!/usr/bin/env bash
set -e

# Default values
COVERAGE_MIN=${COVERAGE_MIN:-78}
SKIP_CLOUD_TESTS=${SKIP_CLOUD_TESTS:-false}

echo "=== Running tests ==="
echo "Coverage minimum: ${COVERAGE_MIN}%"
echo "Skip cloud tests: ${SKIP_CLOUD_TESTS}"

if [ "$SKIP_CLOUD_TESTS" = "true" ]; then
    uv run pytest \
        -k "not (azure and test_ocr_extract_pdf) and not (google and test_ocr_extract_pdf)" \
        --cov=xtra \
        --cov-report=term-missing \
        --cov-fail-under="${COVERAGE_MIN}"
else
    uv run pytest \
        --cov=xtra \
        --cov-report=term-missing \
        --cov-fail-under="${COVERAGE_MIN}"
fi

echo "=== Tests passed ==="
