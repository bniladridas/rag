#!/bin/bash
# RAG Transformer - Test Suite Runner

set -e

PROJECT_ROOT="/Users/niladri/Desktop/rag"

echo "Running RAG Transformer Test Suite"
echo "=================================="

PYTHONPATH="$PROJECT_ROOT" \
python -m pytest \
  --cov=src \
  --cov-report=term-missing \
  --cov-report=html:coverage_report \
  -v "$@"

TEST_RESULT=$?

echo
echo "Test suite completed with exit code: $TEST_RESULT"

if [ "$TEST_RESULT" -eq 0 ]; then
  echo "All tests passed successfully."
  echo "Opening coverage report..."
  open coverage_report/index.html
else
  echo "Some tests failed. Check the output above for details."
fi

exit "$TEST_RESULT"
