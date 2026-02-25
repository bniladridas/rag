#!/bin/bash
# Local release testing script
# Tests the release process without actually pushing

echo "Testing semantic-release in dry-run mode..."

pip install python-semantic-release -q

semantic-release version --dry-run

echo ""
echo "Dry-run complete. No changes were made."
