#!/bin/bash
# Local release script
# Run this to create a release locally (requires GITHUB_TOKEN env var)

if [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: GITHUB_TOKEN environment variable not set"
    echo "Set it with: export GITHUB_TOKEN=your_token_here"
    exit 1
fi

echo "Running semantic-release..."

semantic-release version --push --tag --changelog --vcs-release

echo ""
echo "Release complete!"
