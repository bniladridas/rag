#!/bin/bash

# Script to rewrite commit messages for conventional commits
# Makes them lowercase and truncates to 40 chars if needed

# Function to rewrite a single commit message
rewrite_commit() {
    local commit_hash=$1
    local original_msg=$(git log --format=%B -n 1 $commit_hash)

    # Get first line
    local first_line=$(echo "$original_msg" | head -n1)

    # Make lowercase
    local lower_line=$(echo "$first_line" | tr '[:upper:]' '[:lower:]')

    # Truncate to 40 chars
    local truncated_line=$(echo "$lower_line" | cut -c1-40)

    # Check if it starts with conventional commit type
    if [[ ! "$truncated_line" =~ ^(feat|fix|docs|style|refactor|test|chore|perf|ci|build|revert): ]]; then
        # If not, try to add appropriate type based on content
        if [[ "$truncated_line" == *"add"* || "$truncated_line" == *"create"* ]]; then
            truncated_line="feat: $truncated_line"
        elif [[ "$truncated_line" == *"fix"* || "$truncated_line" == *"bug"* ]]; then
            truncated_line="fix: $truncated_line"
        elif [[ "$truncated_line" == *"update"* || "$truncated_line" == *"change"* ]]; then
            truncated_line="refactor: $truncated_line"
        elif [[ "$truncated_line" == *"test"* ]]; then
            truncated_line="test: $truncated_line"
        elif [[ "$truncated_line" == *"doc"* || "$truncated_line" == *"readme"* ]]; then
            truncated_line="docs: $truncated_line"
        else
            truncated_line="chore: $truncated_line"
        fi
    fi

    # Reconstruct message
    local new_msg="$truncated_line"
    local rest=$(echo "$original_msg" | tail -n +2)
    if [ -n "$rest" ]; then
        new_msg="$new_msg

$rest"
    fi

    echo "$new_msg"
}

# Main script
if [ $# -eq 0 ]; then
    echo "Usage: $0 <commit-range>"
    echo "Example: $0 HEAD~10..HEAD"
    echo "Example: $0 --all"
    exit 1
fi

range=$1

if [ "$range" = "--all" ]; then
    # Rewrite all commits
    git filter-branch --msg-filter 'bash scripts/rewrite_msg.sh --filter "$GIT_COMMIT"' -- --all
else
    # Rewrite specific range
    git filter-branch --msg-filter 'bash scripts/rewrite_msg.sh --filter "$GIT_COMMIT"' -- $range
fi