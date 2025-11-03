#!/usr/bin/env python3
"""
Commit message validation script.
"""

import re
import sys
from pathlib import Path

# Add root to path to import config
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from commit_msg_config import COMMIT_TYPES, MAX_FIRST_LINE_LENGTH, ENFORCE_LOWERCASE, REQUIRE_SCOPE, ALLOW_MULTIPLE_SCOPES, BRACKET_TYPE

def main():
    if len(sys.argv) != 2:
        print("Usage: commit-msg.py <commit-msg-file>")
        sys.exit(1)

    commit_msg_file = sys.argv[1]

    with open(commit_msg_file, 'r') as f:
        lines = f.readlines()

    if not lines:
        print("Commit message is empty")
        sys.exit(1)

    first_line = lines[0].strip()

    # Enforce lowercase if configured
    if ENFORCE_LOWERCASE:
        first_line_lower = first_line.lower()
        if first_line != first_line_lower:
            # Update the file with lowercase
            lines[0] = first_line_lower + '\n'
            with open(commit_msg_file, 'w') as f:
                f.writelines(lines)
            first_line = first_line_lower

    # Check length
    if len(first_line) > MAX_FIRST_LINE_LENGTH:
        print(f"Commit message first line too long (> {MAX_FIRST_LINE_LENGTH} chars)")
        sys.exit(1)

    # Build regex based on config
    types_pattern = '|'.join(COMMIT_TYPES)
    if BRACKET_TYPE == '[':
        open_bracket = r'\['
        close_bracket = r'\]'
        close_char = ']'
    elif BRACKET_TYPE == '(':
        open_bracket = r'\('
        close_bracket = r'\)'
        close_char = ')'
    else:
        print("Invalid BRACKET_TYPE in config (must be '[' or '(')")
        sys.exit(1)

    if REQUIRE_SCOPE:
        if ALLOW_MULTIPLE_SCOPES:
            scope_pattern = f"({open_bracket}[^{close_bracket}]+{close_bracket})+"
            example = f"fix{BRACKET_TYPE}e2e{close_char}{BRACKET_TYPE}ui{close_char}: description"
        else:
            scope_pattern = f"{open_bracket}[^{close_bracket}]+{close_bracket}"
            example = f"fix{BRACKET_TYPE}e2e{close_char}: description"
    else:
        scope_pattern = f"({open_bracket}[^{close_bracket}]+{close_bracket})?"
        example = f"fix or fix{BRACKET_TYPE}e2e{close_char}: description"

    pattern = rf"^({types_pattern}){scope_pattern}:\s"

    if not re.match(pattern, first_line):
        print(f"Commit message must start with a conventional commit type and scope as configured (e.g., {example})")
        sys.exit(1)

    sys.exit(0)

if __name__ == "__main__":
    main()