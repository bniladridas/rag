"""
Configuration for commit message validation.
"""

# Allowed commit types (must match AGENTS.asc)
COMMIT_TYPES = [
    "feat",
    "fix",
    "perf",
    "refactor",
    "docs",
    "chore",
    "test",
    "ci",
    "build",
    "revert",
]

# Maximum length for the first line
MAX_FIRST_LINE_LENGTH = 60

# Whether to enforce lowercase
ENFORCE_LOWERCASE = True

# Whether scope is required
REQUIRE_SCOPE = False

# Whether to allow multiple scopes (e.g., type[scope1][scope2]:)
ALLOW_MULTIPLE_SCOPES = False

# Bracket type for scope: '[' or '('
BRACKET_TYPE = "["
