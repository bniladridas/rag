"""
Configuration for commit message validation.
"""

# Allowed commit types
COMMIT_TYPES = [
    "add",
    "feat",
    "fix",
    "docs",
    "style",
    "refactor",
    "test",
    "chore",
    "perf",
    "ci",
    "build",
    "revert",
]

# Maximum length for the first line
MAX_FIRST_LINE_LENGTH = 40

# Whether to enforce lowercase
ENFORCE_LOWERCASE = True

# Whether scope is required
REQUIRE_SCOPE = True

# Whether to allow multiple scopes (e.g., type[scope1][scope2]:)
ALLOW_MULTIPLE_SCOPES = False

# Bracket type for scope: '[' or '('
BRACKET_TYPE = "["
