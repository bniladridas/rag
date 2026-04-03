#!/usr/bin/env python3
"""
Enhance CHANGELOG.md with additional details.
Adds issue links only for standalone #123 patterns.
"""

import re
from pathlib import Path


def enhance_changelog():
    changelog_path = Path("CHANGELOG.md")
    if not changelog_path.exists():
        print("CHANGELOG.md not found")
        return

    content = changelog_path.read_text()
    original = content

    # Pattern: word-boundary or start of line, then #number, then word-boundary or end
    # Only matches standalone patterns like " fix: handle #123" or "#123)"
    # Does NOT match: "[#123]", "(#123)", already linked "[#123](url)"
    def add_issue_link(match):
        prefix = match.group(1)
        issue_num = match.group(2)
        return f"{prefix}[`#{issue_num}`](https://github.com/bniladridas/rag/issues/{issue_num})"

    # Match standalone #number - after whitespace or start, not in brackets
    pattern = r"(^|\s)#(\d+)(?!\))"
    enhanced_content = re.sub(pattern, add_issue_link, content)

    if enhanced_content != original:
        changelog_path.write_text(enhanced_content)
        print("Enhanced CHANGELOG.md with issue links")
    else:
        print("No changes needed")


if __name__ == "__main__":
    enhance_changelog()
