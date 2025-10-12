#!/usr/bin/env python3
"""
Enhance CHANGELOG.md with additional details.
"""

import re
from pathlib import Path

def enhance_changelog():
    changelog_path = Path("CHANGELOG.md")
    if not changelog_path.exists():
        print("CHANGELOG.md not found")
        return

    content = changelog_path.read_text()

    # Example enhancement: Add issue links if #123 is mentioned
    def add_issue_links(match):
        version_header = match.group(1)
        # Find issue numbers in the section
        section = match.group(2)
        # Replace #123 with [#123](https://github.com/bniladridas/rag/issues/123)
        enhanced_section = re.sub(r'#(\d+)', r'[#\1](https://github.com/bniladridas/rag/issues/\1)', section)
        return version_header + enhanced_section

    # Apply to each version section
    pattern = r'(## v[\d\.]+\s.*?\n)(.*?)(?=\n## v|\n*$)'
    enhanced_content = re.sub(pattern, add_issue_links, content, flags=re.DOTALL)

    # Add a summary at the top or something
    # For now, just write back
    changelog_path.write_text(enhanced_content)
    print("Enhanced CHANGELOG.md")

if __name__ == "__main__":
    enhance_changelog()