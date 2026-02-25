#!/usr/bin/env python3
"""Update version in release-webpage/index.html from __version__.py"""

import re
import sys
from pathlib import Path

VERSION_FILE = Path("src/rag/__version__.py")
HTML_FILE = Path("release-webpage/index.html")


def get_version():
    content = VERSION_FILE.read_text()
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        sys.exit(f"Could not find version in {VERSION_FILE}")
    return match.group(1)


def main():
    version = get_version()
    html = HTML_FILE.read_text()

    patterns = [
        (
            r'<div class="version">v[\d.]+</div>',
            f'<div class="version">v{version}</div>',
        ),
        (r"Successfully installed rag-[\d.]+", f"Successfully installed rag-{version}"),
        (r"Version [\d.]+</div>\s*╰", f"Version {version}</div>\n│"),
    ]

    for pattern, replacement in patterns:
        html = re.sub(pattern, replacement, html)

    HTML_FILE.write_text(html)
    print(f"Updated HTML to version {version}")


if __name__ == "__main__":
    main()
