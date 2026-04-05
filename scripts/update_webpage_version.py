#!/usr/bin/env python3
"""Update version in release-webpage HTML files from __version__.py."""

import re
import sys
from pathlib import Path

VERSION_FILE = Path("src/rag/__version__.py")
HTML_FILES = [
    Path("release-webpage/index.html"),
    Path("release-webpage/index-new.html"),
]


def get_version():
    content = VERSION_FILE.read_text()
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        sys.exit(f"Could not find version in {VERSION_FILE}")
    return match.group(1)


def main():
    version = get_version()
    patterns = [
        (
            r'<span class="version">v[\d.]+</span>',
            f'<span class="version">v{version}</span>',
        ),
        (
            r'<div class="version">v[\d.]+</div>',
            f'<div class="version">v{version}</div>',
        ),
        (
            r'<p class="footer-copy">v[\d.]+</p>',
            f'<p class="footer-copy">v{version}</p>',
        ),
        (r"Successfully installed rag-[\d.]+", f"Successfully installed rag-{version}"),
    ]

    updated = []
    for html_file in HTML_FILES:
        html = html_file.read_text()
        for pattern, replacement in patterns:
            html = re.sub(pattern, replacement, html)
        html_file.write_text(html)
        updated.append(str(html_file))

    print(f"Updated {', '.join(updated)} to version {version}")


if __name__ == "__main__":
    main()
