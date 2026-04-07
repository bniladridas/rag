"""Shared dmgbuild settings for macOS app bundles."""

from __future__ import annotations

import os
from pathlib import Path


app_bundle_path = Path(
    os.environ.get("APP_BUNDLE_PATH", "dist/RAG Transformer.app")
).expanduser()
app_name = app_bundle_path.name

if not app_bundle_path.exists():
    raise FileNotFoundError(f"App bundle not found: {app_bundle_path}")

format = "UDBZ"
size = None
files = [str(app_bundle_path)]
symlinks = {"Applications": "/Applications"}
icon = None
icon_size = 64
background = None
show_status_bar = False
show_tab_view = False
show_toolbar = False
show_pathbar = False
show_sidebar = False
sidebar_width = 180
arrange_by = None
grid_offset = (0, 0)
grid_spacing = 100
scroll_position = (0, 0)
label_pos = (0, 0)
tc_size = 16
text_size = 12
icon_locations = {app_name: (100, 100), "Applications": (300, 100)}
license = None
compression_level = 9
window_rect = ((100, 100), (400, 300))
default_view = "icon-view"
show_icon_preview = False
include_icon_view_settings = "auto"
include_list_view_settings = "auto"
list_icon_size = 16
list_text_size = 12
list_scroll_position = (0, 0)
list_sort_by = "name"
list_use_relative_dates = True
list_calculate_all_sizes = False
list_columns = ("name", "date-modified", "size", "kind", "date-added")
list_column_widths = {
    "name": 300,
    "date-modified": 181,
    "date-created": 181,
    "date-added": 181,
    "date-last-opened": 181,
    "size": 97,
    "kind": 115,
    "label": 100,
    "version": 75,
    "comments": 300,
}
list_column_sort_directions = {
    "name": "ascending",
    "date-modified": "descending",
    "date-created": "descending",
    "date-added": "descending",
    "date-last-opened": "descending",
    "size": "descending",
    "kind": "ascending",
    "label": "ascending",
    "version": "ascending",
    "comments": "ascending",
}
