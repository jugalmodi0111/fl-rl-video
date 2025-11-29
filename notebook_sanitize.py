#!/usr/bin/env python3
"""Sanitize Jupyter notebook metadata.widgets by ensuring each has a 'state' key.

Usage:
    python3 notebook_sanitize.py "rl-fl-ensemble (1).ipynb"

This script will create a backup copy with suffix `.bak` before modifying the file in-place.
"""
import json
import sys
from pathlib import Path


def sanitize_nb(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))

    changed = False

    # Helper to ensure metadata.widgets.state exists where metadata.widgets present
    def fix_meta(meta):
        nonlocal changed
        if not isinstance(meta, dict):
            return
        if "widgets" in meta:
            w = meta["widgets"]
            if isinstance(w, dict) and "state" not in w:
                w["state"] = {}
                changed = True

    # Top-level metadata
    if "metadata" in data:
        fix_meta(data["metadata"])

    # Per-cell metadata
    for cell in data.get("cells", []):
        if "metadata" in cell:
            fix_meta(cell["metadata"])

    if changed:
        bak = path.with_suffix(path.suffix + ".bak")
        print(f"Creating backup: {bak}")
        bak.write_bytes(path.read_bytes())
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"Sanitized notebook saved: {path}")
    else:
        print("No metadata.widgets without state found. No changes made.")


def main():
    if len(sys.argv) < 2:
        print("Usage: notebook_sanitize.py path/to/notebook.ipynb")
        sys.exit(1)
    p = Path(sys.argv[1])
    if not p.exists():
        print(f"File not found: {p}")
        sys.exit(1)
    sanitize_nb(p)


if __name__ == "__main__":
    main()
