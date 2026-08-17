#!/usr/bin/env python3
"""
Strip execution outputs from a notebook on its way into git.

Used as a git `clean` filter: the working copy keeps its outputs so you can see
your own results, while the version committed and pushed carries none. Readers
of the repo therefore run the notebooks and generate their own results rather
than inheriting someone else's.

Wire it up (per clone, since git filters live in .git/config):

    ./tools/setup-git-filters.sh

Reads a notebook on stdin, writes the stripped notebook on stdout. Anything it
cannot parse is passed through untouched, so a malformed file is never silently
emptied.
"""

import json
import sys


def strip(nb: dict) -> dict:
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        # Editor-injected per-cell state that only creates diff noise
        cell.get("metadata", {}).pop("execution", None)
        cell.get("metadata", {}).pop("collapsed", None)
    md = nb.get("metadata", {})
    md.pop("widgets", None)          # ipywidgets state can be large and stale
    if "language_info" in md:
        md["language_info"].pop("version", None)   # varies per machine
    return nb


def main() -> int:
    raw = sys.stdin.read()
    try:
        nb = json.loads(raw)
    except json.JSONDecodeError:
        sys.stdout.write(raw)        # not valid JSON — pass through unchanged
        return 0
    json.dump(strip(nb), sys.stdout, indent=1, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
