#!/usr/bin/env bash
# Configure the notebook-output stripping filter for this clone.
# Git filters are stored in .git/config, which is not committed, so each clone
# runs this once. Without it, notebooks commit with their outputs intact.
set -euo pipefail
cd "$(dirname "$0")/.."
git config filter.nbstrip.clean  "python3 tools/nbstrip.py"
git config filter.nbstrip.smudge cat
git config filter.nbstrip.required true
echo "✅ nbstrip filter configured."
echo "   Working copies keep their outputs; commits are stripped."
echo "   Run 'git add --renormalize .' to apply to already-tracked notebooks."
