#!/usr/bin/env bash
# Reproduce every verification gate: lint, types, tests+coverage, wheel build, clean-venv install + CLI smoke test.
# Usage: scripts/verify.sh "<cli smoke command run inside the clean venv>"
set -euo pipefail
cd "$(dirname "$0")/.."
SMOKE="${1:-}"
echo "== python: $(python3 --version) on $(uname -sm)"
echo "== ruff"; ruff check . ; ruff format --check .
echo "== mypy"; mypy src
echo "== pytest (100% line+branch coverage enforced)"; pytest -q
echo "== build wheel"
rm -rf dist build
python3 -m build --wheel --outdir dist >/dev/null
VENV="$(mktemp -d)/venv"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q dist/*.whl
echo "== clean-venv smoke test"
if [ -n "$SMOKE" ]; then
  (cd "$(mktemp -d)" && PATH="$VENV/bin:$PATH" bash -c "$SMOKE")
fi
echo "== ALL GATES PASSED"
