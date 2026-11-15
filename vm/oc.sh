#!/bin/sh
# Compatibility wrapper. Native Windows callers use the Python module directly.
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec uv run --project "$repo" python -m fieldshift.agent.containers "$@"
