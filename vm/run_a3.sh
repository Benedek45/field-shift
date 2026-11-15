#!/bin/sh
# Pilot A3 matrix (EXPERIMENTS.md "A3"): 12 farm-profile cases x 3 reps per model with the form prompt.
# Usage: vm/run_a3.sh <model> <parallel>   (Anthropic models are forced serial by pilot)
set -u
cd "$(dirname "$0")/.."
set -a
. ./.agent-env
set +a
OUT="$HOME/.opencode-vm/fieldshift/runs/a3"
mkdir -p "$OUT"
uv run python -m fieldshift.agent.pilot --cases vm/a3_cases.json --setups full --parallel "$2" \
    --model "$1" --agent-prompt farm-advisor-a3.md --out "$OUT"
