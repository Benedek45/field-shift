#!/bin/sh
# Full B0 run: one runner process, output directory and owned simulator port per model.
# A runner exits 75 when the provider quota is exhausted; it is restarted after the reset
# (completed episodes are kept, quota-hit episodes are moved aside and rerun).
set -u
cd "$(dirname "$0")/.."
set -a
. ./.agent-env
set +a
OUT="$HOME/.opencode-vm/fieldshift/runs/bench_b0"
STOP_AT=${B0_STOP_AT:-$(date -j -f "%Y-%m-%d %H:%M" "2026-11-15 20:00" +%s)}
mkdir -p "$OUT"
run() {
  model=$1 parallel=$2 port=$3
  slug=$(printf '%s' "$model" | tr '/' '-')
  while [ "$(date +%s)" -lt "$STOP_AT" ]; do
    uv run python -m fieldshift.bench.run --models "$model" --reps 3 --per-family 0 \
      --parallel "$parallel" --port "$port" --out "$OUT/$slug" >> "$OUT/$slug.log" 2>&1
    code=$?
    [ "$code" -eq 75 ] || { echo "runner exit $code" >> "$OUT/$slug.log"; return; }
    ms=$(grep -o 'PROVIDER_LIMIT retry_after_ms=[0-9]*' "$OUT/$slug.log" | tail -1 | cut -d= -f2)
    wait_s=$(( ${ms:-0} / 1000 + 120 ))
    [ "$wait_s" -gt 300 ] || wait_s=1800
    echo "provider limit; sleeping ${wait_s}s from $(date)" >> "$OUT/$slug.log"
    sleep "$wait_s"
  done
  echo "stop time reached $(date)" >> "$OUT/$slug.log"
}
if [ "$#" -gt 0 ]; then
  run "$1" "$2" "$3"
  exit
fi
# The published B0 run also ran anthropic/claude-opus-5-5 through the author's own subscription login;
# that adapter is not part of this repository.
run opencode-go/gpt-6-luna 3 8612 &
run opencode-go/glm-5.3-flash 3 8613 &
run opencode-go/deepseek-v4.1-flash 3 8614 &
wait
