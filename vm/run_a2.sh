#!/bin/sh
# Pilot A2 matrix (EXPERIMENTS.md "A2"): 20 runs per model with the scoreboard prompt.
# Cheap models run side by side, two containers each; Opus runs serially.
set -u
cd "$(dirname "$0")/.."
# Model keys and FARMSIM_TOKEN: environment, or fieldshift.env / .env / .agent-env (read by Python).
OUT="$HOME/.opencode-vm/fieldshift/runs/a2"
CASES="story york sumner lubbock bolivar fresno story-slope12 story-nitrate story-income lubbock-dry known-rye known-marsden"
run_model() {
    model=$1
    parallel=$2
    uv run python -m fieldshift.agent.pilot --only $CASES --setups full --reps 1 --parallel "$parallel" \
        --model "$model" --agent-prompt farm-advisor-a2.md --out "$OUT"
    uv run python -m fieldshift.agent.pilot --only story lubbock --setups full --reps 5 --parallel "$parallel" \
        --model "$model" --agent-prompt farm-advisor-a2.md --out "$OUT"
}
run_model opencode-go/deepseek-v4.1-flash 2 > "$OUT.deepseek.log" 2>&1 &
run_model opencode-go/glm-5.3-flash 2 > "$OUT.glm.log" 2>&1 &
run_model opencode-go/gpt-6-luna 2 > "$OUT.luna.log" 2>&1 &
# The A2 matrix also ran anthropic/claude-opus-5-5 through the author's own subscription login;
# that adapter is not part of this repository.
wait
echo "A2 matrix finished"
