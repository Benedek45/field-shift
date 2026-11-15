# B0 implementation handoff

**Implemented:** generator, opaque twin runner using the pilot machinery, offline
scorer, public specs, method/limits README, and regression tests.

The v24 pre-certification search selected 35 pairs: hidden limit 6, quoted price 6,
field fact 5, priority order 6, infeasible request 6, missing data 6. It discarded
188 field-fact candidates: 128 unchanged answers and 60 tied planted winners
(`data/processed/bench/keys.json`). These are **provisional**, not final-certified
keys: the runner deliberately rejects keys without full-pool uniqueness
certificates. A final generator checks every candidate, not just the displayed
top 40, and refuses publication if simulator/data change during generation.

Example provisional pair `74778f5425c2`, Story County IA: the control's income-only
policy chooses corn–soybean. The planted prose says, “My lease permits only hay
as a harvested main crop.” The key becomes continuous hay. The twin has identical
form/data without that sentence. No key or task-family label enters the container.

**Smoke not yet run:** recomputation encountered an external planner error:
`src/fieldshift/planner/search.py:449`, undefined `market_support`, while building
the physical scoreboard pool. No model episodes have been spent, no simulator
files were edited, and 8601 was untouched. Fix that dependency, then recompute
before the authorized single DeepSeek pair smoke.

**Service features:** economics forks are sufficient; no missing service feature
blocks the six implemented families. Unmodelled-study reversals are excluded
because real cards do not certify a quantitative unique reversal. Distractor-only
tasks require invariance, contradictory to this track's flip requirement.

**Verification:** full offline suite passes 269 tests, including 12 benchmark
tests. Benchmark lint passes. Whole-repo lint remains
blocked by five C408 findings in `sim/alfalfa.py` and F821 (`stand`) in
`sim/engine.py`, outside ownership.

**Public limits:** B0 tests reliable decisive-fact use with a decision tool, not
agronomic truth. The underlying odd-year yield validation is median county
r=0.51/0.53 for corn/soybean across 300 counties (AGENTS.md, round 5). Field facts
are tool-applied; hidden limits and priority order share a mechanism; quotes are
synthetic. Restricted inputs are marked, not bundled. Numeric checks audit
simulator-attributed DECISION evidence using exact metric paths and rounding,
with explicitly reported receipt fallback when transcript output truncates.

After recomputation and smoke, run the four-model overnight benchmark:

```sh
uv run python -m fieldshift.bench.run --models opencode-go/deepseek-v4.1-flash opencode-go/glm-5.3-flash opencode-go/gpt-6-luna anthropic/claude-opus-5-5 --reps 3 --parallel 4
uv run python -m fieldshift.bench.score
```

Default: first three fixed pairs/family, 432 episodes. Full reproduction and
smoke commands are in `bench/README.md`.
