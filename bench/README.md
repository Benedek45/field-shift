# B0: planted-answer farm-advisor benchmark

**Status:** the full run was permanently stopped by the user on 2026-11-15 at
about 01:00, after provider quota failures and a partial first repeat. No restart
or further model calls are authorised. Only completed non-infrastructure repeat-0
results are reported in `data/processed/bench/b0_partial.json`; see the partial
results and scorer audit in `bench/EXPERIMENTS.md`. No pass^3 claim can be made.
The labelled preliminary release draft is `bench/RESULTS_v0.5.md`. The release audits
are `bench/LICENSES_AUDIT.md` and `bench/PUBLIC_READINESS.md`.

**Question:** does an agent reliably find and use a decisive fact with a decision
tool? This is not a test of agronomic truth or an evaluation of real farm advice.
The chosen answer is established *before* model episodes, not judged retrospectively.

## Construction

`specs.json` declares six counties and six task families. `tasks.json`, generated
before any scored episodes, contains the selected public farm records and prose.
`python -m fieldshift.bench.generate --recompute` regenerates the private keys at
`data/processed/bench/keys.json`. Each key retains its simulator version, code/data
fingerprint, baseline answer, planted answer, and both full scoreboard responses.
The search order is deterministic; failures are retained with reasons. Do not tune
tasks after inspecting model answers.

Each pair has a control farm and a twin with one decisive addition:

| Family | Planted addition | Expected response |
|---|---|---|
| Hidden limit | A lease permits only one harvested main crop, in prose only | Translate to exclusions; change cycle |
| Quoted price | A signed $3,000/t crop price, in prose only | Apply `economics`, then rank; change cycle |
| Field fact | Field-specific soil, slope, rainfall or irrigation-capacity override | Rank the changed field; change cycle |
| Priority order | Buyer specification first, profit second | Enforce the crop specification despite income preference; change cycle |
| Infeasible | A request for unsupported pearl millet | Name inability to evaluate in information gaps; invent no millet metrics |
| Missing data | Buyer eligibility approval is unknown | Name the missing approval and set low confidence |

The first four require the planted canonical cycle to differ from the naive
control answer, and a single canonical planted winner beyond the simulator's
weather-noise tie tolerance. Finalists are certified against the full candidate
pool, not only the service's top-40 display. The control follows A3's deterministic policy:
retain the existing cycle if tied; otherwise take rank 1. Control uniqueness is
not required. Task generation does not use model output.

Candidate forms for field facts are fixed in `generate.py`. The twins always have
the same form. Existing farm overrides are applied by the tool: this family tests
tool-mediated sensitivity, **not** proof that the prose was independently read.
The hidden-limit and priority-order families deliberately use similar crop
exclusions; they are different language presentations, not independent mechanisms.
Extreme contract prices are synthetic stress tests, not representative prices.

Unsupported-crop and missing-data pairs use behavioural rubrics, not cycle flips:
the control must not invent the withheld specific gap, while the planted twin must
report it. There is no required control confidence level.

Not implemented: (a) an unmodelled-effect reversal, because the available real
cards do not establish a transferable quantitative effect sufficient to certify
a unique alternative; (b) a distractor-only family, because its proper outcome is
invariance, incompatible with the required answer-flip rule. No fabricated study
cards or unsupported yield coefficients are used.

## Agent and harness

The benchmark reuses `fieldshift.agent.pilot.run_one`, the A3 procedure in
`vm/agents/farm-advisor-a3.md`, and OpenCode **2.3.0** in Docker on Colima.
`farm-advisor-bench.md` tells the agent to read that procedure; additions explain
economics edits, unavailable crops, provisional confidence, and how to quote
numbers. These are prompt differences from A3 and must be disclosed. B0 ran under
prompt v1 (mandatory `numeric_values` source paths). Prompt v2 (2026-11-15, after
B0) asks for numbers quoted exactly as printed, new numbers only with both inputs
shown, and makes `numeric_values` optional; v1 and v2 results must not be pooled.

Models in the stopped partial comparison (aliases, not independently pinned weights):

- `opencode-go/deepseek-v4.1-flash`
- `opencode-go/glm-5.3-flash`
- `opencode-go/gpt-6-luna`
- `anthropic/claude-opus-5-5`

API-key models use image `fieldshift-agent:oc2.3.0-arm64`, locally inspected ID
`sha256:59d3c3245a395add41d937d40cf5270844242e3495fe8592284345e6ee405123`.
The Opus arm was run through the author's own subscription login; that adapter is not part of this repository (it used a separate image and a
request-transforming plugin, a harness difference). Per-run receipts record the actual
served Opus model. The benchmark contract
records image ID and source hashes. No additional provider probes are required.

The private keys and family identifiers are not mounted into the model container.
Case IDs are opaque. Each episode has a new work/config/data directory, study
cards, `FARM.md`, simulator client, and procedure. The brief is anonymous; the
benchmark renderer removes source URLs that could expose locations and mangled
calendar years. This is redaction, not anonymisation of agronomic fingerprints.
Weather outlook responses are excluded rather than redistributed.

## Scoring

`uv run python -m fieldshift.bench.score` scores saved artifacts offline.

- **Exact cycle:** reduce exact repeats and take the smallest cyclic shift,
  preserving main crop, winter cover and double crop. Order matters; phase does not.
- **Mean accuracy:** fraction of non-infrastructure episodes matching the key
  (or satisfying the two special-family rubrics). Hygienic accuracy (v1)
  additionally requires all citation and strict numerical checks; grounded
  accuracy (v2) replaces the strict numerical check by number grounding. Unrun and archived infrastructure
  episodes are not observations. The partial table explicitly uses exact-cycle
  accuracy for every family; the original behavioural-rubric accuracy is retained
  separately in JSON, so a correct cycle alone does not demonstrate gap reporting.
- **Trap rate:** fraction of planted *cycle* episodes choosing the naive answer.
  Behaviour-only families are excluded from this denominator.
- **Twin flip:** same-model, same-repeat control and planted episodes both pass,
  and their cycles differ in the planted direction. Behavioural twins instead
  require both rubrics; reported as `behavioural_twin`, not a crop-choice flip.
- **pass^3:** number of task variants for which **all three** independent repeats
  pass key/rubric and hygiene. Two successes out of three fail. Incomplete cells
  are not evaluable, rather than failures; no complete three-repeat cells exist
  in B0's partial analysis, so pass^3 is **not available**, not zero.
  This is empirical all-three reliability, not accuracy cubed.
- **Citation hygiene:** cited cards must exist in the supplied folder and have
  been opened; simulation labels must exist and not exceed calls made.
- **Number grounding (scorer v2, headline number-honesty metric; post hoc, see
  EXPERIMENTS.md "Scorer v2"):** every number in a `sim`-cited `evidence` claim
  is classified against what the agent saw or sent. `found`: at the claim's
  displayed precision it equals a JSON leaf or a number inside a string anywhere
  in a cited call's output, or a correct optional `numeric_values` annotation.
  Matching is unit-aware: a number stated with a unit ($/USD, %, kg, t, days, mm,
  °C, score, rank/count) only matches leaves whose key names that quantity or
  string numbers printed with that unit; `%` may match a fraction ×100; a
  unit-less number matches any source number (weaker). Sign may be dropped under
  negative-change wording. `request`: a value the agent sent for that call (or a
  farmer economics edit in force), parsed from `./farmsim <cmd> '<json>'` in the
  transcript. `derived`: difference, sum or percent change of two found sim
  numbers quoted in the same claim. `structural`: a rank/count in rank context
  equal to an output rank or count, or 1-4 as cycle years. Anything else is
  `unfounded`. Wrong annotations are counted but do not condemn a number that is
  present. `grounded_pass` = zero unfounded; grounded accuracy additionally needs
  task correctness and the unchanged card, call-existence and output checks.
  Reported per run, model and family: class counts and unfounded per 100 numbers;
  REPORT.md sentences citing `sim:<n>` are reported separately and do not gate.
  It proves presence, not meaning: a value quoted for the wrong row can pass if
  the same-unit value appears elsewhere in the cited call. Planted-fabrication
  detection 82% (300 plants), see EXPERIMENTS.md.
- **Numeric faithfulness (scorer v1, strict, kept for comparison):** every number
  in a simulator-attributed `evidence` claim needs a `numeric_values` source path
  and rounded value. Resolve it against
  that call's JSON; allow half a unit of the reported final decimal place.
  Explicit scale 100 supports fractional source values reported as percentages.
  Explicit negative-change wording (e.g. “20% less”) preserves the sign of a -20
  source value. Correct extra boolean annotations are checked by exact equality;
  numeric literals in an exact string-valued warning path are valid sources too.
  Wrong/missing paths, unsupported values, and additional unaccounted numbers fail.
  This checks evidence fields, not arbitrary numerical prose in `REPORT.md`.
  It cannot prove that a linguistically ambiguous claim describes the right
  concept, even when its exact metric path and value agree.

Transcript export retains up to 10 million characters per tool output, rather
than A3's 4,000. A receipt-only wrapper around the **unchanged** production
`vm/farmsim` also preserves full JSON responses. When original tool display
truncation still prevents parsing, the scorer uses those non-agent-editable
receipts and reports `receipt_only_calls`; transcript/receipt disagreement fails.
Such rows are not silently described as fully verified from transcript text.

Explicit infrastructure rows are excluded and reported separately. Timeouts
remain failures unless independently classified as infrastructure. Missing decisions
in otherwise completed episodes are failures; unrun/killed episodes are not scored.
Results are reported per model and per family, and
must include actual counts and missing cells, not just percentages.

The partial table's twin-pair accuracy means **both exact-cycle keys matched**
among pairs where both twins completed, independently of hygiene. It is not the
original hygiene-inclusive twin-flip metric. Hygiene includes source-verification
failures caused by ambiguous labels or unavailable outputs, separately classified
as harness limitations; these are not automatically model fabrications.

## Reproduction

The launch commands below document the original design, **not authorisation to
resume the stopped run**. To reproduce the saved-data analysis offline:

```sh
uv run python -m fieldshift.bench.score --partial --runs ~/.opencode-vm/fieldshift/runs/bench_b0 --out data/processed/bench/b0_partial.json
# planted-fabrication test of number grounding; writes only to the scratch --out directory
uv run python -m fieldshift.bench.grounding_audit --runs ~/.opencode-vm/fieldshift/runs/bench_b0 --out <scratch dir> [--n 300 --seed 7]
```

Prerequisites: project dependencies installed with `uv`, the local NASA/NASS/
PRISM-derived county data and trained simulator assets described in the project
README/AGENTS, Docker/Colima and the pinned image. A source checkout alone does not
contain gigabytes of local input data. Regeneration requires MPS/CUDA; scoring is
offline and does not require an accelerator or credentials.

```sh
uv run python -m fieldshift.bench.generate --recompute
uv run pytest
uv run ruff check .
uv run python -m fieldshift.bench.run --smoke
uv run python -m fieldshift.bench.score --runs ~/.opencode-vm/fieldshift/runs/bench_smoke --out data/processed/bench/smoke_report.json
```

The one-pair smoke uses DeepSeek only, one repeat per twin, and never counts
towards pass^3. For the full comparison, after reviewing the keys and smoke:

```sh
uv run python -m fieldshift.bench.run --models opencode-go/deepseek-v4.1-flash opencode-go/glm-5.3-flash opencode-go/gpt-6-luna anthropic/claude-opus-5-5 --reps 3 --parallel 4 --out ~/.opencode-vm/fieldshift/runs/bench
uv run python -m fieldshift.bench.score --out data/processed/bench/report.json
```

The default selects the first three fixed pairs per family: 18 pairs × two twins ×
three repeats × four models = **432 episodes**. `--per-family 0` includes all
generated valid pairs. Selection is fixed before model outputs, not based on
smoke success. Read `contract.json` for the exact included pair IDs.

Credentials are read privately from the environment or the gitignored key files
(`fieldshift.env`, `.env`, `.agent-env`); `--models` accepts any model from the main
README's "Choosing a model and API key", and every model's key is checked before the run starts. The original smoke used port **8611**; the full-run registration
used separate owned ports 8611–8614 (see EXPERIMENTS.md). The runner starts/stops its own service
with a new in-memory token and separate registry/cache/log. It never restarts
8601. Do not run multiple benchmark processes on the same port. Existing completed
episodes are reused; a changed contract requires a new output directory.
Regenerate keys after simulator/data changes, and freeze that key file for the
entire comparison. Scoring rejects keys whose hash differs from the run contract.

## Public dev set and hidden test split (plan for v1.0; not implemented)

The 35 pairs in `tasks.json`/`keys.json` (keys SHA-256 `d0b2270e…5e0`) are the
**public dev set**. Once they are published, any model may have seen them. Scores
on them are therefore dev scores. A **hidden test split** will be generated
afterwards and kept private.

1. Choose new counties that are not in `specs.json`, and record them in a private
   specs file. Generate with the frozen simulator commit. Freeze before any model episode.
2. Keep both the hidden **tasks and keys** private. The generator and simulator are
   public, so anyone holding a hidden task can compute its key. Hiding only the keys
   protects nothing.
3. Before any episode, publish a commitment: the SHA-256 of the hidden keys file.
   This file embeds a random 256-bit salt, because a hash over a few small cycle
   answers could otherwise be found by enumeration. Publish it together with
   per-family counts, simulator version and generator commit. Reveal the file after
   the evaluation, or once the split is retired.
4. Publish only aggregate hidden-split scores. Never publish hidden run directories,
   because they contain briefs and answers.

What the generator supports today (`generate.py`): `--specs` (new counties), `--out`
(a separate keys file) and the `--recompute` guard. `run.py` and `score.py` accept `--keys`.
The **187 discarded candidates are not a reserve.** All of them are field-fact
candidates that failed validity: 125 "answer equals naive" and 62 "planted winner
is inside tie tolerance" (`keys.json` → `discarded`). Each county also yields at
most one pair per family, so new pairs require new counties.

Minimal changes needed before a hidden split (none made yet):

- `main()` always overwrites the public `bench/tasks.json`. Add `--tasks-out`,
  and write the hidden tasks only to a private path.
- Pair IDs are `sha256("<adm_id>:<family>")[:12]` with no salt. A reused county
  reproduces the dev ID and the dev task exactly. Refuse counties that appear in
  the dev specs, and add an `--id-salt` so that hidden IDs reveal nothing.
- `local_service()` always uses `data/processed/bench/service` and `service_log.jsonl`.
  Add a separate service root, so hidden farm records and logs are not mixed with the dev ones.
- Add a commitment step: write a random salt into the keys file, then print and
  store its SHA-256.
- `_fingerprint()` hashes local file modification times, so a commitment can only
  be checked by revealing the file. It cannot be checked by regenerating on another
  machine. Replace the file times with content hashes for v1.0.

## Redistribution and limits

Public task specs contain our own invented farmer prose and county identifiers,
not raw QDANN points/maps, KBS content, or Open-Meteo responses. Evidence cards
are our study summaries with short attributed quotations. Review their source
rights before redistributing beyond this project; publication is not a blanket
licence grant. Derived county tables may be shared subject to their source terms.
Every task explicitly marks the potential QDANN-derived county-evidence dependency
(CC-BY-NC-SA 4.0); restricted inputs must be acquired separately. No KBS data are
required. Do not publish entire run directories indiscriminately.

This benchmark measures controlled fact use with our decision tool, not the
truth of the tool's economics, pest effects, nitrogen balance, or recommended
rotation. The underlying simulator's odd-year yield validation is median county
**r = 0.51 corn / 0.53 soybean across 300 counties** (project `AGENTS.md`, round 5);
it is not validation of every rotation or this benchmark's synthetic quotations.
ET anomaly validation and nitrogen budgets remain weak. Candidate search,
small county coverage, shared mechanisms, conspicuous facts, synthetic prices,
study-summary selection, repeated prompts, model alias drift, and the unmeasured
prompt/transport differences limit generalisation. A public report must state
the tested models, simulator version, counts, exclusions, and observed smoke
outcome, not describe this as verified agronomic intelligence.

In this partial snapshot, all 21 exact-cycle key disagreements occur on Lubbock
tasks and explicitly invoke the prompt-permitted rule-5d study override. Keys
remain frozen, but this design tension means those disagreements must not be
presented as established model reasoning errors.
