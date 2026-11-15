# B0 planted-answer benchmark: preliminary results, v0.5

> **Status: PRELIMINARY (v0.5).** Read these limits before the numbers.
>
> - **One repeat of three.** Only repeat 0 of the registered 3 repeats ran before
>   provider quotas stopped the run. pass^3 (all three repeats correct) is
>   **not available** (`b0_partial.json` → `models.*.overall["pass^3"] = null`). It is not zero.
> - **Scorer v2 is post hoc.** The headline number-grounding metric was designed
>   after the results were seen (`bench/EXPERIMENTS.md`, "Scorer v2"). The
>   registered v1 metric is reported next to it, unchanged.
> - **Prompt v1 only.** All episodes used benchmark prompt v1. Prompt v2 exists but
>   has not been run. Results from v1 and v2 must not be pooled.
> - **Keys come from our own simulator (Field Shift `fieldshift-v25`), not from field
>   truth.** A "correct" answer matches what our tool ranks first under the planted
>   fact. It is not a claim that the advice is agronomically right.
> - **Coverage is unbalanced.** Each model completed a different subset of the 70
>   repeat-0 episodes (n = 37 to 70), so this is not a balanced head-to-head comparison.
>
> The full results (three repeats, prompt v2, hidden test split) are planned for v1.0.
> The write-up of this release is `bench/paper/b0_v0.5.md` (PDF: `bench/paper/b0_v0.5.pdf`).

## What was tested

B0 tests whether an agent with a decision tool **finds a decisive fact in a farm
brief and uses it**. There are 35 twin pairs: a control farm, and a planted twin
that differs only in one decisive addition. Method: `bench/README.md`. Registered
contracts and logs: `bench/EXPERIMENTS.md`.

| Family | Planted addition | Valid pairs |
|---|---|---:|
| Hidden limit | Lease allows only one harvested main crop, stated in prose only | 6 |
| Quoted price | Signed $3,000/t contract price, stated in prose only | 6 |
| Field fact | Field-specific soil, slope, rain or irrigation-capacity override | 5 |
| Priority order | Buyer crop specification first, income second | 6 |
| Infeasible request | Asks for pearl millet, which the tool cannot evaluate | 6 |
| Missing data | Buyer eligibility approval unknown | 6 |

Pair counts: `bench/EXPERIMENTS.md` "Smoke result" (35 valid pairs, 187 discarded
candidates), confirmed in `data/processed/bench/keys.json` (`pairs`: 35, `discarded`: 187).
Six counties (`bench/specs.json`): Story IA, York NE, Lubbock TX, Sumner KS, Bolivar MS, Fresno CA.

## Setup

| Item | Value | Source |
|---|---|---|
| Harness | OpenCode 2.3.0 in Docker on Colima (linux-arm64) | run `contract.json` `harness_version`; `bench/README.md` |
| Image, OpenCode Go models | `fieldshift-agent:oc2.3.0-arm64`, `sha256:59d3c3245a395add41d937d40cf5270844242e3495fe8592284345e6ee405123` | `contract.json` `image_id`; `bench/README.md` |
| Image, Opus | `fieldshift-agent:oc2.3.0-arm64-anthropic`, `sha256:d08baaacc4ea8c6eb1db28f104ed8fad257a56d857009de728ec243e28b37e73` (all 69 episodes) | per-run `meta.json` `image_id`; `vm/anthropic-v2/README.md` |
| Simulator | `fieldshift-v25`, fingerprint `dd926469c88297ead3dce37d5199e873ef34a35207523f775449a4d04b786177` | `b0_partial.json` `simulator_versions`; `contract.json` `fingerprint` |
| Keys | `data/processed/bench/keys.json`, SHA-256 `d0b2270e6732002cc3f2f215a05a079e3da4b53e91edbd9c5fd7124147acf5e0` | `contract.json` `keys_sha256` (file hash re-checked for this release) |
| Public tasks | `bench/tasks.json`, SHA-256 `4373bf559427a521fb6694f4f2972b5f2a9c37316c7f7262448a1d3cbf14691b` | identical at commit `06d8a09` and now |
| Harness files | `vm/oc.sh`, `vm/agents/farm-advisor-a3.md`, `vm/agents/farm-advisor-bench.md` (prompt v1), `vm/farmsim`, `bench/farmsim`, as in commit `06d8a09` | `contract.json` `source_hashes` match `git show 06d8a09:<file>`; all five files have changed since |
| Scorer | v1 + v2, commit `2670e98` | `bench/EXPERIMENTS.md` "Scorer v2" |
| Results file | `data/processed/bench/b0_partial.json`, `analysis_utc` 2026-11-15T08:10:49Z, SHA-256 `3b4b41d4606b63d611b205ca44a707bd9f7235ac327dcc10dbc26f7673de4ebc` | file |
| Timeout | 1,500 s per episode | `bench/EXPERIMENTS.md` smoke contract |

### Dates (CET, UTC+1)

- 2026-11-14 ~23:20: one-pair smoke (DeepSeek), rejected on one hygiene violation, which led to a tool fix (`EXPERIMENTS.md`).
- 2026-11-14 23:30: keys recomputed on v25 (file time of `keys.json`).
- 2026-11-14 23:31:12: full-run registration committed (`06d8a09`). The section heading says "~23:45", but the commit time is the reliable timestamp.
- 2026-11-14 23:34 to 2026-11-15 03:42: completed repeat-0 episodes (file times of `events.jsonl`). Opus episodes end at 00:56.
- 2026-11-15 ~01:00: the documented stop of B0 (`EXPERIMENTS.md`, "Partial results"). See the note on timing under "Infrastructure events".
- 2026-11-15 09:10 / 09:14: v2 re-score (`analysis_utc`) and scorer-v2 commit (`2670e98`).

### Models and how they were reached

Model names are provider aliases. The weights were not pinned independently.

| Model alias | Access path | Served-model evidence |
|---|---|---|
| `anthropic/claude-opus-5-5` | The user's Claude subscription login (a Claude Code OAuth access token read from the macOS Keychain). It ran through the third-party OpenCode plugin `opencode-claude-auth` 2.2.1 (MIT), with token refresh disabled by a local patch. The plugin transforms requests (system-prompt relocation, tool renaming, extra headers), so the Opus transport differs from the Go path. | 69/69 episodes have `served_model = claude-opus-5-5` from the SSE stream (per-run `meta.json`). |
| `opencode-go/gpt-6-luna` | OpenCode Go subscription plan through its API key. It has a 5-hour usage limit, and the API reports cost 0. | Not recorded per run. |
| `opencode-go/glm-5.3-flash` | Same OpenCode Go plan | Not recorded per run |
| `opencode-go/deepseek-v4.1-flash` | Same OpenCode Go plan | Not recorded per run |

Sources: `bench/README.md` "Agent and harness"; `AGENTS.md`. The Opus adapter was run through the author's own subscription login; that adapter is not part of this repository.
No API-key (pay-per-token) Anthropic or OpenAI access was used in B0.

## Results (repeat 0, completed non-infrastructure episodes only)

All values come from `data/processed/bench/b0_partial.json`, `models.<alias>.overall` and
`models.<alias>.families.<family>`. Counts are rate × n, and each is checked to be an integer.

Column definitions (JSON fields):

- **n**: `n`, the number of completed episodes.
- **Exact cycle**: `accuracy`. The recommended canonical cycle equals the frozen key. This applies to all families, including the two behavioural ones.
- **v2 grounded accuracy (headline, post hoc)**: `v2.grounded_accuracy`. The task is correct (key, or behavioural rubric), there are zero unfounded numbers in sim-cited evidence, and the unchanged card, call-existence and output-consistency checks pass.
- **v1 hygienic accuracy (registered)**: `hygienic_accuracy_v1`. The task is correct, and the strict v1 `numeric_values` attribution and citation checks pass.
- **Trap rate**: `trap_rate` over `trap_n`. The share of planted cycle tasks that got the naive control answer. Behavioural families are excluded.
- **Twin pairs (both keys)**: `twin_pair_accuracy` over `twin_pairs_n`. Pairs where both twins completed and both exact-cycle keys matched. Hygiene is ignored.
- **Unfounded /100**: `v2.unfounded_per_100_numbers`. The denominator is `v2.numbers` without `unverifiable`, shown as k/checked.
- **Median s**: `median_seconds`.
- **Median generated tokens**: `median_tokens.output_tokens`, which is output plus reasoning (`tokens_definition`).

### Per model

| Model | n | Exact cycle | **v2 grounded acc.** | v1 hygienic acc. | Trap rate | Twin pairs (both keys) | Unfounded /100 (k/checked) | Median s | Median generated tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Claude Opus 5.5 | 69 | 67/69 (97.1%) | 63/69 (91.3%) | 27/69 (39.1%) | 0/23 (0.0%) | 32/34 (94.1%) | 0.09 (1/1164) | 167.6 | 13,853 |
| GPT-6 Luna | 70 | 65/70 (92.9%) | 48/70 (68.6%) | 46/70 (65.7%) | 0/23 (0.0%) | 30/35 (85.7%) | 0.00 (0/191) | 162.8 | 14,191 |
| GLM 5.3 Flash | 60 | 52/60 (86.7%) | 28/60 (46.7%) | 3/60 (5.0%) | 0/20 (0.0%) | 24/30 (80.0%) | 2.60 (30/1155) | 236.85 | 16,454 |
| DeepSeek V4.1 Flash | 37 | 31/37 (83.8%) | 28/37 (75.7%) | 15/37 (40.5%) | 0/12 (0.0%) | 13/18 (72.2%) | 0.25 (2/798) | 358.3 | 27,481 |

Supplementary columns (`overall.v2.numbers`, `overall.v2.report_*`,
`task_accuracy_including_behavioural_rubrics`, `task_twin_pair_accuracy`, `excluded`, `missing_repeat_0`):

| Model | Evidence numbers found / request / derived / structural / unfounded / unverifiable | REPORT.md unfounded /100 (k) | Task accuracy incl. rubrics | Task twin pairs | Excluded records | Missing repeat-0 |
|---|---|---:|---:|---:|---:|---:|
| Claude Opus 5.5 | 1142/0/0/21/1/0 | 0.83 (4) | 64/69 (92.8%) | 30/34 (88.2%) | 10 | 1 |
| GPT-6 Luna | 190/0/0/1/0/6 | 0.00 (0) | 60/70 (85.7%) | 25/35 (71.4%) | 193 | 0 |
| GLM 5.3 Flash | 996/2/7/120/30/43 | 3.22 (16) | 50/60 (83.3%) | 20/30 (66.7%) | 194 | 10 |
| DeepSeek V4.1 Flash | 789/0/0/7/2/0 | 3.18 (7) | 30/37 (81.1%) | 11/18 (61.1%) | 207 | 33 |

"Excluded records" are archived infrastructure episodes, killed or incomplete
episodes, and repeat-1 episodes. None of them enters a denominator.

### Per family

#### Claude Opus 5.5

| Family | n | Exact cycle | **v2 grounded acc.** | v1 hygienic acc. | Trap rate | Twin pairs (both keys) | Unfounded /100 (k/checked) | Median s | Median generated tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hidden limit | 12 | 12/12 (100.0%) | 12/12 (100.0%) | 5/12 (41.7%) | 0/6 (0.0%) | 6/6 (100.0%) | 0.00 (0/187) | 137.05 | 11,667 |
| quoted price | 12 | 12/12 (100.0%) | 12/12 (100.0%) | 6/12 (50.0%) | 0/6 (0.0%) | 6/6 (100.0%) | 0.00 (0/209) | 167.5 | 14,126 |
| field fact | 10 | 10/10 (100.0%) | 9/10 (90.0%) | 2/10 (20.0%) | 0/5 (0.0%) | 5/5 (100.0%) | 0.58 (1/172) | 187.7 | 16,152 |
| priority order | 12 | 11/12 (91.7%) | 11/12 (91.7%) | 2/12 (16.7%) | 0/6 (0.0%) | 5/6 (83.3%) | 0.00 (0/202) | 146.2 | 12,337 |
| infeasible request | 12 | 11/12 (91.7%) | 11/12 (91.7%) | 7/12 (58.3%) | n/a (n=0) | 5/6 (83.3%) | 0.00 (0/192) | 179.25 | 14,682 |
| missing data | 11 | 11/11 (100.0%) | 8/11 (72.7%) | 5/11 (45.5%) | n/a (n=0) | 5/5 (100.0%) | 0.00 (0/202) | 148.4 | 13,177 |

#### GPT-6 Luna

| Family | n | Exact cycle | **v2 grounded acc.** | v1 hygienic acc. | Trap rate | Twin pairs (both keys) | Unfounded /100 (k/checked) | Median s | Median generated tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hidden limit | 12 | 11/12 (91.7%) | 9/12 (75.0%) | 8/12 (66.7%) | 0/6 (0.0%) | 5/6 (83.3%) | 0.00 (0/33) | 179.85 | 14,268 |
| quoted price | 12 | 11/12 (91.7%) | 10/12 (83.3%) | 10/12 (83.3%) | 0/6 (0.0%) | 5/6 (83.3%) | 0.00 (0/38) | 156.3 | 13,944 |
| field fact | 10 | 10/10 (100.0%) | 6/10 (60.0%) | 6/10 (60.0%) | 0/5 (0.0%) | 5/5 (100.0%) | 0.00 (0/46) | 185.6 | 14,438 |
| priority order | 12 | 11/12 (91.7%) | 10/12 (83.3%) | 9/12 (75.0%) | 0/6 (0.0%) | 5/6 (83.3%) | 0.00 (0/29) | 146.4 | 12,824 |
| infeasible request | 12 | 11/12 (91.7%) | 8/12 (66.7%) | 8/12 (66.7%) | n/a (n=0) | 5/6 (83.3%) | 0.00 (0/23) | 173.95 | 15,263 |
| missing data | 12 | 11/12 (91.7%) | 5/12 (41.7%) | 5/12 (41.7%) | n/a (n=0) | 5/6 (83.3%) | 0.00 (0/22) | 164.35 | 14,251 |

#### GLM 5.3 Flash

| Family | n | Exact cycle | **v2 grounded acc.** | v1 hygienic acc. | Trap rate | Twin pairs (both keys) | Unfounded /100 (k/checked) | Median s | Median generated tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hidden limit | 12 | 11/12 (91.7%) | 6/12 (50.0%) | 1/12 (8.3%) | 0/6 (0.0%) | 5/6 (83.3%) | 3.46 (8/231) | 200.65 | 13,402 |
| quoted price | 10 | 9/10 (90.0%) | 5/10 (50.0%) | 0/10 (0.0%) | 0/5 (0.0%) | 4/5 (80.0%) | 0.98 (2/204) | 229.45 | 14,746 |
| field fact | 8 | 7/8 (87.5%) | 5/8 (62.5%) | 0/8 (0.0%) | 0/4 (0.0%) | 3/4 (75.0%) | 0.53 (1/189) | 274.8 | 18,852 |
| priority order | 10 | 9/10 (90.0%) | 4/10 (40.0%) | 0/10 (0.0%) | 0/5 (0.0%) | 4/5 (80.0%) | 5.48 (8/146) | 186 | 12,762 |
| infeasible request | 10 | 8/10 (80.0%) | 4/10 (40.0%) | 2/10 (20.0%) | n/a (n=0) | 4/5 (80.0%) | 3.26 (6/184) | 244.85 | 17,568 |
| missing data | 10 | 8/10 (80.0%) | 4/10 (40.0%) | 0/10 (0.0%) | n/a (n=0) | 4/5 (80.0%) | 2.49 (5/201) | 259.2 | 17,961 |

#### DeepSeek V4.1 Flash

| Family | n | Exact cycle | **v2 grounded acc.** | v1 hygienic acc. | Trap rate | Twin pairs (both keys) | Unfounded /100 (k/checked) | Median s | Median generated tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hidden limit | 7 | 6/7 (85.7%) | 6/7 (85.7%) | 3/7 (42.9%) | 0/3 (0.0%) | 2/3 (66.7%) | 0.00 (0/112) | 373.4 | 25,623 |
| quoted price | 6 | 5/6 (83.3%) | 5/6 (83.3%) | 3/6 (50.0%) | 0/3 (0.0%) | 2/3 (66.7%) | 0.00 (0/143) | 411.15 | 33,766 |
| field fact | 6 | 6/6 (100.0%) | 4/6 (66.7%) | 0/6 (0.0%) | 0/3 (0.0%) | 3/3 (100.0%) | 1.44 (2/139) | 385 | 25,938 |
| priority order | 6 | 5/6 (83.3%) | 5/6 (83.3%) | 5/6 (83.3%) | 0/3 (0.0%) | 2/3 (66.7%) | 0.00 (0/124) | 315.7 | 29,376 |
| infeasible request | 6 | 5/6 (83.3%) | 5/6 (83.3%) | 3/6 (50.0%) | n/a (n=0) | 2/3 (66.7%) | 0.00 (0/148) | 377.6 | 27,098 |
| missing data | 6 | 4/6 (66.7%) | 3/6 (50.0%) | 1/6 (16.7%) | n/a (n=0) | 2/3 (66.7%) | 0.00 (0/132) | 359 | 25,564 |

### Reading the numbers

- **Small n.** Per-family cells hold 6 to 12 episodes from a single repeat. A difference of one episode moves a cell by 8 to 17 points. No confidence intervals are given, and no model ranking should be read from these cells.
- **The trap rate is 0 for every model.** No planted cycle task got the naive control answer (`trap_n` 12 to 23). Every exact-cycle miss is a study override, as described in the next section.
- **v1 vs v2.** v1 failed most otherwise-good answers on bookkeeping. For example, all 42 Opus v1 numeric failures are `unaccounted_claim_number` (`overall.hygiene_failure_classes`). v2 asks whether each number is present in what the agent saw or sent. The hand audit and its limits are in `EXPERIMENTS.md` "Scorer v2": 0/60 unfounded classifications were classifier errors, 38/40 found numbers were confirmed, and 247/300 (82%) planted fabrications were detected. v2 proves that a number is present, not that it is used with the right meaning.
- **Luna quotes few numbers.** It has 191 checkable evidence numbers, plus 6 unverifiable, against 1,164 for Opus. Its gap between `grounded_pass_rate` (70/70) and `grounded_hygiene_rate` (56/70) comes from conflicting or unavailable simulator outputs under concurrent call labels. This is a harness provenance problem (`EXPERIMENTS.md`), not a model fabrication.
- **Behavioural families.** The exact-cycle column for infeasible-request and missing-data tasks does not show that the gap was reported. The rubric result is in "Task accuracy incl. rubrics", and the headline v2 column requires it.

## The 21 Lubbock key disagreements: permitted study overrides

All 21 exact-cycle misses across the four models are on Lubbock County, TX
(`US-48-303`). Every one records `decision_rule = "5d"` with `override_used = true`
(`b0_partial.json` → `key_disagreements`, and `key_disagreement_caveat`). By model:
GLM 8, DeepSeek 6, Luna 5, Opus 2. By twin: 15 are controls (priority order 4, hidden limit 3,
quoted price 3, infeasible 2, missing data 2, field fact 1) and 6 are planted
twins (infeasible 3, missing data 3).

Rule 5d of the A3 procedure (`vm/agents/farm-advisor-a3.md`, step 5d) allows the
agent to override the simulator's tied-set choice "only if a study card you have opened shows that the simulator misses an
effect that is large for this farm and would reverse the order". The frozen keys
do not represent such overrides. These answers therefore disagree with the key while
following the prompt. The keys were not changed after the results. These rows
must not be presented as proven reasoning errors or as evidence of worse agronomic
advice. This is a design tension that v1.0 must resolve: either remove 5d from the
benchmark prompt, or score 5d answers by a separately registered rule.

## Infrastructure events

All of these are infrastructure. None of them is counted as a model failure, and
no infrastructure record enters a denominator.

1. **OpenCode Go 5-hour usage limit** (HTTP 429 `provider.go-limit`), reached about 45 minutes into the run on 2026-11-15 ~00:40. In the first stop, 501 of the 630 Go episodes failed in about 5 s with no model step (`EXPERIMENTS.md`, infrastructure log). After the runner change below, more quota failures were archived. The archive totals in `b0_partial.json` (`excluded`, reason `archived_infrastructure`) are DeepSeek 207, GLM 194 and Luna 172.
2. **Runner change** made after the first stop: any quota or overload error marks the episode as infrastructure, stops that runner, and restarts it after the reset (`vm/run_b0.sh`).
3. **Scorer crash, Opus.** `pilot.canonical` raised an error on an alternative with an empty rotation (pair `1e0b327cdab8`, planted, repeat 0). The Opus runner stopped after 43 episodes. The episode was archived as `_infra_scorer_crash_*`, and `canonical` now returns an empty cycle for an empty rotation. Scoring is otherwise unchanged.
4. **Opus subscription token window.** Episodes had to start with token lifetime ≥ timeout + 600 s. One planted Opus repeat-0 episode (`f03b11ba62ffb`) is missing as infrastructure (`missing_repeat_0`).
5. **Disk-full crash, in the preceding A3 pilot, not in B0.** On 2026-11-14 ~20:50-21:15, each Opus run copied the 233 MB vendored plugin and never removed it, which filled the disk. Runner processes died in `anthropic_auth.prepare` (root `EXPERIMENTS.md`, "A3 run log"). The fix was to hard-link the plugin, and it was in place before B0. No B0 episode is recorded as affected.
6. **Stop.** The user stopped B0 permanently, documented as 2026-11-15 ~01:00 (`EXPERIMENTS.md`). The file times show Go-model repeat-0 episodes completing up to 03:42 CET, and runner logs show quota-sleep restarts until at least 04:47 CET. The runners therefore kept working after the documented stop time, or the documented time is approximate. **This must be reconciled before v1.0.** It does not change any counted episode: every counted episode is a completed, exit-0, non-archived repeat-0 record.
7. **Harness provenance.** Some GLM and Luna runs have duplicate or non-monotonic `sim` labels and malformed receipt tails. Tool output is cut near 51,362 characters. These are classified separately and are not guessed (`EXPERIMENTS.md`, "Sample audit").

## Known gaps in this release

- Go-model episodes record no served-model identifier, so only the requested alias is known.
- The run contract lists the base image ID even for the Opus runner. The Opus episodes actually used the separate anthropic image (`meta.json`). Treat `meta.json` as authoritative.
- The simulator fingerprint includes local file modification times, so it cannot be reproduced on another machine. The current working tree no longer matches `dd926469…`, because later simulator commits changed fingerprinted files. Reproducing the keys requires commit `06d8a09` and the original local data.
- `bench/REPORT.md` is an older implementation handoff (it mentions v24 provisional keys and 188 discarded candidates). It is superseded by this file and by `bench/EXPERIMENTS.md`.

## What v1.0 will add

1. **Three independent repeats** for every task and model, so that pass^3 becomes available.
2. **Prompt v2** (numbers quoted as printed, both inputs shown for any computed number, optional `numeric_values`). It gets a new contract and output directory and is never pooled with v1.
3. **A hidden test split**, with a published SHA-256 commitment made before any model sees it. See `bench/README.md` "Public dev set and hidden test split".
4. **More counties and pairs**, with at least three valid pairs per family. This also reduces the dependence on six counties and on Lubbock overrides.
5. **A resolution of the rule-5d design tension**, registered before the runs.
6. **A reproducible package**: pinned commit, a fingerprint without file times, image digests, task and key files with hashes, scorer, and a documented offline re-score of the saved artifacts.
7. **Served-model receipts** for every provider, where the transport exposes them.

## AI-use disclosure

This disclosure follows the NASA Space Apps 2026 rules, which require AI-generated
code and data to be acknowledged in descriptive text and metadata.

- **Code, task generator, scorer and documentation** were written with AI coding
  agents in OpenCode, **Anthropic Claude Opus 5.5** and **OpenAI GPT-6 Luna**, under the team's
  direction and review.
  The project idea, the benchmark question and the acceptance decisions came from the team.
- **Task prose** (the planted sentences) comes from fixed templates in
  `src/fieldshift/bench/generate.py`, which were written with those agents. **Keys** were computed by our
  simulator, not by a language model.
- **Evaluated models (subjects of the benchmark, not authors):**
  `anthropic/claude-opus-5-5`, `opencode-go/gpt-6-luna`, `opencode-go/glm-5.3-flash`,
  `opencode-go/deepseek-v4.1-flash`.
- This results file was drafted by an AI agent (Claude Opus 5.5) from
  `b0_partial.json` and the registered documents. Every number links to its field.
- No AI-generated images or NASA logos are included.

## Changelog

- **v0.5.0 (2026-11-15, draft, unpublished):** first public snapshot. It contains the method, the 35 public
  dev pairs, the generator and scorer (v1 + post hoc v2), and the repeat-0 partial results
  for four models under prompt v1. Repeats 1-2 are not run. pass^3 is not available.
