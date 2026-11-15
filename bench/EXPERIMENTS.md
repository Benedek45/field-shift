# B0 experiment record

## Registered before any benchmark model episode — 2026-11-14

Hypothesis: an advisor following A3 can extract a decisive prose restriction,
apply it to the decision tool, and produce the precomputed unique planted answer
while its unplanted twin follows the baseline policy.

Smoke contract: one hidden-limit pair, one repeat per twin; only
`opencode-go/deepseek-v4.1-flash`; OpenCode 2.3.0; timeout 1,500 s;
parallelism 2; dedicated output
`~/.opencode-vm/fieldshift/runs/bench_smoke/`. An owned simulator instance listens
on 8611 for the duration. No 8601 restart, model probes, or other episodes.

Confirmation: both decisions are readable, follow their separate computed keys,
and satisfy citation/numeric hygiene; the planted answer differs from control.
Rejection: either answer fails; loop operation can still succeed if both episodes
complete and artifacts are collected. Infrastructure is distinguished from
capability. This two-episode smoke is not a model comparison or pass^3 estimate.

Tasks are searched against the simulator before episodes, rejecting unchanged or
non-unique planted answers. All accepted model-choice tasks require full-pool
uniqueness certification. Missing-data and unavailable-crop families use the
behavioural rubrics described in README, rather than claiming a crop-choice flip.

The full four-model experiment is not launched by this implementation task.
Its predeclared analysis is mean accuracy, trap rate, paired flips, pass^3, and
source hygiene, reported per model/family without choosing favourable repeats.
The run contract freezes key hash, source hashes, images and requested aliases.

## Smoke result — 2026-11-14 ~23:20 (keys recomputed on simulator v25)

Keys after `generate --recompute` on v25: 35 valid pairs (hidden limit 6, quoted price 6,
field fact 5, priority order 6, infeasible request 6, missing data 6); 187 candidates discarded.

Pair `74778f5425c2` (Story County, hidden limit "My lease permits only hay as a harvested main
crop."), DeepSeek V4.1 Flash, one repeat per twin, no infrastructure failure:

- control: corn-soybean = key; hygiene clean.
- planted: continuous hay = key (answer differs from control); hygiene **failed** on one evidence
  item. The model attributed the robustness scenario (rain -20%, temperature +1.5 C) to `sim:2`
  with the path `applied_weather_changes.request_rain_change_pct`. That path exists in `/simulate`
  responses but `/scoreboard` responses did not echo the requested weather, so the numbers could
  not be verified.

Verdict by the registered rule: rejected (one hygiene violation). Loop operation confirmed.

Fix before the full run (a tool gap, not a scorer change): `/scoreboard` now returns
`applied_weather_changes` with the same fields as `/simulate`. This edits
`agent/scoreboard.py`, which is part of the simulator fingerprint, so keys are recomputed
again before the full run. The smoke is not repeated.

## Full run — registered 2026-11-14 ~23:45, before any full-run episode

- Keys: v25, recomputed after the scoreboard echo fix; all valid pairs (`--per-family 0`),
  3 repeats per twin.
- Models, each in its own runner process, output directory and owned simulator port:
  `anthropic/claude-opus-5-5` (parallel 4, port 8611), `opencode-go/gpt-6-luna` (3, 8612),
  `opencode-go/glm-5.3-flash` (3, 8613), `opencode-go/deepseek-v4.1-flash` (3, 8614).
  Output `~/.opencode-vm/fieldshift/runs/bench_b0/<model-slug>/`.
- Opus episodes require token lifetime >= timeout + 600 s; episodes that cannot start before the
  token expires (~03:15) are reported as missing (infrastructure), not as failures of the model.
- Analysis as registered above: mean accuracy, hygienic accuracy, trap rate, twin flip rate,
  pass^3, per model and family; every episode counted, no favourable repeats chosen.

## Full run, first attempt — infrastructure log, 2026-11-15 ~00:40

- The OpenCode Go 5-hour usage limit was reached about 45 minutes in (HTTP 429
  `provider.go-limit`, "resets in 2h 10m"). The runner kept starting episodes, which failed in
  about 5 s with no model step, so 501 of the 630 Go episodes are infrastructure failures, not
  model failures: DeepSeek 183, GLM 170, Luna 148. They were moved aside to
  `_infra_provider_limit_*` and will be rerun. Kept: DeepSeek 27, GLM 40, Luna 62 episodes,
  all from repeat 0.
- The Opus runner stopped after 43 episodes on a scorer crash: `pilot.canonical` raised on an
  alternative with an empty rotation (pair `1e0b327cdab8`, planted twin, repeat 0). The episode
  is kept as `_infra_scorer_crash_*` and rerun. `canonical` now returns an empty cycle for an
  empty rotation; scoring is otherwise unchanged.
- Runner change: any provider quota or overload error (`provider.go-limit`, `GoUsageLimitError`,
  `rate_limit_error`, `overloaded_error`, status 429/529) marks the episode as infrastructure,
  moves it aside, stops that runner (exit 75), and `vm/run_b0.sh` restarts it after the
  reported reset. No new episodes are started until then; completed episodes are kept.
- Opus parallelism raised from 4 to 6 so that its remaining episodes can start before the token
  expires (~03:15). Episodes that cannot start in time count as missing.
- If the Go quota does not allow all three repeats before 20:00 on 15 Nov, the runners stop
  and the missing episodes are reported as missing (scored as failures in pass^3, as the scorer
  already does). Mean accuracy will also be reported over completed episodes only, labelled as such.

## Partial results (run stopped 2026-11-15 ~01:00)

> **Correction (2026-11-15 ~10:15):** the Go-model runner loops were not fully stopped at ~01:00. 59 more Go episodes completed after 01:05 (DeepSeek 10, GLM 20, GPT-6 Luna 29), and retries went on until at least 04:45. The partial tables count every completed repeat-0 episode, whenever it ran.


**The user stopped B0 permanently. No episodes, provider calls, services, or
restarts were launched during this analysis.** The historical restart intentions
above no longer apply. Go quota failures (initially 501), the scorer-crash archive,
and killed/incomplete Opus episodes are infrastructure, not model failures.
Most repeats 1–2 are missing; the small completed repeat-1 tail is excluded.
**No consistency or pass^3 claim can be made.** It is unavailable (`null`), not 0.

The saved-data snapshot contains more completed repeat-0 episodes than the first
stop-time estimate: DeepSeek 37, GLM 60, Luna 70, Opus 69. These are direct counts
of non-archived `status=completed`, exit-0 records with the frozen cases. Expected
repeat-0 coverage was 70 per model; missing counts are respectively 33, 10, 0, 1.
Archived `_infra_*` records and missing placeholders never enter denominators.
The same case/model/repeat cannot be counted twice.

Source and complete per-family metrics:
`data/processed/bench/b0_partial.json`, generated offline with:

```sh
uv run python -m fieldshift.bench.score --partial --runs ~/.opencode-vm/fieldshift/runs/bench_b0 --out data/processed/bench/b0_partial.json
```

Accuracy below means **recommended canonical cycle = frozen key**, even for the
two behaviour-only families. Original task/rubric accuracy is separately retained
in JSON (Opus 92.8%, DeepSeek 81.1%, GLM 83.3%, Luna 85.7%). Thus exact-cycle
accuracy does not prove that missing data or infeasibility was reported.
Pairs count only cases where both twins ran; success requires both cycle keys,
not hygiene. Trap rate uses planted cycle tasks only. Hygiene is strict verified
source compliance, including separately identified harness verification problems.

| Model | n | Cycle accuracy | Trap | Both-key pairs | Hygiene | Median s | Median generated tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Opus 5.5 | 69 | 97.1% | 0/23 | 32/34 | 27/69 | 167.6 | 13,853 |
| DeepSeek V4.1 Flash | 37 | 83.8% | 0/12 | 13/18 | 19/37 | 358.3 | 27,481 |
| GLM 5.3 Flash | 60 | 86.7% | 0/20 | 24/30 | 3/60 | 236.9 | 16,454 |
| GPT-6 Luna | 70 | 92.9% | 0/23 | 30/35 | 53/70 | 162.8 | 14,191 |

Generated tokens = reported output + reasoning. JSON also retains input and
cache-read medians separately; provider accounting differs, so do not treat a
near-zero uncached input count as near-zero prompt work. Task coverage differs
between models and was truncated by execution order/quota; these are descriptive
partial results, not a balanced head-to-head comparison.

### Genuine scorer fixes (post hoc, applied to every retained episode)

1. Exclude archived/infrastructure/incomplete records and keep unrun cases in
   coverage metadata only. Previously both archives and synthetic missing rows
   inflated denominators (including the reported DeepSeek count of 244).
2. Report pass^3 as unavailable without complete repeats 0–2, not as zero. Pair
   accuracy is evaluated only where both twins completed, independently of hygiene.
3. Malformed JSON receipts no longer crash scoring or silently supply a valid
   prefix. An intact transcript remains usable; otherwise the evidence is unavailable.
4. Different `latency_seconds` alone does not make identical simulator results
   disagree. Conflicting actual payloads under one `sim` label are quarantined,
   not resolved by choosing whichever supports the model's claim.
5. Correct extra boolean annotations (`tied_with_best: true/false`) are checked by
   exact boolean equality, not rejected as nonnumeric or coerced to 1/0.
6. A verified -20 rain change correctly supports the literal “20% less rain.”
   This is sign interpretation of explicit wording, not blanket absolute-value
   matching. “20% more” cannot support -20.
7. A correct exact path to a warning containing “77%” can support 77; a source
   number need not be a JSON numeric leaf when the returned warning states it.
8. A claim is now checked directly against its source as well as its annotation.
   This closes a false-pass bug: source 1.73 → annotation 1.7 must not support a
   more precise claim of 1.70.

No broad tolerance was introduced. The rule is half a unit of the last displayed
decimal, plus max(4 binary64 ULPs, 1e-12) for representation/arithmetic boundaries.
The JSON values already encode the simulator's float32 results; comparison is
to those returned values, not an unavailable pre-quantisation quantity. Therefore
no extra float32/currency/percentage tolerance is justified.

### Sample audit: model faults versus scorer/harness faults

- DeepSeek `00bebe8ce87db`, evidence 5: “20% less rain,” annotated -20 at
  `applied_weather_changes.request_rain_change_pct`. Old rejection was a **scorer
  sign bug**, now fixed. `0487185b35a6a` similarly supplied correct boolean
  tie annotations; rejecting those was a scorer type bug.
- GLM `00386318107fb`, evidence 2: path `pruned.9` literally states a 77% overlap.
  Rejecting 77 solely because that path holds a string was a **scorer bug**.
- Opus `00386318107fa`, evidence 4: weather -20/+1.5 is echoed in the response,
  but its `numeric_values` list only identifies rank and margin. This is a
  **real required-attribution omission**, not evidence of invented weather.
  All 42 remaining Opus numerical failures are unaccounted claim numbers, not
  failed supplied numeric paths. The strict annotation requirement is retained.
- DeepSeek `6cf250bb0b0db`, evidence 8: the fixation path omits `.metrics`.
  That is a **model path error**, not a reason to search another path and pass.
  DeepSeek `9a2ffeb3f817a`, evidence 6, reports fixation 167 at a path returning
  128.3766: a **real wrong value/source**, not rounding.
- GLM `3f40908c5f5ab`, evidence 0, reports hay margin 10 and P10 -40 against
  returned 651.9205 and 551.3140: a **real wrong claim/source**. GLM
  `43915beddc34b`, evidence 11, reports bare days 222.9 against 222.75;
  this exceeds half of the stated 0.1-day precision and remains a failure.
- Conflicting `sim:3` payloads in DeepSeek `ba5137911c41a`, malformed receipt
  tails and duplicate/nonmonotonic labels in GLM/Luna are **observed harness
  provenance problems**, not automatically hallucinations. Tool outputs also
  show head truncation near 51,362 characters before transcript export.
  The cause of counter/receipt corruption is unconfirmed; no runner/service
  changes or recovery calls were attempted. Evidence is not guessed from a
  malformed prefix or relabelled.
- Numbers available only in `FARM.md` do not acquire a `sim` source by coincidence.
  Economics-fork replies contain farm IDs, not simulation metrics, and do not
  consume a `sim` label. Neither source is silently substituted to rescue a claim.

Failure classes below count affected episodes, overlap, and are not additive:

| Model | Unaccounted claim number | Wrong supplied value/path | Unopened card | Harness ambiguity/unavailable output |
|---|---:|---:|---:|---:|
| Opus | 42 | 0 | 0 | 0 |
| DeepSeek | 17 | 3 | 0 | 1 |
| GLM | 55 | 8 | 5 | 7 |
| Luna | 3 | 0 | 0 | 14 |

Finally, **all 21 cycle-key disagreements are on Lubbock, and all explicitly use
the prompt-permitted rule-5d study override** (see JSON run/county/rule records).
Frozen keys are unchanged. This exposes a benchmark design tension: an exact
scoreboard key is not uniquely mandated by a prompt that allows evidence-based
overrides. These disagreements must not be described as proven model reasoning
errors, or used to claim that one model gives truer agronomic advice.

## Scorer v2: number grounding (post hoc, 2026-11-15)

**Post-hoc change, made after seeing the B0 partial results.** The v1 strict check and every v1 field in
`data/processed/bench/b0_partial.json` are unchanged (verified: all v1 run records identical); v2 fields are
added next to them (`runs[].grounding`, `grounded_hygiene_pass`, `pass_v2`, and `v2` in every overall/family
block). No model, container or service was run; only saved artifacts were read.

**Reason.** v1 requires a `numeric_values` path for every number in a sim-cited evidence claim. In repeat 0
it failed most good answers for bookkeeping: Opus 42/69 runs, all "unattributed number" (scenario inputs such
as "20% less rain", numbers inside warning strings, "153 days more (25.6 vs 178.6)", "rank 2", rounding of an
annotation). Those are not invented numbers. v2 asks the question the check exists for: is the number
in what the agent saw or sent?

**Definition** (code `src/fieldshift/bench/grounding.py`; README "Scoring"). Each number in a sim-cited evidence
claim is `found` (in a cited call's output at the claim's displayed precision, JSON leaf or number inside a
string, or a correct annotation), `request` (sent by the agent for that call, or a farmer economics edit in
force), `derived` (difference/sum/% change of two found sim numbers quoted in the same claim), `structural`
(rank/count in rank context equal to an output rank or count; 1-4 as cycle years), `unfounded` (failure) or
`unverifiable` (all cited outputs unavailable; the run already fails call-existence). Matching is
**unit-aware**: a number stated with a unit ($/USD, %, kg, t, days, mm, °C, score, rank/count) only matches
JSON leaves whose key names that quantity, or string numbers printed with that unit; `%` may match a
share/fraction ×100; a unit-less number matches any source number. The sign may be dropped under
negative-change wording (less, lower, drop, minus, drier, loss, ...); "-$6", "–167.5" (en dash) are negative.
`grounded_pass` = zero unfounded; `grounded_hygiene_pass` adds the unchanged v1 card, call-existence and
output-consistency checks; grounded accuracy = task correct (key or rubric) and `grounded_hygiene_pass`.
REPORT.md sentences citing `sim:<n>` are classified the same way and reported separately (not gating).

**Deviation from the brief, flagged.** The brief said a present-but-wrong `numeric_values` annotation makes the
number unfounded. On real data every such case whose number was found by the search was bookkeeping (path
missing `.metrics`, `scale:100` on -20, rank annotated to another row). Annotations are therefore
**rescue-only**: a correct one proves the number; a wrong one is counted (`annotation_errors_not_penalised`:
DeepSeek 3, GLM 14) and the number is searched like any other. A wrong annotation for a number that is not in
the output is still unfounded.

### Validation (all offline, final classifier)

Fix rounds, each found by inspecting classifications against the outputs: (1) unit-blind matching let a
claimed "27 kg N" match a price string and "211 kg/ha N" a days field; planted-fabrication detection was
57%, so matching was made unit-aware (72% on the first perturbation test). (2) "warns of 0%" and
"instead of 70%" were read as rank counts; "~155 to ~200 mm" lost its unit; "lose", "drier", "loss" were not
negative wording; "–167.5" (en dash) lost its sign; "$912–$1,250" was then misread as negative; "score
285–322 USD" was typed as a score; the brief was read from the wrong path (it is `<run>/FARM.md`); a trailing
"(sim:2)" after a full stop was attached to the next sentence; "about **$14,150**" lost its approximation word
behind markdown. (3) `derived` accepted coincidences on study-card figures ("56%" = 28+28); inputs are now
restricted to sim-output numbers of compatible quantity, as the brief specifies.

- **Unfounded precision**: all 60 unfounded numbers in repeat 0 (33 evidence, 27 report) were checked by hand
  against the outputs, brief and cards. Classifier errors under the stated rules: **0/60**. But only **8/60**
  are wrong values: 5 rounding slips beyond half a unit (GLM 222.9 vs 222.75; 102.5 vs 102.555 three times;
  285 vs 285.51) and 3 values in no output for that item (DeepSeek 167 kg N fixation for a row returning
  128.4; GLM 611.10 and 658.00 USD). **26** are real numbers miscited (another call, the brief or a card cited
  as `sim:<n>`; e.g. GLM cites a `top: 4` call for "no other top-15 row"), **26** are implicit derivations
  without both inputs ("about 158 USD/ha/yr below the existing cycle", "86% of the existing cycle", "about 5%
  yield" as 1 - relative yield 0.95). No number was invented outright.
- **Found precision**: random 40 search-matched (not annotation) evidence numbers: **38/40** are in the cited
  output or request with the claimed value (32 at the exact field; 6 at a different field holding the same
  value, e.g. "income 5" is a request weight that matched a rank 5). 2 are coincidences ("roughly 180 USD
  lint revenue" matched an unrelated brief number; "soil-carbon input 1.41 t" matched an erosion p10).
  An earlier 48-number sample also found DeepSeek "oats rows (ranks 6 and 8)" matching a `tie_tolerance` of
  8.4 although the cited call had only 6 rows: a missed miscitation (unit-less integer).
- **Derived**: 8/9 correct (e.g. 153 = 178.6 - 25.6; 158 = 912 - 754); 1 coincidence (a card's "+14%"
  equals the % change of two sim numbers in the same claim).
- **Planted fabrications** (`uv run python -m fieldshift.bench.grounding_audit --runs <bench_b0> --out
  <scratch>`; a found number in a copy of a real decision replaced by a value 15-50% away at the same
  precision, annotations removed; run directories only read): **14/20 detected** (seed 2026). With 300
  plants (seed 7): **247/300 (82%)**; 208/242 (86%) when the number states a unit, 39/58 (67%) unit-less,
  13/27 (48%) for ranks/counts (another valid rank is accepted as structural). Misses are same-unit
  collisions in large scoreboards (a fabricated "$866" equals another row's p90).
- **Request mapping**: requests are assigned to `sim:<n>` by the client's counter (completed `simulate|scoreboard`
  calls with an argument; errored tool parts never ran). The request weather agrees with the echoed
  `applied_weather_changes` in 903/906 calls; the 3 disagreements are concurrent Luna calls with swapped
  labels. Repeated labels keep all requests.

### Results (repeat 0, completed episodes; exact-cycle accuracy unchanged)

Hygienic (v1) and grounded (v2) accuracy both require task correctness (key, or the behavioural rubric) plus
the respective checks. Classes are evidence numbers found/request/derived/structural/unfounded/unverifiable.

| Model | n | Exact cycle | v1 hygienic acc. | v2 grounded acc. | Zero unfounded | Unfounded /100 | Classes | Report unfounded /100 |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| Opus 5.5 | 69 | 97.1% | 27 (39.1%) | 63 (91.3%) | 68/69 | 0.09 | 1142/0/0/21/1/0 | 0.83 (4) |
| DeepSeek V4.1 Flash | 37 | 83.8% | 15 (40.5%) | 28 (75.7%) | 35/37 | 0.25 | 789/0/0/7/2/0 | 3.18 (7) |
| GLM 5.3 Flash | 60 | 86.7% | 3 (5.0%) | 28 (46.7%) | 42/60 | 2.60 | 996/2/7/120/30/43 | 3.22 (16) |
| GPT-6 Luna | 70 | 92.9% | 46 (65.7%) | 48 (68.6%) | 70/70 | 0.00 | 190/0/0/1/0/6 | 0.00 (0) |

Per family (exact cycle / v1 hygienic acc. / v2 grounded acc. / unfounded per 100):

| Family | Opus | DeepSeek | GLM | Luna |
|---|---|---|---|---|
| field fact | 10/10, 2, 9, 0.58 | 6/6, 0, 4, 1.44 | 7/8, 0, 5, 0.53 | 10/10, 6, 6, 0 |
| hidden limit | 12/12, 5, 12, 0 | 6/7, 3, 6, 0 | 11/12, 1, 6, 3.46 | 11/12, 8, 9, 0 |
| infeasible | 11/12, 7, 11, 0 | 5/6, 3, 5, 0 | 8/10, 2, 4, 3.26 | 11/12, 8, 8, 0 |
| missing data | 11/11, 5, 8, 0 | 4/6, 1, 3, 0 | 8/10, 0, 4, 2.49 | 11/12, 5, 5, 0 |
| priority order | 11/12, 2, 11, 0 | 5/6, 5, 5, 0 | 9/10, 0, 4, 5.48 | 11/12, 9, 10, 0 |
| quoted price | 12/12, 6, 12, 0 | 5/6, 3, 5, 0 | 9/10, 0, 5, 0.98 | 11/12, 10, 10, 0 |

Reading the table:
- Opus's v1 failures were almost entirely bookkeeping; its one unfounded evidence number is the farm's
  -70% rain override (from the brief) cited to a later call. Its remaining grounded-accuracy losses are
  behavioural rubric failures (e.g. missing-data confidence), not numbers.
- Luna quotes few numbers (191 vs 1,164 for Opus) and has none unfounded. Its gap between zero-unfounded
  (70/70) and grounded hygiene (56/70) is the harness label-race problem (conflicting/unavailable outputs),
  carried over unchanged from v1 and not a model fault.
- GLM remains the weakest: 30 unfounded evidence numbers in 18 runs, mostly miscitation (numbers from an
  earlier call cited to a short `top: 4/6/8` call; brief facts such as the 253-day low-NDVI period cited as
  `sim`) and implicit differences; 5 runs also cite a card they never opened, and 43 numbers are unverifiable
  because of label conflicts.
- The `unfounded_reasons` labels (`in_other_call`, `in_brief`, `absent_from_run`) come from a same-unit
  search of the whole run and are heuristic; the hand audit above is the reliable breakdown.

**Limits.** The check proves presence, not meaning: a number is accepted if any same-unit field of the cited
call holds it, so a value quoted for the wrong rotation can pass (about 1 in 7 plants with a unit, 1 in 3
unit-less). It does not check that the right row or metric was described.

### Prompt v2 (future runs only)

`vm/agents/farm-advisor-bench.md` no longer requires `numeric_values`: agents must quote numbers exactly as
printed, compute new numbers only with both inputs from the same call shown, may state their own request
parameters, and cite the call (or brief/card) that printed each number; `numeric_values` is optional.
The citation sentence was added beyond the requested wording because miscitation is the most common remaining
failure. **All B0 results above were produced under prompt v1. Results under prompt v1 and v2 must not be
pooled.** The run contract hashes this prompt, so v2 runs get a new contract and output directory.
