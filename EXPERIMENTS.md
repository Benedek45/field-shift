# Yield response experiments

## Round 2 plan — 2026-11-13

Previous three-crop reference: year-grouped MLP R² corn 0.2724, soybean
0.1615, winter wheat 0.1011; stage trees 0.2829/0.1901/0.1298.
MLP held-crop R² −1.4269/−0.5130/−8.2947. Seed-only spread was not calibrated.

Before any major fit, record its data, folds and runtime budget here.
All normalisation, yield trends and uncertainty recalibration must exclude
outer test years. Crop-heldout tests must exclude that crop from response
training and calibration; its historical yields may supply its county normal.

### Hypotheses and acceptance criteria

- **H1:** More crops improve transfer. Accept if macro-average held-crop R² >0;
  report every crop, not just the mean. Corn and corn-silage must be held out
  together to avoid near-duplicate trait/label leakage.
- **H2:** Longer history plus irrigation improves drought severity. Accept if
  predicted 2012 corn loss is closer to observed loss than the previous 27.38%
  MLP prediction, on a common evaluation cohort; report 1983/1988 separately.
- **H3:** Heteroscedastic likelihood plus inner-year calibration provides
  predictive spread. Accept if per-crop ±1/±2 SD coverage is within 5 percentage
  points of 68%/95%; report failures and 2012 separately.
- **H4:** A trait-conditioned temporal CNN beats stage trees. Accept if macro
  within-crop R² is higher on exactly the same test rows, without test tuning.
- **H5:** POWER is no worse than AgERA5. Accept if common-cohort 2003–2023 macro
  R² drops by no more than 0.01 and MAPE rises by no more than 1 percentage point.

### Planned protocol

Year-grouped outer folds; deterministic inner held-out years for calibration,
independent of outer test years. Equal-county and harvested-area-weighted metrics.
Strict share=0 rainfed subset. Separate ALL-practice primary cohort from explicit
irrigated/nonirrigated strata to avoid counting the same county crop three times.
All training/inference tensors stay on MPS (CUDA on an authorised Kaggle run).
No per-batch host data transfer. Local epoch timing determines whether the full
sequence-CV ensemble exceeds the two-hour threshold.

### Infrastructure

Implementer delegation failed: harness reports "Subagent depth limit reached (1)".
No configuration changes or permission bypasses attempted; work continues inline.
POWER file was not present at round start. Develop first against CY-Bench.

### R2-A — expanded CY-Bench pipeline and local runtime benchmark

Hypothesis: the expanded stage ensemble is locally tractable and a five-member
CNN can run without host transfers. Acceptance: all parameters, optimiser state,
weather and gathered batches stay on MPS; extrapolate measured epoch time to the
full evaluation before choosing compute.

Data preparation succeeded: 634,236 survey labels across the 20 requested crop
records, 61.35% known irrigation, essentially complete harvested-area joins.
The raw file contains 9,560,646 Census and 14,387,933 Survey records; harvested
county Census extract has 1,012,549 rows. Exact sources:
`data/processed/yield_expanded/{census_audit,label_audit}.json`.

First benchmark failed, not a valid experiment: MPS rejects adaptive average
pooling from length 14 to 4. Replaced it with four disjoint temporal means, all on
MPS; no CPU fallback. Rerun pending. Forage end-of-season now continues to the
first killing frost after August or December 31 (not the annual October cap).
Potential cut cycles are thermal opportunities, never claimed observed cuts.

Rerun succeeded on MPS: 158,056 training rows, MLP epoch times 0.390/0.085 s;
CNN 4.656/3.637 s. Source: benchmark JSON under the CY-Bench feature-cache
directory and `data/processed/yield_expanded_benchmark_v2.log`.
The CNN has independent grouped-convolution weights per seed, trait FiLM and
four temporal bins. All county-day data are float16 on-device, gathered per
batch on-device; arithmetic, parameters and optimiser moments use float32.
No CUDA/Kaggle run is justified for this CY-Bench workload.

### R2-B — twenty-crop calibrated stage ensemble, CY-Bench

Pre-run specification: five grouped-year folds, 2003–2023, 40 epochs, five seeds,
128/64 hidden units, Gaussian mean+variance likelihood after a fixed mean warmup.
Inner calibration years are every fifth sorted non-test year. Trends, scalers
and moisture climatology use only the remaining fit years. No refitting on
calibration years. H1/H3 are tested now; H2 is provisional pending POWER. H4
follows as a separately timed CNN experiment. H5 waits for the new archive.

Every crop is tested without its response labels on one prespecified outer
fold (family index modulo five). Corn and corn-silage are held out together;
all wheat classes are held out together. The held crop's fit-year yields supply
only its normal, never the response/scaler/calibration. Primary metrics use
ALL-practice county rows to avoid counting county aggregates and strata twice.
Area-weighted, all-strata and exact share=0 summaries are saved separately.
Paired old-cohort results recenter predictions onto the original county normal,
so changes in normal-fitting do not masquerade as drought-response improvement.

Acceptance criteria: H1 macro held-crop R² >0; H3 per-crop 68/95 coverage within
five points. No hyperparameter adjustment based on the outer results.

Data-QC clarification before inspecting R2-B results: moisture climatology weights
must count only days with observed RSM, not all days in a stage. Otherwise winter
2003 (missing previous-autumn moisture) compares a spring-only measured mean to an
autumn+spring climatology. Feature cache version 2 corrects this exactly. The
running version-1 evaluation is preserved as an engineering pilot; headline
results must come from version 2. No parameter changes are motivated by outer
scores. Also de-duplicate exact-share-zero strata by county/crop/year, preferring
explicit NON-IRRIGATED when an ALL-practice aggregate is also zero-irrigation.

**R2-B pilot result (engineering, not final):** 195,291 rows; evaluation 391.96 s
on local MPS. Macro within-crop R²: trend −0.00248, season tree 0.07237, stage
tree 0.10436, heteroscedastic MLP 0.11291. Held-crop MLP mean −0.15137 versus
stage trees −0.00289: H1 rejected on this pilot. Corn/soy/winter-wheat held-crop
MLP R² 0.007/−0.040/−0.033 (large improvement over the original failure, not
positive general transfer). Matched 2012 corn: observed loss 43.1402%, predicted
24.5748%, worse than the original MLP's 27.3795%: H2 not supported by more crops
without older weather. Calibration improved greatly but H3 fails per-crop:
±1 SD 53.3–79.1%, ±2 SD 85.1–96.8%. Source:
`data/processed/yield_expanded/cybench_583f74052fd5/evaluation_5fold_40epoch_seq0_2003_2023/metrics.json`.
Conclusion: more crops alone is not sufficient; do not tune outer-fold scores.
Older weather and the sequence model remain the prespecified next steps.

One additional representation correction before the version-2 fit: established
multi-cut forage has no annual emergence/flowering/maturity thresholds. Those
traits and annual maturity/vernalisation diagnostics are now explicitly missing,
not inherited soybean values. Forage uses green-up, killing frost and regrowth
thermal opportunities. Root-crop shared stage slots have explicit explanation
aliases (storage-organ initiation/bulking). This is a physical-schema correction,
not a choice among alternative thresholds based on scores.

### R2-C — corrected CY-Bench sequence comparison

Pre-run: same 2003–2023 five folds, 40 epochs and five seeds. Feature cache v2,
including the representation corrections above. Fit all four expanded baselines
and the trait-FiLM CNN on the identical fit/calibration/test rows. H4 accepts
only if macro within-crop R² exceeds stage trees. Report H1/H3 for both neural
models without hiding negative crops. Estimate under one hour locally from the
measured CNN epoch time; no Kaggle launch planned for this run.

### R2-D — POWER full history, staged heteroscedastic ensemble

Pre-run (2026-11-14): POWER archive delivered as
`data/processed/weather/power_county_daily.f32` + JSON: 3,108 counties,
16,436 days (1981–2025), 14 variables, no missing values. Full-history 20-crop
five-fold CV, 40 epochs, five seeds, same inner-year calibration protocol.
H1/H2/H3 use the original acceptance criteria. Especially report 1983, 1988 and
2012; the 2012 old-cohort comparison uses the original normal and matched rows.
This independent data-history experiment runs concurrently with R2-C; no CNN
outer scores are available or used in its specification. MLP data stay on MPS;
the sequence comparison is already running on accelerator-resident CY-Bench.

### R2-E — matched POWER/CY-Bench weather comparison

Pre-run: 2003–2023, exact county/crop/practice/year intersection, identical grouped
folds, targets, calibration years and area weights. `power_matched` is a virtual
view of the immutable POWER memmap: VPD is converted from FAO-56 mean-saturation
VPD to CY-Bench's Tmax-saturation VPD by adding
0.5 × (es(Tmax) − es(Tmin)). The same conversion is implemented on-device for
sequence windows; no raw POWER file is changed. This removes the known ~0.6 kPa
definition confound. The full-history `power` model retains native FAO-56 VPD.
H5's primary model is the calibrated MLP: accept if macro R² drops by ≤0.01 and
macro MAPE rises by ≤1 percentage point. Report the trees separately. This compares
weather packages (including their different masks/moisture products), not a pure
spatial-resolution intervention or a formal statistical noninferiority test.

### Completed R2-C/D/E baseline results (2026-11-14)

R2-C: 195,291 rows, 3300.26 s evaluation (deployment fitting additional). Within-crop macro R²: {'trend': -0.0024756470868348645, 'season_tree': 0.07247985158700683, 'stage_tree': 0.10328530276742587, 'mlp': 0.11231265911513295, 'sequence': 0.11626772153009605}. Held-crop macro R²: {'trend': -0.04592001529857583, 'season_tree': -0.021049334874981323, 'stage_tree': -0.003014329259930615, 'mlp': -0.13931854746756295, 'sequence': -0.0673019319977334}. Source `data/processed/yield_expanded/cybench_8e14f4f0968b/evaluation_5fold_40epoch_seq1_2003_2023/metrics.json`.

R2-D: 630,528 rows, 1086.55 s evaluation (deployment fitting additional). Within-crop macro R²: {'trend': -0.00037335584468600747, 'season_tree': 0.12931713978716258, 'stage_tree': 0.1530354000500104, 'mlp': 0.17468258674105938}. Held-crop macro R²: {'trend': -0.035933872985821944, 'season_tree': 0.06682926381600605, 'stage_tree': 0.08682726080358799, 'mlp': -0.12752384382383516}. Source `data/processed/yield_expanded/power_8e14f4f0968b/evaluation_5fold_40epoch_seq0_1981_2025/metrics.json`.

R2-E: 195,291 rows, 380.32 s evaluation (deployment fitting additional). Within-crop macro R²: {'trend': -0.0024756470868348645, 'season_tree': 0.08423497081740183, 'stage_tree': 0.1128902486469027, 'mlp': 0.11798039557732562}. Held-crop macro R²: None. Source `data/processed/yield_expanded/power_matched_8e14f4f0968b/evaluation_5fold_40epoch_seq0_2003_2023_cohort_cybench/metrics.json`.

H5 matched-definition result: {'trend': {'delta_r2': 0.0, 'delta_mape_points': 0.0, 'not_materially_worse': True, 'area_weighted_mape_delta_points': 0.0}, 'season_tree': {'delta_r2': 0.011755119230394984, 'delta_mape_points': -0.21887489766786813, 'not_materially_worse': True, 'area_weighted_mape_delta_points': -0.2450308841648674}, 'stage_tree': {'delta_r2': 0.009604945879476768, 'delta_mape_points': -0.18826934330734452, 'not_materially_worse': True, 'area_weighted_mape_delta_points': -0.3579049354448305}, 'mlp': {'delta_r2': 0.005667736462192668, 'delta_mape_points': -0.3423201244843982, 'not_materially_worse': True, 'area_weighted_mape_delta_points': -0.5872081924108965}}. Full per-crop deltas in `data/processed/yield_expanded/weather_comparison.json`.

H7/H8 capacity and architecture sweeps were cancelled by the user before training. No H7/H8 experiment or Kaggle run was started. H6 thermal alignment is absorbed into R3 as a calculated development driver, not a separately selected architecture. Existing running baselines were preserved.

## R3 — recursive field-state emulator (pre-registration, 2026-11-14)

**One configuration, no sweep.** Five independently seeded members. Each has a
3-layer, 384-unit causal GRU, a 128-unit driver embedding, a state-conditioned
128-unit flux/update head and a 256-unit heteroscedastic yield readout. This is
roughly 2.7 million parameters/member (exact count logged by the implementation),
substantially larger than R2's 128/64 MLP. A fused GRU processes the calculated
drivers; a differentiable feedback loop carries named states and predicts fluxes.
This block-triangular recurrent design keeps named-state feedback while avoiding
hundreds of separate large GRU kernel launches per batch. The latent GRU carries
weather history; the flux head consumes its context and the previous named state.

**Data and timing.** Fixed MODIS periods: 46/year, 8 days except the last short
period. POWER daily weather is uploaded once and aggregated on-device. Crop GDD,
winter dormancy/vernalisation, thermal stages, phase-specific heat, crop presence
and astronomical day length are calculated on-device, not inferred by the GRU.
Named states: MERRA root/surface/profile wetness; NDVI, LAI, FPAR, actual ET and a
surface-minus-air temperature proxy. CY-Bench crop-mask NDVI/FPAR are used where
available; new EO/static products are frozen in each uploaded data manifest.
No future wetness/canopy observation is a driver during free-running evaluation.

**Physics.** Three effective water reservoirs, fixed capacities 180/40/400 mm for
root/surface/deep proxies unless a documented soil-capacity conversion is
available. These are operational capacities, NOT measured MERRA storage depths;
the three relative-wetness products are not independently measured water masses.
Precipitation and supplemental irrigation are the only external water inputs;
ET, runoff and deep drainage are non-negative, availability-limited outputs.
Internal percolation/capillary transfer cancels in total water mass. No ad-hoc
soil-water residual is permitted. Irrigated scenarios use a learned automatic
policy with a 60–90% root-wetness target and capped delivery; this is a stated
management assumption, not an observation of farmer irrigation. MERRA water loss
is downweighted by known irrigated share because a land-model proxy cannot be
treated as an observation of irrigation management. Unknown share has an input
missingness flag and no assumed supplemental water.

**Training, fixed before scores.** 24 epochs, 128 batches/epoch, batch 96, AdamW
lr 0.0003, weight decay 0.01, norm clip 5. CUDA mixed precision with accelerator
optimizer state. First four epochs teacher forced; next four scheduled sampling
down to zero; remaining epochs fully free-running. Alternate one- and two-year
batches; two-year training windows contain only adjacent fit years. Final
two-year losses backpropagate through the full 92-step rollout, not a detached
year boundary. One quarter of each batch samples county-state windows without
requiring a yield label; the rest sample yield rows. State loss is primary,
normalised masked MSE; yield loss is 0.5 times Gaussian NLL after mean warm-up.
No stopping rule, width, or epoch choice uses outer scores.

County observations do not reveal a field's rotation. Training uses representative
repeated-crop windows, explicitly imputed drivers rather than invented observed
rotations. The app can change crops between seasons, but nutrient/rotation
causality is not identified by these county data. New crop-progress records check
the sowing/stage priors; they do not silently tune thresholds against outer yields.

**Evaluation.** Exact R2-D outer year folds and inner calibration years; all
state climatologies and normal/scaler fits exclude test/calibration years.
Initial state is allowed at the beginning of each evaluation window; subsequent
observations are used only for scoring. Free-running and teacher-forced yield
scores are both retained. Report named-state RMSE by lead against persistence
and fit-year county/period climatology, including 6/46/92 steps.

To make full held-crop coverage feasible, the pre-existing R2 held groups assigned
to each year fold are united into five larger held-crop groups. Every crop is
held out once, with no labels or crop-mask state supervision for that held group.
This is a harder grouped-transfer test than R2-D's separate 17 fits, not a
like-for-like numerical claim of improvement on its transfer metric.

H9 accepts macro free-running within-crop R² >= 0.1746825867 - 0.02 (best R2-D
non-recursive model). H10 accepts state RMSE below fit-year climatology through
at least six 8-day leads, reported per slot rather than hiding failed channels.
H11 accepts only if rainfall -30% lowers rainfed wetness/canopy/yield with a
smaller high-irrigation response AND an equal heat perturbation hurts corn more
at flowering than vegetative growth. These are model self-consistency checks,
not independent causal validation of interventions. Re-test H2 on matched 2012
corn normals and report 1983/1988. Carry-over is evaluated with preceding
autumn/winter precipitation edits and a separate empirical conditional association.

Up to four private T4 sessions may split folds; both accounts were verified with
`kaggle config view` before staging. Account/kernel references and all failures
will be recorded below. Never cancel unrelated compute or publish source data.

**Pre-training implementation checks:** 14,073,355 ensemble parameters with the
fallback soil table (2,814,671/member). Accelerator-only MPS smoke training on two
batches completed in 0.95 s; state/yield losses finite and no CPU fallback.
Conservation, free-rollout observation exclusion and post-harvest causality tests
pass. This is an engineering check, not a scored model or architecture experiment.

Independent crop-progress diagnostics found the temperature-only sowing prior too
early: corn MAE 17.44 days, soybean 34.84 days, winter wheat 14.51 days, relative
to state 50%-planted dates (full results in `data/processed/yield_expanded/phenology_check.json`).
Before any recursive fit, use the observed state 50% planting date as a **given
management driver** where available; retain temperature rules elsewhere and for
perennial green-up. This is not tuning on yield/test errors and not a planting
forecast. The app accepts an explicit planting date. Accuracy against the same
management records after replacement is within daily/8-day resolution by construction, NOT independent
validation. Generic flowering/maturity GDD priors remain checked, not retrospectively
tuned. Corn silage shares corn planting, not its harvest threshold.
The first corn permutation attempt stopped before fitting because pandas returned
a read-only mask; replaced in-place Boolean assignment with a new Boolean array.

EO assimilation freezes the products actually present in each manifest. SMAP
root/surface and GLDAS root volumetric moisture, when present, are independent
masked targets with an explicit approximate observation operator: mean period
wetness × porosity, porosity from bulk density / mineral particle density 2.65,
default 0.45 if unavailable. Their normalisation is 0.06 m³/m³ and relative loss
weight 0.25. This is not a claim that MERRA wetness is volumetric moisture.
GRACE includes groundwater/other stores absent from this soil-only proxy, so it
is archived but not treated as a soil-water observation; nighttime LST is not
substituted for daytime canopy temperature. Non-assimilated channels are logged.
The EO agent's AVHRR reference harmonisation uses the full MODIS/AVHRR overlap;
optical reference-product calibration is therefore transductive even though all
model/scaler/climatology fits and predictive-spread recalibration exclude test years.

Snapshot ingestion includes the newly arrived soil/terrain table and MODIS
LAI/FPAR. Whole-record GRACE-trend columns are explicitly excluded from "static"
inputs to prevent future-derived hydrologic predictors; grid sample counts are
also excluded. Actual supplied county centroids replace coarse-grid fallbacks.

### R3 Kaggle schedule (registered before upload/launch)

Each account gets its own private `fieldshift-recursive-r3-20261114` dataset;
both usernames were rechecked with `kaggle config view`. All metadata use JSON
`is_private: true`, kernels also `enable_gpu: true`; every push explicitly selects
`NvidiaTeslaT4`, with a hard CUDA/T4 probe before allocation. These runs jointly
test H9/H10/H11 and re-test H2; no architecture/size sweep.

| Account | Kernel ref | Folds | Work |
|---|---|---|---|
| account-A | account-A/fieldshift-recursive-r3-fold0 | 0 | ordinary + union-held crops |
| account-A | account-A/fieldshift-recursive-r3-fold1 | 1 | ordinary + union-held crops |
| account-B | account-B/fieldshift-recursive-r3-fold2 | 2 | ordinary + union-held crops |
| account-B | account-B/fieldshift-recursive-r3-fold34 | 3,4 | ordinary + union-held crops |

Private local upload directories live under the approved session temp directory,
not the repository. Dataset source/input hashes are recorded in `manifest.json`.
Results, actual wall time and any infrastructure failures will be appended.

Pre-launch evaluation correction: two-year yield windows include a context year
that can be a fit year. Those context-year state outcomes must NOT count as
held-out state accuracy. State scoring explicitly masks to the outer test years
(inner calibration years for spread). Leads 1–46 use an independent Jan-1
initialisation in the held year; leads 47–92 use the continuous two-year rollout.
These are two initialisation cohorts, reported as such, not one secretly reset
trajectory. The yield trajectory is always continuously free-running throughout.

Dataset uploads were already in progress when this pre-launch evaluation check
was completed, so their bytes are preserved. Each not-yet-launched kernel embeds
a small, SHA-recorded source overlay containing the correction; no running job
or uploaded dataset is altered. T4 sessions normally provide two 16-GB GPUs:
ordinary and held-group fits run as independent processes on separate devices,
each with its own fully resident weather tensor. At most four sessions are used.
If only one T4 is available, the two fits run serially rather than oversubscribing
its memory. This is compute partitioning, not an architecture comparison.

### R3 post-launch tooling and deployment fit (2026-11-14, before any R3 score)

- Launch state: all four kernels in the table above reported `RUNNING` at
  02:43 and 02:47 CET. The `fold34` session exposes two GPUs (ordinary and held
  children both printed `device: cuda`). No kernel was re-pushed or cancelled.
- The cancelled H7/H8 scaffolding (`model/architectures.py`) was deleted before
  any use; `probabilistic.ARCHITECTURES` keeps only the three reference models.
- `model/recursive_report.py` merges fold outputs, refuses rows whose targets
  differ from R2-D (proves identical normals/folds), and reports state RMSE by
  lead for model/persistence/climatology per slot.
- `model/recursive_checks.py` (what-if, carry-over) passed an engineering smoke
  on an *untrained* model (16 cases, 19 s MPS): edits flow through recomputed
  8-day weather, GDD and stages. Its numbers are meaningless as results.
- Empirical carry-over association (offline, not a model, not causal): rainfed
  -dominated corn (all-practice series, Census irrigated share <= 5%, 26,154
  county-years, 743 counties, 1982-2025, R2-D targets). Within-county regression
  of log(yield/normal) on preceding Sep-Feb precipitation, controlling May-Sep
  precipitation, mean Tmax and ET0, county-clustered SE: **-0.0141 log-yield per
  +100 mm** (95% CI -0.0168 to -0.0114). Wetter autumn/winters go with *slightly
  lower* yields in these mostly humid counties; dry-winter carry-over is not a
  yield-limiting signal in this data. Year effects are not removed, so shared
  regional patterns (e.g. wet springs delaying planting) can confound it.
  File: `data/processed/recursive/empirical_carry_over.json`.
- Deployment model, fixed before any outer score: identical configuration
  trained on every year except the R2 inner-calibration rule (every fifth year,
  used for the SD multiplier and lead RMSE). It runs locally on MPS against the
  **frozen Kaggle input snapshot** (symlinked manifest inputs), because the EO
  product has since gained `et_actual`, `lst_day`, `lst_night` locally; the
  deployment model must match the cross-validated inputs. Those new channels are
  for a later round. Output `data/processed/recursive/r3_final/final.pt`,
  log `data/processed/recursive_final_mps.log`.

### R3 failure 1: post-training calibration crash (2026-11-14 03:15 CET)

- Observed: the local MPS deployment fit finished all 24 epochs (epoch 22: state
  loss 0.425, yield MSE 0.0375) and then raised in `calibrate`:
  `RuntimeError: Inplace update to inference tensor outside InferenceMode is not
  allowed` at `errors[:, :46], counts[:46] = short_errors[:, :46], ...`. Log kept as
  `data/processed/recursive_final_mps_failed_v1.log`. No checkpoint was written,
  because saving came after calibration.
- Cause (verified): `predict` is `@torch.inference_mode()`, so its returned
  `errors`/`counts` are inference tensors; the lead merge wrote into them in place
  in normal mode. Reproduced on a stub row source: the old pattern raises, the
  fixed path passes. The same code is in `evaluate_fold`, so **all four Kaggle v1
  kernels run identical code and will fail at the same point** after ~30 min of
  training per fit. Unit tests missed it because none called `calibrate`.
- Fix (a code bug, not tuning; no hyperparameter or data change): the merge moved
  into `lead_scores`, itself in inference mode, used by both `calibrate` and
  `evaluate_fold`. Uncalibrated weights are now saved as `{fold,held}_k_trained.pt`
  / `final_trained.pt` right after training, so a later failure no longer loses a
  fit. Regression test `test_calibration_outside_inference_mode_sets_buffers`.
- Action: the local deployment fit restarted at 03:19 with the fix. The v1 kernels
  are left to fail on their own (not cancelled); once each reports an error, it is
  re-pushed as v2 with the same configuration, accounts and data, and a source
  overlay that differs only by this fix. Both v1 and v2 will be logged.
- 03:19 `account-B/fieldshift-recursive-r3-fold34` v1 → `ERROR` with exactly this
  traceback in both children (`Fold 3 child failures: [1, 1]`), after 24 training
  epochs (~72 s/epoch). Account re-verified (`kaggle config view`: account-B);
  **v2 pushed 03:19** (`--accelerator NvidiaTeslaT4`, private, same dataset; push
  dir holds only `run.py` + `kernel-metadata.json`).
- 03:23 `account-A/fieldshift-recursive-r3-fold0` v1 → `ERROR`, same traceback
  (`Fold 0 child failures: [1, 1]`). Account re-verified (account-A); **v2 pushed 03:23**.
- 03:29 `account-B/fieldshift-recursive-r3-fold2` v1 → `ERROR`, same traceback
  (`Fold 2 child failures: [1, 1]`). Account re-verified (account-B); **v2 pushed
  03:29**. Decoded overlays of all v2 runners: `recursive_train.py` defines the
  inference-mode `lead_scores` and saves `*_trained.pt`.

### R3 deployment model (local MPS, completed 2026-11-14 03:48 CET)

- Trained on the frozen Kaggle input snapshot (symlinked; the local EO file has since
  grown to 7 channels, so the snapshot keeps static/driver names identical to the CV
  kernels). Fit years = all 1981-2025 except the inner calibration years
  1981, 1986, ..., 2021 (every fifth); no outer test years. 24 epochs, 1,563.6 s on
  MPS (51-88 s/epoch). Final epoch: state loss 0.421, yield MSE 0.0359.
  14,426,635 parameters (5 members). Inner-year yield SD multiplier 1.092 (101,691 rows).
- Saved as `data/processed/models/recursive_field_r3.pt` (copy of
  `data/processed/recursive/r3_final/final.pt`; uncalibrated `final_trained.pt` kept).
- `simulate(...)` engineering smoke with real POWER weather 2011-2012, corn-corn,
  initial state fixed by hand, county statics from the snapshot (in-sample years, so
  **not** an evaluation): relative yield mean (SD), 2011 / 2012:
  | County | rainfed | rain -30% | irrigated share 1 |
  |---|---|---|---|
  | Story IA | 1.018 (.129) / 0.694 (.200) | 1.036 / 0.658 | 0.987 / 0.927 |
  | York NE | 1.111 (.166) / 0.755 (.217) | 1.078 / 0.678 | 0.997 / 0.970 |
  | McLean IL | 0.982 (.115) / 0.687 (.179) | 0.987 / 0.660 | 0.984 / 0.904 |
  Mass-balance residual <= 3e-5 mm/period; 0.3-0.5 s per 2-year simulation.
  Note: Story IA's NASS record shows almost no 2012 loss (157.6 vs ~164 bu/acre), so a
  0.69 rainfed prediction there overstates loss, and the wet-year 2011 rain cut gave a
  slightly *higher* yield in Story/McLean. Both go to the formal what-if (H11) check.

### R3b preregistration: clean, expanded EO targets (2026-11-14 04:00 CET)

Data fault (reported by the main/data agents, independently of test scores):
the 02:10 snapshot leaked MODIS fill codes into county LAI/FPAR means. Snapshot
LAI max 25.50, 15,086 cells >8; FPAR max 2.55, 15,334 cells >1. Against the corrected
cube, 4.43%/3.90% differ by >1% of range and 1.85%/1.52% by >10%; 273 cells change
from finite to missing. NDVI is identical. All v2 fits and the first deployment
model used that snapshot. **Their LAI/FPAR lead scores are invalid**; yield/wetness
scores remain a contaminated-target comparison, not assumed equivalent to clean data.

R3b uses exactly the same architecture, optimizer, loss coefficients, curriculum,
24 epochs/128 batches/96 samples, seeds and R2-D folds. No hyperparameter tuning.
It uses the corrected nine-channel EO cube, adding actual ET, LST and auxiliary
SMAP targets that were absent from the snapshot. Consequently the v2/R3b difference
is the combined effect of correction and additional supervision, not an isolated
estimate of the fill-code bug.

Registered quality rule: before MODIS begins (2000-02-18), mask adjusted AVHRR NDVI
where the source harmonisation fit r2 <0.3 or missing (886 counties below 0.3).
This uses source-product quality, not yield outcomes; source harmonisation itself
uses the full overlap (retained caveat). Range-check all EO channels on ingestion,
fail rather than clip: NDVI [-1,1], source flag [0,1], LAI [0,8] (strict decoder/
operational bound), FPAR and volumetric moisture [0,1], ET [0,25] mm/day, LST
[-100,100] C. NaN is missing; infinity is an error. Unit conversions precede checks.

Do not cancel running v2 kernels. `fold1` v1 is now ERROR; its freed account slot
will run R3b instead of spending another fit on the defective snapshot. Other
R3b kernels launch as v2 slots free. Freeze a new private dataset per account;
retrain deployment on the same clean snapshot, retaining the old weights as
`recursive_field_r3_contaminated_v2.pt`. Report v2 and R3b side by side wherever
v2 completed; no v2 fold1 result will be invented.

R3b snapshot completed with **11** EO channels: the nine listed above plus newly
arrived `grace_tws`/`grace_observed`. GRACE is retained in the private input snapshot
but not assimilated: total terrestrial storage includes groundwater not represented
by the three soil proxy reservoirs, so treating it as soil water would be false.
Night LST is retained but not a separate predicted state. All nine assimilable
channels passed the new ingestion checks on MPS. The clean deployment fit starts
locally from this frozen snapshot, with identical settings; old deployment retained.

Independent checks registered before R3b scores: USDM D2-or-worse county area
versus free-running root wetness anomalies (negative association expected);
Iowa 2012 corn monthly response to replacing deficient rain/excess temperatures
with fit-year monthly climatology (May–September) versus RMA monthly drought
damage share of NASS planted area. These are external diagnostics, not inputs or
tuning objectives. Atmospheric VPD/ET0 are held fixed in the controlled weather
intervention; indemnity timing is not a precise physiological event timestamp.

### First free-running CV result: contaminated R3 v2 fold 0

`account-A/fieldshift-recursive-r3-fold0` v2 COMPLETE. Ordinary fold took
2,445.3 s (40.8 min), per `recursive_kaggle_fold0_v2.log`. Macro R2 = 0.075479
free-running, 0.069104 teacher-forced. On exactly these R2-D test years, stage trees
= 0.087534, MLP = 0.088690, season trees = 0.054497. Thus below both leading
baselines on this fold; not a full-CV H9 verdict. Do not
interpret the teacher/free reversal as absence of error accumulation generally.
LAI/FPAR state scores invalid (contaminated targets). Interim sent immediately
after the first score was read. Outputs downloading for full paired analysis.

R3b infrastructure launch fault: initial `account-A/fieldshift-recursive-r3b-fold0`
and `...-fold1` v1 pushes returned success **and** “not valid dataset sources and
could not be added”. The private dataset was still being published after upload.
Do not count those as training runs or overwrite them while RUNNING. The safe
correction is to require `kaggle datasets status <ref>` = `ready` before retrying
after ERROR. Subsequent dataset status is now `ready`. Accounts were verified
before each push; privacy booleans and T4 accelerator explicit. No model/data/
hyperparameter change. The existing v2 jobs are unaffected.

Correction after row-level merge: the earlier baseline figures included overlapping
irrigation strata. On **identical primary rows and county normals**, fold-0 macro
R2 is recursive 0.075479, stage trees 0.096390, MLP 0.090859, season trees 0.064959.
The recursive score is unchanged (gap to best baseline -0.020911).
`data/processed/recursive/r3/metrics.json` is authoritative.

Both R3b v1 launch stubs now ERROR. Fold-0 log confirms `CUDA_PROBE ... Tesla T4`
then `Expected one private data manifest; got []`: no training occurred. Main
dataset status independently verified `ready`, account re-verified as account-A.
Retry the same kernels with the now-published private dataset; no running job cancelled.

2026-11-14 04:26 CET: both `account-A/fieldshift-recursive-r3b-fold0` and
`...-fold1` v2 pushed successfully with the now-ready dataset (no invalid-source
warning). Same private metadata and explicit NvidiaTeslaT4 accelerator. Their v1
stubs did not train; source overlay/data unchanged.

### Clean R3b deployment completed (2026-11-14, local MPS)

`data/processed/recursive/r3b_final/final.json`: 1,604.55 s (26.7 min), same
14,426,635 parameters, 24 epochs; final state loss 0.4509, yield MSE 0.03536.
Inner-year SD multiplier 1.11474 (101,691 rows). Default model replaced atomically;
contaminated weights retained separately. This is a deployment fit, **not** an
independent yield score. Source snapshot contains 11 EO channels; night LST/GRACE
not assimilated, pre-MODIS low-quality NDVI rule recorded in model metadata.

Actual R3 phenology diagnostic is saved at
`data/processed/yield_expanded/recursive_phenology_check.json` (+ paired Parquet).
Given state 50%-planted dates agree within <0.5 d (rounding, not independent
prediction). Independent corn flowering MAE 4.04 d (866 state-years), maturity
10.89 d (807); winter-wheat heading-proxy 14.24 d (815), maturity 17.43 d (99);
spring-wheat heading-proxy 17.80 d (234), maturity 41.88 d (11).
Soy flowering MAE 12.68 d (914), maturity 10.06 d **only for 55 observed/reached
state-years**. Only 52.9% of soybean and 31.8% of sunflower county seasons reach
the generic maturity threshold; the conditional date error hides failures to
reach maturity. Traits remain unchanged for this registered comparison. Forage
maturity is not applicable (JSON null), not a failed annual crop.

Clean deployment engineering smoke (`recursive_r3b_sim_smoke.log`, same three
counties/weather/hand initial state as v2; in-sample, not CV): Story IA corn 2012
rainfed relative mean 0.712, rainfall -30% 0.657, irrigated share 1 mean 0.874.
York NE 0.806/0.708/0.955; McLean IL 0.672/0.637/0.836. Water residual <=3.1e-5
mm/period; 0.3–0.4 s per two-year scenario. Story/McLean wet-year 2011 still show
small yield increases after rain cuts, so these examples do not establish H11.

Verification after the clean deployment: `uv run pytest -q`: **152 passed** (6.70 s);
`uv run ruff check .`: **all checks passed**.

### R3b compute allocation update (before fold-4 training)

Run ordinary + held fold 4 sequentially on the now-idle local MPS, from the same
clean frozen input snapshot and exact R2-D fold-design JSON. Other clean folds
use Kaggle T4 as slots free. This avoids waiting for the two still-running v2
fallback-account kernels followed by another serial clean fold. Same architecture,
seeds, epochs, batch counts, optimizer settings and curriculum. MPS uses the
already-implemented device AdamW without AMP; CUDA uses fused AdamW with AMP, so
paired version differences are not a perfectly isolated data-only intervention.
No duplicate fold-4 Kaggle job will be launched.

2026-11-14 04:59 CET: contaminated v2 `account-B/fieldshift-recursive-r3-fold34` COMPLETE (folds 3/4 ordinary+held); outputs downloading. Freed slot assigned to `account-B/fieldshift-recursive-r3b-fold3` v1, private/T4 explicit. Account re-verified; clean private dataset status `ready` before push. Fold 4 remains local MPS; no duplicate.

2026-11-14 05:08 CET: contaminated v2 `account-B/fieldshift-recursive-r3-fold2` COMPLETE; outputs downloading. Ordinary fit/evaluation 2,472.82 s, macro free R2 0.129009 (per follower log). Freed slot assigned to `account-B/fieldshift-recursive-r3b-fold2` v1; account re-verified, dataset ready, private and explicit T4. All five clean folds are now allocated (0/1/2/3 Kaggle, 4 local).

Reporting-only fix: numbered `held_*.parquet` selection excludes its own aggregate `held_crop_predictions.parquet`; repeated merge now has a regression test. Training and row predictions unchanged. Partial contaminated folds 0/3/4 paired-primary macro R2: recursive 0.104917, MLP 0.122492, stage trees 0.113427; held-crop recursive 0.045495 vs stage 0.062518 (not full CV).

Secondary carry-over sensitivity registered before running: add year fixed effects to the existing county-effects association (same <=5% irrigation corn subset and May-Sep P/Tmax/ET0 controls), to distinguish common annual shocks from local winter-water associations. Not a model-tuning target; neither regression establishes causality. Primary original estimate remains reported alongside this sensitivity.

### First clean R3b CV result: ordinary fold 4 (MPS)

`r3b/fold_4.json`: 1,793.24 s including calibration/evaluation. Paired primary macro R2: clean recursive 0.090801, contaminated v2 0.106665, MLP 0.125169, stage trees 0.111579. Versions compared on exactly 126,375 all-strata rows with identical county normals; primary macro scores exclude overlapping practice strata. Clean misses H9 margin on this fold (gap 0.03437 to MLP). This comparison includes added EO supervision and a CUDA-AMP→MPS-FP32 numeric change, not just fill-code cleaning. No model tuning.
48-day RMSE (recursive/persistence/climatology): root .0418/.0674/.0641; surface .0524/.0673/.0620; profile .0278/.0647/.0649; NDVI .1284/.1165/.0886; LAI .3606/.1970/.1237; FPAR .1093/.1084/.0709; ET .4476/.4102/.1619 mm/day; surface-air proxy 3.5473/5.1173/3.7567 C. January-start forecasts: water skill is not proof of growing-season canopy skill.
`r3b/what_if_fold4.json` (256 held-year corn rows): rain -30% changes mean root wetness -0.03799, NDVI -0.01431 and arithmetic relative yield -0.05750 (about -5.44% of baseline), versus -0.01864 yield at irrigated share .8. Eight-day +6/+3 C heat: vegetative yield 1.05745 vs baseline 1.05691 (negligible increase), flowering 1.02322 (larger loss; phase-ordering check passes here). Dry Sep-Feb lowers root wetness .00789 and NDVI .00406 but **increases** yield .00172: no universal negative carry-over effect. Full held-year/drought checks still pending other folds. These are controlled consistency checks, not causal validation.
Carry-over observational sensitivity (`recursive/empirical_carry_over_year_effects.json`), same 26,154 rows/743 counties: -0.00922 log yield per +100 mm preceding Sep-Feb P (county-cluster 95% CI [-0.01200,-0.00645]), controlling county and year effects and May-Sep P/Tmax/ET0. Original county-only estimate -0.01412 [-0.01682,-0.01141]. Both oppose the blanket assumption that wetter winter necessarily improves yield; neither identifies a causal effect.

## R3b final: complete clean five-fold CV (logged 2026-11-14, no further training)

Outputs: `data/processed/recursive/r3b/{metrics.json,predictions.parquet,held_crop_predictions.parquet}`;
v2 pairing `data/processed/recursive/paired_versions.json`. Folds 0-3 Kaggle T4 (`account-A/fieldshift-recursive-r3b-fold{0,1}`,
`account-B/fieldshift-recursive-r3b-fold{2,3}`), fold 4 local MPS. Ordinary fits 39-42 min on T4 (fold 4: 29.9 min MPS),
held-crop fits 33-35 min (fold 4: 22.9 min); each Kaggle kernel ran its ordinary and held fit in parallel on 2 GPUs.
Inner-year SD multipliers 0.93-1.22. Baselines are the R2-D predictions on identical rows/normals.

Macro within-crop R² (primary rows), per outer fold and overall:

| fold | trend | season trees | stage trees | MLP | R3b free | R3b teacher | v2 free | v2 teacher |
|---|---|---|---|---|---|---|---|---|
| 0 | -0.031 | 0.065 | 0.096 | 0.091 | 0.056 | 0.050 | 0.076 | 0.069 |
| 1 | -0.045 | 0.137 | 0.161 | 0.215 | 0.106 | 0.121 | — (v1 crashed, not rerun) | — |
| 2 | -0.016 | 0.109 | 0.131 | 0.149 | 0.112 | 0.106 | 0.129 | 0.128 |
| 3 | -0.040 | 0.056 | 0.078 | 0.094 | 0.045 | 0.052 | 0.085 | 0.084 |
| 4 | -0.012 | 0.093 | 0.112 | 0.125 | 0.091 | 0.087 | 0.107 | 0.096 |
| all 5 | -0.000 | 0.129 | 0.153 | 0.175 | 0.114 | 0.118 | | |
| folds 0,2,3,4 paired | -0.007 | 0.100 | 0.125 | 0.136 | 0.095 | 0.094 | 0.117 | 0.113 |

Held-out crop macro R²: trend -0.036, season 0.067, stage 0.087, MLP -0.128, R3b 0.024 (teacher 0.026); paired with v2
on its folds: v2 0.033 vs R3b 0.010. Pooled equal-county R²/MAPE: MLP 0.215/15.98%, stage 0.205/16.12%, R3b 0.160/16.64%;
area-weighted R² MLP 0.310, R3b 0.218. Rainfed (share = 0) macro: MLP 0.237, stage 0.182, R3b 0.119.
Calibration R3b ±1/±2 SD: 70.9/94.0% pooled; per crop 65.7-81.9 / 91.6-97.7%; outside 68±5: potatoes 74.3, rice 81.9.

Drought states (IA IL IN KS MO NE SD) corn, observed vs predicted loss (R²):
1983 obs 25.4% — MLP 21.1 (0.312), stage 14.1 (0.078), R3b 14.1 (-0.024), R3b teacher 16.0, v2 17.3 (0.178);
1988 obs 19.5% — MLP 21.6 (0.459), stage 22.9 (0.359), R3b 14.8 (0.313), teacher 18.2;
2012 obs 35.5% — MLP 25.0 (0.118), stage 25.5 (0.008), R3b 28.7 (0.229), teacher 29.1 (0.299).

State RMSE (free rollout from 1 January; recursive/persistence/climatology) at lead 1 / 6 / 46 steps, and number of
leads from the start that beat climatology: root wetness .017/.022/.069, .040/.066/.068, .056/.096/.074 → 17;
surface .045/.030/.069, .052/.065/.065, .073/.100/.084 → 48; profile .014/.020/.068, .027/.063/.068, .043/.095/.072 → 92 (all);
NDVI .080/.080/.083, .119/.102/.084 → 1; LAI .158/.124/.131 → 0; FPAR .076/.078/.078 → 1; ET .366/.259/.137 mm/day → 0;
surface-air temperature proxy 3.55/4.71/3.92 °C → 9. Wetness beats persistence at all 92 leads (surface 88).
v2 (contaminated, folds 0/2/3/4): root 14, surface 92, profile 92, NDVI 1; v2 LAI/FPAR invalid.

What-if (256 held-year primary corn rows per fold, all five fold models): P -30% in season lowers root wetness 0.038-0.045,
NDVI 0.014-0.017 and relative yield 2.2-5.5% (rainfed) vs 0.4-1.9 points at irrigated share 0.8 (attenuated, all folds).
+6/+3 °C 8-day heat: flowering lowers yield 1.9-4.7 points below baseline in every fold, vegetative changes it by +0.05 to +0.3
(no loss). Dry Sep-Feb (P -30%): root wetness -0.008 to -0.011, yield -0.02 to +0.17 points (≈ no carry-over).
Data: -0.0141 (county effects) / -0.0092 (county+year effects) log yield per +100 mm Sep-Feb rain, i.e. wetter winters go with
slightly lower yields; the model's ~zero response has that sign in 4/5 folds but is much weaker.

Independent 2012 checks (fold 1 model, 2012 held out; `r3b/external_2012_fold1.json`, 99 Iowa counties, rainfed corn):
root-wetness anomaly vs USDM D2+ area r = -0.41 (expected sign). Restoring one month's weather to the fit-year normal raises
yield by May -0.003, Jun 0.064, Jul 0.256, Aug 0.045, Sep 0.000; RMA drought damage share of planted May .006, Jun .044,
Jul .282, Aug .063, Sep .017 (all-cause indemnity 65% in July). The model places the loss in July, as the insured losses do.

Verdicts. **H9 rejected**: free-running R3b is 0.061 below the MLP (margin 0.02), and 0.039 below the stage trees.
Teacher forcing costs nothing (0.118 vs 0.114), so the gap is in the yield readout, not error build-up.
**H10 accepted for soil water** (root 17, surface 48, profile 92 steps ≥ 6), **rejected for NDVI/LAI/FPAR/ET** (0-1 steps).
**H11 accepted** for the registered rain and heat checks (right sign, all folds; irrigation attenuates).
Carry-over is not negative, which agrees in sign with the observed association.
**H2 mixed**: 2012 corn loss 28.7% is closer to 35.5% than the MLP (25.0%), but 1983 (14.1 vs 25.4%) and
1988 (14.8 vs 19.5%) are further off than the MLP. Clean vs contaminated: R3b is 0.022 below v2 on the same rows
(within) and 0.023 (held). The two runs differ in more than the LAI/FPAR fix: R3b also has added ET/LST/SMAP targets,
the pre-2000 NDVI mask and, for fold 4, MPS instead of CUDA AMP. So the drop cannot be attributed to any one change.
The corrupted targets did not inflate v2's yield skill in any way that was visible.
Decision: the tool keeps the MLP for yield; the recursive model is used for soil-water state and what-if behaviour only.

## A1: farming-agent pilot (registered 2026-11-14 ~11:20, before any run)

Change of direction (user): no ground truth exists for the "right" rotation, so an LLM agent decides and our numeric
models serve it. Harness: OpenCode v2.3.0 (`opencode-linux-arm64.tar.gz`, sha256 `3207fb5c...e93711`) in image
`fieldshift-agent:oc2.3.0-arm64` on the Colima `rosetta` VM, agent `vm/agents/farm-advisor.md` (read/glob/grep inside the
workspace, shell only `./farmsim`, writes only `DECISION.json` and `REPORT.md`; no web). The farm state and climate
record are the first message (`fieldshift.agent.brief`), anonymous: no place names, no calendar years. Studies are a
folder of cards searched with grep (`data/evidence/cards`). The simulator is the v22 planner behind a host service.
Model for the matrix: `opencode-go/deepseek-v4.1-flash`; one stronger reference model on the base farms.

Setups: `llm` (farm state only), `evidence` (+ cards), `full` (+ cards + simulator).
Base farms: Story IA rainfed, York NE irrigated, Sumner KS rainfed, Lubbock TX irrigated, Bolivar MS irrigated,
Fresno CA irrigated; default priority text "steady income first, then soil and water".

Hypotheses and pass rules:
- **A1-1 farm-specific.** In `full`, the six base farms get at least 4 different recommended cycles, and no
  recommendation contains a crop the simulator lists as infeasible for that farm.
- **A1-2 sensitivity** (paired runs, same farm, one fact changed; judged on the simulator's numbers for the two
  recommended rotations under the *same* base farm, so only the choice differs): (a) Story with slope 12% instead of
  its own: recommended rotation has lower erosion or fewer days without living cover; (b) Story with priority
  "nitrate leaving the field matters most, some income can go": lower nitrate loss than under "income only";
  (c) Lubbock rainfed with rain -30% vs Lubbock irrigated: lower field water use. Pass: at least 2 of 3 in the
  expected direction and none reversed (a tie counts as neither).
- **A1-3 stability.** Story and Lubbock, 5 runs each in `full`: the modal recommended cycle (crops and winter
  options, rotation-invariant) appears in at least 4 of 5.
- **A1-4 tools are used.** Every `full` run makes at least one simulate call (service log), and at least 70% of the
  evidence items in `DECISION.json` cite a card, a simulation or the brief. Reported, not pass/fail: on how many base
  farms the recommended cycle differs between `llm`, `evidence` and `full`.
- **A1-5 known answers** (the experiment's cards are removed from the folder; farm = the experiment's county):
  (a) Kaspar rye, Boone County IA, priority nitrate in drainage water: recommendation includes a cereal rye winter
  cover; (b) Marsden Farm, Boone County IA, priority "cut purchased fertilizer and herbicide, keep profit": a cycle of
  3+ years with a small grain or forage legume; (c) Great Plains wheat-fallow intensification, Washington County CO,
  priority net return per unit of rain: a cycle with a summer crop replacing part of the fallow. Pass: 2 of 3.
- **A1-6 citations.** Every cited card id exists in the folder given to that run and was opened in that run (read
  events); no `sim:<n>` beyond the number of calls made. Pass: zero violations over all runs with the folder.

Decision rule: if A1-1 or A1-2 fails, fall back to "simulator ranks, LLM explains". If both pass, build the full
evidence library and compare models (Opus against the cheap ones) on the same cases.
Known weakness stated in advance: A1-2 and A1-5 are judged partly with our own simulator, whose cover-crop nitrate
effect (20% vs 61% published) and ET are weak; a pass shows the agent responds to the farm, not that it is right.

### A1 run log

- 2026-11-14 12:05. First run (`story`, `full`) is an **infrastructure failure, not scored**: `farmsim` was
  copied into the workspace without its execute bit, so every call returned "Permission denied" and the agent
  made no simulation. Kept as `_infra_fail_story__full__r0_farmsim_not_executable`. Fix: the runner sets mode 755.
  What the failed run still showed: it opened 19 cards, cited 15, all cited cards existed and had been opened,
  and it reported the missing simulator itself instead of inventing figures. 24 of its 28 shell calls were
  denied (pipes, `cd`, `chmod`, `ls`).
- Two changes made before any scored run: (1) the agent prompt now states the allowed shell form; (2)
  `known-fallow` holds out both Akron card groups (`nielsen-akron-fallow`, `akron-wheat-fallow-water`), since
  both come from the experiment the case is built on. Other Great Plains fallow studies stay in the folder.
- 2026-11-14 12:20. Second `story` / `full` run (438 s, 10 simulate calls, 20 cards opened, 0 cost on the Go
  plan): recommended the existing corn-soybean cycle, 4 alternatives (rye after soybean, rye after corn, rye after
  both, oats/hay variants), 28 evidence items all sourced, every cited card existed and was opened. **A1-6 as
  literally scored fails**: it cited `sim:11` and `sim:12` with 10 simulate calls. Its numbering counted all
  `./farmsim` invocations (options and help were calls 1 and 2), which the prompt ("your n-th simulator call")
  allowed. Treated as a prompt ambiguity, not a fabrication; the run stays in the record with this note and is
  re-run under the fix. Fix before the full matrix: `farmsim simulate` now prints `sim:<n>` with each result
  (counter file in the workspace, errors included, same count the scorer uses) and the prompt says to cite that
  label. 4 of 10 simulate calls were rejected as infeasible (hay stand length, winter wheat after soybean,
  vetch after corn); the agent corrected its requests from the error messages.

### A1 results (33 runs, DeepSeek V4.1 Flash via OpenCode Go, scored 2026-11-14 ~14:00)

Scored by `uv run python -m fieldshift.agent.a1_analysis` (writes `data/processed/agent/a1_report.json`).
All 33 runs completed (median 454 s, 62 denied tool calls in total, cost reported 0 on the Go plan). Cycles are
compared after removing repeats and rotation shifts; free-text winter names are mapped to the simulator's covers.

| Rule | Result | Verdict |
|---|---|---|
| A1-1 distinct, feasible | 5 distinct cycles for 6 farms; no crop outside the farm's options | **pass** |
| A1-2 sensitivity | slope: same cycle as Story's modal answer (no change); nitrate vs income: 42.2 vs 51.2 kg N/ha (expected); dry vs irrigated Lubbock: water use 320.6 vs 319.7 mm (reversed by 0.3%) | **fail** (1 of 3) |
| A1-3 stability | Story modal 3 of 5 (corn+rye/soybean 3, corn/soybean 2); Lubbock 2 of 5 (four different cycles) | **fail** |
| A1-4 tool use and sourcing | all 21 `full` runs simulated; all sourced >= 70% | **pass** |
| A1-5 known answers | rye: pass; Marsden: fail (corn+rye/soybean, no third crop); fallow: not testable | **fail** (1 of 2) |
| A1-6 citation integrity | 1 real violation (`lubbock` evidence run cited `deines2023-cover-yield` without opening it); 4 runs cited 43 cards by shortened IDs, all of which were opened | **fail** |

- Post-hoc choices, stated: A1-2 pairs each sensitivity run with the modal recommendation of its base farm
  (Story and Lubbock had 5 reps). Pairing with rep 0 instead gives slope = expected (Story r0 had no rye), so
  2 of 3 expected but still one reversal (0.9 mm). No tolerance was registered, so the 0.3% difference counts
  as reversed. Either way the rep-to-rep spread in A1-3 is as large as the sensitivity effects, so the
  sensitivity results cannot be told apart from noise.
- The fallow known-answer case was badly built: the brief's existing cycle for Washington County CO is
  winter wheat-corn (inferred from NASS areas), and the simulator has no fallow or summer crop such as millet
  or sunflower for that farm. The agent recommended alfalfa x3 / oats.
- Setups differ: without the simulator (`llm`, `evidence`) the model added a rye cover on 10 of 12 farms;
  with it (`full`) the recommendation differed from `llm` on 6 of 6 farms and kept the existing cycle on 2.
  The simulator shows rye costing ~$150/ha/yr in income.
- **Decision (pre-registered rule):** A1-2 failed, so the agent does not choose the rotation. Fallback:
  the simulator ranks rotations against the farmer's weights, and the LLM explains the ranking with cited
  cards. No model comparison (Opus vs cheap models) is run on the A1 design.

## A2: scoreboard procedure and model comparison (registered 2026-11-14 ~14:40, before any A2 run)

**Why.** In A1 the agent picked rotations freely and repeat runs disagreed as much as changed farms did.
The user asked for a scoreboard and for more models. The literature points the same way:
- LLM + optimiser designs: LISTEN-U, IJCAI 2026; ParetoChat, arXiv 2610.23062. The LLM turns natural-language
  priorities into explicit weights and limits, and a deterministic ranker scores the options.
- Repeated identical runs differ even at temperature 0 (arXiv 2408.04667, 2502.20747), and reliability has to be
  measured over repeats (pass^k, tau-bench).
- Anthropic's consistency guidance: numbered procedures, explicit criteria, fixed output schema, examples.
- Majority voting over runs is the standard variance reducer, and its agreement rate is a usable confidence
  signal (Wang et al. 2022, arXiv 2203.11171).

**Note added 2026-11-14 ~15:10, before any A2 run (user rule):** studies that test models older than the GPT-5 or
Claude 4 series do not count as evidence about the models we use (preferably Claude 4.5+ and GPT-5.5+). None of the
papers above meets that bar: the non-determinism papers test GPT-4o, Claude 3.5 and older; self-consistency tests
PaLM-era models; LISTEN cites Gemini 2.5 and Llama 3; ParetoChat's models were not checked. They motivated the
design but carry no weight for it. The design rests on our own A1 measurement. A2 itself is the evidence for or
against it on current models.

**Change.** New prompt `vm/agents/farm-advisor-a2.md` (A1's prompt is kept unchanged).
- Weight rubric: 5 main priority, 3 secondary, 1 not mentioned, 0 given up. Plus fixed farm-fact adjustments:
  erosion over 11 t/ha/yr, rain under 400 mm or falling groundwater, nitrate over 40 kg.
- Explicit limits only, with a default income floor of 70% of the existing margin.
- One `./farmsim scoreboard` call (new service endpoint) ranks every feasible cycle and flags weather-noise ties.
- Choice rule: existing rotation if it is in the tied set, else rank 1; override only with an opened card
  that shows a missed effect.
- A drier and hotter robustness scoreboard; studies used for checking and explaining.
- Full card IDs required.

**Models.** `opencode-go/deepseek-v4.1-flash`, `opencode-go/glm-5.3-flash`, `opencode-go/gpt-6-luna` (all three
reply-probed in the container at 14:30), and `anthropic/claude-opus-5-5` (subscription route in the VM, run
serially). All on OpenCode v2.3.0. Opus goes through an auth plugin that transforms requests, which is a harness
difference.

**Runs per model.** Setup `full` only. The six base farms, the four sensitivity cases, and two known-answer cases
(rye, Marsden), rep 0 each, plus Story and Lubbock reps 1-4: 20 runs. The fallow case is dropped as
untestable (A1). Farms, briefs and evidence are identical to A1. The service now also returns the seasonal
outlook in the brief.

**Pass rules per model** (A1 rule numbers kept):
- A2-1 = A1-1.
- A2-2 = A1-2, with each sensitivity case paired with its base farm's modal answer.
- A2-3: the modal cycle in at least 4 of 5 on **both** Story and Lubbock.
- A2-4 = A1-4.
- A2-5: rye and Marsden known answers, 1 of 2 needed.
- A2-6: no cited card unopened or missing; full IDs only. One shortened ID that matches an opened card is
  reported but does not fail the rule.
- A2-7 (procedure): the recommendation follows rule 5b/5c given the run's own weights and limits, re-scored
  by the analysis, or has a non-empty `override_reason`. Needed in at least 90% of runs.
- A2-8 (weights): on Story reps 0-4 the weight vector is identical in at least 4 of 5.

**Decision.**
- A model is usable if A2-1, A2-2, A2-3 and A2-6 pass.
- The product uses the cheapest usable model. If only Opus is usable, the demo uses Opus and says so.
- If no model is usable, the simulator ranks with weights set by the rubric from a fixed form, and the LLM
  only explains.
- Also reported: the voted answer and agreement rate for Story and Lubbock; how often rule 5d is used; and
  how often the chosen weights differ between models on the same farm.

**Stated weakness.** The procedure leaves the LLM little room in the choice itself. What A2 tests is whether
the LLM reads priorities and farm facts into weights reliably and correctly. Sensitivity and known answers are
still judged partly by our own simulator. Minimum-maximum score scaling means a farm-wide change, such as a
steeper slope, moves the ranking only through the rubric's adjustments or limits.

### A2 run log

- 15:20 smoke (not scored, `runs/a2_smoke`): DeepSeek V4.1 Flash, Story, 369 s, 2 scoreboard + 1 simulate call.
  Weights followed the rubric (income 5, resilience 5, soil/erosion/water 3, others 1). Limit was the 0.7 floor.
  Tied set: S-C-S-C and S-C-C-C. Recommended the existing rotation by rule 5b, robust under drier/hotter
  weather. No card faults.
- 15:32 matrix started with `vm/run_a2.sh`: 4 models x 20 runs into `~/.opencode-vm/fieldshift/runs/a2`, same farm
  IDs as A1. The service on 8601 was restarted with the scoreboard and the seasonal outlook.
- 16:05 reasoning traces: OpenCode's JSON event stream has no reasoning, but each run's own session database
  (`<run>/data/opencode/opencode.db`) does. New `fieldshift.agent.transcripts` exports reasoning, text and tool calls
  to `<run>/transcript.jsonl`; `pilot` now does this after every run (before the Opus credential cleanup deletes
  `data/`). Backfilled all finished A1 and A2 runs. Visible reasoning: DeepSeek and GLM give full text; GPT-6 Luna
  returns only encrypted reasoning and Opus 5.5 only signed, empty thinking blocks, so for those two only the visible
  text and tool-call sequence can be compared. The first 10 Opus rep-0 runs have no database left (deleted by the
  cleanup); their events.jsonl still holds text and tool calls. Asking the providers for reasoning summaries would
  change request settings, so it is not done inside this matrix.
- 16:50 matrix finished: 80 of 80 runs completed, no infrastructure failures.

### A2 results (80 runs, scored 2026-11-14 ~17:05; `data/processed/agent/a2_report.json`)

**Scorer fix, applied after the runs (post hoc, flagged).** The card-citation pattern in `pilot.score` allowed a dot
inside a card ID. It therefore read the full stop after a sentence-final citation (`card:deines2023-cover-yield.`) as
part of the ID and counted it as a missing card. No card ID contains a dot. The pattern now stops before the dot.
All A1 and A2 runs were re-scored from their saved files (`pilot --rescore <dir>`; the old score is kept in
`meta.json` as `score_before_rescore`). Only two runs changed, both from "missing card" to clean: Opus
known-marsden r0 and DeepSeek story-nitrate r0. Nothing outside the card fields changed in any of the 113 runs.
A1 verdicts are unchanged.

| Rule | Opus 5.5 | DeepSeek V4.1 Flash | GLM 5.3 Flash | GPT-6 Luna |
| --- | --- | --- | --- | --- |
| A2-1 different farms, different cycles | pass (4) | pass (4) | pass (4) | pass (4) |
| A2-2 sensitivity | pass | pass | pass | pass |
| A2-3 repeats agree (Story, Lubbock) | 5/5, 5/5 | 5/5, 5/5 | 5/5, 5/5 | 5/5, 5/5 |
| A2-4 simulator used, sourced | pass | pass | pass | pass |
| A2-5 known answers (rye, Marsden) | rye only | rye only | rye only | rye only |
| A2-6 citations | pass | pass | **fail**: cited `mech-rootworm-rotation` unopened in 2 runs | pass |
| A2-7 follows rule 5b/5c | 100% (1 card override, Fresno) | 100% | 100% | 100% |
| A2-8 Story weights identical | 5/5 | 4/5 | 5/5 | 5/5 |
| usable | yes | yes | no | yes |
| median run time | 116 s | 370 s | 206 s | 116 s |

**Usable under the registered rule: Opus 5.5, DeepSeek V4.1 Flash, GPT-6 Luna.** The cheapest usable model is
DeepSeek V4.1 Flash. GLM fails only on citing a card it never opened (the card exists and appears in `INDEX.md`).

**Why this is weaker than the table looks.**
- Agreement is mostly built into the procedure. All four models chose the same weights from the rubric (the
  identical vector income 5, resilience 5, water/soil/erosion 3, others 1 on Story). The scoreboard then ranks, and
  rule 5b/5c picks. Across models the weight vectors agree in 10 of 20 cases. Recommendations agree far more often,
  because different weights still fall into the same tied set.
- Lubbock (5/5 for every model) and Sumner rest on wrong economics. Irrigation water and pumping cost nothing, and
  alfalfa uses a generic 8 t/ha yield. Every model therefore recommended continuous alfalfa on irrigated Lubbock
  (1,640 mm/yr of field water use, while the existing cotton-sorghum showed -$515/ha/yr), and
  alfalfa-alfalfa-sorghum-soybean on Sumner. The models followed the numbers correctly; the numbers are wrong. A fix
  to the economics is in progress and is a precondition for A3.
- A2-2: the slope pair recommended the same rotation (no change). The drought pair passed on 315 vs 319 mm of field
  water use (about 1%). Only the nitrate pair shows a clear response (42 vs 51 kg N/ha/yr).
- A2-5: the Marsden case failed for every model. The farmer's aim, cutting herbicide, is not a simulator metric.

**Conclusion.** The scoreboard procedure removed the run-to-run spread seen in A1 (Story 3/5 and Lubbock 2/5 became
5/5). It did so by moving the choice from the LLM to the simulator. A3 tests what is left for the model: reading the
farmer's free text into limits, keeping to the procedure, and explaining with evidence. A3 runs only after the
economics fix.

## A3: farmer form and personalisation (registered 2026-11-14 ~16:45, before any A3 run)

**Why.** In A1 and A2 every base farm got one vague sentence ("Steady income first, then soil and water."), so
each model had to guess the weights; most of the remaining disagreement in A2 sits in that step (DeepSeek set
different weights for the same sentence). The challenge asks for decisions personalised to the farmer's own
priorities. The user proposed a short form plus a risk attitude.

**Precondition.** A3 runs only after the irrigated-economics fix found at the end of A2 (Lubbock's existing
cotton-sorghum lost $515/ha/yr because irrigation water and pumping cost nothing, so alfalfa using 1,640 mm/yr
won for every model). A2's Lubbock, York and Fresno choices are read with that fault in mind.

**Change.** The farmer fills a form: importance 0-5 for money, water, soil, nitrogen loss, erosion, ground cover,
forage and local market, a risk attitude (`accept bad years` / `balanced` / `protect me in bad years`), and free
text. The form maps to scoreboard weights without the LLM: each importance is the weight; risk attitude sets
resilience = 1 / 3 / 5. The LLM (A2 procedure otherwise) turns the free text into limits only, may override the
scoreboard only with an opened card (rule 5d), and explains. The brief shows the form.

**Profiles** (same four on each farm; free text tests the limit reading):
- P1 profit: money 5, everything else 1, accept bad years; "I want the most money, I can carry a bad year."
- P2 steady: money 4, soil 2, water 2, others 1, protect me in bad years; "Keep at least 80% of my current
  income." -> `min_margin_share_of_existing` 0.8.
- P3 soil builder: soil 5, erosion 5, cover 4, money 3, others 1, balanced; "No new machinery." ->
  `no_new_equipment`.
- P4 water saver: water 5, nitrogen 4, money 3, others 1, balanced; "Keep a forage crop for my cattle." ->
  `keep_forage_crop`.

**Farms.** Story IA (rainfed), Lubbock TX (irrigated), York NE (irrigated). **Runs:** 3 farms x 4 profiles x
3 reps x 4 models (deepseek-v4.1-flash, glm-5.3-flash, gpt-6-luna, claude-opus-5-5) = 144.

**Pass rules per model:**
- A3-1 repeats: all 3 reps give the same canonical cycle in >= 10 of 12 farm-profile cells.
- A3-2 limits: free-text limits extracted exactly as listed above in >= 90% of runs, with no invented limits.
- A3-3 personalisation: on each farm the four profiles give >= 3 distinct modal recommendations, and P3/P4's modal
  choice beats P1's on its main metric (soil carbon input / field water use) in the simulator.
- A3-4 rule following: >= 90% of runs pick by rule 5b/5c from the scoreboard of the form weights and their own
  limits, or give an override with an opened card.
- A3-5 citations: no cited card missing or unopened (one shortened ID tolerated, as A2-6).
- Across models (reported, not a per-model rule): share of farm-profile cells where all four models' modal
  choices are identical or in one another's tied sets.

**Decision rule.** The product uses the cheapest model passing A3-1 to A3-5. If none passes, the form + scoreboard
choose and the LLM only explains.

**Amendments before any A3 run (2026-11-14 ~17:50).**
- Precondition extended. After the irrigation fix (v23), every Lubbock cycle and Sumner's existing rotation still
  lost money, because "margin" subtracted ownership and overhead, and Kansas borrowed Iowa costs. A3 waits for v24:
  margin = return over operating costs (fixed lines in a second metric), K-State budgets for Kansas, and
  `soil_carbon_input_t_c_ha_yr` in simulate results (the A3-3 metric for P3).
- Prompt `vm/agents/farm-advisor-a3.md` (A2 prompt otherwise unchanged): the nine weights are copied from the form's
  `Weights:` line in the brief (no rubric, no farm-fact adjustment). Limits come only from the farmer's own words;
  **no default income floor** (A2 set 0.7 when none was named). Expected limits: P1 `{}`, P2
  `{min_margin_share_of_existing: 0.8}`, P3 `{no_new_equipment: true}`, P4 `{keep_forage_crop: true}`. A3-2 also
  reports runs whose weights differ from the form.
- Optional economics block on the form: P2 sets labour $22/h and nitrogen $1.40/kg N on every farm. The harness
  applies these to the farm record, not the LLM. No pass rule; reported: whether the explanation names them.
- Cases: `vm/a3_cases.json`. Runner: `vm/run_a3.sh <model> <parallel>`. Scorer: `fieldshift.agent.a3_analysis`.
  For P3 and P4 vs P1, both cycles are simulated on P1's farm record, so P2's economics do not enter.
- Opus runs start after a token renewal (the current token expires 18:41, and each episode needs timeout + 600 s), so
  they may finish later than the other models.

**Stated weakness.** With the weights fixed by the form, agreement is expected to be high by construction; A3
measures whether the LLM reads free text into the right limits, keeps to the scoreboard, and explains with
evidence, not whether it can weigh priorities itself. The profiles' "sensible" direction is judged by our own
simulator.

### A3 run log

- 2026-11-14 ~19:40: v24 economics committed (`7e0dba1`) and the service on 8601 restarted with planner cache
  `fieldshift-v24` (precondition met). The 12 A3 farms were registered once in
  `~/.opencode-vm/fieldshift/runs/a3/farms.json` before launch, so that the four runner processes do not race on
  the registry. Launched `vm/run_a3.sh` for deepseek-v4.1-flash, glm-5.3-flash, gpt-6-luna (parallel 2 each) and
  claude-opus-5-5 (serial; token valid to ~03:15). 36 runs per model, 144 in total. No prompt or rule changes
  since registration.
- 2026-11-14 ~21:45: infrastructure failure. Between ~20:50 and ~21:15 the three unfinished runner processes
  (DeepSeek 23/36, GLM 33/36, Opus 28/36; Luna had finished) died with `shutil.Error` in
  `anthropic_auth.prepare` and lost stderr. Cause: every Opus run copied the 233 MB vendored plugin into its
  `cfg/` and never removed it (~11 GB over A2 and A3), which filled the disk. One GLM container kept running
  orphaned and was stopped; Opus `york-p2 r1` timed out with zero events and was moved to
  `_infra_york-p2__opus__r1_timeout_no_events_disk_full`; runs without `meta.json` are redone. Fixes: the plugin
  is now hard-linked (the vendor tree is still hash-checked before every run and mounted read-only), the old
  copies were deleted, and, on the user's instruction, Opus runs are no longer forced serial. Resumed with
  DeepSeek parallel 4, GLM 3, Opus 4. Opus runs in parallel share one subscription; any 429/529 is
  infrastructure, not a model result. Prompt, cases and rules unchanged.

### A3 results (144 runs, scored 2026-11-14 ~22:30, simulator v24; `data/processed/agent/a3_report.json`)

| Rule | Opus 5.5 | DeepSeek V4.1 Flash | GLM 5.3 Flash | GPT-6 Luna |
| --- | --- | --- | --- | --- |
| A3-1 repeats agree (cells of 12) | pass 12 | pass 11 | pass 11 | pass 11 |
| A3-2 limits exact | pass 100% | pass 100% | pass 100% | pass 100% |
| A3-3 personalization | fail | fail | fail | fail |
| A3-4 procedure | pass | pass | pass | pass |
| A3-5 citations | pass | fail (1 run, 2 cards unopened) | fail (4+ runs) | pass |
| Median seconds per run | 130 | 332 | 239 | 126 |

Across models, the same cell got an identical cycle in 83% of cases. Every model copied the form weights exactly
and wrote exactly the expected limits.

A3-3 failed for all models in the same way, so it measures the test and the simulator, not the models:
- Lubbock P1, P2 and P4 all get continuous alfalfa (the v24 alfalfa fault; v25 fix running).
- The comparison against P1 ignored the profiles' limits: P3 ("no new machinery") cannot use a winter cover, so it
  ends on corn/soybean with less soil carbon input than P1's rye cycle; P4 ("keep a forage crop") must include hay,
  which uses more water than P1's choice. The rule was badly specified by me; it should compare within the limits.
- With weights fixed by the form and limits read correctly, the scoreboard decides; agreement is by construction.

Conclusion: models read farmers' forms and words reliably (A3-1, A3-2, A3-4); citations remain the differentiator
(Opus and Luna clean). A3 cannot show whether the advice is right, which motivates the planted-answer benchmark
(B0, being built): fix the correct answer first, build the task around one decisive fact, and test whether models
reach it and flip on the twin task without that fact.
