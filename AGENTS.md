# Field Shift data helpers

- Use `uv run` for Python, `uv run pytest` for offline tests, and
  `uv run ruff check .` for lint. Data belongs under gitignored `data/`.
- `fieldshift.net.install_dns_fallback()` installs a process-wide, system-first
  `socket.getaddrinfo` wrapper. Only failed lookups use DoH, bootstrapped through
  verified HTTPS IP endpoints at Cloudflare and Google. Do not replace USDA
  hostnames with IP URLs or disable TLS verification.
- Verified 2026-11-13: the keyless Quick Stats bulk index is
  `https://sb.nass.usda.gov/datasets/`. The `www.nass.usda.gov/datasets/`
  route redirected to an unusable origin on this network. Discover the dated
  crops filename from the index rather than hardcoding a date or USDA IP.
- `uv run python -m fieldshift.data.nass` downloads into `data/raw/nass/`,
  resumes validated `.part` downloads, and refuses files above 5 GB. It streams
  survey/annual/county yield, harvested area, planted area and production into
  `data/processed/nass_county_crops.csv.gz`, with a sibling `.summary.json`.
  No NASS key is required for this path. A live transfer was truncated after
  roughly 1.08 GB; rerunning successfully resumed the validated byte range.
  Keep the `.part` and `.download.json` files together when resuming.
- Raw bulk column names are uppercase; processed names are lowercase. Read
  state/county identifiers as strings to preserve leading zeros. Classes,
  production/utilization practices, domains and reference periods distinguish
  observations: do not treat all rows for a county/commodity/year as duplicates.
  Residual NASS county codes remain present; use county ANSI codes to validate
  joins to actual county boundaries. Non-numeric values become missing numeric
  values, with the original code retained in `value_raw`. Missing CV cells can
  contain a NUL sentinel in the raw bulk file; the extractor normalizes it to blank.
- `fieldshift.data.sda.query(sql)` uses SDA's `JSON+COLUMNNAME` response format.
  SDA returns `{}` for no matching rows (verified live); in that case the wrapper
  returns an empty DataFrame without columns. Both USDA helpers install the DNS
  fallback automatically.
- Never interrupt unrelated downloads or change system network settings to
  address DNS failures. No credentials belong in source or documentation.
- Agent-managed credentials live in gitignored `.agent-env` (copy `.env.example`); never commit it.
  No NASS API key is needed (the bulk download path is keyless).

## Data on disk (all under gitignored `data/`, verified 2026-11-13)

- `data/raw/cybench/`: CY-Bench v1.10 (Zenodo record 17279151), md5 verified. Only
  `cybench-data/{maize,wheat}/US/` is unpacked. Per county (`adm_id` =
  `US-<state fips>-<county fips>`): yields 1990-2023 in t/ha, daily weather
  2001-2023, soil moisture from 2003-02, NDVI and FPAR 2001-2023, soil, crop
  calendar. Grain corn and winter wheat only. 499 of 61,703 corn rows have yield 0.
- CY-Bench corn yield equals NASS bu/acre x 0.06277 (Story County IA, 2010-2014,
  agreement within 0.005 t/ha), so the two sources can be mixed.
- NASS units differ by commodity and utilization (corn grain BU / ACRE, silage
  TONS / ACRE); always filter on `unit_desc` and `util_practice_desc`.
- Irrigation matters for the drought signal: York County NE corn 2012 was 210.1
  bu/acre irrigated vs 54.1 non-irrigated.

## Rotation-effect data (checked 2026-11-13, not downloaded)

- Sustainable Corn CAP workbook, CC0, 12 MB:
  `https://ndownloader.figshare.com/files/44334239` (plot yields, crop sequences,
  soil carbon, total N, nitrate, insects; 30 sites, 2011-2015).
- KBS rotation soil C/N 1993-2014: `https://lter.kbs.msu.edu/datatables/102.csv`
  (publishing needs KBS permission).
- QDANN 30 m yield maps 2008-2022 exist only on Google Earth Engine
  (`projects/sat-io/open-datasets/lobell-lab/VAE_QDANN_YIELD_MAP/CORN_SOY_MAP`,
  b1 corn, b2 soybean, kg/ha; CC-BY-NC-SA 4.0, non-commercial).

## Earth Engine (verified 2026-11-13)

- Project: `<your-earth-engine-project>`. A registered Earth Engine Cloud project is needed
  (noncommercial registration is enough). Pass it with `--project` or `EE_PROJECT`.
  Authenticate once with
  `ee.Authenticate(auth_mode="localhost", scopes=[".../auth/earthengine"])`, which
  needs a browser click.
- QDANN: 120 images = 8 states (IA IL IN MN MO OH SD WI) x 2008-2022, one image per
  state-year with property `state_abbv`, EPSG:4326 ~30 m, int bands; 0 = no crop.
  Neighbouring state rasters overlap, so filter by state, never mosaic them.
  In two 500-point test batches, a non-zero yield band sat on the matching CDL class
  (1 corn, 5 soybean) in 94.5% (IA) and 89% (SD) of point-years; the rest fall on
  other CDL classes and are dropped. Corn and soybean are never both non-zero.
- `USDA/NASS/CDL` on Earth Engine covers 1997-2025 (band `cropland`, Albers 30 m).
  Image.projection() fails on the whole CDL image; select `cropland` first.
- `uv run python -m fieldshift.data.qdann --project <your-earth-engine-project>` samples random
  points (default 0.08 per km2 of land) per state in reproducible batches of 2,500
  (seed = crc32 of state and batch), keeps points with at least 5 QDANN yield years,
  caches each batch in `data/raw/qdann/`, and writes `data/processed/qdann_points.csv.gz`
  (wide) and `qdann_point_years.csv.gz` (one row per point-year with the two previous
  CDL classes and an `interior` flag for pixels away from field edges).
  A 500-point Iowa test batch took 29 s and kept 339 points.
  Full run 2026-11-13 (density 0.08, 4 workers): 42 batches in ~15 min, 34,866
  points (IA 7,630, IL 6,652, MN 5,234, SD 3,773, IN 3,725, MO 3,099, OH 2,681,
  WI 2,072), 412,656 point-years.
- `uv run python -m fieldshift.analysis.rotation` writes
  `data/processed/rotation_penalty.json`: continuous vs rotated yield, compared within
  county-year, 95% CI by county bootstrap. Result on the full sample: interior pixels
  corn-after-corn 0.96% lower (0.63-1.23), soy-after-soy 4.03% (3.56-4.47); all pixels
  1.95% and 5.38%. Soy penalty grows with years in soy (2.99% after 1 year, 6.47% after
  4+). Direction matches Seifert 2017 but magnitude is about half (likely because
  QDANN is a satellite estimate that compresses between-field differences; untested).
  Use Seifert for magnitude and QDANN for the pattern.
- Yield-stress model (`uv run python -m fieldshift.model.yield_stress`, details and
  numbers in README "Learned weather-stress yield response"): ~10 min on M5 Pro;
  neural parts require MPS and refuse CPU. Stage trees beat the MLP in-sample and on
  held-out crops; the saved simulator model is the MLP. Its seed spread covers only
  ~35%/~58% at ±1/±2 SD, so do not present it as a confidence interval. On the 4-6%
  of labels with known rainfed status, R² rises (stage trees: corn 0.37, soy 0.27-0.28,
  wheat 0.30-0.33), consistent with irrigated counties hiding water stress.
  CY-Bench `sos` is not a sowing date; CY-Bench weather differs slightly by crop mask.
- NASA POWER county weather (`uv run python -m fieldshift.data.power --project
  <your-earth-engine-project>`; `--cache-only` downloads only, `--validate` checks). Built 2026-11-14:
  `data/processed/weather/power_county_daily.f32`, float32 (3108 counties, 16436 days
  1981-01-01..2025-12-31, 14 vars: tmin tmax tavg prec rad rad_filled vpd et0 cwb rh
  wind gwetroot gwettop gwetprof) + `.json`; load with `load_county_daily()` /
  `county_frame()`. ~1.8 GB over the wire in ~5 min, cached in `data/raw/power/`.
  Source: anonymous zarr at `https://nasa-power.s3.amazonaws.com/merra2/temporal/
  power_merra2_daily_temporal_lst.zarr` (no radiation there); radiation from syn1deg
  2001-2025, srb 1984-2000, Hargreaves fill for 1981-1983 (`rad_filled`). POWER zarr
  radiation is W/m2 (x0.0864 -> MJ/m2/day). zarr v3 rejects fancy indexing.
  County means are weighted by cultivated share from CDL `cropland` (EE has no
  `cultivated` band; non-crop = 24 documented codes); 840 counties fall back to area
  weights. Story County IA matches the POWER point API exactly (one cell). Against
  AgERA5 2003-2023 (median county r / bias): tmax 0.988/+0.36 C, tmin 0.990/-0.76 C,
  prec 0.820/-0.02 mm, rad 0.961/-0.18 MJ, et0 0.961/-0.24 mm, vpd 0.889/-0.62 kPa.
  The VPD gap is a definition difference: CY-Bench VPD = es(Tmax) - ea, ours is the
  FAO-56 mean(es(tmax), es(tmin)) - ea. Elevation comes from time-mean surface
  pressure (no DEM). Story County JJA: 1988 tmax 33.3 C / 221 mm, 2012 32.9 C / 155 mm.
- Validation targets from Seifert et al. 2017 (field data private): rainfed yield
  penalty 4.3% for corn after corn and 10.3% for soybean after soybean.

## State on 2026-11-14 ~03:00 (work in progress, three agents)

- Experiment log: `EXPERIMENTS.md` (hypotheses registered before runs). Baselines on
  POWER 1981-2025, 20 crops, 630,528 rows (R2-D), macro within-crop R2: trend 0.00,
  season trees 0.129, stage trees 0.153, MLP ensemble 0.175; held-out crop: stage
  trees 0.087, MLP -0.128. The recursive field-state model (R3, 8-day steps) trains
  on four private Kaggle T4 kernels (`fieldshift-recursive-r3-fold{0,1,2,34}`, two per
  account); no R3 score yet.
- The temperature-rule sowing date is too early against NASS state 50%-planted dates
  (mean error: corn 17 days, soybean 35, winter wheat 15). Use the observed dates
  (`data/processed/nass_crop_progress_dates.parquet`) where they exist. The soybean
  maturity GDD trait is about 25% too long for Iowa maturity dates.
- Story County IA corn, NASS county record: 2010 163.9, 2011 163.2, 2012 157.6,
  2013 137.2, 2014 169.9 bu/acre. The county barely lost yield in 2012; it is a poor
  drought test. Use many counties, or rainfed Nebraska/Illinois counties.
- New data (built 2026-11-14 by `fieldshift.data.{progress,losses,eo,soil}`; each has
  a `*_qc.json` or sources JSON next to it; review of the coarse-grid sampling and of
  two out-of-range crop-progress values was still open when this was written):
  - `nass_crop_progress.parquet`, `nass_crop_progress_dates.parquet`,
    `nass_crop_condition.parquet`: weekly, state level, 48 states, 47 commodities,
    progress from 1979, condition from 1986.
  - `rma_cause_of_loss.parquet`, `rma_cause_share.parquet`: USDA RMA cause-of-loss
    by county, crop, year, month and cause, 1989-2026, from `pubfs-rma.fpac.usda.gov`.
  - `eo/eo_county_8day.f32` + `.json`: float32 (3108 counties, 2070 steps of 8 days
    from 1981, n_vars), county order as POWER. NDVI (MODIS MOD13A2 2000+, AVHRR CDR
    before, per-county linear adjustment with median R2 0.50, flag `ndvi_source`),
    LAI/FPAR (MOD15A2H), actual ET (MOD16A2GF); more variables were still building.
    County NDVI vs CY-Bench 2003-2023: median r 0.961, bias -0.009.
  - `static/county_static.parquet`: OpenLandMap soil (0-30 and 0-100 cm; AWC, Ksat,
    field capacity and wilting point derived with Saxton & Rawls 2006), SRTM
    elevation and slope, cropland centroid, GRACE trend.
- Data state verified 2026-11-14 ~04:00 (supersedes the list above where they differ):
  - EO cube is (3108, 2070, 9): `ndvi, ndvi_source, lai, fpar, et_actual, lst_day,
    lst_night, smap_rootzone, smap_surface`. MODIS from 2000-02-18, SMAP from
    2015-03-30. Ranges: LAI 0-6.77, FPAR 0-0.974, ET 0-6.47 mm/day, LST day -35.4 to
    55.8 C. SMAP root zone vs POWER `gwetroot`: median county r 0.952 (2016-2024).
    GLDAS was dropped. GRACE was added at ~04:15, making the cube (3108, 2070, 11):
    `grace_tws` (cm, monthly value stamped on each 8-day step of that month,
    2002-04..2024-09, from `NASA/GRACE/MASS_GRIDS_V04/MASCON`) and `grace_observed`.
    Mascons are ~3 degrees, so neighbouring counties share a value. Trend in
    `county_static.grace_tws_trend_cm_per_year`: Hale TX -1.45, Fresno CA -1.09,
    Finney KS -0.73, Story IA -0.14 cm/yr. `eo.publish()` now refuses to write
    any value outside `PHYSICAL_BOUNDS`.
  - PRISM (`fieldshift.data.prism`, asset `OREGONSTATE/PRISM/ANd`, ~4.6 km, daily
    from 1981; `AN81d` is deprecated) delivered 05:36: `weather/prism_county_8day.f32`
    + `.json` + `_qc.json`, (3108, 2070, 9) = `ppt_sum, tmean, tmin, tmax, tdmean,
    vpdmin, vpdmax, wet_days, ppt_max` (VPD in hPa), county order as POWER, no NaN;
    load with `load_prism()`. 2,070 requests, 94 min at 3 workers. An image stamped
    with a date starts at 12:00 UTC; steps with a masked day are masked because EE
    `sum`/`mean` skip masked days silently. Against POWER (median county r / bias):
    precipitation 0.93 / +0.22 mm per step, tmin 0.993 / -0.66 C, tmax 0.993 /
    +0.15 C, June-August rain 0.90. Story County JJA rain PRISM/POWER (8-day steps):
    1988 196/231, 1993 864/784, 2012 231/162 mm, so single-county summer totals differ
    by up to ~40%. Six-state JJA rain anomaly: 1983 -37/-34%, 1988 -39/-36%, 2012
    -41/-42%. Daily PRISM (~12.5 h of requests) was not built.
  - Earth Engine exposes no usage counter through the API; monthly quota use
    cannot be read from code.
  - **Cube copies made before 03:52 on 2026-11-14 are corrupt for LAI and FPAR**:
    MODIS fill codes were not masked before county aggregation (15,086 county-steps
    with LAI > 8, max 25.5; FPAR max 2.55). NDVI was unaffected. The R3 recursive
    model's first Kaggle runs (v1, v2) and the first `recursive_field_r3.pt` used such
    a copy (4 variables, no ET/LST/SMAP). Always range-check a cube against physical
    bounds before training on it.
  - The 0.05 degree bilinear sampling was checked against a strict native-mask box
    average for one July 2012 composite over 201 Corn Belt counties: county r 0.995,
    bias +0.0004 NDVI, +1.8% LAI, +0.3% ET (`eo/eo_grid_check.json`). Kept as is.
  - `eo/ndvi_adjustment.parquet`: per-county AVHRR-to-MODIS fit; median r2 0.50, 886
    counties below 0.3. Pre-2000 NDVI is a weak signal.
  - Story County July+August NDVI: 2011 0.827, 2012 0.789, 2013 0.742 (LAI 2.86, 2.39,
    2.09). 2013 is lower than 2012 because of its cold, late spring.
  - `usdm_county_weekly.parquet` (2000-2026, 4.3 M rows) and `usdm/usdm_8day.f32`.
    Categories are cumulative: `d0` = percent of county area in D0 or worse.
  - RMA: US 2012 indemnity sums to $17.45 bn (published 17.4). Iowa corn 2012 is 93%
    drought with 65% in July; 1993 is 94% excess water; 2019 is 93% excess water with
    45% in May. The commodity spelling changed case in 2019 (CORN, Corn): join on the
    upper-cased name (`commodity_key`, present only in `rma_cause_share.parquet`).
    Use `damage_share_of_planted` (NASS planted acres as denominator), not RMA's own
    planted acres, which cover only policies with a loss.
  - Crop progress: the two bad rows (SD oats 1996, raw 799) are now NaN.
- R3 kernels: v1 of all four failed after training with `Inplace update to inference
  tensor outside InferenceMode` in calibration (fixed; uncalibrated weights are now
  saved right after training). v2 was re-pushed 03:19-03:29.
- Earth Engine returns HTTP 429 when many `computePixels` calls run in parallel;
  the client retries by itself, lower the parallelism instead of adding retries.
- Tool: `uv run streamlit run src/fieldshift/app/main.py`; validation numbers with
  `uv run python -m fieldshift.planner.validate` (writes
  `data/processed/planner/validation.json`). Code in `src/fieldshift/{sim,planner,app}`.
  Known problems in the first version, being fixed: rotations were scored as 4-year
  windows with no previous crop, so cyclic shifts of one rotation ranked differently;
  no winter cover-crop slot; the what-if sliders switch from the learned yield model
  to the FAO-33 path; the page was unreadable in a dark-mode browser (check pages in
  a real browser, Streamlit `AppTest` does not catch styling faults).
- The raw NASS file also holds annual PRICE RECEIVED rows (8,741 recent ones were
  extracted to `data/processed/planner/nass_reference_prices.json`).

## Submission rules (Project Submission Guide, read 2026-11-14)

- Submission opens Sat 14 Nov 09:00 and closes **Sun 15 Nov 2026 23:59 local time**,
  on the team's Project tab at spaceappschallenge.org. Teams of 2-6.
- Fields: project name, high-level summary, link to demo, link to final project,
  detailed description, use of SI (the 2026 term for AI), NASA data used, other
  data and references.
- Demo: at most 7 slides (title slide included) or a video of at most 30 seconds,
  on a public link. Final-project link must open without login; the page accepts
  images (JPG/PNG/GIF up to 10 MB) but no video or document uploads.
- AI use must be disclosed: generated code and data acknowledged "in descriptive
  text and metadata"; generated images and video need a visible watermark; no
  generated NASA logos or mission identifiers. AI may not be the source of the idea.
- Global awards require at least one NASA data source from the challenge's Resources
  list (ours: NASA POWER, MODIS, SMAP, GRACE).
- Licences: code MIT (`LICENSE`), text, cards' summaries, tasks and results CC BY 4.0 (`LICENSE-docs`).

## Tool state 2026-11-14 ~04:40 (commit `12d4e07`)

- Planner cache version v20; restart Streamlit after editing imported modules.
- Nitrogen runs monthly in `sim/nitrogen.py` (living-cover storage, separate soil and
  residue pools, fertilizer rate reduced by legume credit). NASS cotton is class UPLAND.
- Multi-county check (`planner/validate.py`, 300 counties, no fitted parameters):
  FAO-33 yield vs observed, odd-year median r 0.43 corn, 0.47 soybean. Mean corn loss
  observed/simulated: 1983 23.8/-0.5%, 1988 27.2/36.2%, 2012 30.0/42.3%; soybean
  8.8/3.4, 20.7/13.3, 16.6/19.5. Bucket ET vs MODIS MOD16: season mean 442 vs 386 mm,
  summer anomaly r 0.10 (fails). Existing Story County rotation ranks 1 of 393 on
  income alone.
- Open at that time: cereal rye "fails" in 50-61% of Story County years under a
  100-heat-unit autumn rule, which contradicts Kaspar 2007 (Boone County, rye present
  every year); simulated rye cuts nitrate leaching 14% vs 61% published; displayed
  alternatives included a money-losing rotation; satellite evidence was in a
  collapsed section. When reporting cover biomass, separate the all-year mean from
  the mean of successful stands.

## State 2026-11-14 ~05:45

- Tool (planner cache v21): cereal rye grows continuously from autumn plus spring
  thermal time (SARE: germinates at 34 F, grows above 38 F) instead of dying below a
  heat threshold; Story County rye biomass 1.2 t/ha (Iowa trials 0.72-2.23), nitrate
  leaching 22% lower with rye (Kaspar 2007: 61%). Exclude calendar-padding days from
  winter minimum temperatures. Displayed alternatives must be Pareto-optimal and keep
  70% of the existing margin (adjustable). Unmet nitrogen demand is a diagnostic only;
  it does not affect yield and is not shown. Corn N uptake follows a two-thirds-by-
  VT/R1 schedule; cycle-average leaching 49 kg N/ha vs 32-58 in Iowa rate trials; the
  N budget is not validated. Davis 2012 (Marsden Farm) yield gains for longer
  rotations are included as low-confidence transfers. `planner/observations.py`
  builds the satellite panel (MODIS NDVI, SMAP, Drought Monitor, GRACE).
  Bucket ET vs MOD16 anomaly r is 0.10 overall, 0.30 in dry and 0.44 in wet years.
- Recursive model: R3 v2 trained on the corrupt LAI/FPAR copy; R3b is the same
  configuration on the fixed 11-channel cube (deployment file
  `models/recursive_field_r3.pt`, old one kept as
  `recursive_field_r3_contaminated_v2.pt`). R3b folds 0-3 ran on Kaggle
  (`fieldshift-recursive-r3b-fold{0,1}` account A, `-fold{2,3}` account B), fold
  4 on MPS. Fold 4 macro R2: recursive 0.091, MLP 0.125, stage trees 0.112, so the
  tool keeps the MLP for yield. 48-day soil-wetness rollouts beat persistence and
  climatology; NDVI, LAI and ET rollouts do not beat climatology.
- Kaggle: a kernel pushed while its private dataset is still processing reports
  success but runs without the data. Wait for `kaggle datasets status <ref>` = ready.

## Recursive model, final (R3b, complete five-fold CV, 2026-11-14 ~06:15)

- Numbers are in `data/processed/recursive/r3b/metrics.json` and `EXPERIMENTS.md`
  ("R3b final"). Macro within-crop R2, same rows as R2-D: trend 0.000, season trees
  0.129, stage trees 0.153, MLP 0.175, recursive free-running 0.114 (teacher-forced
  0.118). The recursive model is below the MLP in every fold and in 16 of 20 crops;
  rainfed-only 0.119 vs 0.237. **The tool uses the MLP for yield.** Teacher forcing
  barely helps, so the gap is in the yield readout, not in rollout error build-up.
- Held-out crop: recursive 0.024, stage trees 0.087, MLP -0.128.
- Calibration of the recursive model: +-1/+-2 SD cover 70.9% / 94.0%.
- Corn loss in IA IL IN KS MO NE SD, observed / recursive / MLP / stage trees:
  1983 25.4 / 14.1 / 21.1 / 14.1%; 1988 19.5 / 14.8 / 21.6 / 22.9%;
  2012 35.5 / 28.7 / 25.0 / 25.5%.
- Rollouts from 1 January beat county climatology for 17 steps (root wetness), 48
  (surface) and all 92 (profile); NDVI and FPAR for 1 step, LAI and ET for none.
- What-if checks (256 held-year corn rows, all folds): rain -30% lowers rainfed yield
  2.2-5.5% and 0.4-1.9 points at 80% irrigated share; an 8-day heat wave costs
  1.9-4.7 points at flowering and about nothing in the vegetative stage; a dry winter
  changes yield by about zero. In the observed data wetter winters go with slightly
  lower yield (-0.014 log yield per 100 mm; -0.009 with year effects).
- v2 (trained on the corrupt LAI/FPAR copy) scored 0.117 against 0.095 for R3b on the
  four folds both have. The runs differ in several ways (extra ET/LST/SMAP targets,
  pre-2000 NDVI mask, one fold on MPS), so the drop is not attributable to one change.
- In the POWER archive `rad_filled` is a fill indicator, not radiation: the weighted
  share (0 to 1, mostly exactly 0 or 1; mean 0.067) of the county's cells whose
  radiation was Hargreaves-filled that day. `rad` is the radiation itself, already
  gap-filled (0.1-34.9 MJ/m2/day). Never feed `rad_filled` to a model as radiation.

## Tool state 2026-11-14 ~07:45 (round 5, planner cache v22; last modelling round)

- Results of the pre-registered tests are in
  `data/processed/planner/validation_weather_tests.json`; the full check in
  `validation.json` (`uv run python -m fieldshift.planner.validate`, ~85 s).
  Adoption rule, fixed before running: odd-year median r must rise for corn and
  soybean, and the mean absolute loss error over 1983/1988/2012 x 2 crops must not rise.
- Adopted: simulator rain is NASA POWER daily rain rescaled inside each 8-day step to
  the PRISM step total (`prec_power` keeps the original; where POWER is dry and PRISM
  wet, 4,162 steps, the total is spread evenly, an assumption). Odd-year r 0.43 -> 0.49
  corn, 0.47 -> 0.51 soybean. The learned yield MLP still gets unscaled POWER rain,
  which it was trained on.
- Adopted: season-level FAO-33 factor `1 - ky_season (1 - sum ETa / sum ETm)` (ky corn
  1.25, soybean 0.85) instead of the product of stage factors, with one heat slope per
  crop fitted on even years (corn 0.001, soybean 0.0015 per degree day). Odd-year r
  0.51 corn, 0.53 soybean; mean drought error 10.2 -> 6.6 points. Corn loss
  observed/simulated: 1983 23.8/4.0, 1988 27.2/23.7, 2012 30.0/25.4; soybean 8.8/4.9,
  20.7/16.0, 16.6/19.6. 1983 is still missed.
- Rejected: Schlenker-Roberts 2009 degree days (soybean r fell to 0.46; fitted corn
  slope 0 vs ~0.0065 published), alone and combined with the season factor.
- Still failing: bucket ET vs MOD16 summer anomaly r 0.12; rye cuts nitrate leaching
  20% vs 61% in Kaspar 2007.
- Near-tie rule: a candidate is shown as "best match" only if its score lead over the
  existing rotation exceeds 1.96 x SD of the paired score difference over 2,000
  resamples of the 10 weather windows. Story County: rye variant leads by 0.4 points,
  tolerance 0.9, so the existing rotation keeps the label.
- Story County table (median per ha/yr): existing S-C-S-C score 73.1, margin $940
  (691-1,029), nitrate 51 kg N, 234 days without living cover; + cereal rye 73.5, $782,
  42 kg, 36 days; C x4 + rye 70.9, $673, 93 kg; S-C-S-oats 65.7, $786, 55 kg. Existing
  rotation ranks 1 of 434 on income alone.

## Farming agent (OpenCode v2 in the Colima VM), state 2026-11-14 ~14:00

- Historical launch settings below are superseded by "Linux/server portability" at the end.
- Image `fieldshift-agent:oc2.3.0-arm64` (`vm/Dockerfile`), launcher `vm/oc.sh`, agent prompt
  `vm/agents/farm-advisor.md`, simulator client `vm/farmsim` (must be executable; prints `sim:<n>` per call).
  Docker context `colima-rosetta` only; the `default` and `cr-android` profiles and the containers
  `bench-2wiki-amd64-opencode-1`, `buildx_buildkit_rosettabuilder0` belong to other work.
- Simulator service: `uv run python -m fieldshift.agent.service --host 0.0.0.0 --port 8601 --token $FARMSIM_TOKEN`
  (keys in gitignored `.agent-env`: `OPENCODE_GO_API_KEY`, `FARMSIM_TOKEN`). Restart it after editing
  `fieldshift.agent.*`; it serves one simulation at a time (~1 s per call, max seen 11 s).
- Pilot: `uv run python -m fieldshift.agent.pilot [--only ...] [--setups ...] [--reps N] [--parallel N]`,
  cases in `vm/pilot_cases.json`, runs in `~/.opencode-vm/fieldshift/runs/a1/`; scoring
  `uv run python -m fieldshift.agent.a1_analysis`. Containers use ~280 MB and little CPU each; parallel 3 was
  safe, 6 should be. OpenCode Go reports cost 0 (plan quota, not visible from the API).
- A1 result (EXPERIMENTS.md "A1 results"): DeepSeek V4.1 Flash recommendations vary between identical runs
  (Story 3 of 5, Lubbock 2 of 5 agree) as much as between changed inputs, so by the pre-registered rule the
  agent does not choose; the simulator ranks and the LLM explains. Models cite shortened card IDs; normalise or
  require full IDs. Known-answer cases need crops the simulator offers (no fallow/millet exists).

### Subscription-login runs (not part of this repository)

- The Claude Opus 5.5 runs (`anthropic/claude-opus-5-5`: the A2/A3 matrices, the Opus arm of B0, the live
  harness and chat checks) and the live GPT-6 Luna runs of 2026-11-15 (`openai/gpt-6-luna`: advise, app and
  demo-video runs, made after the OpenCode Go weekly limit was exhausted) were run through the author's own
  subscription login; that adapter is not part of this repository. Here `anthropic/*` and `openai/*` work
  only with an API key (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`); without one they are refused
  (`providers.SubscriptionModelError`). See "Model routing" below.
- Every subscription run recorded the provider-reported served model (`meta.json` / `run.json` `served_model`)
  and its credential copy was removed after the episode.
- Fix found during those runs (applies to all models): Luna gave up when the >51 KB scoreboard output was cut
  to its tail. `vm/farmsim` saves `.sim-response-<n>.json` indented (the read tool cuts long lines, and compact
  JSON is one line) and says so after each call; the product prompt points to the file. Rerun
  (`data/processed/advice/live/story-steady-luna-codex-r3`): 62.7 s, all checks pass, rank-1 corn-corn-corn-soybean.

## Rule for reading LLM research (user, 2026-11-14)

- A study about LLM behaviour counts only if it tests at least the GPT-5 or Claude 4 series (preferably Claude 4.5+
  and GPT-5.5+). Older-model results (GPT-4/4o, Claude 3.x, Gemini 2.x, Llama 3, PaLM) may motivate a design but are
  not evidence for it; prefer our own measured runs. Check and state the tested models before citing a paper.

## Economics v24 (planner cache `fieldshift-v24`, 2026-11-14 ~19:30)

- `margin_usd_ha_yr` (income weight, resilience = its P10, `min_margin_share_of_existing`) is **return over
  operating costs**. Fixed ownership/overhead, irrigation system ownership and farmer-quoted land rent enter only
  `return_over_listed_costs_usd_ha_yr`, shown beside it. Unsplit budget fees count as operating.
- Kansas crops use 2026 K-State budgets (`sim/kansas.py`, exact PDF URLs and rows); perennial hay uses K-State
  MF2143 (2015, nominal, low confidence). Budgets taken from another state are flagged `transfer` and LOW.
- `/simulate` and `/scoreboard` share the metric set, including `soil_carbon_input_t_c_ha_yr`.
- Under A2 default weights continuous alfalfa leads Lubbock, Sumner and Fresno. Its yields are weak proxies
  (Lubbock 12.98 t/ha from two 1983-84 county reports; Sumner 6.13 county 2016-18; Fresno 15.62 state) and the
  costs are established-stand budgets; establishment-year cost and yield loss are not modelled. Treat alfalfa
  rankings as low confidence. Receipts: `data/processed/agent/operating_checks.json`.

## Engine v26 (merged 2026-11-15 ~01:30, commit `6ba1268`, planner cache `fieldshift-v26`)

- New: `sim/state.py` (ensemble replay 1981-2025 with SMAP L4 root-zone ensemble update), `sim/forecast.py`
  (conditional forecasts; SEAS5 outlook spliced into historical years), `sim/hindcast.py` (skill test),
  `sim/diagnose.py` (1983 diagnosis only). Endpoints `GET /farm/<id>/state`, `POST /farm/<id>/forecast`;
  `./farmsim state|forecast`; brief section "Current field state". Steady-state simulate/scoreboard equal v25
  (Story, Lubbock exact; York within float32 roundoff 3e-5).
- Measured: Story's 380 SMAP updates cut mean soil-water spread only 14.6 -> 14.3 mm (2%); innovation mean 3.8 mm,
  SD 23.9 mm vs assumed observation SD 60 mm. Forecast 5 rotations x 1 year: 9.1 s cold + 3.8 s farm load, 0.02 s
  cached. Hindcast (200 counties, 2016-2024, SMAP start): assimilated start barely beats a field-capacity start
  (corn CRPS .0904 vs .0906); intervals under-cover (corn +-1/+-2 SD 54/81%); observed-yield climatology beats the
  simulator on CRPS. Soil-water RMSE at leads 1/4/12 steps: model .049/.051/.052, persistence .012/.030/.060,
  climatology .033/.033/.034 m3/m3, so the raw model loses to climatology. Present the state and forecasts as
  scenario tools, not as skilful predictions.
- 1983 corn loss is still simulated near 4% vs 23.8% observed; flowering timing is right (DOY 209 vs 211) and the
  40 C heat term is nearly inactive. No correction is enabled.
- A git worktree whose `data/` is a symlink shows every tracked `data/evidence/cards/*.md` as deleted; never
  stage `data/` from such a worktree.

## Farmer harness (`fieldshift.agent.advise`, 2026-11-15 ~06:30)

- `uv run python -m fieldshift.agent.advise <form.json> [--model M] [--out FRESH_DIR]`; forms in
  `examples/farms/`, schema in README "Advising a farm". Starts its own simulator service on a free port with a
  random token, ranks with the form's weights and limits, runs one agent episode (default
  `opencode-go/gpt-6-luna`), then checks the decision in code.
- Presets have weight 0 for aspects they do not name. With A3's "1 for unnamed", eight minor aspects outweighed
  money 5 and the York "profit" preset chose a rotation earning $161/ha/yr less than the current one.
- OpenCode 2.3.0 keeps only the tail of tool outputs longer than ~51 KB (`options` and full scoreboards), so the
  `Simulator call sim:<n>` header can be lost. `vm/farmsim` now prints the label before and after the response
  and saves the full response as `.sim-response-<n>.json`; `checks.recorded_calls` reads those first.
- Decision checks gate the recommendation: form weights, grounded limits, limits and feasibility, top/tied or
  opened-card override, cards exist and opened, cited `sim:<n>` calls exist. The per-number attribution check is
  advisory (WARN): it accepts numbers from the cited call's numeric and text fields and from the form.
- Live check 2026-11-15 (Opus through the author's subscription login): Story steady, 137.5 s, 9 simulator calls, all checks pass; the agent chose
  corn-soybean, tied with the reference corn-corn-corn-soybean. The OpenCode Go weekly limit was exhausted, so
  no live Luna run has been done.

## Farmer chat (`fieldshift.agent.chat`, 2026-11-15 ~08:30)

- `uv run python -m fieldshift.agent.chat [--model M] [--county "Story, IA"] [--resume DIR] [--script FILE]`;
  default `opencode-go/gpt-6-luna`; sessions in `data/processed/chat/<ts>-<county>/`. One OpenCode session per
  chat (`opencode run --session`); credentials are prepared before and scrubbed after every turn (files, credential
  table rows, token search); if any trace survives, the session is dropped and the next turn replays the
  conversation.
- Nothing is ranked until host code sees the farmer confirm the form table ("yes" without corrections, numbers or
  crop names). Each confirmed form is a new immutable farm version; in a chat `/economics` is refused and
  simulate/scoreboard/forecast accept only the latest confirmed version.
- Live check (Opus through the author's subscription login): `examples/chats/story.txt`, five turns, 153 s, recommendation checks 7/7, served model confirmed,
  every turn scrubbed clean with the session kept. Checks run only when DECISION.json changes.

## Benchmark B0 state (2026-11-15 ~10:15)

- The "stopped ~01:00" note is wrong for the Go models. Killing `vm/run_b0.sh` and the `bench.run` processes
  left the per-model retry loops in the script alive. They kept restarting runners after each provider-limit
  sleep: 59 more Go episodes completed after 01:05 (DeepSeek 10, GLM 20, Luna 29), and retries continued
  until at least 04:45. Nothing was running at 10:15. To stop a looping runner script, kill its whole process
  group, not just the parent.
- Keys in `data/processed/bench/keys.json` were made with an older simulator fingerprint. Later commits changed
  fingerprinted code (scoreboard echo, engine v26 files), so `bench.run` will refuse those keys until
  `uv run python -m fieldshift.bench.generate --recompute` has been rerun.
- Hidden split is not supported yet: the generator always overwrites the public `bench/tasks.json`, pair IDs
  are unsalted, the service folder is shared, there is no commitment step, and the fingerprint uses file
  times. Hidden tasks must stay private as well as keys (a task alone lets anyone compute its key).
- v0.5 release drafts: `bench/RESULTS_v0.5.md`, `bench/LICENSES_AUDIT.md`, `bench/PUBLIC_READINESS.md`,
  `CITATION.cff`, `.zenodo.json`.

## Questionnaire app (`src/fieldshift/app/advisor.py`, 2026-11-15 ~12:10)

- No offline answers (user rule 2026-11-15): agent failure or any failed decision check -> no recommendation shown; the code ranking is an audit file only.
- `uv run streamlit run src/fieldshift/app/advisor.py` (old `app/main.py` is the superseded rotation page).
  One page: farm, current rotation, what matters (0-5 + risk + presets), hard limits, price table, weather
  view, free text -> one agent solution (prompt `vm/agents/farm-advisor-v1.md`) -> questions in the same
  OpenCode session (prompt `vm/agents/farm-qa.md`; the agent may run what-ifs but never changes settings).
  Sessions in `data/processed/app/<ts>-<county>/` (form.json, advise outputs, qa.jsonl, worker.log).
- The worker runs as a subprocess (`advisor_core.spawn`). Anything that loads PROJ (geopandas/pyproj) in the
  page process makes forked children segfault on macOS; `tests/test_app_advisor.py` guards against it.
- Live check (Luna, Story steady, no new machinery): solution 65 s, all 7 checks pass, kept soybean-corn;
  answers 14-28 s. Faults found and fixed: Luna sent `./farmsim simulate --help` as JSON (farmsim now prints
  help for `--help`); "why not alfalfa" ignored the no-new-machinery limit (QA prompt now checks limits
  first); mm were converted to inches wrongly (238 "inches"; both prompts now carry worked conversions);
  farmer-facing text gave raw per-hectare floats (product prompt now asks for plain US units).
- Rerun after the prompt fix (`data/processed/advice/live/story-steady-luna-r4`, 58 s, decision checks pass):
  no raw floats, but the advisory number check WARNs on "about $450 per acre" because it does not convert
  $/ha to $/acre or accept "about" rounding. Expect such WARNs until the checker handles US units.

## CPU vs GPU timing (2026-11-15, M5 Pro, Story County IA)

- Ranking all 434 rotation cycles from cold: MPS 7.3 s, CPU 9.0 s (5 threads); Lubbock (242 cycles) 6.7 vs 6.8 s.
- Full farm pipeline with fresh caches (farm load, scoreboard incl. plan, 32-member state replay with SMAP
  update, 5-rotation 1-year climatology forecast): MPS 2.8 + 3.1 + 4.5 + 5.0 s = 15.4 s; CPU (5 threads)
  2.0 + 4.8 + 1.3 + 1.9 s = 10.0 s, repeated twice. The small state/forecast ensembles run faster on CPU (GPU
  launch overhead). A fully local desktop build without a GPU is fast enough; the code still refuses CPU unless
  the accelerator functions are patched (`sim/components.accelerator`, `model/neural.accelerator_device`).

## Raw download cache removed (2026-11-15 ~14:20)

- `data/raw/prism/` (43 GB) and `data/raw/eo/` (28 GB) were deleted to free disk space, with the user's approval.
  Nothing reads them at run time: the tool uses only `data/processed/weather/prism_county_8day.*` and
  `data/processed/eo/eo_county_8day.*`. Rebuilding those cubes means re-downloading through Earth Engine
  (`fieldshift.data.prism`, ~94 min; `fieldshift.data.eo`, several hours).

## Study search and farmer memory (2026-11-15 ~16:30)

- `./farmsim studies '<json>'` -> `POST /farm/<id>/studies` (`agent/studies.py`): searches only the 56 cards, BM25
  + 21 agronomy synonym groups + farm-fit label close/partial/far (state, climate words or rain, soil texture where
  stated). Counts as `sim:<n>`. Eval `data/processed/agent/study_search_eval.json` (16 queries, not held out):
  hits@3 14/16 BM25, 15/16 with synonyms; fit raises close cards in top-3 from 28 to 33 of 48.
- Farmer profiles in `data/processed/app/profiles/<slug>/` (PROFILE.md notes capped at 1,500 chars,
  last_form.json pre-fills the questionnaire, past sessions copied to the workspace as `history/*.md`). The agent
  only writes `MEMORY_PROPOSALS.json`; code drops proposals whose quote is not in the farmer's own text, the page
  asks Approve/Reject. Approved notes join the free text, so limits from notes pass the same quote check.
- `checks.py` accepts "does not allow / forbids <crop>" and a list of quotes behind one crop-list limit.
- Live (Luna, Story, profile demo-iowa "My landlord does not allow alfalfa."): solution 53.5 s, 7/7 checks,
  soybean-corn; rye question 26 s, still shows almost no simulated nitrate drop with rye (known gap).

## Dollar-figure check (2026-11-15 ~17:30)

- The video's live run (corn $4.60/bu) said "about $470 per acre"; the cited run gave $422. $470 was the worked
  example in the prompts (the default-price Story value), copied by Luna. Prompts now use illustrative numbers
  (1,000 $/ha = about $405/acre) and say to round dollars to the whole dollar, never to tens.
- New gating check `dollar_figures_match_cited_runs`: every "$" figure on a line citing `sim:<n>` must match that
  call (or the form) within half a unit of its last digit; the other numbers stay advisory. Re-scoring all saved
  live runs: only the $470 run and one "$450 vs $453" rounding-to-tens run would have failed. Rerun with the
  fix (`data/processed/advice/live/story-460-luna-dollarcheck`): 65 s, 8/8 checks, $422 / $329 per acre.
- Q&A answers are not checked number by number yet (superseded below).

### US units printed by the tool (2026-11-15 ~19:00)

- `agent/us_units.annotate` adds a rounded US sibling after every metric value in `/simulate`, `/scoreboard` and
  `/forecast` responses (applied in `service.py` only; `scoreboard.py`/`runtime.py` and the bench fingerprint are
  untouched): `*_usd_ha*` -> `*_usd_acre*` (whole $), `*_mm*` -> `*_in*` (1 decimal), `*_kg_n_ha*` ->
  `*_lb_n_acre*`, `*_t[_dm|_c]_ha*` -> `*_us_tons[_dm|_c]_acre*` (2 decimals); never for `_per_` rates, prices,
  %, days or ranks. Prompts tell the agent to quote these and never convert.
- New gating check `us_unit_figures_match_cited_runs`: a number followed by inch(es)/lb on a sentence citing
  `sim:<n>` must match that call's inch or lb fields (or the converted mm/kg fields) to its displayed precision.
- Numbers are now checked per sentence: a sentence citing only a card/brief/history is not checked against
  simulator calls (this caused the false "61%" flag); an uncited sentence on a `sim:` line still is.
- App Q&A answers run the dollar and US-unit checks; a failure shows "This answer was withheld: ..." and the
  original goes to `qa/withheld-<n>.md`. The terminal chat (`agent.chat`) runs the same check on every agent
  answer (`checks.withhold_failing_answer`); the original goes to `withheld-turn-<n>.md`.
- Live (Luna, story-460, `data/processed/advice/live/story-460-usunits-r{1,2}`): 63 s each, 9/9 checks, kept
  soybean-corn, quoted $422 / $329 per acre and 18.8 inches exactly. The host-added comparison table and outlook
  lines in REPORT.md are in $/acre, inches and lb N/acre too.
- Bench keys were already stale before this change (fingerprint `ec1cb708...` differs from `keys.json`;
  needs `bench.generate --recompute`), unchanged by it.

## Model routing (`fieldshift.agent.providers`, 2026-11-15 ~21:00)

- One place decides how `<provider>/<model id>` is served: `providers.resolve(model, environ)` returns a `Route`
  (OpenCode model reference, key variable names, provider block). Used by `pilot.run_one`, `advise.run`,
  `chat.OpenCodeAgent.turn`, `bench.run` and the app (`advisor_core.model_environment`). Key variables come from
  the vendored models.dev table `agent/provider_catalog.json` (324 providers, fetched 2026-11-15; refresh with
  `--update-catalog`; models.dev answers 403 to urllib's default User-Agent). `OPENCODE_GO_API_KEY` is an alias
  for opencode-go's `OPENCODE_API_KEY` (shared with the `opencode` Zen provider).
- Keys: environment first, then gitignored `fieldshift.env`, `.env`, `.agent-env` (`providers.key_environment`).
  The value goes only into the launcher's process environment; `vm/oc.sh` passes `-e NAME` and refuses any
  `FIELDSHIFT_PASS_ENV` entry that is not a variable name. `opencode.json` holds names, `model` and
  `small_model` only. Tests (`tests/conftest.py`) clear all known key variables and key files.
- Verified in the image (OpenCode 2.3.0): with `OPENROUTER_API_KEY` set, `opencode models` lists 360
  `openrouter/...` IDs (nested slashes); `small_model` appears in `debug config` as `agents.title.model`;
  a `providers.custom` block with `package: @opencode/ai/providers/openai-compatible` is active without a key.
  `host.docker.internal:11434` reaches the host's Ollama from `colima-rosetta`. Colima cannot bind-mount
  `/private/var/folders/...` (the opencode temp dir); use `~/.opencode-vm/...` for scratch run dirs.
- Live probes 2026-11-15 (`~/.opencode-vm/fieldshift/runs/provider-probes/`, one try each): OpenRouter with a
  fake key -> `provider.auth` 401 "User not found." from OpenRouter in 1.7 s (routing works); Go -> 429
  `provider.go-limit` weekly limit (key accepted; no advise run); `custom/qwen3.5:9b` via local Ollama -> "Ready",
  exit 0, 113.8 s. No key value in any probe file. No OpenRouter, Anthropic or OpenAI API key exists on this
  machine, so no full episode has been run with them.
- Custom model IDs are slugged for OpenCode's model key (`custom/qwen3.5:9b` -> `custom/qwen3.5-9b`,
  `modelID` keeps the original). Azure/Bedrock/Cloudflare get every listed variable that is set (untested);
   Copilot, Vertex and vLLM discovery are refused with an explanation.

## Linux/server portability (2026-11-15)

- Device-selection instructions in this historical section are superseded by "Automatic CPU selection" below.
- Public checkout only: `field-shift`; do not transfer private history or credentials from `nasa-farming`.
  Inference on a CPU server requires `FIELDSHIFT_DEVICE=cpu`, never a silent accelerator fallback.
  `model.probabilistic.synchronize` synchronizes CUDA/MPS only, not CPU.
- `agent.containers` builds Docker argv without a host shell. Image default
  `fieldshift-agent:oc2.3.0`; override `FIELDSHIFT_IMAGE`. Docker uses its current endpoint by default;
  `FIELDSHIFT_DOCKER_CONTEXT` selects an explicit per-command context. On this development Mac use
  `colima-rosetta` explicitly, leave other profiles and unrelated containers alone.
- `vm/Dockerfile` pins the official OpenCode 2.3.0 npm archives by SHA-512 for arm64 and amd64-baseline.
  Exact version output is `opencode v2.3.0`. Both images built and ran the CLI as the non-root image user;
  `/cfg` must be owned by that user too. Build includes a post-USER version check.
  Local image configs: arm64 `7dd694ff16e79fb87ce4bcad5b1008c4ad6a31ecee0b4fc182a2ecd7aca747cd`,
  amd64 `56083317056b2df09eefcfd04c633bb87f07b6a4e4dabfa3179763be004473ce`.
  Mounted launcher version/auth-list and host-gateway HTTP passed without keys/model calls.
- `locking.py` uses kernel file locks (fcntl on Unix, msvcrt first-byte locks on Windows).
  Registry, app job and benchmark-run locks use it. App workers use a Python supervisor
  `app.worker_process` rather than `/bin/sh`; Windows detach flags and Unix sessions are selected explicitly.
- Current source local CPU suite: 345 passed, 69 unpublished-data skips; focused synthetic engine tests
  42 passed, 4 private acceptance-fixture skips. Ruff clean. Simulator test skips honor explicit CPU selection.
  Native Windows Docker execution remains unverified; OS-matrix CI is `.github/workflows/platforms.yml`.
- Real Linux arm64 CPU physical smoke used the earlier source snapshot (engine files matched current source):
  torch 2.15.0+cpu, 4 threads, farm load 8.46 s, 434-cycle scoreboard 15.55 s, 32-member replay 1.96 s,
  five-rotation one-year climatology forecast 3.78 s, all outputs finite. Receipt on this machine:
  `~/.opencode-vm/fieldshift/smokes/linux-engine-cpu/receipt.json`. Data mounted read-only; caches separate.
  These are execution tests, not evidence of forecast skill.
- The earlier Linux snapshot's full suite was NOT clean: 290 passed, 99 skipped, 4 failed
  (three obsolete shell-launch mocks, one git-ignore test without `.git`). Current-source snapshot packaging
  was blocked by Safety Net `secret.basename.env`, even with explicit exclusions; no bypass/retry.
  Do not claim latest-source Linux success from that image; use the normal checked-in platform CI results.
- First platform CI (`43764540221`) built/passed the amd64 image but found a second unconditional MPS
  synchronization in `model.neural.train_ensemble` (CPU Linux/Windows error, hosted Mac SIGSEGV).
  It now synchronizes only the selected MPS/CUDA device; an explicit-CPU regression forbids both calls.
  Windows also exposed double-escaped missing-data paths in the test-only skip hook, two tests replacing
  mapped files without releasing their readers, and a POSIX-only mode assertion. Normalize path separators,
  release fixture maps before rewriting, and retain the mode assertion only on POSIX; Windows directories
  inherit ACLs and must live in private user storage. These fixes do not relax decision/scoring checks.
- Final latest-source platform CI `43764938454`, commit `f7752de`, all four jobs passed:
  Linux 349 passed/69 skipped in 31.60 s; macOS 349/69 in 28.24 s; Windows 349/69 in 45.84 s;
  amd64 Docker build + non-root pinned CLI check passed. The 69 skips require unpublished data/private fixtures.
  Native Windows Docker and a paid-provider episode are still not validated. Deprecation notices for the
  Actions Node runtime and runner-image migration are warnings, not test failures.

## Automatic CPU selection (user request, 2026-11-15)

- `device.runtime_device()` is shared by `sim.components.accelerator` and
  `model.neural.accelerator_device`: default CUDA -> Apple MPS -> CPU. `FIELDSHIFT_DEVICE=cpu`
  still forces CPU. No environment variable is required on CPU-only Windows/Linux.
  This chooses a device before computation; GPU runtime errors are never caught and retried on CPU.
  The separate `PYTORCH_ENABLE_MPS_FALLBACK=1` prohibition in the neural helper remains.
- Removed obsolete GPU-availability test skips so plain `uv run pytest` exercises synthetic model/simulator
  tests on CPU too. Tests requiring unpublished data/private fixtures still skip without those assets.
  CI no longer supplies a device override: each native OS runner tests automatic selection.
- The friend's 14 failures all came from the former strict GPU selectors, not Windows-specific math errors.
  Regression coverage now tests shared device precedence and real training with both GPU probes false and
  no device override. A full CPU-only simulated-availability run passed 355 tests with 69 data/fixture skips.
