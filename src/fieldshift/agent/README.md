# Farming-agent host tools

## Finished farmer command

`uv run python -m fieldshift.agent.advise examples/farms/story-steady.json`
validates the form, requires a usable advisor model, creates an owned ephemeral
service, runs one advisor episode, optionally forecasts, and independently checks
the result. If no checked recommendation can be produced, it gives no answer.
See [Advising a farm](../../../README.md#advising-a-farm) for the public
schema, presets, US units, output files, model setup, costs and limitations.

Registry `existing_rotation` is optional: 1–6 `{crop,winter?,double_crop?}` seasons.
It replaces the inferred reference, preserves per-season covers, and must pass
local feasibility checks. The brief distinguishes farmer-stated history from
county satellite/crop-area proxies. Its historical extrapolation is an assumption.
For such records scoring uses the common LCM(4, cycle length) horizon/phases;
absent this field, the existing simulation/scoring path is unchanged.
The product prompt is `vm/agents/farm-advisor-v1.md`; experiment prompts and
pilot/scorer references were not renamed. Postchecks currently use a disclosed
numeric presence/rounding audit, not the separately developed benchmark scorer.

Run from the repository root. The Linux agent container calls this **host**
service; the simulator needs host MPS (or CUDA). It never runs numeric simulation
on CPU. No new dependencies. Cache v25 adds explicit alfalfa stand accounting.
V24 separates operating and listed-cost
returns and adds Kansas budgets; v23 capped/priced irrigation and farmer
economics remain. The v22 learned/FAO response parameters are retained.
Do not restart an active evaluation service: code changes take effect only when
its owner deliberately starts a fresh process.

## V26 state and conditional forecast

New bearer-authenticated endpoints:

- `GET /farm/<id>/state` (`./farmsim state`): historical state replay and
  provenance; the brief includes a **Current field state** section.
- `POST /farm/<id>/forecast` (`./farmsim forecast '<json>'`):
  `{"rotations":[[{"crop":"corn"}]],"years":1,"weather":"outlook"}`.
  Years must be 1–4; weather is `outlook` or `climatology`. Results include
  per-season distributions, risk proxies and differences from climatology.
  An unavailable provider returns HTTP503, not stale forecast data.

The replay is a county surrogate, not a measured field. Weather currently ends
2025-12-31. Subsequent unobserved weather is **climate-bridged**, not reconstructed
observations. The 32-particle ensemble uses explicitly assumed uncertainty ranges;
SMAP L4 (itself a model analysis) updates only water, using an assumed
0.06m3/m3 observation SD. NDVI is a calendar check, not assimilated. Other stores
remain model-only unless farmer-supplied. Root/profile water are projections of
one mixed bucket, not independent observed layers.

Forecasts pair all 51SEAS5 members ×45historical donors with the 32state particles
deterministically, rather than taking a full state×weather Cartesian product.
Outside the available outlook horizon each member uses every donor with equal
weight. Monthly mean-bias transfer follows `outlook.py`; it is not distributional
calibration. Future yields use FAO-33, not fabricated MLP covariates. Probabilities
and ensemble ranges are uncalibrated. Frost means Tmin<0 between sowing and
flowering (emergence is not modelled); cover failure is a thermal/survival screen.
SEAS5/Open-Meteo/ECMWF attribution and CC-BY4/noncommercial-access caveats remain.

Cache version is `fieldshift-v26`; state JSON/NPZ pairs, forecast JSON, outlook,
scoreboard and default service logs live under `data/processed/engine_v26/`.
These caches rebuild automatically: none of the acceptance receipts, scripts or
`protected_assets.json` is required to serve requests. Retain the existing
`data/processed/agent/farms.json` registry and original model/weather/soil/EO/QDANN
inputs. Do not overwrite or touch fingerprinted shared inputs during a benchmark.
Only the service owner may replace an active process; the worktree acceptance
checks use port8621, never benchmark services.

`sim/hindcast.py` records C without tuning. Actual-weather runs are an upper-bound
diagnostic, not operational forecast skill; historical donors include their
observed planting calendars. Report model-only2003–2015 separately from
SMAP-initialised2016–2024. Yield intervals under-cover, and the raw soil-water
forecast loses to SMAP climatology; an observation-anchored increment diagnostic
has short-lead skill but is not the deployed water state. `sim/diagnose.py` only
diagnoses1983; it changes no service parameter and fits no correction. The old
MOD16 ET anomaly failure and unvalidated N budget remain. Steady-state
`/simulate`/`/scoreboard` have regression tests against the private v25 receipts;
York uses a stated float32 tolerance because independent MPS runs differ slightly.

## V25 alfalfa audit

`sim/alfalfa.py` defines a **scenario**, not an agronomically validated cutting model.
Uniform alfalfa requests use one four-calendar-year stand lifecycle, irrespective
of whether the caller writes A×1, A×2 or A×4: one no-hay fall-seeding year and three
production years. Termination by the last year's fall-seeding date and reseeding
on that date next year preserve a full one-year autotoxicity gap. Regional dates
are September15 (TX/CA/MS), August15 elsewhere; the latter transfers UMN's
southern-Minnesota window and is LOW confidence. The last production season is
shortened, with retained potential-ET share as a LOW harvest-window proxy.
Nonuniform stands use fall establishment in TX/CA (no harvest) and spring elsewhere
(ISU2.5/6 of production; UNL irrigated spring3.8/6.7 in Nebraska).
There is no winterkill, density decline, cutting-by-cutting scheduling or explicit
validated termination-operation model. Termination is assumed effective; its
operation timing/cost remains an unvalidated part of the transferred management budgets.

Establishment **replaces**, not adds to, production costs. Identified annual
amortisation is removed (TX100USD/ac, UC315USD/ac, ISUone-third seeding; Kansas annual
seed/tillage lines). Kansas unsplit ownership remains conservatively retained:
listed-cost returns may retain some drill ownership. `/options` exposes spring/fall
parts, starter N, labour, prices, source/confidence and stand policy. Price-only
edits still reprice cached physical stand events without simulation.

Recent NASS2022 Census entire/none-irrigated alfalfa yields take precedence over old
county proxies. Mississippi's 2022 practice yields are suppressed (one entirely
irrigated operation); its ALL-practice total164short tons/115harvested acres is used
instead of the three-operation2017 irrigated cohort. All state/county transfers are LOW.
Hay yield and NASS price are as-reported baled-hay mass; grade and actual moisture
are unspecified. No premium-grade price is invented. The88%DM carbon/reference-mass
convention is an assumption; UC's90%DM custom-harvest reference is converted90/88.
`local_market_fraction` is now the cycle mean NASS county harvested-area share
among supported crops, a production-depth proxy, **not** verified hay buyers,
quality, transport costs or liquidity.

FAO-33 alfalfa ky1.1 and capped/uncapped FAO multiplication of the learned baseline
were already present. Established-alfalfa→corn's30lb/ac N target against181lb/ac
continuous corn already implies151lb/ac credit; it is not added again as N or cash.
One-year stands use a LOW45lb/ac midpoint of UMN's30–60; other crops no longer
erroneously receive a zero-N target. Phosphate's incidental N is in the physical
ledger but its cost is charged only once; farmer N-price changes are marginal
where a source bundles N with phosphate. No N-yield feedback or validated N budget.

### Audit sources

- [UC Davis2016 SJV establishment/production, pp3,8,10–14](https://coststudyfiles.ucdavis.edu/uploads/cs_public/1c/e2/1ce256d0-957e-4bd4-b17e-18fef4efcedd/16alfalfasjv300acfinal_41916.pdf):
  fall seeding, three productive years, separate establishment and capital recovery;
  nominal2016 dollars, not inflation-adjusted.
- [TAMU2026 D3 establishment](https://agecoext.tamu.edu/wp-content/uploads/2026/02/2026D3AlfalfaIrrigatedEstablish.pdf),
  transferred to D2 LOW; production remains [D2](https://agecoext.tamu.edu/wp-content/uploads/2026/02/2026D2AlfalfaHay.pdf).
- [TAMU West Texas2023](https://legacy.research.agrilife.org/wp-content/uploads/sites/3/2023/04/Alfalfa-West-Texas-2023.pdf):
  spring harvest reduction, South Plains fall window, water demand, rotation/autotoxicity.
- [UNL2026 budgets1,6,10](https://cap.unl.edu/sites/unl.edu.ianr.agecon.center-for-ag-profitability/files/media/file/2026-nebraska-crop-budgets.pdf)
  (canonical URL is stored in `sim/alfalfa.py`); labour hours use the explicitly LOW2025 wage transfer.
- [ISU2025 pp9–10](https://www.extension.iastate.edu/agdm/crops/pdf/a1-20-2025.pdf);
  [K-State2026](https://agmanager.info/farm-mgmt-guides/crop-budgets) exact archived PDF rows in `sim/kansas.py`.
- [UMN autotoxicity](https://extension.umn.edu/agriculture/crop-production/forages/reducing-autotoxicity-in-alfalfa),
  [seeding windows](https://extension.umn.edu/agriculture/crop-production/forages/alfalfa-establishment-management-strategies),
  [corn N credits](https://extension.umn.edu/agriculture/crop-production/corn/tools-to-predict-nitrogen-needs-in-corn-following-alfalfa),
  [one-year stand credit](https://blog-crop-news.extension.umn.edu/2022/04/alfalfa-and-red-clover-can-supply-n-for.html).
- [NRCS irrigation facts](https://efotg.sc.egov.usda.gov/references/public/UT/AlfalfaIrrigationFacts2013Final.pdf):
  approximately linear yield/ET and12% moisture example; not NASS moisture metadata.
- [NASS Quick Stats](https://quickstats.nass.usda.gov/): raw Census records retained in
  `data/processed/agent/alfalfa_sources/nass_state_recent.json`; state cache
  `planner/nass_state_yields_v3.parquet`. Annual prices are ungraded, not grade-specific.

```sh
uv run python -m fieldshift.agent.farms add --adm-id US-19-169 \
  --priority "Protect soil without losing too much income" \
  --weight soil=5 --constraint "no new equipment"
# Save the opaque ID printed above as FARM_ID.
uv run python -m fieldshift.agent.brief "$FARM_ID" > FARM.md
uv run python -m fieldshift.agent.service --host 0.0.0.0 --port 8601 --token "$TOKEN"
```

Use a random high-entropy token. Host binding defaults to `0.0.0.0` for
`http://host.docker.internal:8601` in Docker Desktop. On Linux Docker, configure
`--add-host=host.docker.internal:host-gateway` if needed. The token authenticates
the entire local service, not one farm; protect it and use a trusted network
(plain HTTP, not Internet-facing). `Ctrl-C` stops the service.

## Registry

`data/processed/agent/farms.json` is a permission-0600, atomically replaced map of
opaque tokens to `{adm_id, irrigated, overrides, economics, priorities, constraints, anonymous}`.
Writes use a process file lock. `priorities` is `{text, weights}`. An optional
`--anonymous` omits identity and calendar years from the brief. Registry records
still contain identity; do not give the registry or host data files to the agent.

Repeat `--override KEY=VALUE`, `--priority TEXT`, `--weight KEY=VALUE`, or
`--constraint TEXT`. Supported overrides:

| Key | Meaning |
| --- | --- |
| rain_change_pct | Percent change to PRISM-corrected POWER rain |
| temp_change_c | Additive daily temperature change, °C |
| slope_pct | Slope in percent rise/run |
| organic_matter_pct | Upper-0.3-m organic matter, percent soil mass |
| awc_mm_per_m | Available water capacity, mm per metre of soil |
| irrigation_capacity_mm_yr | Annual applied-irrigation cap, mm/year (also accepted in economics, not both) |

OM also updates EPIC erodibility using the fixed observed texture. Soil overrides
are not clamped away. Climate overrides use the simulator's existing weather-edit
routine and its baseline-to-edited FAO multiplier; brief climate and simulator
weather are identical. The learned model retains the original POWER features it
was trained on. Observed satellite/yield/loss records are never rewritten as if
the hypothetical changes actually happened.

## HTTP

Every endpoint requires `Authorization: Bearer TOKEN`.

* `GET /farm/FARM_ID/options`: local crop screen, winter covers,
  maturity/survival, exclusions, and economic values/units/sources/confidence/ranges.
* `GET /farm/FARM_ID/brief`: `text/markdown`.
* `POST /farm/FARM_ID/simulate`: `Content-Type: application/json`, at most 64 KiB.
* `POST /farm/FARM_ID/scoreboard`: deterministic exhaustive planner ranking (below).
* `POST /farm/FARM_ID/economics`: merge economic overrides into a **new** immutable
  registry entry; returns `farm_id` and `parent_farm_id`.
* `POST /farm/FARM_ID/form` (chat sessions only, `./farmsim form`): proposes a patch to
  the public farmer form. Nothing is applied; the chat host applies it after the farmer
  confirms. While a chat runs, `/economics` is refused and `/simulate`, `/scoreboard`
  and `/forecast` accept only the latest confirmed farm version (`chat_forms.ChatForms.gate`).
  Outside a chat (`runtime.chat_form` unset) all other routes behave as before.
* `POST /farm/FARM_ID/studies` (`./farmsim studies`, counted as `sim:<n>`): body
  `{"query": str, "crops": [..]?, "topics": [..]?, "top": 1-15}`. Searches only the fixed
  card library (`studies.py`: BM25 + agronomy synonym table, x farm fit) and returns
  ranked cards with snippets and `farm_fit` = `close` / `partial` / `far` plus reasons
  (state or neighbouring state, aridity P/ET0 and thermal class against the card's
  climate words or stated rain, soil texture where stated). Read-only and not gated in chats.

## Farmer economics and irrigation (v23)

`farms.add({...,"economics":{...},"priorities":{"text":"...","weights":{...}}})`
accepts the economics block alongside the existing priority form. The endpoint
body is that block directly, not an extra `economics` wrapper:

```json
{
  "crop_price_usd_t": {"cotton": 1600, "cottonseed": 300},
  "crop_cost_usd_ha": {"cotton": {"seed": 180}},
  "labour_usd_hr": 25,
  "fertilizer_n_usd_kg": 1.5,
  "usd_per_acre_foot": 100,
  "irrigation_energy_usd_per_mm_ha": 0.9,
  "irrigation_capacity_mm_yr": 457.2,
  "cover_seed_usd_ha": 75,
  "land_rent_usd_ha": 0
}
```

These example quotes are illustrative **farmer inputs**, not suggested prices.
`crop_price_usd_t` uses harvested metric tonnes; cotton means **lint**, and the
optional `cottonseed` key prices its byproduct separately. Cottonseed quantity
uses the Texas budget's turnout ratio, not a field measurement. Unharvested covers
have no sale price; use their seed/cost parts. Allowed crop cost parts are `seed`,
`fertilizer_other`, `chemicals`, `fuel_repairs`, `labour`, `other_operating`,
`fixed_overhead`, `irrigation_fixed`, `repeat_crop`, `repeat_crop_fixed`, `harvest`. N is priced separately.
Component quotes are absolute USD/ha values, not percentage changes.
`repeat_crop` and `repeat_crop_fixed` apply only when the main crop repeats;
nonzero conditional repeat fees for unharvested covers/double crops are rejected.
Budget custom harvest
charges scale with actual harvested mass; an explicit farmer `harvest` USD/ha
quote instead supplies a fixed per-area bill. Its reference quantity is exposed.
Source budgets without a defensible split retain costs in `other_operating`:
to edit another component there, also supply revised `other_operating` to avoid
double counting. Bundled contractor services are not invented operator hours.

| Optional scalar | Inclusive allowed range | Unit |
| --- | --- | --- |
| labour_usd_hr | 0–500 | USD/hour |
| fertilizer_n_usd_kg | 0–100 | USD/kg N |
| water_charge_usd_per_mm_ha | 0–1000 | USD/(mm ha) |
| usd_per_acre_foot | 0–100000 | USD/acre-foot; alternative water-charge unit |
| irrigation_energy_usd_per_mm_ha | 0–1000 | USD/(mm ha) |
| irrigation_capacity_mm_yr | 0–5000 | mm/year; zero means no irrigation |
| cover_seed_usd_ha | 0–10000 | USD/ha sown |
| land_rent_usd_ha | 0–100000 | USD/ha/year; default zero/excluded |

Crop prices allow 0–1,000,000 USD/t; cost parts 0–100,000 USD/ha.
These are validation limits, not plausibility endorsements. Unknown keys,
booleans, NaN/infinity and out-of-range values are rejected. One acre-foot/acre
is 304.8 mm; USD/acre-foot converts to USD/(mm ha) by ×2.471053814671653/304.8.
Do not submit both water-charge unit aliases. All overrides are optional.

`options.economics` and crop records expose current values and provenance.
Farmer quotes are labelled `farmer`, high confidence **as supplied inputs**, not
independently verified contracts or a yield confidence interval. Model income
confidence can rise to medium with major quoted drivers; physical/model uncertainty
remains. Default N prices may be crop-specific budget blends; the global N field
is a regional reference. A farmer global N quote replaces all crop rates, so forms
should not silently submit a prefilled default as an explicit farmer quote.
Labour uses cited hours × wage, with hours inferred from bills explicitly marked
low confidence where necessary. Explicit per-crop labour bills take precedence.
Known budgets remove embedded reference N and price the simulated N quantity.
Unsplit generic bundles stay operating and retain their documented reference
N-credit savings; farmer N-price changes there are marginal actual-N adjustments.
This uncertainty is explicit; missing component splits are not fabricated.

**Accounting v24:** `margin_usd_ha_yr` is **return over operating costs**, including
operating labour, insurance and working-capital interest. `income` scores its
median, `resilience` its P10, and margin-share limits use it. Ownership/overhead
and rent never affect that ranking. `return_over_listed_costs_usd_ha_yr`
subtracts separately listed fixed costs plus farmer rent. Both endpoints expose
both quantile metrics and definitions; options also show existing returns.
Source-unsplit services remain operating. ISU labour follows its explicit
crop-choice variable-cost advice (p1), despite its placement in fixed columns.
ISU repeat-corn N is not billed twice: actual N plus the non-N operating and
ownership differences from pp2–3. Irrigation repairs are operating; ownership fixed.
`soil_carbon_input_t_c_ha_yr` is the same retained/humified C input used by the
scoreboard soil objective (not observed stock change). Simulate also exposes
forage and local-market fractions, so metric sets match.

Price/cost-only forks share field preparations, physical cycle ledgers and the
scoreboard's disk simulation cache. Repricing uses stored yearly harvest,
irrigation, fertilizer, cover and repeat-crop quantities, then recalculates window
margin quantiles, resilience, hard limits and ranking. Weight/limit edits also
reuse physics. A cap change is **physical** and requires a new simulation.

Yield normals prefer county, 100-km neighbours, then official STATE series (or
a same-state **county mean** if absent) for the same practice before all-practices.
The state series is separately extracted from the raw bulk: the processed county
file has no state rows. For example CA alfalfa is present at STATE, not county level.
The last available reports
can be old or sparse: dates/practice/fallback confidence are shown explicitly.
Fallback county means are not published statewide aggregates; official STATE rows
are labelled as such. Cotton's
LB/ACRE lint conversion is 0.00112085 t/ha per lb/acre, with UPLAND lint prices.

Default sources: [Texas South Plains 2026](https://agecoext.tamu.edu/wp-content/uploads/2026/02/2026D2CottonPivot.pdf)
and its sorghum, wheat and alfalfa budgets; [UNL 2026](https://cap.unl.edu/sites/unl.edu.ianr.agecon.center-for-ag-profitability/files/media/file/2026-nebraska-crop-budgets.pdf)
(with the cited 2025 pumping conversion/hours proxy); [UC Davis SJV alfalfa 2016](https://coststudyfiles.ucdavis.edu/uploads/cs_public/1c/e2/1ce256d0-957e-4bd4-b17e-18fef4efcedd/16alfalfasjv300acfinal_41916.pdf);
[Iowa 2025](https://www.extension.iastate.edu/agdm/crops/pdf/a1-20-2025.pdf).
Kansas uses the [archived 2026 K-State FMGs](https://www.agmanager.info/farm-budgets/historical-farm-budget-projections/2026-historical-budget-projections-non-irrigated)
(PDFs dated Nov-01-2025): south-central wheat, corn, sorghum, soybean and alfalfa;
southeast double-crop soybean is a regional transfer. Irrigated budgets use the
corresponding 2026 irrigated guides. Exact PDF URLs and native wage/N/energy
rates are in `sim/kansas.py` and options. Perennial mixed hay uses
[MF2143 brome, April2015](https://www.coffey.k-state.edu/farm-management/Brome%20Hay%20Cost%20Return%20Budget.pdf),
LOW, nominal2015 and a crop/region transfer: do not substitute annual forage
sorghum hay. The newer KFMA 2025 presentation
(`https://agmanager.info/sites/default/files/pdf/6_Dikeman_CropProfitability_2.pdf`)
reports operator-share rather than whole-field expenses and does not identify
N quantities/hours; it is not silently substituted for the reproducible FMG.
Every cross-state budget is flagged `transfer`; defaults remain LOW confidence.
CA's nominal 2016 costs are not inflation-adjusted. Transfers, old yields and
unmeasured well conditions are LOW confidence, not current farm quotes.
Source pumping lines are removed before actual simulator-applied mm are billed;
irrigation system ownership/repair is included **once** in costs and its reported
subtotal. CA's $130/acre-foot combines water and pumping, so no separate default
energy charge is added. The irrigation subtotal covers water/energy and separately
identified ownership (repairs sit in operating fuel/repairs); irrigation labour
remains in crop labour. Land is only in the listed-cost return when quoted;
payouts, government payments and wheat grazing receipts remain excluded.

Lubbock uses the [HPWD Rule 5.2(a)](https://www.hpwd.org/files/5fa25ede6/Amended+Rules+by+HB+2443+-+Rules+-+11%7B2%7D15%7B2%7D2023+%28ID+49614%29.pdf)
1.50 acre-foot/acre/year proxy (457.2 mm). The bucket caps annual net application
and resets each calendar year; it does not model daily pumping capacity or
distribution losses, so equating this to a gross pumping allocation is optimistic.
Fresno and York have no verified farm-wide total cap: unlimited supply remains
an explicit warning. Westlands' groundwater-only allocation is **not** imposed
on all Fresno surface-plus-groundwater supply. Capped yields use uncapped,
same-weather FAO reference ratios, so mean deficits are not normalized away.
ET anomaly validation still fails; irrigation volumes are uncertain scenarios.

Example request (1–12 rotations; each contains 1–6 years):

```json
{
  "rotations": [
    [{"crop": "soybean"}, {"crop": "corn"}],
    [{"crop": "soybean", "winter": "cereal_rye"}, {"crop": "corn"}]
  ],
  "rain_change_pct": 0,
  "temp_change_c": 0
}
```

Winter is **after** that year's cash harvest and before the next cash crop
(including the cycle boundary). Valid aliases: `cereal_rye`, `hairy_vetch`;
canonical `_cover` keys also work. Omit winter or use `"none"` for no cover.
These are unharvested covers. A supported autumn-sown crop can instead have
`"double_crop": "soybean"` (see `double_crop_option` in `/options`). Do not combine
double crop and winter cover on the same year. This reuses v22's existing
double-crop model; it is not a new model. `/simulate` evaluates only requested
cycles; `/scoreboard` enumerates them. No LLM choice happens in the service.

Request weather edits are extra: rain multipliers multiply, temperature offsets
add. E.g. farm rain −30%, request −20% means a total −44%. `existing` always uses
the registered farm's weather; `existing_under_request_weather` also includes the
extra what-if, so both comparisons are available.

The response has exactly these top-level fields:

```text
farm_id, simulator_version, device,
existing, existing_under_request_weather, results[],
applied_weather_changes, method, units, latency_seconds
```

Every result (also the two existing references) has:

```text
rotation, metrics, relative_yield_by_crop, family_pressure_index,
warnings[], income_confidence, income_confidence_drivers,
weather_runs, spin_years, scored_years_per_run
```

Every metric and each relative-yield / family-pressure value has
`{"p10": number|null, "median": number|null, "p90": number|null}`. Metrics:

```text
margin_usd_ha_yr                 field_water_use_mm_yr
soil_water_at_planting_mm         nitrate_leached_kg_n_ha_yr
fertilizer_n_kg_n_ha_yr           nitrogen_fixed_kg_n_ha_yr
erosion_t_ha_yr                   days_without_living_cover_yr
cover_biomass_t_dm_ha_yr          cover_establishment_fraction
soil_mineral_n_kg_n_ha            mineralized_n_kg_n_ha_yr
organic_matter_trend_pct_points_decade
double_crop_yield_t_ha_yr
return_over_listed_costs_usd_ha_yr
soil_carbon_input_t_c_ha_yr        forage_fraction
local_market_fraction
```

Relative yields are fractions of each crop's county/reference normal, not tonnes.
P10/P90 describe cycle-mean variation over windows × phases, not confidence
intervals. Cover establishment is the fraction of **all** scored crop-years,
including unsown years; biomass is the all-year mean. Failed sowings cost money.
Headline income includes operating irrigation charges, not fixed ownership.
Only the listed-cost return subtracts fixed costs and quoted land; neither is a new-equipment
purchase/financing plan. Applied irrigation and its cost are returned as
`applied_irrigation_mm_yr` and `irrigation_cost_usd_ha_yr`.

Errors are JSON `{"error": "...", "message": "actionable explanation"}`:
401 authentication, 404 unknown farm/route, 405 method, 422 invalid/infeasible
input, 500 unavailable data/accelerator. No stack traces in HTTP responses.
The inferred existing reference is still returned when its calendar is infeasible,
with explicit warnings; a matching requested existing cycle receives that same
reference exception. Alternatives with calendar clashes, <20% cover survival,
<60% maturity (except known existing crops), or isolated perennial stands fail
with reasons. Exact recognized equipment/forage constraints are enforced; other
free text is listed for agent review.

## Scope and numerical limitations

Preparation is serialized behind a lock and cached for three farms (LRU), with
three edited-weather preparations and 96 cycle summaries per farm. Other inputs
are immutable copies. `service_log.jsonl` logs requests, status, farm, rotations
and elapsed time, never the bearer token. Normal HTTP request logs are suppressed.

One/two/four-year cycles retain the planner's four-year spin and four-year scoring
windows (so Story parity holds). Other lengths score one full cycle and spin for
at least four years. This is v22's finite spin-up, not proof of physical steady
state. The v22 diversity bonus inspects four consecutive crops; five/six-year
transfers are marked low-confidence. A single perennial cycle renews annually,
as in v22; use a multiyear stand for a realistic renewal interval. Generic
family-pressure indices are reconstructed from v22's half-decay/+1 recurrence.
Nitrogen shortage does not reduce yield or automatically raise fertilizer rates.
GRACE is total water, not groundwater-only. Satellite records are county proxies.

## Deterministic scoreboard

```sh
curl "http://host.docker.internal:8601/farm/$FARM_ID/scoreboard" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  --data '{"weights":{"income":5,"water":2,"soil":3,"nitrogen":1,"erosion":2,"cover":1,"forage":0,"resilience":3,"market":2},"limits":{"min_margin_share_of_existing":0.7},"top":15}'
```

The nine accepted keys above are the entire scoreboard weight vocabulary;
missing keys are zero, unknown keys (including `nitrate`) and all-zero weights
return 422. Weights are normalized to sum to one. `GET /options` returns
`weights_help` (scored metrics, direction, plain-language meaning) and
`default_weights`. **Here `nitrogen` scores lower nitrate leaching**, not the
legacy planner's retained mineral-N-plus-credit objective. The underlying
planner's default weights are unchanged; v24 uses the operating-return definition above.

Optional request fields:

* `rain_change_pct`, `temp_change_c`: same multiplicative/additive semantics as
  `/simulate`, on top of registered farm overrides.
* `top`: integer 1–40, default 15.
* `limits`: any of:
  * `min_margin_share_of_existing`: 0–1, referenced to the existing rotation
    under the **same requested weather**. If existing margin is nonpositive,
    its actual margin is the floor (a negative margin does not license a larger loss).
  * `max_nitrate_kg_n_ha_yr`, `max_field_water_use_mm_yr`,
    `max_erosion_t_ha_yr`, `max_days_without_living_cover_yr`.
  * `require_crops`, `exclude_crops`: lists of `/options` crop keys. Requirements
    mean each named crop appears somewhere, not in every year. Winter crop keys
    and feasible `double_crop_soybean` may also be used.
  * `require_winter_cover`: at least one unharvested winter cover in the cycle.
    Double-crop soybean is not an unharvested winter cover.
  * `no_new_equipment`: main crops limited to the existing set, no new covers or
    double-cropping. `keep_forage_crop`: at least one forage crop-year.

Hard numerical limits apply to **medians of cycle means**, not P90s, worst years
or chance constraints. Explicit false values cannot relax an equipment/forage
constraint already in the farm registry. Free-text constraints outside the
recognized phrases still need agent review.

Every feasible cycle enumerated by the existing planner is simulated in one
batched pass: four main crop-years, one representative per cyclic shift, all
retained cover choices and feasible double-crop choices. No Pareto or income
floor is silently imposed. No trait-sensitivity reruns are needed. Soil and
weather overrides apply before enumeration and every simulation.
Calendar-infeasible variants of an inferred existing sequence are pruned using
the same checks as `/simulate`; the existing reference itself is still returned
and explicitly flagged. `/options` crop keys remain valid limit keys when a
weather edit makes their cycles infeasible (excluding an absent crop is a no-op).

Only survivors set min/max scaling bounds. `planner.rank` then produces the
weighted 0–100 score; a constant metric scores 50. Sorting is stable.
For each returned row, `tie_tolerance` is 1.96 × SD of the score difference
between the best and that row over 2,000 seed-0 paired resamples of weather
windows (all phases paired; bounds fixed to the survivor pool).
`tied_with_best` is true exactly when the displayed score lead is no larger
than that tolerance. The best row has zero tolerance and is tied with itself.
These are weather-resampling ties, not calibrated uncertainty in model traits,
costs, or the scientific adequacy of the simulator.

Response fields:

```text
farm_id, simulator_version, weights_normalised, limits_applied,
n_cycles_scored, n_after_limits,
existing {rank, score, metrics, rotation, violates_limits[], rank_is_counterfactual},
rows[] {rank, rotation, label, score, tie_tolerance, tied_with_best,
        metrics, income_confidence, warnings[]},
pruned[], method, latency_seconds
```

Row rotations are executable `/simulate` input objects, including `double_crop`
when present. Metrics have `{p10, median, p90}` and units in their names, as for
`/simulate`; additions are `soil_carbon_input_t_c_ha_yr`, `forage_fraction` and
`local_market_fraction`. Income confidence notes are in warnings.

Existing is **always** returned separately, whether or not it is in the top
rows or satisfies limits. Its rank is its placement against survivors. If it
violates limits/feasibility, that rank is explicitly counterfactual; it does
not become eligible or change scaling bounds. Scores can lie outside 0–100 for
an excluded reference projected onto survivor bounds. If no cycle survives,
`rows=[]`, existing rank/score are null, and violations explain why.

`data/processed/agent/scoreboard/` contains separate simulation `.json`/`.npz`
caches and `rank_*.json` response caches (directory mode 0700, files 0600).
Keys include soil/weather/irrigation/
registry constraints, simulator and adapter source hashes, relevant asset
metadata, weights/limits/top where appropriate. Changing only request weights,
limits or top never repeats numeric simulation. A fresh process can serve a
cached response without preparing a farm or loading a model; only
`latency_seconds` changes. `rows` are byte-identical on repeated requests.
Scoreboard requests, including weights and limits, are recorded in the normal
service request log. None of this restarts a running service; the owner must
restart after the pilot is done.

The local HTTP acceptance receipt is
`data/processed/agent/scoreboard_checks.json`, generated on the separate test
port 8611 and including complete default/nitrogen scoreboards, exact timings,
the byte-identical-rows check, and the numeric-simulation call count.

`brief.weather_outlook(farm)` now calls `outlook.build`: cached Open-Meteo SEAS5
51-member forecasts at the county cropland centroid. Initialization comes from
the provider's model metadata, not an inferred issue date. Verified daily JSON
uses the unnumbered control plus `_member01` … `_member50`; precipitation, mean/
minimum/maximum temperature and FAO ET0 are requested from `/v1/seasonal`.
Only complete months are reported (up to seven). No simulator weather changes.

Bias method: the API's monthly ensemble mean minus monthly anomaly supplies the
ECMWF hindcast-climatology **mean**. Monthly rain is scaled by the county/model
normal ratio; temperature receives the county-minus-model normal offset. County
normals and terciles use 1991–2020 PRISM-corrected POWER. This is NOT quantile
calibration: raw member forecasts are retained only one month; the live seasonal
archive probe rejects dates before 2025-09-01, and short-range historical/previous
runs do not supply decades of matching seasonal-member distributions. Spread,
wet-day frequency, daily frost tails, trends and the differing baseline period
remain uncorrected. `outlook.quantile_map` is a tested utility for a future proper
hindcast archive, not silently applied to the live forecast ensemble itself.

Each rendered outlook explicitly labels tercile member fractions as uncalibrated,
low-skill central-US seasonal guidance, particularly for rain. No county skill
metric is available through this API; no local skill is claimed. Published SEAS5
reliability is generally poorer for precipitation outside the tropics:
https://doi.org/10.1029/2021GL094662 . Official verification products are available
at https://charts.ecmwf.int (Verification → Long), but are not an API-supplied
calibration of this county transformation.

Cache: `data/processed/agent/outlook/<adm_id>/<initialization>.json`, actual raw
response plus provenance; initialization checked at most daily, failures cached
ten minutes. Outages, malformed/null responses, missing climatology and cache
errors yield “No forecast available” without breaking the brief. The source is
Open-Meteo/ECMWF under CC BY 4.0; free API non-commercial only, below 10,000 calls/
day, 5,000/hour and 600/minute. Commercial deployment requires a subscription.
Docs: https://open-meteo.com/en/docs/seasonal-forecast-api and
https://open-meteo.com/en/terms . The separate EC46 46-day model is not mixed with
SEAS5 here because that would require a separate lead/model bias treatment.

The anonymous brief keeps climate/soil numbers; calendar years become offsets
from the archive's final year. It does not promise geographic non-inferability
from climate patterns, only removal of explicit identifiers.

Local integration examples and measurements:
`data/processed/agent/{integration_checks,example_request,example_response}.json`
and `{story,anonymous_story,lubbock}.md`. Tests:

```sh
uv run pytest tests/test_agent_farms.py tests/test_agent_service.py
uv run pytest
uv run ruff check .
```
