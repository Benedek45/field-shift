# Field Shift: Adapting Farms with NASA Data

NASA Space Apps Challenge 2026 —
[challenge page](https://www.spaceappschallenge.org/2026/challenges/adapting-farms-with-nasa-data/)

Decision-support tool that helps farmers compare crop-rotation strategies (including their
current rotation) using NASA Earth observations, local soil data, crop characteristics and
farmer priorities.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 automatically).

```bash
git clone https://github.com/Benedek45/field-shift.git
cd field-shift
uv sync                 # creates .venv with all dependencies
cp .env.example .env    # then fill in your credentials
```

Run code with `uv run python ...`, tests with `uv run pytest`, lint with `uv run ruff check`.

## Layout

```
src/fieldshift/   Python package (shared code)
notebooks/        exploration notebooks
tests/            tests
data/raw/         downloaded data   (not in git)
data/processed/   derived data      (not in git)
```

## Advising a farm

```bash
uv run python -m fieldshift.agent.advise examples/farms/story-steady.json --out data/processed/advice/my-farm
```

Use a **fresh output folder** for each run. The command owns a temporary, random-token
simulator service on a free port and closes it afterwards; it does not use or restart
the development service. Local weather/model/soil/EO data and MPS/CUDA are required.
Record-view template runs are offline. Outlook requires the Open-Meteo seasonal API.

### Farmer form

See the four complete JSON files in [`examples/farms/`](examples/farms/).
Unknown keys, nonfinite values, wrong units and locally infeasible stated cycles
produce errors rather than silent substitutions.

| Field | Input |
|---|---|
| `location` | `adm_id` or `county: "Story County, IA"`; explicit boolean `irrigated`; optional `well_capacity: {"value":18,"unit":"in/yr"}` (or `mm/yr`) |
| `current_rotation` | Optional 1–6 seasons, e.g. `[{"crop":"corn"},{"crop":"soybean"},{"crop":"soybean"}]`; optional `winter:"cereal_rye_cover"` or `double_crop:"soybean"` per season, not both |
| `field_facts` | `slope_pct`, `organic_matter_pct`, `awc_mm_per_m`, or `water_holding_capacity: {"value":2,"unit":"in/ft"}` (`mm/m` also supported) |
| `priorities` | `preset`: `profit` (money 5), `steady` (money 4, water 2, soil 2, protect in bad years), `soil_builder` (soil 5, erosion 5, ground cover 4, money 3), `water_saver` (water 5, nitrogen loss 4, money 3); aspects a preset does not name are 0. Optional 0–5 `money`, `water`, `soil`, `nitrogen_loss`, `erosion`, `ground_cover`, `forage`, `local_market`; explicit values override the preset |
| `risk_attitude` (inside priorities) | `accept bad years`, `balanced`, `protect me in bad years` map to resilience 1/3/5 |
| `limits` | `minimum_share_current_income` (fraction or `{"value":80,"unit":"%"}`), `no_new_machinery`, `keep_forage_crop`, `require_winter_cover`, `include_crops`, `exclude_crops`; scoreboard `max_nitrate_kg_n_ha_yr`, `max_field_water_use_mm_yr`, `max_erosion_t_ha_yr`, `max_days_without_living_cover_yr` |
| `economics` | Optional `labour_usd_hr`, `fertilizer_n_usd_lb`, `water_usd_acre_foot`, `energy_usd_acre_inch`, `land_rent_usd_acre`; `crop_prices: {"corn":{"value":5.5,"unit":"usd/bu"}}`; `crop_cost_parts_usd_acre: {"corn":{"seed":100}}` |
| `weather_view` | `record` (historical comparison) or `outlook` (also run conditional SEAS5 scenarios) |
| `free_text`, `anonymous` | Farmer's words; optional boolean anonymity for the brief. The output package still contains private location/form data: do not assume it is safe to publish. |

Numeric max limits default to their named SI unit; nitrate also accepts
`lb_n/ac/yr`, water `in/yr`, erosion `ton/ac/yr`, bare days `days/yr`.
Economics also accepts the existing SI API keys documented in
[`agent/README.md`](src/fieldshift/agent/README.md).
Bushel prices use the simulator's NASS crop mass conversions; cotton prices are
`usd/lb` **lint**, hay `usd/ton` means a **US short ton as reported**.
No extra moisture or hay-quality adjustment is invented. Farmer facts/quotes are
labelled farmer, not independently verified.

The stated cycle replaces the satellite/county-inferred reference everywhere.
When supplied, all candidate cycles share an LCM(4, stated cycle length)-year
scoring horizon and equally repeated phases; a three-season cycle stays three
seasons. Continuous alfalfa uses the existing four-year stand enterprise policy.
Without a stated cycle the old four-year comparison is unchanged. Income means
**return over operating costs**, not profit after rent/ownership; the bad-year
value is historical P10, not a confidence bound. Limits constrain medians, not
every possible year. Water importance means field ET, not pumped irrigation.

### Model, checks and files

Default: `opencode-go/gpt-6-luna`, plus the existing Colima farming image/launcher.
Any model with an API key works: `--model openrouter/openai/gpt-6-luna`,
`anthropic/claude-sonnet-5-5`, `custom/<model>` and so on (see "Choosing a model and
API key" below). If the model's key is missing, the command names the variable and
stops before writing output. If the advisor fails or its decision does not pass every check,
the tool gives no answer; it never substitutes the code ranking for advice.

The form, **not the LLM**, sets weights and hard limits. Our measured
[A2 results](EXPERIMENTS.md#a2-results-80-runs-scored-2026-11-14-1705-dataprocessedagenta2_reportjson) and
[A3 results](EXPERIMENTS.md#a3-results-144-runs-scored-2026-11-14-2230-simulator-v24-dataprocessedagenta3_reportjson)
motivated this design: form weights were copied exactly and A3 choices repeated
in 11–12 of 12 cells. The product prompt preserves the older experiment prompts.
The model opens study cards, explains trade-offs and may add only explicitly
quoted free-text limits.

Code independently re-runs the scoreboard and chosen simulation, checks exact
form weights, unchanged/grounded limits, feasibility, top/tied ranking (or an
opened-card override within the top 15), opened/existing cards, and simulator
numbers. The current numeric check is **exact-or-rounding numeric presence in
the cited call, with unit conversions, not semantic metric attribution**.
A failed check means no recommendation is shown; the deterministic ranking is
retained only as an audit file. A pass is not agronomic validation. State/forecast
scenarios are low-skill and uncalibrated; the historical ranking is never replaced
by the forecast.

Outputs: `REPORT.md`, `DECISION.json`, `FARM.md`, `scoreboard.json`,
`checked_scoreboard.json`, `forecast.json` when requested, full `transcript.jsonl`,
normalised SI `form.json`, and `run.json` (versions/source hashes, timings, model,
usage and check results). `form.json` is a normalised receipt, not the public input
schema. Output folders also retain episode diagnostics, but not credential/SQLite
data. Treat briefs, transcripts and forms as private.

Go's API reported cost zero in our experiments: this is **not** proof of zero
subscription/quota cost. Episode token usage and reported cost are in `run.json`;
provider limits still apply. The free Open-Meteo API is noncommercial and requires
Open-Meteo/ECMWF CC-BY4 attribution. This tool supports, does not replace your judgment.

## Choosing a model and API key

The advisor runs in OpenCode inside the container, so any provider OpenCode knows works.
Write the model as `<provider>/<model id>`; the provider is the part before the first
slash. `src/fieldshift/agent/providers.py` decides how it is served and which variable
holds the key, from a vendored copy of the [models.dev](https://models.dev) catalog
(`provider_catalog.json`, 324 providers, fetched 2026-11-15; refresh with
`uv run python -m fieldshift.agent.providers --update-catalog`).

| Provider | Model example | Key variable |
| --- | --- | --- |
| OpenCode Go | `opencode-go/gpt-6-luna` | `OPENCODE_API_KEY` (or our alias `OPENCODE_GO_API_KEY`) |
| OpenRouter | `openrouter/openai/gpt-6-luna` | `OPENROUTER_API_KEY` |
| Anthropic API | `anthropic/claude-sonnet-5-5` | `ANTHROPIC_API_KEY` |
| OpenAI API | `openai/gpt-6-luna` | `OPENAI_API_KEY` |
| Google | `google/gemini-2.5-pro` | `GOOGLE_API_KEY`, `GOOGLE_GENERATIVE_AI_API_KEY` or `GEMINI_API_KEY` |
| xAI, DeepSeek, Mistral, Groq | `xai/grok-4.7`, `deepseek/deepseek-v4-flash`, ... | `XAI_API_KEY`, `DEEPSEEK_API_KEY`, `MISTRAL_API_KEY`, `GROQ_API_KEY` |
| Any other catalog provider | `<id>/<model>` | as listed in `provider_catalog.json` |
| Custom OpenAI-compatible server | `custom/<model id>` | `FIELDSHIFT_BASE_URL`, optional `FIELDSHIFT_API_KEY` |
| Ollama / LM Studio on this machine | `ollama/qwen3.5:9b` | none (`FIELDSHIFT_BASE_URL` overrides the port) |

Keys are read from the environment first, then from the gitignored `fieldshift.env`
(copy `fieldshift.env.example`), `.env` and `.agent-env`. `uv run python -m
fieldshift.agent.providers` lists the variable names it found; `--check <model>` shows
how a model would be served. A missing key is reported with the variable's name.

```bash
# OpenRouter: the full OpenRouter model ID follows the provider, slashes included
export OPENROUTER_API_KEY=...
uv run python -m fieldshift.agent.advise examples/farms/story-steady.json \
  --model openrouter/anthropic/claude-sonnet-5.5

# Any OpenAI-compatible server, here a local Ollama
export FIELDSHIFT_BASE_URL=http://localhost:11434/v1
uv run python -m fieldshift.agent.advise examples/farms/story-steady.json --model custom/qwen3.5:9b
```

How the key is handled: the launcher `vm/oc.sh` receives it in its process environment
and passes it into the container by name (`docker run -e NAME`), so it never appears on
a command line. The run's `opencode.json` holds only variable names, the model and
`small_model` (set to the same model, so OpenCode's hidden title and summary calls stay
on the provider that has a key). `localhost` in `FIELDSHIFT_BASE_URL` is rewritten to
`host.docker.internal`, because inside the container `localhost` is the container itself.

Not supported: subscription logins (Claude Pro/Max, ChatGPT Plus/Pro; `anthropic/*` and
`openai/*` without an API key are refused with that explanation), GitHub Copilot (device
login) and Vertex (credential file). Providers that need several variables (Azure, Bedrock,
Cloudflare) get every listed variable that is set; they were not tested. vLLM's own
discovery starts with tools disabled, so use the custom route for vLLM.

The measured results in this repository and the paper used OpenCode Go models and the
author's own subscription login (Claude Opus 5.5, and GPT-6 Luna for the live runs of
2026-11-15). Other models and providers have not been evaluated with the benchmark.

## Questionnaire app

```bash
uv run streamlit run src/fieldshift/app/advisor.py
```

Choose a county, check the inferred rotation and soil defaults, then set priorities,
hard limits and weather view. The editable price tables use US farm units; only
changed cells enter the farmer economics, and those rows are labelled **farmer**.
Submit validates the form before starting a background advisor. The page shows
elapsed time, simulator calls, a recommendation and two alternatives only when
the advisor's decision passes every check. Otherwise no answer is shown; the code
ranking is audit-only.

The default model is `opencode-go/gpt-6-luna`. The sidebar has a provider picker
(OpenCode Go, OpenRouter, Anthropic API, OpenAI API, custom endpoint), a free-text
model field, an endpoint field for custom servers and a password field for the key.
A typed key stays in the browser session's memory and reaches the advisor process
only through its environment; it is never saved and is used only for the provider it
was typed for. The sidebar lists the key variable names (not values) found in the
environment and key files. No credentials enter the questionnaire or its saved files. The existing VM image and Docker context
must be available (see the advising setup above). The live app runs shown in the demo
video used GPT-6 Luna run through the author's own subscription login; that adapter is not part of this repository.

Questions continue the solution's OpenCode session. Credentials are scrubbed
after each turn; if the session cannot be safely retained, the next turn replays
the conversation. Question-time simulator access cannot change questionnaire
settings; edit the named field and submit a new run instead. A changed decision
is checked again before it becomes the displayed solution.

Sessions are saved under `data/processed/app/<timestamp>-<county>/` with
`form.json`, `questionnaire.json`, `advice/`, `latest.json`, `qa.jsonl` and
`memory_proposals.json`. Reopen one from the sidebar; the sidebar also holds the
farmer profile picker (see "Study search and farmer memory"). Each session allows only one background job at a
time; closing the browser does not cancel it. The app supports, not replaces,
farmer and local agronomist judgment; simulator limitations still apply.

On macOS, do not import pyproj/PROJ into the Streamlit process before launching
workers: its at-fork SQLite cleanup can crash a child before exec. County-weight
construction imports geopandas lazily; the app tests guard this import boundary.

Validation (2026-11-15): 386 offline tests passed in 183 s, including scripted
solution/two-question/session-replay checks. The real-browser Story County run
(steady, corn $4.60/bu, no new machinery) reached the Opus lifetime gate:
1,893 s remaining versus 2,100 s required. No renewal or retry was attempted;
the app showed no recommendation and disabled question time. Live Q&A therefore
remains unverified for this app. Light/dark screenshots are
in `.playwright-mcp/advisor-{light,dark}-solution.png`; dark mode was tested with
CLI theme overrides because the repository config fixes a single light theme.

## Study search and farmer memory

**Study search.** `./farmsim studies '{"query":"cereal rye nitrate","top":5}'`
(`POST /farm/<id>/studies`, counted as `sim:<n>`, full response in
`.sim-response-<n>.json`) searches only the 56 cards in `data/evidence/cards`; it never
uses the web. Ranking (`src/fieldshift/agent/studies.py`) is BM25 over front matter and
Findings / Conditions / Quotes, a 21-group agronomy synonym table (catch crop -> cover
crop, lucerne -> alfalfa, N leaching -> nitrate, milo -> sorghum, ...; expanded words count
0.4), times a farm-fit factor (close 1.3, partial 1.1, far 1.0). No embedding model is
used. Each hit gives citation, access, region/climate/soil, up to three best Findings or
Conditions lines and `farm_fit`: `close` / `partial` / `far` with reasons (same or
neighbouring state; the farm's UNEP aridity class P/ET0 and thermal class from frost-free
days and mean temperature, compared with the card's climate words or stated rain; soil
texture from the county sand/clay when the card states one). Snippets are not citations:
the checks still require every cited card to be opened with the read tool.

Evaluation (`uv run python -m fieldshift.agent.studies --eval`, 16 queries with expected
cards over Story IA, Washington CO, Lubbock TX, Fresno CA and York NE; result in
`data/processed/agent/study_search_eval.json`): hits@3 BM25 alone 14/16, + synonyms
15/16, + synonyms and fit 15/16; `close` cards in the 48 top-3 slots 28 -> 28 -> 33.
The query set was written with the ranking (not held out), and one expected set
("N leaching under catch crops") was widened to kaspar2007 after inspection. The
remaining miss: "does a cover crop reduce the next corn yield" ranks the Kaspar
nitrate card above the cover-crop yield cards.

**Farmer memory.** The sidebar's *Farmer profile* picker creates or selects
`data/processed/app/profiles/<slug>/` with `PROFILE.md` (approved notes, at most 1,500
characters), `last_form.json` (last questionnaire, used to pre-fill the page) and
`sessions.json`. Approved notes are farmer statements: the page appends them to the
form's `free_text` under "Approved notes from earlier sessions", so a hard limit taken
from a note must quote it exactly and passes the unchanged grounding check
(`checks.text_limits` now also reads "does not allow / forbids <crop>"). The first agent
message lists them in "What the farmer told us before (approved notes)". Earlier sessions
of the profile are summarised into the workspace as `history/<session>.md` (date, county,
outcome, questions, first line of each answer). The agent may write only one extra file,
`MEMORY_PROPOSALS.json` (`[{"note", "source": "farmer said \"...\""}]`); the worker keeps
a proposal only if its quoted words occur in what the farmer typed in that session, and
the page lists it with Approve / Reject. If the notes would exceed the limit, the page
asks which old note to drop. Nothing is saved without a click.

## Chatting with the advisor

```bash
uv run python -m fieldshift.agent.chat                                # interactive; type quit to end
uv run python -m fieldshift.agent.chat --county "Story, IA"
uv run python -m fieldshift.agent.chat --script examples/chats/story.txt   # one farmer message per line
uv run python -m fieldshift.agent.chat --resume data/processed/chat/<session>
```

The farmer can start with only a county ("my farm is in Story County, Iowa"). The
advisor then asks, in plain words: irrigated or not, the current rotation (confirming
the one inferred from satellite crop maps, and saying so), priorities and bad-year
protection (presets profit, steady, soil builder, water saver), hard limits, own prices
and costs, and past weather or past weather plus the seasonal outlook. It proposes the
answers with `./farmsim form`; **the chat program (not the model)** shows them as a table
and applies them only if the farmer's next message is a plain "yes" (`chat_forms.is_confirmation`:
a yes-word and no correction, number, question or new fact). Until then ranking and
simulation are refused by the service. Any later change (a price, a limit, irrigation)
is a new immutable farm version and must be confirmed again.

After confirmation the advisor ranks with the exact confirmed weights and limits,
follows the product rules (top or tied choice; override only with an opened study card;
sources for every claim) and writes `DECISION.json` and `REPORT.md`. Whenever the
recommendation is new or changed, the host re-runs the product checks of
`agent.advise` and appends one line, e.g. `Checks: 7/7 passed`, or reports no checked
recommendation and names the failed checks. Follow-ups the farmer can ask:

- "What if next summer is 30% drier?": `simulate` with `rain_change_pct` (it applies to
  all rain in every historical year), or the outlook forecast for next season.
- "Water costs me $60 an acre-foot", "I get $5 more per ton for corn": a form change
  (confirm, then rerank). Prices that cannot matter (water on a rainfed field) are said so.
- "Why not alfalfa?": the alternative simulated against the choice.

**Models.** Default `opencode-go/gpt-6-luna` (`chat.DEFAULT_MODEL`); `--model` takes any
model from "Choosing a model and API key". The live chat check was
made with Claude Opus 5.5 run through the author's own subscription login; that adapter is not part of this repository.

**Continuity and credentials.** One OpenCode session per chat: each farmer turn is
`opencode run --session <id>` against the same run folder, so the agent keeps its
context. The provider credential is prepared fresh for every turn. After the turn the
host deletes the adapter's credential files, every row of OpenCode's `credential` and
`account` tables, and any other row containing the credential (SQLite secure-delete,
WAL truncated, VACUUM), then byte-searches every remaining file for the credential and
records the result per turn (`credential_scrub.verified_clean`). If the credential is
found anywhere, a turn is interrupted, or a continued session gives no answer, the
session store is deleted and later turns are fresh episodes whose first message replays
the conversation, the confirmed form and a list of earlier `sim:<n>` results.

**Session folder** `data/processed/chat/<UTC time>-<county>-<state>/` (private, 0700):
`conversation.md`; `turns.jsonl` (farmer text, reply, simulator calls, checks, seconds,
tokens, form version, credential scrub, per turn); `session.json` (resume state);
`forms/` (draft `v0-draft.json`, confirmed `v<n>.json`, every `proposal-<id>.json`);
latest `DECISION.json`, `REPORT.md` (product report and checks)
and `checks.json`; `transcripts/` and `events/` per turn; `agent/` (the run folder;
its `data/` holds only the scrubbed conversation store between turns).

**Limits.** One turn takes about one to three minutes with Opus (container start, tool
calls, model time). The confirmation test is deliberately strict: "yes, but…" or
"yes, 80%" is treated as an answer, and the table is shown again. What-if weather changes
are uniform over all historical years, not one season. The same simulator caveats as the
one-shot report apply; the checks test the decision's consistency, not agronomy.

## Notes

- Downloaded data never goes in git; keep it under `data/`.
- Credentials live only in `.env` (gitignored). See `.env.example`.
- USDA helpers try system DNS first, falling back to cached DNS-over-HTTPS only
  on lookup failure; hostname-based TLS verification stays enabled. No network
  settings need changing. A warning identifies each host using the fallback.
- Run `uv run python -m fieldshift.data.nass` to download the keyless crops bulk
  file resumably and stream county/annual/survey rows into
  `data/processed/nass_county_crops.csv.gz` (plus a summary JSON). No API key needed.

## Licence and data credits

- Code: MIT (`LICENSE`). Text, study-card summaries, benchmark tasks and results: CC BY 4.0
  (`LICENSE-docs`). Passages quoted inside `data/evidence/cards/*.md` remain their original authors'.
- No raw or processed data files are included (`data/` is gitignored apart from the study cards).
  The `fieldshift.data.*` modules rebuild them. The Earth Engine steps need your own registered
  Earth Engine project: pass `--project <your-earth-engine-project>` or set `EE_PROJECT`.
  Without the processed data, the tests that need it are skipped.
- **QDANN** 30 m corn and soybean yield maps (Ma et al. 2024, Lobell lab, via the Earth Engine
  community catalog), CC BY-NC-SA 4.0: used only for derived statistics (the rotation-penalty
  numbers); no QDANN pixels are redistributed. Their use is non-commercial.
- **NASA POWER** (NASA Langley Research Center POWER Project, daily MERRA-2 meteorology and
  CERES/SRB radiation), **MODIS** Collection 6.1 (MOD13A2 NDVI, MOD15A2H LAI/FPAR, MOD16A2GF ET, MOD11A2 LST;
  NASA LP DAAC), **SMAP** L4 soil moisture SPL4SMGP v8 (NASA NSIDC DAAC), **GRACE/GRACE-FO** mascon water
  storage (NASA JPL), plus AVHRR NDVI CDR (NOAA) before 2000, accessed through Google Earth Engine and the POWER archive.
- **PRISM** daily climate (PRISM Climate Group, Oregon State University).
- **Open-Meteo** seasonal outlook from **ECMWF** SEAS5: data CC BY 4.0 (Open-Meteo.com, ECMWF);
  the free Open-Meteo API is for non-commercial use.
- USDA NASS Quick Stats, crop progress and RMA cause-of-loss data (US public domain), CY-Bench
  (Zenodo), OpenLandMap soil, SRTM elevation, US Drought Monitor; details and terms in
  `bench/LICENSES_AUDIT.md`.

## Learned weather-stress yield response

### Expanded model (round 2)

```bash
uv run python -m fieldshift.data.crop_labels
uv run python -m fieldshift.model.expanded --weather cybench --benchmark
uv run python -m fieldshift.model.expanded --weather cybench --min-year 2003 --epochs 40 --sequence
# POWER archive 1981–2025:
uv run python -m fieldshift.model.expanded --weather power --epochs 40
# Matched-cohort weather comparison, with the same years and folds:
uv run python -m fieldshift.model.expanded --weather power_matched --min-year 2003 --max-year 2023 --cohort cybench --epochs 40
uv run pytest
uv run ruff check .
```

`EXPERIMENTS.md` records hypotheses **before** fits and outcomes afterwards.
The expanded modules are `data/crop_labels.py`, `model/expanded_data.py`,
`model/probabilistic.py` and `model/expanded.py`. No new dependency is required
beyond the first-round model dependencies. Neural code selects CUDA if available,
otherwise MPS, and refuses CPU or MPS CPU fallback.

#### Labels, irrigation and physiology

- Twenty records: corn grain/silage, soybean, winter/spring/durum wheat, alfalfa,
  other hay, oats, sorghum grain, barley, upland cotton, peanuts, dry beans
  (including chickpeas), rye, potatoes, oil sunflower, sugarbeets, rice, flaxseed.
  The broad dry-bean and other-hay labels represent mixtures, not pure cultivars.
- The new label module streams the raw NASS bulk **once** for county Census
  harvested areas. Survey labels come from the existing processed extract.
  Only harvested-acre yield units are used; wheat ALL CLASSES is excluded.
  County ANSI must match FIPS, series need eight observations overall, and zero,
  suppressed or otherwise nonnumeric yields are excluded from log-ratio fitting.
- Irrigated/total Census acres give shares at 1997/2002/2007/2012/2017/2022.
  Interpolate within each county/crop; hold the nearest available endpoint.
  This is a retrospective infrastructure proxy, especially before 1997—not
  observed annual historical irrigation. Missing/suppressed acres stay unknown.
  Explicit IRRIGATED/NON-IRRIGATED survey series are separate observations at 1/0.
  The old >10% exclusion is gone. Share and its missingness flag are inputs.
- `labels.parquet`, Census snapshots, interpolation provenance, source counts,
  unit conversions and audits live under `data/processed/yield_expanded/`.
  Initially 634,236 eligible labels; 61.35% have known irrigation.
- Traits retain a source URL and confidence: extracted AquaCrop-OSPy GDD
  defaults, FAO AquaCrop/FAO-56/FAO-66 priors, and explicitly low-confidence
  related-crop proxies where defaults are missing. Additional numeric traits
  represent C3/C4, legume, harvested organ, determinate habit, photoperiod,
  winter habit and perennial status. No crop/county ID or year is a predictor.
  Local cultivar, sowing dates and photoperiod responses remain unvalidated.
- `corn_silage` is the `corn` parent plus early whole-plant harvest overrides,
  not an unrelated crop. `load_traits()` resolves the inheritance.
- Established forage has **no annual emergence/flowering/maturity** in this
  representation. Green-up uses a temperature threshold; the season ends at
  first killing frost after August or December 31. Thermal cut opportunities
  summarise repeated regrowth, not observed cutting schedules. Other hay's
  perennial status and C3/C4/legume mixture are explicitly low-confidence.
  `explanation_stage_names(traits)` supplies appropriate forage/root-crop labels;
  root-crop shared reproductive slots mean organ initiation and bulking.

#### Neural model and honest evaluation

- Five independently initialised Gaussian mean/variance MLPs (128/64 hidden
  units), with fixed mean warm-up then heteroscedastic NLL. Total predictive
  variance includes aleatoric variance **and** between-member disagreement.
- The sequence variant uses three independent grouped temporal convolutions per
  member, trait FiLM, four temporal bins and the named stage/soil/irrigation
  features. Daily windows align to each crop's own sowing or forage green-up.
  All county-day weather is stored once as float16 on the accelerator; window
  gathering, normalisation, training, optimiser state and inference stay there.
  No per-batch host copies. Offline label/feature preparation and the requested
  sklearn references run on the host.
- All models share grouped harvest-year folds. Inside each outer training fold,
  every fifth sorted year is reserved for calibration; county/practice trends,
  feature scaling and county/month moisture climatology use only the other
  fit years. Calibration years are never refitted. Moisture calendar weights
  count only days with observed RSM. This replaces the old transductive
  leave-calendar-year-out climatology.
- A global predictive-SD multiplier is estimated from inner-year normalised
  error quantiles, never outer test years. It is a predictive calibration
  procedure, **not a guarantee** of 68/95% coverage for every crop or drought.
  Report actual held-out coverage, including failures. Coverage is in log-ratio
  space; intervals in ratio space are asymmetric.
- Each crop is held out on at least one prespecified outer fold; corn/silage
  and all wheat classes are held out as families. No held crop's response labels
  enter its predictor, scaler or calibrator. Its other-year yields supply only
  its county normal: conditional response transfer, not history-free yield.
- Trend, season trees, stage trees, MLP and CNN use identical rows. Primary
  reporting uses ALL-practice county yields, not those aggregates plus their
  overlapping irrigation strata. Separate all-strata and exact-share-zero
  rainfed summaries are retained, with rainfed county/year duplicates removed.
  Both equal-county and harvested-area weights are reported. Dedicated
  1983/1988/2012 summaries use IA/IL/IN/KS/MO/NE/SD and all counties.
- Predictions, fold/calibration years, timings, per-crop metrics and paired
  old-cohort comparisons are durable Parquet/JSON files beneath each weather/
  trait cache. Weather comparisons must use `--cohort cybench` and common years,
  rather than compare different county/crop samples.

Pilot evidence already rejects the assumption that more crops alone guarantees
transfer: mean held-crop MLP R² was −0.151, despite improved corn/soy/wheat transfer
and much better coverage. That engineering pilot precedes the corrected forage/
moisture representation; do not use it as the final headline experiment.

#### Calibrated simulator interface

```python
from fieldshift.features.stress import compute_stress, explanation_stage_names
from fieldshift.features.traits import load_traits
from fieldshift.model.expanded import predict_calibrated_relative_yield

traits = load_traits("corn_silage")
features = compute_stress(daily_weather, traits, sowing_day, harvest_year).iloc[0]
ratio_mean, predictive_sd = predict_calibrated_relative_yield(
    features, traits, soil, irrigated_share=0.0,
    model_path="data/processed/models/yield_response_v2_cybench.pt",
)
stage_labels = explanation_stage_names(traits)
```

Returns arithmetic mean and SD of the calibrated lognormal relative-yield
distribution, not seed-only spread or a confidence interval for the mean.
Evaluation point estimates remain mean log ratios. Supply the external county/
crop normal separately. Irrigation `None` means unknown, not rainfed.
Changing share is an observational scenario, not a validated causal irrigation
effect or a monotonic guarantee. County predictions are not calibrated field-
level risk estimates.

Match weather units to the checkpoint: CY-Bench RSM is GLDAS kg/m², POWER
`gwetroot`/`rsm` is a 0–1 wetness fraction; their climatologies are not
interchangeable. Other units remain °C, mm/day, kPa, MJ/m²/day. Soil uses the
original CY-Bench `awc`, `bulk_density`, `drainage_class` units.
VPD definitions also differ: native POWER uses FAO-56 mean saturation VPD,
CY-Bench uses saturation at Tmax. `power_matched` converts POWER VPD for the H5
comparison without modifying its memmap. Its checkpoint must receive that
matched definition; the native full-history POWER model uses unmodified POWER
VPD. `align_power_vpd_to_tmax()` is available in `model.expanded_data`.
Saved stage checkpoints are `yield_response_v2_{weather}.pt`; optional final
CNNs append `_sequence`. Matched-overlap POWER checkpoints have a `_cohort_...`
suffix and do not overwrite the full-history POWER model.

The original `yield_stress.predict_relative_yield(...)` now prefers a calibrated
v2 POWER/CY-Bench checkpoint if present and accepts `irrigated_share=`. Explicitly
selecting the archived v1 file retains its old seed-only behaviour with a warning.
Use the explicit v2 function/checkpoint above to avoid accidental source mixing.

### Recursive field-state model (round 3)

**Which model the tool uses.** For yield, the tool uses the calibrated MLP ensemble
(`yield_response_v2_power`). In the complete five-fold, year-grouped CV on POWER
1981-2025 (R3b, clean inputs), the recursive model is worse at yield. Macro
within-crop R² is 0.114 free-running and 0.118 teacher-forced, against 0.175 for the
MLP and 0.153 for the stage trees. It is below the MLP in every fold and in 16 of 20
crops. The recursive model is an experiment. Use it for what the yield models
cannot do:
- Roll soil water forward. Root, surface and profile wetness beat persistence at
  every lead and climatology for the first 17, 48 and 92 eight-day steps.
- Answer edited-weather what-ifs consistently, because soil water and canopy respond
  to edited weather instead of being fixed inputs.

It does not forecast canopy or ET better than the county climatology. Full
numbers are in `EXPERIMENTS.md` (R3b final) and
`data/processed/recursive/r3b/metrics.json`.

```bash
uv run python -m fieldshift.model.recursive_data          # canopy/static/planting helpers
uv run python -m fieldshift.model.diagnostics --progress --corn   # phenology vs crop progress, corn permutation ranking
uv run python -m fieldshift.model.stage_recursive --directory <dir outside repo> --owner <account> [--kernel --folds 0,1]
uv run python -m fieldshift.model.recursive_train --folds 0 [--held]   # outer fold(s)
uv run python -m fieldshift.model.recursive_train --final             # deployment model
uv run python -m fieldshift.model.recursive_report        # merge folds, compare with R2-D
uv run python -m fieldshift.model.recursive_checks --model <fold .pt> --output <json>
```

Modules: `model/recursive_data.py` (8-day drivers and observations, built on the
accelerator), `model/recursive.py` (network and water balance),
`model/recursive_train.py` (training, calibration, free-running evaluation),
`model/recursive_report.py`, `model/recursive_checks.py` (what-if and carry-over),
`model/simulator.py` (app entry point), `model/diagnostics.py`,
`model/stage_recursive.py`. Hypotheses, runs, failures and results are in
`EXPERIMENTS.md` (R3).

- **Time step:** 8-day periods, period *k* of a year starts on day-of-year 8k+1
  (MODIS convention, last period short), 46 per year. Daily POWER weather is held as
  float16 on the device and aggregated to periods there.
- **State (8 slots, fed back):** root/surface/profile wetness (POWER MERRA-2
  `gwetroot/gwettop/gwetprof`; the model learns to emulate the land model), NDVI, LAI,
  FPAR, actual ET (mm/day) and a surface-minus-air temperature proxy. A recurrent
  latent (3-layer GRU, 384 units) carries what is not observed. Observations are
  targets only; after the initial state, none enter the model during a rollout.
- **Clean R3b data:** MODIS NDVI/LAI/FPAR/actual ET/daytime LST and SMAP volumetric
  moisture supplement POWER. Before 2000-02-18, adjusted AVHRR NDVI is masked where
  its county harmonisation R² is below 0.3 or missing. Source harmonisation uses
  the full sensor overlap, a disclosed source-processing limitation. GRACE total
  storage and night LST are retained but not assimilated as extra soil/canopy slots.
  EO loading rejects infinity and out-of-range physical values rather than clipping.
  The first R3/v2 snapshot contained unmasked MODIS fill codes in LAI/FPAR; its
  corresponding state scores are invalid. `EXPERIMENTS.md` tracks the paired
  contaminated/clean comparison. The default deployment file is the **clean R3b**
  fit; old weights are `recursive_field_r3_contaminated_v2.pt`.
- **Drivers (never predicted):** 8-day weather, heat degree-days above the crop's
  critical temperature and 30/35 °C, frost and dry days, GDD increment and cumulative
  GDD as a fraction of maturity (the thermal-time phase), stage one-hots, flowering
  heat, crop-present flag, day length, numeric crop traits, soil/terrain statics,
  irrigated share. No crop or county IDs, no year.
- **Water balance:** three non-overlapping proxy reservoirs (root 180, surface 40,
  deep 400 mm). Only precipitation and irrigation add water; ET, runoff and drainage
  are non-negative and limited by what is stored, so water cannot appear. The
  per-period residual is reported (≈1e-5 mm). Irrigation is a learned automatic rule
  (refill to 60–90 % root wetness, at most 20 mm/day, scaled by irrigated share,
  only while a crop grows): a management assumption, not observed farmer behaviour.
  The reservoirs are surrogates: the MERRA-2 wetness products overlap physically.
- **Yield head:** reads only the target season of the free rollout (final active
  state, season- and stage-pooled states and latent) and returns mean and variance
  of log(yield / county normal), five-member deep ensemble, SD multiplier fitted on
  inner calibration years (never on test years).
- **Training:** 24 epochs; teacher forcing for 4 epochs, scheduled sampling down to
  zero over the next 4, then free-running full-window BPTT on 1- and 2-year windows
  (the state carries across 1 January). Loss: masked state error on every observed
  slot and step plus the Gaussian yield likelihood. CUDA uses fused AdamW with AMP;
  MPS uses the on-device AdamW. CPU is refused.
- **Sowing:** observed NASS state 50 %-planted dates are used as a management driver
  where available; otherwise the temperature rule (which is 2–5 weeks too early, see
  `data/processed/yield_expanded/phenology_check.json`).

#### Simulator entry point

```python
from fieldshift.model.simulator import simulate

out = simulate(
    initial_state,            # dict of the 8 slots, or a previous out["final_state"]
    daily_weather,            # whole Jan 1–Dec 31 years: tmin tmax tavg prec rad vpd et0 wind rh
    [{"crop": "corn", "year": 2026}, {"crop": "soybean", "year": 2027,
                                       "sowing_date": "2027-05-10"}],
    soil=soil,                # the checkpoint's static names (latitude required)
    irrigated_share=0.0,      # or per crop in the sequence; None = unknown
)
out["states"], out["state_std"]   # per 8-day step; SD = max(member spread, inner-year lead RMSE)
out["yields"]                     # per season: relative_yield_mean/std (lognormal), log moments
out["water_fluxes"]               # P, irrigation, ET, runoff, drainage, residual (mm)
out["final_state"]                # pass as the next initial_state to chain years
```

Weather can be edited freely (for example `prec * 0.7`); soil water, canopy and
yield then respond through the rollout instead of being fixed inputs. The default
checkpoint is `data/processed/models/recursive_field_r3.pt` (fit on all years except
the inner calibration years). Overlapping seasons are rejected; a winter crop in
the first year needs the preceding year of weather. Limitations: rotation and
nutrient effects are not identified from county data (a crop sequence changes
traits and timing only); irrigation response is learned from observational
county data; county-scale, not field-scale, risk.

### Archived three-crop reference (round 1)

The following describes the first-round protocol and saved results, not the
expanded pipeline's current data or calibration. Its OOF tables remain available.

```bash
uv run python -m fieldshift.model.yield_stress --build-only
uv run python -m fieldshift.model.yield_stress --folds 5 --epochs 50
uv run pytest
uv run ruff check .
```

Requires Apple MPS; neural training/inference explicitly refuses CPU fallback.
PyArrow provides streaming CSV-to-Parquet caches, scikit-learn provides the two
HistGradientBoosting baselines, and torch provides a five-seed batched MLP.
The AdamW implementation keeps every moment tensor on MPS (torch's standard
MPS AdamW otherwise creates a CPU step tensor). Numerical preprocessing of
daily data and the explicitly requested sklearn baselines run on the host.

Source: `features/traits.py`, `features/stress.py`, and `model/{dataset,trend,neural,yield_stress}.py`
under `src/fieldshift/`. Each numeric trait has a URL, confidence and, where
necessary, an explicit operational assumption. No crop/county identifier, year,
yield, irrigation flag or one-hot is a learned-model input.

#### Original physical assumptions and data handling

- Representative AquaCrop thermal-time traits, FAO-56 rooting depths, and FAO
  wheat physiology provide sourced priors, not locally calibrated cultivars.
  Corn uses base 8°C (AquaCrop), not a base-10 maize extension convention.
- Sowing is the first 10-day mean temperature ≥13°C for corn or ≥15°C for
  soybean within March 1–June 15; no crossing falls back to June 15. Soybean's
  threshold is a low-confidence warm-soil proxy using air temperature.
  Wheat uses the first 10-day mean ≤17°C in September 1–December 1 of the
  previous year, falling back to December 1 (FAO-56 table 11 footnote 2).
- Wheat requires 45 days at mean 0–7°C. Development is capped at 300 GDD until
  chilling is complete and January 1 has passed; GDD is zero below mean 5°C.
  These are deliberately simple dormancy/vernalisation approximations.
  Spring crop windows end October 31 and winter wheat August 31.
- GDD clips daily minimum/maximum to the crop's base/upper temperatures.
  Threshold-crossing days belong to the preceding stage; the maturity-crossing
  day is included. Flowering uses a crop-specific GDD interval. All stage
  quantities have physical names; unavailable stages/moisture remain missing.
- CY-Bench `sos` is **not used as a verified sowing date**, especially because
  US wheat often starts in January/February. Weather is crop-mask-weighted,
  not identical across crops. Use maize support first and wheat-only counties
  second, identically for all crops. The canonical union has 3,082 counties.
  Preserve original per-mask Parquet caches and quantitative overlap audits.
- AgERA5 radiation is converted J/m²/day → MJ/m²/day, VPD hPa → kPa;
  precipitation/ET0 are mm/day. CWB is recomputed as precipitation − ET0.
  Small source min/max inversions (≤0.952°C) are replaced by their midpoint;
  larger than 1°C raises an error. Counts are in `temperature_qc.json`.
- RSM anomalies use county/month-day climatology, excluding the observation's
  calendar year. This is label-free but uses other folds' weather, not a
  strictly training-fold-only operational climatology. Winter 2003 lacks
  prior-autumn moisture; missingness is retained. No NDVI/FPAR is used.
- NASS filters: TOTAL domain, YEAR reference, positive BU / ACRE yield,
  corn GRAIN / ALL CLASSES, soybean ALL CLASSES, wheat WINTER. County ANSI
  must match reported FIPS; residual county codes are excluded.
  Convert using 56 lb/bu corn and 60 lb/bu soybean/wheat.
- Prefer county NON-IRRIGATED series with at least eight years; otherwise
  retain ALL-practice series unless a known irrigated harvested-area share
  exceeds 10%. Unknown share is explicitly flagged, not assumed zero.
  **About 94–96% of retained labels have unknown irrigation share.**

#### Original evaluation and limitations

`data/processed/yield_stress/metrics.json` contains per-crop R² and RMSE in
log-ratio units, mean absolute percentage yield error, 2012 summaries, held
crop-and-year results, calibration, trait provenance and fold definitions.
OOF prediction tables allow independent recomputation. All are gitignored.

Five-fold harvest-year-grouped evaluation covers 2003–2023, 79,338 observations.
It is a full-season hindcast, not in-season or rolling-origin forecasting.
Trends are log-linear with county intercepts and county slopes shrunk toward
crop-wide slopes; trends, feature centring, scaling and imputation are fitted
only on training rows. No validation/test-based hyperparameter selection is used.
Counties need at least eight observations overall; missing county histories
fall back to the training crop-wide trend.

Initial measured R² / mean absolute percentage yield error:

| Crop | Trend | Season tree | Stage tree | MLP |
|---|---:|---:|---:|---:|
| Corn | 0.000 / 17.92% | 0.127 / 16.59% | 0.283 / 15.02% | 0.272 / 15.26% |
| Soybean | 0.000 / 16.33% | 0.121 / 15.24% | 0.190 / 14.64% | 0.162 / 14.94% |
| Winter wheat | 0.000 / 17.72% | 0.097 / 16.77% | 0.130 / 16.39% | 0.101 / 16.80% |

Held-crop tests also exclude the test harvest years from response training.
The held crop's **other years are used only to supply its county normal**,
not to fit the response or scaler; this is conditional response transfer,
not absolute-yield prediction without any crop history. MLP held-crop R² is
−1.43 corn, −0.51 soybean and −8.29 wheat: **zero-shot crop generalisation is
not validated**. Stage trees transfer better for corn/soybean but fail wheat.
Generic traits omit photoperiod and cultivar adaptation; only 69% of soybean
seasons reach prescribed thermal maturity. Do not present these as field-scale
causal stress effects or as crop-rotation-effect estimates.

The drought-state subset is fixed to IA, IL, IN, KS, MO, NE, SD; all statistics
weight county-seasons equally, not by harvested area. In 2012 observed corn
loss was 43.14% there; stage trees predict 29.13%, MLP 27.38%. Extreme losses
remain underestimated despite improvement over the trend-only baseline.
Uncertainty is **seed disagreement, not a calibrated predictive interval**:
±1/±2 ensemble log-space SD coverage is 36.5/60.8% corn, 34.0/57.7% soybean,
30.2/53.6% wheat, far below a Gaussian 68/95% reference.

#### Original simulator interface

```python
from fieldshift.features.stress import compute_stress
from fieldshift.features.traits import load_traits
from fieldshift.model.yield_stress import predict_relative_yield

traits = load_traits("corn")  # alternatively provide the same numeric trait keys
features = compute_stress(daily_weather, traits, sowing_day).iloc[0]
ratio_mean, ratio_seed_std = predict_relative_yield(
    features, traits, soil, model_path="data/processed/models/yield_stress_mlp.pt"
)
```

`daily_weather` uses date, tmin/tmax/tavg (°C), prec/et0 (mm/day), vpd (kPa),
rad (MJ/m²/day); optional rsm and rsm_anomaly use GLDAS kg/m². Optional adm_id
enables vectorised counties. `soil` supplies awc, bulk_density and drainage_class
in the original CY-Bench units. Supply an external RSM climatology/anomaly for
field use; absence is imputed by training statistics, not fabricated.
The saved ensemble is `data/processed/models/yield_stress_mlp.pt`.
Outputs are arithmetic mean and sample SD of exponentiated seed predictions;
evaluation averages seed log-ratios. The caller supplies the county-normal
absolute yield separately. A new crop needs a trait record, not retraining,
but the held-crop failure means its predictions must be flagged experimental.

## Running the tool

```bash
uv run streamlit run src/fieldshift/app/main.py      # opens the rotation explorer
uv run python -m fieldshift.planner.validate         # multi-county NASA/NASS + Story + five contrasting counties
uv run python -m fieldshift.planner.prices           # one-off: NASS prices received -> data/processed/planner/
```

Uses an Apple MPS or CUDA device by default and never falls back to the CPU silently; set `FIELDSHIFT_DEVICE=cpu` to run on the CPU (Story County, full farm pipeline: about 12 s on an M5 Pro CPU against 15 s on its GPU). Story County, IA
(`US-19-169`) opens from the cache in a few seconds. Any other county in the NASA POWER archive is
computed on demand and cached as compressed `.npz` arrays plus JSON under
`data/processed/planner/`. The seven-crop search retains cyclically distinct rotations, optional
winter covers and feasible wheat/double-crop soybean. Each weather run spins up a full cycle,
then scores the next at every starting phase. Priority sliders only re-rank stored results.
Weather edits preserve the learned baseline and multiply it by edited/unedited FAO-33 yield
(season-level factor, see validation);
a zero edit exactly reproduces the baseline. The pinned light theme works in dark-mode browsers.
After editing imported simulator/planner modules during development, restart the Streamlit process:
a page rerun can retain old imported functions even while source-based cache keys change.
Current irrigation/economics code uses `fieldshift-v25`; the round-5 validation figures below
describe v22, not a new validation of irrigated-farm economics. Do not restart an active
evaluation API; its owner must restart it after the evaluation finishes.

Irrigated sites now prefer historical NASS IRRIGATED yields (rainfed: NON-IRRIGATED), followed by
same-practice neighbours/official state series (county means if absent), then all-practices. Dates and low-confidence old/sparse
fallbacks are shown. Cotton is lint, using LB/ACRE ×0.00112085 t/ha and UPLAND prices.
Applied irrigation is priced from cited regional budgets, with fixed system costs once.
Headline margin is now return over **operating** costs; ownership/overhead and quoted rent
are in the separate `return_over_listed_costs_usd_ha_yr` metric, not income/resilience scores.
Kansas uses cited 2026 K-State guides (2015 brome proxy for mixed perennial hay); cross-state
and crop/region budget transfers are explicitly LOW confidence. Source-unsplit fees stay operating.
V25 adds explicit alfalfa establishment and a cited four-calendar-year stand scenario
(fall/no-hay establishment plus three production years), not four full-yield harvests.
Termination precedes reseeding by a year; dates and cross-region stand-life transfers are LOW confidence.
Mixed stands establish in-year: TX/CA fall (no hay), otherwise reduced spring harvest.
Identified annualised establishment is removed before cash establishment is charged.
Recent state Census whole-crop irrigated/non-irrigated hay yields replace obsolete county proxies;
suppressed practice yields can use newer state production/area totals, explicitly ALL-practice.
Hay is as-reported baled mass with unspecified grade/moisture; 88% DM is a working convention,
not observed NASS moisture. `market` now uses county harvested-area share, not binary presence
or verified buyer demand. Sources and stand/cost caveats: agent README.
Lubbock has an optimistic 457.2-mm/year HPWD allocation proxy; unknown total supplies remain
uncapped with warnings, not falsely assigned Westlands' groundwater-only limit.
The farming-agent service supports immutable farmer price/cost overrides and reprices cached
physical outputs without field simulation; a capacity edit does change physics. See
[`src/fieldshift/agent/README.md`](src/fieldshift/agent/README.md) for schemas, sources and limitations.
Simulate and scoreboard expose the same retained/humified soil-carbon input quantiles
(`soil_carbon_input_t_c_ha_yr`), not a measured stock change.

Rain for the water balance is NASA POWER rescaled to PRISM (4 km, gauge-based) 8-day totals,
keeping POWER's wet days inside each step; where POWER is dry and PRISM is wet, PRISM's total is
split equally over `wet_days` evenly spaced days (an assumption). The learned yield model keeps the
POWER rain it was trained on (`prec_power`). Both choices, and the season-level FAO-33 factor, were
adopted under rules fixed before running (`validation_weather_tests.json`).

Rye is sown after the preceding crop's own harvest operation (soybean typically before corn).
It germinates at about 1.1°C and grows above 3.3°C (SARE); little autumn growth no longer means
death. Autumn plus spring thermal time controls biomass. Extreme winter air temperature is
a conservative failure proxy, without snow insulation. Vetch retains its time-to-frost,
heat-unit and hardiness screens; late Iowa vetch is pruned. Failures still cost money.
Non-leap calendar padding is excluded from winter temperature minima (it is not lethal frost).
The calendar shows flowering, covers, failed covers, bare periods and RMA insured-loss months.
Where field history is unavailable, the baseline is explicitly an inferred alternating sequence
of the two largest *supported* county crop areas, not an observed farm rotation. Cotton uses
NASS's UPLAND class; corn uses grain, not silage. Orchards are outside this tool's scope.

Nitrogen is tracked monthly using bucket drainage, seasonal cash-crop uptake, a living winter-cover
store and separate soil/residue N pools. Applied N is a reference recommendation minus preceding
legume and legume-cover credits (floored at zero); additional cover fertilizer savings enter margin.
The corn rate is an explicitly historical Iowa MRTN scenario, not current fertilizer advice.
Corn uptake now follows Bender's two-thirds by VT/R1, with explicitly approximate within-stage
interpolation; a potential-ET curve had back-loaded N demand. Other crops retain the ET proxy.
Mineralisation seasonality (Q10=2, 2%/yr at 20°C), complete-mixing leaching, fertilizer prices
and transfer to other states remain uncertain. Unmet N demand does not affect yield and is **not**
shown as an agronomic outcome; it remains an internal diagnostic. No N-response curve is implied.

Displayed alternatives must be Pareto-optimal on every actively scored outcome and retain at
least 70% of the existing median margin (adjustable). A negative baseline cannot justify a larger
loss. Goal labels name the actual measure; more carbon input does not mean less nitrate.
The trade-off scatter uses cached results, not fresh simulations. All-zero comparison rows are hidden.
Near ties: weather windows are resampled 2,000 times (phases paired, both rotations on the same draw),
and a goal winner whose score lead over the existing rotation is within 1.96 × the SD of the paired
difference is not shown; the existing rotation carries the label ("as good a match as any candidate").
Remaining slots are filled only with candidates that beat the existing rotation beyond that noise.

The visible NASA section shows recent MODIS greenness and a low-greenness winter-gap proxy, SMAP
Level-4 root-zone moisture, US Drought Monitor D2+ summer frequency, and GRACE/GRACE-FO total-water
storage. NDVI is not proof of bare soil; SMAP includes a land model; GRACE mascons are roughly 3°
and combine groundwater, soil water, surface water and snow. Drought is not a NASA-only product.
Storage trend below −0.5 cm/year or drought in at least 30% of summers triggers an **optional**
water-priority suggestion, never a silent slider change. These thresholds are screening choices,
not statistical significance tests.

Davis et al. 2012 ([Marsden Farm](https://doi.org/10.1371/journal.pone.0047149)) found corn +4% and
soybean +9%, manufactured N 80/16/11 kg/ha/year and established-system margins $953/965/913 for
2/3/4-year systems. The simulator transfers the yield gains to diverse cycles as LOW-confidence
scenarios, and uses cited corn-after-established-alfalfa N reductions. It does **not** reproduce the
study's underseeded clover/alfalfa, manure, straw sales, reduced herbicides or additional labor.
`validation_diverse.json` reports actual candidate ranks and a separately simulated three-year oats
proxy (without clover). The exact Marsden four-year system is not in the search.

### Validation and limitations

`data/processed/planner/validation.json` contains the reproducible report;
`validation_counties.json`, `validation_et.csv`, `nitrogen_before.json`,
`nitrogen_after.json` and `validation_other_counties.json` retain the diagnostics.
The 300-county state-stratified selection requires 25 yield years per crop: explicit
NON-IRRIGATED series are preferred; low-irrigation and humid-state unknown-practice series
are flagged rainfed proxies. County yield trends and FAO normalisation use even years only.
No simulator parameters are calibrated in the tool (the rejected Schlenker-Roberts variants fitted
one heat slope per crop on even years only).

Round 5 tests (300 counties, odd-year median r corn/soybean; mean |observed − simulated| loss over
1983/1988/2012 × two crops): POWER rain 0.426/0.475, 10.2 points; PRISM-scaled 0.492/0.505, 10.2
(adopted). On PRISM rain: stage product 0.492/0.505, 10.2; Schlenker-Roberts degree days above
29/30 °C 0.492/0.465, 9.3 (rejected, soybean r fell); season-level FAO-33 0.509/0.527, 6.6
(adopted); season + Schlenker-Roberts 0.488/0.488, 5.2 (rejected, r fell although drought error was
lowest). Current corn loss observed/simulated: 1983 23.8/4.0, 1988 27.2/23.7, 2012 30.0/25.4%;
soybean 8.8/4.9, 20.7/16.0, 16.6/19.6%. 1983 is still largely missed.
MOD16 comparison covers 170 corn-dominated counties: bucket ET 443 vs MODIS 386 mm, summer anomaly
correlation only 0.12.
**This is mixed evidence for yield and a failed summer-water-anomaly check**, not a general
validation pass. County crop mixtures, no capillary rise, irrigation ambiguity and seasonal
phenology need further work; a generic FAO seasonal-water range is not observational validation.

The dry/wet ET split uses within-county lower/upper thirds of summer rain/ET₀. Median ET anomaly
r is 0.23/0.42 (round 5 run). In wet summers MOD16 vs LAI r is 0.85, vs humidity 0.09; bucket vs ET₀ r is 0.87.
This is consistent with a canopy-versus-reference-demand difference, not proof of its cause.
The overall 0.10 anomaly check still fails; no humidity-based explanation is established.

For Story County, the corrected model gives mean rye nitrate loss reduction about 20%, versus
61% load / 59% concentration reduction in Kaspar et al. 2007
([doi:10.2134/jeq2006.0468](https://doi.org/10.2134/jeq2006.0468)). Subsequent Iowa years
showed 48% concentration reduction but no significant cumulative-load reduction (Kaspar 2012).
Correct sign does **not** mean magnitude is calibrated.
Iowa rye trial means span about 0.72–2.23 t dry matter/ha, while late-sown vetch mostly winterkilled
([SARE Iowa trials](https://projects.sare.org/project-reports/lnc13-352/)).
Simulated rye averages 1.21 t/ha over all years, 1.28 in successful years, with 95% simulated
establishment/survival (100% median four-year-window rate). Vetch averages 0.33 overall / 3.29 in
its rare successful years and is pruned.
The all-year mean must not be presented as validation of successful-stand biomass. The cover
GDD rule now uses daily mean temperature: clipping Tmin before averaging had created spurious
growth on freezing-mean days. Biomass/WP and unobserved snow protection still need local validation.

Code: `src/fieldshift/sim/` (trait extension, FAO-56 bucket, N, carbon, RUSLE, pest ledgers, the
batched rollout), `src/fieldshift/planner/` (site data, candidate search, ranking, explanations,
validation), `src/fieldshift/app/` (Streamlit). The tool supports and does not replace farmer judgment.

## Linux server and Windows setup

The Python app/simulator runs on the host; only the OpenCode agent runs in a Linux
container. Install Python 3.12+, `uv`, and Docker Engine 20.10+ with BuildKit on a
Linux amd64 or arm64 server. The account running Field Shift needs access to Docker
(Docker socket access is effectively root access; do not expose it to app users).
Use a local Docker daemon: bind-mount paths and the simulator must be reachable
from that daemon. Restore or rebuild the processed data described above; the public
checkout does not include it.

From the repository root:

```bash
uv sync --locked
docker build -t fieldshift-agent:oc2.3.0 vm
docker run --rm --entrypoint opencode fieldshift-agent:oc2.3.0 --version
export FIELDSHIFT_DEVICE=cpu   # explicit CPU operation on a server without CUDA
uv run streamlit run src/fieldshift/app/advisor.py --server.address 127.0.0.1
```

The version check requires no model, API key or paid call. `vm/Dockerfile` downloads
the official OpenCode 2.3.0 npm binary for the build's `TARGETARCH`, verifies a pinned
SHA-512 checksum, and checks its version. The amd64 binary is the baseline build
for CPUs without AVX2; arm64 is native. To publish both architectures with your own
Buildx builder, use `docker buildx build --platform linux/amd64,linux/arm64 -t
<registry>/fieldshift-agent:oc2.3.0 --push vm`. Neither image contains provider keys.

Configure the provider using "Choosing a model and API key" above, then open the
questionnaire through an SSH tunnel or an authenticated TLS reverse proxy. Do not
publish Streamlit directly on an untrusted network. Agent episodes start an
authenticated simulator listener on the host; allow Docker bridge traffic to it,
but firewall it from external clients. `host.docker.internal:host-gateway` is added
per container, including on Linux Engine. A local-model server must likewise
listen on a host address reachable from Docker, not only loopback.

Docker launch, image inspection and timeout cleanup use the same endpoint:

- Default: Docker's current context / `DOCKER_CONTEXT` / `DOCKER_HOST`; Field Shift
  never switches the global context.
- Override only Field Shift: `FIELDSHIFT_DOCKER_CONTEXT=<name>`.
- Image override: `FIELDSHIFT_IMAGE=<registry>/fieldshift-agent:<tag>` (default
  `fieldshift-agent:oc2.3.0`).

The Python launcher (`uv run python -m fieldshift.agent.containers <run-dir>
[opencode args...]`) requires existing `cfg/`, `data/`, `work/` directories. Only
those absolute paths are mounted. Pilot/chat call Docker directly with argv lists,
without a host shell; keys are forwarded as `-e VARIABLE_NAME`, never key literals.
`FIELDSHIFT_PASS_ENV` accepts space-separated environment variable names only.
POSIX launches use the host UID/GID so output remains writable without making run
folders world-writable. `vm/oc.sh` remains a Unix compatibility wrapper.

**Native Windows:** install Python/uv and Docker Desktop in **Linux containers**
mode, with the project drive available for bind mounts. Use the same `uv` and
`docker build` commands in PowerShell; set CPU mode with
`$env:FIELDSHIFT_DEVICE = "cpu"` instead of `export`. No Git Bash, WSL shell, Colima,
or host `chmod` command is required for pilot/chat launches. Run the Python module
above rather than `vm/oc.sh`. Multi-platform CI covers the offline suite; native
Windows Docker execution still needs an end-to-end check on a Windows host.

**Verification:** both arm64 and amd64 agent images were built and ran the pinned
CLI as their non-root user. The mounted launcher and container-to-host HTTP
connection were also checked without provider credentials. The current source
passed 345 offline tests (69 skipped for unpublished data) with explicit CPU mode
on macOS. A real arm64 Linux CPU smoke, using an earlier source snapshot whose
physical engine was unchanged, ranked 434 Story County cycles, replayed 32 state
members and produced five one-year scenarios in about 30 seconds with finite
outputs. This is an execution check, not evidence of agronomic prediction skill.
The latest-source [platform CI run](https://github.com/Benedek45/field-shift/actions/runs/43764938454)
passed on Linux, macOS and Windows: **349 tests passed, 69 unpublished-data tests
skipped on each OS**. The Linux amd64 Docker build and non-root CLI check also
passed. This verifies the offline runtime, not a key-funded agent episode or native
Windows Docker execution.
On Windows, session folders inherit the parent folder's ACLs; keep the checkout and
session storage in a private user directory. POSIX `0700` mode bits are not a Windows
access-control guarantee.
