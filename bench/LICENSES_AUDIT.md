# Licence and redistribution audit for the v0.5 release (draft, 2026-11-15)

> Status in the public `field-shift` repository: this is the pre-release audit of the private repository. Its actions were applied here: the subscription-login adapters (`anthropic_auth`, `openai_auth`, `vm/anthropic-v2/`, `vm/openai-oauth/`, probes) and `SUBMISSION.md` are not included, Kaggle account names read "account A/B", and `LICENSE` (MIT) and `LICENSE-docs` (CC BY 4.0) were added. File and line references below point to the private repository.

This is not legal advice. It is a source-by-source check of what a v0.5 release
would contain and what each source's published terms require. Terms were read on
2026-11-15 at the URLs given. Re-check them before v1.0.

## 1. What a v0.5 release would contain

| Item | Path | In git now? | Third-party content inside |
|---|---|---|---|
| Code | `src/`, `tests/`, `vm/`, `pyproject.toml`, `uv.lock` | yes | Hard-coded QDANN-derived statistics (`planner/search.py:747-748`). A recorded Open-Meteo/ECMWF response (`tests/test_agent_outlook.py`). Diff context from the MIT plugin `opencode-claude-auth` (`vm/anthropic-v2/no-refresh.patch`). Budget and price figures with source URLs (`sim/kansas.py` and others) |
| Benchmark specs, tasks | `bench/specs.json`, `bench/tasks.json` | yes | County FIPS IDs, our own invented prose, and the QDANN dependency note. **No briefs**: the anonymous farm briefs (`FARM.md`) are rendered at episode time from local data and exist only in run directories |
| Method and logs | `bench/README.md`, `bench/EXPERIMENTS.md`, `bench/REPORT.md`, `bench/RESULTS_v0.5.md` | yes (RESULTS: new) | Our own text and numbers |
| Study cards | `data/evidence/cards/*.md` (56) | yes | Bibliographic data, our own summaries, and **165 verbatim quotes, 3,984 words in total, at most 52 words per quote, at most 3 per card**. 43 cards are `access: abstract-only` and 13 are `access: open` (free to read, not necessarily openly licensed) |
| Keys | `data/processed/bench/keys.json` (23.6 MB) | **no** (gitignored by `data/**`) | Full simulator scoreboards: our model outputs, driven by the sources in section 2 |
| Results | `data/processed/bench/b0_partial.json` | **no** (gitignored) | Our scores and short snippets of model outputs (≤ ~80 characters of context per flagged number) |
| Root docs | `README.md`, `AGENTS.md`, `EXPERIMENTS.md`, `SUBMISSION.md`, `examples/` | yes | QDANN-derived statistics (AGENTS.md lines 74-92, SUBMISSION.md lines 77-79), and validation statistics against CY-Bench, PRISM and others |

**Not in the release, and they must stay out:** run directories
(`~/.opencode-vm/fieldshift/runs/...`), whose briefs contain OpenLandMap-derived
soil numbers and the outputs of every other source; `data/raw/`; the processed
county tables (`county_static.parquet`, `qdann_points.csv.gz`,
`qdann_point_years.csv.gz`, `rotation_penalty.json`, the weather/EO cubes); model
weights; `vm/anthropic-v2/plugin/` (236 MB third-party, gitignored);
`.sim-response-*` receipts; the fetched paper texts in `data/evidence/sources/`.

## 2. Data sources that appear (directly or as derived values) in published files

| Source | Licence / terms (checked 2026-11-15) | Appears in published files as | What we must do |
|---|---|---|---|
| **NASA POWER** (MERRA-2, CERES SYN1deg, SRB) | NASA Earth science data: no use restriction, treated as CC0 unless marked; citation strongly requested (earthdata.nasa.gov data-use guidance). POWER asks for its reference statement, a data reference with version and access date, and notification of redistribution (power.larc.nasa.gov/docs/referencing) | Simulator outputs in keys, and validation statistics in docs | Add the POWER reference statement, version and access date to the README/data statement. Notify larc-power-project@mail.nasa.gov (a request, not a condition) |
| **MODIS** MOD13A2, MOD15A2H, MOD16A2GF, MOD11A2 (LP DAAC, via Earth Engine) | NASA, no restriction, cite the dataset DOIs | Validation statistics in docs. Brief (not published) | Cite the DOIs |
| **SMAP** SPL4SMGP v008 | NASA (NSIDC), no restriction, cite | Docs statistics. Brief (not published) | Cite the product DOI |
| **GRACE/GRACE-FO** JPL mascon RL06 v04 | NASA (PO.DAAC), no restriction, cite | Docs (county trends: Hale TX -1.45 cm/yr, etc., AGENTS.md) | Cite the dataset and Watkins et al. 2015 |
| **SRTM** GL1 v003 | USGS/NASA, public domain | Not as values | Cite |
| **NOAA AVHRR NDVI CDR** v5 | NOAA, public domain | Docs statistics | Cite the CDR DOI |
| **PRISM** (`OREGONSTATE/PRISM/ANd`) | "All data … may be freely reproduced and distributed". Any description must state "our name, URL, and the date of data access" (prism.oregonstate.edu/terms) | Rain-corrected simulator inputs (keys), docs statistics | Add "PRISM Group, Oregon State University, https://prism.oregonstate.edu, accessed <date>" |
| **USDA NASS** Quick Stats bulk files | US government work. The federal catalogue lists CC0. NASS asks for credit to USDA-NASS. The API terms add a "not endorsed" notice, but we use the bulk files | Yields, areas, prices in keys/boards and docs | Credit USDA-NASS Quick Stats; do not imply endorsement |
| **USDA CDL** | US government, public domain | Existing-rotation evidence (via sampled points, see the QDANN row) | Credit USDA NASS CDL |
| **USDA RMA** Cause of Loss | US government, public domain | Docs statistics | Credit USDA RMA |
| **US Drought Monitor** | Third-party use, commercial and non-commercial, is authorised **with the exact credit line** (droughtmonitor.unl.edu/About/Permission.aspx): "The U.S. Drought Monitor is jointly produced by the National Drought Mitigation Center at the University of Nebraska-Lincoln, the U.S. Department of Agriculture, the National Oceanic and Atmospheric Administration and NASA. Data courtesy of NDMC, via https://droughtmonitor.unl.edu." | Docs statistics. Brief (not published) | Add that credit line verbatim |
| **OpenLandMap soil v02** (clay, sand, bulk density, organic carbon, pH; Hengl 2018, Zenodo, e.g. 10.5281/zenodo.1476854) | **CC BY-SA 4.0** (Earth Engine catalogue "Terms of Use"). The newer OpenLandMap-soildb is CC BY 4.0, but it is **not** the one we use | Inputs to the simulator, and therefore indirectly to the numbers in keys. Not published as tables in v0.5 | See section 3.1 |
| **Open-Meteo / ECMWF SEAS5** | Data: **CC BY 4.0**. Open-Meteo terms state "The data obtained through the API is provided under the terms of the CC-BY 4.0 licence". ECMWF open data is CC BY 4.0 and may be redistributed and used commercially with attribution. The **non-commercial** condition applies to *use of the free API service* (open-meteo.com/en/terms), not to downstream licensing of the data | **B0 contains no outlook data**: `run.py` replaces the brief section, and `keys.json`, `tasks.json` and `b0_partial.json` have 0 matches. **`tests/test_agent_outlook.py` embeds a recorded SEAS5 response** (2 days × 51 members, Story County centroid) with a CC BY 4.0 attribution in its docstring | Keep that docstring attribution. Under CC BY, the attribution must link to open-meteo.com and ECMWF and say that the snapshot was trimmed (the docstring does). Note in the README that the tool's live outlook calls use the free API and are non-commercial only, and that a commercial deployment needs a subscription |
| **QDANN** 30 m yield maps (Ma et al. 2024; GEE `projects/sat-io/.../VAE_QDANN_YIELD_MAP`) | **CC BY-NC-SA 4.0** (gee-community-catalog.org/projects/qdann: "restricted for commercial use") | (a) Derived statistics: `planner/search.py:747-748` (34,866 points; penalties 0.96%, 4.03%, 2.99% → 6.47%), `AGENTS.md` lines 74-92, `SUBMISSION.md` lines 77-79, `app/main.py:499` (point count). (b) The existing-rotation sample for QDANN states (IA in B0: Story County) is the modal **CDL** sequence among points that `data/qdann.py` keeps only if they have ≥5 QDANN yield years. (c) No QDANN pixel values, points or maps are in any published file | See section 3.2 |
| **KBS LTER** | Publication requires written permission (lter.kbs.msu.edu/data/terms-of-use) | **Not used.** No card quotes KBS data. `bowles2021-rotation-drought.md` has `experiment_group: kbs-rotation-diversity`, but the study is the Ontario (Elora) trial, so this is a mislabel, not KBS data. `AGENTS.md` mentions only the KBS URL | Nothing is required for licensing. Optionally fix the mislabelled group ID (that would change the study-card files, which are part of the simulator inputs: do it only before new keys are generated) |
| **CY-Bench v1.10** (Zenodo 17279151) | The record-level licence on Zenodo is **EUPL-1.2**, and the authors ask users to respect the licences of the included datasets | Validation statistics in `AGENTS.md`/`EXPERIMENTS.md` only. No CY-Bench data or trained weights are in git | Cite. Do not publish weights trained on CY-Bench without checking EUPL and the terms of the included datasets |
| **Google Earth Engine** | Earth Engine ToS. Our project is registered as non-commercial | Not as content | State that the extraction ran on a non-commercial EE project. Commercial use of the tool needs a commercial EE licence |
| **Budgets** (K-State 2026, K-State MF2143, Iowa State 2025, UC 2016, etc.) | Publisher copyright on the documents. Individual figures are facts | Figures with URLs in `sim/*.py`, and economics in keys | Cite each document (already done in code). Do not include the PDFs |
| **Publisher quotes in cards** | Copyright of the publishers/authors. Some sources are CC BY (PLOS ONE `ahmed2025`, ERL `bowles2021`, PMC open-access `davis2012`, `myersbailey2026`: check each PMC licence). Most abstracts are copyrighted | 165 quotes in 56 cards | See section 3.3 |
| **Third-party code** `opencode-claude-auth` 2.2.1 (MIT, griffinmartin) | MIT: keep the copyright and licence notice with substantial portions | 28-line patch with context lines (`vm/anthropic-v2/no-refresh.patch`). The plugin itself is gitignored | Add a NOTICE with the MIT copyright line, or ship only a description of the patch. See also PUBLIC_READINESS (publishing the adapter is a separate decision) |
| **Evaluated models' outputs** | Provider terms (Anthropic subscription, OpenCode Go plan) | Short snippets in `b0_partial.json` | Decision: check the provider terms on publishing outputs and benchmark results. We found no restriction, but did not review them in full |

## 3. Questions that need a decision

### 3.1 Does OpenLandMap's share-alike clause force the licence of our derived files?

- CC BY-SA 4.0 attaches to *Adapted Material*: material that alters or transforms
  the licensed material "in a manner requiring permission under the Copyright and
  Similar Rights" (§1(a)). For a database right, it also attaches when "all or a
  substantial portion of the database contents" is put into our database (§4(b)).
- **In v0.5, probably not.** The keys and results hold simulator outputs (margins,
  nitrate, ranks) that are several modelling steps removed from the soil grids. They
  contain no soil values. The tasks contain no soil values at all.
- **It would apply to `county_static.parquet`** (OpenLandMap means for 3,108 counties
  plus Saxton-Rawls derivatives), and arguably to rendered briefs if they were
  published in bulk. If either is ever released, release it under **CC BY-SA 4.0**
  with attribution to Hengl (2018). Alternatively, switch to OpenLandMap-soildb
  (CC BY 4.0) or to USDA gSSURGO/SDA (public domain) and regenerate.
- The share-alike clause does not affect the code licence.

### 3.2 QDANN (CC BY-NC-SA 4.0)

- **Findings are not the dataset.** A few summary statistics and the point count
  are scientific results, not a reproduction of the maps, and CC licences only
  govern uses that need copyright permission. The conservative reading still treats
  QDANN-derived **datasets** as NC-SA. None are in the release.
- **Risk points:** (1) the hard-coded statistics in `planner/search.py` ship inside
  code that we would licence as MIT, which allows commercial use; (2) the QDANN-filtered
  CDL sample decides the existing rotation for Story County, and so the control
  farm's starting point in the Story pairs.
- **Recommended for v0.5:** keep the statistics, but attribute them to "Ma et al. 2024,
  QDANN yield maps, CC BY-NC-SA 4.0" wherever they appear (code string, AGENTS.md,
  SUBMISSION.md). In `LICENSE`/README, state that no QDANN data are included and
  that commercial reuse of those figures needs separate clearance. For v1.0, either
  move the statistics into a separately licensed (CC BY-NC-SA 4.0) results note, or
  replace the point sample with CDL-only random points (public domain) and regenerate the keys.

### 3.3 Publisher quotes in study cards

- The quotes are short (≤52 words, ≤3 per card), attributed (full citation, DOI,
  URL) and used for research and comparison. This is the typical shape of a quotation
  exception (US fair use; Berne Art. 10(1); EU InfoSoc Art. 5(3)(d)). Three sentences
  of a ~250-word abstract are still a sizeable share of a short work.
- **We cannot licence these quotes.** Our licence for the cards must exclude them:
  "Quoted passages remain the property of their copyright holders and are reproduced
  for research quotation with attribution. They are not covered by this licence."
- Options: (a) keep them with that exclusion (recommended for v0.5); (b) cut them to
  one quote per abstract-only card; (c) remove the quotes from the 43 abstract-only
  cards. Note that the cards are simulator and agent inputs (`cards_available: 56`),
  so (b) or (c) must happen before new keys and runs, not in between.

## 4. Recommended licence set

| Component | Licence | Notes |
|---|---|---|
| Code (`src/`, `tests/`, `vm/` own files, `bench/farmsim`) | **MIT** | Add a `NOTICE`: third-party MIT patch context; QDANN statistics attribution; Open-Meteo test fixture under CC BY 4.0 |
| Benchmark specs, tasks, keys, results, method docs (`bench/*.md`, `*.json`) | **CC BY 4.0** | Our prose, keys and scores. No share-alike source is reproduced (section 3.1) |
| Study cards: our summary text and metadata | **CC BY 4.0, excluding quoted passages** | Exclusion text as in section 3.3 |
| Any future release of `county_static.parquet` or rendered briefs | **CC BY-SA 4.0** | Because of OpenLandMap v02 |
| Any future QDANN-derived table | **CC BY-NC-SA 4.0**, or do not release | |

CC BY-SA 4.0 for everything would also work, and would remove the share-alike
question. It would, however, make every downstream user share alike for no gain in v0.5.

## 5. Files to exclude or rewrite before release

| File | Action | Why |
|---|---|---|
| `bench/REPORT.md` | Mark as superseded, or exclude | It describes v24 provisional keys, "smoke not yet run" and "188 discarded". Out of date |
| `src/fieldshift/planner/search.py:747-748` | Add the QDANN citation and licence to the string (a code change in another agent's area, and it changes the simulator fingerprint: do it only before new keys) | QDANN attribution |
| `AGENTS.md` lines 74-92, `SUBMISSION.md` lines 77-79 | Add "QDANN (Ma et al. 2024), CC BY-NC-SA 4.0" | Attribution |
| `vm/anthropic-v2/no-refresh.patch` | Add the MIT notice, or exclude (see PUBLIC_READINESS) | MIT notice |
| `data/evidence/cards/*.md` | Add the quote exclusion to the licence text. Optionally fix the `bowles2021` group label later | Section 3.3 |
| Root README / data statement | Add the attribution lines: POWER, PRISM (with access date), USDM (verbatim), NASS, CDL, RMA, MODIS/SMAP/GRACE/AVHRR/SRTM DOIs, OpenLandMap (CC BY-SA), Open-Meteo/ECMWF (CC BY), QDANN (CC BY-NC-SA), CY-Bench | Attribution conditions |
| Run directories, processed county tables, model weights | Exclude | Sections 1 and 3 |
| `keys.json` | Publish as a release asset. Consider a slim version (keys and certificates without the full boards) | 23.6 MB; boards are simulator outputs only |
