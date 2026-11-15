# Field Shift demo video (HyperFrames)

30 s, 1920x1080 MP4 with a synthesized soundtrack, for the Space Apps submission. Built with
[HyperFrames](https://github.com/heygen-com/hyperframes) 0.9.52 (Apache 2.0); GSAP 3.15.0 is vendored in
`assets/vendor/`. Every frame is a real app screenshot or text/graphics drawn by code from project data.
No generated images, voice or music; no NASA logos (dataset names appear as plain text only).
The sound is computed by `scripts/make_audio.py` (numpy/scipy oscillators and filtered noise; no samples,
no downloaded tracks, no AI audio): a D-major pad (Dmaj9, Bm7, Gmaj7, A7sus4, 7.5 s each, slow attack and
release) with a soft plucked arpeggio, a noise swell into each cut, ticks when the form values and tool calls
appear, a two-note chime (A5, D6) on "9/9 checks passed" and a low tone on the end screen. Event times are
the scene starts in `index.html` plus the in-scene offsets of the GSAP timelines. The mix is gain-matched to
-16 LUFS integrated (true peak about -4 dBTP); render output is AAC 48 kHz stereo, 192 kbps.

## Build

```bash
cd video
uv run python scripts/extract_layers.py   # real Story County series -> assets/data/story_layers.js
uv run python scripts/crop_shots.py       # assets/raw/*.png + scripts/shots.json -> assets/shots/*.png
uv run python scripts/make_audio.py       # -> assets/audio/soundtrack.wav (gitignored, 5.8 MB)
npx hyperframes check                     # lint, layout, motion, contrast
npx hyperframes snapshot --at 1,4,8,12    # look at frames before rendering
npx hyperframes render --quality delivery -o renders/final.mp4
ffprobe -v error -show_entries format=duration renders/final.mp4   # must be <= 30.0
ffmpeg -i renders/final.mp4 -map 0:a -af ebur128=peak=true -f null -   # about -16 LUFS, peak <= -1.5 dBTP
```

`renders/` and `assets/audio/*.wav` are gitignored; never commit an MP4.

## Swapping screenshots

Compositions only reference `assets/shots/{questionnaire,progress,comparison,checks,qa}.png`. To replace
one: put the new raw PNG in `assets/raw/`, update `source` and the crop `box` in `scripts/shots.json`, rerun
`crop_shots.py`, then render. Shot sizes are read from `assets/shots/shots.js`. The corn-price highlight
in `form.html` uses a fixed row position inside the questionnaire crop (comment there); move it if that
crop changes.

Sources (all live app runs on 2026-11-15, GPT-6 Luna served as `gpt-6-luna`, Story County IA, "steady"
preset, no new machinery, corn price changed to $4.60/bu, light mode, 1600 px wide, Playwright):
- questionnaire: `live2-questionnaire-full.png`, session `data/processed/app/20261115-150339-story-ia/`
  (git 1886881; the form is identical in both sessions).
- progress, comparison, checks, qa: `live4-progress.png` and `live4-solution-qa-full.png`, session
  `data/processed/app/20261115-190206-story-ia/` (git de1cb70, 9 checks). One run, one try each question.
Older raws (`live-*.png`, `live3-*.png`, `live2-progress.png`, `live2-nosolution-full.png`, `advisor-light-questionnaire.png`,
`qa-luna-*-dark.png`) are kept but unused.

## Shot list (30.0 s)

| Time | Composition | What it shows | Source |
|---|---|---|---|
| 0-2.4 | `title.html` | "Field Shift", subtitle, soybean/corn rotation strip | text |
| 2.4-5.8 | `form.html` | Questionnaire scrolled from Hard limits (no new machinery ticked) to the price table; corn price row (4.60, default 5.05, source "farmer") outlined | `live2-questionnaire-full.png` |
| 5.8-9.6 | `layers.html` | Story County IA: NASA POWER summer rain 1981-2025 (1988, 2012 marked), MODIS NDVI 2011/2012/2013, SMAP root-zone moisture, GRACE water storage | Drawn from `data/processed/weather/power_county_daily.f32`, `eo/eo_county_8day.f32` by `scripts/extract_layers.py` |
| 9.6-12.2 | `why.html` beat 1 | Drought years -20 to -36% yield vs corn after corn -1 to -4%; no data says which rotation is right | Corn loss 1983/1988/2012, 7 states (AGENTS.md "Recursive model, final"); QDANN 0.96% and Seifert 2017 4.3% (AGENTS.md) |
| 12.2-15.2 | `why.html` beat 2 | Best network explains 17.5% of yield swings (24% rainfed), fails on unseen crops; field-state network 11%; so the network sits inside the simulator and the LLM reads, runs, explains | `data/processed/recursive/r3b/metrics.json`, EXPERIMENTS.md (MLP 0.175, rainfed 0.237, recursive 0.114) |
| 15.2-18.6 | `agent.html` | Live progress at 26 s (options, a failed shell read of the notes then a normal read, scoreboard, study search, two study cards, scoreboard output); "434 -> 6" rotations; study search with farm-fit labels (Kaspar 2007 close, Myers-Bailey 2026 close, Abdalla 2019 partial) | `live4-progress.png`; `agent/work/.sim-response-1.json` (`n_cycles_scored` 434, `n_after_limits` 6) and `-2.json` (study search, 56 cards) |
| 18.6-22.4 | `solution.html` | "Keep soybean -> corn, $422/acre typical, $329 in a bad year"; recommended banner and comparison table (alternatives $390/$306, $398/$296; 18.8 inches); then the code checks table, 9 PASS, "9/9 checks passed" | `live4-solution-qa-full.png`; `advice/run.json` (63.2 s, `all_passed` true), `advice/REPORT.md` |
| 22.4-26.2 | `qa.html` | "Why not alfalfa?" (18 s): about -$6/acre vs $422 (bad year -$34 vs $329), excluded by the no-new-machinery limit; "What if next summer is dry?" (-20% rain, 22 s): $392 vs $422 typical, $316 vs $329 bad year, plus a low-skill 2027 outlook | `live4-solution-qa-full.png`; `qa.jsonl`, `agent/work/.sim-response-3.json` to `-5.json` |
| 26.2-30.0 | `bench.html` | Benchmark: right rotation / right and grounded: Opus 97/91, Luna 93/69, DeepSeek 84/76, GLM 87/47; "No answer is better than a bad answer."; repo link | `bench/RESULTS_v0.5.md` (repeat 0) |

## Known limits of the current cut

- The alfalfa answer's first sentence contradicts itself ("wasn't ruled out by the equipment limit; it was
  excluded ... because no new equipment allows only crops already grown"). Its first session attempt failed
  (exit 1) and the app replayed it once as a fresh episode (`qa.jsonl` `host` note).
- `check` warns that several lines are on screen shorter than its reading-time rule (inherent to a 30 s
  cut), that SVG chart axis labels in `layers.html` are 20 px.
- The repo link (`bench.html`, variable `repo`) points to `github.com/Benedek45/field-shift`.
