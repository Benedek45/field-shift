# Public-readiness audit: repo and git history (draft, 2026-11-15)

> Status in the public `field-shift` repository: this is the pre-release audit of the private repository. Its actions were applied here: the subscription-login adapters (`anthropic_auth`, `openai_auth`, `vm/anthropic-v2/`, `vm/openai-oauth/`, probes) and `SUBMISSION.md` are not included, Kaggle account names read "account A/B", and `LICENSE` (MIT) and `LICENSE-docs` (CC BY 4.0) were added. File and line references below point to the private repository.

Scope: `Benedek45/nasa-farming`, all refs (`git log -p --all`). There are 45
commits up to `6590755`; `main` equals `origin/main`, so **making the repo public
publishes the whole history**. The `engine-v26` branch is fully merged. The audit
also covers the tracked working tree. Secret values are never reproduced here: only
commit, file, line and match length. Severity: **HIGH** = fix or decide before going
public, **MED** = decide, **LOW** = cosmetic or informational.

## 1. Secrets scan of the full history

Method: a stream scan of every added line in `git log -p --all` for `KGAT_…`,
`sk-ant-…`, `sk-…`, `Authorization: Bearer <literal>`, `Bearer <20+ chars>`, JWTs,
GitHub PATs, AWS/Google keys, private-key headers, and 32+ character hex/base64
strings within 40 characters of key/token/secret/password/auth. A tracked-tree grep
was run as well. Script: `$TMPDIR/opencode/scan_history.py` (not committed). It prints no values.

**Result: no credential found.** All 12 hits are false positives:

| Commit | File:line | Pattern (length) | Classification |
|---|---|---|---|
| `a5322fb` | `tests/test_agent_chat.py:228`, `:251` | keyword+base64 (32) | The invalid refresh sentinel string (uppercase words, "not a credential") |
| `2c24007` | `vm/anthropic-v2/entry.py:13` | keyword+base64 (32) | Same sentinel |
| `a26a9ac` | `src/fieldshift/agent/checks.py:64` | keyword+base64 (32) | Dictionary key name (lowercase words) |
| `7e0dba1` | `tests/test_agent_economics.py:225` | keyword+base64 (34) | Metric name |
| `12d4e07` | `src/fieldshift/planner/validate.py:276` | keyword+base64 (33) | Metric name |
| `4c02360`, `de5a96f` | `uv.lock:573`, `:586`, `:26` | hex (64) | Package `sha256` hashes |

Other checks:

- `tests/test_agent_chat.py:14` (current tree) builds a **synthetic** token that starts with
  the Anthropic OAuth prefix, followed by a repeated 3-character pattern, and is commented "synthetic". It is not a
  credential, but **GitHub secret scanning or push protection may flag it** once the repo is public. **LOW-MED**: build the prefix at
  runtime (e.g. `"sk-" + "ant-oat01-"`) to avoid alert noise.
- `Authorization: Bearer` appears only with placeholders or env variables:
  `src/fieldshift/agent/README.md:189`, `:438`; `src/fieldshift/agent/service.py:66`; `vm/farmsim:7`. OK.
- `.env.example` (tracked since its first commit): key **names** only, with every value empty in every
  revision. OK.
- `.agent-env`, `.env`, `.kaggle*/` and `**/credentials*` are gitignored and not tracked now.
  The history pattern scan found no credential in any added line. However, a direct
  history query by those **file names** was blocked by the harness safety rule
  (`secret.basename.env`), so it was not run. **Before going public, the user should run
  `git log --all --name-only --format= | sort -u`, and check that no such file name appears in it.**
  The name `.agent-env` appears in 11 tracked files, as a path only, never with content. **LOW**.
- No file was ever deleted from history (`--diff-filter=D` is empty), so there are no hidden past blobs.
- `git remote -v` holds no embedded credential.

## 2. Personal and identifying information

| Finding | Where | Severity | Decision |
|---|---|---|---|
| Personal Gmail address of the committer | **Author/committer metadata of all commits**; `pyproject.toml:6` | **MED** | Accept, or (a) change `pyproject.toml` to a team or noreply address and use the GitHub noreply address for future commits. Rewriting history to remove it from old commits is destructive (force push, new hashes; release hashes would change) |
| Kaggle usernames `account-A`, `account-B` and private kernel slugs | `EXPERIMENTS.md` (18 lines, e.g. 297-298, 383-384, 463-492, 557), `AGENTS.md:274` | **MED** | Keep or redact. The docs describe running parallel GPU jobs on two accounts. If both accounts belong to one person, check the Kaggle terms on multiple accounts before publishing that description |
| Google Cloud / Earth Engine project ID with the owner's first name | `AGENTS.md:65` | LOW | The ID alone grants no access. Optional redaction |
| GitHub username | `README.md:15` (clone URL), `pyproject.toml:6` | LOW | Expected for a public repo |
| Absolute home paths `/Users/<name>` | **none tracked**, in history or now. A `~/Library/Application Support/...` path is built at runtime (`src/fieldshift/agent/openai_auth.py:36`) | none | — |
| The machine's username in run artifacts | `~/.opencode-vm/...` run directories (not tracked) | none if run dirs stay out | Never publish run dirs wholesale |
| Story County cropland centroid coordinates | `tests/test_agent_outlook.py:2` | none | A public county centroid, not a farm |

## 3. Credential-handling code and docs: decisions, not defects

These files are well guarded technically: access-only tokens, an invalid refresh
sentinel, blocked refresh endpoints, 0600 files and scrubbing. Publishing them
exposes no secret. **What the user must decide is whether to publish them at all.**
This is stated neutrally; it is not legal advice.

| Files | What they do | Decision |
|---|---|---|
| `src/fieldshift/agent/anthropic_auth.py`, `tests/test_anthropic_auth.py`, `vm/anthropic-v2/{README.md,entry.py,no-refresh.patch,audit/}`, `vm/Dockerfile.anthropic`, `vm/opus_probe.py` | Read the user's **Claude subscription** (Claude Code) OAuth access token from the macOS Keychain, and run it in a container through the third-party plugin `opencode-claude-auth` 2.2.1 (MIT). Per `vm/anthropic-v2/README.md`, the plugin "injects Claude Code identity and content-derived billing system entries" and sets CLI headers | **HIGH (decision):** consumer-subscription terms may limit the use of subscription credentials outside the provider's own clients. Options: (a) publish as is; (b) publish without the adapter (remove the files in the release tree, but history still contains them unless rewritten), and describe the access path in prose only; (c) for v1.0, rerun Opus through the paid API with an API key and drop the adapter. The access path is stated factually in `RESULTS_v0.5.md` either way |
| `src/fieldshift/agent/openai_auth.py`, `tests/test_openai_auth.py`, `vm/openai-oauth/`, `vm/luna_probe.py`, `FIELDSHIFT_OPENAI` in `vm/oc.sh` | Use the user's **ChatGPT/Codex subscription login** (host OpenCode `auth export` or `~/.codex/auth.json`) with OpenCode's built-in ChatGPT plugin. Not used in B0 | **MED (decision):** the same question for OpenAI's subscription terms. It does not affect the B0 results. These files are under active edit by another agent; audit them again at release time |
| `AGENTS.md` sections "Opus subscription episodes" and "ChatGPT/Codex login" | Describe Keychain service use, token lifetimes and the internals of the safety-rule harness | **MED**: publish with the adapters or redact with them |
| `vm/anthropic-v2/no-refresh.patch` | Contains context lines from the MIT plugin | LOW: add the MIT notice (`LICENSES_AUDIT.md`) |

## 4. Size and content

- No tracked blob exceeds 1 MB in any revision. The largest is `uv.lock` (747 KB).
  The 236 MB vendored plugin and all `data/**` are gitignored.
- The release needs two gitignored files as **release assets**, not commits:
  `data/processed/bench/keys.json` (23.6 MB) and `data/processed/bench/b0_partial.json` (0.78 MB).
  Check them before upload: both have 0 matches for `/Users/`, `opencode-vm`,
  `KGAT_`, `sk-ant` and `Bearer` (checked 2026-11-15).
- No images and no NASA logos are tracked (Space Apps rule on generated NASA insignia: OK).
- There is no `LICENSE` file, and no `license` field in `pyproject.toml`. **HIGH**: add one before going
  public (`LICENSES_AUDIT.md` §4). Without it the code is "all rights reserved" by default.

## 5. Accuracy issues to fix before the release text is final

| Issue | Evidence | Severity |
|---|---|---|
| The documented B0 stop time (~01:00) conflicts with file times | Go-model repeat-0 `events.jsonl` files dated up to 2026-11-15 03:42 CET; runner logs show quota sleeps at 04:47 CET; Opus ends 00:56 | **MED**: reconcile in `bench/EXPERIMENTS.md` |
| The registration heading says "~23:45", but the commit is 23:31:12 and the first episodes end 23:34 | `06d8a09`; `events.jsonl` times | LOW: the commit precedes the episodes. Cite the commit time |
| The run contract lists the base image for the Opus runner, but the episodes used the anthropic image | `contract.json` `image_id` vs `meta.json` `image_id` (69/69 `d08baa…`) | LOW: documented in RESULTS |
| `bench/REPORT.md` is an obsolete handoff (v24 keys, smoke not run) | file | LOW: mark it superseded |
| The simulator fingerprint uses file modification times, so it cannot be reproduced elsewhere, and the current tree no longer matches `dd926469…` | `agent/scoreboard.py:98-116` | MED for v1.0 reproducibility |
| The Go-model runs record no served model | `meta.json` has no `served_model` | LOW: stated in RESULTS |

## 6. Decisions for the user before going public

1. **Publish the subscription-credential adapters (Anthropic, OpenAI)?** (§3)
2. **Licence set**: MIT for code, CC BY 4.0 for tasks, results and text, with quotes
   excluded (`LICENSES_AUDIT.md`)? Add the `LICENSE` and `NOTICE` files.
3. **Personal data in history** (Gmail, Kaggle accounts, EE project): accept, redact
   in the tree only, or rewrite history (destructive).
4. **QDANN statistics in code and docs**: attribute only (recommended for v0.5), or move
   them out of the MIT-licensed code.
5. **Study-card quotes**: keep with the exclusion, or trim them before v1.0 runs.
6. **Release assets**: publish full `keys.json` (dev keys), or a slim version without boards.
7. **Authors and affiliation** for `CITATION.cff` and `.zenodo.json` (currently `TODO-author`).
