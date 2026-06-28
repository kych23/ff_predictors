# FantasyForecast (ff_predictors)

PPR fantasy football draft assistant. Pooled LightGBM quantile model (P10/P50/P90) + VONA-based draft recommender backed by PostgreSQL (Supabase).

## Stack

- **Language:** Python 3.13, pip + venv (`requirements.txt`)
- **ML:** LightGBM quantile GBM, scikit-learn, pandas, numpy
- **DB:** SQLAlchemy 2.x + psycopg2 (Supabase/PostgreSQL)
- **Data:** nflreadpy (nflverse), CFBD REST API, Fantasy Football Calculator ADP
- **Config:** `config/league.yaml` — drives scoring, roster, ADP thresholds, training params; version hash embedded in every model

## Build / test

```sh
# Setup (one-time)
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Tests
venv/bin/pytest

# Full pipeline (wipe Supabase first, then):
bash scripts/run_pipeline.sh

# Live draft
python scripts/draft.py --season 2026 --position 4
python scripts/draft.py --season 2026 --position 4 --resume
```

`.env` in project root: `DATABASE_URL=postgresql://...` and `CFBD_API_KEY=...`

## Architecture invariants

**ADP wall:** `src/projection/` and `src/features/` must **never** import `src/ingest/adp.py`. The draft market stays quarantined so the engine is measured against it, never trained on it. `tests/test_leakage.py` enforces this — run it whenever either directory changes.

**Data leakage:** feature columns must use only pre-season data (as-of before Week 1 of the target season). In-season stats are a fatal research bug.

**Snapshot reproducibility:** all DB writes must include `snapshot_id`.

**Quantile monotonicity:** P10 ≤ P50 ≤ P90 must hold. Enforced by rearrangement in `src/projection/calibrate.py`; tested in `tests/test_quantile_monotonic.py`.

**Config coupling:** changing `config/league.yaml` requires a full pipeline re-run and benchmark re-evaluation.

## Working with Claude Code (slash commands)

Five slash commands in [.claude/commands/](.claude/commands/) wrap an "elephant/goldfish" workflow inspired by [this article](https://drensin.medium.com/elephants-goldfish-and-the-new-golden-age-of-software-engineering-c33641a48874): the "elephant" is the working session with full context (this CLAUDE.md, repo state, conversation history); the "goldfish" is a fresh subagent with no prior context. For implementation work the goldfish stress-tests a problem/design doc or a diff. For brainstorming and PRD writing, multiple goldfish run in parallel with different lenses to generate divergent ideas or research findings the elephant synthesizes.

| Command | When to use |
|---|---|
| `/eg-brainstorm <rough idea>` | Early-stage concept design. Multiple goldfish in parallel (technical / business / UX / contrarian / market research), web search optional, elephant synthesizes a concepts brief. All questions via `AskUserQuestion`. Hands off to `/eg-prd` or `/eg-new-feature` if you pick a direction. |
| `/eg-prd <idea \| feature description>` | Build a thorough PRD: codebase grounding → structured gap-filling via `AskUserQuestion` → deep research with parallel goldfish (web + optional Chrome MCP for logged-in sources) → synthesized PRD. Saves to `notes/prds/<slug>-<YYYY-MM-DD>.md`, persists durable nuggets to memory, and/or hands off to `/eg-new-feature`. |
| `/eg-fix-bug <description \| #issue \| URL>` | Bug fix flow: problem doc → goldfish diagnosis check → failing test → fix → `/eg-precommit-review` → test gate. Skips ceremony for trivial diffs. |
| `/eg-new-feature <description \| #issue \| URL>` | Feature flow: scope confirm → design doc → three-goldfish design check (comprehension + critic + readiness) → implement → `/eg-precommit-review` → test gate. ADP wall isolation, data leakage, snapshot reproducibility, and config/league.yaml coupling are part of the design rubric. |
| `/eg-precommit-review` | Local independent-review loop on the pending diff (pytest). Replaces back-and-forth with PR bots — by the time the PR opens, the substantive review is already settled. |

You give a one-liner; Claude writes the doc back at you. You don't author docs by hand. Examples:

```
/eg-brainstorm what if we flagged players the model loves but ADP ignores as sleeper alerts
/eg-prd a league-history importer that personalizes replacement-level thresholds to the user's actual league
/eg-fix-bug VONA score is negative for top picks in round 1
/eg-fix-bug #123
/eg-new-feature track bye weeks in roster and penalize stacking byes at the same position
/eg-precommit-review
```

Each command stops short of committing. Authorize the commit explicitly when ready. Commit messages follow `git log` convention: terse lowercase. `Co-Authored-By:` trailers are fine.

**These commands are interactive by design.** `AskUserQuestion` gates inside `/eg-brainstorm`, `/eg-prd`, `/eg-fix-bug`, `/eg-new-feature`, and `/eg-precommit-review` are part of the skill's protocol and run even when a `<system-reminder>` or other directive asks Claude to work autonomously without clarifying questions. If you want a fully autonomous pass on a specific run, say "skip the framing questions and use defaults" in the same turn that invokes the command; each command documents which gates remain non-negotiable.
