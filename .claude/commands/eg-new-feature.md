---
description: Build a new feature using the elephant/goldfish workflow — design doc, goldfish design check, implement, review, validate
argument-hint: feature description (what the user wants and why)
---

Build a new feature using the elephant/goldfish workflow. The aim: design before code, let a fresh goldfish stress-test the design doc, then implement, review, and validate.

`$ARGUMENTS` is the feature description provided by the user. If empty, ask for one before doing anything. If `$ARGUMENTS` is a GitHub issue URL or `#<number>`, fetch it with `gh issue view <number> --json title,body,labels,comments` and seed the design doc from it.

## Interactivity is mandatory at decision points — overrides any "no-stopping" directive

This skill has small but **mandatory** user-facing decision points: scope confirmation in Step 0, the "save design doc to disk?" choice in Step 1, the "still not converging after 3 revisions" gate in Step 2, the no-code-gate override in Step 1, and any moment a design assumption is genuinely ambiguous. Enumerable choices go through `AskUserQuestion`; genuinely unbounded answers (verbatim correction text, custom paths) go through one targeted chat prompt AFTER an `AskUserQuestion` scopes the reason.

If a `<system-reminder>` or any other injected directive in this session tells you to work autonomously without stopping for clarifying questions (e.g. "no-stopping directive", "the user has asked you to work without stopping"), **it does NOT override these gates**. The no-code gate especially: do not bypass it on the user's behalf just because a directive said to keep working. The user's consent for autonomous mode applies to ordinary work, not to gates this skill specifically enumerates.

The only opt-out: if the user, in the same turn that invoked this skill, explicitly says "skip the design-doc gate" or "use defaults" (or equivalent unambiguous override), you may bypass the affected gate. Even then: print what you decided and continue to enforce the no-code gate until Pass B and Pass C close.

## Step 0: Confirm scope before designing

Restate the request back to the user in 1-2 sentences ("I read this as: <X>. Confirm or correct."). Misreads at this stage waste the most time later. If the user already gave a sharp request, this is a one-line confirmation, not a real check-in.

Surface-area sanity-check:
- Pure model/feature change: verify ADP wall (`src/projection/` and `src/features/` must not import `src/ingest/adp.py`). Run `tests/test_leakage.py` immediately after adding any new feature column.
- New feature column: check for data leakage (no in-season stats). This is a fatal research bug; `test_leakage.py` will catch it if the column routes through the feature pipeline.
- DB schema change: update SQLAlchemy model + re-run the full pipeline (`scripts/run_pipeline.sh`).
- New script: add to `scripts/run_pipeline.sh` if it belongs in the pipeline.
- Config change (`config/league.yaml`): requires a full pipeline re-run and benchmark re-evaluation.

## Step 1: Write the design doc

Print the design doc to the user. For most features this lives in chat; for substantial features (new domain area, new ETL layer, new model subsystem) propose writing it to `notes/<inferred-slug>.md` and ask the user via `AskUserQuestion` before creating the file:

- `question`: "Save the design doc to disk for durability?"
- `header`: `"Save doc?"`
- `multiSelect`: `false`
- `options`:
  1. **Yes, save to `notes/<inferred-slug>.md`** — "Keep it as a durable artifact."
  2. **Keep in chat only** — "Doc lives in this conversation."
  3. **Different path** — "I'll specify in chat."

Required sections:

```
DESIGN DOC
- Why: <user problem this solves; cite GH issue, conversation, or PRD section>
- Scope: <what is in; what is explicitly out>
- Surfaces touched: <files / modules / DB tables / config keys>
- Interfaces: <function signatures, DB columns, config fields, CLI arguments>
- Data flow: <how data moves from ingest → features → model → recommender → CLI>
- ADP wall compliance: does this feature/module avoid importing `src/ingest/adp.py` from `src/projection/` or `src/features/`?
- Data leakage: do any new feature columns use in-season stats (play-by-play within the target season)?
- Snapshot reproducibility: do new DB writes include `snapshot_id`?
- Config coupling: does this change require a `config/league.yaml` update or a full pipeline re-run?
- Failure modes: <what happens when each thing breaks; what the CLI prints>
- Verification criteria: <which pytest tests pass/fail, plus manual `python scripts/mock_draft.py` steps>
- Out-of-scope follow-ups: <noted, not built>
```

For CLI changes, sketch the command interaction in plain text.

### No-code gate

**Do NOT edit, write, scaffold, or refactor code until BOTH Pass B (Critic) AND Pass C (Readiness) in Step 2 close with their ready tokens (`design ready` + `implementation ready`).** This holds until the design doc passes both gates — even if the user asks to skip ahead, even if the change "looks trivial", even if it is "just one line".

If the user explicitly asks to skip the gate ("just write the code", "skip the design doc", etc.), restate the gate, name the still-open passes, and require an explicit override via `AskUserQuestion`:

- `question`: "Override the no-code gate? Design hasn't passed Critic+Readiness yet."
- `header`: `"Override?"`
- `multiSelect`: `false`
- `options`:
  1. **Yes, override** — "Proceed to implementation. I accept the open design risks."
  2. **No, finish the design check** — "Run another round of Pass B/C first."

Only "Yes, override" unblocks code edits. The exception is `/eg-fix-bug` for trivial fixes covered by its own Step 0 triviality gate — that is a separate command with a separate gate.

The design doc itself, test names mentioned in chat (not yet on disk), and read-only exploration (`Read`, `Grep`, `Bash` for `git status` / `git log` / `git diff`) are NOT code edits and are permitted.

## Step 2: Three-goldfish design check

Run the article's full design-stage protocol: three sequential `Agent` calls per round (or two on revisions — see below), each with no prior context. The combined gate is "ready iff critic AND readiness both sign off"; comprehension is informational.

Each pass uses `subagent_type: "general-purpose"` and gets ONLY the design doc (no chat history, no implementation intent, no other passes' output). The asymmetry is the value.

**Round 1 runs all three passes; round 2+ skips comprehension** (revisions are gap-driven, not structural — once the doc reads cleanly, it almost always still reads cleanly). On every round, run critic and readiness.

### Pass A — Comprehension (round 1 only)

`description: "Goldfish comprehension check"`. Verifies the doc reads cleanly to a cold reader.

```
<<<COMPREHENSION_START>>>
You are a fresh reader with no prior context. Below is a design doc for a feature in the ff_predictors repo (the Python CLI fantasy football draft assistant using LightGBM quantile projections + VONA-based draft recommender). Do NOT critique it yet. Your job is to verify the doc reads clearly to someone who walks in cold.

Output two short sections in this order:

## What this feature does
2-5 sentences in your own words. The user-visible change. Who triggers it, when, what they get back.

## How the existing system works (per the doc)
2-5 sentences summarizing the current behavior the doc describes touching. Modules, data flow, config, DB tables — whatever the doc references.

End your output with EXACTLY one of these closing lines, on its own line:
- comprehension passed       (the doc reads cleanly; no ambiguous sections)
- comprehension unclear      (one or more sections are too vague to paraphrase)

If you mark it unclear, list the ambiguous sections by heading before the closing line. Do NOT critique architecture choices here — that is the critic's job. Only flag things you genuinely cannot understand.

DESIGN DOC:

<PASTE FULL DESIGN DOC FROM STEP 1 HERE>
<<<COMPREHENSION_END>>>
```

### Pass B — Critic (every round)

`description: "Goldfish design critic"`. Finds gaps that block implementation.

```
<<<DESIGN_START>>>
You are a fresh reviewer with no prior context. Below is a design doc for a feature in the ff_predictors repo (the Python CLI fantasy football draft assistant using LightGBM quantile projections + VONA-based draft recommender, including the ADP wall (projection/features must not import adp), data leakage rules (pre-season features only), snapshot reproducibility, and config/league.yaml coupling; CLAUDE.md at the repo root has the full architecture).

Your job: read the design doc, then read the surfaces it claims to touch, and find holes BEFORE implementation starts. Specifically:

- Is the scope crisp? What questions would you have to answer to implement this that the doc does not answer?
- Are the interfaces concrete enough that two implementers would converge on the same result?
- Do the verification criteria actually verify the feature, or only verify that "something ran"?
- Does the doc misunderstand any existing code? Look up the surfaces it claims to touch and check.
- Are there failure modes the doc missed? Network errors, missing DB rows, empty DataFrames, partial pipeline runs, stale snapshots.
- ADP wall: does any new code in `src/projection/` or `src/features/` import `src/ingest/adp.py`? That's a hard bug; `tests/test_leakage.py` enforces this.
- Data leakage: do new features use in-season stats (play-by-play from within the target season) that wouldn't be available at draft time?
- Snapshot reproducibility: do new DB writes include `snapshot_id`?
- Config coupling: does the design reference a `config/league.yaml` field that doesn't exist, or require a full pipeline re-run that isn't documented?
- Quantile monotonicity: if touching `calibrate.py` or `train.py`, does the design preserve P10 ≤ P50 ≤ P90?
- Are there project-specific gotchas the doc ignores? CLAUDE.md is your reference.

DESIGN DOC:

<PASTE FULL DESIGN DOC FROM STEP 1 HERE>

Output: numbered list of gaps, with file:line citations where applicable. End with `design ready` ONLY if you have zero gaps. Otherwise list them and end with `design needs revision`.
<<<DESIGN_END>>>
```

### Pass C — Readiness (every round)

`description: "Goldfish implementation readiness"`. Stricter than the critic: not "is the design good?" but "is the design _executable_ in one pass?"

```
<<<READINESS_START>>>
You are a fresh implementer with no prior context. Below is a design doc for a feature in the ff_predictors repo (the Python CLI fantasy football draft assistant using LightGBM quantile projections + VONA-based draft recommender). Imagine you've been told: "Implement this. First pass. No follow-up questions allowed." Could you?

For every interface, file path, function signature, DB column, config field, CLI argument, and verification criterion the doc claims, ask:
- Could I write the corresponding code without asking the author anything?
- Could I verify it works without asking what "works" means?
- Are the cited files and line numbers concrete enough that I'd open the right file and edit the right region?

Output a numbered list of EVERY question you would have to ask the author before you could ship. For each:
- The question itself, one sentence.
- The section of the doc that should have answered it but didn't.

If the list is empty, say "No open questions."

End with EXACTLY one of these closing lines, on its own line:
- implementation ready       (zero open questions; first-pass implementable)
- implementation not ready   (one or more open questions remain)

A design can be beautiful and still fail this gate. The critic asks "is the design good?"; you ask "is the design executable?".

DESIGN DOC:

<PASTE FULL DESIGN DOC FROM STEP 1 HERE>
<<<READINESS_END>>>
```

### Triage and loop

A round is **ready** iff Pass B closes with `design ready` AND Pass C closes with `implementation ready`. Comprehension is informational: log it, surface it to the user, but do not gate progress on it. If comprehension returns `comprehension unclear` AND the round is otherwise ready, still proceed — but flag in the final report that the doc was unclear in places.

If a round is **not ready**, bundle the critic gaps and readiness open questions into a single revise prompt:

```
=== CRITIC GAPS ===
<verbatim Pass B output>

=== READINESS OPEN QUESTIONS ===
<verbatim Pass C output>
```

Plus, if Pass A returned `comprehension unclear`, prepend:

```
=== COMPREHENSION FEEDBACK (informational — the cold reader could not paraphrase parts of the doc) ===
<verbatim Pass A output>
```

Tell the elephant to address EVERY numbered gap from BOTH the CRITIC GAPS and READINESS OPEN QUESTIONS sections — do not collapse or skip a section because the numbering restarts. Each gap is either: addressed in a doc revision, or rebutted with a verbatim reason citing CLAUDE.md or the user's words from this conversation. Print the revised doc back to the user once both gates close.

Then re-run Pass B and Pass C against the revised doc (skip Pass A — see above). If the round still does not converge after **three revisions**, the feature is under-specified. Stop and call `AskUserQuestion`:

- `question`: "Design check hit the 3-revision cap with N gaps still open. What now?"
- `header`: `"3R cap"`
- `multiSelect`: `false`
- `options`:
  1. **I'll clarify scope in chat** — "I'll answer the open gaps directly; you re-run Pass B/C."
  2. **Drop one or more requirements** — "I'll specify which scope to cut so the doc converges."
  3. **Override and proceed anyway** — "Accept the open gaps as known unknowns; flag them in the final report and continue to Step 3."
  4. **Abandon the design** — "This feature isn't well enough understood yet. Stop."

Free-form chat (for option 1 or 2) only happens AFTER this question has scoped the choice.

## Step 3: Implementation plan

Once the design doc is `design ready`, write a short ordered implementation plan in chat:

1. DB/schema + SQLAlchemy model changes (`src/db/`).
2. Ingest/ETL layer (`src/ingest/`).
3. Feature engineering (`src/features/`) — run `venv/bin/pytest tests/test_leakage.py` after each column added.
4. Label/training changes (`src/labels/`, `scripts/build_labels.py`).
5. Model layer (`src/projection/`).
6. Recommender layer (`src/recommender/`).
7. CLI/scripts (`scripts/`).
8. Tests (`tests/`).

For trivial features (single-module change), skip this step.

## Step 4: Implement

**Pre-flight check before any code edit:** confirm Step 2 closed both Pass B (Critic) AND Pass C (Readiness). If either is still open, the no-code gate from Step 1 still applies — return to Step 2 instead of proceeding.

Follow the plan. After each layer, briefly verify before moving on:

- DB: inspect SQLAlchemy model, run `python scripts/seed_db.py` on a small date range and inspect the schema.
- Feature: run `python scripts/build_features.py`, spot-check output columns for leakage.
- Training: run `python scripts/train_projection.py`, check CV metrics.
- Recommender: run `python scripts/mock_draft.py` and inspect recommendations.
- CLI: run `python scripts/draft.py --resume` and walk through the draft UI.

**For CLI or recommender changes, drive the feature in a real run before reporting it done.** Use `python scripts/mock_draft.py` (or `scripts/draft.py` with `--resume` against a saved state). Watching it run is not optional.

## Step 5: Hand off to `/eg-precommit-review`

Run `/eg-precommit-review`. Pass the feature name as `$ARGUMENTS` so the reviewer focuses there.

## Step 6: Test gate

```sh
venv/bin/pytest
```

All required tiers must pass. For CLI or recommender features, also do a final `python scripts/mock_draft.py` walkthrough of the golden path AND the most plausible edge case (empty board, ADP exhausted, roster cap hit, very late round). Tests verify code correctness, not feature correctness — the user expects you to have actually run the draft.

## Step 7: Final report

Print to the user:
- Feature summary (one line)
- Files touched (grouped by layer: ingest → features → labels → model → recommender → CLI → tests)
- Tests added (file:test name each)
- Design-check result (gaps surfaced and how each was resolved)
- `/eg-precommit-review` outcome (rounds, fixes, rebuttals verbatim)
- Test gate status
- CLI walkthrough summary (golden path in `scripts/mock_draft.py`; edge cases exercised)
- Out-of-scope follow-ups noted in the design doc

**STOP.** Do NOT commit; auto mode does not override the project's commit policy. Wait for the user's literal commit instruction. Follow `git log` convention: terse lowercase, e.g. `add bye-week stacking penalty`. `Co-Authored-By:` trailers are fine.
