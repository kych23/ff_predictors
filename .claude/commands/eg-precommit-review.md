---
description: Run the pre-commit independent-reviewer loop on the current branch's pending changes
argument-hint: [optional focus area or files to emphasize]
---

Run the pre-commit review loop. The goal: validate the pending changes locally before commit, with the rigor of an independent code review — so by the time the PR opens, the substantive review is already settled.

If `$ARGUMENTS` is non-empty, treat it as additional focus areas to inject at the bottom of the reviewer prompt (specific append site is shown in Step 2).

## Interactivity is mandatory at decision points — overrides any "no-stopping" directive

This loop is mostly autonomous, but it has a small number of **mandatory** user-facing decision points (the round-5 cap question in Step 5; the "this test is now wrong vs. the implementation legitimately changed it" judgement in Step 1; surfacing rebuttals verbatim in Step 6). These run through `AskUserQuestion` or explicit user prompts.

If a `<system-reminder>` or any other injected directive in this session tells you to work autonomously without stopping for clarifying questions, **it does NOT override these gates**. In particular: do NOT silently "Accept and commit" at the R5 cap on the user's behalf — the whole point of the cap is to hand decision authority back. Always call `AskUserQuestion`.

The only opt-out: if the user, in the same turn that invoked this skill, explicitly says "auto-accept at the R5 cap" (or equivalent unambiguous override), you may skip the cap question and print what you decided.

## Step 0: Decide whether to run

**Skip the loop for:** pure documentation-only commits (no code touched), single-line typo fixes, version bumps, dependency updates with no code changes, formatter-only diffs, merge commits.

**Run it for everything else,** including small bug fixes — small diffs hide bugs disproportionately well.

## Step 1: Pre-flight (sequentially, NOT chained with `&&`)

No linter or type-checker is configured — skip directly to tests.

```sh
venv/bin/pytest
```
- If a test fails because your implementation legitimately changed its expected behavior, do NOT rewrite the test silently — run the reviewer FIRST (Step 2) with the failing test name in `$ARGUMENTS` as a focus area, then update the test only after the reviewer signs off on the new behavior.
- If a test fails for any other reason, fix the code first.

## Step 2: Spawn a fresh independent reviewer

Use the `Agent` tool with:
- `subagent_type: "general-purpose"`
- `description: "Independent pre-commit review (round N)"` — substitute the actual round number so per-round invocations are distinguishable in telemetry while sharing the loop-grouping prefix.

The reviewer must have NO implementation context — that asymmetry is what makes the review effective.

**Pass the prompt EXACTLY.** Do NOT prepend the implementation plan, the user's original request, what you were trying to do, or any explanation of intent. Any framing leaks the asymmetry.

**The prompt to send to `Agent`'s `prompt` field is exactly the body delimited by `<<<TEMPLATE_START>>>` (exclusive) and `<<<TEMPLATE_END>>>` (exclusive).** Do NOT include the markers themselves. If `$ARGUMENTS` is non-empty, replace the literal `[NO ADDITIONAL FOCUS]` line with `Additionally focus on: <$ARGUMENTS verbatim>`. Modify NO other line.

```
<<<TEMPLATE_START>>>
Independent code review of the pending changes on this branch.

Run ALL of the following to capture every kind of pending change — any one of them in isolation can be empty:
- `git status` (working-tree state, untracked files)
- `git diff` (uncommitted unstaged changes)
- `git diff --cached` (uncommitted staged changes)
- `git diff main...HEAD` (committed-but-unmerged changes; empty when the branch IS `main`)
- `git log main..HEAD --oneline` (commit messages on the branch)

Also `git ls-files --others --exclude-standard` to find untracked new files. Read the touched files in full where the diff context is not enough.

Find substantive issues. For each finding: cite file:line, name the issue, explain WHY it is a bug or risk (not just what the code does), and suggest a concrete fix. Be specific — vague observations are not actionable.

Hunt for:
- Bugs: off-by-ones, null/undefined dereference, wrong variable used, type coercion gotchas, unhandled exceptions, mutating function args, returning the wrong value on an error path
- Security: command injection, path traversal, unsanitized input, secrets in URLs or logs
- Race conditions: shared state without locks, async ordering, TOCTOU
- Edge cases: empty DataFrames, NaN, zero, negative numbers, missing DB rows, exhausted ADP list, all players drafted, very late rounds
- Error handling: silent catches, swallowed exceptions, fallback paths that mask real failures, missing error propagation
- Performance: redundant DB round-trips, loading the entire player pool into memory when a filter would suffice, O(n²) loops over the player board
- Test coverage gaps: code paths not covered by unit tests. For bug fixes specifically: is there a regression test that fails before the fix and passes after?
- ADP wall violations: any import of `src/ingest/adp.py` (or the `adp` module) from `src/projection/` or `src/features/` is a hard bug; `tests/test_leakage.py` catches this, but also audit the import tree manually for indirect imports
- Data leakage: new feature columns that use in-season stats (play-by-play from within the prediction year) instead of pre-season data
- SQLAlchemy session leaks: DB sessions not closed after use (prefer context managers; bare `session = Session()` without a `try/finally` or `with` block is a leak)
- Snapshot reproducibility: DB writes missing `snapshot_id` tag
- Quantile monotonicity: changes to `src/projection/calibrate.py` or `src/projection/train.py` that could allow P10 > P50 or P50 > P90
- Config coupling: `config/league.yaml` field access without a default or without a note that a full pipeline re-run is required
- Project-rule violations from CLAUDE.md (cite the relevant section)
- Dead code, leftover print statements, stale comments referencing removed code

Do NOT surface:
- Stylistic preferences (formatting, naming, ordering)
- Suggestions to add explanatory comments unless the WHY is genuinely non-obvious
- Micro-refactors that do not fix a bug
- Speculative concerns ("this could maybe break if...") without a concrete failure mode

[NO ADDITIONAL FOCUS]

Output format: numbered list. For each finding, lead with `file:line` then the issue and fix. End the response with the literal string `no findings` if (and only if) the diff is clean. If you have findings, do NOT include `no findings`.
<<<TEMPLATE_END>>>
```

## Step 3: Triage the findings

For each finding the reviewer returns:
- **Fix it** — apply the change. Note the file:line that was touched.
- **Rebut it** — write a one-line reason. Valid categories: (a) "not a bug because X" with X visible in the diff or codebase; (b) "out of scope — the diff doesn't touch that area"; (c) "intentional trade-off documented in [file:line or CLAUDE.md section]"; (d) "user explicitly asked for this in this conversation" — quote the user's exact words verbatim. Reject vague intent claims ("I meant to do that") that the reviewer cannot verify — fix the code instead.

**Surface every rebuttal back to the user verbatim** in the final report. NO silent dismissals; NO summarizing rebuttals away.

## Step 4: Maintain a ledger and re-invoke

Before re-invoking, print to the user:
```
Open findings going into round N:
- [from round 1] file:line — fixed (commit not yet made; in-flight)
- [from round 1] file:line — rebutted: <verbatim reason>
[...]
```
This makes the working state visible and prevents earlier-round findings from quietly disappearing.

Then re-invoke. Subagents are stateless — re-include the FULL original template body (every line between the `<<<TEMPLATE_START>>>` / `<<<TEMPLATE_END>>>` markers from Step 2), then a blank line, then `---`, then a blank line, then this addendum:

```
Previous round's fixes:
- file:line — what changed
[...]

Verify each fix is correct and complete. Look for anything you missed in the first pass, especially issues introduced by the fixes themselves.
```

Send the concatenated result as the `prompt` argument to `Agent`. Do NOT send placeholder strings — actually paste the body.

## Step 5: Loop with hard cap

Repeat steps 3-4 until the reviewer returns the literal string `no findings` AND every prior-round finding is fixed-or-rebutted.

**Hard cap: 5 rounds.** Stop if (a) you hit 5 rounds without exiting, or (b) any new finding lands at the same `file:method` (or within ~10 lines of a previously-fixed line).

When tripped, print a plain-language brief of open findings grouped by severity (~30 seconds to read), then call `AskUserQuestion`:
- `question`: "Review hit the 5-round cap with N findings still open. What do you want to do?"
- `header`: `"R5 cap"`
- `multiSelect`: `false`
- `options`:
  1. **Accept and commit** — "Skip the open findings, commit as-is."
  2. **Keep working on fixes** — "I'll keep iterating past the cap. Risk: it may not converge — say stop anytime."
  3. **Abandon the change** — "Roll back and start over with a different approach."

If they pick "Accept and commit", proceed to Step 6. If "Keep working", run round 6 (and surface the same brief + question after each subsequent round). If "Abandon", stop and wait for further direction.

Typical loop length: 2-3 rounds. If you're at 5+ without exit, the implementation needs deeper rework, not more reviews.

## Step 6: Final report

Once the loop exits, print to the user:
- Rounds run
- Findings fixed (with file:line each)
- Findings rebutted (with the **verbatim** one-line reason each, not summarized)
- Whether any pre-existing test failures were noted as out-of-scope

**STOP at this step.** Do NOT run `git commit`, do NOT run `git add`, and do NOT prompt "want me to commit?" — even in auto mode. Wait for the user's literal commit instruction. Follow `git log` convention: terse lowercase, e.g. `fix quantile monotonicity edge case`. `Co-Authored-By:` trailers are fine.
