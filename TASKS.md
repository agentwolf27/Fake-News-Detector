# Tasks — Fake News Detector

Live task list. Every task has an owner and a VERIFY condition. A task is done when the
VERIFY passes, not when someone says it is.

**VERIFY is a command, not a verdict.** Write what should be run, in backticks: `` `npm test` exits 0 ``.
`/verify` runs the first backticked command and reports what happened. If a task genuinely
cannot be checked by a command, say so in plain words with **no backticks** — *reviewed by eye* —
so it is counted as uncheckable rather than quietly counted as passed.

## In progress

| # | Task | Owner | VERIFY |
|---|------|-------|--------|
| 7 | Record the walkthrough | user | Vishrut records it |

## Queued

| # | Task | Owner | VERIFY |
|---|------|-------|--------|
| 8 | Save what the live server served during the take | user | `.venv/bin/python -m unittest tests.test_rehearsal` exits 0 after `scripts/rehearse.py --record` |
| 13 | Push `dev` and the branch, open the PR into `dev` | user | Vishrut approves the push |
| 14 | Portfolio case file and shot list follow what landed | claude | `grep -c "Too little to go on" "/Users/vish/Projects/Portfolio Website/design-system/WALKTHROUGHS.md"` prints a non-zero count |
| 15 | Decide on CLAUDE.md, .claude/ and a LICENSE file | user | Vishrut decides |

## Done

| # | Task | Owner | VERIFY | Completed |
|---|------|-------|--------|-----------|
| 1 | Capture the golden bodies from 23207ce before any code edit | claude | `.venv/bin/python -B scripts/capture_golden.py --check` exits 0 | 2026-09-16 |
| 2 | Server: load once, input_check, error handlers, caps, health and models | claude | `.venv/bin/python -m unittest tests.test_golden tests.test_api_contract tests.test_errors tests.test_thresholds` exits 0 | 2026-09-16 |
| 3 | Demo samples, each held to its on-screen state | claude | `.venv/bin/python -m unittest tests.test_samples` exits 0 | 2026-09-16 |
| 4 | Page: one formatter, named states, new layout | claude | `.venv/bin/python -m unittest tests.test_page tests.test_present` exits 0 | 2026-09-16 |
| 5 | No skipped or empty test modules | claude | `.venv/bin/python -m unittest tests.test_hygiene` exits 0 | 2026-09-16 |
| 6 | Rehearse every shot in a headed browser at the recording size | claude | screenshots of every state, looked at by eye | 2026-09-16 |
| 9 | Pinned requirements, run modes from any directory | claude | `.venv/bin/python -m unittest tests.test_run_mode` exits 0 | 2026-09-16 |
| 10 | Docker image that starts and answers | claude | `bash scripts/docker_smoke.sh` exits 0 | 2026-09-16 |
| 11 | Mutants each fail their module | claude | `.venv/bin/python scripts/mutants.py` exits 0 | 2026-09-16 |
| 12 | README matches what runs | claude | `grep -q "docker run --rm -p 8080:8080" README.md` exits 0 | 2026-09-16 |
