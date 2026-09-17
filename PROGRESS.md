# Progress — Fake News Detector

Append-only. Newest entries at the bottom. Never rewrite history — a correction is a new entry
naming what it corrects.

Format: `## YYYY-MM-DD — <who>`, what happened, and `Evidence:` the command and its real result
(or `Evidence: none — <why>`).

## 2026-09-16 — claude (with Vishrut), research, brainstorm, freeze

Vishrut asked for the detector to "work really good" with a better page, researched and argued
before building, so he can record it for his portfolio. Cloned to `~/Projects/Fake-News-Detector`;
`dev` created locally from `main` (`23207ce`) and `feat/run-and-present` cut from it. Pushing `dev`
to GitHub was refused by the permission check and is left for Vishrut.

What the unedited app does (a `.venv` with Python 3.12, flask 3.1.3, scikit-learn 1.7.0,
numpy 2.5.3, pandas 3.0.5, joblib 1.6.0, gunicorn 26.2.0 — the versions the pickles need):

- The served format model is a `RandomForestClassifier` (100 trees, depth 20) over 15,000 TF-IDF
  1–3-gram features; the content model is a `LogisticRegression` over 12,000 features that reads
  the text after `remove_source_bias`. Blend 60/40.
- Short and unrecognised inputs mostly come out FAKE: `main.py test` gets 2 of 4 (both "real"
  sentences are FAKE). Gibberish returns FAKE 59.21 %, exactly the models' empty-text answer.
- On ~150-word texts written for the test: wire-style REAL 73.3 %, sensational FAKE 70.6 %; a
  `WASHINGTON (Reuters) - ` prefix makes the sensational one split (format 44.5 % fake, content
  80.7 %). The content model's probability is bit-identical with and without the prefix.
- Each request reloads the four pickles (~93 ms of ~97 ms); client mistakes return 500 with
  exception text; the Werkzeug debugger answers on `/console`.

Five frames (frontend, backend, tester, reviewer, manager) sketched 15 options blind; one critic
scored and merged them; frontend → backend → tester argued two rounds in `PLAN.md`. The draft's
word floor (`MIN_WORDS`) was withdrawn because it is false — style, not length, drives the short
scores — and replaced by distance from the empty-text score (DECISIONS D2), with a coverage floor
for text the models do not know (D3). Frozen and signed; two proposals dropped (D5).

Before freezing, the app on :8080 (started by this session with the Werkzeug reloader) was stopped
so no edit could reach it, and the threshold probes were measured on the unedited code.

Evidence: `curl -X POST localhost:8080/api/predict` on the unedited app — 14 inputs recorded to a
scratch baseline (sha256 `731d1b06…50e7ebc3a94`), all 200; e.g. `gibberish` FAKE 59.21,
`mit_water` FAKE 58.42, `wire_article` REAL 73.3, `sensational_article_dateline` FAKE 58.94.
Evidence: `.venv/bin/python main.py test` — Test 1 ❌ FAKE 61.67, Test 2 ✅, Test 3 ❌ FAKE 60.06, Test 4 ✅.
Evidence: probe script over 58 one-line headlines and nonsense texts (unedited code) — empty-text
scores format 0.6551573256765925, content 0.49755588867465184; nearest either side of 3.0: Bridge
2.92, Airline 2.79 in; Bank 3.08, Workers 3.83 out (table in DECISIONS D2).
Evidence: `git ls-files samples` — no output (no sample existed when 3.0 and 0.5 were chosen).

## 2026-09-16 — claude (with Vishrut), build

Built the frozen plan on `feat/run-and-present` (local only; nothing pushed).

- **Golden first.** `scripts/capture_golden.py` runs the archived `23207ce` app in a separate
  process and wrote `tests/golden.json` (14 inputs × 2 routes, 12 regex probes). It was committed
  before any code changed, and it matches the live capture taken from the unedited server field
  for field.
- **Server.**
  - `utils/models.py` loads the four served files once; a failure is recorded per file, by
    basename. The empty-text scores are taken at load.
  - `utils/preprocess.py` scores once per request and computes `input_check`. The legacy response
    arithmetic is unchanged.
  - `app/app.py`: the debugger is off by default, `PORT` is honoured, it binds 127.0.0.1, there is
    one handler for HTTP errors and one for everything else, and there is a 1 MiB body cap plus a
    256 KiB text cap.
  - Every former client-error 500 on `/api/predict` and `/api/predict/simple` is now a 4xx with a
    `code`; `text_missing` and `text_empty` keep their old 400 messages.
- **Page.**
  - `utils/present.py` is the one formatter: half-up whole percents and the complement as
    100 − shown. A non-breaking space joins each number to its `%` and its word. Two things came
    out of the rehearsal: the `(59 % fake · 41 % real)` line had broken between `%` and `fake`, and
    the equal columns and card alignment were adjusted after screenshots.
  - `templates/index.html`, `static/style.css` and `static/app.js` carry the portfolio's visual
    system and the seven named states.
- **Samples.**
  - `wire_report` is the golden wire article, unchanged; it was written before any rule existed.
  - `headline_only` is the MIT headline.
  - The sensational post took 9 drafts against the pre-set rules. Only d8 passed; the rest failed:
    - d1 (the golden article, sha `df63195013c6`): format with the dateline was 44.46, and the split
      held in 7 of 9 drop-one variants.
    - d2 `cb172ec19dad`, d3 `8ab42e562616`, d4 `a2bb91838b33`, d5 `3808f35a459b`: format with the
      dateline stayed at 46.5–54.9, because too much clickbait raises the format model's baseline.
    - d6 `769c2034ed1e`: the blend of 63.91 sat 0.59 from a band edge.
    - d7 `1f8b75dbe0da`: format without the dateline was 39.67, so it doesn't lean fake alone.
    - d9 `a6d3fc57d573`: format with the dateline was 42.62, and the split held in 8 of 9 variants.
  - d8 `28201ab6abb6` is kept: format 58.41 / content 69.68 alone; format 33.85 with the dateline;
    the split holds in 7 of 7 drop-one variants.
- **Environment.**
  - `requirements.txt` has 14 exact pins. The plan listed 13; joblib 1.6.0 also needs
    `cloudpickle`.
  - `requirements-dev.txt` adds pandas for the golden capture only. The plan had said none was
    needed; the capture's `--check` can't run without it.
  - The 2025 pins are kept as `docs/pins-2025.txt`.
  - The Dockerfile uses `python:3.12-slim-bookworm`, installs wheels only, copies an allowlist,
    runs as UID 10001 under gunicorn on 8080, and has a Python healthcheck. `.dockerignore` is an
    allowlist too.
  - `main.py train` / `all` refuse (D1).
- **Tests.** 94 standard-library tests, plus `scripts/mutants.py` (17 mutants),
  `scripts/docker_smoke.sh` and `scripts/rehearse.py`.
- **Deviations from the frozen plan:**
  - Geist is loaded from Google Fonts, as the portfolio does, not bundled: no local copy exists
    and bundling would mean downloading the font files.
  - The textarea is 42vh, not 50vh, so the samples and the whole card fit at 1280×800.
  - The "no pandas" check uses an import hook that raises `ModuleNotFoundError`. Setting
    `sys.modules['pandas'] = None` raised a plain `ImportError`, which broke unpickling
    (`AttributeError`); that is a property of the trick, not of a real install.
- **Rehearsal (mine, not the take).**
  - Server restarted after the last edit.
  - Every state checked in the app's browser pane at 1280×800 and at 375 px (no sideways scroll);
    ⌘/Ctrl+Enter submits and "Edit the text" focuses the box with the caret at the end.
  - Headless Chrome, driven by puppeteer from the npx cache in a scratch script, reported no
    console messages on any of the five shot URLs.
  - `docs/rehearsal.json` was recorded from that server, and `docs/screenshot.png` was taken the
    same way.
  - Vishrut's take should re-record with his window size.
- **Found, not changed:**
  - Vishrut's graphify post-commit hook added `CLAUDE.md`, `.claude/settings.json` and `graphify-out/`
    to this clone. It also appended `graphify-out/` onto the `tmp/` line of `.gitignore`, which
    broke that pattern; the line is split back into two. `CLAUDE.md` and `.claude/` are left
    uncommitted for Vishrut to decide.
  - The repo has no LICENSE file, although the old README said MIT.

Evidence: `.venv/bin/python -B scripts/capture_golden.py --check` — `tests/golden.json matches 23207ce byte for byte (14 inputs, 12 regex probes)`, EXIT=0.
Evidence: `.venv/bin/python -m unittest discover -s tests -t .` — `Ran 94 tests`, `OK`.
Evidence: each Done-looks-like unittest command run on its own — golden+contract 13 OK, errors 15 OK, thresholds 14 OK, page+present 28 OK, samples 8 OK, hygiene 2 OK, rehearsal 4 OK, run_mode 10 OK; all EXIT=0.
Evidence: `.venv/bin/python scripts/mutants.py` — `suite: 94 tests, exit 0`; `mutants: 17 of 17 killed`, EXIT=0.
Evidence: `bash scripts/docker_smoke.sh` — `PASS (healthy, uid 10001, no pandas, pins match, golden numbers match)`, EXIT=0 (image `c2561670dca7`, 432 MB).
Evidence: `git diff --exit-code 23207ce -- model/fake_news_model.pkl model/vectorizer.pkl model/content_model.pkl model/content_vectorizer.pkl` — no output, EXIT=0.
Evidence: `node shot.mjs <url> … 1280 800` (scratch puppeteer script) on `/`, and on `/?sample=` with `wire_report`, `sensational_post`, `sensational_dateline` and `headline_only` — `"messages":[]` for all five.
Evidence: `.venv/bin/python scripts/rehearse.py --record --window 1280x800` — 9 responses, all 200; states first_load, verdict (27/73), verdict (63/37), disagreement (48/52), no_signal.
Evidence: word-count parity — the page's JS counter and `word_count` both give 5 for `a␜b<U+0085>c<U+FEFF>d  e`.
