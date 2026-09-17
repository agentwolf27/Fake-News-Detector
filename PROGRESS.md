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
