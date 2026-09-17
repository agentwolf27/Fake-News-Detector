# Fake News Detector

<!-- pulse -->

## What we are building
A small Flask service and web page that scores whether a pasted news article's writing style
resembles the fake or the real articles its two committed models were trained on (a TF-IDF random
forest, the "format model", and a TF-IDF logistic regression that reads the text with agency names
and datelines removed, the "content model", blended 60/40). This round makes it start from a fresh
clone — locally and in Docker — makes its page say only what the models can support, and gives it
committed demo texts that tests hold to the states the page shows, so Vishrut can record a
45-second walkthrough for his portfolio.

## Why
As of `23207ce` (2025-06-11): a clean `pip install -r requirements.txt` does not start the app
(Flask 2.0.1 against Werkzeug 3; numpy 1.21.2 has no wheel for Python ≥ 3.12; the pickles need
scikit-learn 1.7.0 and NumPy 2). The Docker image builds but exits on import, listens on 8080
while exposing and health-checking 5000 with a `curl` it lacks, and "trains" on a dataset that is
not there. The page headlines "Likely Fake News" on a 52/48 split, shows `VERY_LOW` and two-decimal
percents, and gives unrecognised input a confident FAKE (gibberish → FAKE 59.21 %, which is exactly
the models' empty-text answer). Client mistakes come back as 500s with Python exception text; the
Werkzeug debugger is served on 0.0.0.0. The models score writing style, not truth — a wire-style
story about the Moon becoming a US state scored REAL 90 % — and the planned shot list's two example
headlines both come out FAKE on camera. The recording is for a data & ML engineering job search; a
demo that implies fact checking, or crashes mid-take, costs more than no demo.

## Done looks like
A short list of concrete, checkable outcomes. If you cannot check it, it does not belong here.
Tick one only when its VERIFY passes, not when someone says it is done.
Lines marked **[take]** must pass before Vishrut records; **[at take]** is written right after the take.

- [ ] [take] The committed golden bodies re-derive byte for byte from 23207ce, run in isolation — `.venv/bin/python -B scripts/capture_golden.py --check`
- [ ] [take] Existing API fields and the golden bodies are exactly unchanged on both predict routes — `.venv/bin/python -m unittest tests.test_golden tests.test_api_contract`
- [ ] [take] Malformed, non-text and oversized requests get a 4xx with a named code and no Python internals — `.venv/bin/python -m unittest tests.test_errors`
- [ ] [take] The no-signal and coverage rules refuse or admit the pinned probes either side of each edge — `.venv/bin/python -m unittest tests.test_thresholds`
- [ ] [take] Real inputs reach every named state in whole percents with the complement, with no decimal percent, raw enum, emoji or banned word — `.venv/bin/python -m unittest tests.test_page tests.test_present`
- [ ] [take] With all four models loaded, each sample shows its claimed state with margin; the dateline sample is the post plus a prefix and moves only the format model — `.venv/bin/python -m unittest tests.test_samples`
- [ ] [take] No test is skipped and no listed test module is empty — `.venv/bin/python -m unittest tests.test_hygiene`
- [ ] [take] The four served model files are byte-identical to 23207ce — `git diff --exit-code 23207ce -- model/fake_news_model.pkl model/vectorizer.pkl model/content_model.pkl model/content_vectorizer.pkl`
- [ ] [take] Every recorded state was looked at in a headed browser at the recorded inner size, on a server started after the last edit, with a clean console and the result card in view — Vishrut and claude look at the rehearsal screenshots
- [ ] [at take] The pages served during the take are byte-identical to what the current code renders — `.venv/bin/python -m unittest tests.test_rehearsal`
- [ ] The documented commands serve with the debugger off from any directory and without pandas; two pickles give format-only states, none gives 503 — `.venv/bin/python -m unittest tests.test_run_mode`
- [ ] Every listed mutant, including no-signal 3.0→2.0 and 3.0→4.0, fails its module — `.venv/bin/python scripts/mutants.py`
- [ ] A freshly built image turns healthy, runs as non-root, has no pandas and returns the golden numbers — `bash scripts/docker_smoke.sh`
- [ ] The take was watched against the shot list: no narration implying fact checking or that headlines are refused, no zoom, no overlays — Vishrut watches it

## Out of scope
- Retraining, downloading any dataset, new or bigger models (DECISIONS D1).
- Word-level explanations (occlusion or `top_terms`). The cheap, exact first step later is
  content-model coefficient × tf-idf contributions.
- A probe endpoint or a client-rendered page; one server-rendered path only.
- Re-implementing the models without scikit-learn; hash pins, lock files, CI, published images,
  compose.
- A headless-Chrome console gate and a Node word-count parity test (DECISIONS D5).
- Deleting, moving or packaging the unused apps, notebooks and training scripts, and the 92 MB
  `model/advanced/advanced_ensemble.pkl` (DECISIONS D6). History is never rewritten.
- A sealed hand-written holdout set; a hosted public demo.
- The portfolio case file and shot list — a follow-on PR in that repo, quoting only numbers
  measured on this branch.

## Constraints
- Python 3.12 or 3.13: numpy 2.5.3 and scipy 1.18.1 need ≥ 3.12, and scikit-learn 1.7.0 wheels
  stop at 3.13. Runtime deps pinned exactly with `==` from the working `.venv`: flask 3.1.3,
  werkzeug 3.1.8, jinja2 3.1.6, markupsafe 3.0.3, itsdangerous 2.2.0, click 8.5.0, blinker 1.9.0,
  scikit-learn 1.7.0, scipy 1.18.1, numpy 2.5.3, joblib 1.6.0, threadpoolctl 3.7.0,
  gunicorn 26.2.0. No pandas.
- The four served pickles are not modified. Existing routes and response fields stay backward
  compatible; the portfolio cites `/api/predict`'s `model_breakdown` and `/api/models`.
- Tests use `python -m unittest` (pytest is not installed in `.venv`).
- One Python formatter produces every number the page shows; JS never formats a probability.
- Page follows the portfolio's visual system (DECISIONS D4): bg `#0A0A0B`, surfaces `#121214` /
  `#17171A`, text `#E6E6E3` / `#B8B8B6` / `#9A9AA1`, accent amber `#FFB000`, red `#F26D5B` for
  faults only; Geist Mono for machine voice and numbers, Geist for prose; no gradients, emoji or
  glow; contrast ≥ 4.5 : 1, including the placeholder and the busy button.
- Git: feature branch `feat/run-and-present` → PR into `dev`. Creating `dev` on the remote,
  pushing and opening the PR need Vishrut's sign-off.

## Approach (frozen 2026-09-16 — see the freeze line at the end)

### Ordered work
0. **Before any edit** (the app on :8080 was stopped first): `scripts/capture_golden.py` unpacks
   `git archive 23207ce -- app utils templates` plus the four pickles into a temp dir and, in a
   subprocess there (`PYTHONPATH`, `FLASK_DEBUG`, `PORT` scrubbed), records the exact bodies and
   status codes of both predict routes for 14 inputs into `tests/golden.json`; it asserts the
   imported modules are the archive's, every body is a 200 with FAKE or REAL, and no body has
   `input_check`. `--check` re-derives and byte-compares (needs pandas, which only the dev `.venv`
   has; it exits non-zero naming pandas if absent). `test_golden` compares the app with the
   sha256-pinned file. A second file, `tests/golden_probes.json`, holds backend·1's regex probes.
1. **Server** (`utils/preprocess.py`, new `utils/models.py`, `app/app.py`):
   - Load all four files once at import (`FND_MODEL_DIR` overrides the directory, default from
     `__file__`) into an immutable record per file: the object, or the exception class and
     basename. A pair is loaded only if both files load and its empty-text scoring runs; that
     scoring, stored as float64, is the warm-up and the `no_signal` baseline.
   - FAKE/REAL by `classes_`; the forest's label from `predict_proba` (it runs once); raw floats
     inside, two decimals only at the API edge; no pandas.
   - HTML stripping keeps `<[^>]+>` but applies it only up to the last `>` (identical output,
     linear time).
   - `app.py` binds 127.0.0.1, reads `PORT` (default 8080) and `FLASK_DEBUG` (default off); no
     `sys.path.append` of the parent — it inserts the repo root first. `main.py run` execs
     `sys.executable` on the absolute path; `train` / `all` refuse with a message (D1).
   - No try/except in routes: one `HTTPException` handler and one `Exception` handler (JSON under
     `/api/` and `/health`, HTML elsewhere). The 500 body carries a uuid4 `request_id` and nothing
     else; the log carries the traceback, never the text.
   - 1 MiB transport cap (`MAX_CONTENT_LENGTH` and `MAX_FORM_MEMORY_SIZE`), then
     `len(text.encode()) <= 256 KiB` on both paths → 413 `too_large` with `text_bytes`,
     `limit_bytes`.
   - `input_check` computed once on the server (below). `/health` and `/api/models` built from
     what loaded.
2. **Samples** under `samples/` (written for this repo; every scored draft listed in PROGRESS with
   its sha256 and both scores, kept or dropped): `wire_report` (~150 words, a real-leaning verdict,
   both models agree), `sensational_post` (~130 words, a fake-leaning verdict, both agree),
   `sensational_dateline` (exactly `"WASHINGTON (Reuters) - " + sensational_post`, which must show
   `disagreement`), `headline_only` (the MIT headline, `no_signal`). Each file is the JSON body
   `{"text": …}`; `samples/index.json` names the id, label and expected state. If the MIT headline
   misses its margin once committed, shot 5 is cut rather than the sample rewritten.
3. **Page:** `utils/present.py` (the one formatter and the state precedence), template, CSS,
   bundled Geist woff2 (copied from the portfolio repo if present, OFL), ~40 lines of JS.
4. **Rehearse** in a headed browser: stop any server, start `.venv/bin/python app/app.py`, walk
   every shot, screenshot each state, note `innerWidth×innerHeight`, check the console.
   ▶ **Recording point.** Right after the take, `scripts/rehearse.py --record` saves
   `docs/rehearsal.json`.
5. **Environment and Docker:** `requirements.txt` (the pins above; `requirements-dev` not needed);
   the old pins recorded in `docs/requirements-2025.txt` ("not installable; record only");
   Dockerfile (`python:3.12-slim-bookworm`, `pip install --only-binary=:all:`, no compiler, no
   training, an allowlist COPY, fixed-UID user, `gunicorn --bind 0.0.0.0:8080 --workers 2
   --preload app.app:app`, a Python healthcheck with `--start-period=30s --start-interval=1s`);
   `.dockerignore`; `scripts/docker_smoke.sh`; `test_run_mode`; `scripts/mutants.py`.
6. **README:** what runs and what does not, the commands, the port, the measured behaviour,
   accuracy figures labelled "author-reported; not re-runnable without the dataset", the false
   "REAL 92 % / FAKE 88 %" examples removed, and the JSON-header curl lines.
7. **PR into `dev`** (sign-off), naming which 500s on cited routes became 4xx.

### `input_check` (both predict routes)
`{status, word_count, known_features: {format_model, content_model}, coverage: {…},
points_from_empty: {…}}`. `word_count` = `len(text.split())` with U+001C–U+001F, U+0085 and
U+FEFF treated as spaces. `known_features` = non-zero features of that model's transform (int).
`coverage` and `points_from_empty` are 2-dp floats; `null` when the model did not load; `coverage`
also `null` when the vectorizer yields no tokens; never NaN. Status precedence:
1. `url_only` — every whitespace token is a link (`http(s)://…`, `www.…`, or a space-free
   `host.tld/…`).
2. `no_known_words` — every non-null coverage `< COVERAGE_FLOOR` (0.5), or all null (D3).
3. `no_signal` — every loaded model within `NO_SIGNAL_POINTS` (3.0) of its empty-text score, on
   raw floats, `<=` (D2).
4. `ok`.

Status code: 422 only when `clean_text` yields `""` (code `url_only` or `no_known_words`, body
carries `input_check`); everything else that is valid JSON with string text stays 200 with every
existing field.

### Errors (JSON under `/api/`; `error` string kept, `code` added)
400 `bad_json` (malformed, not an object, nested too deep) · 400 `text_missing` · 400
`text_not_string` · 400 `text_empty` (whitespace only) · 415 `json_required` · 413 `too_large` ·
422 `url_only` / `no_known_words` · 503 `model_unavailable` · 404 `not_found` · 405
`method_not_allowed` · 500 `internal_error` + `request_id`. No body contains `Traceback`, a path,
`Werkzeug` or exception text.

### `/health` and `/api/models`
`/health` keeps `status`, `service`, `models.{format_model, content_model}` with values from the
loaded objects, adds `sklearn_version`, and returns 503 when the format pair did not load.
`/api/models` keeps `{available, description}` for `format_model`, `content_model`, `ensemble` and
adds `loaded`, `type`, `n_features`, `ngram_range`, `weight`, `empty_text_fake_prob` (2 dp) and a
top-level `no_signal_points` and `coverage_floor`.

### Page
- **Layout:** two columns at ≥ 960 px (input left, result right, top-aligned, the result column
  reserves its height so switching samples doesn't jump); one column below. Textarea fixed at
  50vh, scrolls inside. `novalidate`; `color-scheme: dark`; inline `html{background:#0A0A0B}`;
  Geist preloaded; amber `:focus-visible` ring; explicit placeholder colour; busy state without
  opacity.
- **Always:** title `Fake News Detector` · lede `Paste a news article. Two text models score
  whether its writing style resembles the fake or the real articles they were trained on.` · scope
  line `Trained on about 1,600 labelled news articles. It scores writing style; it does not check
  facts.` · label `Article text` · placeholder `Paste the full article, not just the headline.` ·
  counter `0 words` / `1 word` / `1,152 words` · button `Analyse` → `Analysing…` (guarded by a
  flag and `aria-disabled`, never `disabled`; reset on `pageshow`) · ⌘/Ctrl+Enter uses
  `requestSubmit()` · samples `Try a text written for this demo:` `Wire-style report` ·
  `Sensational post` · `Sensational post + agency dateline` · `Headline only` — each
  `GET /?sample=<id>`, which fills the box **and scores it** · footer `API: POST /api/predict ·
  GET /api/models · Source on GitHub`. A draft is kept in `sessionStorage` (try/catch) before a
  sample click or a 413, with `Restore your text`.
- **One `[data-state]`** per page: `first_load`, `verdict`, `disagreement`, `no_signal`,
  `url_only`, `unrecognised`, `error` (+ `data-code`); the 404 page is `not_found`.
- **Precedence:** error → url_only → unrecognised → no_signal → disagreement → verdict.
- **first_load:** `No text scored yet` · `Paste an article or load a sample. Scores appear here.`
  No dashes, no 0 %, no empty bars.
- **verdict:** one rounding, half-up, from raw floats; bands read the **shown** blended integer of
  the top class: 50–54 `Too close to call`; 55–64 `Leans fake` / `Leans real`; 65–79 `Reads like
  fake news` / `Reads like real reporting`; ≥ 80 `Reads strongly like fake news` / `Reads strongly
  like real reporting`. The page ignores `confidence_level`. Then a table in fixed fake · real
  order, blended first: `Blended 60/40  38 % fake · 62 % real`; `Format model  …` + `Random forest
  over word patterns, including datelines and agency names.`; `Content model  …` + `Logistic
  regression; agency names and datelines are removed first.` A model leans iff its shown fake is
  ≤ 45 or ≥ 55. Agreement line, one of: `Both models lean fake.` · `Both models lean real.` ·
  `Neither model leans either way.` · `The format model leans fake; the content model has no
  lean.` (and the other three one-lean combinations). A model with zero known features reads `no
  words it recognises` in its row. Content model not loaded: its row reads `not scored — model
  file not loaded` and the blended row reads `Format model only — content model not loaded`.
  Foot: `Read 152 words.` + the scope line.
- **disagreement** (one model leans fake, the other leans real; amber): `The two models disagree` ·
  the table as above with `— leans fake` / `— leans real` on the model rows · `The format model
  reacts to agency datelines and wire-service phrasing; the content model has those removed first.
  A split means the styling and the wording point different ways.`
- **no_signal:** `Too little to go on` · `Both models scored this within 3 points of an empty text
  (59 % fake · 41 % real), so the score says nothing about these words.` (format only: `The format
  model scored this within 3 points of an empty text (66 % fake · 34 % real), …`) · `Edit the text`
  (focuses the textarea, caret at the end, never clears it). The two percents come from the stored
  empty scores via `present`; the input's own score is not shown.
- **unrecognised:** `Too few words the models know` · `Most of this text isn't in the English news
  vocabulary the models learned, so there is little for them to read.` · `Edit the text`. No
  number.
- **url_only:** `Links aren't opened` · `The detector reads text only. Paste the article itself,
  not its address.` · `Edit the text`. No number.
- **error** (red only here): empty → `Nothing to score` · `The text box is empty.` · `Paste an
  article, or load a sample.`; 413 → `Too much text` · the size and the 256 KB limit, noting the
  form path counts encoded characters · `Paste one article.` + `Restore your text`; 503 → `The
  detector isn't available` · which file did not load · `Start the server from the repository, or
  see README › Run.`; 500 → `Something went wrong on the server` · `Logged as request <id>; no
  details are shown here.` · `Try again.`; 404 → `Nothing at <path>` (≤ 80 chars, wraps) · `Go to
  the detector`.
- **Numbers:** whole percent, half-up; complement = 100 − shown; `<1 %` / `>99 %` clamps with the
  complement `>99 %` / `<1 %`; a value is shown only when `isinstance(x, (int, float))` and
  `math.isfinite(x)`, else `not scored`; Jinja uses `is not none`; number and `%` joined by
  U+00A0; Geist Mono `tabular-nums`; no colour for fake vs real. Banned on the page (visible text
  and `title`/`aria-label`/`placeholder`/`value`): decimal percents, the words `HIGH`, `MEDIUM`,
  `LOW`, `VERY_LOW`, `FAKE`, `REAL`, `None`, `nan`, `True`, `False`, "consider carefully",
  "calibrated", "accurate", "verified", emoji.

### Tests (stdlib `unittest`; the expected copy is a fixture transcribed from this section)
- `test_golden` — exact equality with the sha256-pinned `tests/golden.json` on both routes, status
  included; the only new key allowed is `input_check`; asserts 14 inputs; through the Flask test
  client.
- `test_api_contract` — keys and types on all routes; `input_check` shape; `/api/models` new
  fields; a `joblib.load` counter reads 4 however many predictions run; JSON parsed with a raising
  `parse_constant`.
- `test_errors` — the error table on both predict routes and `POST /`; the banned-internals scan;
  `'<' * 256000` cleans in < 1 s; the golden regex probes unchanged.
- `test_thresholds` — the D2 and D3 probe tables, each on its side; the stored empty scores by
  `repr`; a `<=` stub with base exactly 0.0; coverage equal to the floor admitted; floor ≤ 0.5.
- `test_present` — raw 0–100 by 0.001: shown + complement = 100, clamps; band edges 45/46, 54/55,
  64/65, 79/80; all agreement lines.
- `test_page` — each state reached with a real input (only 413/503/500 by injection); exactly one
  `[data-state]`; its headline present and every other state's headline absent (headline element,
  case-sensitive); banned-pattern scan; page percents equal `present` recomputed from raw scores;
  refusals contain no `%` except the no_signal empty-score line; content 0.0 still renders; `None`
  renders `not scored`; no `disabled` in the JS.
- `test_samples` — first asserts the four basenames loaded; each sample's state; every on-camera
  shown integer ≥ 1 point (raw) from its band and lean edges and from 3.0; the dateline sample's
  bytes are `PREFIX + sensational_post`; on every dateline pair (samples and golden) the content
  probability is identical and the format fake drops ≥ 10 points; the hold sample clears the lean
  edges by ≥ 2 and still splits with any one body sentence removed.
- `test_hygiene` — no `skip`/`SkipTest` in `tests/`; every listed module has ≥ 1 test.
- `test_rehearsal` — renders every URL in `docs/rehearsal.json` through the test client and
  compares sha256s; two renders are byte-identical.
- `test_run_mode` — free ports (never 8080), temp cwd, `start_new_session` + `killpg`; `python
  app/app.py`, `main.py run` and gunicorn `--chdir`: `/health` 200, `/console` 404, a prediction
  succeeds, pandas blocked; `FND_MODEL_DIR` with the two format files → format-only states; empty
  → 503 `model_unavailable`.
- `scripts/mutants.py` — runs the suite (fails on any skip or zero tests), then in a temp copy
  (no `.venv`, no `model/advanced`) applies each mutant and requires its module to fail: blend
  0.6→0.5; FAKE/REAL index swap; whole-string `<[^<>]*>`; `NO_SIGNAL_POINTS` 3.0→2.0 and 3.0→4.0;
  `COVERAGE_FLOOR` 0.5→0.3; half-up → `int()`; complement from `round(real)`; debug on by default;
  cap `<=` → `<`; exception text in the 500 body. A mutant whose target text is missing fails.
- `scripts/docker_smoke.sh` — `set -euo pipefail`; non-zero when the daemon is down; `docker build
  --iidfile`; runs that id on `127.0.0.1::8080`; waits ≤ 60 s for `healthy`; two golden inputs
  compared field by field; `input_check` and `model_breakdown.content_model` present; `/console`
  404; `id -u` ≠ 0; `import pandas` must raise `ModuleNotFoundError`; `pip check`; `pip freeze`
  equals `requirements.txt`; `trap` removes the container.

### Run story
Local: `uv venv --python 3.12 .venv` · `uv pip install --python .venv/bin/python -r
requirements.txt` · `.venv/bin/python app/app.py` → `http://127.0.0.1:8080`, debugger off
(`FLASK_DEBUG=1` for UI work; `PORT=` to move). Or `python3.12 -m venv .venv` and
`.venv/bin/pip install -r requirements.txt`. Also `.venv/bin/python main.py run`,
`.venv/bin/gunicorn --bind 127.0.0.1:8080 --workers 2 --preload app.app:app`, and
`.venv/bin/python -m unittest discover -s tests -v`. Docker: `docker build -t fake-news-detector .`
· `docker run --rm -p 8080:8080 fake-news-detector` · `curl -fsS localhost:8080/health`.

### Recording (~45 s; every number is measured when the samples are committed)
1. First load (title, lede, scope line, `No text scored yet`). 2. `Wire-style report` → a
real-leaning verdict, both models agree. 3. `Sensational post` → a fake-leaning verdict. 4.
`Sensational post + agency dateline` → `The two models disagree` (hold). 5. `Headline only` → `Too
little to go on`. 6. Terminal (≥ 40 rows, `clear` between): `curl -s localhost:8080/api/models |
python3 -m json.tool`, then `curl -s -H 'Content-Type: application/json' --data
@samples/wire_report.json localhost:8080/api/predict | python3 -m json.tool`, holding on
`model_breakdown` and `input_check`. Record at the rehearsed window size with DevTools closed.

## Options considered
Five frames sketched blind (frontend, backend, tester, reviewer, manager — 15 sketches), then one
critic scored, clustered and merged them (2026-09-16).

- Cluster — **withhold what isn't earned**: page refuses short/unrecognised text, names the
  refusal. *Kept.*
- Cluster — **the demo texts are the test suite**: every text the camera shows is committed and
  test-pinned. *Kept.*
- Cluster — **make the one path run**: freeze the working env, fix the live app in place, test
  the service against the pickles. *Kept.*
- Cluster — **show what moves the score**: probes, attributions, an audit table. *Kept only as the
  dateline sample and its test.*
- Cluster — **the artefact is the pin** (manifest, lock file, CI, published image). *Rejected:*
  publishing needs sign-off, recurring upkeep, nothing on camera.
- Cluster — **reshape the tree** (delete / archive / package). *Rejected for now:* large diff,
  needs Vishrut's OK, gains nothing for the recording — its own PR later.
- Cluster — **replace the runtime** (numpy re-implementation). *Rejected:* days of parity work,
  swaps the code behind the cited numbers.
- Trap — demo texts chosen because they "come out right": the models score style, so a fake/real
  montage implies fact checking. Every sample is labelled as written for the demo.
- Trap — "work really good" read as retrain / download / BERT: no data, and the training scripts
  would overwrite the served models.
- Trap — a hard 422 from the API for short text: breaks a 200 on a cited route; the page refuses
  instead and the API reports `input_check`.
- Trap — a word floor stated as measured when it rests on two texts: needs the sweep, a DECISIONS
  entry, and "in our tests" copy.
- Trap — trusting `models_agree`: it says `false` when one model sits at 48 %; the page uses its
  own lean band and derives the complement as 100 − shown.
- Trap — "calibrated", "98.75 %" and two decimals: nothing is calibrated, the figure cannot be
  re-run, the decimals are false precision.
- Rejected — probe bench with in-place JS (two render paths, JS formatting nullable numbers).
- Rejected — casebook table as the front page (static on video, reads as curated).
- Rejected — occlusion `top_terms` (3–4 days, credit split across correlated n-grams, unmeasured
  on long text).
- Rejected — 5,000-word truncation (changes numbers silently; a 256 KB refusal instead).
- Rejected — sealed 60-item holdout now (needs Vishrut to write 60 texts; offer later).
- Rejected — CI badge and `docker run` on camera (red-badge cost; not an ML insight).
- Rejected — the audit script as the whole deliverable (leaves the page unimproved).

## Critiques

### frontend · round 1
Measured 2026-09-16, read-only: `curl -X POST :8080/api/predict -d @file` without `-H` → 500, with the JSON header → 200.
Flask 3.1.3 `jsonify` with debug off prints one compact line. A warm predict takes 2.3 ms. Lorem ipsum (69 words)
matches 6/5 n-grams → 64.9/52.5 % fake. `WASHINGTON (Reuters) -` matches 0 content terms → 49.8 % (the model's
default). All palette text is ≥ 6.06 : 1; Chrome's default placeholder #757575 on #121214 is 4.06 : 1. The first
tree's root weight is 1600.0, which backs "about 1,600".

1. BLOCKING — headlines contradict the numbers beside them. The ladder reads the raw `confidence_level` (> .55/.65/.80),
   so 54.6 gives `Too close to call` and 55.4 gives `Leans fake`, yet both print `55 % fake · 45 % real` (the same
   happens at 65 and 80). The draft also counts 45 % as a lean under "≥ 5 points from 50" and as no lean under
   "45–55". Change: `present.py` rounds once, half-up, from unrounded floats (not the API's 2-dp values: 52.4951 →
   52.50 → 53), and every band reads the shown integer: 50–54 · 55–64 · 65–79 · ≥ 80. A model leans iff its shown
   fake is ≤ 45 or ≥ 55. The page ignores `confidence_level`. `test_present` pins 45/46, 54/55, 64/65, 79/80 and
   all five agreement lines (both fake, both real, neither, one lean + one without ×2); the draft writes two.
2. BLOCKING — `Score it anyway` will loop. The browser builds form data after `submit` and skips disabled
   controls, so disabling buttons in the handler drops `force=1` and the refusal comes back. Make it
   `<button type="submit" name="force" value="1">` in the same form (it scores the current text). Block repeats
   with a flag + `aria-disabled`, never `disabled`. On `pageshow` with `persisted`, reset the button, or Back
   restores the page stuck on `Analysing…`.
3. BLOCKING — shot 6 fails on camera. `-d @samples/wire_report.json` sends curl's default form type: 500 today,
   415 `json_required` under this draft. With the debugger off, the output is also one wrapped line. Shot list
   and README: `curl -s -H 'Content-Type: application/json' --data @samples/wire_report.json
   localhost:8080/api/predict | python3 -m json.tool` (same pipe for `/api/models`). Rehearse exactly these lines.
4. BLOCKING — likely inputs land in the wrong state. (a) A pasted URL, a stranger's likely first try, cleans to
   nothing and gets "…English news vocabulary", the wrong cause. Add `Links aren't opened` · `The detector reads
   text only. Paste the article itself, not its address.` (b) Lorem ipsum clears both the word floor and the
   zero-term rule, so it would show `Leans fake · 60 % fake · 40 % real`. Unrecognised needs a coverage floor,
   not zero (backend/tester set it), and lorem and a 70-word key-mash become named tests. (c) A model with 0 known
   terms shows `no words it recognises` in its row, even when forced. The draft's `The content model has no lean
   (50 % fake)` is exactly what that default looks like.
5. BLOCKING — state precedence is only partly defined. Make one ordered list, with each adjacent pair tested:
   error → URL-only → Unrecognised → Too short (unforced) → Disagreement → ladder. Short gibberish is
   Unrecognised, with no override. Forced short text that also splits keeps the `Scored from 11 words…` headline
   and shows the split as the amber agreement line. With no content model, `Blended 60/40` reads
   `Format model only — content model not loaded`.
6. BLOCKING — "Paste the full article" has no defined behaviour and reads as a paste action, which a page cannot
   do without a permission prompt. Label it `Edit the text`: it focuses the textarea, caret at the end, and never
   clears it.
7. BLOCKING — "fits the first viewport at 1280×800" is undefined. The copy says "on the left", but the draft
   never specifies columns, and a browser captured at 1280×800 has a shorter inner viewport (not measured).
   Specify two columns at ≥ 960 px, with the result column top-aligned and reserving the verdict's min-height so
   switching samples doesn't make the page jump. The rehearsal records `innerWidth×innerHeight` and the take uses
   the same size. First-load copy: `Paste an article or load a sample. Scores appear here.`
8. NOTE — input. `required` shows the browser's off-palette tooltip, so `Nothing to score` only appears for
   whitespace; add `novalidate`. `19 more needed` contradicts `Score it anyway`: use `41 of 60 words`, get
   singular/plural right (`1 word`) and add thousands separators. Python also splits on U+001C–1F and U+0085, JS
   on U+FEFF (measured): normalise these on both sides.
9. NOTE — numbers. Join the number and `%` with a literal U+00A0; `&nbsp;` via `|safe` would also un-escape
   `<1 %`. Use Geist Mono `tabular-nums`. The headline line is real-first and the rows are fake-first, so `66`
   sits over `38`. Make blended the first table row, in a fixed fake · real order, and let the headline alone name
   the top class.
10. NOTE — lost input. A 413 returns an empty box, and a sample click replaces a draft. Save the draft to
   `sessionStorage` (try/catch) before either, restore it after a 413, and offer `Restore your text`. Form
   encoding inflates punctuation, so the form path accepts less than 256 KB of text; the 413 copy should say so.
   Cap the echoed `<path>` at ~80 chars with `overflow-wrap:anywhere`.
11. NOTE — first frames. On a cold cache, Google-hosted Geist swaps in after first paint, and offline the failed
   request logs a console error. Bundle the two OFL woff2 files and preload them. Add an inline
   `html{background:#0A0A0B}` and a `color-scheme: dark` meta. Set an explicit `::placeholder` colour. Show busy
   without `opacity` (the amber button at 0.5 is 3.45 : 1). Add an amber `:focus-visible` ring.
12. NOTE — timing. At ~2 ms, `Analysing…` flashes for at most a frame; add no minimum delay. Say whether
   `GET /?sample=` also scores. I'd have it score (the GET is idempotent): that halves the clicks in a 45 s take.
   Don't reload a POST result on camera (resubmit dialog). Agreed, non-obviously: the fixed-height textarea and
   dropping the 300-char echo keep the result column still between takes.

TURN: backend

### backend · round 1
Measured 2026-09-16 (curl on :8080; `.venv/bin/python -B -c`; one throwaway loopback werkzeug server; `docker image ls`).
Preloaded scoring: short 2.8 ms, 20,000 words 53 ms, 256 KB 98 ms; today's code reloads four pickles per call (161 ms).
Cold start ~8.5 s (pandas 2.4, sklearn 5.0, pickles 1.1). `classes_` = `['FAKE' 'REAL']` on both; RF `n_jobs=None`.

1. BLOCKING — the routes' `except Exception` swallows every `HTTPException`. In-process, `RequestEntityTooLarge`,
   `BadRequest` and `UnsupportedMediaType` all return 500, so no 4xx code or cap reaches a client. Change: no try/except
   in routes; one `HTTPException` handler and one `Exception` handler (JSON on `/api/*` and `/health`). Not `is_json` →
   415; `get_json(silent=True)` not a `dict` → 400 `bad_json` (:8080 `["text"]` → 500); `RecursionError` (100 KB of `[`)
   → 400 `bad_json`. Add the missing codes: `text_empty` 400 (`/simple` gives 500 today), `url_only` 422, and JSON
   `not_found` 404 / `method_not_allowed` 405 under `/api/` (text/html today).
2. BLOCKING — `<[^<>]*>` is linear, but it changes cleaned text: `x < y < z > w` → `x  w` today, `x < y  w` after, and
   no golden input would notice. Keep `<[^>]+>` but apply it only up to the last `>`, then append the tail unchanged:
   same output on every probe, 0.1 ms on `'<'*256000`. The regex today takes 0.106 s at 16,000 chars; it is quadratic,
   so ~27 s at 256 KB, near gunicorn's 30 s timeout. Add the probes to golden and a <1 s `'<'*256000` case to `test_errors`.
3. BLOCKING — one body cap is not "the same cap on both paths". Flask 3.1.3 has `MAX_CONTENT_LENGTH` None and
   `MAX_FORM_MEMORY_SIZE` 500,000; form encoding made the README 1.33× bigger; with a 256 KiB body cap, 1 of 52
   oversized posts got `ConnectionResetError` instead of 413. Change: a 1 MiB transport cap (set both configs), then one
   `len(text.encode()) ≤ 256 KiB` check after decoding on both paths → 413 `too_large` + `{text_bytes, limit_bytes}`.
4. BLOCKING — loading. One `load_models(model_dir)` call at import, so `--preload` shares the models. It returns an
   immutable per-file record: the object, or the exception class + basename (never a path). Catch `Exception`. The dir
   comes from `__file__`, overridable by `FND_MODEL_DIR`: the only way `test_run_mode` gets a 503 without moving pickles.
   Score a warm-up text at load, because the first `transform` writes `_stop_words_id` on both vectorizers (measured;
   nothing else changes). The `joblib.load` counter patches before the loader runs.
5. BLOCKING — `input_check` is computed once, server-side, and the page reads `status`. `nnz` counts distinct n-grams,
   not words, so call it `known_features` (lorem 6/5, 68-word news 40/73). Add `coverage`: the share of the
   vectorizer's own tokens (its `token_pattern`, stop words as a frozenset) that are in its vocabulary, format/content.
   Lorem .09/.07, 70-word key-mash 0/0, French .27/.24, news .97/.75, sensational .76/.73; +12 ms at 256 KB. Order:
   `url_only` (every token matches `clean_text`'s URL regex) → `no_known_words` (cleans to nothing → 422 with
   `input_check`; max coverage < the sweep's `COVERAGE_FLOOR` → 200) → `too_short` → `ok`. `word_count` =
   `len(text.split())` with U+001C–1F/U+0085/U+FEFF counted as spaces; models get the raw text. Frontend 4–5 hold.
6. BLOCKING — run mode. `main.py run` calls `os.system("python app/app.py")`. Here `python` is anaconda 3.13.5 with
   sklearn 1.6.1 (`InconsistentVersionWarning`), and the relative path breaks outside the repo. Use
   `os.execv(sys.executable, …)` on the absolute path. `app.py` binds only 127.0.0.1 and reads `PORT`. `test_run_mode`:
   free ports, never 8080 (held all branch); temp cwd; `start_new_session` + `killpg` (the reloader forks); debugger off
   = `GET /console` → 404 (200 on :8080 today); gunicorn `--chdir <repo>`; `-W error::…InconsistentVersionWarning`.
7. BLOCKING — pins. The Python ≥ 3.12 floor comes from numpy 2.5.3 and scipy 1.18.1; scikit-learn 1.7.0 accepts ≥ 3.10,
   so fix the Constraint's stated reason. Pin the whole runtime set from `.venv` with `==`: flask 3.1.3, werkzeug 3.1.8,
   jinja2 3.1.6, markupsafe 3.0.3, itsdangerous 2.2.0, click 8.5.0, blinker 1.9.0, scikit-learn 1.7.0, scipy 1.18.1,
   numpy 2.5.3, joblib 1.6.0, threadpoolctl 3.7.0, gunicorn 26.2.0. An unpinned scipy is the likeliest break next year.
8. BLOCKING — Docker. `docker image ls` shows no python image at all, so the first build pulls; the smoke script should
   say so. Use `python:3.12-slim-<codename>`. Drop gcc/g++ and add `pip --only-binary=:all:` (arm64 host). COPY an
   allowlist: app/app.py, utils/, templates/, static/, samples/, the four pickles (`.venv` 209 MB, `model/advanced`
   89 MB). Fixed-UID user; `gunicorn --bind 0.0.0.0:8080 --workers 2 --preload app.app:app`; `--start-period=30s`.
   Smoke: `-p 127.0.0.1::8080` + `docker port` (8080 is taken), `trap 'docker rm -f …' EXIT`, ≤ 60 s `/health` wait, a
   JSON POST that asserts `model_breakdown.content_model` + `input_check`, `id -u` ≠ 0, `pip check`, and `pip freeze` ==
   `requirements.txt`.
9. NOTE — scoring returns raw floats; only the API rounds to 2 dp (agree with frontend 1). Take the RF label from
   `predict_proba`, so the forest runs once: 3.7 → 1.8 ms.
10. NOTE — at 100× traffic, two sync workers handle ~20 req/s at the cap. `GET /?sample=<id>` looks the id up in a dict and
    never builds a path from it; scoring on that GET (frontend 12) is pure and takes ~3 ms.
11. NOTE — the 500 handler logs the traceback and a uuid4 `request_id`, never the text. With no content model, `/health`
    returns 200 and `known_features.content_model` is null. Some dependency scanners flag `requirements*.txt`, so move
    `requirements-legacy.txt` under `docs/` or rely on `git show 23207ce:requirements.txt`.

TURN: tester

### tester · round 1
Measured 2026-09-16, read-only (unedited `utils/preprocess.py` in-process via `.venv/bin/python -B`; scratch files only).
Baseline sha256 `731d1b069da2…50e7ebc3a94`. Prefix sweep, % fake: wire article REAL at every prefix from 10 words (26–29),
14–17 with dateline; sensational FAKE 69–76 throughout; sensational + dateline R50 at 10, F54–59 from 20. Golden
`html_wrapped` (10 words) and `central_bank_dateline` (18) are REAL. Each dateline pair leaves the content model
bit-identical (`washington` is not in its vocabulary) and moves format −29.3 / −18.8 / −19.5. Golden dateline sensational:
format 44.46, 0.54 inside the 5-point edge; 2 of 8 drop-one-body-sentence variants lose the split. unittest: all-skipped
module exits 0. Headless Chrome logs console error/warn as `INFO:CONSOLE` and writes a 1280×800 PNG, but never exits alone.

1. BLOCKING — the too-short copy is already false: wire prefixes are REAL from 10 words and two golden short inputs are REAL.
   Style, not length, makes the headlines fake. Change: pre-register the MIN_WORDS rule in DECISIONS, committed before the
   script first runs (e.g. smallest N from which every sweep text's prefix verdict equals its full-length verdict); name the
   corpus in the script with sha256s, no glob; commit its TSV. New `test_sweep` re-runs it, compares to the TSV byte for byte,
   asserts `MIN_WORDS` = the rule's output, and checks each claim in the refusal copy against the table (the copy takes its
   number from the constant). If the rule lands below 12, shot 5 changes, not the rule. `COVERAGE_FLOOR` gets the same:
   pre-registered rule, named inputs on each side (lorem, key-mash, French .27/.24 — pick its side and test it, news).
2. BLOCKING — the dateline sample invites tuning, and the golden shows the split is knife-edge. Change: (a) its bytes are
   asserted to be `PREFIX + sensational_post`, so it cannot be tuned alone; (b) the camera claim is the mechanism, tested on
   every dateline pair in golden and samples: content raw probability identical, format fake down ≥ 10 points; (c) the hold
   sample clears the lean edge by ≥ 2 points on both models and splits in every drop-one-body-sentence variant; (d) every
   draft scored is listed in DECISIONS (sha256, both scores, kept or dropped); (e) every on-camera raw value sits ≥ 1 point
   from each band edge, so the page and shot 6's `confidence_level` cannot contradict each other.
3. BLOCKING — the golden can pass while broken. Exact equality, no tolerance (blend 0.60→0.59 moves the dateline sensational
   by 0.36; ±0.5 hides it), on the whole recorded body of both routes, status included; extra keys = `{input_check}` only;
   assert 14 inputs (an empty loop passes); go through the Flask test client, not `predict_ensemble`; pin golden.json's
   sha256 and assert `git log 23207ce..HEAD -- tests/golden.json` shows one commit. Backend 2's probes go in a second
   pre-edit file captured from :8080 now; golden.json stays byte-identical to the capture.
4. BLOCKING — nothing shows a test fails on wrong code, and skips pass green. Add stdlib `scripts/mutants.py`: run the whole
   suite, fail on any skip or 0 tests; then in a temp copy apply ~9 one-line mutants, each of which must fail its module:
   blend 0.6→0.5; FAKE/REAL index swap; whole-string `<[^<>]*>`; `MIN_WORDS` ±1; half-up → `int()`; complement from
   `round(real)`; debug on by default; cap `>` → `>=`; `str(e)` in the 500 body. A mutant whose target text is gone fails.
5. BLOCKING — page tests must not only find the implementer's strings. Expected copy is a fixture transcribed from this plan,
   committed with the tests. States come from real inputs via `POST /` and `GET /?sample=` (only 413/503/500 by injection).
   Assert one `[data-state]` with the expected value, its headline present and every other state's headline absent. Scan
   visible text plus `title`/`aria-label`/`placeholder`/`value` (not the textarea's content) for `\d[.,]\d+\s*%`, whole-word
   `HIGH|MEDIUM|LOW|VERY_LOW|FAKE|REAL|None|nan|True|False`, emoji and the banned words. Each page percent must equal
   `present` recomputed from raw scores. Unforced short/unrecognised: no `%` in the card. `test_present` sweeps raw 0–100 by
   0.001 (fake + real = 100, clamps). Frontend 2's loop is browser-only: test the markup and no `disabled` in JS; click it by hand.
6. BLOCKING — `docker_smoke.sh` can pass on the wrong server: the unedited app answers 200 on :8080 all branch. Change:
   `set -euo pipefail`; exit non-zero, never "skipped", when the daemon is down; `docker build --iidfile` and run that id;
   host port from `docker port`; wait for `docker inspect` health = `healthy`; POST two golden inputs and compare every field
   to `tests/golden.json` (same pickles, same numpy/sklearn numbers); `import pandas` inside must fail; plus backend 8.
7. BLOCKING — the take can show the old app or stale numbers. With :8080 held, `app/app.py` fails to bind, or the tab and
   shot 6 reach the old process. Stop it first (Vishrut's OK); rehearsal opens with `/api/models` showing `loaded`. The
   recording point precedes steps 5–6, so `docs/rehearsal.json` records, per shot, the state and every shown integer,
   `innerWidth×innerHeight`, `devicePixelRatio`, and sha256s of `templates/`, `static/`, `utils/present.py`, `samples/`.
   `test_rehearsal` fails if those files changed since, or if the samples now render different integers.
8. NOTE — "clean console" becomes `scripts/rehearse.py`: headless Chrome per recorded URL at the recorded inner size, with a
   watchdog kill, failing on any `INFO:CONSOLE` line and writing into rehearsal.json. That needs frontend 12 (the sample GET
   scores), so every recorded state has a URL. "Fits" stays a look: the card's scope line in view, unscrolled, at the
   recorded size (headless 1280×800 is taller than a headed window's viewport).
9. NOTE — `.venv` keeps pandas 3.0.5, so a leftover `import pandas` passes locally. `test_run_mode`: import with
   `sys.modules['pandas'] = None`; pins vs `importlib.metadata` (no pip); commands and frontend 3's curl from README's code
   block, not literals; `FLASK_DEBUG`/`PORT` scrubbed; loaded basenames = the four pickles; backend 3's reset (real socket).
10. NOTE — frontend 8's word-count parity: a shared table under `node` (v22.16.0 here), failing if absent. No command sees
    narration implying fact checking, zoom or extension overlays; Vishrut watches the take against the shot list.

Proposed `## Done looks like`:
- [ ] Existing API fields and the pre-edit golden bodies are exactly unchanged on both routes — `.venv/bin/python -m unittest tests.test_golden tests.test_api_contract`
- [ ] Malformed, non-text and oversized requests get a 4xx with a named code and no Python internals — `.venv/bin/python -m unittest tests.test_errors`
- [ ] Real inputs reach every named state, in whole percents with the complement, with no decimal percent, raw enum, emoji or banned word — `.venv/bin/python -m unittest tests.test_page tests.test_present`
- [ ] The committed sweep reproduces, sets MIN_WORDS and COVERAGE_FLOOR by the pre-registered rules, and backs the refusal copy — `.venv/bin/python -m unittest tests.test_sweep`
- [ ] Each sample shows its claimed state with margin; the dateline sample is the post plus a prefix and moves only the format model — `.venv/bin/python -m unittest tests.test_samples`
- [ ] The documented commands serve with the debugger off from any directory, without pandas, and give 503 with no models — `.venv/bin/python -m unittest tests.test_run_mode`
- [ ] The suite runs with no skips, and every listed mutant fails its module — `.venv/bin/python scripts/mutants.py`
- [ ] A freshly built image turns healthy, runs as non-root and returns the golden numbers — `bash scripts/docker_smoke.sh`
- [ ] The four served model files are byte-identical to `23207ce` — `git diff --exit-code 23207ce -- model/fake_news_model.pkl model/vectorizer.pkl model/content_model.pkl model/content_vectorizer.pkl`
- [ ] The rehearsal matches the current page and samples, and its console was clean — `.venv/bin/python -m unittest tests.test_rehearsal`
- [ ] Vishrut watched every recorded state at the recorded window size, with the result card in view

TURN: frontend

### frontend · round 2
Measured 2026-09-16, read-only (`.venv/bin/python -B -W ignore -c` importing `utils.preprocess`; JSON curl on :8080). `transform([""])` scores
format 65.52 % fake, content 49.76, blended 59.21. Distance from those, format/content (blended): MIT headline −0.23/−1.62 (58.42);
`City council votes to expand bike lanes downtown` 0.00/−1.70; `Stock markets fall as investors weigh inflation data` 0.00/−2.45;
`Apple unveils new iPhone with faster chip and longer battery life` 0.00/+10.15 (63.27); `NASA probe sends back first close-up images
of distant asteroid` +4.81/+12.20 (66.98); birds −6.60/−9.54 (51.44, split); `The central bank held interest rates steady on Tuesday,
officials said.` −24.27/−33.29 (31.34); `Breaking: you will not believe what they are hiding from you` +2.68/+25.43 (71.00). Scheme-less
`bbc.co.uk/news/world-europe-12345678` misses the URL regex and scores exactly the empty answer. `max(nan, .2)` is nan; `max(.2, nan)` is .2.
`/api/predict | python3 -m json.tool` prints 24 lines today.

1. BLOCKING — drop the word floor; "Too short" becomes `no_signal`. 5 of the 6 headlines above lean fake, and 4 get the format model's
   empty-text answer to within 0.23, while 11-word wire and sensational sentences move a model 24–33 points. Length can't separate these;
   distance can, per input, with no sweep. Trigger: every loaded model's raw fake % is within `NO_SIGNAL_POINTS = 3.0` of its own
   empty-string score (computed at load, never typed). Both models, not format alone: tester·1's numbers put the undated sensational
   article's format near 64 (inferred), so a format-only rule would refuse shot 3. 3.0 was chosen after seeing only the inputs above,
   before any sample exists; DECISIONS lists them. Pinned in: MIT, City council. Out: Apple, NASA, birds, both sentences. `Stock markets`
   (2.45) makes a 3.0 → 2.0 mutant fail. Long samples sit ≥ 4 from empty on some model; the headline sample ≥ 1 inside on both (MIT 1.38).
   Copy: `Too little to go on` · `Both models scored this within 3 points of an empty text (59 % fake · 41 % real), so the score says
   nothing about these words.` · `Edit the text` (no content model: `The format model scored…`, 66 % · 34 %). Those numbers come from the
   stored empty scores via `present` and are the card's only percents: no input score, no override. Gone: `MIN_WORDS`, the length sweep
   (`test_sweep` keeps `COVERAGE_FLOOR`), `force=1`, `Score it anyway`, `Scored from…`, `41 of 60 words` (counter: `152 words`); my round-1
   point 2 and point 5's forced-short line are withdrawn. If MIT misses the margin once committed, cut shot 5; don't rewrite the sample.
   Order: error → url_only → unrecognised → no_signal → disagreement → ladder. `status` ∈ `ok | url_only | no_known_words | no_signal`,
   no `min_words`; the empty scores go on `/api/models`, not in every predict body.
2. BLOCKING — once coverage decides it, the Unrecognised copy is false and trips tester 5's scan: `None of this text matches…` is untrue
   for lorem (.09/.07) and French (.27/.24), and opens with the whole word `None`. New: `Too few words the models know` · `Most of this
   text isn't in the English news vocabulary the models learned, so there is little for them to read.` · `Edit the text`. "Most" needs
   `COVERAGE_FLOOR ≤ 0.5` (asserted). French goes on the refused side; news at .73+ clears a .5 floor by .2. A model with no tokens has
   coverage `null`, never NaN, and the max skips nulls (both null → `no_known_words`).
3. NOTE — `url_only` first: agreed. Also treat a single space-free `host.tld/…` token as a link; a typed address without a scheme
   otherwise gets the vocabulary card, the wrong cause. Copied addresses carry a scheme.
4. NOTE — margins: agreed. The ±1 also applies to `NO_SIGNAL_POINTS` and to the 45/55 lean edges on each model row.
5. NOTE — one `[data-state]`: agreed. Values `first_load`, `verdict`, `disagreement`, `no_signal`, `url_only`, `unrecognised`, `error`
   (+ `data-code`), and `not_found` on the 404 page. Absence checks read the headline element, case-sensitively: otherwise the split
   card's `— leans real` matches the `Leans real` headline.
6. NOTE — fixture: fold the accepted amendments into `### Page states and copy` before the freeze, and transcribe once; the body still
   says `Paste the full article`, `60 needed` and `None of this text`. The fixture carries U+00A0 before `%`.
7. NOTE — before the recording point, only what changes a pixel or the take: golden, errors, samples, present, page, rehearsal.
   `scripts/mutants.py`, `docker_smoke.sh` and `test_run_mode` come after it. Record `innerWidth×innerHeight` with DevTools undocked
   (docked DevTools shrinks the viewport). Shot 6 is ~35 lines with `input_check`: `clear` between the two curls and use ≥ 40 rows.

TURN: backend

### backend · round 2
Measured 2026-09-16, read-only (`.venv/bin/python -B -W ignore` from stdin, importing `utils.preprocess`; `lsof`/`ps` on :8080; werkzeug 3.1.8
source; `docker version` 28.0.4). Raw empty-text fake: format 0.6551573256765925, content 0.49755588867465184, blend 0.5921167508758163; both take
5.05 ms once. Zero known n-grams scores bit-identical to `transform([""])` (key-mash 0.0/0.0 from empty; `the and of a` 0.00/1.88; `Reuters -`
17.96/0.00). Old `predict_ensemble` vs load-once + `classes_` index + label from `predict_proba`: 0 diffs in 36 calls. On 200,000 random pairs,
real as `1 − fake`, swapped operands, `(3f+2c)/5` and numpy `round` change no 2-dp value. `jsonify` emits a bare `NaN`. `.venv` has no pip.

1. BLOCKING — :8080 will not stay "the unedited app". PIDs 92303/92322 are a Werkzeug reloader pair (`.venv`, one socket via `WERKZEUG_SERVER_FD`).
   The first save under `app/` or `utils/` reloads it onto new code; after the save that turns debug off it serves that snapshot and stops
   reloading; a save with an import error kills both and frees 8080. Tester 6/7 assume otherwise, and the golden capture races the first edit.
   Change: capture `golden.json` and my round-1 probe file through the Flask test client from `git archive 23207ce` unpacked in a temp dir
   (`.venv` interpreter, cwd there); `test_golden` re-derives and compares bytes, which survives the rebase. Stop :8080 before step 1 (Vishrut's OK).
2. BLOCKING — `no_signal` is exact and costs nothing per request, if defined on raw floats. The load warm-up becomes `transform([""])` per loaded
   pair (replaces my round-1 warm-up text), stored as float64 in the immutable load record; per request `abs(p − p0) * 100 <= NO_SIGNAL_POINTS` on
   each loaded model's raw fake probability. Pin `<=` with a stub (p0 0.0, p 0.03 → exactly 3.0). Never from the 2-dp values.
3. BLOCKING — `input_check` (both predict routes): `status` `"ok"|"url_only"|"no_known_words"|"no_signal"`; `word_count` int; `known_features`
   (int), `coverage` (float 2 dp, 0–1) and `points_from_empty` (float 2 dp, ≥ 0), each `{format_model, content_model}`; no `min_words`. The server's
   distance goes in the body because a client subtracting two 2-dp values can disagree with `status` within 0.01 of the edge. `null` = model not
   loaded; `coverage` is also `null` when the vectorizer yields no tokens; never NaN (tests parse with a raising `parse_constant`). Precedence:
   `url_only` (every token a link, frontend 3's `host.tld/…` included) → `no_known_words` (every non-null coverage < `COVERAGE_FLOOR`, or all null)
   → `no_signal` → `ok`. 422 only when `clean_text` gives `""` (code `url_only` / `no_known_words`, body carries `input_check`); all else stays
   200 — scheme-less `bbc.co.uk/…` is 200 today and stays 200 with status `url_only`. `/api/models` adds per-model `empty_text_fake_prob` (2 dp |
   null) and `no_signal_points`; with no content model the blended empty score is the format one.
4. BLOCKING — a missing model changes states, not just rows. Without the content model, `no_signal` runs on format alone, so the sensational post
   (format ~64 vs 65.52, frontend 2's inference) is refused while `/health` says 200. `model_breakdown.content_model` stays omitted, as today;
   `input_check`'s content fields are null. `test_samples` first asserts the four basenames loaded; `test_run_mode` adds a two-pickle
   `FND_MODEL_DIR` case (200, format-only status, `The format model scored…`). A pair is loaded only if both files load and its empty scoring runs.
5. NOTE — exact golden equality costs no false failures (no harmless rewrite above moves a digit); those rewrites are equivalent mutants, so list
   only semantic ones: `MIN_WORDS ±1` → `NO_SIGNAL_POINTS 3.0→2.0`. The runner's temp copy skips `.venv` (209 MB) and `model/advanced` (89 MB).
6. NOTE — Docker: the default 30 s healthcheck interval puts the first probe ~30 s in, so a 60 s `healthy` wait is tight: add `--start-interval=1s`
   (Engine ≥ 25). Scoring uses no BLAS (sparse matvec, float32 tree thresholds): expect exact 2-dp golden values on linux/arm64, with no tolerance.
   `! docker run … 'import pandas'` never trips `set -e`: use `if …; then exit 1; fi` and require `ModuleNotFoundError`. `sys.modules['pandas'] =
   None` can only wrap a `-c` import check; for the documented commands, the pandas-free image is the proof.
7. NOTE — before the take (with frontend 7): stop :8080, archive golden, `test_api_contract`, `test_errors`, loader + empty scores + `input_check`,
   the `COVERAGE_FLOOR` rule, samples, present, page, and tester 7's hash file written at the take (cheap; it is what makes steps 5–6 safe). After:
   `mutants.py`, `docker_smoke.sh`, `test_run_mode`, `rehearse.py` (a manual console check at the recorded size will do for the take). Mutants and
   the in-image golden cost more than they buy before the take: neither proves anything the camera shows.

TURN: tester

### tester · round 2
Measured 2026-09-16, read-only (`.venv/bin/python -B -W ignore` from stdin, joblib-loading the four pickles; `lsof` + curl on :8080; `unittest discover` on an
empty scratch dir). Raw empty scores match backend 2. For 25 one-line headlines I wrote, the largest per-model distance from empty was: ≤ 3 for Tech recalls
0.06, Hospital 0.13, New study 0.17, University 1.33, Company profits 1.49, MIT 1.62, City council 1.70, Court rules 2.25, Stock markets 2.45; > 3 for Police
arrest 3.51, Museum 3.65, Airline 3.66, Senate 4.15, Mayor 4.27, Teachers 4.45, Heavy rain 5.16, and the rest ≥ 8. There is no gap near 3. 9 of the 16 scored
would read `Leans fake` (55–62 %), and none leans real. Largest coverage: MIT .80. `Researchers find new species of frog in rainforest` has .40/.14, and four
others are exactly .50 on content. `abs(0.03-0.0)*100 == 3.0`, but `abs(0.53-0.5)*100` is 3.0000000000000027. A run of 0 tests exits 5; all-skipped exits 0.
The old `app.py` appends its root to `sys.path` (it does not prepend) and loads `model/…` from the cwd, and the old `preprocess.py` imports pandas.

1. BLOCKING — nothing checks the upper edge of `no_signal`. Frontend 2's nearest "out" probe is birds at 9.54, so `NO_SIGNAL_POINTS` 3.0 → 9.0 passes every
   probe and every sample. `tests/test_thresholds.py` replaces `test_sweep`. It checks Stock markets and Court (in) and Police, Museum and Airline (out), plus
   frontend 2's set; mutants 3.0→2.0 and 3.0→4.0 both fail it. It also asserts the raw empty scores by `repr`, `/api/models` 65.52/49.76, the fixture's `59 %`
   and `66 %` lines, and the `<=` stub with a base of exactly 0.0 (a 0.5 base gives 3.0000000000000027, so that stub would test nothing).
2. BLOCKING — record the choice of 3.0 in DECISIONS rather than test its timing: a git-ancestry test breaks on a squash merge. At freeze, before `samples/`
   exists, D<n> records 3.0, the probe table (text, sha256, both distances, side) and the withdrawn MIN_WORDS rule with its reason. It also says plainly that 3.0 is a
   convention: nothing measured separates 2.45 from 3.51. PROGRESS evidence: `git ls-files samples` prints nothing. The margins make tuning pointless: samples sit ≥ 1
   from 3.0 and probes ≤ 0.66 either side. Any value in (2.45, 3.51) leaves every sample's state unchanged, and a value outside that range fails a probe.
3. NOTE — the rule refuses only readings close to the empty-text score; it does not refuse headlines as such. 9 of the 16 headlines it scored still read `Leans fake`. README, DECISIONS and the case file say "within
   3 points of an empty text", never "short text is refused". Any threshold has a scored neighbour just outside it, so another round would only move this edge.
4. BLOCKING — `COVERAGE_FLOOR` cannot change any sample (MIT .80/.70; long samples ≥ .73), so its risk is text from strangers. Pre-register 0.5 in the same D<n>.
   Named as refused: lorem, key-mash, French, and the English frog headline (.40; the refusal is stated in D<n>, not left to be found later). Named as admitted: news,
   sensational, MIT. A stub checks that coverage exactly equal to the floor is admitted (`<`), because 1/2 and 3/6 are common in short text. The test asserts
   `floor ≤ 0.5` and a margin of ≥ floor + .1 for every sample, and parses with a raising `parse_constant`.
5. BLOCKING — the archive capture can silently run the new code. The old `app.py` appends its root to `sys.path`, so a process that already has the repo on its
   path imports the edited `utils`, and a `test_golden` that re-derives in-process compares the new code with itself. Fix: capture in a subprocess (cwd = the
   unpacked archive; `PYTHONPATH`, `FLASK_DEBUG` and `PORT` scrubbed) from `git archive 23207ce -- app utils templates` plus the four pickles, leaving out the
   89 MB `model/advanced`. The capture asserts `utils.preprocess.__file__` and `app.__file__` are inside the archive; that every predict body is a 200 with FAKE
   or REAL (a wrong cwd returns `Format model not found`); and that no body has `input_check`.
6. BLOCKING — the re-derivation cannot be part of the suite that runs on a fresh clone. The old `preprocess.py` imports pandas, and step 5 removes pandas from
   `requirements.txt`, so `unittest discover` would fail on a fresh clone. Split it: `scripts/capture_golden.py --check` re-derives and byte-compares (with
   no pandas it exits non-zero with a message naming pandas, and never skips). It runs in this `.venv` before the take. `test_golden` compares the app only with
   the sha256-pinned file.
7. NOTE — :8080: agreed, stop it before step 1. Test-client bodies are compact, while :8080's are pretty-printed under debug, so golden is never compared with curl output.
8. BLOCKING — the take can still show a stale server after :8080 is stopped. With debug off there is no reloader and Jinja caches templates, so an edit
   made after `app.py` starts never reaches the camera, and my round-1 source hashes cannot detect that. Right after the take, `scripts/rehearse.py --record`
   (urllib only) fetches each shot URL and its CSS/JS from the live server into `docs/rehearsal.json`: a sha256 per URL, the integers shown, and the
   `innerWidth×innerHeight` Vishrut reads off. `test_rehearsal` renders the same URLs through the test client and compares the hashes. It also asserts two
   renders are byte-identical, so those pages carry no request id and no nonce.
9. Split — I agree with backend 7 and frontend 7, with one change: the skip guard is cheap, so it moves before the take (`tests/test_hygiene`: no `skip` or
   `SkipTest` in `tests/`, and each listed module has ≥ 1 test). `test_rehearsal` makes it safe to leave mutants, Docker and run mode until after: a later fix that
   changes a served byte fails it and forces a re-take.

Final `## Done looks like`. Lines marked [take] must be green before Vishrut records. The [at take] line is written just after the take. The rest can come after.
- [ ] [take] The committed golden bodies re-derive byte for byte from 23207ce, run in isolation — `.venv/bin/python -B scripts/capture_golden.py --check`
- [ ] [take] Existing API fields and the golden bodies are exactly unchanged on both routes — `.venv/bin/python -m unittest tests.test_golden tests.test_api_contract`
- [ ] [take] Malformed, non-text and oversized requests get a 4xx with a named code and no Python internals — `.venv/bin/python -m unittest tests.test_errors`
- [ ] [take] The no-signal and coverage rules refuse or admit the pinned probes either side of each edge — `.venv/bin/python -m unittest tests.test_thresholds`
- [ ] [take] Real inputs reach every named state in whole percents with the complement, with no decimal percent, raw enum, emoji or banned word — `.venv/bin/python -m unittest tests.test_page tests.test_present`
- [ ] [take] With all four models loaded, each sample shows its claimed state with margin; the dateline sample is the post plus a prefix and moves only the format model — `.venv/bin/python -m unittest tests.test_samples`
- [ ] [take] No test is skipped and no listed test module is empty — `.venv/bin/python -m unittest tests.test_hygiene`
- [ ] [take] The four served model files are byte-identical to 23207ce — `git diff --exit-code 23207ce -- model/fake_news_model.pkl model/vectorizer.pkl model/content_model.pkl model/content_vectorizer.pkl`
- [ ] [take] Vishrut looked at every recorded state in a headed browser at the recorded inner size, on a server started after the last edit, with a clean console and the result card in view
- [ ] [at take] The pages served during the take are byte-identical to what the current code renders — `.venv/bin/python -m unittest tests.test_rehearsal`
- [ ] The documented commands serve with the debugger off from any directory and without pandas; two pickles give format-only states, none gives 503 — `.venv/bin/python -m unittest tests.test_run_mode`
- [ ] Every listed mutant, including no-signal 3.0→2.0 and 3.0→4.0, fails its module — `.venv/bin/python scripts/mutants.py`
- [ ] A freshly built image turns healthy, runs as non-root, has no pandas and returns the golden numbers — `bash scripts/docker_smoke.sh`
- [ ] Each recorded URL loads in headless Chrome at the recorded size with no console message — `.venv/bin/python scripts/rehearse.py --console`
- [ ] Vishrut watched the take against the shot list: no narration implying fact checking or that headlines are refused, no zoom, no overlays

TURN: freeze

---

PLAN FROZEN: 2026-09-16 — argued in two rounds by frontend, backend and tester (the tester called the freeze); every BLOCKING point above is folded into the sections above or dropped in DECISIONS D5. Signed claude for Vishrut, who asked for research, a brainstorm, then implementation.
