# Decisions — Fake News Detector

Append-only. Each decision carries a reason, the alternative it beat, and a Status:
`settled` (argued and stands) · `taken` (decided by one person, contestable) · `contested`
(a challenge is open) · `open` (nobody has decided). Never edit an entry; supersede it with a
new one. Contest by appending `### Challenge — D<n> — <who>` and flipping the status.

## D1 — Serve the committed models unchanged; no retraining this round · settled · claude (plan-together, signed for Vishrut) · 2026-09-16

The four served pickles (`model/fake_news_model.pkl`, `model/vectorizer.pkl`,
`model/content_model.pkl`, `model/content_vectorizer.pkl`) are not modified, and `main.py train` /
`all` refuse to run.

*Reason:* retraining was the literal reading of "make it work really good". It lost because the
dataset is not in the repo (a download needs Vishrut's Kaggle credentials and sign-off), the
committed training scripts do not reproduce the served pickles (`model/train_model.py` produces
the older 10,000-feature logistic regression and would overwrite the served forest), and every
measured number the portfolio will quote would move.

*Revisit if:* Vishrut downloads the dataset and wants a retrained or re-evaluated model.

## D2 — "Too little to go on" means every loaded model scores within 3.0 points of its empty-text score · settled · frontend, backend, tester; signed claude for Vishrut · 2026-09-16

The page refuses to show a score (state `no_signal`) when, for every loaded model,
`abs(p − p_empty) × 100 <= 3.0`, where `p` is the raw FAKE probability for the input and `p_empty`
is the raw FAKE probability that model gives an empty string, computed when the models load. The
API still scores the text and reports `input_check.status = "no_signal"`.

Empty-text scores (raw): format 0.6551573256765925, content 0.49755588867465184,
blend 0.5921167508758163.

*Reason:* the first draft refused text under a word count (`MIN_WORDS`, default 60, to be set by
a length sweep). It was withdrawn because it is false: prefixes of a wire-style article score REAL
from 10 words on, and two short golden inputs are REAL (tester · round 1). Length does not decide
the score; style does. Distance from the empty-text answer shows, per input, when the models have
read nothing that moves them.

**3.0 is a convention, not a measured boundary.** Nothing measured separates 2.92 from 3.08. It
was chosen after seeing only the probes below, before any file under `samples/` existed. The rule
does not refuse headlines as such: most one-line headlines score outside it and read "Leans fake".

Probe table (unedited code, 2026-09-16; distance = max over the two models, in points):

| side | max distance | format · content | text |
|---|---|---|---|
| in | 2.92 | 0.00 · 2.92 | Bridge closes for inspection after cracks found |
| in | 2.79 | 0.00 · 2.79 | Airline cancels hundreds of flights as storm approaches |
| in | 2.45 | 0.00 · 2.45 | Stock markets fall as investors weigh inflation data |
| in | 1.70 | 0.00 · 1.70 | City council votes to expand bike lanes downtown |
| in | 1.62 | 0.23 · 1.62 | Scientists at MIT develop new water purification technology using solar energy |
| out | 3.08 | 1.11 · 3.08 | Bank raises savings rates for customers |
| out | 3.83 | 0.00 · 3.83 | Workers finish repairs on main water line |
| out | 3.87 | 2.63 · 3.87 | Retail sales rise for third straight month |
| out | 4.15 | 0.88 · 4.15 | Senate passes bill to fund highway repairs |
| out | 9.53 | 6.59 · 9.53 | Government plans to replace all birds with surveillance drones |
| out | 10.15 | 0.00 · 10.15 | Apple unveils new iPhone with faster chip and longer battery life |

A change of the constant to 2.0 flips "Stock markets"; to 4.0 flips "Bank", "Workers" and
"Retail sales". `tests/test_thresholds.py` pins these.

*Revisit if:* the models are retrained (every empty score and distance changes), or a probe set
shows a real gap somewhere else.

## D3 — "Too few words the models know" means every loaded model's vocabulary coverage is below 0.5 · settled · frontend, backend, tester; signed claude for Vishrut · 2026-09-16

`coverage` = the share of the vectorizer's own tokens (its preprocessor, tokenizer and stop words)
that are in its vocabulary, computed on the text that model actually receives. The page shows
`Too few words the models know` when every non-null coverage is `< COVERAGE_FLOOR = 0.5`
(equality is admitted), or when every model yields no tokens. "Most of this text isn't in the
English news vocabulary" is only true while the floor is ≤ 0.5; a test asserts that.

Named as refused: lorem ipsum (0.09 · 0.07), a key-mash (0 · 0), a French sentence (0.24 · 0.31),
and the English headline `Researchers find new species of frog in rainforest` (0.40 · 0.14) —
refused on purpose, stated here rather than found later. Named as admitted: `Museum reopens after
two-year renovation` (0.50 · 0.40, exactly at the floor), the MIT headline (0.80 · 0.70), and the
long samples.

*Reason:* the draft refused only text with zero known terms. Lorem ipsum matches a few terms and
would have shown `Leans fake`. A floor at 0.5 separates the named nonsense from news at 0.7+.

*Revisit if:* a real English news paragraph is refused, or the vocabulary changes.

## D4 — The page uses the portfolio's visual system · taken · claude · 2026-09-16

Dark ground `#0A0A0B`, amber `#FFB000` as the one accent (look here / split), red `#F26D5B` for
faults only, Geist Mono for machine voice and numbers, Geist for prose, no emoji, no gradients.

*Reason:* the recording sits on vishrutmalhotra.com; the old purple-gradient, emoji-heading page is
exactly what that design system bans. The alternative, a neutral light theme, would look like a
different author's work on that page.

*Revisit if:* the app is shown somewhere other than the portfolio.

## D5 — Two proposals dropped at freeze · taken · claude (signer) · 2026-09-16

- `scripts/rehearse.py --console` (headless Chrome failing on any console line): no browser driver
  is installed in this repo, and headless Chrome does not exit on its own. The console check is a
  person looking at DevTools during the rehearsal.
- A Node test for word-count parity between the page's counter and the server: the page's counter
  is a typing aid; the number the result shows comes from the server.

*Reason:* both cost more than they protect before the recording.

*Revisit if:* the page gets a client-side render path.

## D6 — Unused code stays until after the recording · taken · claude · 2026-09-16

`app/advanced_app.py`, `app/final_app.py`, `backend/`, `notebook/`, the training scripts and the
92 MB `model/advanced/advanced_ensemble.pkl` are left in place; the README says which path is live.

*Reason:* deleting or archiving them is a large diff that needs Vishrut's OK and changes nothing
the camera sees. History is never rewritten.

*Revisit if:* Vishrut asks for the cleanup PR.
