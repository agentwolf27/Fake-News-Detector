"""Everything the page says about a score, decided in one place.

Rules (PLAN.md › Page):
- one rounding, half-up, from the raw probability; the complement is 100 minus the shown number;
- 0 and 100 are never shown: they become "<1 %" and ">99 %";
- a value is shown only when it is a finite number;
- verdict bands and model leans read the shown integer, so the words and the numbers beside them
  can never disagree;
- the page ignores the API's `confidence_level`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from utils.preprocess import CONTENT_WEIGHT, FORMAT_WEIGHT, NO_SIGNAL_POINTS, Reading

NBSP = "\u00a0"
SCOPE_LINE = "Trained on about 1,600 labelled news articles. It scores writing style; it does not check facts."
LEAN_FAKE_AT = 55  # a model leans fake when its shown fake % is at least this
LEAN_REAL_AT = 45  # and leans real when it is at most this

MODEL_NAMES = {"format_model": "Format model", "content_model": "Content model"}
MODEL_NOTES = {
    "format_model": "Random forest over word patterns, including datelines and agency names.",
    "content_model": "Logistic regression; agency names and datelines are removed first.",
}

HEADLINES = {
    "first_load": "No text scored yet",
    "no_signal": "Too little to go on",
    "unrecognised": "Too few words the models know",
    "url_only": "Links aren't opened",
    "disagreement": "The two models disagree",
}


def is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def shown(probability) -> int | None:
    """Whole percent, rounded half-up once from the raw probability (0–1)."""
    if not is_number(probability):
        return None
    return int(math.floor(float(probability) * 100 + 0.5))


def pct_text(value: int) -> str:
    if value <= 0:
        return f"<1{NBSP}%"
    if value >= 100:
        return f">99{NBSP}%"
    return f"{value}{NBSP}%"


def fake_real(probability_fake) -> str:
    """'38 % fake · 62 % real' — fake first, complement derived from the shown number.

    Non-breaking spaces keep each number with its % and its word, so a pair only wraps at the dot.
    """
    fake = shown(probability_fake)
    if fake is None:
        return "not scored"
    return f"{pct_text(fake)}{NBSP}fake · {pct_text(100 - fake)}{NBSP}real"


def words(n: int) -> str:
    return f"{n:,} word" if n == 1 else f"{n:,} words"


def points(p: float) -> str:
    return f"{p:g}"


def lean(probability_fake) -> str | None:
    fake = shown(probability_fake)
    if fake is None:
        return None
    if fake >= LEAN_FAKE_AT:
        return "fake"
    if fake <= LEAN_REAL_AT:
        return "real"
    return None


def verdict_headline(blended_fake: float) -> str:
    fake = shown(blended_fake)
    top, value = ("fake", fake) if fake >= 50 else ("real", 100 - fake)
    if value < 55:
        return "Too close to call"
    if value < 65:
        return f"Leans {top}"
    if value < 80:
        return "Reads like fake news" if top == "fake" else "Reads like real reporting"
    return "Reads strongly like fake news" if top == "fake" else "Reads strongly like real reporting"


@dataclass
class Row:
    name: str
    numbers: str
    note: str | None = None
    lean_text: str | None = None
    missing: bool = False


@dataclass
class View:
    state: str
    headline: str
    body: list[str] = field(default_factory=list)
    rows: list[Row] = field(default_factory=list)
    agreement: str | None = None
    foot: str | None = None
    action: str | None = None  # "edit" | "home" | "restore"
    code: str | None = None
    tone: str = "plain"  # plain | amber | fault


def first_load() -> View:
    return View("first_load", HEADLINES["first_load"], ["Paste an article or load a sample. Scores appear here."])


def error(code: str, *, detail: str | None = None, request_id: str | None = None,
          text_bytes: int | None = None, limit_bytes: int | None = None, path: str | None = None) -> View:
    if code == "text_empty":
        return View("error", "Nothing to score", ["The text box is empty."], action="edit", code=code, tone="fault",
                    foot="Paste an article, or load a sample.")
    if code == "too_large":
        if text_bytes is not None and limit_bytes is not None:
            cause = f"This text is {text_bytes:,} bytes; the limit is {limit_bytes:,} bytes (256 KB)."
        else:
            cause = "The request was over 1 MB, which is more than one article."
        return View("error", "Too much text", [cause], action="restore", code=code, tone="fault", foot="Paste one article.")
    if code == "model_unavailable":
        reason = f" ({detail})" if detail else ""
        return View("error", "The detector isn't available", [f"The model files didn't load on this server{reason}."],
                    code=code, tone="fault", foot="Start the server from the repository, or see README › Run.")
    if code == "not_found":
        shown_path = (path or "")[:80]
        return View("not_found", f"Nothing at {shown_path}", [], action="home", code=code, tone="fault")
    if code == "internal_error":
        return View("error", "Something went wrong on the server",
                    [f"Logged as request {request_id}; no details are shown here."], code=code, tone="fault",
                    foot="Try again.")
    return View("error", "That request isn't supported here", [detail] if detail else [], action="home", code=code,
                tone="fault")


def _model_row(r, *, with_lean: bool) -> Row:
    if r.known_features == 0:
        return Row(MODEL_NAMES[r.key], "no words it recognises", MODEL_NOTES[r.key])
    lean_word = lean(r.fake)
    return Row(
        MODEL_NAMES[r.key],
        fake_real(r.fake),
        MODEL_NOTES[r.key],
        lean_text=f"leans {lean_word}" if (with_lean and lean_word) else None,
    )


def _model_lean(r) -> str | None:
    return None if r.known_features == 0 else lean(r.fake)


def _agreement(fmt, content) -> str | None:
    if content is None:
        return None
    lf, lc = _model_lean(fmt), _model_lean(content)
    if lf and lf == lc:
        return f"Both models lean {lf}."
    if not lf and not lc:
        return "Neither model leans either way."
    if lf and not lc:
        return f"The format model leans {lf}; the content model has no lean."
    if lc and not lf:
        return f"The content model leans {lc}; the format model has no lean."
    return None  # opposite leans: the disagreement state handles it


def _empty_blend(models) -> float:
    if models.content.loaded:
        return FORMAT_WEIGHT * models.format.empty_fake + CONTENT_WEIGHT * models.content.empty_fake
    return models.format.empty_fake


def scored(reading: Reading, models) -> View:
    """The view for a text that was read. Precedence: url_only → unrecognised → no_signal →
    disagreement → verdict."""
    fmt, content = reading.format, reading.content
    if reading.status == "url_only":
        return View("url_only", HEADLINES["url_only"],
                    ["The detector reads text only. Paste the article itself, not its address."], action="edit")
    if reading.status == "no_known_words":
        return View("unrecognised", HEADLINES["unrecognised"],
                    ["Most of this text isn't in the English news vocabulary the models learned, "
                     "so there is little for them to read."], action="edit")
    if reading.status == "no_signal":
        who = "Both models" if content is not None else "The format model"
        return View("no_signal", HEADLINES["no_signal"],
                    [f"{who} scored this within {points(NO_SIGNAL_POINTS)} points of an empty text "
                     f"({fake_real(_empty_blend(models))}), so the score says nothing about these words."],
                    action="edit")

    blended_fake = (FORMAT_WEIGHT * fmt.fake + CONTENT_WEIGHT * content.fake) if content is not None else fmt.fake
    foot = f"Read {words(reading.word_count)}."
    if content is None:
        rows = [
            Row("Format model only — content model not loaded", fake_real(blended_fake)),
            _model_row(fmt, with_lean=False),
            Row(MODEL_NAMES["content_model"], "not scored — model file not loaded", missing=True),
        ]
        return View("verdict", verdict_headline(blended_fake), rows=rows, foot=foot)

    lf, lc = _model_lean(fmt), _model_lean(content)
    if lf and lc and lf != lc:
        rows = [
            Row("Blended 60/40", fake_real(blended_fake)),
            _model_row(fmt, with_lean=True),
            _model_row(content, with_lean=True),
        ]
        return View("disagreement", HEADLINES["disagreement"],
                    ["The format model reacts to agency datelines and wire-service phrasing; the content model "
                     "has those removed first. A split means the styling and the wording point different ways."],
                    rows=rows, foot=foot, tone="amber")

    rows = [
        Row("Blended 60/40", fake_real(blended_fake)),
        _model_row(fmt, with_lean=False),
        _model_row(content, with_lean=False),
    ]
    return View("verdict", verdict_headline(blended_fake), rows=rows, agreement=_agreement(fmt, content), foot=foot)
